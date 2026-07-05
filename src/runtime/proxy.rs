use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Mutex;

use pyo3::PyTraverseError;
use pyo3::exceptions::{PyAttributeError, PyKeyError};
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use serde::{Deserialize, Serialize};

use crate::compile::execution_graph::{ExecutionGraph, ExecutionGraphState, ExecutionNodeId};

use super::executor::{ContextData, ExecutionState, execute, node_is_writable, write_node};
use super::resource_plan::resource_plan_for_node;
use super::resources::RuntimeResources;
use super::resources::RuntimeResourcesState;

#[derive(Serialize, Deserialize)]
struct DelegatedDictState {
    graph: ExecutionGraphState,
    members: Vec<ContextMemberState>,
    resources: RuntimeResourcesState,
}

#[derive(Serialize, Deserialize)]
struct ContextProxyState {
    graph: ExecutionGraphState,
    members: Vec<ContextMemberState>,
    values: Vec<NamedPyRefState>,
    writable: Vec<String>,
    resources: RuntimeResourcesState,
}

#[derive(Serialize, Deserialize)]
struct NamedPyRefState {
    name: String,
    value_ref: usize,
}

#[derive(Serialize, Deserialize)]
struct ContextMemberState {
    name: String,
    node: usize,
}

const INTERNAL_ATTRS: &[&str] = &["__members", "__writable"];

#[pyclass(weakref, module = "inlay")]
pub(crate) struct ContextProxy {
    graph: Arc<ExecutionGraph>,
    members: HashMap<Arc<str>, ExecutionNodeId>,
    writable: HashSet<Arc<str>>,
    values: Mutex<HashMap<Arc<str>, Py<PyAny>>>,
    resources: Mutex<RuntimeResources>,
}

impl ContextProxy {
    pub(crate) fn new(
        graph: Arc<ExecutionGraph>,
        members: HashMap<Arc<str>, ExecutionNodeId>,
        writable: HashSet<Arc<str>>,
        resources: RuntimeResources,
    ) -> Self {
        Self {
            graph,
            members,
            writable,
            values: Mutex::new(HashMap::new()),
            resources: Mutex::new(resources),
        }
    }

    pub(crate) fn from_single_member(
        graph: Arc<ExecutionGraph>,
        name: Arc<str>,
        node_id: ExecutionNodeId,
        resources: RuntimeResources,
    ) -> Self {
        let writable = node_is_writable(&graph, node_id)
            .then(|| Arc::clone(&name))
            .into_iter()
            .collect();
        Self {
            graph,
            members: HashMap::from([(name, node_id)]),
            writable,
            values: Mutex::new(HashMap::new()),
            resources: Mutex::new(resources),
        }
    }

    fn to_state(
        &self,
        py: Python<'_>,
        refs: &mut crate::pickle::PyRefCollector,
    ) -> ContextProxyState {
        let values = self
            .values
            .lock()
            .expect("poisoned")
            .iter()
            .map(|(name, value)| NamedPyRefState {
                name: name.to_string(),
                value_ref: refs.push(py, value),
            })
            .collect();
        ContextProxyState {
            graph: self.graph.to_state(py, refs),
            members: self
                .members
                .iter()
                .map(|(name, node)| ContextMemberState {
                    name: name.to_string(),
                    node: node.index(),
                })
                .collect(),
            values,
            writable: self.writable.iter().map(ToString::to_string).collect(),
            resources: self.resources.lock().expect("poisoned").to_state(py, refs),
        }
    }

    fn execute_member(
        &self,
        py: Python<'_>,
        name: Arc<str>,
        node_id: ExecutionNodeId,
    ) -> PyResult<Py<PyAny>> {
        if let Some(value) = self
            .values
            .lock()
            .expect("poisoned")
            .get(&name)
            .map(|value| value.clone_ref(py))
        {
            return Ok(value);
        }
        let plan = resource_plan_for_node(&self.graph, node_id, &Default::default());
        let resources = self
            .resources
            .lock()
            .expect("poisoned")
            .capture_plan(py, &plan)?;
        let data = ContextData {
            graph: Arc::clone(&self.graph),
            root_node: node_id,
        };
        let value = execute(py, &data, resources, true)?;
        if !self.graph[node_id].source_deps.is_empty() {
            return Ok(value);
        }
        let mut values = self.values.lock().expect("poisoned");
        Ok(values
            .entry(name)
            .or_insert_with(|| value.clone_ref(py))
            .clone_ref(py))
    }

    fn write_member(
        &self,
        py: Python<'_>,
        node_id: ExecutionNodeId,
        value: Py<PyAny>,
    ) -> PyResult<()> {
        let resources = {
            let mut guard = self.resources.lock().expect("poisoned");
            std::mem::take(&mut *guard)
        };
        let data = ContextData {
            graph: Arc::clone(&self.graph),
            root_node: node_id,
        };
        let mut state = ExecutionState::new(resources, true);
        let result = write_node(py, &data, &mut state, node_id, value);
        let mut guard = self.resources.lock().expect("poisoned");
        *guard = state.resources;
        result
    }
}

#[pyfunction]
pub(crate) fn _rebuild_context_proxy(
    state: &Bound<'_, PyAny>,
    refs: &Bound<'_, PyAny>,
) -> PyResult<ContextProxy> {
    let state: ContextProxyState = crate::pickle::depythonize_state(state)?;
    let refs = crate::pickle::PyRefResolver::new(refs)?;
    let graph = Arc::new(ExecutionGraph::from_state(state.graph, &refs)?);
    let members = state
        .members
        .into_iter()
        .map(|member| {
            (
                Arc::from(member.name.as_str()),
                ExecutionNodeId::from_index(member.node),
            )
        })
        .collect();
    let values = state
        .values
        .into_iter()
        .map(|value| Ok((Arc::from(value.name.as_str()), refs.get(value.value_ref)?)))
        .collect::<PyResult<HashMap<_, _>>>()?;
    let writable = state.writable.into_iter().map(Arc::<str>::from).collect();
    let resources = RuntimeResources::from_state(state.resources, &refs)?;

    Ok(ContextProxy {
        graph,
        members,
        writable,
        values: Mutex::new(values),
        resources: Mutex::new(resources),
    })
}

#[pymethods]
impl ContextProxy {
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let (member_name, node_id) = self
            .members
            .get_key_value(name)
            .map(|(name, &node_id)| (Arc::clone(name), node_id))
            .ok_or_else(|| PyAttributeError::new_err(format!("'{name}'")))?;
        self.execute_member(py, member_name, node_id)
    }

    fn __setattr__(&self, py: Python<'_>, name: &str, value: Py<PyAny>) -> PyResult<()> {
        if INTERNAL_ATTRS.contains(&name) {
            return Err(PyAttributeError::new_err(format!(
                "cannot set internal attribute '{name}'"
            )));
        }
        if !self.writable.contains(name) {
            return Err(PyAttributeError::new_err(format!(
                "attribute '{name}' is not writable"
            )));
        }
        let node_id = *self
            .members
            .get(name)
            .ok_or_else(|| PyAttributeError::new_err(format!("'{name}'")))?;
        let result = self.write_member(py, node_id, value);
        if result.is_ok() {
            self.values.lock().expect("poisoned").clear();
        }
        result
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Ok(values) = self.values.try_lock() {
            for value in values.values() {
                visit.call(value)?;
            }
        }
        if let Ok(resources) = self.resources.try_lock() {
            resources.traverse_py_refs(&visit)?;
        }
        Ok(())
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let mut refs = crate::pickle::PyRefCollector::default();
        let state = self.to_state(py, &mut refs);
        crate::pickle::reduce_with_state_and_refs(
            py,
            "_rebuild_context_proxy",
            crate::pickle::pythonize_state(py, &state)?,
            refs.into_tuple(py)?,
        )
    }

    fn __clear__(&mut self) {
        if let Ok(mut values) = self.values.lock() {
            values.clear();
        }
        self.members.clear();
        self.writable.clear();
        if let Ok(mut resources) = self.resources.lock() {
            resources.clear();
        }
    }

    fn __delattr__(&self, name: &str) -> PyResult<()> {
        Err(PyAttributeError::new_err(format!(
            "cannot delete attribute '{name}'"
        )))
    }
}

#[pyclass(module = "inlay")]
pub(crate) struct DelegatedDict {
    graph: Arc<ExecutionGraph>,
    members: HashMap<Arc<str>, ExecutionNodeId>,
    resources: Mutex<RuntimeResources>,
}

impl DelegatedDict {
    pub(crate) fn new(
        graph: Arc<ExecutionGraph>,
        members: HashMap<Arc<str>, ExecutionNodeId>,
        resources: RuntimeResources,
    ) -> Self {
        Self {
            graph,
            members,
            resources: Mutex::new(resources),
        }
    }

    fn to_state(
        &self,
        py: Python<'_>,
        refs: &mut crate::pickle::PyRefCollector,
    ) -> DelegatedDictState {
        DelegatedDictState {
            graph: self.graph.to_state(py, refs),
            members: self
                .members
                .iter()
                .map(|(name, node)| ContextMemberState {
                    name: name.to_string(),
                    node: node.index(),
                })
                .collect(),
            resources: self.resources.lock().expect("poisoned").to_state(py, refs),
        }
    }

    fn execute_item(&self, py: Python<'_>, node_id: ExecutionNodeId) -> PyResult<Py<PyAny>> {
        let plan = resource_plan_for_node(&self.graph, node_id, &Default::default());
        let resources = self
            .resources
            .lock()
            .expect("poisoned")
            .capture_plan(py, &plan)?;
        let data = ContextData {
            graph: Arc::clone(&self.graph),
            root_node: node_id,
        };
        execute(py, &data, resources, true)
    }

    fn write_item(
        &self,
        py: Python<'_>,
        node_id: ExecutionNodeId,
        value: Py<PyAny>,
    ) -> PyResult<()> {
        let resources = {
            let mut guard = self.resources.lock().expect("poisoned");
            std::mem::take(&mut *guard)
        };
        let data = ContextData {
            graph: Arc::clone(&self.graph),
            root_node: node_id,
        };
        let mut state = ExecutionState::new(resources, true);
        let result = write_node(py, &data, &mut state, node_id, value);
        let mut guard = self.resources.lock().expect("poisoned");
        *guard = state.resources;
        result
    }
}

#[pyfunction]
pub(crate) fn _rebuild_delegated_dict(
    state: &Bound<'_, PyAny>,
    refs: &Bound<'_, PyAny>,
) -> PyResult<DelegatedDict> {
    let state: DelegatedDictState = crate::pickle::depythonize_state(state)?;
    let refs = crate::pickle::PyRefResolver::new(refs)?;
    let graph = Arc::new(ExecutionGraph::from_state(state.graph, &refs)?);
    let members = state
        .members
        .into_iter()
        .map(|member| {
            (
                Arc::from(member.name.as_str()),
                ExecutionNodeId::from_index(member.node),
            )
        })
        .collect();
    let resources = RuntimeResources::from_state(state.resources, &refs)?;
    Ok(DelegatedDict {
        graph,
        members,
        resources: Mutex::new(resources),
    })
}

#[pymethods]
impl DelegatedDict {
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        let node_id = *self
            .members
            .get(key)
            .ok_or_else(|| PyKeyError::new_err(key.to_owned()))?;
        self.execute_item(py, node_id)
    }

    fn __setitem__(&self, py: Python<'_>, key: &str, value: Py<PyAny>) -> PyResult<()> {
        let node_id = *self
            .members
            .get(key)
            .ok_or_else(|| PyKeyError::new_err(key.to_owned()))?;
        if !node_is_writable(&self.graph, node_id) {
            return Err(PyKeyError::new_err(format!("key '{key}' is not writable")));
        }
        self.write_item(py, node_id, value)
    }

    fn __contains__(&self, key: &str) -> bool {
        self.members.contains_key(key)
    }

    fn __len__(&self) -> usize {
        self.members.len()
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let keys: Vec<&str> = self.members.keys().map(|k| &**k).collect();
        let list = PyList::new(py, &keys)?;
        list.call_method0("__iter__")
    }

    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.members.keys().map(|k| PyString::new(py, k)))
    }

    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let vals: Vec<Py<PyAny>> = self
            .members
            .keys()
            .map(|key| self.__getitem__(py, key))
            .collect::<PyResult<_>>()?;
        PyList::new(py, &vals)
    }

    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Bound<'py, PyTuple>> = self
            .members
            .keys()
            .map(|k| {
                let v = self.__getitem__(py, k)?;
                PyTuple::new(py, [PyString::new(py, k).as_any(), v.bind(py)])
            })
            .collect::<PyResult<_>>()?;
        PyList::new(py, &items)
    }

    #[pyo3(name = "get")]
    fn get_item(
        &self,
        py: Python<'_>,
        key: &str,
        default: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if self.members.contains_key(key) {
            self.__getitem__(py, key)
        } else {
            Ok(default.unwrap_or_else(|| py.None()))
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Ok(resources) = self.resources.try_lock() {
            resources.traverse_py_refs(&visit)?;
        }
        Ok(())
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let mut refs = crate::pickle::PyRefCollector::default();
        let state = self.to_state(py, &mut refs);
        crate::pickle::reduce_with_state_and_refs(
            py,
            "_rebuild_delegated_dict",
            crate::pickle::pythonize_state(py, &state)?,
            refs.into_tuple(py)?,
        )
    }

    fn __clear__(&mut self) {
        self.members.clear();
        if let Ok(mut resources) = self.resources.lock() {
            resources.clear();
        }
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other_dict) = other.cast::<DelegatedDict>() {
            let other_ref = other_dict.borrow();
            if self.members.len() != other_ref.members.len() {
                return Ok(false);
            }
            for key in self.members.keys() {
                if !other_ref.members.contains_key(key) {
                    return Ok(false);
                }
                let self_val = self.__getitem__(py, key)?;
                let other_val = other_ref.__getitem__(py, key)?;
                if !self_val.bind(py).eq(other_val.bind(py))? {
                    return Ok(false);
                }
            }
            Ok(true)
        } else if let Ok(other_dict) = other.cast::<PyDict>() {
            if self.members.len() != other_dict.len() {
                return Ok(false);
            }
            for key in self.members.keys() {
                let self_val = self.__getitem__(py, key)?;
                match other_dict.get_item(&**key)? {
                    Some(other_val) => {
                        if !self_val.bind(py).eq(&other_val)? {
                            return Ok(false);
                        }
                    }
                    None => return Ok(false),
                }
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }
}
