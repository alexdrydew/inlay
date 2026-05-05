use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Mutex;

use pyo3::PyTraverseError;
use pyo3::exceptions::{PyAttributeError, PyKeyError};
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use crate::types::MemberAccessKind;

use crate::compile::flatten::{ExecutionGraph, ExecutionNodeId, resource_plan_for_node};

use super::executor::{ContextData, execute};
use super::resources::RuntimeResources;

#[pyclass(frozen, module = "inlay")]
pub(crate) struct DelegatedMember {
    pub(crate) source: Py<PyAny>,
    pub(crate) name: Arc<str>,
    pub(crate) access_kind: MemberAccessKind,
}

impl DelegatedMember {
    pub(crate) fn read<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.access_kind {
            MemberAccessKind::Attribute => self.source.bind(py).getattr(self.name.as_ref()),
            MemberAccessKind::DictItem => self.source.bind(py).get_item(self.name.as_ref()),
        }
    }

    pub(crate) fn write(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        match self.access_kind {
            MemberAccessKind::Attribute => self.source.bind(py).setattr(self.name.as_ref(), value),
            MemberAccessKind::DictItem => self.source.bind(py).set_item(self.name.as_ref(), value),
        }
    }
}

#[pymethods]
impl DelegatedMember {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.source)
    }
}

pub(crate) fn unwrap_delegated<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(member) = value.cast::<DelegatedMember>() {
        member.borrow().read(value.py())
    } else {
        Ok(value.clone())
    }
}

const INTERNAL_ATTRS: &[&str] = &["__members", "__writable"];

#[pyclass(weakref, module = "inlay")]
pub(crate) struct ContextProxy {
    graph: Arc<ExecutionGraph>,
    members: HashMap<Arc<str>, Option<ExecutionNodeId>>,
    values: Mutex<HashMap<Arc<str>, Py<PyAny>>>,
    writable: HashSet<Arc<str>>,
    resources: RuntimeResources,
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
            members: members
                .into_iter()
                .map(|(name, node_id)| (name, Some(node_id)))
                .collect(),
            values: Mutex::new(HashMap::new()),
            writable,
            resources,
        }
    }

    pub(crate) fn from_materialized(
        graph: Arc<ExecutionGraph>,
        values: HashMap<Arc<str>, Py<PyAny>>,
    ) -> Self {
        Self {
            graph,
            members: values.keys().map(|name| (Arc::clone(name), None)).collect(),
            values: Mutex::new(values),
            writable: HashSet::new(),
            resources: RuntimeResources::empty(),
        }
    }

    fn materialize_member(
        &self,
        py: Python<'_>,
        name: Arc<str>,
        node_id: Option<ExecutionNodeId>,
    ) -> PyResult<Py<PyAny>> {
        if let Some(value) = self.values.lock().expect("poisoned").get(&name) {
            return Ok(value.clone_ref(py));
        }
        let node_id = node_id.ok_or_else(|| PyAttributeError::new_err(format!("'{name}'")))?;

        let plan = resource_plan_for_node(&self.graph, node_id, &Default::default());
        let data = ContextData {
            graph: Arc::clone(&self.graph),
            root_node: node_id,
        };
        let value = execute(py, &data, self.resources.capture_plan(py, &plan)?, true)?;
        let mut values = self.values.lock().expect("poisoned");
        match values.get(&name) {
            Some(existing) => Ok(existing.clone_ref(py)),
            None => {
                values.insert(name, value.clone_ref(py));
                Ok(value)
            }
        }
    }
}

#[pymethods]
impl ContextProxy {
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let (member_name, node_id) = self
            .members
            .get_key_value(name)
            .map(|(key, &node_id)| (Arc::clone(key), node_id))
            .ok_or_else(|| PyAttributeError::new_err(format!("'{name}'")))?;
        let value = self.materialize_member(py, member_name, node_id)?;
        let bound = value.bind(py);
        if let Ok(member) = bound.cast::<DelegatedMember>() {
            Ok(member.borrow().read(py)?.unbind())
        } else {
            Ok(value.clone_ref(py))
        }
    }

    fn __setattr__(&mut self, py: Python<'_>, name: &str, value: Py<PyAny>) -> PyResult<()> {
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
        let (member_name, node_id) = self
            .members
            .get_key_value(name)
            .map(|(key, &node_id)| (Arc::clone(key), node_id))
            .ok_or_else(|| PyAttributeError::new_err(format!("'{name}'")))?;
        let current = self.materialize_member(py, member_name, node_id)?;
        let current_bound = current.bind(py);
        if let Ok(member) = current_bound.cast::<DelegatedMember>() {
            member.borrow().write(py, value.bind(py))
        } else {
            self.values
                .lock()
                .expect("poisoned")
                .insert(Arc::from(name), value);
            Ok(())
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Ok(values) = self.values.try_lock() {
            for value in values.values() {
                visit.call(value)?;
            }
        }
        self.resources.traverse_py_refs(&visit)?;
        Ok(())
    }

    fn __clear__(&mut self) {
        if let Ok(mut values) = self.values.lock() {
            values.clear();
        }
        self.members.clear();
        self.writable.clear();
        self.resources.clear();
    }

    fn __delattr__(&self, name: &str) -> PyResult<()> {
        Err(PyAttributeError::new_err(format!(
            "cannot delete attribute '{name}'"
        )))
    }
}

#[pyclass(module = "inlay")]
pub(crate) struct DelegatedDict {
    members: HashMap<Arc<str>, Py<PyAny>>,
}

impl DelegatedDict {
    pub(crate) fn new(members: HashMap<Arc<str>, Py<PyAny>>) -> Self {
        Self { members }
    }
}

#[pymethods]
impl DelegatedDict {
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        let member = self
            .members
            .get(key)
            .ok_or_else(|| PyKeyError::new_err(key.to_owned()))?;
        Ok(unwrap_delegated(member.bind(py))?.unbind())
    }

    fn __setitem__(&mut self, py: Python<'_>, key: &str, value: Py<PyAny>) -> PyResult<()> {
        let member = self
            .members
            .get(key)
            .ok_or_else(|| PyKeyError::new_err(key.to_owned()))?;
        if let Ok(member) = member.bind(py).cast::<DelegatedMember>() {
            member.borrow().write(py, value.bind(py))
        } else {
            self.members.insert(Arc::from(key), value);
            Ok(())
        }
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
            .values()
            .map(|member| unwrap_delegated(member.bind(py)).map(|v| v.unbind()))
            .collect::<PyResult<_>>()?;
        PyList::new(py, &vals)
    }

    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Bound<'py, PyTuple>> = self
            .members
            .iter()
            .map(|(k, member)| {
                let v = unwrap_delegated(member.bind(py))?;
                PyTuple::new(py, [PyString::new(py, k).as_any(), &v])
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
        match self.members.get(key) {
            Some(member) => Ok(unwrap_delegated(member.bind(py))?.unbind()),
            None => Ok(default.unwrap_or_else(|| py.None())),
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for member in self.members.values() {
            visit.call(member)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.members.clear();
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
