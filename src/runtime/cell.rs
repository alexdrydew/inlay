use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Arc, Condvar, Mutex};

use pyo3::PyTraverseError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use serde::{Deserialize, Serialize};

use crate::compile::execution_graph::{ExecutionGraph, ExecutionGraphState, ExecutionNodeId};

use super::executor::{
    ContextData, ExecutionState, InProgressGuard, current_thread_owns_in_progress_node,
    current_thread_owns_node, execute_node, write_node,
};
use super::resource_plan::resource_plan_for_node;
use super::resources::{InProgressOwners, RuntimeResources, RuntimeResourcesState};

#[derive(Serialize, Deserialize)]
struct LiveCellState {
    graph: ExecutionGraphState,
    target: usize,
    resources: RuntimeResourcesState,
}

pub(crate) struct CellHandle {
    graph: Arc<ExecutionGraph>,
    target: ExecutionNodeId,
    resources: Mutex<Option<RuntimeResources>>,
    resources_available: Condvar,
    in_progress: Mutex<InProgressOwners>,
}

struct ResourceLease<'a> {
    handle: &'a CellHandle,
    backup: Option<RuntimeResources>,
}

impl ResourceLease<'_> {
    fn restore(&mut self, resources: RuntimeResources) {
        self.handle.restore_resources(resources);
        self.backup = None;
    }
}

impl Drop for ResourceLease<'_> {
    fn drop(&mut self) {
        if let Some(resources) = self.backup.take() {
            self.handle.restore_resources(resources);
        }
    }
}

impl CellHandle {
    pub(crate) fn new(
        graph: Arc<ExecutionGraph>,
        target: ExecutionNodeId,
        resources: RuntimeResources,
    ) -> Self {
        let in_progress = resources.in_progress();
        Self {
            graph,
            target,
            resources: Mutex::new(Some(resources)),
            resources_available: Condvar::new(),
            in_progress: Mutex::new(in_progress),
        }
    }

    fn in_progress(&self) -> InProgressOwners {
        Arc::clone(&self.in_progress.lock().expect("poisoned"))
    }

    fn take_resources(
        &self,
        py: Python<'_>,
        except: Option<ExecutionNodeId>,
    ) -> PyResult<RuntimeResources> {
        let mut resources = self.resources.lock().expect("poisoned");
        if let Some(resources) = resources.take() {
            return Ok(resources);
        }
        drop(resources);

        if current_thread_owns_in_progress_node(&self.in_progress(), except) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "execution node accessed while it was still being computed",
            ));
        }

        Ok(py.detach(|| {
            let mut resources = self.resources.lock().expect("poisoned");
            loop {
                if let Some(resources) = resources.take() {
                    return resources;
                }
                resources = self.resources_available.wait(resources).expect("poisoned");
            }
        }))
    }

    fn restore_resources(&self, resources: RuntimeResources) {
        *self.resources.lock().expect("poisoned") = Some(resources);
        self.resources_available.notify_one();
    }

    fn with_resources<T>(
        &self,
        py: Python<'_>,
        except: Option<ExecutionNodeId>,
        operation: impl FnOnce(&ContextData, &mut ExecutionState) -> PyResult<T>,
    ) -> PyResult<T> {
        let resources = self.take_resources(py, except)?;
        let backup = resources.clone_ref(py);
        let mut lease = ResourceLease {
            handle: self,
            backup: Some(backup),
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            let data = ContextData {
                graph: Arc::clone(&self.graph),
                root_node: self.target,
            };
            let mut state = ExecutionState::new(resources, true);
            let result = operation(&data, &mut state);
            (result, state.resources)
        }));
        match result {
            Ok((result, resources)) => {
                lease.restore(resources);
                result
            }
            Err(payload) => {
                drop(lease);
                resume_unwind(payload)
            }
        }
    }

    fn get(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.with_resources(py, None, |data, state| {
            execute_node(py, data, state, self.target)
        })
    }

    fn set(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        let in_progress = self.in_progress();
        let already_owned = current_thread_owns_node(&in_progress, self.target);
        let guard = if already_owned {
            None
        } else {
            Some(InProgressGuard::enter(&in_progress, self.target)?)
        };
        let result = self.with_resources(
            py,
            (!already_owned).then_some(self.target),
            |data, state| write_node(py, data, state, self.target, value),
        );
        drop(guard);
        result
    }

    fn relink(&self, py: Python<'_>, resources: &mut RuntimeResources) -> PyResult<()> {
        let plan = resource_plan_for_node(&self.graph, self.target, &HashSet::new());
        let captured = resources.capture_plan(py, &plan)?;
        *self.in_progress.lock().expect("poisoned") = captured.in_progress();
        *self.resources.lock().expect("poisoned") = Some(captured);
        self.resources_available.notify_all();
        Ok(())
    }

    fn to_state(
        &self,
        py: Python<'_>,
        refs: &mut crate::pickle::PyRefCollector,
    ) -> PyResult<LiveCellState> {
        loop {
            let resources = self.resources.lock().expect("poisoned");
            if let Some(resources) = resources.as_ref() {
                return Ok(LiveCellState {
                    graph: self.graph.to_state(py, refs),
                    target: self.target.index(),
                    resources: resources.to_state(py, refs),
                });
            }
            drop(resources);
            if current_thread_owns_in_progress_node(&self.in_progress(), None) {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "execution node accessed while it was still being computed",
                ));
            }
            py.detach(|| {
                let mut resources = self.resources.lock().expect("poisoned");
                while resources.is_none() {
                    resources = self.resources_available.wait(resources).expect("poisoned");
                }
            });
        }
    }

    fn from_state(
        py: Python<'_>,
        state: LiveCellState,
        refs: &crate::pickle::PyRefResolver<'_>,
    ) -> PyResult<Self> {
        let graph = Arc::new(ExecutionGraph::from_state(state.graph, refs)?);
        let mut resources = RuntimeResources::from_state(state.resources, refs)?;
        relink_cached_cells(py, &mut resources)?;
        Ok(Self::new(
            graph,
            ExecutionNodeId::from_index(state.target),
            resources,
        ))
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Ok(resources) = self.resources.try_lock()
            && let Some(resources) = resources.as_ref()
        {
            resources.traverse_py_refs(visit)?;
        }
        Ok(())
    }

    fn clear(&mut self) {
        if let Ok(mut resources) = self.resources.lock()
            && let Some(resources) = resources.as_mut()
        {
            resources.clear();
        }
    }
}

#[pyclass(module = "inlay")]
pub(crate) struct ReadCellImpl {
    handle: CellHandle,
}

impl ReadCellImpl {
    pub(crate) fn new(handle: CellHandle) -> Self {
        Self { handle }
    }
}

#[pyclass(module = "inlay")]
pub(crate) struct CellImpl {
    handle: CellHandle,
}

impl CellImpl {
    pub(crate) fn new(handle: CellHandle) -> Self {
        Self { handle }
    }
}

pub(crate) fn relink_cached_cells(
    py: Python<'_>,
    resources: &mut RuntimeResources,
) -> PyResult<()> {
    for value in resources.cached_values(py) {
        if let Ok(cell) = value.bind(py).cast::<ReadCellImpl>() {
            cell.borrow().handle.relink(py, resources)?;
        } else if let Ok(cell) = value.bind(py).cast::<CellImpl>() {
            cell.borrow().handle.relink(py, resources)?;
        }
    }
    Ok(())
}

#[pyfunction]
pub(crate) fn _rebuild_read_cell(
    state: &Bound<'_, PyAny>,
    refs: &Bound<'_, PyAny>,
) -> PyResult<ReadCellImpl> {
    let py = state.py();
    let state: LiveCellState = crate::pickle::depythonize_state(state)?;
    let refs = crate::pickle::PyRefResolver::new(refs)?;
    Ok(ReadCellImpl::new(CellHandle::from_state(py, state, &refs)?))
}

#[pyfunction]
pub(crate) fn _rebuild_cell(
    state: &Bound<'_, PyAny>,
    refs: &Bound<'_, PyAny>,
) -> PyResult<CellImpl> {
    let py = state.py();
    let state: LiveCellState = crate::pickle::depythonize_state(state)?;
    let refs = crate::pickle::PyRefResolver::new(refs)?;
    Ok(CellImpl::new(CellHandle::from_state(py, state, &refs)?))
}

#[pymethods]
impl ReadCellImpl {
    fn get(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.handle.get(py)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.handle.traverse(&visit)
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let mut refs = crate::pickle::PyRefCollector::default();
        crate::pickle::reduce_with_state_and_refs(
            py,
            "_rebuild_read_cell",
            crate::pickle::pythonize_state(py, &self.handle.to_state(py, &mut refs)?)?,
            refs.into_tuple(py)?,
        )
    }

    fn __clear__(&mut self) {
        self.handle.clear();
    }
}

#[pymethods]
impl CellImpl {
    fn get(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.handle.get(py)
    }

    fn set(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        self.handle.set(py, value)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.handle.traverse(&visit)
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let mut refs = crate::pickle::PyRefCollector::default();
        crate::pickle::reduce_with_state_and_refs(
            py,
            "_rebuild_cell",
            crate::pickle::pythonize_state(py, &self.handle.to_state(py, &mut refs)?)?,
            refs.into_tuple(py)?,
        )
    }

    fn __clear__(&mut self) {
        self.handle.clear();
    }
}
