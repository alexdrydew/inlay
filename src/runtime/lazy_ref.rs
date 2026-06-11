use std::sync::OnceLock;

use pyo3::PyTraverseError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct LazyRefState {
    value_ref: Option<usize>,
}

#[pyclass(frozen, module = "inlay")]
pub(crate) struct LazyRefImpl {
    value: OnceLock<Py<PyAny>>,
}

impl LazyRefImpl {
    pub(crate) fn new() -> Self {
        Self {
            value: OnceLock::new(),
        }
    }

    pub(crate) fn bind_value(&self, value: Py<PyAny>) {
        self.value.set(value).expect("LazyRef bound twice");
    }

    fn to_state(&self, py: Python<'_>, refs: &mut crate::pickle::PyRefCollector) -> LazyRefState {
        LazyRefState {
            value_ref: self.value.get().map(|value| refs.push(py, value)),
        }
    }
}

#[pyfunction]
pub(crate) fn _rebuild_lazy_ref(
    state: &Bound<'_, PyAny>,
    refs: &Bound<'_, PyAny>,
) -> PyResult<LazyRefImpl> {
    let state: LazyRefState = crate::pickle::depythonize_state(state)?;
    let refs = crate::pickle::PyRefResolver::new(refs)?;
    let result = LazyRefImpl::new();
    if let Some(value_ref) = state.value_ref {
        result.bind_value(refs.get(value_ref)?);
    }
    Ok(result)
}

#[pymethods]
impl LazyRefImpl {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(val) = self.value.get() {
            visit.call(val)?;
        }
        Ok(())
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let mut refs = crate::pickle::PyRefCollector::default();
        let state = self.to_state(py, &mut refs);
        crate::pickle::reduce_with_state_and_refs(
            py,
            "_rebuild_lazy_ref",
            crate::pickle::pythonize_state(py, &state)?,
            refs.into_tuple(py)?,
        )
    }

    fn get(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.value.get().map(|v| v.clone_ref(py)).ok_or_else(|| {
            PyRuntimeError::new_err("LazyRef accessed before context was fully built")
        })
    }

    fn bind(&self, value: Py<PyAny>) {
        self.bind_value(value);
    }
}
