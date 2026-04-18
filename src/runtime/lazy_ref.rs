use std::sync::OnceLock;

use pyo3::PyTraverseError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;

#[pyclass(frozen, module = "inlay")]
pub(crate) struct LazyRefImpl {
    value: OnceLock<Py<PyAny>>,
    target_type_name: String,
}

impl LazyRefImpl {
    pub(crate) fn new(target_type_name: String) -> Self {
        Self {
            value: OnceLock::new(),
            target_type_name,
        }
    }

    pub(crate) fn bind_value(&self, value: Py<PyAny>) {
        self.value.set(value).expect("LazyRef bound twice");
    }
}

#[pymethods]
impl LazyRefImpl {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(val) = self.value.get() {
            visit.call(val)?;
        }
        Ok(())
    }

    fn get(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.value.get().map(|v| v.clone_ref(py)).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "LazyRef<{}> accessed before context was fully built",
                self.target_type_name
            ))
        })
    }

    fn bind(&self, value: Py<PyAny>) {
        self.bind_value(value);
    }
}
