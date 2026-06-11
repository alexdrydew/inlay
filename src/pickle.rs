use pyo3::prelude::*;
use pyo3::types::PyTuple;
use serde::{Serialize, de::DeserializeOwned};

pub(crate) fn pythonize_state<T: Serialize>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>> {
    pythonize::pythonize(py, value)
        .map(|obj| obj.unbind())
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

pub(crate) fn depythonize_state<T: DeserializeOwned>(obj: &Bound<'_, PyAny>) -> PyResult<T> {
    pythonize::depythonize(obj)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

#[derive(Default)]
pub(crate) struct PyRefCollector {
    refs: Vec<Py<PyAny>>,
}

impl PyRefCollector {
    pub(crate) fn push(&mut self, py: Python<'_>, value: &Py<PyAny>) -> usize {
        let index = self.refs.len();
        self.refs.push(value.clone_ref(py));
        index
    }

    pub(crate) fn into_tuple(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        py_tuple(py, self.refs)
    }
}

pub(crate) struct PyRefResolver<'py> {
    refs: Bound<'py, PyTuple>,
}

impl<'py> PyRefResolver<'py> {
    pub(crate) fn new(refs: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(Self {
            refs: refs.cast::<PyTuple>()?.clone(),
        })
    }

    pub(crate) fn get(&self, index: usize) -> PyResult<Py<PyAny>> {
        Ok(self.refs.get_item(index)?.unbind())
    }
}

pub(crate) fn py_tuple(py: Python<'_>, items: Vec<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    Ok(PyTuple::new(py, items)?.into_any().unbind())
}

pub(crate) fn reduce<'py>(
    py: Python<'py>,
    rebuild_name: &str,
    args: Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyTuple>> {
    let rebuild = py.import("inlay._native")?.getattr(rebuild_name)?;
    PyTuple::new(py, [rebuild.into_any().unbind(), args.into_any().unbind()])
}

pub(crate) fn reduce_with_single_state<'py>(
    py: Python<'py>,
    rebuild_name: &str,
    state: Py<PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let args = PyTuple::new(py, [state])?;
    reduce(py, rebuild_name, args)
}

pub(crate) fn reduce_with_state_and_refs<'py>(
    py: Python<'py>,
    rebuild_name: &str,
    state: Py<PyAny>,
    refs: Py<PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let args = PyTuple::new(py, [state, refs])?;
    reduce(py, rebuild_name, args)
}
