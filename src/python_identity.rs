use std::{fmt, sync::Arc};

use pyo3::{Bound, Py, PyAny, ffi};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct PythonIdentity(usize);

impl PythonIdentity {
    pub(crate) fn from_bound(object: &Bound<'_, PyAny>) -> Self {
        Self::from_ptr(object.as_ptr())
    }

    pub(crate) fn from_arc_py_any(object: &Arc<Py<PyAny>>) -> Self {
        Self::from_ptr(object.as_ref().as_ptr())
    }

    pub(crate) fn from_ptr(object: *mut ffi::PyObject) -> Self {
        Self(object as usize)
    }
}

impl fmt::Display for PythonIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
