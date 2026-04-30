use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::PyTraverseError;
use pyo3::exceptions::{PyAttributeError, PyKeyError};
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use super::executor::{ScopeHandle, attach_scope};

// ---------------------------------------------------------------------------
// DelegatedAttr — a lazy read/write handle to a dict entry
// ---------------------------------------------------------------------------

#[pyclass(frozen, module = "inlay")]
pub(crate) struct DelegatedAttr {
    pub(crate) source: Py<PyAny>,
    pub(crate) name: Arc<str>,
}

impl DelegatedAttr {
    pub(crate) fn read<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        read_attr_or_dict_item(self.source.bind(py), self.name.as_ref())
    }

    pub(crate) fn write(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        write_attr_or_dict_item(self.source.bind(py), self.name.as_ref(), value)
    }
}

pub(crate) fn read_attr_or_dict_item<'py>(
    source: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(dict) = source.cast::<PyDict>() {
        dict.get_item(name)?
            .ok_or_else(|| PyKeyError::new_err(name.to_owned()))
    } else {
        source.getattr(name)
    }
}

fn write_attr_or_dict_item(
    source: &Bound<'_, PyAny>,
    name: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if let Ok(dict) = source.cast::<PyDict>() {
        dict.set_item(name, value)
    } else {
        source.setattr(name, value)
    }
}

#[pymethods]
impl DelegatedAttr {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.source)
    }
}

/// Unwrap a value: if it's a DelegatedAttr, read through to the actual value.
pub(crate) fn unwrap_delegated<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(da) = value.cast::<DelegatedAttr>() {
        da.borrow().read(value.py())
    } else {
        Ok(value.clone())
    }
}

// ---------------------------------------------------------------------------
// ContextProxy — runtime proxy for Protocol structural types
// ---------------------------------------------------------------------------

const INTERNAL_ATTRS: &[&str] = &["__members", "__writable"];

#[pyclass(weakref, module = "inlay")]
pub(crate) struct ContextProxy {
    members: HashMap<Arc<str>, Py<PyAny>>,
    writable: HashSet<Arc<str>>,
    scope_owner: Option<ScopeHandle>,
}

impl ContextProxy {
    pub(crate) fn new(members: HashMap<Arc<str>, Py<PyAny>>, writable: HashSet<Arc<str>>) -> Self {
        Self {
            members,
            writable,
            scope_owner: None,
        }
    }

    pub(crate) fn set_scope_owner(&mut self, handle: ScopeHandle) {
        self.scope_owner = Some(handle);
    }
}

#[pymethods]
impl ContextProxy {
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let value = self
            .members
            .get(name)
            .ok_or_else(|| PyAttributeError::new_err(format!("'{name}'")))?;
        let bound = value.bind(py);
        if let Ok(da) = bound.cast::<DelegatedAttr>() {
            Ok(da.borrow().read(py)?.unbind())
        } else if let Some(handle) = &self.scope_owner {
            attach_scope(py, value.clone_ref(py), Arc::clone(handle))
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
        let current = self
            .members
            .get(name)
            .ok_or_else(|| PyAttributeError::new_err(format!("'{name}'")))?;
        let current_bound = current.bind(py);
        if let Ok(da) = current_bound.cast::<DelegatedAttr>() {
            da.borrow().write(py, value.bind(py))
        } else {
            self.members.insert(Arc::from(name), value);
            Ok(())
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for value in self.members.values() {
            visit.call(value)?;
        }
        if let Some(handle) = &self.scope_owner {
            if let Some(scope) = handle.get() {
                scope.traverse_py_refs(&visit)?;
            }
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.members.clear();
        self.writable.clear();
        self.scope_owner = None;
    }

    fn __delattr__(&self, name: &str) -> PyResult<()> {
        Err(PyAttributeError::new_err(format!(
            "cannot delete attribute '{name}'"
        )))
    }
}

// ---------------------------------------------------------------------------
// DelegatedDict — runtime proxy for TypedDict structural types
// ---------------------------------------------------------------------------

#[pyclass(module = "inlay")]
pub(crate) struct DelegatedDict {
    delegates: HashMap<Arc<str>, Py<DelegatedAttr>>,
}

impl DelegatedDict {
    pub(crate) fn new(delegates: HashMap<Arc<str>, Py<DelegatedAttr>>) -> Self {
        Self { delegates }
    }
}

#[pymethods]
impl DelegatedDict {
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        let da = self
            .delegates
            .get(key)
            .ok_or_else(|| PyKeyError::new_err(key.to_owned()))?;
        Ok(da.borrow(py).read(py)?.unbind())
    }

    fn __setitem__(&self, py: Python<'_>, key: &str, value: Py<PyAny>) -> PyResult<()> {
        let da = self
            .delegates
            .get(key)
            .ok_or_else(|| PyKeyError::new_err(key.to_owned()))?;
        da.borrow(py).write(py, value.bind(py))
    }

    fn __contains__(&self, key: &str) -> bool {
        self.delegates.contains_key(key)
    }

    fn __len__(&self) -> usize {
        self.delegates.len()
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let keys: Vec<&str> = self.delegates.keys().map(|k| &**k).collect();
        let list = PyList::new(py, &keys)?;
        list.call_method0("__iter__")
    }

    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.delegates.keys().map(|k| PyString::new(py, k)))
    }

    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let vals: Vec<Py<PyAny>> = self
            .delegates
            .values()
            .map(|da| da.borrow(py).read(py).map(|v| v.unbind()))
            .collect::<PyResult<_>>()?;
        PyList::new(py, &vals)
    }

    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Bound<'py, PyTuple>> = self
            .delegates
            .iter()
            .map(|(k, da)| {
                let v = da.borrow(py).read(py)?;
                PyTuple::new(py, [PyString::new(py, &**k).as_any(), &v])
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
        match self.delegates.get(key) {
            Some(da) => Ok(da.borrow(py).read(py)?.unbind()),
            None => Ok(default.unwrap_or_else(|| py.None())),
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for da in self.delegates.values() {
            visit.call(da)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.delegates.clear();
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other_dict) = other.cast::<DelegatedDict>() {
            let other_ref = other_dict.borrow();
            if self.delegates.len() != other_ref.delegates.len() {
                return Ok(false);
            }
            for key in self.delegates.keys() {
                if !other_ref.delegates.contains_key(key) {
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
            if self.delegates.len() != other_dict.len() {
                return Ok(false);
            }
            for key in self.delegates.keys() {
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
