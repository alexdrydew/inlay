use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::PyTraverseError;
use pyo3::exceptions::{PyAttributeError, PyKeyError};
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use crate::types::MemberAccessKind;

use super::executor::{ScopeHandle, attach_scope};

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
        if let Ok(member) = bound.cast::<DelegatedMember>() {
            Ok(member.borrow().read(py)?.unbind())
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
        if let Ok(member) = current_bound.cast::<DelegatedMember>() {
            member.borrow().write(py, value.bind(py))
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

#[pyclass(module = "inlay")]
pub(crate) struct DelegatedDict {
    delegates: HashMap<Arc<str>, Py<DelegatedMember>>,
}

impl DelegatedDict {
    pub(crate) fn new(delegates: HashMap<Arc<str>, Py<DelegatedMember>>) -> Self {
        Self { delegates }
    }
}

#[pymethods]
impl DelegatedDict {
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        let member = self
            .delegates
            .get(key)
            .ok_or_else(|| PyKeyError::new_err(key.to_owned()))?;
        Ok(member.borrow(py).read(py)?.unbind())
    }

    fn __setitem__(&self, py: Python<'_>, key: &str, value: Py<PyAny>) -> PyResult<()> {
        let member = self
            .delegates
            .get(key)
            .ok_or_else(|| PyKeyError::new_err(key.to_owned()))?;
        member.borrow(py).write(py, value.bind(py))
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
            .map(|member| member.borrow(py).read(py).map(|v| v.unbind()))
            .collect::<PyResult<_>>()?;
        PyList::new(py, &vals)
    }

    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Bound<'py, PyTuple>> = self
            .delegates
            .iter()
            .map(|(k, member)| {
                let v = member.borrow(py).read(py)?;
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
            Some(member) => Ok(member.borrow(py).read(py)?.unbind()),
            None => Ok(default.unwrap_or_else(|| py.None())),
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for member in self.delegates.values() {
            visit.call(member)?;
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
