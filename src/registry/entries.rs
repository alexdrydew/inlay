use std::sync::Arc;
use std::{cmp::Ordering, cmp::PartialOrd, hash::Hash, hash::Hasher};

use derive_where::derive_where;
use pyo3::{Py, PyAny};

use crate::types::{
    CallableKey, Parametric, ProtocolKey, PyTypeParametricKey, TypeVarSupport, TypedDictKey,
};

#[derive_where(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum SourceType<G: TypeVarSupport> {
    TypedDict(TypedDictKey<G>),
    Protocol(ProtocolKey<G>),
}

#[derive(Clone)]
pub(crate) struct Constructor {
    pub(crate) fn_type: CallableKey<Parametric>,
    pub(crate) implementation: Arc<Py<PyAny>>,
}

impl PartialEq for Constructor {
    fn eq(&self, other: &Self) -> bool {
        self.fn_type == other.fn_type
            && self.implementation.as_ref().as_ptr() == other.implementation.as_ref().as_ptr()
    }
}

impl Eq for Constructor {}

impl Hash for Constructor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.fn_type.hash(state);
        self.implementation.as_ref().as_ptr().hash(state);
    }
}

impl PartialOrd for Constructor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Constructor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.fn_type.cmp(&other.fn_type).then_with(|| {
            (self.implementation.as_ref().as_ptr() as usize)
                .cmp(&(other.implementation.as_ref().as_ptr() as usize))
        })
    }
}

#[derive(Clone)]
pub(crate) struct MethodImplementation {
    pub(crate) name: Arc<str>,
    pub(crate) fn_type: CallableKey<Parametric>,
    pub(crate) implementation: Arc<Py<PyAny>>,
    pub(crate) bound_to: Option<PyTypeParametricKey>,
}

impl PartialEq for MethodImplementation {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.fn_type == other.fn_type
            && self.bound_to == other.bound_to
            && self.implementation.as_ref().as_ptr() == other.implementation.as_ref().as_ptr()
    }
}

impl Eq for MethodImplementation {}

impl Hash for MethodImplementation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.fn_type.hash(state);
        self.bound_to.hash(state);
        self.implementation.as_ref().as_ptr().hash(state);
    }
}

impl PartialOrd for MethodImplementation {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MethodImplementation {
    fn cmp(&self, other: &Self) -> Ordering {
        self.name
            .cmp(&other.name)
            .then_with(|| self.fn_type.cmp(&other.fn_type))
            .then_with(|| self.bound_to.cmp(&other.bound_to))
            .then_with(|| {
                (self.implementation.as_ref().as_ptr() as usize)
                    .cmp(&(other.implementation.as_ref().as_ptr() as usize))
            })
    }
}

#[derive(Clone)]
pub(crate) struct Hook {
    pub(crate) name: Arc<str>,
    pub(crate) fn_type: CallableKey<Parametric>,
    pub(crate) implementation: Arc<Py<PyAny>>,
}

impl PartialEq for Hook {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.fn_type == other.fn_type
            && self.implementation.as_ref().as_ptr() == other.implementation.as_ref().as_ptr()
    }
}

impl Eq for Hook {}

impl Hash for Hook {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.fn_type.hash(state);
        self.implementation.as_ref().as_ptr().hash(state);
    }
}

impl PartialOrd for Hook {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hook {
    fn cmp(&self, other: &Self) -> Ordering {
        self.name
            .cmp(&other.name)
            .then_with(|| self.fn_type.cmp(&other.fn_type))
            .then_with(|| {
                (self.implementation.as_ref().as_ptr() as usize)
                    .cmp(&(other.implementation.as_ref().as_ptr() as usize))
            })
    }
}
