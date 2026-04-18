use std::sync::Arc;
use std::{cmp::Ordering, cmp::PartialOrd, hash::Hash, hash::Hasher};

use derive_where::derive_where;
use pyo3::{Py, PyAny};

use crate::types::{
    ArenaFamily, CallableKey, Concrete, Parametric, PlainKey, ProtocolKey, PyType,
    PyTypeConcreteKey, PyTypeParametricKey, TypeVarSupport, TypedDictKey,
};

#[derive_where(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum SourceType<S: ArenaFamily, G: TypeVarSupport> {
    TypedDict(TypedDictKey<S, G>),
    Protocol(ProtocolKey<S, G>),
}

#[derive_where(Clone)]
pub(crate) struct Constructor<S: ArenaFamily> {
    pub(crate) fn_type: CallableKey<S, Parametric>,
    pub(crate) implementation: Arc<Py<PyAny>>,
}

impl<S: ArenaFamily> PartialEq for Constructor<S> {
    fn eq(&self, other: &Self) -> bool {
        self.fn_type == other.fn_type
            && self.implementation.as_ref().as_ptr() == other.implementation.as_ref().as_ptr()
    }
}

impl<S: ArenaFamily> Eq for Constructor<S> {}

impl<S: ArenaFamily> Hash for Constructor<S> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.fn_type.hash(state);
        self.implementation.as_ref().as_ptr().hash(state);
    }
}

impl<S: ArenaFamily> PartialOrd for Constructor<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<S: ArenaFamily> Ord for Constructor<S> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.fn_type.cmp(&other.fn_type).then_with(|| {
            (self.implementation.as_ref().as_ptr() as usize)
                .cmp(&(other.implementation.as_ref().as_ptr() as usize))
        })
    }
}

#[derive_where(Clone)]
pub(crate) struct MethodImplementation<S: ArenaFamily> {
    pub(crate) name: Arc<str>,
    pub(crate) fn_type: CallableKey<S, Parametric>,
    pub(crate) implementation: Arc<Py<PyAny>>,
    pub(crate) bound_to: Option<PyTypeParametricKey<S>>,
}

impl<S: ArenaFamily> PartialEq for MethodImplementation<S> {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.fn_type == other.fn_type
            && self.bound_to == other.bound_to
            && self.implementation.as_ref().as_ptr() == other.implementation.as_ref().as_ptr()
    }
}

impl<S: ArenaFamily> Eq for MethodImplementation<S> {}

impl<S: ArenaFamily> Hash for MethodImplementation<S> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.fn_type.hash(state);
        self.bound_to.hash(state);
        self.implementation.as_ref().as_ptr().hash(state);
    }
}

impl<S: ArenaFamily> PartialOrd for MethodImplementation<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<S: ArenaFamily> Ord for MethodImplementation<S> {
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

#[derive_where(Clone)]
pub(crate) struct Hook<S: ArenaFamily> {
    pub(crate) name: Arc<str>,
    pub(crate) fn_type: CallableKey<S, Parametric>,
    pub(crate) implementation: Arc<Py<PyAny>>,
}

impl<S: ArenaFamily> PartialEq for Hook<S> {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.fn_type == other.fn_type
            && self.implementation.as_ref().as_ptr() == other.implementation.as_ref().as_ptr()
    }
}

impl<S: ArenaFamily> Eq for Hook<S> {}

impl<S: ArenaFamily> Hash for Hook<S> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.fn_type.hash(state);
        self.implementation.as_ref().as_ptr().hash(state);
    }
}

impl<S: ArenaFamily> PartialOrd for Hook<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<S: ArenaFamily> Ord for Hook<S> {
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

// --- ConstantType ---

#[derive_where(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum ConstantType<S: ArenaFamily> {
    Plain(PlainKey<S, Concrete>),
    Protocol(ProtocolKey<S, Concrete>),
    TypedDict(TypedDictKey<S, Concrete>),
}

impl<S: ArenaFamily> From<ConstantType<S>> for PyTypeConcreteKey<S> {
    fn from(value: ConstantType<S>) -> Self {
        match value {
            ConstantType::Plain(k) => PyType::Plain(k),
            ConstantType::Protocol(k) => PyType::Protocol(k),
            ConstantType::TypedDict(k) => PyType::TypedDict(k),
        }
    }
}

/// Try to narrow a `ConcreteRef` to a `ConstantType`.
///
/// Only `Plain`, `Protocol`, and `TypedDict` can appear as scope constants.
pub(crate) fn to_constant_type<S: ArenaFamily>(
    type_ref: PyTypeConcreteKey<S>,
) -> Option<ConstantType<S>> {
    match type_ref {
        PyType::Plain(k) => Some(ConstantType::Plain(k)),
        PyType::Protocol(k) => Some(ConstantType::Protocol(k)),
        PyType::TypedDict(k) => Some(ConstantType::TypedDict(k)),
        _ => None,
    }
}
