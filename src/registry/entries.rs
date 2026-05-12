use std::sync::Arc;
use std::{cmp::Ordering, cmp::PartialOrd, hash::Hash, hash::Hasher};

use derive_where::derive_where;
use pyo3::{Py, PyAny};

use crate::python_identity::PythonIdentity;
use crate::types::{
    CallableKey, Parametric, ProtocolKey, PyTypeParametricKey, TypeVarSupport, TypedDictKey,
};

fn python_identity(object: &Arc<Py<PyAny>>) -> PythonIdentity {
    PythonIdentity::from_arc_py_any(object)
}

#[derive_where(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum SourceType<'ty, G: TypeVarSupport> {
    TypedDict(TypedDictKey<'ty, G>),
    Protocol(ProtocolKey<'ty, G>),
}

#[derive(Clone)]
pub(crate) struct Constructor<'ty> {
    pub(crate) fn_type: CallableKey<'ty, Parametric>,
    pub(crate) implementation: Arc<Py<PyAny>>,
}

impl PartialEq for Constructor<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.fn_type == other.fn_type
            && python_identity(&self.implementation) == python_identity(&other.implementation)
    }
}

impl Eq for Constructor<'_> {}

impl Hash for Constructor<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.fn_type.hash(state);
        python_identity(&self.implementation).hash(state);
    }
}

impl PartialOrd for Constructor<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Constructor<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.fn_type.cmp(&other.fn_type).then_with(|| {
            python_identity(&self.implementation).cmp(&python_identity(&other.implementation))
        })
    }
}

#[derive(Clone)]
pub(crate) struct MethodImplementation<'ty> {
    pub(crate) name: Arc<str>,
    pub(crate) registration_protocol: ProtocolKey<'ty, Parametric>,
    pub(crate) public_fn_type: CallableKey<'ty, Parametric>,
    pub(crate) implementation_fn_type: CallableKey<'ty, Parametric>,
    pub(crate) implementation: Arc<Py<PyAny>>,
    pub(crate) bound_to: Option<PyTypeParametricKey<'ty>>,
    pub(crate) order: usize,
}

impl PartialEq for MethodImplementation<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.registration_protocol == other.registration_protocol
            && self.public_fn_type == other.public_fn_type
            && self.implementation_fn_type == other.implementation_fn_type
            && self.bound_to == other.bound_to
            && self.order == other.order
            && python_identity(&self.implementation) == python_identity(&other.implementation)
    }
}

impl Eq for MethodImplementation<'_> {}

impl Hash for MethodImplementation<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.registration_protocol.hash(state);
        self.public_fn_type.hash(state);
        self.implementation_fn_type.hash(state);
        self.bound_to.hash(state);
        self.order.hash(state);
        python_identity(&self.implementation).hash(state);
    }
}

impl PartialOrd for MethodImplementation<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MethodImplementation<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.name
            .cmp(&other.name)
            .then_with(|| self.order.cmp(&other.order))
            .then_with(|| self.registration_protocol.cmp(&other.registration_protocol))
            .then_with(|| self.public_fn_type.cmp(&other.public_fn_type))
            .then_with(|| {
                self.implementation_fn_type
                    .cmp(&other.implementation_fn_type)
            })
            .then_with(|| self.bound_to.cmp(&other.bound_to))
            .then_with(|| {
                python_identity(&self.implementation).cmp(&python_identity(&other.implementation))
            })
    }
}
