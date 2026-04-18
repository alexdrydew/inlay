use std::sync::Arc;
use std::{cmp::Ordering, hash::Hash, hash::Hasher};

use pyo3::{Py, PyAny};

// --- Function identity ---

#[derive(Clone)]
pub(crate) enum FnIdentity {
    Explicit(Arc<Py<PyAny>>),
    AutoMethod,
}

fn python_object_addr(object: &Arc<Py<PyAny>>) -> usize {
    object.as_ref().as_ptr() as usize
}

fn same_python_object(a: &Arc<Py<PyAny>>, b: &Arc<Py<PyAny>>) -> bool {
    a.as_ref().as_ptr() == b.as_ref().as_ptr()
}

fn cmp_python_object(a: &Arc<Py<PyAny>>, b: &Arc<Py<PyAny>>) -> Ordering {
    python_object_addr(a).cmp(&python_object_addr(b))
}

fn hash_python_object<H: Hasher>(object: &Arc<Py<PyAny>>, state: &mut H) {
    object.as_ref().as_ptr().hash(state);
}

impl PartialEq for FnIdentity {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Explicit(a), Self::Explicit(b)) => same_python_object(a, b),
            (Self::AutoMethod, Self::AutoMethod) => true,
            _ => false,
        }
    }
}

impl Eq for FnIdentity {}

impl PartialOrd for FnIdentity {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FnIdentity {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::AutoMethod, Self::AutoMethod) => Ordering::Equal,
            (Self::AutoMethod, Self::Explicit(_)) => Ordering::Less,
            (Self::Explicit(_), Self::AutoMethod) => Ordering::Greater,
            (Self::Explicit(a), Self::Explicit(b)) => cmp_python_object(a, b),
        }
    }
}

impl Hash for FnIdentity {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        if let Self::Explicit(function) = self {
            hash_python_object(function, state);
        }
    }
}

// --- Source ---

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PyArg {
    pub(crate) name: Arc<str>,
    pub(crate) function: FnIdentity,
}

#[derive(Clone)]
pub(crate) enum SourceKind {
    FnResult(Arc<Py<PyAny>>), // function identity
    FnArg(PyArg),
}

impl PartialEq for SourceKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::FnResult(a), Self::FnResult(b)) => same_python_object(a, b),
            (Self::FnArg(a), Self::FnArg(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for SourceKind {}

impl PartialOrd for SourceKind {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SourceKind {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::FnArg(a), Self::FnArg(b)) => a.cmp(b),
            (Self::FnArg(_), Self::FnResult(_)) => Ordering::Less,
            (Self::FnResult(_), Self::FnArg(_)) => Ordering::Greater,
            (Self::FnResult(a), Self::FnResult(b)) => cmp_python_object(a, b),
        }
    }
}

impl Hash for SourceKind {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::FnResult(function) => hash_python_object(function, state),
            Self::FnArg(arg) => arg.hash(state),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Source {
    pub(crate) kind: SourceKind,
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;

    use pyo3::Python;
    use pyo3::types::PyDict;

    use super::*;

    fn hash_value(value: &impl Hash) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn identity_uses_python_object_not_arc_allocation() {
        Python::initialize();

        let (identities_equal, identity_hashes_equal, sources_equal, source_hashes_equal) =
            Python::attach(|py| {
                let object = PyDict::new(py).into_any().unbind();
                let left = Arc::new(object.clone_ref(py));
                let right = Arc::new(object);

                let left_identity = FnIdentity::Explicit(left.clone());
                let right_identity = FnIdentity::Explicit(right.clone());

                let left_source = SourceKind::FnResult(left);
                let right_source = SourceKind::FnResult(right);

                (
                    left_identity == right_identity,
                    hash_value(&left_identity) == hash_value(&right_identity),
                    left_source == right_source,
                    hash_value(&left_source) == hash_value(&right_source),
                )
            });

        assert!(identities_equal);
        assert!(identity_hashes_equal);
        assert!(sources_equal);
        assert!(source_hashes_equal);
    }
}
