use std::sync::Arc;
use std::{cmp::Ordering, hash::Hash, hash::Hasher};

use pyo3::{Py, PyAny};

use crate::types::PyTypeConcreteKey;

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

// --- Source ---

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct TransitionBindingKey {
    pub(crate) name: Arc<str>,
    pub(crate) type_ref: PyTypeConcreteKey,
}

impl TransitionBindingKey {
    pub(crate) fn from_type_ref(name: Arc<str>, type_ref: PyTypeConcreteKey) -> Self {
        Self { name, type_ref }
    }
}

#[derive(Clone)]
pub(crate) enum SourceKind {
    ProviderResult(Arc<Py<PyAny>>),
    TransitionBinding(TransitionBindingKey),
    TransitionResult(PyTypeConcreteKey),
}

impl PartialEq for SourceKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ProviderResult(a), Self::ProviderResult(b)) => same_python_object(a, b),
            (Self::TransitionBinding(a), Self::TransitionBinding(b)) => a == b,
            (Self::TransitionResult(a), Self::TransitionResult(b)) => a == b,
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
            (Self::TransitionBinding(a), Self::TransitionBinding(b)) => a.cmp(b),
            (Self::TransitionBinding(_), _) => Ordering::Less,
            (_, Self::TransitionBinding(_)) => Ordering::Greater,
            (Self::TransitionResult(a), Self::TransitionResult(b)) => a.cmp(b),
            (Self::TransitionResult(_), Self::ProviderResult(_)) => Ordering::Less,
            (Self::ProviderResult(_), Self::TransitionResult(_)) => Ordering::Greater,
            (Self::ProviderResult(a), Self::ProviderResult(b)) => cmp_python_object(a, b),
        }
    }
}

impl Hash for SourceKind {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::ProviderResult(function) => hash_python_object(function, state),
            Self::TransitionBinding(binding) => binding.hash(state),
            Self::TransitionResult(type_ref) => type_ref.hash(state),
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
    use crate::qualifier::Qualifier;
    use crate::types::{Concrete, Keyed, PyType, Qual, Qualified, TypeArenas};
    use crate::types::{PlainType, PyTypeDescriptor};

    fn hash_value(value: &impl Hash) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn identity_uses_python_object_not_arc_allocation() {
        Python::initialize();

        let (sources_equal, source_hashes_equal) = Python::attach(|py| {
            let object = PyDict::new(py).into_any().unbind();
            let left = Arc::new(object.clone_ref(py));
            let right = Arc::new(object);

            let left_source = SourceKind::ProviderResult(left);
            let right_source = SourceKind::ProviderResult(right);

            (
                left_source == right_source,
                hash_value(&left_source) == hash_value(&right_source),
            )
        });

        assert!(sources_equal);
        assert!(source_hashes_equal);
    }

    #[test]
    fn arg_identity_uses_name_and_type() {
        let mut arenas = TypeArenas::default();
        let first = arenas.concrete.plains.insert(Some(Qualified {
            inner: PlainType::<Qual<Keyed>, Concrete> {
                descriptor: PyTypeDescriptor {
                    id: crate::types::PyTypeId::new("First".to_string()),
                    display_name: Arc::from("First"),
                },
                args: Vec::new(),
            },
            qualifier: Qualifier::any(),
        }));
        let second = arenas.concrete.plains.insert(Some(Qualified {
            inner: PlainType::<Qual<Keyed>, Concrete> {
                descriptor: PyTypeDescriptor {
                    id: crate::types::PyTypeId::new("Second".to_string()),
                    display_name: Arc::from("Second"),
                },
                args: Vec::new(),
            },
            qualifier: Qualifier::any(),
        }));

        let left = SourceKind::TransitionBinding(TransitionBindingKey {
            name: Arc::from("session_id"),
            type_ref: PyType::Plain(first),
        });
        let right = SourceKind::TransitionBinding(TransitionBindingKey {
            name: Arc::from("session_id"),
            type_ref: PyType::Plain(first),
        });
        let different_type = SourceKind::TransitionBinding(TransitionBindingKey {
            name: Arc::from("session_id"),
            type_ref: PyType::Plain(second),
        });
        let different_name = SourceKind::TransitionBinding(TransitionBindingKey {
            name: Arc::from("branch_id"),
            type_ref: PyType::Plain(first),
        });

        assert!(left == right);
        assert_eq!(hash_value(&left), hash_value(&right));
        assert!(left != different_type);
        assert!(left != different_name);
    }

    #[test]
    fn transition_result_identity_uses_type() {
        let mut arenas = TypeArenas::default();
        let first = arenas.concrete.plains.insert(Some(Qualified {
            inner: PlainType::<Qual<Keyed>, Concrete> {
                descriptor: PyTypeDescriptor {
                    id: crate::types::PyTypeId::new("FirstResult".to_string()),
                    display_name: Arc::from("FirstResult"),
                },
                args: Vec::new(),
            },
            qualifier: Qualifier::any(),
        }));
        let second = arenas.concrete.plains.insert(Some(Qualified {
            inner: PlainType::<Qual<Keyed>, Concrete> {
                descriptor: PyTypeDescriptor {
                    id: crate::types::PyTypeId::new("SecondResult".to_string()),
                    display_name: Arc::from("SecondResult"),
                },
                args: Vec::new(),
            },
            qualifier: Qualifier::any(),
        }));

        let left = SourceKind::TransitionResult(PyType::Plain(first));
        let right = SourceKind::TransitionResult(PyType::Plain(first));
        let different = SourceKind::TransitionResult(PyType::Plain(second));

        assert!(left == right);
        assert_eq!(hash_value(&left), hash_value(&right));
        assert!(left != different);
    }
}
