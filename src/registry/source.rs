use std::sync::Arc;
use std::{cmp::Ordering, hash::Hash, hash::Hasher};

use derive_where::derive_where;
use pyo3::{Py, PyAny};

use super::ConstantType;
use crate::types::ArenaFamily;

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

#[derive_where(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct TransitionBindingKey<S: ArenaFamily> {
    pub(crate) name: Arc<str>,
    pub(crate) constant_type: ConstantType<S>,
}

impl<S: ArenaFamily> TransitionBindingKey<S> {
    pub(crate) fn from_constant_type(name: Arc<str>, constant_type: ConstantType<S>) -> Self {
        Self {
            name,
            constant_type,
        }
    }
}

#[derive_where(Clone)]
pub(crate) enum SourceKind<S: ArenaFamily> {
    ProviderResult(Arc<Py<PyAny>>),
    TransitionBinding(TransitionBindingKey<S>),
    TransitionResult(ConstantType<S>),
}

impl<S: ArenaFamily> PartialEq for SourceKind<S> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ProviderResult(a), Self::ProviderResult(b)) => same_python_object(a, b),
            (Self::TransitionBinding(a), Self::TransitionBinding(b)) => a == b,
            (Self::TransitionResult(a), Self::TransitionResult(b)) => a == b,
            _ => false,
        }
    }
}

impl<S: ArenaFamily> Eq for SourceKind<S> {}

impl<S: ArenaFamily> PartialOrd for SourceKind<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<S: ArenaFamily> Ord for SourceKind<S> {
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

impl<S: ArenaFamily> Hash for SourceKind<S> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::ProviderResult(function) => hash_python_object(function, state),
            Self::TransitionBinding(binding) => binding.hash(state),
            Self::TransitionResult(constant_type) => constant_type.hash(state),
        }
    }
}

#[derive_where(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Source<S: ArenaFamily> {
    pub(crate) kind: SourceKind<S>,
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;

    use pyo3::Python;
    use pyo3::types::PyDict;

    use super::*;
    use crate::qualifier::Qualifier;
    use crate::types::storage::Arena;
    use crate::types::{Concrete, Keyed, Qual, Qualified, SlotBackend, TypeArenas};
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

            let left_source = SourceKind::<SlotBackend>::ProviderResult(left);
            let right_source = SourceKind::<SlotBackend>::ProviderResult(right);

            (
                left_source == right_source,
                hash_value(&left_source) == hash_value(&right_source),
            )
        });

        assert!(sources_equal);
        assert!(source_hashes_equal);
    }

    #[test]
    fn arg_identity_uses_name_and_constant_type() {
        let mut arenas = TypeArenas::<SlotBackend>::default();
        let first = arenas.concrete.plains.insert(Qualified {
            inner: PlainType::<Qual<Keyed<SlotBackend>>, Concrete> {
                descriptor: PyTypeDescriptor {
                    id: crate::types::PyTypeId::new("First".to_string()),
                    display_name: Arc::from("First"),
                },
                args: Vec::new(),
            },
            qualifier: Qualifier::any(),
        });
        let second = arenas.concrete.plains.insert(Qualified {
            inner: PlainType::<Qual<Keyed<SlotBackend>>, Concrete> {
                descriptor: PyTypeDescriptor {
                    id: crate::types::PyTypeId::new("Second".to_string()),
                    display_name: Arc::from("Second"),
                },
                args: Vec::new(),
            },
            qualifier: Qualifier::any(),
        });

        let left = SourceKind::TransitionBinding(TransitionBindingKey::<SlotBackend> {
            name: Arc::from("session_id"),
            constant_type: ConstantType::Plain(first),
        });
        let right = SourceKind::TransitionBinding(TransitionBindingKey::<SlotBackend> {
            name: Arc::from("session_id"),
            constant_type: ConstantType::Plain(first),
        });
        let different_type = SourceKind::TransitionBinding(TransitionBindingKey::<SlotBackend> {
            name: Arc::from("session_id"),
            constant_type: ConstantType::Plain(second),
        });
        let different_name = SourceKind::TransitionBinding(TransitionBindingKey::<SlotBackend> {
            name: Arc::from("branch_id"),
            constant_type: ConstantType::Plain(first),
        });

        assert!(left == right);
        assert_eq!(hash_value(&left), hash_value(&right));
        assert!(left != different_type);
        assert!(left != different_name);
    }

    #[test]
    fn transition_result_identity_uses_constant_type() {
        let mut arenas = TypeArenas::<SlotBackend>::default();
        let first = arenas.concrete.plains.insert(Qualified {
            inner: PlainType::<Qual<Keyed<SlotBackend>>, Concrete> {
                descriptor: PyTypeDescriptor {
                    id: crate::types::PyTypeId::new("FirstResult".to_string()),
                    display_name: Arc::from("FirstResult"),
                },
                args: Vec::new(),
            },
            qualifier: Qualifier::any(),
        });
        let second = arenas.concrete.plains.insert(Qualified {
            inner: PlainType::<Qual<Keyed<SlotBackend>>, Concrete> {
                descriptor: PyTypeDescriptor {
                    id: crate::types::PyTypeId::new("SecondResult".to_string()),
                    display_name: Arc::from("SecondResult"),
                },
                args: Vec::new(),
            },
            qualifier: Qualifier::any(),
        });

        let left = SourceKind::<SlotBackend>::TransitionResult(ConstantType::Plain(first));
        let right = SourceKind::<SlotBackend>::TransitionResult(ConstantType::Plain(first));
        let different = SourceKind::<SlotBackend>::TransitionResult(ConstantType::Plain(second));

        assert!(left == right);
        assert_eq!(hash_value(&left), hash_value(&right));
        assert!(left != different);
    }
}
