use std::sync::Arc;
use std::{cmp::Ordering, hash::Hash, hash::Hasher};

use pyo3::{Py, PyAny};

use crate::python_identity::PythonIdentity;
use crate::types::PyTypeConcreteKey;

fn python_object_identity(object: &Arc<Py<PyAny>>) -> PythonIdentity {
    PythonIdentity::from_arc_py_any(object)
}

fn same_python_object(a: &Arc<Py<PyAny>>, b: &Arc<Py<PyAny>>) -> bool {
    python_object_identity(a) == python_object_identity(b)
}

fn cmp_python_object(a: &Arc<Py<PyAny>>, b: &Arc<Py<PyAny>>) -> Ordering {
    python_object_identity(a).cmp(&python_object_identity(b))
}

fn hash_python_object<H: Hasher>(object: &Arc<Py<PyAny>>, state: &mut H) {
    python_object_identity(object).hash(state);
}

#[derive(Clone)]
pub(crate) enum SourceKind<'ty> {
    ProviderResult(Arc<Py<PyAny>>),
    Transition {
        name: Option<Arc<str>>,
        type_ref: PyTypeConcreteKey<'ty>,
    },
}

impl PartialEq for SourceKind<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ProviderResult(a), Self::ProviderResult(b)) => same_python_object(a, b),
            (
                Self::Transition {
                    name: a_name,
                    type_ref: a_type,
                },
                Self::Transition {
                    name: b_name,
                    type_ref: b_type,
                },
            ) => a_name == b_name && a_type == b_type,
            _ => false,
        }
    }
}

impl Eq for SourceKind<'_> {}

impl PartialOrd for SourceKind<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SourceKind<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (
                Self::Transition {
                    name: a_name,
                    type_ref: a_type,
                },
                Self::Transition {
                    name: b_name,
                    type_ref: b_type,
                },
            ) => a_name.cmp(b_name).then_with(|| a_type.cmp(b_type)),
            (Self::Transition { .. }, Self::ProviderResult(_)) => Ordering::Less,
            (Self::ProviderResult(_), Self::Transition { .. }) => Ordering::Greater,
            (Self::ProviderResult(a), Self::ProviderResult(b)) => cmp_python_object(a, b),
        }
    }
}

impl Hash for SourceKind<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::ProviderResult(function) => hash_python_object(function, state),
            Self::Transition { name, type_ref } => {
                name.hash(state);
                type_ref.hash(state);
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Source<'ty> {
    pub(crate) kind: SourceKind<'ty>,
}

impl<'ty> Source<'ty> {
    pub(crate) fn provider_result(provider: Arc<Py<PyAny>>) -> Self {
        Self {
            kind: SourceKind::ProviderResult(provider),
        }
    }

    pub(crate) fn transition(name: Option<Arc<str>>, type_ref: PyTypeConcreteKey<'ty>) -> Self {
        Self {
            kind: SourceKind::Transition { name, type_ref },
        }
    }

    pub(crate) fn transition_type_ref(&self) -> Option<PyTypeConcreteKey<'ty>> {
        match &self.kind {
            SourceKind::Transition { type_ref, .. } => Some(*type_ref),
            SourceKind::ProviderResult(_) => None,
        }
    }

    pub(crate) fn transition_name(&self) -> Option<&Arc<str>> {
        match &self.kind {
            SourceKind::Transition { name, .. } => name.as_ref(),
            SourceKind::ProviderResult(_) => None,
        }
    }
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
        let first = arenas.concrete.plains.insert(Qualified {
            inner: PlainType::<Qual<Keyed>, Concrete> {
                descriptor: PyTypeDescriptor {
                    id: crate::types::PyTypeId::new("First".to_string()),
                    display_name: Arc::from("First"),
                },
                args: Vec::new(),
            },
            qualifier: Qualifier::any(),
        });
        let second = arenas.concrete.plains.insert(Qualified {
            inner: PlainType::<Qual<Keyed>, Concrete> {
                descriptor: PyTypeDescriptor {
                    id: crate::types::PyTypeId::new("Second".to_string()),
                    display_name: Arc::from("Second"),
                },
                args: Vec::new(),
            },
            qualifier: Qualifier::any(),
        });

        let left = SourceKind::Transition {
            name: Some(Arc::from("session_id")),
            type_ref: PyType::Plain(first),
        };
        let right = SourceKind::Transition {
            name: Some(Arc::from("session_id")),
            type_ref: PyType::Plain(first),
        };
        let different_type = SourceKind::Transition {
            name: Some(Arc::from("session_id")),
            type_ref: PyType::Plain(second),
        };
        let different_name = SourceKind::Transition {
            name: Some(Arc::from("branch_id")),
            type_ref: PyType::Plain(first),
        };
        let anonymous = SourceKind::Transition {
            name: None,
            type_ref: PyType::Plain(first),
        };

        assert!(left == right);
        assert_eq!(hash_value(&left), hash_value(&right));
        assert!(left != different_type);
        assert!(left != different_name);
        assert!(left != anonymous);
    }

    #[test]
    fn anonymous_transition_identity_uses_type() {
        let mut arenas = TypeArenas::default();
        let first = arenas.concrete.plains.insert(Qualified {
            inner: PlainType::<Qual<Keyed>, Concrete> {
                descriptor: PyTypeDescriptor {
                    id: crate::types::PyTypeId::new("FirstResult".to_string()),
                    display_name: Arc::from("FirstResult"),
                },
                args: Vec::new(),
            },
            qualifier: Qualifier::any(),
        });
        let second = arenas.concrete.plains.insert(Qualified {
            inner: PlainType::<Qual<Keyed>, Concrete> {
                descriptor: PyTypeDescriptor {
                    id: crate::types::PyTypeId::new("SecondResult".to_string()),
                    display_name: Arc::from("SecondResult"),
                },
                args: Vec::new(),
            },
            qualifier: Qualifier::any(),
        });

        let left = SourceKind::Transition {
            name: None,
            type_ref: PyType::Plain(first),
        };
        let right = SourceKind::Transition {
            name: None,
            type_ref: PyType::Plain(first),
        };
        let different_type = SourceKind::Transition {
            name: None,
            type_ref: PyType::Plain(second),
        };
        let named = SourceKind::Transition {
            name: Some(Arc::from("result")),
            type_ref: PyType::Plain(first),
        };

        assert!(left == right);
        assert_eq!(hash_value(&left), hash_value(&right));
        assert!(left != different_type);
        assert!(left != named);
    }
}
