use std::collections::HashMap;
use std::hash::Hash;

use crate::qualifier::Qualifier;

use super::{
    Arena, ArenaFamily, Bindings, Concrete, MapChildren, OpaqueParamSpec, OpaqueTypeVar,
    Parametric, PyType, PyTypeConcreteKey, PyTypeParametricKey, Qualified, SentinelType,
    StoreGroup, TypeArenas,
};

// --- TypeArenas method ---

impl<S: ArenaFamily> TypeArenas<S> {
    pub(crate) fn apply_bindings(
        &mut self,
        source: PyTypeParametricKey<S>,
        bindings: &Bindings<S>,
    ) -> PyTypeConcreteKey<S> {
        apply_bindings_inner(
            source,
            bindings,
            &mut self.concrete,
            &self.parametric,
            &self.sentinels,
            &mut HashMap::new(),
        )
    }
}

fn apply_bindings_inner<S: ArenaFamily>(
    source: PyTypeParametricKey<S>,
    bindings: &Bindings<S>,
    concrete: &mut StoreGroup<S, Concrete>,
    parametric: &StoreGroup<S, Parametric>,
    sentinels: &S::Store<Qualified<SentinelType>>,
    memo: &mut HashMap<PyTypeParametricKey<S>, PyTypeConcreteKey<S>>,
) -> PyTypeConcreteKey<S> {
    if let Some(&cached) = memo.get(&source) {
        return cached;
    }

    let result = match source {
        // Binding lookup uses Python TypeVar identity (PyTypeId), not arena
        // slot key. The same logical TypeVar may occupy different slots due to
        // different qualifier contexts (return type vs params after include).
        PyType::TypeVar(key) => {
            let tv = parametric.type_vars.get(&key).expect("dangling key");
            match bindings.type_vars.get(&tv.inner.descriptor.id) {
                Some(&bound) => {
                    if tv.qualifier.is_unqualified() {
                        bound
                    } else {
                        requalify_concrete(bound, &tv.qualifier, concrete)
                    }
                }
                None => {
                    let opaque = Qualified {
                        inner: OpaqueTypeVar {
                            descriptor: tv.inner.descriptor.clone(),
                        },
                        qualifier: tv.qualifier.clone(),
                    };
                    PyType::TypeVar(concrete.type_vars.insert(opaque))
                }
            }
        }
        PyType::ParamSpec(key) => {
            let ps = parametric.param_specs.get(&key).expect("dangling key");
            match bindings.param_specs.get(&ps.inner.descriptor.id) {
                Some(&bound) => {
                    if ps.qualifier.is_unqualified() {
                        bound
                    } else {
                        requalify_concrete(bound, &ps.qualifier, concrete)
                    }
                }
                None => {
                    let opaque = Qualified {
                        inner: OpaqueParamSpec {
                            descriptor: ps.inner.descriptor.clone(),
                        },
                        qualifier: ps.qualifier.clone(),
                    };
                    PyType::ParamSpec(concrete.param_specs.insert(opaque))
                }
            }
        }
        PyType::Sentinel(key) => PyType::Sentinel(key),
        PyType::Plain(key) => {
            let placeholder = concrete.plains.insert_placeholder();
            let result = PyType::Plain(placeholder);
            memo.insert(source, result);
            let val = parametric.plains.get(&key).expect("dangling key").clone();
            let output = val.map_children(&mut |k| {
                apply_bindings_inner(k, bindings, concrete, parametric, sentinels, memo)
            });
            assert!(
                concrete
                    .plains
                    .replace(placeholder, output)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::Protocol(key) => {
            let placeholder = concrete.protocols.insert_placeholder();
            let result = PyType::Protocol(placeholder);
            memo.insert(source, result);
            let val = parametric
                .protocols
                .get(&key)
                .expect("dangling key")
                .clone();
            let output = val.map_children(&mut |k| {
                apply_bindings_inner(k, bindings, concrete, parametric, sentinels, memo)
            });
            assert!(
                concrete
                    .protocols
                    .replace(placeholder, output)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::TypedDict(key) => {
            let placeholder = concrete.typed_dicts.insert_placeholder();
            let result = PyType::TypedDict(placeholder);
            memo.insert(source, result);
            let val = parametric
                .typed_dicts
                .get(&key)
                .expect("dangling key")
                .clone();
            let output = val.map_children(&mut |k| {
                apply_bindings_inner(k, bindings, concrete, parametric, sentinels, memo)
            });
            assert!(
                concrete
                    .typed_dicts
                    .replace(placeholder, output)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::Union(key) => {
            let placeholder = concrete.unions.insert_placeholder();
            let result = PyType::Union(placeholder);
            memo.insert(source, result);
            let val = parametric.unions.get(&key).expect("dangling key").clone();
            let output = val.map_children(&mut |k| {
                apply_bindings_inner(k, bindings, concrete, parametric, sentinels, memo)
            });
            assert!(
                concrete
                    .unions
                    .replace(placeholder, output)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::Callable(key) => {
            let placeholder = concrete.callables.insert_placeholder();
            let result = PyType::Callable(placeholder);
            memo.insert(source, result);
            let val = parametric
                .callables
                .get(&key)
                .expect("dangling key")
                .clone();
            let output = val.map_children(&mut |k| {
                apply_bindings_inner(k, bindings, concrete, parametric, sentinels, memo)
            });
            assert!(
                concrete
                    .callables
                    .replace(placeholder, output)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::LazyRef(key) => {
            let placeholder = concrete.lazy_refs.insert_placeholder();
            let result = PyType::LazyRef(placeholder);
            memo.insert(source, result);
            let val = parametric
                .lazy_refs
                .get(&key)
                .expect("dangling key")
                .clone();
            let output = val.map_children(&mut |k| {
                apply_bindings_inner(k, bindings, concrete, parametric, sentinels, memo)
            });
            assert!(
                concrete
                    .lazy_refs
                    .replace(placeholder, output)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
    };

    memo.insert(source, result);
    result
}

fn reinsert_requalified<T, A>(store: &mut A, key: A::Key, additional: &Qualifier) -> A::Key
where
    T: Hash + Eq + Clone + 'static,
    A: Arena<Qualified<T>>,
{
    let (inner, new_qual) = {
        let val = store.get(&key).expect("dangling key");
        let new_qual = val.qualifier.intersect(additional);
        if new_qual == val.qualifier {
            return key;
        }
        (val.inner.clone(), new_qual)
    };
    store.insert(Qualified {
        inner,
        qualifier: new_qual,
    })
}

pub(crate) fn requalify_concrete<S: ArenaFamily>(
    target: PyTypeConcreteKey<S>,
    additional: &Qualifier,
    concrete: &mut StoreGroup<S, Concrete>,
) -> PyTypeConcreteKey<S> {
    match target {
        PyType::Sentinel(_) => target,
        PyType::TypeVar(key) => PyType::TypeVar(reinsert_requalified(
            &mut concrete.type_vars,
            key,
            additional,
        )),
        PyType::ParamSpec(key) => PyType::ParamSpec(reinsert_requalified(
            &mut concrete.param_specs,
            key,
            additional,
        )),
        PyType::Plain(key) => {
            PyType::Plain(reinsert_requalified(&mut concrete.plains, key, additional))
        }
        PyType::Protocol(key) => PyType::Protocol(reinsert_requalified(
            &mut concrete.protocols,
            key,
            additional,
        )),
        PyType::TypedDict(key) => PyType::TypedDict(reinsert_requalified(
            &mut concrete.typed_dicts,
            key,
            additional,
        )),
        PyType::Union(key) => {
            PyType::Union(reinsert_requalified(&mut concrete.unions, key, additional))
        }
        PyType::Callable(key) => PyType::Callable(reinsert_requalified(
            &mut concrete.callables,
            key,
            additional,
        )),
        PyType::LazyRef(key) => PyType::LazyRef(reinsert_requalified(
            &mut concrete.lazy_refs,
            key,
            additional,
        )),
    }
}
