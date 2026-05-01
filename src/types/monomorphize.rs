use std::{collections::BTreeMap, sync::Arc};

use indexmap::IndexMap;
use inlay_instrument::instrumented;
use rustc_hash::FxHashMap as HashMap;

use crate::qualifier::Qualifier;

use super::{
    Arena, ArenaKey, Bindings, CallableType, Concrete, Keyed, LazyRefType, OpaqueParamSpec,
    OpaqueTypeVar, ParamKind, PlainType, ProtocolType, PyType, PyTypeConcreteKey, PyTypeDescriptor,
    PyTypeParametricKey, Qual, Qualified, TypeArenas, TypedDictType, UnionType, WrapperKind,
};

// --- TypeArenas method ---

impl<'types> TypeArenas<'types> {
    #[instrumented(
        name = "inlay.types.apply_bindings",
        target = "inlay",
        level = "trace",
        skip_all
    )]
    pub(crate) fn apply_bindings(
        &mut self,
        source: PyTypeParametricKey<'types>,
        bindings: &Bindings<'types>,
    ) -> PyTypeConcreteKey<'types> {
        let mut temp = TempConcreteArenas::default();
        let root = apply_bindings_inner(source, bindings, self, &mut temp, &mut HashMap::default());
        let root = commit_concrete_temp(self, temp, root);
        self.canonicalize_concrete(root)
    }

    fn canonicalize_concrete(
        &mut self,
        key: PyTypeConcreteKey<'types>,
    ) -> PyTypeConcreteKey<'types> {
        let mut canonical_concrete = std::mem::take(&mut self.canonical_concrete_qualified);
        let canonical = canonical_concrete
            .get(key, self)
            .copied()
            .unwrap_or_else(|| {
                canonical_concrete.insert(key, key, self);
                key
            });
        self.canonical_concrete_qualified = canonical_concrete;
        canonical
    }
}

fn canonicalize_if_resolved<'types>(
    key: PyTypeConcreteKey<'types>,
    arenas: &mut TypeArenas<'types>,
) -> PyTypeConcreteKey<'types> {
    arenas.canonicalize_concrete(key)
}

type ConcretePlain<'types> = Qualified<PlainType<Qual<Keyed<'types>>, Concrete>>;
type ConcreteProtocol<'types> = Qualified<ProtocolType<Qual<Keyed<'types>>, Concrete>>;
type ConcreteTypedDict<'types> = Qualified<TypedDictType<Qual<Keyed<'types>>, Concrete>>;
type ConcreteUnion<'types> = Qualified<UnionType<Qual<Keyed<'types>>, Concrete>>;
type ConcreteCallable<'types> = Qualified<CallableType<Qual<Keyed<'types>>, Concrete>>;
type ConcreteLazyRef<'types> = Qualified<LazyRefType<Qual<Keyed<'types>>, Concrete>>;

#[derive(Clone)]
struct BuildPlainType<'types, 'temp> {
    descriptor: PyTypeDescriptor,
    args: Vec<BuildConcreteKey<'types, 'temp>>,
}

#[derive(Clone)]
struct BuildProtocolType<'types, 'temp> {
    descriptor: PyTypeDescriptor,
    methods: BTreeMap<Arc<str>, BuildConcreteKey<'types, 'temp>>,
    attributes: BTreeMap<Arc<str>, BuildConcreteKey<'types, 'temp>>,
    properties: BTreeMap<Arc<str>, BuildConcreteKey<'types, 'temp>>,
    type_params: Vec<BuildConcreteKey<'types, 'temp>>,
}

#[derive(Clone)]
struct BuildTypedDictType<'types, 'temp> {
    descriptor: PyTypeDescriptor,
    attributes: BTreeMap<Arc<str>, BuildConcreteKey<'types, 'temp>>,
    type_params: Vec<BuildConcreteKey<'types, 'temp>>,
}

#[derive(Clone)]
struct BuildUnionType<'types, 'temp> {
    variants: Vec<BuildConcreteKey<'types, 'temp>>,
}

#[derive(Clone)]
struct BuildCallableType<'types, 'temp> {
    params: IndexMap<Arc<str>, BuildConcreteKey<'types, 'temp>>,
    param_kinds: Vec<ParamKind>,
    param_has_default: Vec<bool>,
    accepts_varargs: bool,
    accepts_varkw: bool,
    return_type: BuildConcreteKey<'types, 'temp>,
    return_wrapper: WrapperKind,
    type_params: Vec<BuildConcreteKey<'types, 'temp>>,
    function_name: Option<Arc<str>>,
}

#[derive(Clone)]
struct BuildLazyRefType<'types, 'temp> {
    target: BuildConcreteKey<'types, 'temp>,
}

type TempConcretePlain<'types, 'temp> = Qualified<BuildPlainType<'types, 'temp>>;
type TempConcreteProtocol<'types, 'temp> = Qualified<BuildProtocolType<'types, 'temp>>;
type TempConcreteTypedDict<'types, 'temp> = Qualified<BuildTypedDictType<'types, 'temp>>;
type TempConcreteUnion<'types, 'temp> = Qualified<BuildUnionType<'types, 'temp>>;
type TempConcreteCallable<'types, 'temp> = Qualified<BuildCallableType<'types, 'temp>>;
type TempConcreteLazyRef<'types, 'temp> = Qualified<BuildLazyRefType<'types, 'temp>>;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum BuildConcreteKey<'types, 'temp> {
    Sentinel(ArenaKey<'types, Qualified<super::SentinelType>>),
    MainTypeVar(ArenaKey<'types, Qualified<OpaqueTypeVar>>),
    TempTypeVar(ArenaKey<'temp, Qualified<OpaqueTypeVar>>),
    MainParamSpec(ArenaKey<'types, Qualified<OpaqueParamSpec>>),
    TempParamSpec(ArenaKey<'temp, Qualified<OpaqueParamSpec>>),
    MainPlain(ArenaKey<'types, ConcretePlain<'types>>),
    TempPlain(ArenaKey<'temp, TempConcretePlain<'types, 'temp>>),
    MainProtocol(ArenaKey<'types, ConcreteProtocol<'types>>),
    TempProtocol(ArenaKey<'temp, TempConcreteProtocol<'types, 'temp>>),
    MainTypedDict(ArenaKey<'types, ConcreteTypedDict<'types>>),
    TempTypedDict(ArenaKey<'temp, TempConcreteTypedDict<'types, 'temp>>),
    MainUnion(ArenaKey<'types, ConcreteUnion<'types>>),
    TempUnion(ArenaKey<'temp, TempConcreteUnion<'types, 'temp>>),
    MainCallable(ArenaKey<'types, ConcreteCallable<'types>>),
    TempCallable(ArenaKey<'temp, TempConcreteCallable<'types, 'temp>>),
    MainLazyRef(ArenaKey<'types, ConcreteLazyRef<'types>>),
    TempLazyRef(ArenaKey<'temp, TempConcreteLazyRef<'types, 'temp>>),
}

impl<'types, 'temp> BuildConcreteKey<'types, 'temp> {
    fn main(key: PyTypeConcreteKey<'types>) -> Self {
        match key {
            PyType::Sentinel(key) => Self::Sentinel(key),
            PyType::TypeVar(key) => Self::MainTypeVar(key),
            PyType::ParamSpec(key) => Self::MainParamSpec(key),
            PyType::Plain(key) => Self::MainPlain(key),
            PyType::Protocol(key) => Self::MainProtocol(key),
            PyType::TypedDict(key) => Self::MainTypedDict(key),
            PyType::Union(key) => Self::MainUnion(key),
            PyType::Callable(key) => Self::MainCallable(key),
            PyType::LazyRef(key) => Self::MainLazyRef(key),
        }
    }
}

#[derive(Default)]
struct TempConcreteArenas<'types, 'temp> {
    type_vars: Arena<'temp, Qualified<OpaqueTypeVar>, Option<Qualified<OpaqueTypeVar>>>,
    param_specs: Arena<'temp, Qualified<OpaqueParamSpec>, Option<Qualified<OpaqueParamSpec>>>,
    plains:
        Arena<'temp, TempConcretePlain<'types, 'temp>, Option<TempConcretePlain<'types, 'temp>>>,
    protocols: Arena<
        'temp,
        TempConcreteProtocol<'types, 'temp>,
        Option<TempConcreteProtocol<'types, 'temp>>,
    >,
    typed_dicts: Arena<
        'temp,
        TempConcreteTypedDict<'types, 'temp>,
        Option<TempConcreteTypedDict<'types, 'temp>>,
    >,
    unions:
        Arena<'temp, TempConcreteUnion<'types, 'temp>, Option<TempConcreteUnion<'types, 'temp>>>,
    callables: Arena<
        'temp,
        TempConcreteCallable<'types, 'temp>,
        Option<TempConcreteCallable<'types, 'temp>>,
    >,
    lazy_refs: Arena<
        'temp,
        TempConcreteLazyRef<'types, 'temp>,
        Option<TempConcreteLazyRef<'types, 'temp>>,
    >,
}

struct ConcreteCommitKeys<'types> {
    type_vars: Vec<ArenaKey<'types, Qualified<OpaqueTypeVar>>>,
    param_specs: Vec<ArenaKey<'types, Qualified<OpaqueParamSpec>>>,
    plains: Vec<ArenaKey<'types, ConcretePlain<'types>>>,
    protocols: Vec<ArenaKey<'types, ConcreteProtocol<'types>>>,
    typed_dicts: Vec<ArenaKey<'types, ConcreteTypedDict<'types>>>,
    unions: Vec<ArenaKey<'types, ConcreteUnion<'types>>>,
    callables: Vec<ArenaKey<'types, ConcreteCallable<'types>>>,
    lazy_refs: Vec<ArenaKey<'types, ConcreteLazyRef<'types>>>,
}

fn future_keys<'types, T>(store: &Arena<'types, T>, count: usize) -> Vec<ArenaKey<'types, T>> {
    (0..count).map(|offset| store.future_key(offset)).collect()
}

fn remap_temp_key<'temp, 'types, T, U>(
    key: ArenaKey<'temp, T>,
    keys: &[ArenaKey<'types, U>],
) -> ArenaKey<'types, U> {
    keys[key.index()]
}

fn commit_build_key<'types, 'temp>(
    key: BuildConcreteKey<'types, 'temp>,
    keys: &ConcreteCommitKeys<'types>,
) -> PyTypeConcreteKey<'types> {
    match key {
        BuildConcreteKey::Sentinel(key) => PyType::Sentinel(key),
        BuildConcreteKey::MainTypeVar(key) => PyType::TypeVar(key),
        BuildConcreteKey::TempTypeVar(key) => PyType::TypeVar(remap_temp_key(key, &keys.type_vars)),
        BuildConcreteKey::MainParamSpec(key) => PyType::ParamSpec(key),
        BuildConcreteKey::TempParamSpec(key) => {
            PyType::ParamSpec(remap_temp_key(key, &keys.param_specs))
        }
        BuildConcreteKey::MainPlain(key) => PyType::Plain(key),
        BuildConcreteKey::TempPlain(key) => PyType::Plain(remap_temp_key(key, &keys.plains)),
        BuildConcreteKey::MainProtocol(key) => PyType::Protocol(key),
        BuildConcreteKey::TempProtocol(key) => {
            PyType::Protocol(remap_temp_key(key, &keys.protocols))
        }
        BuildConcreteKey::MainTypedDict(key) => PyType::TypedDict(key),
        BuildConcreteKey::TempTypedDict(key) => {
            PyType::TypedDict(remap_temp_key(key, &keys.typed_dicts))
        }
        BuildConcreteKey::MainUnion(key) => PyType::Union(key),
        BuildConcreteKey::TempUnion(key) => PyType::Union(remap_temp_key(key, &keys.unions)),
        BuildConcreteKey::MainCallable(key) => PyType::Callable(key),
        BuildConcreteKey::TempCallable(key) => {
            PyType::Callable(remap_temp_key(key, &keys.callables))
        }
        BuildConcreteKey::MainLazyRef(key) => PyType::LazyRef(key),
        BuildConcreteKey::TempLazyRef(key) => PyType::LazyRef(remap_temp_key(key, &keys.lazy_refs)),
    }
}

fn commit_concrete_temp<'types, 'temp>(
    arenas: &mut TypeArenas<'types>,
    temp: TempConcreteArenas<'types, 'temp>,
    root: BuildConcreteKey<'types, 'temp>,
) -> PyTypeConcreteKey<'types> {
    let keys = ConcreteCommitKeys {
        type_vars: future_keys(&arenas.concrete.type_vars, temp.type_vars.values().len()),
        param_specs: future_keys(
            &arenas.concrete.param_specs,
            temp.param_specs.values().len(),
        ),
        plains: future_keys(&arenas.concrete.plains, temp.plains.values().len()),
        protocols: future_keys(&arenas.concrete.protocols, temp.protocols.values().len()),
        typed_dicts: future_keys(
            &arenas.concrete.typed_dicts,
            temp.typed_dicts.values().len(),
        ),
        unions: future_keys(&arenas.concrete.unions, temp.unions.values().len()),
        callables: future_keys(&arenas.concrete.callables, temp.callables.values().len()),
        lazy_refs: future_keys(&arenas.concrete.lazy_refs, temp.lazy_refs.values().len()),
    };
    let root = commit_build_key(root, &keys);

    let TempConcreteArenas {
        type_vars,
        param_specs,
        plains,
        protocols,
        typed_dicts,
        unions,
        callables,
        lazy_refs,
    } = temp;

    for value in type_vars.into_values() {
        arenas
            .concrete
            .type_vars
            .push_committed(value.expect("concrete temp type var should be filled"));
    }
    for value in param_specs.into_values() {
        arenas
            .concrete
            .param_specs
            .push_committed(value.expect("concrete temp param spec should be filled"));
    }
    for value in plains.into_values() {
        let value = value.expect("concrete temp plain should be filled");
        arenas.concrete.plains.push_committed(Qualified {
            inner: super::PlainType {
                descriptor: value.inner.descriptor,
                args: value
                    .inner
                    .args
                    .into_iter()
                    .map(|child| commit_build_key(child, &keys))
                    .collect(),
            },
            qualifier: value.qualifier,
        });
    }
    for value in protocols.into_values() {
        let value = value.expect("concrete temp protocol should be filled");
        arenas.concrete.protocols.push_committed(Qualified {
            inner: super::ProtocolType {
                descriptor: value.inner.descriptor,
                methods: value
                    .inner
                    .methods
                    .into_iter()
                    .map(|(name, child)| (name, commit_build_key(child, &keys)))
                    .collect(),
                attributes: value
                    .inner
                    .attributes
                    .into_iter()
                    .map(|(name, child)| (name, commit_build_key(child, &keys)))
                    .collect(),
                properties: value
                    .inner
                    .properties
                    .into_iter()
                    .map(|(name, child)| (name, commit_build_key(child, &keys)))
                    .collect(),
                type_params: value
                    .inner
                    .type_params
                    .into_iter()
                    .map(|child| commit_build_key(child, &keys))
                    .collect(),
            },
            qualifier: value.qualifier,
        });
    }
    for value in typed_dicts.into_values() {
        let value = value.expect("concrete temp typed dict should be filled");
        arenas.concrete.typed_dicts.push_committed(Qualified {
            inner: super::TypedDictType {
                descriptor: value.inner.descriptor,
                attributes: value
                    .inner
                    .attributes
                    .into_iter()
                    .map(|(name, child)| (name, commit_build_key(child, &keys)))
                    .collect(),
                type_params: value
                    .inner
                    .type_params
                    .into_iter()
                    .map(|child| commit_build_key(child, &keys))
                    .collect(),
            },
            qualifier: value.qualifier,
        });
    }
    for value in unions.into_values() {
        let value = value.expect("concrete temp union should be filled");
        arenas.concrete.unions.push_committed(Qualified {
            inner: super::UnionType {
                variants: value
                    .inner
                    .variants
                    .into_iter()
                    .map(|child| commit_build_key(child, &keys))
                    .collect(),
            },
            qualifier: value.qualifier,
        });
    }
    for value in callables.into_values() {
        let value = value.expect("concrete temp callable should be filled");
        arenas.concrete.callables.push_committed(Qualified {
            inner: super::CallableType {
                params: value
                    .inner
                    .params
                    .into_iter()
                    .map(|(name, child)| (name, commit_build_key(child, &keys)))
                    .collect(),
                param_kinds: value.inner.param_kinds,
                param_has_default: value.inner.param_has_default,
                accepts_varargs: value.inner.accepts_varargs,
                accepts_varkw: value.inner.accepts_varkw,
                return_type: commit_build_key(value.inner.return_type, &keys),
                return_wrapper: value.inner.return_wrapper,
                type_params: value
                    .inner
                    .type_params
                    .into_iter()
                    .map(|child| commit_build_key(child, &keys))
                    .collect(),
                function_name: value.inner.function_name,
            },
            qualifier: value.qualifier,
        });
    }
    for value in lazy_refs.into_values() {
        let value = value.expect("concrete temp lazy ref should be filled");
        arenas.concrete.lazy_refs.push_committed(Qualified {
            inner: super::LazyRefType {
                target: commit_build_key(value.inner.target, &keys),
            },
            qualifier: value.qualifier,
        });
    }

    root
}

fn apply_bindings_inner<'types, 'temp>(
    source: PyTypeParametricKey<'types>,
    bindings: &Bindings<'types>,
    arenas: &mut TypeArenas<'types>,
    temp: &mut TempConcreteArenas<'types, 'temp>,
    memo: &mut HashMap<PyTypeParametricKey<'types>, BuildConcreteKey<'types, 'temp>>,
) -> BuildConcreteKey<'types, 'temp> {
    if let Some(&cached) = memo.get(&source) {
        return cached;
    }

    let result = match source {
        // Binding lookup uses Python TypeVar identity (PyTypeId), not arena
        // slot key. The same logical TypeVar may occupy different slots due to
        // different qualifier contexts (return type vs params after include).
        PyType::TypeVar(key) => {
            let tv = arenas.parametric.type_vars.get(key);
            let qualifier = tv.qualifier.clone();
            let descriptor = tv.inner.descriptor.clone();
            match bindings.type_vars.get(&descriptor.id) {
                Some(&bound) => {
                    if qualifier.is_unqualified() {
                        BuildConcreteKey::main(bound)
                    } else {
                        BuildConcreteKey::main(requalify_concrete(bound, &qualifier, arenas))
                    }
                }
                None => {
                    let opaque = Qualified {
                        inner: OpaqueTypeVar { descriptor },
                        qualifier,
                    };
                    BuildConcreteKey::TempTypeVar(temp.type_vars.insert(Some(opaque)))
                }
            }
        }
        PyType::ParamSpec(key) => {
            let ps = arenas.parametric.param_specs.get(key);
            let qualifier = ps.qualifier.clone();
            let descriptor = ps.inner.descriptor.clone();
            match bindings.param_specs.get(&descriptor.id) {
                Some(&bound) => {
                    if qualifier.is_unqualified() {
                        BuildConcreteKey::main(bound)
                    } else {
                        BuildConcreteKey::main(requalify_concrete(bound, &qualifier, arenas))
                    }
                }
                None => {
                    let opaque = Qualified {
                        inner: OpaqueParamSpec { descriptor },
                        qualifier,
                    };
                    BuildConcreteKey::TempParamSpec(temp.param_specs.insert(Some(opaque)))
                }
            }
        }
        PyType::Sentinel(key) => BuildConcreteKey::Sentinel(key),
        PyType::Plain(key) => {
            let placeholder = temp.plains.insert(None);
            let result = BuildConcreteKey::TempPlain(placeholder);
            memo.insert(source, result);
            let val = arenas.parametric.plains.get(key).clone();
            let output = Qualified {
                inner: BuildPlainType {
                    descriptor: val.inner.descriptor,
                    args: val
                        .inner
                        .args
                        .into_iter()
                        .map(|child| apply_bindings_inner(child, bindings, arenas, temp, memo))
                        .collect(),
                },
                qualifier: val.qualifier,
            };
            assert!(
                temp.plains.get_mut(placeholder).replace(output).is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::Protocol(key) => {
            let placeholder = temp.protocols.insert(None);
            let result = BuildConcreteKey::TempProtocol(placeholder);
            memo.insert(source, result);
            let val = arenas.parametric.protocols.get(key).clone();
            let output = Qualified {
                inner: BuildProtocolType {
                    descriptor: val.inner.descriptor,
                    methods: val
                        .inner
                        .methods
                        .into_iter()
                        .map(|(name, child)| {
                            (
                                name,
                                apply_bindings_inner(child, bindings, arenas, temp, memo),
                            )
                        })
                        .collect(),
                    attributes: val
                        .inner
                        .attributes
                        .into_iter()
                        .map(|(name, child)| {
                            (
                                name,
                                apply_bindings_inner(child, bindings, arenas, temp, memo),
                            )
                        })
                        .collect(),
                    properties: val
                        .inner
                        .properties
                        .into_iter()
                        .map(|(name, child)| {
                            (
                                name,
                                apply_bindings_inner(child, bindings, arenas, temp, memo),
                            )
                        })
                        .collect(),
                    type_params: val
                        .inner
                        .type_params
                        .into_iter()
                        .map(|child| apply_bindings_inner(child, bindings, arenas, temp, memo))
                        .collect(),
                },
                qualifier: val.qualifier,
            };
            assert!(
                temp.protocols
                    .get_mut(placeholder)
                    .replace(output)
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::TypedDict(key) => {
            let placeholder = temp.typed_dicts.insert(None);
            let result = BuildConcreteKey::TempTypedDict(placeholder);
            memo.insert(source, result);
            let val = arenas.parametric.typed_dicts.get(key).clone();
            let output = Qualified {
                inner: BuildTypedDictType {
                    descriptor: val.inner.descriptor,
                    attributes: val
                        .inner
                        .attributes
                        .into_iter()
                        .map(|(name, child)| {
                            (
                                name,
                                apply_bindings_inner(child, bindings, arenas, temp, memo),
                            )
                        })
                        .collect(),
                    type_params: val
                        .inner
                        .type_params
                        .into_iter()
                        .map(|child| apply_bindings_inner(child, bindings, arenas, temp, memo))
                        .collect(),
                },
                qualifier: val.qualifier,
            };
            assert!(
                temp.typed_dicts
                    .get_mut(placeholder)
                    .replace(output)
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::Union(key) => {
            let placeholder = temp.unions.insert(None);
            let result = BuildConcreteKey::TempUnion(placeholder);
            memo.insert(source, result);
            let val = arenas.parametric.unions.get(key).clone();
            let output = Qualified {
                inner: BuildUnionType {
                    variants: val
                        .inner
                        .variants
                        .into_iter()
                        .map(|child| apply_bindings_inner(child, bindings, arenas, temp, memo))
                        .collect(),
                },
                qualifier: val.qualifier,
            };
            assert!(
                temp.unions.get_mut(placeholder).replace(output).is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::Callable(key) => {
            let placeholder = temp.callables.insert(None);
            let result = BuildConcreteKey::TempCallable(placeholder);
            memo.insert(source, result);
            let val = arenas.parametric.callables.get(key).clone();
            let output = Qualified {
                inner: BuildCallableType {
                    params: val
                        .inner
                        .params
                        .into_iter()
                        .map(|(name, child)| {
                            (
                                name,
                                apply_bindings_inner(child, bindings, arenas, temp, memo),
                            )
                        })
                        .collect(),
                    param_kinds: val.inner.param_kinds,
                    param_has_default: val.inner.param_has_default,
                    accepts_varargs: val.inner.accepts_varargs,
                    accepts_varkw: val.inner.accepts_varkw,
                    return_type: apply_bindings_inner(
                        val.inner.return_type,
                        bindings,
                        arenas,
                        temp,
                        memo,
                    ),
                    return_wrapper: val.inner.return_wrapper,
                    type_params: val
                        .inner
                        .type_params
                        .into_iter()
                        .map(|child| apply_bindings_inner(child, bindings, arenas, temp, memo))
                        .collect(),
                    function_name: val.inner.function_name,
                },
                qualifier: val.qualifier,
            };
            assert!(
                temp.callables
                    .get_mut(placeholder)
                    .replace(output)
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::LazyRef(key) => {
            let placeholder = temp.lazy_refs.insert(None);
            let result = BuildConcreteKey::TempLazyRef(placeholder);
            memo.insert(source, result);
            let val = arenas.parametric.lazy_refs.get(key).clone();
            let output = Qualified {
                inner: BuildLazyRefType {
                    target: apply_bindings_inner(val.inner.target, bindings, arenas, temp, memo),
                },
                qualifier: val.qualifier,
            };
            assert!(
                temp.lazy_refs
                    .get_mut(placeholder)
                    .replace(output)
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
    };

    memo.insert(source, result);
    result
}

fn requalified_qualifier(current: &Qualifier, additional: &Qualifier) -> Qualifier {
    current.intersect(additional)
}

enum RequalifiedKey<'types, 'temp, T> {
    Main(ArenaKey<'types, Qualified<T>>),
    Temp(ArenaKey<'temp, Qualified<T>>),
}

fn reinsert_requalified_temp<'types, 'temp, T>(
    main: &Arena<'types, Qualified<T>>,
    temp: &mut Arena<'temp, Qualified<T>, Option<Qualified<T>>>,
    key: ArenaKey<'types, Qualified<T>>,
    additional: &Qualifier,
) -> RequalifiedKey<'types, 'temp, T>
where
    T: Clone,
{
    let val = main.get(key);
    let new_qual = requalified_qualifier(&val.qualifier, additional);
    if new_qual == val.qualifier {
        return RequalifiedKey::Main(key);
    }
    RequalifiedKey::Temp(temp.insert(Some(Qualified {
        inner: val.inner.clone(),
        qualifier: new_qual,
    })))
}

fn requalify_concrete_inner<'types, 'temp>(
    target: PyTypeConcreteKey<'types>,
    additional: &Qualifier,
    arenas: &mut TypeArenas<'types>,
    temp: &mut TempConcreteArenas<'types, 'temp>,
    memo: &mut HashMap<PyTypeConcreteKey<'types>, BuildConcreteKey<'types, 'temp>>,
) -> BuildConcreteKey<'types, 'temp> {
    if let Some(&cached) = memo.get(&target) {
        return cached;
    }

    let result = match target {
        PyType::Sentinel(key) => BuildConcreteKey::Sentinel(key),
        PyType::TypeVar(key) => match reinsert_requalified_temp(
            &arenas.concrete.type_vars,
            &mut temp.type_vars,
            key,
            additional,
        ) {
            RequalifiedKey::Main(key) => BuildConcreteKey::MainTypeVar(key),
            RequalifiedKey::Temp(key) => BuildConcreteKey::TempTypeVar(key),
        },
        PyType::ParamSpec(key) => match reinsert_requalified_temp(
            &arenas.concrete.param_specs,
            &mut temp.param_specs,
            key,
            additional,
        ) {
            RequalifiedKey::Main(key) => BuildConcreteKey::MainParamSpec(key),
            RequalifiedKey::Temp(key) => BuildConcreteKey::TempParamSpec(key),
        },
        PyType::Plain(key) => {
            let placeholder = temp.plains.insert(None);
            let result = BuildConcreteKey::TempPlain(placeholder);
            memo.insert(target, result);
            let value = arenas.concrete.plains.get(key).clone();
            let output = Qualified {
                inner: BuildPlainType {
                    descriptor: value.inner.descriptor,
                    args: value
                        .inner
                        .args
                        .into_iter()
                        .map(|child| {
                            requalify_concrete_inner(child, additional, arenas, temp, memo)
                        })
                        .collect(),
                },
                qualifier: requalified_qualifier(&value.qualifier, additional),
            };
            assert!(
                temp.plains.get_mut(placeholder).replace(output).is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::Protocol(key) => {
            let placeholder = temp.protocols.insert(None);
            let result = BuildConcreteKey::TempProtocol(placeholder);
            memo.insert(target, result);
            let value = arenas.concrete.protocols.get(key).clone();
            let output = Qualified {
                inner: BuildProtocolType {
                    descriptor: value.inner.descriptor,
                    methods: value
                        .inner
                        .methods
                        .into_iter()
                        .map(|(name, child)| {
                            (
                                name,
                                requalify_concrete_inner(child, additional, arenas, temp, memo),
                            )
                        })
                        .collect(),
                    attributes: value
                        .inner
                        .attributes
                        .into_iter()
                        .map(|(name, child)| {
                            (
                                name,
                                requalify_concrete_inner(child, additional, arenas, temp, memo),
                            )
                        })
                        .collect(),
                    properties: value
                        .inner
                        .properties
                        .into_iter()
                        .map(|(name, child)| {
                            (
                                name,
                                requalify_concrete_inner(child, additional, arenas, temp, memo),
                            )
                        })
                        .collect(),
                    type_params: value
                        .inner
                        .type_params
                        .into_iter()
                        .map(|child| {
                            requalify_concrete_inner(child, additional, arenas, temp, memo)
                        })
                        .collect(),
                },
                qualifier: requalified_qualifier(&value.qualifier, additional),
            };
            assert!(
                temp.protocols
                    .get_mut(placeholder)
                    .replace(output)
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::TypedDict(key) => {
            let placeholder = temp.typed_dicts.insert(None);
            let result = BuildConcreteKey::TempTypedDict(placeholder);
            memo.insert(target, result);
            let value = arenas.concrete.typed_dicts.get(key).clone();
            let output = Qualified {
                inner: BuildTypedDictType {
                    descriptor: value.inner.descriptor,
                    attributes: value
                        .inner
                        .attributes
                        .into_iter()
                        .map(|(name, child)| {
                            (
                                name,
                                requalify_concrete_inner(child, additional, arenas, temp, memo),
                            )
                        })
                        .collect(),
                    type_params: value
                        .inner
                        .type_params
                        .into_iter()
                        .map(|child| {
                            requalify_concrete_inner(child, additional, arenas, temp, memo)
                        })
                        .collect(),
                },
                qualifier: requalified_qualifier(&value.qualifier, additional),
            };
            assert!(
                temp.typed_dicts
                    .get_mut(placeholder)
                    .replace(output)
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::Union(key) => {
            let placeholder = temp.unions.insert(None);
            let result = BuildConcreteKey::TempUnion(placeholder);
            memo.insert(target, result);
            let value = arenas.concrete.unions.get(key).clone();
            let output = Qualified {
                inner: BuildUnionType {
                    variants: value
                        .inner
                        .variants
                        .into_iter()
                        .map(|child| {
                            requalify_concrete_inner(child, additional, arenas, temp, memo)
                        })
                        .collect(),
                },
                qualifier: requalified_qualifier(&value.qualifier, additional),
            };
            assert!(
                temp.unions.get_mut(placeholder).replace(output).is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::Callable(key) => {
            let placeholder = temp.callables.insert(None);
            let result = BuildConcreteKey::TempCallable(placeholder);
            memo.insert(target, result);
            let value = arenas.concrete.callables.get(key).clone();
            let output = Qualified {
                inner: BuildCallableType {
                    params: value
                        .inner
                        .params
                        .into_iter()
                        .map(|(name, child)| {
                            (
                                name,
                                requalify_concrete_inner(child, additional, arenas, temp, memo),
                            )
                        })
                        .collect(),
                    param_kinds: value.inner.param_kinds,
                    param_has_default: value.inner.param_has_default,
                    accepts_varargs: value.inner.accepts_varargs,
                    accepts_varkw: value.inner.accepts_varkw,
                    return_type: requalify_concrete_inner(
                        value.inner.return_type,
                        additional,
                        arenas,
                        temp,
                        memo,
                    ),
                    return_wrapper: value.inner.return_wrapper,
                    type_params: value
                        .inner
                        .type_params
                        .into_iter()
                        .map(|child| {
                            requalify_concrete_inner(child, additional, arenas, temp, memo)
                        })
                        .collect(),
                    function_name: value.inner.function_name,
                },
                qualifier: requalified_qualifier(&value.qualifier, additional),
            };
            assert!(
                temp.callables
                    .get_mut(placeholder)
                    .replace(output)
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::LazyRef(key) => {
            let placeholder = temp.lazy_refs.insert(None);
            let result = BuildConcreteKey::TempLazyRef(placeholder);
            memo.insert(target, result);
            let value = arenas.concrete.lazy_refs.get(key).clone();
            let output = Qualified {
                inner: BuildLazyRefType {
                    target: requalify_concrete_inner(
                        value.inner.target,
                        additional,
                        arenas,
                        temp,
                        memo,
                    ),
                },
                qualifier: requalified_qualifier(&value.qualifier, additional),
            };
            assert!(
                temp.lazy_refs
                    .get_mut(placeholder)
                    .replace(output)
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
    };

    memo.insert(target, result);
    result
}

pub(crate) fn requalify_concrete<'types>(
    target: PyTypeConcreteKey<'types>,
    additional: &Qualifier,
    arenas: &mut TypeArenas<'types>,
) -> PyTypeConcreteKey<'types> {
    let mut temp = TempConcreteArenas::default();
    let root = requalify_concrete_inner(
        target,
        additional,
        arenas,
        &mut temp,
        &mut HashMap::default(),
    );
    let root = commit_concrete_temp(arenas, temp, root);
    canonicalize_if_resolved(root, arenas)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use super::*;
    use crate::types::{PlainType, PyTypeDescriptor, PyTypeId};

    fn duplicate_plain_key<'types>(
        arenas: &mut TypeArenas<'types>,
        descriptor: &PyTypeDescriptor,
        args: Vec<PyTypeConcreteKey<'types>>,
    ) -> PyTypeConcreteKey<'types> {
        let key = arenas.concrete.plains.insert(Qualified {
            inner: PlainType {
                descriptor: descriptor.clone(),
                args,
            },
            qualifier: Qualifier::any(),
        });
        PyType::Plain(key)
    }

    fn qualifier(tag: &str) -> Qualifier {
        let mut alternative = BTreeSet::new();
        alternative.insert(tag.to_string());
        let mut alternatives = BTreeSet::new();
        alternatives.insert(alternative);
        Qualifier::from(alternatives)
    }

    #[test]
    fn apply_bindings_reuses_structurally_equal_concrete_keys() {
        let mut arenas = TypeArenas::default();
        let source = PyType::Plain(arenas.parametric.plains.insert(Qualified {
            inner: PlainType {
                descriptor: PyTypeDescriptor {
                    id: PyTypeId::new("builtins.int".to_string()),
                    display_name: Arc::from("int"),
                },
                args: vec![],
            },
            qualifier: Qualifier::any(),
        }));

        let first = arenas.apply_bindings(source, &Bindings::default());
        let second = arenas.apply_bindings(source, &Bindings::default());

        assert!(first == second);
    }

    #[test]
    fn requalify_concrete_reuses_structurally_equal_nested_keys() {
        let mut arenas = TypeArenas::default();
        let child_descriptor = PyTypeDescriptor {
            id: PyTypeId::new("bench.ChildState".to_string()),
            display_name: Arc::from("ChildState"),
        };
        let parent_descriptor = PyTypeDescriptor {
            id: PyTypeId::new("bench.WriteState".to_string()),
            display_name: Arc::from("WriteState"),
        };

        let child_a = duplicate_plain_key(&mut arenas, &child_descriptor, vec![]);
        let child_b = duplicate_plain_key(&mut arenas, &child_descriptor, vec![]);
        let parent_a = duplicate_plain_key(&mut arenas, &parent_descriptor, vec![child_a]);
        let parent_b = duplicate_plain_key(&mut arenas, &parent_descriptor, vec![child_b]);

        let first = requalify_concrete(parent_a, &qualifier("write"), &mut arenas);
        let second = requalify_concrete(parent_b, &qualifier("write"), &mut arenas);

        assert!(first == second);
    }

    #[test]
    fn requalify_concrete_requalifies_nested_children() {
        let mut arenas = TypeArenas::default();
        let child_descriptor = PyTypeDescriptor {
            id: PyTypeId::new("bench.VectorClock".to_string()),
            display_name: Arc::from("VectorClock"),
        };
        let parent_descriptor = PyTypeDescriptor {
            id: PyTypeId::new("bench.Constants".to_string()),
            display_name: Arc::from("Constants"),
        };

        let child = duplicate_plain_key(&mut arenas, &child_descriptor, vec![]);
        let parent = duplicate_plain_key(&mut arenas, &parent_descriptor, vec![child]);

        let requalified = requalify_concrete(parent, &qualifier("write"), &mut arenas);
        let PyType::Plain(parent_key) = requalified else {
            panic!("expected plain requalified parent");
        };
        let parent_value = arenas.concrete.plains.get(parent_key);
        let child_key = parent_value.inner.args[0];

        assert_eq!(
            arenas.qualifier_of_concrete(child_key).display_compact(),
            "ANY"
        );
    }
}
