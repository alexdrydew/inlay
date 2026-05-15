use std::sync::Arc;

use indexmap::IndexMap;
use inlay_instrument::{inlay_event, instrumented};
use rustc_hash::FxHashMap as HashMap;

use crate::qualifier::Qualifier;

use super::{
    ApplyBindingsCacheKey, Arena, ArenaKey, Bindings, CallableType, ClassInit, ClassType, Concrete,
    Keyed, LazyRefType, OpaqueParamSpec, OpaqueTypeVar, ParamKind, Parametric, PlainType,
    ProtocolBase, ProtocolMethod, ProtocolType, PyType, PyTypeConcreteKey, PyTypeDescriptor,
    PyTypeParametricKey, Qual, Qualified, RequalifyConcreteCacheKey, TypeArenas, TypedDictType,
    UnionType, WrapperKind,
};

// --- TypeArenas method ---

impl<'ty> TypeArenas<'ty> {
    #[instrumented(
        name = "inlay.types.apply_bindings",
        target = "inlay",
        level = "trace",
        skip_all,
        fields(
            type_bindings = bindings.type_vars.len() as u64,
            param_bindings = bindings.param_specs.len() as u64,
            cache_hit
        )
    )]
    pub(crate) fn apply_bindings(
        &mut self,
        source: PyTypeParametricKey<'ty>,
        bindings: &Bindings<'ty>,
    ) -> PyTypeConcreteKey<'ty> {
        let cache_key = apply_bindings_cache_key(source, bindings);
        if let Some(cached) = self.apply_bindings_cache.get(&cache_key).copied() {
            inlay_event!(
                name: "inlay.types.apply_bindings.result",
                cache_hit = true,
            );
            return cached;
        }

        let mut temp = TempConcreteArenas::default();
        let root = apply_bindings_inner(source, bindings, self, &mut temp, &mut HashMap::default());
        let root = commit_concrete_temp(self, temp, root);
        let root = self.canonicalize_concrete(root);
        self.apply_bindings_cache.insert(cache_key, root);
        inlay_event!(
            name: "inlay.types.apply_bindings.result",
            cache_hit = false,
        );
        root
    }

    fn canonicalize_concrete(&mut self, key: PyTypeConcreteKey<'ty>) -> PyTypeConcreteKey<'ty> {
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

fn apply_bindings_cache_key<'ty>(
    source: PyTypeParametricKey<'ty>,
    bindings: &Bindings<'ty>,
) -> ApplyBindingsCacheKey<'ty> {
    let mut type_vars = bindings
        .type_vars
        .iter()
        .map(|(id, &type_ref)| (id.clone(), type_ref))
        .collect::<Vec<_>>();
    type_vars.sort_by(|left, right| left.0.cmp(&right.0));

    let mut param_specs = bindings
        .param_specs
        .iter()
        .map(|(id, &type_ref)| (id.clone(), type_ref))
        .collect::<Vec<_>>();
    param_specs.sort_by(|left, right| left.0.cmp(&right.0));

    ApplyBindingsCacheKey {
        source,
        type_vars,
        param_specs,
    }
}

fn canonicalize_if_resolved<'ty>(
    key: PyTypeConcreteKey<'ty>,
    arenas: &mut TypeArenas<'ty>,
) -> PyTypeConcreteKey<'ty> {
    arenas.canonicalize_concrete(key)
}

type ConcretePlain<'ty> = Qualified<PlainType<Qual<Keyed<'ty>>, Concrete>>;
type ConcreteClass<'ty> = Qualified<ClassType<Qual<Keyed<'ty>>, Concrete>>;
type ConcreteProtocol<'ty> = Qualified<ProtocolType<Qual<Keyed<'ty>>, Concrete>>;
type ConcreteProtocolBase<'ty> = ProtocolBase<Qual<Keyed<'ty>>, Concrete>;
type ConcreteTypedDict<'ty> = Qualified<TypedDictType<Qual<Keyed<'ty>>, Concrete>>;
type ConcreteUnion<'ty> = Qualified<UnionType<Qual<Keyed<'ty>>, Concrete>>;
type ConcreteCallable<'ty> = Qualified<CallableType<Qual<Keyed<'ty>>, Concrete>>;
type ConcreteLazyRef<'ty> = Qualified<LazyRefType<Qual<Keyed<'ty>>, Concrete>>;
type ConcreteProtocolMethod<'ty> = ProtocolMethod<Qual<Keyed<'ty>>, Concrete>;
type ConcreteProtocolMethodList<'ty> = Arc<[(Arc<str>, ConcreteProtocolMethod<'ty>)]>;
type ParametricProtocolMethod<'ty> = ProtocolMethod<Qual<Keyed<'ty>>, Parametric>;
type ParametricProtocolBase<'ty> = ProtocolBase<Qual<Keyed<'ty>>, Parametric>;

#[derive(Clone)]
struct BuildPlainType<'ty, 'tmp> {
    descriptor: PyTypeDescriptor,
    args: Vec<BuildConcreteKey<'ty, 'tmp>>,
}

#[derive(Clone)]
struct BuildClassInit<'ty, 'tmp> {
    params: IndexMap<Arc<str>, BuildConcreteKey<'ty, 'tmp>>,
    param_kinds: Vec<ParamKind>,
    param_has_default: Vec<bool>,
}

#[derive(Clone)]
struct BuildClassType<'ty, 'tmp> {
    descriptor: PyTypeDescriptor,
    constructor: Arc<pyo3::Py<pyo3::PyAny>>,
    args: Vec<BuildConcreteKey<'ty, 'tmp>>,
    init: Option<BuildClassInit<'ty, 'tmp>>,
}

#[derive(Clone)]
struct BuildProtocolType<'ty, 'tmp> {
    descriptor: PyTypeDescriptor,
    protocol_mro: Vec<BuildProtocolBase<'ty, 'tmp>>,
    direct_methods: Vec<Arc<str>>,
    methods: Arc<[(Arc<str>, BuildProtocolMethod<'ty, 'tmp>)]>,
    attributes: Arc<[(Arc<str>, BuildConcreteKey<'ty, 'tmp>)]>,
    properties: Arc<[(Arc<str>, BuildConcreteKey<'ty, 'tmp>)]>,
    type_params: Vec<BuildConcreteKey<'ty, 'tmp>>,
}

#[derive(Clone)]
struct BuildProtocolBase<'ty, 'tmp> {
    descriptor: PyTypeDescriptor,
    type_params: Vec<BuildConcreteKey<'ty, 'tmp>>,
    direct_methods: Vec<Arc<str>>,
}

#[derive(Clone)]
struct BuildProtocolMethod<'ty, 'tmp> {
    callable: BuildConcreteKey<'ty, 'tmp>,
}

#[derive(Clone)]
struct BuildTypedDictType<'ty, 'tmp> {
    descriptor: PyTypeDescriptor,
    attributes: Arc<[(Arc<str>, BuildConcreteKey<'ty, 'tmp>)]>,
    type_params: Vec<BuildConcreteKey<'ty, 'tmp>>,
}

#[derive(Clone)]
struct BuildUnionType<'ty, 'tmp> {
    variants: Vec<BuildConcreteKey<'ty, 'tmp>>,
}

#[derive(Clone)]
struct BuildCallableType<'ty, 'tmp> {
    params: IndexMap<Arc<str>, BuildConcreteKey<'ty, 'tmp>>,
    param_kinds: Vec<ParamKind>,
    param_has_default: Vec<bool>,
    accepts_varargs: bool,
    accepts_varkw: bool,
    return_type: BuildConcreteKey<'ty, 'tmp>,
    return_wrapper: WrapperKind,
    type_params: Vec<BuildConcreteKey<'ty, 'tmp>>,
    function_name: Option<Arc<str>>,
}

#[derive(Clone)]
struct BuildLazyRefType<'ty, 'tmp> {
    target: BuildConcreteKey<'ty, 'tmp>,
}

type TempConcretePlain<'ty, 'tmp> = Qualified<BuildPlainType<'ty, 'tmp>>;
type TempConcreteClass<'ty, 'tmp> = Qualified<BuildClassType<'ty, 'tmp>>;
type TempConcreteProtocol<'ty, 'tmp> = Qualified<BuildProtocolType<'ty, 'tmp>>;
type TempConcreteTypedDict<'ty, 'tmp> = Qualified<BuildTypedDictType<'ty, 'tmp>>;
type TempConcreteUnion<'ty, 'tmp> = Qualified<BuildUnionType<'ty, 'tmp>>;
type TempConcreteCallable<'ty, 'tmp> = Qualified<BuildCallableType<'ty, 'tmp>>;
type TempConcreteLazyRef<'ty, 'tmp> = Qualified<BuildLazyRefType<'ty, 'tmp>>;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum BuildConcreteKey<'ty, 'tmp> {
    Sentinel(ArenaKey<'ty, Qualified<super::SentinelType>>),
    MainTypeVar(ArenaKey<'ty, Qualified<OpaqueTypeVar>>),
    TempTypeVar(ArenaKey<'tmp, Qualified<OpaqueTypeVar>>),
    MainParamSpec(ArenaKey<'ty, Qualified<OpaqueParamSpec>>),
    TempParamSpec(ArenaKey<'tmp, Qualified<OpaqueParamSpec>>),
    MainPlain(ArenaKey<'ty, ConcretePlain<'ty>>),
    TempPlain(ArenaKey<'tmp, TempConcretePlain<'ty, 'tmp>>),
    MainClass(ArenaKey<'ty, ConcreteClass<'ty>>),
    TempClass(ArenaKey<'tmp, TempConcreteClass<'ty, 'tmp>>),
    MainProtocol(ArenaKey<'ty, ConcreteProtocol<'ty>>),
    TempProtocol(ArenaKey<'tmp, TempConcreteProtocol<'ty, 'tmp>>),
    MainTypedDict(ArenaKey<'ty, ConcreteTypedDict<'ty>>),
    TempTypedDict(ArenaKey<'tmp, TempConcreteTypedDict<'ty, 'tmp>>),
    MainUnion(ArenaKey<'ty, ConcreteUnion<'ty>>),
    TempUnion(ArenaKey<'tmp, TempConcreteUnion<'ty, 'tmp>>),
    MainCallable(ArenaKey<'ty, ConcreteCallable<'ty>>),
    TempCallable(ArenaKey<'tmp, TempConcreteCallable<'ty, 'tmp>>),
    MainLazyRef(ArenaKey<'ty, ConcreteLazyRef<'ty>>),
    TempLazyRef(ArenaKey<'tmp, TempConcreteLazyRef<'ty, 'tmp>>),
}

impl<'ty> BuildConcreteKey<'ty, '_> {
    fn main(key: PyTypeConcreteKey<'ty>) -> Self {
        match key {
            PyType::Sentinel(key) => Self::Sentinel(key),
            PyType::TypeVar(key) => Self::MainTypeVar(key),
            PyType::ParamSpec(key) => Self::MainParamSpec(key),
            PyType::Plain(key) => Self::MainPlain(key),
            PyType::Class(key) => Self::MainClass(key),
            PyType::Protocol(key) => Self::MainProtocol(key),
            PyType::TypedDict(key) => Self::MainTypedDict(key),
            PyType::Union(key) => Self::MainUnion(key),
            PyType::Callable(key) => Self::MainCallable(key),
            PyType::LazyRef(key) => Self::MainLazyRef(key),
        }
    }
}

#[derive(Default)]
struct TempConcreteArenas<'ty, 'tmp> {
    type_vars: Arena<'tmp, Qualified<OpaqueTypeVar>, Option<Qualified<OpaqueTypeVar>>>,
    param_specs: Arena<'tmp, Qualified<OpaqueParamSpec>, Option<Qualified<OpaqueParamSpec>>>,
    plains: Arena<'tmp, TempConcretePlain<'ty, 'tmp>, Option<TempConcretePlain<'ty, 'tmp>>>,
    classes: Arena<'tmp, TempConcreteClass<'ty, 'tmp>, Option<TempConcreteClass<'ty, 'tmp>>>,
    protocols:
        Arena<'tmp, TempConcreteProtocol<'ty, 'tmp>, Option<TempConcreteProtocol<'ty, 'tmp>>>,
    typed_dicts:
        Arena<'tmp, TempConcreteTypedDict<'ty, 'tmp>, Option<TempConcreteTypedDict<'ty, 'tmp>>>,
    unions: Arena<'tmp, TempConcreteUnion<'ty, 'tmp>, Option<TempConcreteUnion<'ty, 'tmp>>>,
    callables:
        Arena<'tmp, TempConcreteCallable<'ty, 'tmp>, Option<TempConcreteCallable<'ty, 'tmp>>>,
    lazy_refs: Arena<'tmp, TempConcreteLazyRef<'ty, 'tmp>, Option<TempConcreteLazyRef<'ty, 'tmp>>>,
}

struct ConcreteCommitKeys<'ty> {
    type_vars: Vec<ArenaKey<'ty, Qualified<OpaqueTypeVar>>>,
    param_specs: Vec<ArenaKey<'ty, Qualified<OpaqueParamSpec>>>,
    plains: Vec<ArenaKey<'ty, ConcretePlain<'ty>>>,
    classes: Vec<ArenaKey<'ty, ConcreteClass<'ty>>>,
    protocols: Vec<ArenaKey<'ty, ConcreteProtocol<'ty>>>,
    typed_dicts: Vec<ArenaKey<'ty, ConcreteTypedDict<'ty>>>,
    unions: Vec<ArenaKey<'ty, ConcreteUnion<'ty>>>,
    callables: Vec<ArenaKey<'ty, ConcreteCallable<'ty>>>,
    lazy_refs: Vec<ArenaKey<'ty, ConcreteLazyRef<'ty>>>,
}

fn future_keys<'ty, T>(store: &Arena<'ty, T>, count: usize) -> Vec<ArenaKey<'ty, T>> {
    (0..count).map(|offset| store.future_key(offset)).collect()
}

fn remap_temp_key<'ty, T, U>(key: ArenaKey<'_, T>, keys: &[ArenaKey<'ty, U>]) -> ArenaKey<'ty, U> {
    keys[key.index()]
}

fn commit_build_key<'ty>(
    key: BuildConcreteKey<'ty, '_>,
    keys: &ConcreteCommitKeys<'ty>,
) -> PyTypeConcreteKey<'ty> {
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
        BuildConcreteKey::MainClass(key) => PyType::Class(key),
        BuildConcreteKey::TempClass(key) => PyType::Class(remap_temp_key(key, &keys.classes)),
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

fn map_member_list<T: Copy, U>(
    members: &[(Arc<str>, T)],
    mut map_value: impl FnMut(T) -> U,
) -> Arc<[(Arc<str>, U)]> {
    let values: Vec<_> = members
        .iter()
        .map(|(name, value)| (Arc::clone(name), map_value(*value)))
        .collect();
    Arc::from(values.into_boxed_slice())
}

fn map_protocol_method_list<'ty, 'tmp>(
    methods: &[(Arc<str>, ConcreteProtocolMethod<'ty>)],
    mut map_value: impl FnMut(PyTypeConcreteKey<'ty>) -> BuildConcreteKey<'ty, 'tmp>,
) -> Arc<[(Arc<str>, BuildProtocolMethod<'ty, 'tmp>)]> {
    let values: Vec<_> = methods
        .iter()
        .map(|(name, method)| {
            (
                Arc::clone(name),
                BuildProtocolMethod {
                    callable: map_value(method.callable),
                },
            )
        })
        .collect();
    Arc::from(values.into_boxed_slice())
}

fn map_parametric_protocol_method_list<'ty, 'tmp>(
    methods: &[(Arc<str>, ParametricProtocolMethod<'ty>)],
    mut map_value: impl FnMut(PyTypeParametricKey<'ty>) -> BuildConcreteKey<'ty, 'tmp>,
) -> Arc<[(Arc<str>, BuildProtocolMethod<'ty, 'tmp>)]> {
    let values: Vec<_> = methods
        .iter()
        .map(|(name, method)| {
            (
                Arc::clone(name),
                BuildProtocolMethod {
                    callable: map_value(method.callable),
                },
            )
        })
        .collect();
    Arc::from(values.into_boxed_slice())
}

fn map_concrete_protocol_base<'ty, 'tmp>(
    scope: &ConcreteProtocolBase<'ty>,
    mut map_value: impl FnMut(PyTypeConcreteKey<'ty>) -> BuildConcreteKey<'ty, 'tmp>,
) -> BuildProtocolBase<'ty, 'tmp> {
    BuildProtocolBase {
        descriptor: scope.descriptor.clone(),
        type_params: scope
            .type_params
            .iter()
            .map(|&child| map_value(child))
            .collect(),
        direct_methods: scope.direct_methods.clone(),
    }
}

fn map_parametric_protocol_base<'ty, 'tmp>(
    scope: &ParametricProtocolBase<'ty>,
    mut map_value: impl FnMut(PyTypeParametricKey<'ty>) -> BuildConcreteKey<'ty, 'tmp>,
) -> BuildProtocolBase<'ty, 'tmp> {
    BuildProtocolBase {
        descriptor: scope.descriptor.clone(),
        type_params: scope
            .type_params
            .iter()
            .map(|&child| map_value(child))
            .collect(),
        direct_methods: scope.direct_methods.clone(),
    }
}

fn commit_protocol_base<'ty>(
    scope: &BuildProtocolBase<'ty, '_>,
    keys: &ConcreteCommitKeys<'ty>,
) -> ConcreteProtocolBase<'ty> {
    ProtocolBase {
        descriptor: scope.descriptor.clone(),
        type_params: scope
            .type_params
            .iter()
            .map(|&child| commit_build_key(child, keys))
            .collect(),
        direct_methods: scope.direct_methods.clone(),
    }
}

fn commit_protocol_method_list<'ty>(
    methods: &[(Arc<str>, BuildProtocolMethod<'ty, '_>)],
    keys: &ConcreteCommitKeys<'ty>,
) -> ConcreteProtocolMethodList<'ty> {
    let values: Vec<_> = methods
        .iter()
        .map(|(name, method)| {
            (
                Arc::clone(name),
                ProtocolMethod {
                    callable: commit_build_key(method.callable, keys),
                },
            )
        })
        .collect();
    Arc::from(values.into_boxed_slice())
}

fn commit_concrete_temp<'ty, 'tmp>(
    arenas: &mut TypeArenas<'ty>,
    temp: TempConcreteArenas<'ty, 'tmp>,
    root: BuildConcreteKey<'ty, 'tmp>,
) -> PyTypeConcreteKey<'ty> {
    let keys = ConcreteCommitKeys {
        type_vars: future_keys(&arenas.concrete.type_vars, temp.type_vars.values().len()),
        param_specs: future_keys(
            &arenas.concrete.param_specs,
            temp.param_specs.values().len(),
        ),
        plains: future_keys(&arenas.concrete.plains, temp.plains.values().len()),
        classes: future_keys(&arenas.concrete.classes, temp.classes.values().len()),
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
        classes,
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
    for value in classes.into_values() {
        let value = value.expect("concrete temp class should be filled");
        arenas.concrete.classes.push_committed(Qualified {
            inner: super::ClassType {
                descriptor: value.inner.descriptor,
                constructor: value.inner.constructor,
                args: value
                    .inner
                    .args
                    .into_iter()
                    .map(|child| commit_build_key(child, &keys))
                    .collect(),
                init: value.inner.init.map(|init| ClassInit {
                    params: init
                        .params
                        .into_iter()
                        .map(|(name, child)| (name, commit_build_key(child, &keys)))
                        .collect(),
                    param_kinds: init.param_kinds,
                    param_has_default: init.param_has_default,
                }),
            },
            qualifier: value.qualifier,
        });
    }
    for value in protocols.into_values() {
        let value = value.expect("concrete temp protocol should be filled");
        arenas.concrete.protocols.push_committed(Qualified {
            inner: super::ProtocolType {
                descriptor: value.inner.descriptor,
                protocol_mro: value
                    .inner
                    .protocol_mro
                    .iter()
                    .map(|scope| commit_protocol_base(scope, &keys))
                    .collect(),
                direct_methods: value.inner.direct_methods,
                methods: commit_protocol_method_list(&value.inner.methods, &keys),
                attributes: map_member_list(&value.inner.attributes, |child| {
                    commit_build_key(child, &keys)
                }),
                properties: map_member_list(&value.inner.properties, |child| {
                    commit_build_key(child, &keys)
                }),
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
                attributes: map_member_list(&value.inner.attributes, |child| {
                    commit_build_key(child, &keys)
                }),
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

fn apply_bindings_inner<'ty, 'tmp>(
    source: PyTypeParametricKey<'ty>,
    bindings: &Bindings<'ty>,
    arenas: &mut TypeArenas<'ty>,
    temp: &mut TempConcreteArenas<'ty, 'tmp>,
    memo: &mut HashMap<PyTypeParametricKey<'ty>, BuildConcreteKey<'ty, 'tmp>>,
) -> BuildConcreteKey<'ty, 'tmp> {
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
        PyType::Class(key) => {
            let placeholder = temp.classes.insert(None);
            let result = BuildConcreteKey::TempClass(placeholder);
            memo.insert(source, result);
            let val = arenas.parametric.classes.get(key).clone();
            let output = Qualified {
                inner: BuildClassType {
                    descriptor: val.inner.descriptor,
                    constructor: val.inner.constructor,
                    args: val
                        .inner
                        .args
                        .into_iter()
                        .map(|child| apply_bindings_inner(child, bindings, arenas, temp, memo))
                        .collect(),
                    init: val.inner.init.map(|init| BuildClassInit {
                        params: init
                            .params
                            .into_iter()
                            .map(|(name, child)| {
                                (
                                    name,
                                    apply_bindings_inner(child, bindings, arenas, temp, memo),
                                )
                            })
                            .collect(),
                        param_kinds: init.param_kinds,
                        param_has_default: init.param_has_default,
                    }),
                },
                qualifier: val.qualifier,
            };
            assert!(
                temp.classes.get_mut(placeholder).replace(output).is_none(),
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
                    protocol_mro: val
                        .inner
                        .protocol_mro
                        .iter()
                        .map(|scope| {
                            map_parametric_protocol_base(scope, |child| {
                                apply_bindings_inner(child, bindings, arenas, temp, memo)
                            })
                        })
                        .collect(),
                    direct_methods: val.inner.direct_methods,
                    methods: map_parametric_protocol_method_list(&val.inner.methods, |child| {
                        apply_bindings_inner(child, bindings, arenas, temp, memo)
                    }),
                    attributes: map_member_list(&val.inner.attributes, |child| {
                        apply_bindings_inner(child, bindings, arenas, temp, memo)
                    }),
                    properties: map_member_list(&val.inner.properties, |child| {
                        apply_bindings_inner(child, bindings, arenas, temp, memo)
                    }),
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
                    attributes: map_member_list(&val.inner.attributes, |child| {
                        apply_bindings_inner(child, bindings, arenas, temp, memo)
                    }),
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
    if current.is_unqualified() {
        return additional.clone();
    }
    if additional.is_unqualified() {
        return current.clone();
    }
    current.intersect(additional)
}

enum RequalifiedKey<'ty, 'tmp, T> {
    Main(ArenaKey<'ty, Qualified<T>>),
    Temp(ArenaKey<'tmp, Qualified<T>>),
}

fn reinsert_requalified_temp<'ty, 'tmp, T>(
    main: &Arena<'ty, Qualified<T>>,
    temp: &mut Arena<'tmp, Qualified<T>, Option<Qualified<T>>>,
    key: ArenaKey<'ty, Qualified<T>>,
    additional: &Qualifier,
) -> RequalifiedKey<'ty, 'tmp, T>
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

fn requalify_concrete_inner<'ty, 'tmp>(
    target: PyTypeConcreteKey<'ty>,
    additional: &Qualifier,
    arenas: &mut TypeArenas<'ty>,
    temp: &mut TempConcreteArenas<'ty, 'tmp>,
    memo: &mut HashMap<PyTypeConcreteKey<'ty>, BuildConcreteKey<'ty, 'tmp>>,
) -> BuildConcreteKey<'ty, 'tmp> {
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
        PyType::Class(key) => {
            let placeholder = temp.classes.insert(None);
            let result = BuildConcreteKey::TempClass(placeholder);
            memo.insert(target, result);
            let value = arenas.concrete.classes.get(key).clone();
            let output = Qualified {
                inner: BuildClassType {
                    descriptor: value.inner.descriptor,
                    constructor: value.inner.constructor,
                    args: value
                        .inner
                        .args
                        .into_iter()
                        .map(|child| {
                            requalify_concrete_inner(child, additional, arenas, temp, memo)
                        })
                        .collect(),
                    init: value.inner.init.map(|init| BuildClassInit {
                        params: init
                            .params
                            .into_iter()
                            .map(|(name, child)| {
                                (
                                    name,
                                    requalify_concrete_inner(child, additional, arenas, temp, memo),
                                )
                            })
                            .collect(),
                        param_kinds: init.param_kinds,
                        param_has_default: init.param_has_default,
                    }),
                },
                qualifier: requalified_qualifier(&value.qualifier, additional),
            };
            assert!(
                temp.classes.get_mut(placeholder).replace(output).is_none(),
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
                    protocol_mro: value
                        .inner
                        .protocol_mro
                        .iter()
                        .map(|scope| {
                            map_concrete_protocol_base(scope, |child| {
                                requalify_concrete_inner(child, additional, arenas, temp, memo)
                            })
                        })
                        .collect(),
                    direct_methods: value.inner.direct_methods,
                    methods: map_protocol_method_list(&value.inner.methods, |child| {
                        requalify_concrete_inner(child, additional, arenas, temp, memo)
                    }),
                    attributes: map_member_list(&value.inner.attributes, |child| {
                        requalify_concrete_inner(child, additional, arenas, temp, memo)
                    }),
                    properties: map_member_list(&value.inner.properties, |child| {
                        requalify_concrete_inner(child, additional, arenas, temp, memo)
                    }),
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
                    attributes: map_member_list(&value.inner.attributes, |child| {
                        requalify_concrete_inner(child, additional, arenas, temp, memo)
                    }),
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

#[instrumented(
    name = "inlay.types.requalify_concrete",
    target = "inlay",
        level = "trace",
        skip_all,
        fields(
        additional = %additional.display_compact(),
        cache_hit
    )
)]
pub(crate) fn requalify_concrete<'ty>(
    target: PyTypeConcreteKey<'ty>,
    additional: &Qualifier,
    arenas: &mut TypeArenas<'ty>,
) -> PyTypeConcreteKey<'ty> {
    if additional.is_unqualified() {
        inlay_event!(
            name: "inlay.types.requalify_concrete.result",
            cache_hit = true,
        );
        return target;
    }
    let cache_key = RequalifyConcreteCacheKey {
        source: target,
        additional: additional.clone(),
    };
    if let Some(cached) = arenas.requalify_concrete_cache.get(&cache_key).copied() {
        inlay_event!(
            name: "inlay.types.requalify_concrete.result",
            cache_hit = true,
        );
        return cached;
    }

    let mut temp = TempConcreteArenas::default();
    let root = requalify_concrete_inner(
        target,
        additional,
        arenas,
        &mut temp,
        &mut HashMap::default(),
    );
    let root = commit_concrete_temp(arenas, temp, root);
    let root = canonicalize_if_resolved(root, arenas);
    arenas.requalify_concrete_cache.insert(cache_key, root);
    inlay_event!(
        name: "inlay.types.requalify_concrete.result",
        cache_hit = false,
    );
    root
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use super::*;
    use crate::types::{PlainType, PyTypeDescriptor, PyTypeId};

    fn duplicate_plain_key<'ty>(
        arenas: &mut TypeArenas<'ty>,
        descriptor: &PyTypeDescriptor,
        args: Vec<PyTypeConcreteKey<'ty>>,
    ) -> PyTypeConcreteKey<'ty> {
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
            "write"
        );
    }
}
