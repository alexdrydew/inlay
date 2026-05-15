use std::marker::PhantomData;

use derive_where::derive_where;
use rustc_hash::FxHashMap as HashMap;

use crate::qualifier::Qualifier;

use super::{
    CallableType, ClassType, Concrete, Keyed, LazyRefType, Parametric, PlainType, ProtocolType,
    PyType, PyTypeConcreteKey, PyTypeId, PyTypeParametricKey, Qual, Qualified, QualifiedMode,
    SentinelType, TypeKeyMap, TypeVarSupport, TypedDictType, UnionType, ViewRef, Viewed, Wrapper,
};

pub type KeyOf<'arena, T> = ArenaKey<'arena, T>;
type ArenaMarker<'arena, T> = PhantomData<(&'arena (), fn(T) -> T)>;

// --- ArenaKey ---

#[derive_where(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ArenaKey<'arena, T> {
    index: u32,
    _marker: ArenaMarker<'arena, T>,
}

impl<T> PartialOrd for ArenaKey<'_, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for ArenaKey<'_, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.index.cmp(&other.index)
    }
}

impl<T> std::fmt::Debug for ArenaKey<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ArenaKey({})", self.index)
    }
}

impl<T> ArenaKey<'_, T> {
    pub(crate) fn index(self) -> usize {
        self.index as usize
    }

    fn new(index: usize) -> Self {
        Self {
            index: index
                .try_into()
                .expect("type arena cannot exceed u32::MAX entries"),
            _marker: PhantomData,
        }
    }
}

// --- Arena ---

pub struct Arena<'arena, T, V = T> {
    values: Vec<V>,
    _marker: ArenaMarker<'arena, T>,
}

impl<'arena, T, V> Arena<'arena, T, V> {
    pub(crate) fn insert(&mut self, value: V) -> ArenaKey<'arena, T> {
        let key = ArenaKey::new(self.values.len());
        self.values.push(value);
        key
    }

    pub(crate) fn get(&self, key: ArenaKey<'arena, T>) -> &V {
        &self.values[key.index()]
    }

    pub(crate) fn get_mut(&mut self, key: ArenaKey<'arena, T>) -> &mut V {
        &mut self.values[key.index()]
    }

    pub(crate) fn future_key(&self, offset: usize) -> ArenaKey<'arena, T> {
        ArenaKey::new(self.values.len() + offset)
    }

    pub(crate) fn push_committed(&mut self, value: V) {
        self.values.push(value);
    }

    pub(crate) fn values(&self) -> &[V] {
        &self.values
    }

    pub(crate) fn truncate(&mut self, len: usize) {
        self.values.truncate(len);
    }

    pub(crate) fn into_values(self) -> Vec<V> {
        self.values
    }
}

// --- ConcreteArenaSnapshot ---

#[derive(Clone, Copy)]
pub(crate) struct ConcreteArenaSnapshot {
    type_vars: usize,
    param_specs: usize,
    plains: usize,
    classes: usize,
    protocols: usize,
    typed_dicts: usize,
    unions: usize,
    callables: usize,
    lazy_refs: usize,
}

impl<T, V> Default for Arena<'_, T, V> {
    fn default() -> Self {
        Self {
            values: Vec::new(),
            _marker: PhantomData,
        }
    }
}

// --- StoreGroup ---

#[derive_where(Default)]
pub struct StoreGroup<'arena, G: TypeVarSupport> {
    pub(crate) plains: Arena<'arena, Qualified<PlainType<Qual<Keyed<'arena>>, G>>>,
    pub(crate) classes: Arena<'arena, Qualified<ClassType<Qual<Keyed<'arena>>, G>>>,
    pub(crate) protocols: Arena<'arena, Qualified<ProtocolType<Qual<Keyed<'arena>>, G>>>,
    pub(crate) typed_dicts: Arena<'arena, Qualified<TypedDictType<Qual<Keyed<'arena>>, G>>>,
    pub(crate) unions: Arena<'arena, Qualified<UnionType<Qual<Keyed<'arena>>, G>>>,
    pub(crate) callables: Arena<'arena, Qualified<CallableType<Qual<Keyed<'arena>>, G>>>,
    pub(crate) lazy_refs: Arena<'arena, Qualified<LazyRefType<Qual<Keyed<'arena>>, G>>>,
    pub(crate) type_vars: Arena<'arena, Qualified<G::TypeVar>>,
    pub(crate) param_specs: Arena<'arena, Qualified<G::ParamSpec>>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct ApplyBindingsCacheKey<'arena> {
    pub(crate) source: PyTypeParametricKey<'arena>,
    pub(crate) type_vars: Vec<(PyTypeId, PyTypeConcreteKey<'arena>)>,
    pub(crate) param_specs: Vec<(PyTypeId, PyTypeConcreteKey<'arena>)>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct RequalifyConcreteCacheKey<'arena> {
    pub(crate) source: PyTypeConcreteKey<'arena>,
    pub(crate) additional: Qualifier,
}

// --- TypeArenas ---

#[derive(Default)]
pub struct TypeArenas<'arena> {
    pub(crate) concrete: StoreGroup<'arena, Concrete>,
    pub(crate) parametric: StoreGroup<'arena, Parametric>,
    pub(crate) sentinels: Arena<'arena, Qualified<SentinelType>>,
    pub(crate) deep_hash_caches: super::DeepHashCaches<'arena>,
    pub(crate) canonical_concrete_qualified:
        TypeKeyMap<'arena, QualifiedMode, PyTypeConcreteKey<'arena>>,
    pub(crate) apply_bindings_cache:
        HashMap<ApplyBindingsCacheKey<'arena>, PyTypeConcreteKey<'arena>>,
    pub(crate) requalify_concrete_cache:
        HashMap<RequalifyConcreteCacheKey<'arena>, PyTypeConcreteKey<'arena>>,
}

// --- ArenaSelector ---

pub trait ArenaSelector<'arena>: TypeVarSupport + Sized {
    fn stores<'a>(arenas: &'a TypeArenas<'arena>) -> &'a StoreGroup<'arena, Self>;
}

impl<'arena> ArenaSelector<'arena> for Concrete {
    fn stores<'a>(arenas: &'a TypeArenas<'arena>) -> &'a StoreGroup<'arena, Self> {
        &arenas.concrete
    }
}

impl<'arena> ArenaSelector<'arena> for Parametric {
    fn stores<'a>(arenas: &'a TypeArenas<'arena>) -> &'a StoreGroup<'arena, Self> {
        &arenas.parametric
    }
}

// --- QualView::qualifier ---

impl<G: TypeVarSupport> PyType<Qual<Viewed<'_>>, Qual<Keyed<'_>>, G> {
    pub(crate) fn qualifier(&self) -> &Qualifier {
        match self {
            PyType::Sentinel(v) => &v.qualifier,
            PyType::ParamSpec(v) => &v.qualifier,
            PyType::Plain(v) => &v.qualifier,
            PyType::Class(v) => &v.qualifier,
            PyType::Protocol(v) => &v.qualifier,
            PyType::TypedDict(v) => &v.qualifier,
            PyType::Union(v) => &v.qualifier,
            PyType::Callable(v) => &v.qualifier,
            PyType::LazyRef(v) => &v.qualifier,
            PyType::TypeVar(v) => &v.qualifier,
        }
    }
}

// --- ResolveMode ---

pub(crate) trait ResolveMode {
    type View<'a>: Wrapper;

    fn resolve_one<'a, 'arena, T>(
        store: &'a Arena<'arena, Qualified<T>>,
        key: &KeyOf<'arena, Qualified<T>>,
    ) -> <Self::View<'a> as Wrapper>::Wrap<T>;
}

impl ResolveMode for super::QualifiedMode {
    type View<'a> = Qual<Viewed<'a>>;

    fn resolve_one<'a, 'arena, T>(
        store: &'a Arena<'arena, Qualified<T>>,
        key: &KeyOf<'arena, Qualified<T>>,
    ) -> ViewRef<'a, Qualified<T>> {
        ViewRef::new(store.get(*key))
    }
}

impl ResolveMode for super::UnqualifiedMode {
    type View<'a> = Viewed<'a>;

    fn resolve_one<'a, 'arena, T>(
        store: &'a Arena<'arena, Qualified<T>>,
        key: &KeyOf<'arena, Qualified<T>>,
    ) -> ViewRef<'a, T> {
        ViewRef::new(&store.get(*key).inner)
    }
}

// --- TypeArenas::get / get_as ---

impl<'arena> TypeArenas<'arena> {
    pub(crate) fn concrete_snapshot(&self) -> ConcreteArenaSnapshot {
        ConcreteArenaSnapshot {
            type_vars: self.concrete.type_vars.values().len(),
            param_specs: self.concrete.param_specs.values().len(),
            plains: self.concrete.plains.values().len(),
            classes: self.concrete.classes.values().len(),
            protocols: self.concrete.protocols.values().len(),
            typed_dicts: self.concrete.typed_dicts.values().len(),
            unions: self.concrete.unions.values().len(),
            callables: self.concrete.callables.values().len(),
            lazy_refs: self.concrete.lazy_refs.values().len(),
        }
    }

    pub(crate) fn truncate_concrete_to(&mut self, snapshot: ConcreteArenaSnapshot) {
        self.concrete.type_vars.truncate(snapshot.type_vars);
        self.concrete.param_specs.truncate(snapshot.param_specs);
        self.concrete.plains.truncate(snapshot.plains);
        self.concrete.classes.truncate(snapshot.classes);
        self.concrete.protocols.truncate(snapshot.protocols);
        self.concrete.typed_dicts.truncate(snapshot.typed_dicts);
        self.concrete.unions.truncate(snapshot.unions);
        self.concrete.callables.truncate(snapshot.callables);
        self.concrete.lazy_refs.truncate(snapshot.lazy_refs);
        self.deep_hash_caches
            .retain_concrete(|key| concrete_key_before_snapshot(key, snapshot));
    }

    pub fn get<G: ArenaSelector<'arena>>(
        &self,
        key: PyType<Qual<Keyed<'arena>>, Qual<Keyed<'arena>>, G>,
    ) -> PyType<Qual<Viewed<'_>>, Qual<Keyed<'arena>>, G> {
        self.get_as::<super::QualifiedMode, G>(key)
    }

    pub(crate) fn get_as<M: ResolveMode, G: ArenaSelector<'arena>>(
        &self,
        key: PyType<Qual<Keyed<'arena>>, Qual<Keyed<'arena>>, G>,
    ) -> PyType<M::View<'_>, Qual<Keyed<'arena>>, G> {
        let sg = G::stores(self);

        match key {
            PyType::Sentinel(p) => PyType::Sentinel(M::resolve_one(&self.sentinels, &p)),
            PyType::ParamSpec(p) => PyType::ParamSpec(M::resolve_one(&sg.param_specs, &p)),
            PyType::Plain(p) => PyType::Plain(M::resolve_one(&sg.plains, &p)),
            PyType::Class(p) => PyType::Class(M::resolve_one(&sg.classes, &p)),
            PyType::Protocol(p) => PyType::Protocol(M::resolve_one(&sg.protocols, &p)),
            PyType::TypedDict(p) => PyType::TypedDict(M::resolve_one(&sg.typed_dicts, &p)),
            PyType::Union(p) => PyType::Union(M::resolve_one(&sg.unions, &p)),
            PyType::Callable(p) => PyType::Callable(M::resolve_one(&sg.callables, &p)),
            PyType::LazyRef(p) => PyType::LazyRef(M::resolve_one(&sg.lazy_refs, &p)),
            PyType::TypeVar(p) => PyType::TypeVar(M::resolve_one(&sg.type_vars, &p)),
        }
    }

    pub(crate) fn qualifier_of_concrete(&self, r: PyTypeConcreteKey<'arena>) -> &Qualifier {
        match r {
            PyType::Sentinel(key) => &self.sentinels.get(key).qualifier,
            PyType::ParamSpec(key) => &self.concrete.param_specs.get(key).qualifier,
            PyType::Plain(key) => &self.concrete.plains.get(key).qualifier,
            PyType::Class(key) => &self.concrete.classes.get(key).qualifier,
            PyType::Protocol(key) => &self.concrete.protocols.get(key).qualifier,
            PyType::TypedDict(key) => &self.concrete.typed_dicts.get(key).qualifier,
            PyType::Union(key) => &self.concrete.unions.get(key).qualifier,
            PyType::Callable(key) => &self.concrete.callables.get(key).qualifier,
            PyType::LazyRef(key) => &self.concrete.lazy_refs.get(key).qualifier,
            PyType::TypeVar(key) => &self.concrete.type_vars.get(key).qualifier,
        }
    }
}

fn concrete_key_before_snapshot(
    key: PyTypeConcreteKey<'_>,
    snapshot: ConcreteArenaSnapshot,
) -> bool {
    match key {
        PyType::Sentinel(_) => true,
        PyType::ParamSpec(key) => key.index() < snapshot.param_specs,
        PyType::Plain(key) => key.index() < snapshot.plains,
        PyType::Class(key) => key.index() < snapshot.classes,
        PyType::Protocol(key) => key.index() < snapshot.protocols,
        PyType::TypedDict(key) => key.index() < snapshot.typed_dicts,
        PyType::Union(key) => key.index() < snapshot.unions,
        PyType::Callable(key) => key.index() < snapshot.callables,
        PyType::LazyRef(key) => key.index() < snapshot.lazy_refs,
        PyType::TypeVar(key) => key.index() < snapshot.type_vars,
    }
}
