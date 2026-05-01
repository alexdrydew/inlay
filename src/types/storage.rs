use std::marker::PhantomData;

use derive_where::derive_where;
use slotmap::{KeyData, SlotMap};

use crate::qualifier::Qualifier;

use super::{
    CallableType, Concrete, Keyed, LazyRefType, Parametric, PlainType, ProtocolType, PyType,
    PyTypeConcreteKey, Qual, Qualified, QualifiedMode, SentinelType, TypeKeyMap, TypeVarSupport,
    TypedDictType, UnionType, Viewed, Wrapper,
};

pub type KeyOf<T> = ArenaKey<T>;

// --- ArenaKey ---

#[repr(transparent)]
#[derive_where(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ArenaKey<T: 'static>(KeyData, PhantomData<T>);

impl<T: 'static> PartialOrd for ArenaKey<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: 'static> Ord for ArenaKey<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.as_ffi().cmp(&other.0.as_ffi())
    }
}

impl<T: 'static> std::fmt::Debug for ArenaKey<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ArenaKey({:?})", self.0)
    }
}

impl<T: 'static> From<KeyData> for ArenaKey<T> {
    fn from(data: KeyData) -> Self {
        Self(data, PhantomData)
    }
}

unsafe impl<T: 'static> slotmap::Key for ArenaKey<T> {
    fn data(&self) -> KeyData {
        self.0
    }
}

// --- StoreGroup ---

#[derive_where(Default)]
pub struct StoreGroup<G: TypeVarSupport> {
    pub(crate) plains: SlotMap<
        ArenaKey<Qualified<PlainType<Qual<Keyed>, G>>>,
        Option<Qualified<PlainType<Qual<Keyed>, G>>>,
    >,
    pub(crate) protocols: SlotMap<
        ArenaKey<Qualified<ProtocolType<Qual<Keyed>, G>>>,
        Option<Qualified<ProtocolType<Qual<Keyed>, G>>>,
    >,
    pub(crate) typed_dicts: SlotMap<
        ArenaKey<Qualified<TypedDictType<Qual<Keyed>, G>>>,
        Option<Qualified<TypedDictType<Qual<Keyed>, G>>>,
    >,
    pub(crate) unions: SlotMap<
        ArenaKey<Qualified<UnionType<Qual<Keyed>, G>>>,
        Option<Qualified<UnionType<Qual<Keyed>, G>>>,
    >,
    pub(crate) callables: SlotMap<
        ArenaKey<Qualified<CallableType<Qual<Keyed>, G>>>,
        Option<Qualified<CallableType<Qual<Keyed>, G>>>,
    >,
    pub(crate) lazy_refs: SlotMap<
        ArenaKey<Qualified<LazyRefType<Qual<Keyed>, G>>>,
        Option<Qualified<LazyRefType<Qual<Keyed>, G>>>,
    >,
    pub(crate) type_vars: SlotMap<ArenaKey<Qualified<G::TypeVar>>, Option<Qualified<G::TypeVar>>>,
    pub(crate) param_specs:
        SlotMap<ArenaKey<Qualified<G::ParamSpec>>, Option<Qualified<G::ParamSpec>>>,
}

// --- TypeArenas ---

#[derive(Default)]
pub struct TypeArenas {
    pub(crate) concrete: StoreGroup<Concrete>,
    pub(crate) parametric: StoreGroup<Parametric>,
    pub(crate) sentinels:
        SlotMap<ArenaKey<Qualified<SentinelType>>, Option<Qualified<SentinelType>>>,
    pub(crate) deep_hash_caches: super::DeepHashCaches,
    pub(crate) canonical_concrete_qualified: TypeKeyMap<QualifiedMode, PyTypeConcreteKey>,
}

// --- ArenaSelector ---

pub trait ArenaSelector: TypeVarSupport + Sized {
    fn stores(arenas: &TypeArenas) -> &StoreGroup<Self>;
}

impl ArenaSelector for Concrete {
    fn stores(arenas: &TypeArenas) -> &StoreGroup<Self> {
        &arenas.concrete
    }
}

impl ArenaSelector for Parametric {
    fn stores(arenas: &TypeArenas) -> &StoreGroup<Self> {
        &arenas.parametric
    }
}

// --- QualView::qualifier ---

impl<'a, G: TypeVarSupport> PyType<Qual<Viewed<'a>>, Qual<Keyed>, G> {
    pub(crate) fn qualifier(&self) -> &'a Qualifier {
        match self {
            PyType::Sentinel(v) => &v.qualifier,
            PyType::ParamSpec(v) => &v.qualifier,
            PyType::Plain(v) => &v.qualifier,
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

    fn resolve_one<'a, T: 'static>(
        store: &'a SlotMap<KeyOf<Qualified<T>>, Option<Qualified<T>>>,
        key: &KeyOf<Qualified<T>>,
    ) -> Option<<Self::View<'a> as Wrapper>::Wrap<T>>;
}

impl ResolveMode for super::QualifiedMode {
    type View<'a> = Qual<Viewed<'a>>;

    fn resolve_one<'a, T: 'static>(
        store: &'a SlotMap<KeyOf<Qualified<T>>, Option<Qualified<T>>>,
        key: &KeyOf<Qualified<T>>,
    ) -> Option<&'a Qualified<T>> {
        store.get(*key)?.as_ref()
    }
}

impl ResolveMode for super::UnqualifiedMode {
    type View<'a> = Viewed<'a>;

    fn resolve_one<'a, T: 'static>(
        store: &'a SlotMap<KeyOf<Qualified<T>>, Option<Qualified<T>>>,
        key: &KeyOf<Qualified<T>>,
    ) -> Option<&'a T> {
        store.get(*key)?.as_ref().map(|q| &q.inner)
    }
}

// --- TypeArenas::get / get_as ---

impl TypeArenas {
    pub fn get<G: ArenaSelector>(
        &self,
        key: PyType<Qual<Keyed>, Qual<Keyed>, G>,
    ) -> Option<PyType<Qual<Viewed<'_>>, Qual<Keyed>, G>> {
        self.get_as::<super::QualifiedMode, G>(key)
    }

    pub(crate) fn get_as<M: ResolveMode, G: ArenaSelector>(
        &self,
        key: PyType<Qual<Keyed>, Qual<Keyed>, G>,
    ) -> Option<PyType<M::View<'_>, Qual<Keyed>, G>> {
        let sg = G::stores(self);

        match key {
            PyType::Sentinel(p) => M::resolve_one(&self.sentinels, &p).map(PyType::Sentinel),
            PyType::ParamSpec(p) => M::resolve_one(&sg.param_specs, &p).map(PyType::ParamSpec),
            PyType::Plain(p) => M::resolve_one(&sg.plains, &p).map(PyType::Plain),
            PyType::Protocol(p) => M::resolve_one(&sg.protocols, &p).map(PyType::Protocol),
            PyType::TypedDict(p) => M::resolve_one(&sg.typed_dicts, &p).map(PyType::TypedDict),
            PyType::Union(p) => M::resolve_one(&sg.unions, &p).map(PyType::Union),
            PyType::Callable(p) => M::resolve_one(&sg.callables, &p).map(PyType::Callable),
            PyType::LazyRef(p) => M::resolve_one(&sg.lazy_refs, &p).map(PyType::LazyRef),
            PyType::TypeVar(p) => M::resolve_one(&sg.type_vars, &p).map(PyType::TypeVar),
        }
    }

    pub(crate) fn qualifier_of_concrete(&self, r: PyTypeConcreteKey) -> Option<&Qualifier> {
        self.get(r).map(|v| v.qualifier())
    }
}
