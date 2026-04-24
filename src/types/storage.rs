use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::marker::PhantomData;

use derive_where::derive_where;
use slotmap::{KeyData, SlotMap};

use crate::qualifier::Qualifier;

use super::{
    CallableType, Concrete, Keyed, LazyRefType, Parametric, PlainType, ProtocolType, PyType,
    PyTypeConcreteKey, Qual, Qualified, QualifiedMode, SentinelType, TypeKeyMap, TypeVarSupport,
    TypedDictType, UnionType, Viewed, Wrapper,
};

// --- Arena / ArenaFamily ---

pub trait Arena<T: 'static>: Default + 'static {
    type Key: Copy + Eq + Hash + Ord + std::fmt::Debug + 'static;

    fn insert(&mut self, val: T) -> Self::Key
    where
        T: Hash + Eq;

    fn insert_placeholder(&mut self) -> Self::Key;

    fn replace(&mut self, key: Self::Key, val: T) -> Result<Option<T>, ReplaceError>
    where
        T: Hash + Eq;

    fn get(&self, key: &Self::Key) -> Option<&T>;

    fn len(&self) -> usize;
}

pub trait ArenaFamily: 'static {
    type Store<T: 'static>: Arena<T>;
}

pub type KeyOf<S, T> = <<S as ArenaFamily>::Store<T> as Arena<T>>::Key;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplaceError {
    InvalidKey,
}

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
pub struct StoreGroup<S: ArenaFamily, G: TypeVarSupport> {
    pub(crate) plains: S::Store<Qualified<PlainType<Qual<Keyed<S>>, G>>>,
    pub(crate) protocols: S::Store<Qualified<ProtocolType<Qual<Keyed<S>>, G>>>,
    pub(crate) typed_dicts: S::Store<Qualified<TypedDictType<Qual<Keyed<S>>, G>>>,
    pub(crate) unions: S::Store<Qualified<UnionType<Qual<Keyed<S>>, G>>>,
    pub(crate) callables: S::Store<Qualified<CallableType<Qual<Keyed<S>>, G>>>,
    pub(crate) lazy_refs: S::Store<Qualified<LazyRefType<Qual<Keyed<S>>, G>>>,
    pub(crate) type_vars: S::Store<Qualified<G::TypeVar>>,
    pub(crate) param_specs: S::Store<Qualified<G::ParamSpec>>,
}

// --- TypeArenas ---

#[derive_where(Default)]
pub struct TypeArenas<S: ArenaFamily> {
    pub(crate) concrete: StoreGroup<S, Concrete>,
    pub(crate) parametric: StoreGroup<S, Parametric>,
    pub(crate) sentinels: S::Store<Qualified<SentinelType>>,
    pub(crate) deep_hash_caches: super::DeepHashCaches<S>,
    pub(crate) canonical_concrete_qualified: TypeKeyMap<S, QualifiedMode, PyTypeConcreteKey<S>>,
}

// --- ArenaSelector ---

pub trait ArenaSelector: TypeVarSupport + Sized {
    fn stores<S: ArenaFamily>(arenas: &TypeArenas<S>) -> &StoreGroup<S, Self>;
}

impl ArenaSelector for Concrete {
    fn stores<S: ArenaFamily>(arenas: &TypeArenas<S>) -> &StoreGroup<S, Self> {
        &arenas.concrete
    }
}

impl ArenaSelector for Parametric {
    fn stores<S: ArenaFamily>(arenas: &TypeArenas<S>) -> &StoreGroup<S, Self> {
        &arenas.parametric
    }
}

// --- QualView::qualifier ---

impl<'a, S: ArenaFamily, G: TypeVarSupport> PyType<Qual<Viewed<'a>>, Qual<Keyed<S>>, G> {
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

pub(crate) trait ResolveMode<S: ArenaFamily> {
    type View<'a>: Wrapper;

    fn resolve_one<'a, T: 'static>(
        store: &'a S::Store<Qualified<T>>,
        key: &KeyOf<S, Qualified<T>>,
    ) -> Option<<Self::View<'a> as Wrapper>::Wrap<T>>;
}

impl<S: ArenaFamily> ResolveMode<S> for super::QualifiedMode {
    type View<'a> = Qual<Viewed<'a>>;

    fn resolve_one<'a, T: 'static>(
        store: &'a S::Store<Qualified<T>>,
        key: &KeyOf<S, Qualified<T>>,
    ) -> Option<&'a Qualified<T>> {
        store.get(key)
    }
}

impl<S: ArenaFamily> ResolveMode<S> for super::UnqualifiedMode {
    type View<'a> = Viewed<'a>;

    fn resolve_one<'a, T: 'static>(
        store: &'a S::Store<Qualified<T>>,
        key: &KeyOf<S, Qualified<T>>,
    ) -> Option<&'a T> {
        store.get(key).map(|q| &q.inner)
    }
}

// --- TypeArenas::get / get_as ---

impl<S: ArenaFamily> TypeArenas<S> {
    pub fn get<G: ArenaSelector>(
        &self,
        key: PyType<Qual<Keyed<S>>, Qual<Keyed<S>>, G>,
    ) -> Option<PyType<Qual<Viewed<'_>>, Qual<Keyed<S>>, G>> {
        self.get_as::<super::QualifiedMode, G>(key)
    }

    pub(crate) fn get_as<'a, M: ResolveMode<S>, G: ArenaSelector>(
        &'a self,
        key: PyType<Qual<Keyed<S>>, Qual<Keyed<S>>, G>,
    ) -> Option<PyType<M::View<'a>, Qual<Keyed<S>>, G>> {
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

    pub(crate) fn qualifier_of_concrete(&self, r: PyTypeConcreteKey<S>) -> Option<&Qualifier> {
        self.get(r).map(|v| v.qualifier())
    }

    pub(crate) fn concrete_item_count(&self) -> usize {
        self.concrete.plains.len()
            + self.concrete.protocols.len()
            + self.concrete.typed_dicts.len()
            + self.concrete.unions.len()
            + self.concrete.callables.len()
            + self.concrete.lazy_refs.len()
            + self.concrete.type_vars.len()
            + self.concrete.param_specs.len()
    }

    pub(crate) fn parametric_item_count(&self) -> usize {
        self.parametric.plains.len()
            + self.parametric.protocols.len()
            + self.parametric.typed_dicts.len()
            + self.parametric.unions.len()
            + self.parametric.callables.len()
            + self.parametric.lazy_refs.len()
            + self.parametric.type_vars.len()
            + self.parametric.param_specs.len()
    }

    pub(crate) fn sentinel_item_count(&self) -> usize {
        self.sentinels.len()
    }

    pub(crate) fn canonical_concrete_count(&self) -> usize {
        self.canonical_concrete_qualified.len()
    }
}

fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

// --- DedupSlotStore ---

#[derive_where(Default)]
pub struct DedupSlotStore<T: 'static> {
    slots: SlotMap<ArenaKey<T>, Option<T>>,
    index: HashMap<u64, Vec<ArenaKey<T>>>,
}

// --- SlotBackend ---

pub struct SlotBackend;

impl ArenaFamily for SlotBackend {
    type Store<T: 'static> = DedupSlotStore<T>;
}

impl<T: 'static> Arena<T> for DedupSlotStore<T> {
    type Key = ArenaKey<T>;

    fn insert(&mut self, val: T) -> ArenaKey<T>
    where
        T: Hash + Eq,
    {
        let hash = hash_value(&val);

        if let Some(candidates) = self.index.get(&hash) {
            for &k in candidates {
                if self.slots[k].as_ref() == Some(&val) {
                    return k;
                }
            }
        }

        let key = self.slots.insert(Some(val));
        self.index.entry(hash).or_default().push(key);
        key
    }

    fn insert_placeholder(&mut self) -> ArenaKey<T> {
        self.slots.insert(None)
    }

    fn replace(&mut self, key: ArenaKey<T>, val: T) -> Result<Option<T>, ReplaceError>
    where
        T: Hash + Eq,
    {
        let hash = hash_value(&val);
        let old = self
            .slots
            .get_mut(key)
            .ok_or(ReplaceError::InvalidKey)?
            .replace(val);

        if let Some(old_value) = old.as_ref() {
            let old_hash = hash_value(old_value);
            let remove_bucket = {
                let candidates = self
                    .index
                    .get_mut(&old_hash)
                    .expect("stored value must exist in index");
                let original_len = candidates.len();
                candidates.retain(|candidate| *candidate != key);
                assert!(
                    candidates.len() + 1 == original_len,
                    "stored value must include its key in index"
                );
                candidates.is_empty()
            };

            if remove_bucket {
                self.index.remove(&old_hash);
            }
        }

        self.index.entry(hash).or_default().push(key);
        Ok(old)
    }

    fn get(&self, key: &ArenaKey<T>) -> Option<&T> {
        self.slots.get(*key)?.as_ref()
    }

    fn len(&self) -> usize {
        self.slots.len()
    }
}
