use std::{
    hash::{Hash, Hasher},
    marker::PhantomData,
};

use derive_where::derive_where;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};

use super::{
    ArenaSelector, CallableType, Concrete, Keyed, LazyRefType, Parametric, PlainType, ProtocolType,
    PyType, PyTypeConcreteKey, PyTypeKey, PyTypeParametricKey, Qual, QualifiedMode, SentinelType,
    ShallowHash, TypeArenas, TypeChildren, TypedDictType, UnionType, UnqualifiedMode, Wrapper,
};

// --- DeepHashValue ---

#[derive(Debug)]
#[derive_where(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct DeepHashValue<M>(u64, PhantomData<M>);

impl<M> DeepHashValue<M> {
    pub(crate) fn raw(self) -> u64 {
        self.0
    }
}

// --- DeepHashCaches ---

#[derive(Default)]
pub(crate) struct DeepHashCaches {
    concrete_unqualified: HashMap<PyTypeConcreteKey, u64>,
    concrete_qualified: HashMap<PyTypeConcreteKey, u64>,
    parametric_unqualified: HashMap<PyTypeParametricKey, u64>,
    parametric_qualified: HashMap<PyTypeParametricKey, u64>,
}

// --- DeepHashMode trait ---

pub(crate) trait DeepHashMode<G: ArenaSelector> {
    fn resolve_and_hash(
        key: PyTypeKey<G>,
        arenas: &TypeArenas,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<G>>,
    ) where
        G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<G>>,
        G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<G>>;

    fn cache(caches: &DeepHashCaches) -> &HashMap<PyTypeKey<G>, u64>;
    fn cache_mut(caches: &mut DeepHashCaches) -> &mut HashMap<PyTypeKey<G>, u64>;
}

// --- Recursion skeleton ---

fn deep_hash_impl<M: DeepHashMode<G>, G: ArenaSelector>(
    key: PyTypeKey<G>,
    arenas: &TypeArenas,
    state: &mut impl Hasher,
    visited: &mut HashSet<PyTypeKey<G>>,
) where
    G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<G>>,
    G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<G>>,
{
    if !visited.insert(key) {
        return;
    }
    std::mem::discriminant(&key).hash(state);
    M::resolve_and_hash(key, arenas, state, visited);
    visited.remove(&key);
}

// --- One-level helper ---

fn hash_and_recurse<V, M: DeepHashMode<G>, G: ArenaSelector>(
    v: V,
    arenas: &TypeArenas,
    state: &mut impl Hasher,
    visited: &mut HashSet<PyTypeKey<G>>,
) where
    V: ShallowHash + TypeChildren<PyTypeKey<G>>,
    G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<G>>,
    G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<G>>,
{
    v.shallow_hash(state);
    for &dep in v.children() {
        deep_hash_impl::<M, G>(dep, arenas, state, visited);
    }
}

// --- Generic dispatch (single 9-arm match, generic over O and G) ---

impl<O: Wrapper, G: ArenaSelector> PyType<O, Qual<Keyed>, G> {
    fn dispatch_deep_hash<M: DeepHashMode<G>>(
        self,
        arenas: &TypeArenas,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<G>>,
    ) where
        O::Wrap<SentinelType>: ShallowHash + TypeChildren<PyTypeKey<G>>,
        O::Wrap<G::TypeVar>: ShallowHash + TypeChildren<PyTypeKey<G>>,
        O::Wrap<G::ParamSpec>: ShallowHash + TypeChildren<PyTypeKey<G>>,
        O::Wrap<PlainType<Qual<Keyed>, G>>: ShallowHash + TypeChildren<PyTypeKey<G>>,
        O::Wrap<ProtocolType<Qual<Keyed>, G>>: ShallowHash + TypeChildren<PyTypeKey<G>>,
        O::Wrap<TypedDictType<Qual<Keyed>, G>>: ShallowHash + TypeChildren<PyTypeKey<G>>,
        O::Wrap<UnionType<Qual<Keyed>, G>>: ShallowHash + TypeChildren<PyTypeKey<G>>,
        O::Wrap<CallableType<Qual<Keyed>, G>>: ShallowHash + TypeChildren<PyTypeKey<G>>,
        O::Wrap<LazyRefType<Qual<Keyed>, G>>: ShallowHash + TypeChildren<PyTypeKey<G>>,
        G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<G>>,
        G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<G>>,
    {
        match self {
            PyType::Sentinel(v) => hash_and_recurse::<_, M, G>(v, arenas, state, visited),
            PyType::ParamSpec(v) => hash_and_recurse::<_, M, G>(v, arenas, state, visited),
            PyType::Plain(v) => hash_and_recurse::<_, M, G>(v, arenas, state, visited),
            PyType::Protocol(v) => hash_and_recurse::<_, M, G>(v, arenas, state, visited),
            PyType::TypedDict(v) => hash_and_recurse::<_, M, G>(v, arenas, state, visited),
            PyType::Union(v) => hash_and_recurse::<_, M, G>(v, arenas, state, visited),
            PyType::Callable(v) => hash_and_recurse::<_, M, G>(v, arenas, state, visited),
            PyType::LazyRef(v) => hash_and_recurse::<_, M, G>(v, arenas, state, visited),
            PyType::TypeVar(v) => hash_and_recurse::<_, M, G>(v, arenas, state, visited),
        }
    }
}

// --- Mode impls ---

impl DeepHashMode<Concrete> for UnqualifiedMode {
    fn resolve_and_hash(
        key: PyTypeKey<Concrete>,
        arenas: &TypeArenas,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<Concrete>>,
    ) {
        arenas
            .get_as::<Self, Concrete>(key)
            .expect("dangling key")
            .dispatch_deep_hash::<Self>(arenas, state, visited);
    }

    fn cache(caches: &DeepHashCaches) -> &HashMap<PyTypeConcreteKey, u64> {
        &caches.concrete_unqualified
    }

    fn cache_mut(caches: &mut DeepHashCaches) -> &mut HashMap<PyTypeConcreteKey, u64> {
        &mut caches.concrete_unqualified
    }
}

impl DeepHashMode<Parametric> for UnqualifiedMode {
    fn resolve_and_hash(
        key: PyTypeKey<Parametric>,
        arenas: &TypeArenas,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<Parametric>>,
    ) {
        arenas
            .get_as::<Self, Parametric>(key)
            .expect("dangling key")
            .dispatch_deep_hash::<Self>(arenas, state, visited);
    }

    fn cache(caches: &DeepHashCaches) -> &HashMap<PyTypeParametricKey, u64> {
        &caches.parametric_unqualified
    }

    fn cache_mut(caches: &mut DeepHashCaches) -> &mut HashMap<PyTypeParametricKey, u64> {
        &mut caches.parametric_unqualified
    }
}

impl DeepHashMode<Concrete> for QualifiedMode {
    fn resolve_and_hash(
        key: PyTypeKey<Concrete>,
        arenas: &TypeArenas,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<Concrete>>,
    ) {
        arenas
            .get_as::<Self, Concrete>(key)
            .expect("dangling key")
            .dispatch_deep_hash::<Self>(arenas, state, visited);
    }

    fn cache(caches: &DeepHashCaches) -> &HashMap<PyTypeConcreteKey, u64> {
        &caches.concrete_qualified
    }

    fn cache_mut(caches: &mut DeepHashCaches) -> &mut HashMap<PyTypeConcreteKey, u64> {
        &mut caches.concrete_qualified
    }
}

impl DeepHashMode<Parametric> for QualifiedMode {
    fn resolve_and_hash(
        key: PyTypeKey<Parametric>,
        arenas: &TypeArenas,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<Parametric>>,
    ) {
        arenas
            .get_as::<Self, Parametric>(key)
            .expect("dangling key")
            .dispatch_deep_hash::<Self>(arenas, state, visited);
    }

    fn cache(caches: &DeepHashCaches) -> &HashMap<PyTypeParametricKey, u64> {
        &caches.parametric_qualified
    }

    fn cache_mut(caches: &mut DeepHashCaches) -> &mut HashMap<PyTypeParametricKey, u64> {
        &mut caches.parametric_qualified
    }
}

// --- TypeArenas methods ---

impl TypeArenas {
    fn deep_hash_of_uncached<M: DeepHashMode<G>, G: ArenaSelector>(
        &self,
        id: PyTypeKey<G>,
    ) -> DeepHashValue<M>
    where
        G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<G>>,
        G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<G>>,
    {
        let mut hasher = FxHasher::default();
        let mut visited = HashSet::default();
        deep_hash_impl::<M, G>(id, self, &mut hasher, &mut visited);
        DeepHashValue(hasher.finish(), PhantomData)
    }

    pub(crate) fn deep_hash_concrete<M: DeepHashMode<Concrete>>(
        &mut self,
        key: PyTypeConcreteKey,
    ) -> DeepHashValue<M> {
        if let Some(&h) = M::cache(&self.deep_hash_caches).get(&key) {
            return DeepHashValue(h, PhantomData);
        }
        let value = self.deep_hash_of_uncached::<M, Concrete>(key);
        M::cache_mut(&mut self.deep_hash_caches).insert(key, value.raw());
        value
    }
}
