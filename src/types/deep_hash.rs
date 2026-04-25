use std::{
    collections::{HashMap, HashSet},
    hash::{DefaultHasher, Hash, Hasher},
    marker::PhantomData,
};

use derive_where::derive_where;

use super::{
    ArenaFamily, ArenaSelector, CallableType, Concrete, Keyed, LazyRefType, Parametric, PlainType,
    ProtocolType, PyType, PyTypeConcreteKey, PyTypeKey, PyTypeParametricKey, Qual, QualifiedMode,
    SentinelType, ShallowHash, TypeArenas, TypeChildren, TypedDictType, UnionType, UnqualifiedMode,
    Wrapper,
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

#[derive_where(Default)]
pub(crate) struct DeepHashCaches<S: ArenaFamily> {
    concrete_unqualified: HashMap<PyTypeConcreteKey<S>, u64>,
    concrete_qualified: HashMap<PyTypeConcreteKey<S>, u64>,
    parametric_unqualified: HashMap<PyTypeParametricKey<S>, u64>,
    parametric_qualified: HashMap<PyTypeParametricKey<S>, u64>,
}

impl<S: ArenaFamily> DeepHashCaches<S> {
    pub(crate) fn len(&self) -> usize {
        self.concrete_unqualified.len()
            + self.concrete_qualified.len()
            + self.parametric_unqualified.len()
            + self.parametric_qualified.len()
    }
}

// --- DeepHashMode trait ---

pub(crate) trait DeepHashMode<S: ArenaFamily, G: ArenaSelector> {
    fn resolve_and_hash(
        key: PyTypeKey<S, G>,
        arenas: &TypeArenas<S>,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<S, G>>,
    ) where
        G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
        G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<S, G>>;

    fn cache(caches: &DeepHashCaches<S>) -> &HashMap<PyTypeKey<S, G>, u64>;
    fn cache_mut(caches: &mut DeepHashCaches<S>) -> &mut HashMap<PyTypeKey<S, G>, u64>;
}

// --- Recursion skeleton ---

fn deep_hash_impl<S: ArenaFamily, M: DeepHashMode<S, G>, G: ArenaSelector>(
    key: PyTypeKey<S, G>,
    arenas: &TypeArenas<S>,
    state: &mut impl Hasher,
    visited: &mut HashSet<PyTypeKey<S, G>>,
) where
    G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
    G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
{
    if !visited.insert(key) {
        return;
    }
    std::mem::discriminant(&key).hash(state);
    M::resolve_and_hash(key, arenas, state, visited);
    visited.remove(&key);
}

// --- One-level helper ---

fn hash_and_recurse<V, S: ArenaFamily, M: DeepHashMode<S, G>, G: ArenaSelector>(
    v: V,
    arenas: &TypeArenas<S>,
    state: &mut impl Hasher,
    visited: &mut HashSet<PyTypeKey<S, G>>,
) where
    V: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
    G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
    G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
{
    v.shallow_hash(state);
    for &dep in v.children() {
        deep_hash_impl::<S, M, G>(dep, arenas, state, visited);
    }
}

// --- Generic dispatch (single 9-arm match, generic over O and G) ---

impl<O: Wrapper, S: ArenaFamily, G: ArenaSelector> PyType<O, Qual<Keyed<S>>, G> {
    fn dispatch_deep_hash<M: DeepHashMode<S, G>>(
        self,
        arenas: &TypeArenas<S>,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<S, G>>,
    ) where
        O::Wrap<SentinelType>: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
        O::Wrap<G::TypeVar>: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
        O::Wrap<G::ParamSpec>: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
        O::Wrap<PlainType<Qual<Keyed<S>>, G>>: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
        O::Wrap<ProtocolType<Qual<Keyed<S>>, G>>: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
        O::Wrap<TypedDictType<Qual<Keyed<S>>, G>>: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
        O::Wrap<UnionType<Qual<Keyed<S>>, G>>: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
        O::Wrap<CallableType<Qual<Keyed<S>>, G>>: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
        O::Wrap<LazyRefType<Qual<Keyed<S>>, G>>: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
        G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
        G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
    {
        match self {
            PyType::Sentinel(v) => hash_and_recurse::<_, S, M, G>(v, arenas, state, visited),
            PyType::ParamSpec(v) => hash_and_recurse::<_, S, M, G>(v, arenas, state, visited),
            PyType::Plain(v) => hash_and_recurse::<_, S, M, G>(v, arenas, state, visited),
            PyType::Protocol(v) => hash_and_recurse::<_, S, M, G>(v, arenas, state, visited),
            PyType::TypedDict(v) => hash_and_recurse::<_, S, M, G>(v, arenas, state, visited),
            PyType::Union(v) => hash_and_recurse::<_, S, M, G>(v, arenas, state, visited),
            PyType::Callable(v) => hash_and_recurse::<_, S, M, G>(v, arenas, state, visited),
            PyType::LazyRef(v) => hash_and_recurse::<_, S, M, G>(v, arenas, state, visited),
            PyType::TypeVar(v) => hash_and_recurse::<_, S, M, G>(v, arenas, state, visited),
        }
    }
}

// --- Mode impls ---

impl<S: ArenaFamily> DeepHashMode<S, Concrete> for UnqualifiedMode {
    fn resolve_and_hash(
        key: PyTypeKey<S, Concrete>,
        arenas: &TypeArenas<S>,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<S, Concrete>>,
    ) {
        arenas
            .get_as::<Self, Concrete>(key)
            .expect("dangling key")
            .dispatch_deep_hash::<Self>(arenas, state, visited);
    }

    fn cache(caches: &DeepHashCaches<S>) -> &HashMap<PyTypeConcreteKey<S>, u64> {
        &caches.concrete_unqualified
    }

    fn cache_mut(caches: &mut DeepHashCaches<S>) -> &mut HashMap<PyTypeConcreteKey<S>, u64> {
        &mut caches.concrete_unqualified
    }
}

impl<S: ArenaFamily> DeepHashMode<S, Parametric> for UnqualifiedMode {
    fn resolve_and_hash(
        key: PyTypeKey<S, Parametric>,
        arenas: &TypeArenas<S>,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<S, Parametric>>,
    ) {
        arenas
            .get_as::<Self, Parametric>(key)
            .expect("dangling key")
            .dispatch_deep_hash::<Self>(arenas, state, visited);
    }

    fn cache(caches: &DeepHashCaches<S>) -> &HashMap<PyTypeParametricKey<S>, u64> {
        &caches.parametric_unqualified
    }

    fn cache_mut(caches: &mut DeepHashCaches<S>) -> &mut HashMap<PyTypeParametricKey<S>, u64> {
        &mut caches.parametric_unqualified
    }
}

impl<S: ArenaFamily> DeepHashMode<S, Concrete> for QualifiedMode {
    fn resolve_and_hash(
        key: PyTypeKey<S, Concrete>,
        arenas: &TypeArenas<S>,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<S, Concrete>>,
    ) {
        arenas
            .get_as::<Self, Concrete>(key)
            .expect("dangling key")
            .dispatch_deep_hash::<Self>(arenas, state, visited);
    }

    fn cache(caches: &DeepHashCaches<S>) -> &HashMap<PyTypeConcreteKey<S>, u64> {
        &caches.concrete_qualified
    }

    fn cache_mut(caches: &mut DeepHashCaches<S>) -> &mut HashMap<PyTypeConcreteKey<S>, u64> {
        &mut caches.concrete_qualified
    }
}

impl<S: ArenaFamily> DeepHashMode<S, Parametric> for QualifiedMode {
    fn resolve_and_hash(
        key: PyTypeKey<S, Parametric>,
        arenas: &TypeArenas<S>,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<S, Parametric>>,
    ) {
        arenas
            .get_as::<Self, Parametric>(key)
            .expect("dangling key")
            .dispatch_deep_hash::<Self>(arenas, state, visited);
    }

    fn cache(caches: &DeepHashCaches<S>) -> &HashMap<PyTypeParametricKey<S>, u64> {
        &caches.parametric_qualified
    }

    fn cache_mut(caches: &mut DeepHashCaches<S>) -> &mut HashMap<PyTypeParametricKey<S>, u64> {
        &mut caches.parametric_qualified
    }
}

// --- TypeArenas methods ---

impl<S: ArenaFamily> TypeArenas<S> {
    fn deep_hash_of_uncached<M: DeepHashMode<S, G>, G: ArenaSelector>(
        &self,
        id: PyTypeKey<S, G>,
    ) -> DeepHashValue<M>
    where
        G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
        G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<S, G>>,
    {
        let mut hasher = DefaultHasher::new();
        let mut visited = HashSet::new();
        deep_hash_impl::<S, M, G>(id, self, &mut hasher, &mut visited);
        DeepHashValue(hasher.finish(), PhantomData)
    }

    pub(crate) fn deep_hash_concrete<M: DeepHashMode<S, Concrete>>(
        &mut self,
        key: PyTypeConcreteKey<S>,
    ) -> DeepHashValue<M> {
        if let Some(&h) = M::cache(&self.deep_hash_caches).get(&key) {
            return DeepHashValue(h, PhantomData);
        }
        let value = self.deep_hash_of_uncached::<M, Concrete>(key);
        M::cache_mut(&mut self.deep_hash_caches).insert(key, value.raw());
        value
    }

    pub(crate) fn deep_hash_parametric<M: DeepHashMode<S, Parametric>>(
        &mut self,
        key: PyTypeParametricKey<S>,
    ) -> DeepHashValue<M> {
        if let Some(&h) = M::cache(&self.deep_hash_caches).get(&key) {
            return DeepHashValue(h, PhantomData);
        }
        let value = self.deep_hash_of_uncached::<M, Parametric>(key);
        M::cache_mut(&mut self.deep_hash_caches).insert(key, value.raw());
        value
    }
}
