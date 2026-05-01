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
pub(crate) struct DeepHashCaches<'types> {
    concrete_unqualified: HashMap<PyTypeConcreteKey<'types>, u64>,
    concrete_qualified: HashMap<PyTypeConcreteKey<'types>, u64>,
    parametric_unqualified: HashMap<PyTypeParametricKey<'types>, u64>,
    parametric_qualified: HashMap<PyTypeParametricKey<'types>, u64>,
}

// --- DeepHashMode trait ---

pub(crate) trait DeepHashMode<'types, G: ArenaSelector<'types>> {
    fn resolve_and_hash(
        key: PyTypeKey<'types, G>,
        arenas: &TypeArenas<'types>,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<'types, G>>,
    ) where
        G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
        G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<'types, G>>;

    fn cache<'a>(caches: &'a DeepHashCaches<'types>) -> &'a HashMap<PyTypeKey<'types, G>, u64>;
    fn cache_mut<'a>(
        caches: &'a mut DeepHashCaches<'types>,
    ) -> &'a mut HashMap<PyTypeKey<'types, G>, u64>;
}

// --- Recursion skeleton ---

fn deep_hash_impl<'types, M: DeepHashMode<'types, G>, G: ArenaSelector<'types>>(
    key: PyTypeKey<'types, G>,
    arenas: &TypeArenas<'types>,
    state: &mut impl Hasher,
    visited: &mut HashSet<PyTypeKey<'types, G>>,
) where
    G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
    G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
{
    if !visited.insert(key) {
        return;
    }
    std::mem::discriminant(&key).hash(state);
    M::resolve_and_hash(key, arenas, state, visited);
    visited.remove(&key);
}

// --- One-level helper ---

fn hash_and_recurse<'types, V, M: DeepHashMode<'types, G>, G: ArenaSelector<'types>>(
    v: V,
    arenas: &TypeArenas<'types>,
    state: &mut impl Hasher,
    visited: &mut HashSet<PyTypeKey<'types, G>>,
) where
    V: ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
    G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
    G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
{
    v.shallow_hash(state);
    for &dep in v.children() {
        deep_hash_impl::<M, G>(dep, arenas, state, visited);
    }
}

// --- Generic dispatch (single 9-arm match, generic over O and G) ---

impl<'types, O: Wrapper, G: ArenaSelector<'types>> PyType<O, Qual<Keyed<'types>>, G> {
    fn dispatch_deep_hash<M: DeepHashMode<'types, G>>(
        self,
        arenas: &TypeArenas<'types>,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<'types, G>>,
    ) where
        O::Wrap<SentinelType>: ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
        O::Wrap<G::TypeVar>: ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
        O::Wrap<G::ParamSpec>: ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
        O::Wrap<PlainType<Qual<Keyed<'types>>, G>>:
            ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
        O::Wrap<ProtocolType<Qual<Keyed<'types>>, G>>:
            ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
        O::Wrap<TypedDictType<Qual<Keyed<'types>>, G>>:
            ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
        O::Wrap<UnionType<Qual<Keyed<'types>>, G>>:
            ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
        O::Wrap<CallableType<Qual<Keyed<'types>>, G>>:
            ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
        O::Wrap<LazyRefType<Qual<Keyed<'types>>, G>>:
            ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
        G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
        G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
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

impl<'types> DeepHashMode<'types, Concrete> for UnqualifiedMode {
    fn resolve_and_hash(
        key: PyTypeKey<'types, Concrete>,
        arenas: &TypeArenas<'types>,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<'types, Concrete>>,
    ) {
        arenas
            .get_as::<Self, Concrete>(key)
            .dispatch_deep_hash::<Self>(arenas, state, visited);
    }

    fn cache<'a>(
        caches: &'a DeepHashCaches<'types>,
    ) -> &'a HashMap<PyTypeConcreteKey<'types>, u64> {
        &caches.concrete_unqualified
    }

    fn cache_mut<'a>(
        caches: &'a mut DeepHashCaches<'types>,
    ) -> &'a mut HashMap<PyTypeConcreteKey<'types>, u64> {
        &mut caches.concrete_unqualified
    }
}

impl<'types> DeepHashMode<'types, Parametric> for UnqualifiedMode {
    fn resolve_and_hash(
        key: PyTypeKey<'types, Parametric>,
        arenas: &TypeArenas<'types>,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<'types, Parametric>>,
    ) {
        arenas
            .get_as::<Self, Parametric>(key)
            .dispatch_deep_hash::<Self>(arenas, state, visited);
    }

    fn cache<'a>(
        caches: &'a DeepHashCaches<'types>,
    ) -> &'a HashMap<PyTypeParametricKey<'types>, u64> {
        &caches.parametric_unqualified
    }

    fn cache_mut<'a>(
        caches: &'a mut DeepHashCaches<'types>,
    ) -> &'a mut HashMap<PyTypeParametricKey<'types>, u64> {
        &mut caches.parametric_unqualified
    }
}

impl<'types> DeepHashMode<'types, Concrete> for QualifiedMode {
    fn resolve_and_hash(
        key: PyTypeKey<'types, Concrete>,
        arenas: &TypeArenas<'types>,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<'types, Concrete>>,
    ) {
        arenas
            .get_as::<Self, Concrete>(key)
            .dispatch_deep_hash::<Self>(arenas, state, visited);
    }

    fn cache<'a>(
        caches: &'a DeepHashCaches<'types>,
    ) -> &'a HashMap<PyTypeConcreteKey<'types>, u64> {
        &caches.concrete_qualified
    }

    fn cache_mut<'a>(
        caches: &'a mut DeepHashCaches<'types>,
    ) -> &'a mut HashMap<PyTypeConcreteKey<'types>, u64> {
        &mut caches.concrete_qualified
    }
}

impl<'types> DeepHashMode<'types, Parametric> for QualifiedMode {
    fn resolve_and_hash(
        key: PyTypeKey<'types, Parametric>,
        arenas: &TypeArenas<'types>,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<'types, Parametric>>,
    ) {
        arenas
            .get_as::<Self, Parametric>(key)
            .dispatch_deep_hash::<Self>(arenas, state, visited);
    }

    fn cache<'a>(
        caches: &'a DeepHashCaches<'types>,
    ) -> &'a HashMap<PyTypeParametricKey<'types>, u64> {
        &caches.parametric_qualified
    }

    fn cache_mut<'a>(
        caches: &'a mut DeepHashCaches<'types>,
    ) -> &'a mut HashMap<PyTypeParametricKey<'types>, u64> {
        &mut caches.parametric_qualified
    }
}

// --- TypeArenas methods ---

impl<'types> TypeArenas<'types> {
    fn deep_hash_of_uncached<M: DeepHashMode<'types, G>, G: ArenaSelector<'types>>(
        &self,
        id: PyTypeKey<'types, G>,
    ) -> DeepHashValue<M>
    where
        G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
        G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<'types, G>>,
    {
        let mut hasher = FxHasher::default();
        let mut visited = HashSet::default();
        deep_hash_impl::<M, G>(id, self, &mut hasher, &mut visited);
        DeepHashValue(hasher.finish(), PhantomData)
    }

    pub(crate) fn deep_hash_concrete<M: DeepHashMode<'types, Concrete>>(
        &mut self,
        key: PyTypeConcreteKey<'types>,
    ) -> DeepHashValue<M> {
        if let Some(&h) = M::cache(&self.deep_hash_caches).get(&key) {
            return DeepHashValue(h, PhantomData);
        }
        let value = self.deep_hash_of_uncached::<M, Concrete>(key);
        M::cache_mut(&mut self.deep_hash_caches).insert(key, value.raw());
        value
    }
}
