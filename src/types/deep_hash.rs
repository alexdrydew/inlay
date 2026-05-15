use std::{
    hash::{Hash, Hasher},
    marker::PhantomData,
};

use derive_where::derive_where;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};

use super::{
    ArenaSelector, CallableType, ClassType, Concrete, Keyed, LazyRefType, Parametric, PlainType,
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

#[derive(Default)]
pub(crate) struct DeepHashCaches<'ty> {
    concrete_unqualified: HashMap<PyTypeConcreteKey<'ty>, u64>,
    concrete_qualified: HashMap<PyTypeConcreteKey<'ty>, u64>,
    parametric_unqualified: HashMap<PyTypeParametricKey<'ty>, u64>,
    parametric_qualified: HashMap<PyTypeParametricKey<'ty>, u64>,
}

impl<'ty> DeepHashCaches<'ty> {
    pub(crate) fn retain_concrete(
        &mut self,
        mut retain: impl FnMut(PyTypeConcreteKey<'ty>) -> bool,
    ) {
        self.concrete_unqualified.retain(|key, _| retain(*key));
        self.concrete_qualified.retain(|key, _| retain(*key));
    }
}

// --- DeepHashMode trait ---

pub(crate) trait DeepHashMode<'ty, G: ArenaSelector<'ty>> {
    fn resolve_and_hash(
        key: PyTypeKey<'ty, G>,
        arenas: &TypeArenas<'ty>,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<'ty, G>>,
    ) where
        G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>;

    fn cache<'a>(caches: &'a DeepHashCaches<'ty>) -> &'a HashMap<PyTypeKey<'ty, G>, u64>;
    fn cache_mut<'a>(
        caches: &'a mut DeepHashCaches<'ty>,
    ) -> &'a mut HashMap<PyTypeKey<'ty, G>, u64>;
}

// --- Recursion skeleton ---

fn deep_hash_impl<'ty, M: DeepHashMode<'ty, G>, G: ArenaSelector<'ty>>(
    key: PyTypeKey<'ty, G>,
    arenas: &TypeArenas<'ty>,
    state: &mut impl Hasher,
    visited: &mut HashSet<PyTypeKey<'ty, G>>,
) where
    G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
    G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
{
    if !visited.insert(key) {
        return;
    }
    std::mem::discriminant(&key).hash(state);
    M::resolve_and_hash(key, arenas, state, visited);
    visited.remove(&key);
}

// --- One-level helper ---

fn hash_and_recurse<'ty, V, M: DeepHashMode<'ty, G>, G: ArenaSelector<'ty>>(
    v: V,
    arenas: &TypeArenas<'ty>,
    state: &mut impl Hasher,
    visited: &mut HashSet<PyTypeKey<'ty, G>>,
) where
    V: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
    G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
    G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
{
    v.shallow_hash(state);
    for &dep in v.children() {
        deep_hash_impl::<M, G>(dep, arenas, state, visited);
    }
}

// --- Generic dispatch (single 9-arm match, generic over O and G) ---

impl<'ty, O: Wrapper, G: ArenaSelector<'ty>> PyType<O, Qual<Keyed<'ty>>, G> {
    fn dispatch_deep_hash<M: DeepHashMode<'ty, G>>(
        self,
        arenas: &TypeArenas<'ty>,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<'ty, G>>,
    ) where
        O::Wrap<SentinelType>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<G::TypeVar>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<G::ParamSpec>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<PlainType<Qual<Keyed<'ty>>, G>>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<ClassType<Qual<Keyed<'ty>>, G>>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<ProtocolType<Qual<Keyed<'ty>>, G>>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<TypedDictType<Qual<Keyed<'ty>>, G>>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<UnionType<Qual<Keyed<'ty>>, G>>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<CallableType<Qual<Keyed<'ty>>, G>>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<LazyRefType<Qual<Keyed<'ty>>, G>>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
    {
        match self {
            PyType::Sentinel(v) => hash_and_recurse::<_, M, G>(v, arenas, state, visited),
            PyType::ParamSpec(v) => hash_and_recurse::<_, M, G>(v, arenas, state, visited),
            PyType::Plain(v) => hash_and_recurse::<_, M, G>(v, arenas, state, visited),
            PyType::Class(v) => hash_and_recurse::<_, M, G>(v, arenas, state, visited),
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

impl<'ty> DeepHashMode<'ty, Concrete> for UnqualifiedMode {
    fn resolve_and_hash(
        key: PyTypeKey<'ty, Concrete>,
        arenas: &TypeArenas<'ty>,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<'ty, Concrete>>,
    ) {
        arenas
            .get_as::<Self, Concrete>(key)
            .dispatch_deep_hash::<Self>(arenas, state, visited);
    }

    fn cache<'a>(caches: &'a DeepHashCaches<'ty>) -> &'a HashMap<PyTypeConcreteKey<'ty>, u64> {
        &caches.concrete_unqualified
    }

    fn cache_mut<'a>(
        caches: &'a mut DeepHashCaches<'ty>,
    ) -> &'a mut HashMap<PyTypeConcreteKey<'ty>, u64> {
        &mut caches.concrete_unqualified
    }
}

impl<'ty> DeepHashMode<'ty, Parametric> for UnqualifiedMode {
    fn resolve_and_hash(
        key: PyTypeKey<'ty, Parametric>,
        arenas: &TypeArenas<'ty>,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<'ty, Parametric>>,
    ) {
        arenas
            .get_as::<Self, Parametric>(key)
            .dispatch_deep_hash::<Self>(arenas, state, visited);
    }

    fn cache<'a>(caches: &'a DeepHashCaches<'ty>) -> &'a HashMap<PyTypeParametricKey<'ty>, u64> {
        &caches.parametric_unqualified
    }

    fn cache_mut<'a>(
        caches: &'a mut DeepHashCaches<'ty>,
    ) -> &'a mut HashMap<PyTypeParametricKey<'ty>, u64> {
        &mut caches.parametric_unqualified
    }
}

impl<'ty> DeepHashMode<'ty, Concrete> for QualifiedMode {
    fn resolve_and_hash(
        key: PyTypeKey<'ty, Concrete>,
        arenas: &TypeArenas<'ty>,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<'ty, Concrete>>,
    ) {
        arenas
            .get_as::<Self, Concrete>(key)
            .dispatch_deep_hash::<Self>(arenas, state, visited);
    }

    fn cache<'a>(caches: &'a DeepHashCaches<'ty>) -> &'a HashMap<PyTypeConcreteKey<'ty>, u64> {
        &caches.concrete_qualified
    }

    fn cache_mut<'a>(
        caches: &'a mut DeepHashCaches<'ty>,
    ) -> &'a mut HashMap<PyTypeConcreteKey<'ty>, u64> {
        &mut caches.concrete_qualified
    }
}

impl<'ty> DeepHashMode<'ty, Parametric> for QualifiedMode {
    fn resolve_and_hash(
        key: PyTypeKey<'ty, Parametric>,
        arenas: &TypeArenas<'ty>,
        state: &mut impl Hasher,
        visited: &mut HashSet<PyTypeKey<'ty, Parametric>>,
    ) {
        arenas
            .get_as::<Self, Parametric>(key)
            .dispatch_deep_hash::<Self>(arenas, state, visited);
    }

    fn cache<'a>(caches: &'a DeepHashCaches<'ty>) -> &'a HashMap<PyTypeParametricKey<'ty>, u64> {
        &caches.parametric_qualified
    }

    fn cache_mut<'a>(
        caches: &'a mut DeepHashCaches<'ty>,
    ) -> &'a mut HashMap<PyTypeParametricKey<'ty>, u64> {
        &mut caches.parametric_qualified
    }
}

// --- TypeArenas methods ---

impl<'ty> TypeArenas<'ty> {
    fn deep_hash_of_uncached<M: DeepHashMode<'ty, G>, G: ArenaSelector<'ty>>(
        &self,
        id: PyTypeKey<'ty, G>,
    ) -> DeepHashValue<M>
    where
        G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
    {
        let mut hasher = FxHasher::default();
        let mut visited = HashSet::default();
        deep_hash_impl::<M, G>(id, self, &mut hasher, &mut visited);
        DeepHashValue(hasher.finish(), PhantomData)
    }

    pub(crate) fn deep_hash_concrete<M: DeepHashMode<'ty, Concrete>>(
        &mut self,
        key: PyTypeConcreteKey<'ty>,
    ) -> DeepHashValue<M> {
        if let Some(&h) = M::cache(&self.deep_hash_caches).get(&key) {
            return DeepHashValue(h, PhantomData);
        }
        let value = self.deep_hash_of_uncached::<M, Concrete>(key);
        M::cache_mut(&mut self.deep_hash_caches).insert(key, value.raw());
        value
    }
}
