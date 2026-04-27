use std::hash::Hash;
use std::marker::PhantomData;

use derive_where::derive_where;
use inlay_dedup::{DedupTable, HashWith, PartialEqWith};

use super::{
    ArenaSelector, Concrete, DeepEqMode, DeepHashMode, Parametric, PyType, PyTypeConcreteKey,
    PyTypeKey, PyTypeParametricKey, ShallowEqMode, ShallowHash, ShallowHashMode, TypeArenas,
};

struct DeepTypeKeyQuery<M, G: ArenaSelector> {
    key: PyTypeKey<G>,
    _mode: PhantomData<M>,
}

impl<M, G: ArenaSelector> DeepTypeKeyQuery<M, G> {
    fn new(key: PyTypeKey<G>) -> Self {
        Self {
            key,
            _mode: PhantomData,
        }
    }
}

impl<M: DeepHashMode<Concrete>> HashWith<TypeArenas> for DeepTypeKeyQuery<M, Concrete> {
    fn hash_with<H: std::hash::Hasher>(&self, hasher: &mut H, ctx: &mut TypeArenas) {
        ctx.deep_hash_concrete::<M>(self.key).raw().hash(hasher);
    }
}

impl<M: DeepEqMode<Concrete>> PartialEqWith<TypeArenas, PyTypeConcreteKey>
    for DeepTypeKeyQuery<M, Concrete>
{
    fn eq_with(&self, other: &PyTypeConcreteKey, ctx: &mut TypeArenas) -> bool {
        ctx.deep_eq_concrete::<M>(self.key, *other)
    }
}

struct ShallowTypeKeyQuery<M, G: ArenaSelector> {
    key: PyTypeKey<G>,
    _mode: PhantomData<M>,
}

impl<M, G: ArenaSelector> ShallowTypeKeyQuery<M, G> {
    fn new(key: PyTypeKey<G>) -> Self {
        Self {
            key,
            _mode: PhantomData,
        }
    }
}

impl<M, G> HashWith<&TypeArenas> for ShallowTypeKeyQuery<M, G>
where
    M: ShallowHashMode,
    G: ArenaSelector,
    G::TypeVar: ShallowHash,
    G::ParamSpec: ShallowHash,
{
    fn hash_with<H: std::hash::Hasher>(&self, hasher: &mut H, ctx: &mut &TypeArenas) {
        (*ctx).shallow_hash_of::<M, G>(self.key).raw().hash(hasher);
    }
}

impl<M: ShallowEqMode> PartialEqWith<&TypeArenas, PyTypeParametricKey>
    for ShallowTypeKeyQuery<M, Concrete>
{
    fn eq_with(&self, other: &PyTypeParametricKey, ctx: &mut &TypeArenas) -> bool {
        M::cross_eq(*ctx, self.key, *other)
    }
}

impl<M: ShallowEqMode> PartialEqWith<&TypeArenas, PyTypeParametricKey>
    for ShallowTypeKeyQuery<M, Parametric>
{
    fn eq_with(&self, other: &PyTypeParametricKey, ctx: &mut &TypeArenas) -> bool {
        M::eq::<Parametric>(*ctx, self.key, *other)
    }
}

#[derive_where(Default)]
pub(crate) struct TypeKeyMap<M, V> {
    table: DedupTable<PyTypeConcreteKey, V>,
    _mode: PhantomData<M>,
}

impl<M, V> TypeKeyMap<M, V>
where
    M: DeepHashMode<Concrete> + DeepEqMode<Concrete>,
{
    pub(crate) fn new() -> Self {
        Self {
            table: DedupTable::new(),
            _mode: PhantomData,
        }
    }

    pub(crate) fn get(&self, key: PyTypeConcreteKey, arenas: &mut TypeArenas) -> Option<&V> {
        self.table
            .find(&DeepTypeKeyQuery::<M, Concrete>::new(key), arenas)
            .map(|(_, value)| value)
    }

    pub(crate) fn insert(
        &mut self,
        key: PyTypeConcreteKey,
        value: V,
        arenas: &mut TypeArenas,
    ) -> Option<V> {
        self.table.insert_or_replace(
            &DeepTypeKeyQuery::<M, Concrete>::new(key),
            key,
            value,
            arenas,
        )
    }

    pub(crate) fn get_or_insert_default(
        &mut self,
        key: PyTypeConcreteKey,
        arenas: &mut TypeArenas,
    ) -> &mut V
    where
        V: Default,
    {
        self.table
            .get_or_insert_default(&DeepTypeKeyQuery::<M, Concrete>::new(key), key, arenas)
    }
}

// --- ShallowTypeKeyMap ---

#[derive_where(Default)]
pub(crate) struct ShallowTypeKeyMap<M, V> {
    table: DedupTable<PyTypeParametricKey, V>,
    wildcard: Option<V>,
    _mode: PhantomData<M>,
}

impl<M, V> ShallowTypeKeyMap<M, V>
where
    M: ShallowHashMode + ShallowEqMode,
{
    pub(crate) fn get(
        &self,
        key: PyTypeConcreteKey,
        arenas: &TypeArenas,
    ) -> impl Iterator<Item = &V> {
        let mut ctx = arenas;
        let exact = self
            .table
            .find(&ShallowTypeKeyQuery::<M, Concrete>::new(key), &mut ctx)
            .map(|(_, value)| value);
        exact.into_iter().chain(self.wildcard.as_ref())
    }

    pub(crate) fn get_or_insert_default(
        &mut self,
        key: PyTypeParametricKey,
        arenas: &TypeArenas,
    ) -> &mut V
    where
        V: Default,
    {
        if matches!(key, PyType::TypeVar(_) | PyType::ParamSpec(_)) {
            return self.wildcard.get_or_insert_with(V::default);
        }
        let mut ctx = arenas;
        self.table.get_or_insert_default(
            &ShallowTypeKeyQuery::<M, Parametric>::new(key),
            key,
            &mut ctx,
        )
    }
}
