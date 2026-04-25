use std::hash::Hash;
use std::marker::PhantomData;

use derive_where::derive_where;
use inlay_dedup::{DedupTable, HashWith, PartialEqWith};

use super::{
    ArenaFamily, ArenaSelector, Concrete, DeepEqMode, DeepHashMode, Parametric, PyType,
    PyTypeConcreteKey, PyTypeKey, PyTypeParametricKey, ShallowEqMode, ShallowHash, ShallowHashMode,
    TypeArenas,
};

struct DeepTypeKeyQuery<S: ArenaFamily, M, G: ArenaSelector> {
    key: PyTypeKey<S, G>,
    _mode: PhantomData<M>,
}

impl<S: ArenaFamily, M, G: ArenaSelector> DeepTypeKeyQuery<S, M, G> {
    fn new(key: PyTypeKey<S, G>) -> Self {
        Self {
            key,
            _mode: PhantomData,
        }
    }
}

impl<S: ArenaFamily, M: DeepHashMode<S, Concrete>> HashWith<TypeArenas<S>>
    for DeepTypeKeyQuery<S, M, Concrete>
{
    fn hash_with<H: std::hash::Hasher>(&self, hasher: &mut H, ctx: &mut TypeArenas<S>) {
        ctx.deep_hash_concrete::<M>(self.key).raw().hash(hasher);
    }
}

impl<S: ArenaFamily, M: DeepEqMode<S, Concrete>> PartialEqWith<TypeArenas<S>, PyTypeConcreteKey<S>>
    for DeepTypeKeyQuery<S, M, Concrete>
{
    fn eq_with(&self, other: &PyTypeConcreteKey<S>, ctx: &mut TypeArenas<S>) -> bool {
        ctx.deep_eq_concrete::<M>(self.key, *other)
    }
}

struct ShallowTypeKeyQuery<S: ArenaFamily, M, G: ArenaSelector> {
    key: PyTypeKey<S, G>,
    _mode: PhantomData<M>,
}

impl<S: ArenaFamily, M, G: ArenaSelector> ShallowTypeKeyQuery<S, M, G> {
    fn new(key: PyTypeKey<S, G>) -> Self {
        Self {
            key,
            _mode: PhantomData,
        }
    }
}

impl<S, M, G> HashWith<&TypeArenas<S>> for ShallowTypeKeyQuery<S, M, G>
where
    S: ArenaFamily,
    M: ShallowHashMode<S>,
    G: ArenaSelector,
    G::TypeVar: ShallowHash,
    G::ParamSpec: ShallowHash,
{
    fn hash_with<H: std::hash::Hasher>(&self, hasher: &mut H, ctx: &mut &TypeArenas<S>) {
        (*ctx).shallow_hash_of::<M, G>(self.key).raw().hash(hasher);
    }
}

impl<S: ArenaFamily, M: ShallowEqMode<S>> PartialEqWith<&TypeArenas<S>, PyTypeParametricKey<S>>
    for ShallowTypeKeyQuery<S, M, Concrete>
{
    fn eq_with(&self, other: &PyTypeParametricKey<S>, ctx: &mut &TypeArenas<S>) -> bool {
        M::cross_eq(*ctx, self.key, *other)
    }
}

impl<S: ArenaFamily, M: ShallowEqMode<S>> PartialEqWith<&TypeArenas<S>, PyTypeParametricKey<S>>
    for ShallowTypeKeyQuery<S, M, Parametric>
{
    fn eq_with(&self, other: &PyTypeParametricKey<S>, ctx: &mut &TypeArenas<S>) -> bool {
        M::eq::<Parametric>(*ctx, self.key, *other)
    }
}

#[derive_where(Default)]
pub(crate) struct TypeKeyMap<S: ArenaFamily, M, V> {
    table: DedupTable<PyTypeConcreteKey<S>, V>,
    _mode: PhantomData<M>,
}

impl<S: ArenaFamily, M, V> TypeKeyMap<S, M, V>
where
    M: DeepHashMode<S, Concrete> + DeepEqMode<S, Concrete>,
{
    pub(crate) fn new() -> Self {
        Self {
            table: DedupTable::new(),
            _mode: PhantomData,
        }
    }

    pub(crate) fn get(&self, key: PyTypeConcreteKey<S>, arenas: &mut TypeArenas<S>) -> Option<&V> {
        self.table
            .find(&DeepTypeKeyQuery::<S, M, Concrete>::new(key), arenas)
            .map(|(_, value)| value)
    }

    pub(crate) fn get_mut(
        &mut self,
        key: PyTypeConcreteKey<S>,
        arenas: &mut TypeArenas<S>,
    ) -> Option<&mut V> {
        self.table
            .find_mut(&DeepTypeKeyQuery::<S, M, Concrete>::new(key), arenas)
            .map(|(_, value)| value)
    }

    pub(crate) fn insert(
        &mut self,
        key: PyTypeConcreteKey<S>,
        value: V,
        arenas: &mut TypeArenas<S>,
    ) -> Option<V> {
        self.table.insert_or_replace(
            &DeepTypeKeyQuery::<S, M, Concrete>::new(key),
            key,
            value,
            arenas,
        )
    }

    pub(crate) fn get_or_insert_default(
        &mut self,
        key: PyTypeConcreteKey<S>,
        arenas: &mut TypeArenas<S>,
    ) -> &mut V
    where
        V: Default,
    {
        self.table
            .get_or_insert_default(&DeepTypeKeyQuery::<S, M, Concrete>::new(key), key, arenas)
    }

    pub(crate) fn len(&self) -> usize {
        self.table.len()
    }

    pub(crate) fn value_len_sum<T>(&self) -> usize
    where
        V: AsRef<[T]>,
    {
        self.table.values().map(|value| value.as_ref().len()).sum()
    }
}

// --- ShallowTypeKeyMap ---

#[derive_where(Default)]
pub(crate) struct ShallowTypeKeyMap<S: ArenaFamily, M, V> {
    table: DedupTable<PyTypeParametricKey<S>, V>,
    wildcard: Option<V>,
    _mode: PhantomData<M>,
}

impl<S: ArenaFamily, M, V> ShallowTypeKeyMap<S, M, V>
where
    M: ShallowHashMode<S> + ShallowEqMode<S>,
{
    pub(crate) fn new() -> Self {
        Self {
            table: DedupTable::new(),
            wildcard: None,
            _mode: PhantomData,
        }
    }

    pub(crate) fn get(
        &self,
        key: PyTypeConcreteKey<S>,
        arenas: &TypeArenas<S>,
    ) -> impl Iterator<Item = &V> {
        let mut ctx = arenas;
        let exact = self
            .table
            .find(&ShallowTypeKeyQuery::<S, M, Concrete>::new(key), &mut ctx)
            .map(|(_, value)| value);
        exact.into_iter().chain(self.wildcard.as_ref())
    }

    pub(crate) fn get_or_insert_default(
        &mut self,
        key: PyTypeParametricKey<S>,
        arenas: &TypeArenas<S>,
    ) -> &mut V
    where
        V: Default,
    {
        if matches!(key, PyType::TypeVar(_) | PyType::ParamSpec(_)) {
            return self.wildcard.get_or_insert_with(V::default);
        }
        let mut ctx = arenas;
        self.table.get_or_insert_default(
            &ShallowTypeKeyQuery::<S, M, Parametric>::new(key),
            key,
            &mut ctx,
        )
    }
}
