use std::hash::Hash;
use std::marker::PhantomData;

use derive_where::derive_where;
use inlay_dedup::{DedupTable, HashWith, PartialEqWith};

use super::{
    ArenaSelector, Concrete, DeepEqMode, DeepHashMode, Parametric, PyType, PyTypeConcreteKey,
    PyTypeKey, PyTypeParametricKey, ShallowEqMode, ShallowHash, ShallowHashMode, TypeArenas,
};

struct DeepTypeKeyQuery<'ty, M, G: ArenaSelector<'ty>> {
    key: PyTypeKey<'ty, G>,
    _mode: PhantomData<M>,
}

impl<'ty, M, G: ArenaSelector<'ty>> DeepTypeKeyQuery<'ty, M, G> {
    fn new(key: PyTypeKey<'ty, G>) -> Self {
        Self {
            key,
            _mode: PhantomData,
        }
    }
}

impl<'ty, M: DeepHashMode<'ty, Concrete>> HashWith<TypeArenas<'ty>>
    for DeepTypeKeyQuery<'ty, M, Concrete>
{
    fn hash_with<H: std::hash::Hasher>(&self, hasher: &mut H, ctx: &mut TypeArenas<'ty>) {
        ctx.deep_hash_concrete::<M>(self.key).raw().hash(hasher);
    }
}

impl<'ty, M: DeepEqMode<'ty, Concrete>> PartialEqWith<TypeArenas<'ty>, PyTypeConcreteKey<'ty>>
    for DeepTypeKeyQuery<'ty, M, Concrete>
{
    fn eq_with(&self, other: &PyTypeConcreteKey<'ty>, ctx: &mut TypeArenas<'ty>) -> bool {
        ctx.deep_eq_concrete::<M>(self.key, *other)
    }
}

struct ShallowTypeKeyQuery<'ty, M, G: ArenaSelector<'ty>> {
    key: PyTypeKey<'ty, G>,
    _mode: PhantomData<M>,
}

impl<'ty, M, G: ArenaSelector<'ty>> ShallowTypeKeyQuery<'ty, M, G> {
    fn new(key: PyTypeKey<'ty, G>) -> Self {
        Self {
            key,
            _mode: PhantomData,
        }
    }
}

impl<'ty, M, G> HashWith<&TypeArenas<'ty>> for ShallowTypeKeyQuery<'ty, M, G>
where
    M: ShallowHashMode,
    G: ArenaSelector<'ty>,
    G::TypeVar: ShallowHash,
    G::ParamSpec: ShallowHash,
{
    fn hash_with<H: std::hash::Hasher>(&self, hasher: &mut H, ctx: &mut &TypeArenas<'ty>) {
        (*ctx).shallow_hash_of::<M, G>(self.key).raw().hash(hasher);
    }
}

impl<'ty, M: ShallowEqMode> PartialEqWith<&TypeArenas<'ty>, PyTypeParametricKey<'ty>>
    for ShallowTypeKeyQuery<'ty, M, Concrete>
{
    fn eq_with(&self, other: &PyTypeParametricKey<'ty>, ctx: &mut &TypeArenas<'ty>) -> bool {
        M::cross_eq(ctx, self.key, *other)
    }
}

impl<'ty, M: ShallowEqMode> PartialEqWith<&TypeArenas<'ty>, PyTypeParametricKey<'ty>>
    for ShallowTypeKeyQuery<'ty, M, Parametric>
{
    fn eq_with(&self, other: &PyTypeParametricKey<'ty>, ctx: &mut &TypeArenas<'ty>) -> bool {
        M::eq::<Parametric>(ctx, self.key, *other)
    }
}

#[derive_where(Default)]
pub(crate) struct TypeKeyMap<'ty, M, V> {
    table: DedupTable<PyTypeConcreteKey<'ty>, V>,
    _mode: PhantomData<M>,
}

impl<'ty, M, V> TypeKeyMap<'ty, M, V>
where
    M: DeepHashMode<'ty, Concrete> + DeepEqMode<'ty, Concrete>,
{
    pub(crate) fn new() -> Self {
        Self {
            table: DedupTable::new(),
            _mode: PhantomData,
        }
    }

    pub(crate) fn get(
        &self,
        key: PyTypeConcreteKey<'ty>,
        arenas: &mut TypeArenas<'ty>,
    ) -> Option<&V> {
        self.table
            .find(&DeepTypeKeyQuery::<'ty, M, Concrete>::new(key), arenas)
            .map(|(_, value)| value)
    }

    pub(crate) fn insert(
        &mut self,
        key: PyTypeConcreteKey<'ty>,
        value: V,
        arenas: &mut TypeArenas<'ty>,
    ) -> Option<V> {
        self.table.insert_or_replace(
            &DeepTypeKeyQuery::<'ty, M, Concrete>::new(key),
            key,
            value,
            arenas,
        )
    }

    pub(crate) fn get_or_insert_default(
        &mut self,
        key: PyTypeConcreteKey<'ty>,
        arenas: &mut TypeArenas<'ty>,
    ) -> &mut V
    where
        V: Default,
    {
        self.table.get_or_insert_default(
            &DeepTypeKeyQuery::<'ty, M, Concrete>::new(key),
            key,
            arenas,
        )
    }
}

// --- ShallowTypeKeyMap ---

#[derive_where(Default)]
pub(crate) struct ShallowTypeKeyMap<'ty, M, V> {
    table: DedupTable<PyTypeParametricKey<'ty>, V>,
    wildcard: Option<V>,
    _mode: PhantomData<M>,
}

impl<'ty, M, V> ShallowTypeKeyMap<'ty, M, V>
where
    M: ShallowHashMode + ShallowEqMode,
{
    pub(crate) fn get(
        &self,
        key: PyTypeConcreteKey<'ty>,
        arenas: &TypeArenas<'ty>,
    ) -> impl Iterator<Item = &V> {
        let mut ctx = arenas;
        let exact = self
            .table
            .find(&ShallowTypeKeyQuery::<'ty, M, Concrete>::new(key), &mut ctx)
            .map(|(_, value)| value);
        exact.into_iter().chain(self.wildcard.as_ref())
    }

    pub(crate) fn get_or_insert_default(
        &mut self,
        key: PyTypeParametricKey<'ty>,
        arenas: &TypeArenas<'ty>,
    ) -> &mut V
    where
        V: Default,
    {
        if matches!(key, PyType::TypeVar(_) | PyType::ParamSpec(_)) {
            return self.wildcard.get_or_insert_with(V::default);
        }
        let mut ctx = arenas;
        self.table.get_or_insert_default(
            &ShallowTypeKeyQuery::<'ty, M, Parametric>::new(key),
            key,
            &mut ctx,
        )
    }
}
