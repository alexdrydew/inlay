use std::hash::Hash;
use std::marker::PhantomData;

use derive_where::derive_where;
use inlay_dedup::{DedupTable, HashWith, PartialEqWith};

use super::{
    ArenaSelector, Concrete, DeepEqMode, DeepHashMode, Parametric, PyType, PyTypeConcreteKey,
    PyTypeKey, PyTypeParametricKey, ShallowEqMode, ShallowHash, ShallowHashMode, TypeArenas,
};

struct DeepTypeKeyQuery<'types, M, G: ArenaSelector<'types>> {
    key: PyTypeKey<'types, G>,
    _mode: PhantomData<M>,
}

impl<'types, M, G: ArenaSelector<'types>> DeepTypeKeyQuery<'types, M, G> {
    fn new(key: PyTypeKey<'types, G>) -> Self {
        Self {
            key,
            _mode: PhantomData,
        }
    }
}

impl<'types, M: DeepHashMode<'types, Concrete>> HashWith<TypeArenas<'types>>
    for DeepTypeKeyQuery<'types, M, Concrete>
{
    fn hash_with<H: std::hash::Hasher>(&self, hasher: &mut H, ctx: &mut TypeArenas<'types>) {
        ctx.deep_hash_concrete::<M>(self.key).raw().hash(hasher);
    }
}

impl<'types, M: DeepEqMode<'types, Concrete>>
    PartialEqWith<TypeArenas<'types>, PyTypeConcreteKey<'types>>
    for DeepTypeKeyQuery<'types, M, Concrete>
{
    fn eq_with(&self, other: &PyTypeConcreteKey<'types>, ctx: &mut TypeArenas<'types>) -> bool {
        ctx.deep_eq_concrete::<M>(self.key, *other)
    }
}

struct ShallowTypeKeyQuery<'types, M, G: ArenaSelector<'types>> {
    key: PyTypeKey<'types, G>,
    _mode: PhantomData<M>,
}

impl<'types, M, G: ArenaSelector<'types>> ShallowTypeKeyQuery<'types, M, G> {
    fn new(key: PyTypeKey<'types, G>) -> Self {
        Self {
            key,
            _mode: PhantomData,
        }
    }
}

impl<'types, M, G> HashWith<&TypeArenas<'types>> for ShallowTypeKeyQuery<'types, M, G>
where
    M: ShallowHashMode,
    G: ArenaSelector<'types>,
    G::TypeVar: ShallowHash,
    G::ParamSpec: ShallowHash,
{
    fn hash_with<H: std::hash::Hasher>(&self, hasher: &mut H, ctx: &mut &TypeArenas<'types>) {
        (*ctx).shallow_hash_of::<M, G>(self.key).raw().hash(hasher);
    }
}

impl<'types, M: ShallowEqMode> PartialEqWith<&TypeArenas<'types>, PyTypeParametricKey<'types>>
    for ShallowTypeKeyQuery<'types, M, Concrete>
{
    fn eq_with(&self, other: &PyTypeParametricKey<'types>, ctx: &mut &TypeArenas<'types>) -> bool {
        M::cross_eq(ctx, self.key, *other)
    }
}

impl<'types, M: ShallowEqMode> PartialEqWith<&TypeArenas<'types>, PyTypeParametricKey<'types>>
    for ShallowTypeKeyQuery<'types, M, Parametric>
{
    fn eq_with(&self, other: &PyTypeParametricKey<'types>, ctx: &mut &TypeArenas<'types>) -> bool {
        M::eq::<Parametric>(ctx, self.key, *other)
    }
}

#[derive_where(Default)]
pub(crate) struct TypeKeyMap<'types, M, V> {
    table: DedupTable<PyTypeConcreteKey<'types>, V>,
    _mode: PhantomData<M>,
}

impl<'types, M, V> TypeKeyMap<'types, M, V>
where
    M: DeepHashMode<'types, Concrete> + DeepEqMode<'types, Concrete>,
{
    pub(crate) fn new() -> Self {
        Self {
            table: DedupTable::new(),
            _mode: PhantomData,
        }
    }

    pub(crate) fn get(
        &self,
        key: PyTypeConcreteKey<'types>,
        arenas: &mut TypeArenas<'types>,
    ) -> Option<&V> {
        self.table
            .find(&DeepTypeKeyQuery::<'types, M, Concrete>::new(key), arenas)
            .map(|(_, value)| value)
    }

    pub(crate) fn insert(
        &mut self,
        key: PyTypeConcreteKey<'types>,
        value: V,
        arenas: &mut TypeArenas<'types>,
    ) -> Option<V> {
        self.table.insert_or_replace(
            &DeepTypeKeyQuery::<'types, M, Concrete>::new(key),
            key,
            value,
            arenas,
        )
    }

    pub(crate) fn get_or_insert_default(
        &mut self,
        key: PyTypeConcreteKey<'types>,
        arenas: &mut TypeArenas<'types>,
    ) -> &mut V
    where
        V: Default,
    {
        self.table.get_or_insert_default(
            &DeepTypeKeyQuery::<'types, M, Concrete>::new(key),
            key,
            arenas,
        )
    }
}

// --- ShallowTypeKeyMap ---

#[derive_where(Default)]
pub(crate) struct ShallowTypeKeyMap<'types, M, V> {
    table: DedupTable<PyTypeParametricKey<'types>, V>,
    wildcard: Option<V>,
    _mode: PhantomData<M>,
}

impl<'types, M, V> ShallowTypeKeyMap<'types, M, V>
where
    M: ShallowHashMode + ShallowEqMode,
{
    pub(crate) fn get(
        &self,
        key: PyTypeConcreteKey<'types>,
        arenas: &TypeArenas<'types>,
    ) -> impl Iterator<Item = &V> {
        let mut ctx = arenas;
        let exact = self
            .table
            .find(
                &ShallowTypeKeyQuery::<'types, M, Concrete>::new(key),
                &mut ctx,
            )
            .map(|(_, value)| value);
        exact.into_iter().chain(self.wildcard.as_ref())
    }

    pub(crate) fn get_or_insert_default(
        &mut self,
        key: PyTypeParametricKey<'types>,
        arenas: &TypeArenas<'types>,
    ) -> &mut V
    where
        V: Default,
    {
        if matches!(key, PyType::TypeVar(_) | PyType::ParamSpec(_)) {
            return self.wildcard.get_or_insert_with(V::default);
        }
        let mut ctx = arenas;
        self.table.get_or_insert_default(
            &ShallowTypeKeyQuery::<'types, M, Parametric>::new(key),
            key,
            &mut ctx,
        )
    }
}
