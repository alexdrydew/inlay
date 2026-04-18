use std::marker::PhantomData;

use derive_where::derive_where;
use hashbrown::HashTable;

use super::{
    ArenaFamily, Concrete, DeepEqMode, DeepHashMode, Parametric, PyType, PyTypeConcreteKey,
    PyTypeParametricKey, ShallowEqMode, ShallowHashMode, TypeArenas,
};

#[derive_where(Default)]
pub(crate) struct TypeKeyMap<S: ArenaFamily, M, V> {
    table: HashTable<(u64, PyTypeConcreteKey<S>, V)>,
    _mode: PhantomData<M>,
}

impl<S: ArenaFamily, M, V> TypeKeyMap<S, M, V>
where
    M: DeepHashMode<S, Concrete> + DeepEqMode<S, Concrete>,
{
    pub(crate) fn new() -> Self {
        Self {
            table: HashTable::new(),
            _mode: PhantomData,
        }
    }

    pub(crate) fn get(&self, key: PyTypeConcreteKey<S>, arenas: &mut TypeArenas<S>) -> Option<&V> {
        let hash = arenas.deep_hash_concrete::<M>(key).raw();
        self.table
            .find(hash, |(_, k, _)| arenas.deep_eq_concrete::<M>(key, *k))
            .map(|(_, _, v)| v)
    }

    pub(crate) fn get_mut(
        &mut self,
        key: PyTypeConcreteKey<S>,
        arenas: &mut TypeArenas<S>,
    ) -> Option<&mut V> {
        let hash = arenas.deep_hash_concrete::<M>(key).raw();
        self.table
            .find_mut(hash, |(_, k, _)| arenas.deep_eq_concrete::<M>(key, *k))
            .map(|(_, _, v)| v)
    }

    pub(crate) fn insert(
        &mut self,
        key: PyTypeConcreteKey<S>,
        value: V,
        arenas: &mut TypeArenas<S>,
    ) -> Option<V> {
        let hash = arenas.deep_hash_concrete::<M>(key).raw();
        match self
            .table
            .find_mut(hash, |(_, k, _)| arenas.deep_eq_concrete::<M>(key, *k))
        {
            Some((_, _, existing)) => Some(std::mem::replace(existing, value)),
            None => {
                self.table
                    .insert_unique(hash, (hash, key, value), |(h, _, _)| *h);
                None
            }
        }
    }

    pub(crate) fn get_or_insert_default(
        &mut self,
        key: PyTypeConcreteKey<S>,
        arenas: &mut TypeArenas<S>,
    ) -> &mut V
    where
        V: Default,
    {
        let hash = arenas.deep_hash_concrete::<M>(key).raw();
        let entry = self.table.entry(
            hash,
            |(_, k, _)| arenas.deep_eq_concrete::<M>(key, *k),
            |(h, _, _)| *h,
        );
        match entry {
            hashbrown::hash_table::Entry::Occupied(o) => &mut o.into_mut().2,
            hashbrown::hash_table::Entry::Vacant(v) => {
                &mut v.insert((hash, key, V::default())).into_mut().2
            }
        }
    }
}

// --- ShallowTypeKeyMap ---

#[derive_where(Default)]
pub(crate) struct ShallowTypeKeyMap<S: ArenaFamily, M, V> {
    table: HashTable<(u64, PyTypeParametricKey<S>, V)>,
    wildcard: Option<V>,
    _mode: PhantomData<M>,
}

impl<S: ArenaFamily, M, V> ShallowTypeKeyMap<S, M, V>
where
    M: ShallowHashMode<S> + ShallowEqMode<S>,
{
    pub(crate) fn new() -> Self {
        Self {
            table: HashTable::new(),
            wildcard: None,
            _mode: PhantomData,
        }
    }

    pub(crate) fn get(
        &self,
        key: PyTypeConcreteKey<S>,
        arenas: &TypeArenas<S>,
    ) -> impl Iterator<Item = &V> {
        let hash = arenas.shallow_hash_of::<M, Concrete>(key).raw();
        let exact = self
            .table
            .find(hash, |(h, pk, _)| {
                *h == hash && M::cross_eq(arenas, key, *pk)
            })
            .map(|(_, _, v)| v);
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
        let hash = arenas.shallow_hash_of::<M, Parametric>(key).raw();
        let entry = self.table.entry(
            hash,
            |(h, pk, _)| *h == hash && M::eq::<Parametric>(arenas, key, *pk),
            |(h, _, _)| *h,
        );
        match entry {
            hashbrown::hash_table::Entry::Occupied(o) => &mut o.into_mut().2,
            hashbrown::hash_table::Entry::Vacant(v) => {
                &mut v.insert((hash, key, V::default())).into_mut().2
            }
        }
    }
}
