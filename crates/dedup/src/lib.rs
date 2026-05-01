use std::hash::Hasher;

use hashbrown::HashTable;
use rustc_hash::FxHasher;

pub trait PartialEqWith<Ctx: ?Sized, Rhs: ?Sized = Self> {
    fn eq_with(&self, other: &Rhs, ctx: &mut Ctx) -> bool;
}

pub trait HashWith<Ctx: ?Sized> {
    fn hash_with<H: Hasher>(&self, hasher: &mut H, ctx: &mut Ctx);
}

pub fn hash_with<T: HashWith<Ctx> + ?Sized, Ctx: ?Sized>(value: &T, ctx: &mut Ctx) -> u64 {
    let mut hasher = FxHasher::default();
    value.hash_with(&mut hasher, ctx);
    hasher.finish()
}

pub struct DedupTable<K, V> {
    table: HashTable<(u64, K, V)>,
}

impl<K, V> Default for DedupTable<K, V> {
    fn default() -> Self {
        Self {
            table: HashTable::new(),
        }
    }
}

impl<K, V> DedupTable<K, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn find<Q, Ctx>(&self, query: &Q, ctx: &mut Ctx) -> Option<(&K, &V)>
    where
        Q: HashWith<Ctx> + PartialEqWith<Ctx, K> + ?Sized,
        Ctx: ?Sized,
    {
        let hash = hash_with(query, ctx);
        self.table
            .find(hash, |(_, key, _)| query.eq_with(key, ctx))
            .map(|(_, key, value)| (key, value))
    }

    pub fn get_or_insert_default<Q, Ctx>(&mut self, query: &Q, key: K, ctx: &mut Ctx) -> &mut V
    where
        Q: HashWith<Ctx> + PartialEqWith<Ctx, K> + ?Sized,
        Ctx: ?Sized,
        V: Default,
    {
        let hash = hash_with(query, ctx);
        match self.table.entry(
            hash,
            |(_, stored_key, _)| query.eq_with(stored_key, ctx),
            |(stored_hash, _, _)| *stored_hash,
        ) {
            hashbrown::hash_table::Entry::Occupied(entry) => &mut entry.into_mut().2,
            hashbrown::hash_table::Entry::Vacant(entry) => {
                &mut entry.insert((hash, key, V::default())).into_mut().2
            }
        }
    }

    pub fn insert_or_replace<Q, Ctx>(
        &mut self,
        query: &Q,
        key: K,
        value: V,
        ctx: &mut Ctx,
    ) -> Option<V>
    where
        Q: HashWith<Ctx> + PartialEqWith<Ctx, K> + ?Sized,
        Ctx: ?Sized,
    {
        let hash = hash_with(query, ctx);
        if let Some((_, _, existing)) = self
            .table
            .find_mut(hash, |(_, stored_key, _)| query.eq_with(stored_key, ctx))
        {
            return Some(std::mem::replace(existing, value));
        }

        self.table
            .insert_unique(hash, (hash, key, value), |(stored_hash, _, _)| *stored_hash);
        None
    }
}
