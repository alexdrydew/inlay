use std::hash::{Hash, Hasher};

use hashbrown::HashTable;
use rustc_hash::FxHasher;

pub trait PartialEqWith<Ctx: ?Sized, Rhs: ?Sized = Self> {
    fn eq_with(&self, other: &Rhs, ctx: &mut Ctx) -> bool;
}

pub trait EqWith<Ctx: ?Sized, Rhs: ?Sized = Self>: PartialEqWith<Ctx, Rhs> {}

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

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.table.iter().map(|(_, key, value)| (key, value))
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.table.iter().map(|(_, _, value)| value)
    }

    pub fn find<Q, Ctx>(&self, query: &Q, ctx: &mut Ctx) -> Option<(&K, &V)>
    where
        Q: HashWith<Ctx> + PartialEqWith<Ctx, K> + ?Sized,
        Ctx: ?Sized,
    {
        self.find_with_hash(hash_with(query, ctx), query, ctx)
    }

    pub fn find_with_hash<Q, Ctx>(
        &self,
        hash: u64,
        query: &Q,
        ctx: &mut Ctx,
    ) -> Option<(&K, &V)>
    where
        Q: PartialEqWith<Ctx, K> + ?Sized,
        Ctx: ?Sized,
    {
        self.table
            .find(hash, |(_, key, _)| query.eq_with(key, ctx))
            .map(|(_, key, value)| (key, value))
    }

    pub fn find_mut<Q, Ctx>(&mut self, query: &Q, ctx: &mut Ctx) -> Option<(&K, &mut V)>
    where
        Q: HashWith<Ctx> + PartialEqWith<Ctx, K> + ?Sized,
        Ctx: ?Sized,
    {
        self.find_mut_with_hash(hash_with(query, ctx), query, ctx)
    }

    pub fn find_mut_with_hash<Q, Ctx>(
        &mut self,
        hash: u64,
        query: &Q,
        ctx: &mut Ctx,
    ) -> Option<(&K, &mut V)>
    where
        Q: PartialEqWith<Ctx, K> + ?Sized,
        Ctx: ?Sized,
    {
        self.table
            .find_mut(hash, |(_, key, _)| query.eq_with(key, ctx))
            .map(|(_, key, value)| (&*key, value))
    }

    pub fn get_or_insert_default<Q, Ctx>(&mut self, query: &Q, key: K, ctx: &mut Ctx) -> &mut V
    where
        Q: HashWith<Ctx> + PartialEqWith<Ctx, K> + ?Sized,
        Ctx: ?Sized,
        V: Default,
    {
        self.get_or_insert_default_with_hash(hash_with(query, ctx), query, key, ctx)
    }

    pub fn get_or_insert_default_with_hash<Q, Ctx>(
        &mut self,
        hash: u64,
        query: &Q,
        key: K,
        ctx: &mut Ctx,
    ) -> &mut V
    where
        Q: PartialEqWith<Ctx, K> + ?Sized,
        Ctx: ?Sized,
        V: Default,
    {
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
        self.insert_or_replace_with_hash(hash_with(query, ctx), query, key, value, ctx)
    }

    pub fn insert_or_replace_with_hash<Q, Ctx>(
        &mut self,
        hash: u64,
        query: &Q,
        key: K,
        value: V,
        ctx: &mut Ctx,
    ) -> Option<V>
    where
        Q: PartialEqWith<Ctx, K> + ?Sized,
        Ctx: ?Sized,
    {
        if let Some((_, existing)) = self.find_mut_with_hash(hash, query, ctx) {
            return Some(std::mem::replace(existing, value));
        }

        self.insert_unique_hashed(hash, key, value);
        None
    }

    pub fn insert_unique<Q, Ctx>(&mut self, key: K, value: V, ctx: &mut Ctx)
    where
        K: HashWith<Ctx>,
        Ctx: ?Sized,
    {
        self.insert_unique_hashed(hash_with(&key, ctx), key, value);
    }

    pub fn insert_unique_hashed(&mut self, hash: u64, key: K, value: V) {
        self.table
            .insert_unique(hash, (hash, key, value), |(stored_hash, _, _)| *stored_hash);
    }

    pub fn remove<Q, Ctx>(&mut self, query: &Q, ctx: &mut Ctx) -> Option<(K, V)>
    where
        Q: HashWith<Ctx> + PartialEqWith<Ctx, K> + ?Sized,
        Ctx: ?Sized,
    {
        self.remove_with_hash(hash_with(query, ctx), query, ctx)
    }

    pub fn remove_with_hash<Q, Ctx>(
        &mut self,
        hash: u64,
        query: &Q,
        ctx: &mut Ctx,
    ) -> Option<(K, V)>
    where
        Q: PartialEqWith<Ctx, K> + ?Sized,
        Ctx: ?Sized,
    {
        let ((_, key, value), _) = self
            .table
            .find_entry(hash, |(_, stored_key, _)| query.eq_with(stored_key, ctx))
            .ok()?
            .remove();
        Some((key, value))
    }
}

pub struct ValueEq;

impl<T: PartialEq> PartialEqWith<ValueEq> for T {
    fn eq_with(&self, other: &Self, _ctx: &mut ValueEq) -> bool {
        self == other
    }
}

impl<T: Eq> EqWith<ValueEq> for T {}

impl<T: Hash> HashWith<ValueEq> for T {
    fn hash_with<H: Hasher>(&self, hasher: &mut H, _ctx: &mut ValueEq) {
        self.hash(hasher);
    }
}
