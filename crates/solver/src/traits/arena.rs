use std::hash::Hash;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplaceError {
    InvalidKey,
}

pub trait Arena<T>: Default {
    type Key: Copy + Eq + Hash + std::fmt::Debug;

    fn insert(&mut self, val: T) -> Self::Key
    where
        T: Hash + Eq;

    fn insert_placeholder(&mut self) -> Self::Key;

    fn replace(&mut self, key: Self::Key, val: T) -> Result<Option<T>, ReplaceError>
    where
        T: Hash + Eq;

    fn get(&self, key: &Self::Key) -> Option<&T>;

    fn len(&self) -> usize;
}
