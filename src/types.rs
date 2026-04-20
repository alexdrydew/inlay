mod deep_eq;
mod deep_hash;
mod definitions;
mod monomorphize;
mod shallow_eq;
mod shallow_hash;
pub mod storage;
mod traverse;
mod type_key_map;
mod unify;

use crate::qualifier::Qualifier;

pub(crate) use deep_eq::*;
pub(crate) use deep_hash::*;
pub use definitions::*;
pub(crate) use monomorphize::requalify_concrete;
pub(crate) use shallow_eq::*;
pub(crate) use shallow_hash::*;
pub use storage::*;
pub(crate) use traverse::*;
pub(crate) use type_key_map::*;
pub(crate) use unify::*;

pub trait Wrapper {
    type Wrap<T: 'static>;
}

// --- Hash/eq mode markers ---

pub(crate) struct QualifiedMode;
pub(crate) struct UnqualifiedMode;

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Qualified<T> {
    pub(crate) inner: T,
    pub(crate) qualifier: Qualifier,
}
