use std::fmt::Debug;
use std::hash::Hash;

pub trait RuleLookupSupport: Hash + Eq + Clone + Debug {
    fn merge_lookup_support(&self, other: &Self) -> Option<Self>;
}
