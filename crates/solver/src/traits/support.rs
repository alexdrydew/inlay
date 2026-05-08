use std::fmt::Debug;
use std::hash::Hash;

pub trait RuleLookupSupport: Hash + Eq + Clone + Debug {
    fn merge_lookup_support(&self, other: &Self) -> Option<Self>;
}

fn insert_compacted_lookup_support<T: RuleLookupSupport>(compacted: &mut Vec<T>, mut support: T) {
    let mut index = 0;
    while index < compacted.len() {
        let Some(merged) = compacted[index].merge_lookup_support(&support) else {
            index += 1;
            continue;
        };

        if merged == compacted[index] {
            return;
        }

        if merged == support {
            compacted.swap_remove(index);
            continue;
        }

        compacted.swap_remove(index);
        support = merged;
        index = 0;
    }

    compacted.push(support);
}

impl<T> RuleLookupSupport for Vec<T>
where
    T: RuleLookupSupport,
{
    fn merge_lookup_support(&self, other: &Self) -> Option<Self> {
        let mut compacted = Vec::with_capacity(self.len() + other.len());
        for support in self.iter().chain(other).cloned() {
            insert_compacted_lookup_support(&mut compacted, support);
        }
        Some(compacted)
    }
}
