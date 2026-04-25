use std::collections::HashMap;
use std::sync::Arc;

use derive_where::derive_where;

use crate::{
    rule::{RuleEnv, RuleQuery, RuleResultRef},
    search_graph::GoalKey,
    traits::Rule,
};

pub(crate) type CacheKey<R> = (RuleQuery<R>, <R as Rule>::RuleStateId);

#[derive_where(Default)]
pub(crate) struct Cache<R: Rule> {
    buckets: HashMap<CacheKey<R>, HashMap<Arc<RuleEnv<R>>, RuleResultRef<R>>>,
}

impl<R: Rule> Cache<R> {
    pub(crate) fn len(&self) -> usize {
        self.buckets.len()
    }

    pub(crate) fn get(
        &self,
        key: &CacheKey<R>,
    ) -> Option<&HashMap<Arc<RuleEnv<R>>, RuleResultRef<R>>> {
        self.buckets.get(key)
    }

    pub(crate) fn insert_entry(
        &mut self,
        key: CacheKey<R>,
        env: Arc<R::Env>,
        result_ref: RuleResultRef<R>,
    ) {
        self.buckets.entry(key).or_default().insert(env, result_ref);
    }
}

pub(crate) fn cache_key<R: Rule>(goal: &GoalKey<R>) -> CacheKey<R> {
    (goal.query.clone(), goal.state_id)
}
