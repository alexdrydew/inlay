use std::sync::Arc;

use derive_where::derive_where;
use rustc_hash::FxHashMap as HashMap;

use crate::{
    lookup_support::AnswerSupport,
    rule::{RuleEnv, RuleQuery, RuleResultRef},
    search_graph::GoalKey,
    traits::Rule,
};

pub(crate) type CacheKey<R> = (RuleQuery<R>, <R as Rule>::RuleStateId);

pub(crate) struct CacheableEntry<R: Rule> {
    pub(crate) key: CacheKey<R>,
    pub(crate) env: Arc<RuleEnv<R>>,
    pub(crate) result_ref: RuleResultRef<R>,
    pub(crate) answer_support: Option<AnswerSupport<R>>,
}

#[derive_where(Default)]
pub(crate) struct Cache<R: Rule> {
    buckets: HashMap<CacheKey<R>, HashMap<Arc<RuleEnv<R>>, RuleResultRef<R>>>,
    // Supports for the same answer are alternatives: each support is an AND of checks,
    // while this list is an OR. Merging alternatives would make reuse too strict.
    answer_supports: HashMap<RuleResultRef<R>, Vec<AnswerSupport<R>>>,
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

    pub(crate) fn insert_entry(&mut self, entry: CacheableEntry<R>) {
        let previous_result_ref = self
            .buckets
            .entry(entry.key)
            .or_default()
            .insert(entry.env, entry.result_ref);
        match previous_result_ref {
            Some(previous_result_ref) if previous_result_ref != entry.result_ref => {
                self.answer_supports.remove(&previous_result_ref);
            }
            Some(_) | None => {}
        }

        if let Some(answer_support) = entry.answer_support {
            self.store_answer_support(entry.result_ref, answer_support);
        }
    }

    pub(crate) fn stored_answer_support(
        &self,
        result_ref: RuleResultRef<R>,
    ) -> Option<&AnswerSupport<R>> {
        self.answer_supports
            .get(&result_ref)
            .and_then(|supports| supports.first())
    }

    pub(crate) fn stored_answer_supports(
        &self,
        result_ref: RuleResultRef<R>,
    ) -> Option<&[AnswerSupport<R>]> {
        self.answer_supports
            .get(&result_ref)
            .map(Vec::as_slice)
            .filter(|supports| !supports.is_empty())
    }

    pub(crate) fn store_answer_support(
        &mut self,
        result_ref: RuleResultRef<R>,
        answer_support: AnswerSupport<R>,
    ) {
        let supports = self.answer_supports.entry(result_ref).or_default();
        if !supports.contains(&answer_support) {
            supports.push(answer_support);
        }
    }
}

pub(crate) fn cache_key<R: Rule>(goal: &GoalKey<R>) -> CacheKey<R> {
    (goal.query.clone(), goal.state_id)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        example::{
            ExampleEnv, ExampleResultsArena, ExampleRule, ExampleSharedState, definition, leaf,
        },
        lookup_support::AnswerSupport,
        traits::{Arena, ResolutionEnv},
    };

    use super::*;

    fn answer_support(query: &str) -> AnswerSupport<ExampleRule> {
        let env = Arc::new(ExampleEnv::new([definition(query, leaf(query))]));
        let mut shared_state = ExampleSharedState::default();
        let result = env.lookup(&mut shared_state, &query.to_string());
        AnswerSupport {
            checks: vec![env.lookup_support(&mut shared_state, &query.to_string(), &result)],
        }
    }

    #[test]
    fn cache_keeps_distinct_alternative_answer_supports() {
        // given
        let mut cache = Cache::<ExampleRule>::default();
        let mut arena = ExampleResultsArena::default();
        let result_ref = arena.insert_placeholder();
        let first = answer_support("first");
        let second = answer_support("second");

        // when
        cache.store_answer_support(result_ref, first.clone());
        cache.store_answer_support(result_ref, second.clone());
        cache.store_answer_support(result_ref, first.clone());

        // then
        let supports = cache
            .stored_answer_supports(result_ref)
            .expect("stored supports should exist");
        assert_eq!(supports.len(), 2);
        assert!(supports.contains(&first));
        assert!(supports.contains(&second));
    }
}
