use std::sync::Arc;

use derive_where::derive_where;
use rustc_hash::FxHashMap as HashMap;

use crate::{
    lookup_support::AnswerSupport,
    rule::{RuleEnv, RuleQuery, RuleResultRef},
    search_graph::{Answer, GoalKey},
    traits::Rule,
};

pub(crate) type CacheKey<R> = (RuleQuery<R>, <R as Rule>::RuleStateId);

pub(crate) struct CacheEntry<R: Rule> {
    pub(crate) goal: GoalKey<R>,
    pub(crate) answer: Answer<R>,
    pub(crate) answer_support: Option<AnswerSupport<R>>,
}

pub(crate) struct CachedAnswer<R: Rule> {
    pub(crate) goal: GoalKey<R>,
    pub(crate) answer: Answer<R>,
    answer_support: Option<AnswerSupport<R>>,
}

#[derive_where(Default)]
pub(crate) struct Cache<R: Rule> {
    buckets: HashMap<CacheKey<R>, HashMap<Arc<RuleEnv<R>>, RuleResultRef<R>>>,
    answers: HashMap<RuleResultRef<R>, CachedAnswer<R>>,
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

    pub(crate) fn insert_entry(&mut self, entry: CacheEntry<R>) {
        let result_ref = entry.answer.result_ref;
        let CacheEntry {
            goal,
            answer,
            answer_support,
        } = entry;

        if let Some(existing) = self.answers.get(&result_ref) {
            debug_assert!(
                existing.goal == goal && existing.answer == answer,
                "cached answer records must be immutable"
            );
        } else {
            self.answers.insert(
                result_ref,
                CachedAnswer {
                    goal: goal.clone(),
                    answer,
                    answer_support: None,
                },
            );
        }

        if let Some(answer_support) = answer_support {
            self.store_answer_support(result_ref, answer_support);
        }

        self.buckets
            .entry(cache_key(&goal))
            .or_default()
            .insert(goal.env, result_ref);
    }

    pub(crate) fn goal_for_result_ref(&self, result_ref: RuleResultRef<R>) -> Option<&GoalKey<R>> {
        self.answers.get(&result_ref).map(|answer| &answer.goal)
    }

    pub(crate) fn answer_for(&self, result_ref: RuleResultRef<R>) -> Option<&Answer<R>> {
        self.answers.get(&result_ref).map(|answer| &answer.answer)
    }

    pub(crate) fn stored_answer_support(
        &self,
        result_ref: RuleResultRef<R>,
    ) -> Option<&AnswerSupport<R>> {
        self.answers
            .get(&result_ref)
            .and_then(|answer| answer.answer_support.as_ref())
    }

    pub(crate) fn store_answer_support(
        &mut self,
        result_ref: RuleResultRef<R>,
        answer_support: AnswerSupport<R>,
    ) -> bool {
        let Some(cached_answer) = self.answers.get_mut(&result_ref) else {
            return false;
        };

        if let Some(existing) = &cached_answer.answer_support {
            debug_assert!(
                existing == &answer_support,
                "cached answer support must never be written twice"
            );
            return true;
        }
        cached_answer.answer_support = Some(answer_support);
        true
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
            ExampleEnv, ExampleResultsArena, ExampleRule, ExampleSharedState, ExampleState,
            definition, leaf,
        },
        lookup_support::AnswerSupport,
        search_graph::{Answer, GoalKey, LazyDepth},
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
    fn cache_keeps_single_answer_support() {
        // given
        let mut cache = Cache::<ExampleRule>::default();
        let mut arena = ExampleResultsArena::default();
        let result_ref = arena.insert_placeholder();
        let support = answer_support("first");
        cache.insert_entry(CacheEntry {
            goal: GoalKey {
                query: "first".to_string(),
                state_id: ExampleState::Resolve,
                env: Arc::new(ExampleEnv::new([definition("first", leaf("first"))])),
                lazy_depth: LazyDepth(0),
            },
            answer: Answer {
                result_ref,
                direct_supports: vec![],
                dependencies: vec![],
            },
            answer_support: None,
        });

        // when
        assert!(cache.store_answer_support(result_ref, support.clone()));
        assert!(cache.store_answer_support(result_ref, support.clone()));

        // then
        assert!(cache.stored_answer_support(result_ref) == Some(&support));
    }
}
