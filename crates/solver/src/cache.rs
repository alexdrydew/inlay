use std::sync::Arc;

use derive_where::derive_where;
use rustc_hash::FxHashMap as HashMap;

use crate::{
    lookup_support::AnswerSupport,
    rule::{RuleEnv, RuleQuery, RuleResultRef},
    search_graph::{Answer, GoalKey},
    traits::Rule,
};

#[derive_where(PartialEq, Eq, Hash)]
struct CacheKey<R: Rule>(RuleQuery<R>, <R as Rule>::RuleStateId);

impl<R: Rule> From<&GoalKey<R>> for CacheKey<R> {
    fn from(goal: &GoalKey<R>) -> Self {
        Self(goal.query.clone(), goal.state_id)
    }
}

pub(crate) struct CacheEntry<R: Rule> {
    pub(crate) goal: GoalKey<R>,
    pub(crate) answer: Answer<R>,
    pub(crate) answer_support: Option<AnswerSupport<R>>,
}

#[derive_where(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CachedResultRef<R: Rule>(RuleResultRef<R>);

impl<R: Rule> CachedResultRef<R> {
    fn new(result_ref: RuleResultRef<R>) -> Self {
        Self(result_ref)
    }

    pub(crate) fn result_ref(self) -> RuleResultRef<R> {
        self.0
    }
}

#[derive_where(Default)]
pub(crate) struct Cache<R: Rule> {
    buckets: HashMap<CacheKey<R>, HashMap<Arc<RuleEnv<R>>, RuleResultRef<R>>>,
    answers: HashMap<RuleResultRef<R>, CacheEntry<R>>,
}

impl<R: Rule> Cache<R> {
    pub(crate) fn len(&self) -> usize {
        self.answers.len()
    }

    pub(crate) fn get_same_env_result(&self, goal: &GoalKey<R>) -> Option<CachedResultRef<R>> {
        self.buckets
            .get(&CacheKey::from(goal))?
            .get(&goal.env)
            .copied()
            .map(CachedResultRef::new)
    }

    pub(crate) fn get_result_candidates(&self, goal: &GoalKey<R>) -> Vec<CachedResultRef<R>> {
        let Some(bucket) = self.buckets.get(&CacheKey::from(goal)) else {
            return vec![];
        };
        let exact_result_ref = bucket.get(&goal.env).copied();
        bucket
            .values()
            .copied()
            .filter(|result_ref| Some(*result_ref) != exact_result_ref)
            .map(CachedResultRef::new)
            .collect()
    }

    pub(crate) fn insert_entry(&mut self, mut entry: CacheEntry<R>) {
        let result_ref = entry.answer.result_ref;
        let key = CacheKey::from(&entry.goal);
        let env = Arc::clone(&entry.goal.env);
        let answer_support = entry.answer_support.take();

        if let Some(existing) = self.answers.get_mut(&result_ref) {
            debug_assert!(
                existing.goal == entry.goal && existing.answer == entry.answer,
                "cached answer records must be immutable"
            );
            if let Some(answer_support) = answer_support {
                Self::store_entry_answer_support(existing, answer_support);
            }
        } else {
            entry.answer_support = answer_support;
            self.answers.insert(result_ref, entry);
        }

        self.buckets.entry(key).or_default().insert(env, result_ref);
    }

    pub(crate) fn cached_result_ref(
        &self,
        result_ref: RuleResultRef<R>,
    ) -> Option<CachedResultRef<R>> {
        self.answers
            .contains_key(&result_ref)
            .then_some(CachedResultRef::new(result_ref))
    }

    fn entry(&self, result_ref: CachedResultRef<R>) -> &CacheEntry<R> {
        self.answers
            .get(&result_ref.result_ref())
            .expect("cache-created result ref must remain in cache")
    }

    fn entry_mut(&mut self, result_ref: CachedResultRef<R>) -> &mut CacheEntry<R> {
        self.answers
            .get_mut(&result_ref.result_ref())
            .expect("cache-created result ref must remain in cache")
    }

    #[cfg(feature = "tracing")]
    pub(crate) fn goal_for_result_ref(&self, result_ref: CachedResultRef<R>) -> &GoalKey<R> {
        &self.entry(result_ref).goal
    }

    pub(crate) fn answer_for(&self, result_ref: CachedResultRef<R>) -> &Answer<R> {
        &self.entry(result_ref).answer
    }

    pub(crate) fn stored_answer_support(
        &self,
        result_ref: CachedResultRef<R>,
    ) -> Option<&AnswerSupport<R>> {
        self.entry(result_ref).answer_support.as_ref()
    }

    pub(crate) fn store_answer_support(
        &mut self,
        result_ref: CachedResultRef<R>,
        answer_support: AnswerSupport<R>,
    ) {
        Self::store_entry_answer_support(self.entry_mut(result_ref), answer_support);
    }

    fn store_entry_answer_support(
        cached_answer: &mut CacheEntry<R>,
        answer_support: AnswerSupport<R>,
    ) {
        if let Some(existing) = &cached_answer.answer_support {
            debug_assert!(
                existing == &answer_support,
                "cached answer support must never be written twice"
            );
            return;
        }
        cached_answer.answer_support = Some(answer_support);
    }
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
        let cached_result_ref = cache
            .cached_result_ref(result_ref)
            .expect("inserted entry must be cached");

        // when
        cache.store_answer_support(cached_result_ref, support.clone());
        cache.store_answer_support(cached_result_ref, support.clone());

        // then
        assert!(cache.stored_answer_support(cached_result_ref) == Some(&support));
    }
}
