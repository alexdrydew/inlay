use std::{convert::Infallible, sync::Arc};

use derive_where::derive_where;
use inlay_instrument_macros::instrumented;
use rustc_hash::FxHashMap as HashMap;

use crate::{
    lookup_support::{
        AnswerSupport, AnswerSupportSource, DependencyResolution, SupportAnswer,
        build_answer_support as build_answer_support_from_source,
    },
    rule::{RuleEnv, RuleQuery, RuleResultRef},
    search_graph::{Answer, Dependency, GoalKey},
    traits::Rule,
};

#[derive_where(PartialEq, Eq, Hash)]
struct CacheKey<R: Rule>(RuleQuery<R>, <R as Rule>::RuleStateId);

impl<R: Rule> From<&GoalKey<R>> for CacheKey<R> {
    fn from(goal: &GoalKey<R>) -> Self {
        Self(goal.query.clone(), goal.state_id)
    }
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

pub(crate) type CachedAnswer<R> = Answer<R, CachedResultRef<R>>;

pub(crate) struct CacheEntry<R: Rule> {
    pub(crate) goal: GoalKey<R>,
    pub(crate) answer: CachedAnswer<R>,
    pub(crate) answer_support: Option<AnswerSupport<R>>,
}

impl<R: Rule> CacheEntry<R> {
    pub(crate) fn new(
        goal: GoalKey<R>,
        answer: Answer<R>,
        answer_support: Option<AnswerSupport<R>>,
    ) -> Self {
        Self {
            goal,
            answer: CachedAnswer {
                result_ref: CachedResultRef::new(answer.result_ref),
                direct_supports: answer.direct_supports,
                dependencies: answer
                    .dependencies
                    .into_iter()
                    .map(|dependency| Dependency {
                        result_ref: CachedResultRef::new(dependency.result_ref),
                        env_delta: dependency.env_delta,
                    })
                    .collect(),
            },
            answer_support,
        }
    }

    fn store_answer_support(&mut self, answer_support: AnswerSupport<R>) {
        if let Some(existing) = &self.answer_support {
            debug_assert!(
                existing == &answer_support,
                "cached answer support must never be written twice"
            );
            return;
        }
        self.answer_support = Some(answer_support);
    }
}

#[derive_where(Default)]
pub(crate) struct Cache<R: Rule> {
    buckets: HashMap<CacheKey<R>, HashMap<Arc<RuleEnv<R>>, RuleResultRef<R>>>,
    answers: HashMap<RuleResultRef<R>, CacheEntry<R>>,
}

impl<R: Rule> AnswerSupportSource<R> for Cache<R> {
    type AnswerRef = CachedResultRef<R>;
    type DependencyRef = CachedResultRef<R>;
    type Error = Infallible;

    fn answer_for(
        &mut self,
        result_ref: Self::AnswerRef,
    ) -> Result<SupportAnswer<R, Self::DependencyRef>, Self::Error> {
        let answer = Cache::answer_for(self, result_ref).clone();
        Ok(SupportAnswer {
            direct_supports: answer.direct_supports,
            dependencies: answer.dependencies,
        })
    }

    fn resolve_dependency(
        &mut self,
        result_ref: Self::DependencyRef,
    ) -> Result<DependencyResolution<R, Self::AnswerRef>, Self::Error> {
        if let Some(answer_support) = Cache::stored_answer_support(self, result_ref) {
            return Ok(DependencyResolution::Precomputed(answer_support.clone()));
        }
        Ok(DependencyResolution::Traverse(result_ref))
    }
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
        let result_ref = entry.answer.result_ref.result_ref();
        let key = CacheKey::from(&entry.goal);
        let env = Arc::clone(&entry.goal.env);
        let answer_support = entry.answer_support.take();

        if let Some(existing) = self.answers.get_mut(&result_ref) {
            debug_assert!(
                existing.goal == entry.goal && existing.answer == entry.answer,
                "cached answer records must be immutable"
            );
            if let Some(answer_support) = answer_support {
                existing.store_answer_support(answer_support);
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

    fn answer_for(&self, result_ref: CachedResultRef<R>) -> &CachedAnswer<R> {
        &self.entry(result_ref).answer
    }

    fn stored_answer_support(&self, result_ref: CachedResultRef<R>) -> Option<&AnswerSupport<R>> {
        self.entry(result_ref).answer_support.as_ref()
    }

    pub(crate) fn answer_support(&mut self, result_ref: CachedResultRef<R>) -> AnswerSupport<R> {
        if let Some(answer_support) = self.stored_answer_support(result_ref) {
            return answer_support.clone();
        }

        let answer_support = self.build_answer_support(result_ref);
        self.entry_mut(result_ref)
            .store_answer_support(answer_support.clone());
        answer_support
    }

    #[instrumented(
        name = "solver.build_cached_answer_support",
        target = "context_solver",
        level = "trace",
        skip(self),
        fields(result_ref = ?result_ref, answer_nodes, checks, missing_answer)
    )]
    fn build_answer_support(&mut self, result_ref: CachedResultRef<R>) -> AnswerSupport<R> {
        match build_answer_support_from_source(self, result_ref, result_ref) {
            Ok(support) => support,
            Err(error) => match error {},
        }
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
        let env = Arc::new(ExampleEnv::default());
        let mut shared_state = ExampleSharedState::new([definition(query, leaf(query))]);
        let result = env.lookup(&mut shared_state, &query.to_string());
        AnswerSupport {
            checks: vec![env.lookup_support(&mut shared_state, &query.to_string(), &result)],
        }
    }

    #[test]
    fn cache_builds_and_reuses_answer_support() {
        // given
        let mut cache = Cache::<ExampleRule>::default();
        let mut arena = ExampleResultsArena::default();
        let result_ref = arena.insert_placeholder();
        let support = answer_support("first");
        cache.insert_entry(CacheEntry::new(
            GoalKey {
                query: "first".to_string(),
                state_id: ExampleState::Resolve,
                env: Arc::new(ExampleEnv::default()),
                lazy_depth: LazyDepth(0),
            },
            Answer {
                result_ref,
                direct_supports: support.checks.clone(),
                dependencies: vec![],
            },
            None,
        ));
        let cached_result_ref = cache
            .cached_result_ref(result_ref)
            .expect("inserted entry must be cached");

        // when
        let first = cache.answer_support(cached_result_ref);
        let second = cache.answer_support(cached_result_ref);

        // then
        assert_eq!(first, support);
        assert_eq!(second, support);
        assert!(cache.stored_answer_support(cached_result_ref) == Some(&support));
    }
}
