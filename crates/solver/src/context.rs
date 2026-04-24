#![cfg_attr(not(feature = "tracing"), allow(unused_variables))]

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use crate::{
    arena::Arena,
    instrument::solver_event,
    rule::{ResolutionEnv, Rule, RuleEnv, RuleEnvSharedState, RuleResultRef, RuleResultsArena},
    search_graph::{Answer, CacheBucket, CacheKey, DepthFirstNumber, GoalKey, SearchGraph},
    stack::{Stack, StackDepth, StackError},
};

#[derive(Clone, Copy)]
pub(crate) enum AnswerMatchMemo {
    InProgress,
    Resolved(bool),
}

type AnswerMatchMemoKey<R> = (RuleResultRef<R>, Arc<RuleEnv<R>>);
type BlockedCrossEnvReuse<R> = (RuleResultRef<R>, Arc<RuleEnv<R>>);
type RebasedEnvCacheKey<R> = (
    Arc<RuleEnv<R>>,
    <RuleEnv<R> as crate::rule::ResolutionEnv>::DependencyEnvDelta,
);

pub(crate) struct Context<R: Rule> {
    pub(crate) results_arena: RuleResultsArena<R>,
    pub(crate) result_refs: HashMap<GoalKey<R>, RuleResultRef<R>>,
    pub(crate) result_goals: HashMap<RuleResultRef<R>, GoalKey<R>>,
    pub(crate) result_answers: HashMap<RuleResultRef<R>, Answer<R>>,
    pub(crate) answer_fingerprints: HashMap<RuleResultRef<R>, u64>,
    pub(crate) answer_dependents: HashMap<RuleResultRef<R>, HashSet<RuleResultRef<R>>>,
    pub(crate) answer_match_memo: HashMap<AnswerMatchMemoKey<R>, AnswerMatchMemo>,
    rebased_env_cache: HashMap<RebasedEnvCacheKey<R>, Arc<RuleEnv<R>>>,
    pub(crate) blocked_cross_env_reuses: HashSet<BlockedCrossEnvReuse<R>>,
    pub(crate) search_graph: SearchGraph<R>,
    pub(crate) cache: HashMap<CacheKey<R>, CacheBucket<R>>,
    pub(crate) stack: Stack,
    pub(crate) fixpoint_iteration_limit: usize,
    pub(crate) shared_state: RuleEnvSharedState<R>,
    pub(crate) cache_reuse_enabled: bool,
    pub(crate) cache_dedup_enabled: bool,
}

impl<R: Rule> fmt::Debug for Context<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context")
            .field("results", &self.results_arena.len())
            .field("result_refs", &self.result_refs.len())
            .field("result_goals", &self.result_goals.len())
            .field("result_answers", &self.result_answers.len())
            .field("answer_fingerprints", &self.answer_fingerprints.len())
            .field("answer_dependents", &self.answer_dependents.len())
            .field("answer_match_memo", &self.answer_match_memo.len())
            .field("rebased_env_cache", &self.rebased_env_cache.len())
            .field(
                "blocked_cross_env_reuses",
                &self.blocked_cross_env_reuses.len(),
            )
            .field("cache", &self.cache.len())
            .field("fixpoint_iteration_limit", &self.fixpoint_iteration_limit)
            .field(
                "cross_env_active_reuse",
                &cfg!(feature = "cross-env-active-reuse"),
            )
            .field("cache_reuse_enabled", &self.cache_reuse_enabled)
            .field("cache_dedup_enabled", &self.cache_dedup_enabled)
            .finish()
    }
}

impl<R: Rule> Context<R> {
    pub(crate) fn new(
        env_shared_state: RuleEnvSharedState<R>,
        fixpoint_iteration_limit: usize,
        stack_depth_limit: usize,
    ) -> Self {
        Self {
            results_arena: RuleResultsArena::<R>::default(),
            result_refs: HashMap::new(),
            result_goals: HashMap::new(),
            result_answers: HashMap::new(),
            answer_fingerprints: HashMap::new(),
            answer_dependents: HashMap::new(),
            answer_match_memo: HashMap::new(),
            rebased_env_cache: HashMap::new(),
            blocked_cross_env_reuses: HashSet::new(),
            search_graph: SearchGraph::new(),
            cache: HashMap::new(),
            stack: Stack::new(stack_depth_limit),
            fixpoint_iteration_limit,
            shared_state: env_shared_state,
            cache_reuse_enabled: std::env::var_os("INLAY_DISABLE_CACHE_REUSE").is_none(),
            cache_dedup_enabled: std::env::var_os("INLAY_DISABLE_CACHE_DEDUP").is_none(),
        }
    }

    pub(crate) fn result_ref_for(&mut self, goal: &GoalKey<R>) -> RuleResultRef<R> {
        if let Some(result_ref) = self.result_refs.get(goal).copied() {
            return result_ref;
        }

        let result_ref = self.results_arena.insert_placeholder();
        self.result_refs.insert(goal.clone(), result_ref);
        self.result_goals.insert(result_ref, goal.clone());
        result_ref
    }

    pub(crate) fn call_on_stack<T, E>(
        &mut self,
        goal: &GoalKey<R>,
        result_ref: RuleResultRef<R>,
        f: impl FnOnce(&mut Self, DepthFirstNumber, StackDepth) -> Result<T, E>,
    ) -> Result<(DepthFirstNumber, T), E>
    where
        E: From<StackError>,
    {
        let stack_depth = self.stack.push().map_err(E::from)?;
        let dfn = self.search_graph.insert(goal, stack_depth, result_ref);
        let result = f(self, dfn, stack_depth);
        self.search_graph.pop_stack_goal(dfn);
        self.stack.pop(stack_depth);
        result.map(|value| (dfn, value))
    }

    pub(crate) fn goal_for_result_ref(&self, result_ref: RuleResultRef<R>) -> Option<&GoalKey<R>> {
        self.result_goals.get(&result_ref)
    }

    pub(crate) fn replace_answer(&mut self, result_ref: RuleResultRef<R>, answer: Answer<R>) {
        let changed = self
            .result_answers
            .get(&result_ref)
            .is_none_or(|old| old != &answer);
        let dependency_count = answer.dependencies.len() as u64;
        let old_dependencies = self
            .result_answers
            .insert(result_ref, answer)
            .map(|old| old.dependencies)
            .unwrap_or_default();

        for dependency in old_dependencies {
            if let Some(dependents) = self.answer_dependents.get_mut(&dependency.result_ref) {
                dependents.remove(&result_ref);
                if dependents.is_empty() {
                    self.answer_dependents.remove(&dependency.result_ref);
                }
            }
        }

        let dependencies = self
            .result_answers
            .get(&result_ref)
            .expect("inserted answer must exist")
            .dependencies
            .clone();
        for dependency in dependencies {
            self.answer_dependents
                .entry(dependency.result_ref)
                .or_default()
                .insert(result_ref);
        }

        let memo_entries_cleared = self.answer_match_memo.len() as u64;
        solver_event!(
            name: "solver.answer_replaced",
            ?result_ref,
            changed,
            dependency_count,
            memo_entries_cleared
        );
        self.answer_match_memo.clear();
        self.invalidate_fingerprint_closure(result_ref);
    }

    fn invalidate_fingerprint_closure(&mut self, result_ref: RuleResultRef<R>) {
        let mut stack = vec![result_ref];
        let mut visited = HashSet::new();

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            self.answer_fingerprints.remove(&current);
            if let Some(dependents) = self.answer_dependents.get(&current) {
                stack.extend(dependents.iter().copied());
            }
        }
    }

    pub(crate) fn answer_for(&self, result_ref: RuleResultRef<R>) -> Option<&Answer<R>> {
        self.result_answers.get(&result_ref)
    }

    pub(crate) fn rebased_env_for_dependency(
        &mut self,
        parent: &Arc<RuleEnv<R>>,
        delta: &<RuleEnv<R> as crate::rule::ResolutionEnv>::DependencyEnvDelta,
    ) -> Arc<RuleEnv<R>> {
        let key = (Arc::clone(parent), delta.clone());
        let parent_items = R::Env::env_item_count(parent.as_ref()) as u64;
        let delta_items = R::Env::dependency_env_delta_item_count(delta) as u64;
        if let Some(env) = self.rebased_env_cache.get(&key).cloned() {
            solver_event!(
                name: "solver.env_rebased",
                cache_hit = true,
                parent_items,
                delta_items,
                child_items = R::Env::env_item_count(env.as_ref()) as u64
            );
            return env;
        }

        let env = RuleEnv::<R>::apply_dependency_env_delta(parent, delta);
        self.rebased_env_cache.insert(key, Arc::clone(&env));
        solver_event!(
            name: "solver.env_rebased",
            cache_hit = false,
            parent_items,
            delta_items,
            child_items = R::Env::env_item_count(env.as_ref()) as u64
        );
        env
    }
}
