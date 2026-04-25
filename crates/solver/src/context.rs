#![cfg_attr(not(feature = "tracing"), allow(unused_variables))]

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use inlay_instrument_macros::instrumented;

use crate::{
    cache::Cache,
    instrument::solver_span_record,
    rule::{RuleEnv, RuleEnvSharedState, RuleResultRef, RuleResultsArena},
    search_graph::{Answer, DepthFirstNumber, GoalKey, SearchGraph},
    stack::{Stack, StackDepth, StackError},
    traits::{Arena, ResolutionEnv, Rule},
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
    <RuleEnv<R> as ResolutionEnv>::DependencyEnvDelta,
);

pub(crate) struct Context<R: Rule> {
    pub(crate) results_arena: RuleResultsArena<R>,
    pub(crate) result_refs: HashMap<GoalKey<R>, RuleResultRef<R>>,
    pub(crate) result_goals: HashMap<RuleResultRef<R>, GoalKey<R>>,
    answer_match_memo: HashMap<AnswerMatchMemoKey<R>, AnswerMatchMemo>,
    answer_match_memo_envs: HashMap<RuleResultRef<R>, HashSet<Arc<RuleEnv<R>>>>,
    rebased_env_cache: HashMap<RebasedEnvCacheKey<R>, Arc<RuleEnv<R>>>,
    pub(crate) blocked_cross_env_reuses: HashSet<BlockedCrossEnvReuse<R>>,
    pub(crate) search_graph: SearchGraph<R>,
    pub(crate) cache: Cache<R>,
    pub(crate) stack: Stack,
    pub(crate) fixpoint_iteration_limit: usize,
    pub(crate) shared_state: RuleEnvSharedState<R>,
}

impl<R: Rule> fmt::Debug for Context<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context")
            .field("results", &self.results_arena.len())
            .field("result_refs", &self.result_refs.len())
            .field("result_goals", &self.result_goals.len())
            .field("answer_match_memo", &self.answer_match_memo.len())
            .field("answer_match_memo_envs", &self.answer_match_memo_envs.len())
            .field("rebased_env_cache", &self.rebased_env_cache.len())
            .field(
                "blocked_cross_env_reuses",
                &self.blocked_cross_env_reuses.len(),
            )
            .field("cache", &self.cache.len())
            .field("fixpoint_iteration_limit", &self.fixpoint_iteration_limit)
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
            answer_match_memo: HashMap::new(),
            answer_match_memo_envs: HashMap::new(),
            rebased_env_cache: HashMap::new(),
            blocked_cross_env_reuses: HashSet::new(),
            search_graph: SearchGraph::new(),
            cache: Cache::default(),
            stack: Stack::new(stack_depth_limit),
            fixpoint_iteration_limit,
            shared_state: env_shared_state,
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

    #[instrumented(
        name = "solver.replace_answer",
        target = "context_solver",
        level = "trace",
        fields(
            result_ref = ?answer.result_ref,
            changed,
            dependency_count,
            memo_entries_cleared,
            support_entries_cleared
        )
    )]
    pub(crate) fn store_graph_answer(&mut self, dfn: DepthFirstNumber, answer: Answer<R>) {
        let replacement = self.search_graph.replace_answer(dfn, answer);
        let changed = replacement.changed;
        let dependency_count = replacement.dependency_count;
        let support_entries_cleared = replacement.support_entries_cleared;
        let memo_entries_cleared =
            self.invalidate_answer_match_memos(replacement.affected_result_refs);
        solver_span_record!(
            changed,
            dependency_count,
            memo_entries_cleared,
            support_entries_cleared
        );
    }

    fn invalidate_answer_match_memos(
        &mut self,
        result_refs: impl IntoIterator<Item = RuleResultRef<R>>,
    ) -> u64 {
        let mut removed = 0_u64;
        for affected_result_ref in result_refs {
            let Some(envs) = self.answer_match_memo_envs.remove(&affected_result_ref) else {
                continue;
            };
            for env in envs {
                if self
                    .answer_match_memo
                    .remove(&(affected_result_ref, env))
                    .is_some()
                {
                    removed += 1;
                }
            }
        }
        removed
    }

    pub(crate) fn answer_match_memo(
        &self,
        result_ref: RuleResultRef<R>,
        env: &Arc<RuleEnv<R>>,
    ) -> Option<AnswerMatchMemo> {
        self.answer_match_memo
            .get(&(result_ref, Arc::clone(env)))
            .copied()
    }

    pub(crate) fn insert_answer_match_memo(
        &mut self,
        result_ref: RuleResultRef<R>,
        env: &Arc<RuleEnv<R>>,
        memo: AnswerMatchMemo,
    ) {
        self.answer_match_memo
            .insert((result_ref, Arc::clone(env)), memo);
        self.answer_match_memo_envs
            .entry(result_ref)
            .or_default()
            .insert(Arc::clone(env));
    }

    #[instrumented(
        name = "solver.rebase_env_for_dependency",
        target = "context_solver",
        level = "trace",
        fields(parent_items, delta_items, child_items, cache_hit)
    )]
    pub(crate) fn rebase_env_for_dependency(
        &mut self,
        parent: &Arc<RuleEnv<R>>,
        delta: &<RuleEnv<R> as ResolutionEnv>::DependencyEnvDelta,
    ) -> Arc<RuleEnv<R>> {
        let key = (Arc::clone(parent), delta.clone());
        solver_span_record!(
            parent_items = R::Env::env_item_count(parent.as_ref()) as u64,
            delta_items = R::Env::dependency_env_delta_item_count(delta) as u64
        );

        if let Some(env) = self.rebased_env_cache.get(&key).cloned() {
            solver_span_record!(
                cache_hit = true,
                child_items = R::Env::env_item_count(env.as_ref()) as u64
            );
            return env;
        }

        let env = RuleEnv::<R>::apply_dependency_env_delta(parent, delta);
        self.rebased_env_cache.insert(key, Arc::clone(&env));
        solver_span_record!(
            cache_hit = false,
            child_items = R::Env::env_item_count(env.as_ref()) as u64
        );
        env
    }
}
