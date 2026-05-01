#![cfg_attr(not(feature = "tracing"), allow(unused_variables))]

use std::{
    fmt,
    hash::{Hash, Hasher},
};
use std::sync::Arc;

use inlay_instrument::{inlay_span_record, instrumented};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::{
    cache::Cache,
    rule::{RuleEnv, RuleEnvSharedState, RuleResultRef, RuleResultsArena},
    search_graph::{Answer, DepthFirstNumber, GoalKey, SearchGraph},
    stack::{Stack, StackDepth, StackError},
    traits::{Arena, Rule},
};

#[derive(Clone, Copy)]
pub(crate) enum AnswerMatchMemo {
    InProgress,
    Resolved(bool),
}

struct AnswerMatchMemoEnv<R: Rule>(Arc<RuleEnv<R>>);

impl<R: Rule> Clone for AnswerMatchMemoEnv<R> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<R: Rule> AnswerMatchMemoEnv<R> {
    fn new(env: &Arc<RuleEnv<R>>) -> Self {
        Self(Arc::clone(env))
    }
}

impl<R: Rule> PartialEq for AnswerMatchMemoEnv<R> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<R: Rule> Eq for AnswerMatchMemoEnv<R> {}

impl<R: Rule> Hash for AnswerMatchMemoEnv<R> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

struct AnswerMatchMemoKey<R: Rule> {
    result_ref: RuleResultRef<R>,
    env: AnswerMatchMemoEnv<R>,
}

impl<R: Rule> AnswerMatchMemoKey<R> {
    fn new(result_ref: RuleResultRef<R>, env: &Arc<RuleEnv<R>>) -> Self {
        Self {
            result_ref,
            env: AnswerMatchMemoEnv::new(env),
        }
    }
}

impl<R: Rule> PartialEq for AnswerMatchMemoKey<R> {
    fn eq(&self, other: &Self) -> bool {
        self.result_ref == other.result_ref && self.env == other.env
    }
}

impl<R: Rule> Eq for AnswerMatchMemoKey<R> {}

impl<R: Rule> Hash for AnswerMatchMemoKey<R> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.result_ref.hash(state);
        self.env.hash(state);
    }
}

type BlockedCrossEnvReuse<R> = (RuleResultRef<R>, Arc<RuleEnv<R>>);
pub(crate) struct Context<R: Rule> {
    pub(crate) results_arena: RuleResultsArena<R>,
    answer_match_memo: HashMap<AnswerMatchMemoKey<R>, AnswerMatchMemo>,
    answer_match_memo_envs: HashMap<RuleResultRef<R>, HashSet<AnswerMatchMemoEnv<R>>>,
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
            .field("answer_match_memo", &self.answer_match_memo.len())
            .field("answer_match_memo_envs", &self.answer_match_memo_envs.len())
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
            answer_match_memo: HashMap::default(),
            answer_match_memo_envs: HashMap::default(),
            blocked_cross_env_reuses: HashSet::default(),
            search_graph: SearchGraph::default(),
            cache: Cache::default(),
            stack: Stack::new(stack_depth_limit),
            fixpoint_iteration_limit,
            shared_state: env_shared_state,
        }
    }

    pub(crate) fn call_on_stack<T, E>(
        &mut self,
        goal: &GoalKey<R>,
        f: impl FnOnce(&mut Self, DepthFirstNumber, StackDepth, RuleResultRef<R>) -> Result<T, E>,
    ) -> Result<(DepthFirstNumber, RuleResultRef<R>, T), E>
    where
        E: From<StackError>,
    {
        let stack_depth = self.stack.push().map_err(E::from)?;
        let (dfn, result_ref) =
            self.search_graph
                .insert(goal, stack_depth, &mut self.results_arena);
        let result = f(self, dfn, stack_depth, result_ref);
        self.search_graph.pop_stack_goal(dfn);
        self.stack.pop(stack_depth);
        result.map(|value| (dfn, result_ref, value))
    }

    #[instrumented(
        name = "solver.replace_answer",
        target = "inlay",
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
        let memo_entries_cleared =
            self.invalidate_answer_match_memos(replacement.affected_result_refs);
        inlay_span_record!(
            changed = replacement.changed,
            dependency_count = replacement.dependency_count,
            support_entries_cleared = replacement.support_entries_cleared,
            memo_entries_cleared = memo_entries_cleared,
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
                    .remove(&AnswerMatchMemoKey {
                        result_ref: affected_result_ref,
                        env,
                    })
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
            .get(&AnswerMatchMemoKey::new(result_ref, env))
            .copied()
    }

    pub(crate) fn insert_answer_match_memo(
        &mut self,
        result_ref: RuleResultRef<R>,
        env: &Arc<RuleEnv<R>>,
        memo: AnswerMatchMemo,
    ) {
        let env = AnswerMatchMemoEnv::new(env);
        self.answer_match_memo
            .insert(AnswerMatchMemoKey { result_ref, env: env.clone() }, memo);
        self.answer_match_memo_envs
            .entry(result_ref)
            .or_default()
            .insert(env);
    }
}
