#![cfg_attr(not(feature = "tracing"), allow(unused_variables, unused_assignments))]

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use derive_where::derive_where;
use inlay_instrument::{inlay_event, inlay_in_span, inlay_span_record, instrumented};
#[cfg(feature = "tracing")]
use rustc_hash::FxHasher;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use thiserror::Error;

use crate::{
    cache::{Cache, CachedResultRef},
    lookup_support::{AnswerSupportBuildError, compact_lookup_supports},
    rule::{RuleEnv, RuleEnvSharedState, RuleQuery, RuleResult, RuleResultRef, RuleResultsArena},
    search_graph::{Answer, Dependency, GoalKey, LazyDepth, Minimums, SearchGraph},
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

#[derive_where(Debug)]
pub(crate) enum GoalSolveResult<R: Rule> {
    Resolved { result_ref: RuleResultRef<R> },
    Lazy { result_ref: RuleResultRef<R> },
    LazyCrossEnv { result_ref: RuleResultRef<R> },
}

pub enum SolveResult<'a, R: Rule> {
    Resolved {
        result: &'a RuleResult<R>,
        result_ref: RuleResultRef<R>,
    },
    Lazy {
        result_ref: RuleResultRef<R>,
    },
}

impl<R: Rule> std::fmt::Debug for SolveResult<'_, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resolved { result, result_ref } => f
                .debug_struct("Resolved")
                .field("result_ref", result_ref)
                .field("result_kind", &if result.is_ok() { "ok" } else { "err" })
                .finish(),
            Self::Lazy { result_ref } => f
                .debug_struct("Lazy")
                .field("result_ref", result_ref)
                .finish(),
        }
    }
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum SolveError {
    #[error("fixpoint iteration limit reached")]
    FixpointIterationLimitReached,
    #[error("stack overflow depth reached")]
    StackOverflowDepthReached,
    #[error("same depth cycle")]
    SameDepthCycle,
    #[error("answer support closure is incomplete")]
    AnswerSupportClosureIncomplete,
}

impl From<StackError> for SolveError {
    fn from(error: StackError) -> Self {
        match error {
            StackError::Overflow => Self::StackOverflowDepthReached,
        }
    }
}

impl From<AnswerSupportBuildError> for SolveError {
    fn from(_: AnswerSupportBuildError) -> Self {
        Self::AnswerSupportClosureIncomplete
    }
}

pub struct Solver<R: Rule> {
    pub(crate) rule: Arc<R>,
    pub(crate) results_arena: RuleResultsArena<R>,
    answer_match_memo: HashMap<AnswerMatchMemoKey<R>, AnswerMatchMemo>,
    answer_match_memo_envs: HashMap<RuleResultRef<R>, HashSet<AnswerMatchMemoEnv<R>>>,
    pub(crate) blocked_cross_env_reuses: HashSet<BlockedCrossEnvReuse<R>>,
    pub(crate) search_graph: SearchGraph<R>,
    pub(crate) cache: Cache<R>,
    pub(crate) stack: Stack,
    pub(crate) fixpoint_iteration_limit: usize,
    stack_depth_limit: usize,
    pub(crate) shared_state: RuleEnvSharedState<R>,
}

impl<R: Rule> std::fmt::Debug for Solver<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Solver")
            .field("rule", &self.rule)
            .field("results", &self.results_arena.len())
            .field("answer_match_memo", &self.answer_match_memo.len())
            .field("answer_match_memo_envs", &self.answer_match_memo_envs.len())
            .field(
                "blocked_cross_env_reuses",
                &self.blocked_cross_env_reuses.len(),
            )
            .field("cache", &self.cache.len())
            .field("fixpoint_iteration_limit", &self.fixpoint_iteration_limit)
            .field("stack_depth_limit", &self.stack_depth_limit)
            .finish()
    }
}

impl<R: Rule> Solver<R> {
    pub fn new(
        rule: R,
        shared_state: RuleEnvSharedState<R>,
        fixpoint_iteration_limit: usize,
        stack_depth_limit: usize,
    ) -> Self {
        Self {
            rule: Arc::new(rule),
            results_arena: RuleResultsArena::<R>::default(),
            answer_match_memo: HashMap::default(),
            answer_match_memo_envs: HashMap::default(),
            blocked_cross_env_reuses: HashSet::default(),
            search_graph: SearchGraph::default(),
            cache: Cache::default(),
            stack: Stack::new(stack_depth_limit),
            fixpoint_iteration_limit,
            stack_depth_limit,
            shared_state,
        }
    }

    fn reset_active_state(&mut self) {
        self.answer_match_memo.clear();
        self.answer_match_memo_envs.clear();
        self.blocked_cross_env_reuses.clear();
        self.search_graph = SearchGraph::default();
        self.stack = Stack::new(self.stack_depth_limit);
    }

    pub fn result(&self, result_ref: RuleResultRef<R>) -> Option<&RuleResult<R>> {
        self.results_arena.get(&result_ref)
    }

    pub fn results(&self) -> &RuleResultsArena<R> {
        &self.results_arena
    }

    pub fn shared_state(&self) -> &RuleEnvSharedState<R> {
        &self.shared_state
    }

    pub fn shared_state_mut(&mut self) -> &mut RuleEnvSharedState<R> {
        &mut self.shared_state
    }

    pub fn clear_cache(&mut self) {
        self.cache = Cache::default();
    }

    pub fn clear_results_and_cache(&mut self) {
        self.reset_active_state();
        self.results_arena = RuleResultsArena::<R>::default();
        self.cache = Cache::default();
    }

    pub fn into_parts(self) -> (R, RuleEnvSharedState<R>, RuleResultsArena<R>) {
        let Self {
            rule,
            shared_state,
            results_arena,
            ..
        } = self;
        let rule = match Arc::try_unwrap(rule) {
            Ok(rule) => rule,
            Err(_) => panic!("solver rule must not be shared when consumed"),
        };
        (rule, shared_state, results_arena)
    }

    pub fn into_shared_state(self) -> RuleEnvSharedState<R> {
        let (_, shared_state, _) = self.into_parts();
        shared_state
    }

    #[instrumented(
        name = "solver.solve",
        target = "inlay",
        level = "trace",
        ret,
        skip(self),
        fields(
            query_hash = hash_value(&query),
            env_hash = debug_env_hash::<R>(&Default::default()),
            state_hash = hash_value(&initial_rule),
            lazy_depth = 0_u64
        )
    )]
    pub fn solve(
        &mut self,
        query: RuleQuery<R>,
        initial_rule: R::RuleStateId,
    ) -> Result<RuleResultRef<R>, SolveError> {
        let root_goal = GoalKey {
            query,
            state_id: initial_rule,
            env: Arc::new(Default::default()),
            lazy_depth: LazyDepth(0),
        };

        self.reset_active_state();
        self.solve_goal(root_goal).map(|(solve_result, _)| {
            match solve_result {
                GoalSolveResult::Resolved { result_ref } => result_ref,
                GoalSolveResult::Lazy { .. } | GoalSolveResult::LazyCrossEnv { .. } => {
                    unreachable!(
                        "root solve_goal cannot resolve lazily because lazy results require an active ancestor"
                    )
                }
            }
        })
    }

    #[instrumented(
        name = "solver.solve_goal",
        target = "inlay",
        level = "trace",
        ret,
        err,
        skip(self),
        fields(
            query_hash = hash_value(&goal.query),
            env_hash = debug_env_hash::<R>(Arc::as_ref(&goal.env)),
            state_hash = hash_value(&goal.state_id),
            lazy_depth = goal.lazy_depth.0 as u64
        )
    )]
    pub(crate) fn solve_goal(
        &mut self,
        goal: GoalKey<R>,
    ) -> Result<(GoalSolveResult<R>, Minimums), SolveError> {
        if let Some(result) = self.try_close_active_cycle(&goal)? {
            return Ok(result);
        }

        if let Some(result) = self.try_close_cross_env_active_cycle(&goal) {
            return Ok(result);
        }

        if let Some(result) = self.try_cache_reuse(&goal) {
            return Ok(result);
        }

        if let Some(result) = self.try_graph_goal_reuse(&goal) {
            return Ok(result);
        }

        self.solve_new_goal(goal)
    }
}

impl<R: Rule> Solver<R> {
    pub(crate) fn call_on_stack<T, E>(
        &mut self,
        goal: &GoalKey<R>,
        f: impl FnOnce(
            &mut Self,
            crate::search_graph::DepthFirstNumber,
            StackDepth,
            RuleResultRef<R>,
        ) -> Result<T, E>,
    ) -> Result<(crate::search_graph::DepthFirstNumber, RuleResultRef<R>, T), E>
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
    pub(crate) fn store_graph_answer(
        &mut self,
        dfn: crate::search_graph::DepthFirstNumber,
        answer: Answer<R>,
    ) {
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
        self.answer_match_memo.insert(
            AnswerMatchMemoKey {
                result_ref,
                env: env.clone(),
            },
            memo,
        );
        self.answer_match_memo_envs
            .entry(result_ref)
            .or_default()
            .insert(env);
    }
}

impl<R: Rule> Solver<R> {
    fn replace_result(&mut self, result_ref: RuleResultRef<R>, result: RuleResult<R>) {
        self.results_arena
            .replace(result_ref, result)
            .expect("solver-managed result ref must remain valid");
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActiveAnswerMatch {
    Matches,
    Mismatch,
}

impl<R: Rule> Solver<R> {
    #[instrumented(
        name = "solver.answer_match",
        target = "inlay",
        level = "trace",
        ret,
        skip(self),
        fields(
            result_ref = ?result_ref,
            env_hash = debug_env_hash::<R>(Arc::as_ref(env))
        )
    )]
    fn answer_matches_env(&mut self, result_ref: CachedResultRef<R>, env: &Arc<R::Env>) -> bool {
        let raw_result_ref = result_ref.result_ref();
        match self.answer_match_memo(raw_result_ref, env) {
            Some(AnswerMatchMemo::Resolved(matches)) => {
                inlay_event!(
                    name: "solver.answer_match_memo",
                    matched = matches
                );
                return matches;
            }
            Some(AnswerMatchMemo::InProgress) => {
                inlay_event!(name: "solver.answer_match_in_progress");
                return true;
            }
            None => {}
        }

        self.insert_answer_match_memo(raw_result_ref, env, AnswerMatchMemo::InProgress);

        let support = self.cached_answer_support(result_ref);
        let matches = self.answer_support_matches_env(raw_result_ref, &support, env);
        self.insert_answer_match_memo(raw_result_ref, env, AnswerMatchMemo::Resolved(matches));
        matches
    }

    #[instrumented(
        name = "solver.answer_matches_env_for_backref",
        target = "inlay",
        level = "trace",
        ret,
        skip(self),
        fields(
            result_ref = ?result_ref,
            env_hash = debug_env_hash::<R>(Arc::as_ref(env))
        )
    )]
    fn answer_matches_env_for_backref(
        &mut self,
        result_ref: RuleResultRef<R>,
        env: &Arc<R::Env>,
        resolved_memo: &mut HashMap<(RuleResultRef<R>, Arc<R::Env>), ActiveAnswerMatch>,
    ) -> Result<ActiveAnswerMatch, AnswerSupportBuildError> {
        let key = (result_ref, Arc::clone(env));
        if let Some(result) = resolved_memo.get(&key).copied() {
            return Ok(result);
        }

        let support = self.graph_answer_support(result_ref)?;
        let result = if self.answer_support_matches_env(result_ref, &support, env) {
            ActiveAnswerMatch::Matches
        } else {
            ActiveAnswerMatch::Mismatch
        };

        resolved_memo.insert(key, result);
        Ok(result)
    }

    #[instrumented(
        name = "solver.update_blocked_cross_env_reuses",
        target = "inlay",
        level = "trace",
        ret,
        skip(self),
        fields(
            dfn = dfn.index() as u64,
            cross_env_reuses,
            blocked_total
        )
    )]
    fn update_blocked_cross_env_reuses_in_suffix(
        &mut self,
        dfn: crate::search_graph::DepthFirstNumber,
    ) -> Result<bool, AnswerSupportBuildError> {
        let mut resolved_memo = HashMap::default();
        let mut blocked_grew = false;
        let cross_env_reuses = self.search_graph.suffix_cross_env_reuses(dfn);
        inlay_span_record!(cross_env_reuses = cross_env_reuses.len() as u64);

        for (result_ref, env) in cross_env_reuses {
            let blocked_key = (result_ref, Arc::clone(&env));
            if self.blocked_cross_env_reuses.contains(&blocked_key) {
                continue;
            }

            if self.answer_matches_env_for_backref(result_ref, &env, &mut resolved_memo)?
                == ActiveAnswerMatch::Mismatch
            {
                blocked_grew |= self.blocked_cross_env_reuses.insert(blocked_key);
            }
        }

        inlay_span_record!(blocked_total = self.blocked_cross_env_reuses.len() as u64);
        Ok(blocked_grew)
    }
}

type SuffixSnapshot<R> = HashMap<RuleResultRef<R>, RuleResult<R>>;

#[cfg(feature = "tracing")]
pub(crate) fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut hasher = FxHasher::default();
    value.hash(&mut hasher);
    hasher.finish()
}

#[cfg(feature = "tracing")]
pub(crate) fn debug_env_hash<R: Rule>(env: &R::Env) -> u64 {
    hash_value(env)
}

impl<R: Rule> Solver<R> {
    fn snapshot_suffix(&self, dfn: crate::search_graph::DepthFirstNumber) -> SuffixSnapshot<R> {
        self.search_graph
            .suffix_result_refs(dfn)
            .into_iter()
            .map(|result_ref| {
                let result = self
                    .results_arena
                    .get(&result_ref)
                    .cloned()
                    .expect("solver-managed suffix node must have a stored result");
                (result_ref, result)
            })
            .collect()
    }

    #[instrumented(
        name = "solver.evaluate_goal",
        target = "inlay",
        level = "trace",
        skip(self),
        fields(
            dfn = dfn.index() as u64,
            query_hash = hash_value(&self.search_graph[dfn].goal.query),
            env_hash = debug_env_hash::<R>(Arc::as_ref(&self.search_graph[dfn].goal.env)),
            state_hash = hash_value(&self.search_graph[dfn].goal.state_id),
            lazy_depth = self.search_graph[dfn].goal.lazy_depth.0 as u64
        )
    )]
    fn evaluate_goal_once(
        &mut self,
        dfn: crate::search_graph::DepthFirstNumber,
    ) -> Result<Minimums, SolveError> {
        let goal = self.search_graph[dfn].goal.clone();

        let mut minimums = Minimums::new();
        let (direct_supports, raw_lookup_support_count, dependencies, cross_env_reuses, result_ref) = {
            let rule = Arc::clone(&self.rule);
            let mut rule_ctx =
                crate::rule::RuleContext::new(self, goal.state_id, goal.env, dfn, &mut minimums);

            let result = inlay_in_span!("solver.rule_run", {}, {
                match rule.run(goal.query, &mut rule_ctx) {
                    Ok(output) => Ok(output),
                    Err(crate::rule::RunError::Rule(err)) => Err(err),
                    Err(crate::rule::RunError::Solve(error)) => return Err(error),
                }
            });
            let raw_lookup_support_count = rule_ctx.lookup_supports.len() as u64;
            let direct_supports =
                compact_lookup_supports::<R>(std::mem::take(&mut rule_ctx.lookup_supports));
            let dependencies: Vec<Dependency<R>> =
                rule_ctx.child_dependencies.iter().cloned().collect();
            let cross_env_reuses: Vec<(RuleResultRef<R>, Arc<R::Env>)> =
                rule_ctx.cross_env_reuses.iter().cloned().collect();
            let result_ref = rule_ctx.solver.search_graph[dfn].answer.result_ref;

            rule_ctx.solver.replace_result(result_ref, result);
            (
                direct_supports,
                raw_lookup_support_count,
                dependencies,
                cross_env_reuses,
                result_ref,
            )
        };

        self.search_graph[dfn].cross_env_reuses = cross_env_reuses;
        let direct_support_count = direct_supports.len() as u64;
        let dependency_count = dependencies.len() as u64;
        let cross_env_reuse_count = self.search_graph[dfn].cross_env_reuses.len() as u64;
        self.store_graph_answer(
            dfn,
            crate::search_graph::Answer {
                result_ref,
                direct_supports,
                dependencies,
            },
        );
        let result_kind = match self
            .results_arena
            .get(&result_ref)
            .expect("result just stored")
        {
            Ok(_) => "ok",
            Err(_) => "err",
        };
        inlay_event!(
            name: "solver.goal_eval",
            ?result_ref,
            lookup_supports_raw = raw_lookup_support_count,
            direct_supports = direct_support_count,
            dependencies = dependency_count,
            cross_env_reuses = cross_env_reuse_count,
            result_kind
        );
        self.search_graph[dfn].links = minimums;
        Ok(minimums)
    }

    #[instrumented(
        name = "solver.new_goal",
        target = "inlay",
        level = "trace",
        skip(self),
        fields(
            query_hash = hash_value(&goal.query),
            env_hash = debug_env_hash::<R>(Arc::as_ref(&goal.env)),
            state_hash = hash_value(&goal.state_id),
            lazy_depth = goal.lazy_depth.0 as u64
        )
    )]
    fn solve_new_goal(
        &mut self,
        goal: GoalKey<R>,
    ) -> Result<(GoalSolveResult<R>, Minimums), SolveError> {
        let (dfn, result_ref, final_minimums) =
            self.call_on_stack(&goal, |solver, dfn, stack_depth, result_ref| {
                let mut reruns: usize = 0;
                let mut previous_snapshot = None;
                let final_minimums = loop {
                    let iteration_minimums = solver.evaluate_goal_once(dfn)?;

                    if !solver.stack[stack_depth].read_and_reset_cycle_flag() {
                        break iteration_minimums;
                    }

                    let blocked_grew = solver.update_blocked_cross_env_reuses_in_suffix(dfn)?;
                    let current_snapshot = solver.snapshot_suffix(dfn);
                    let snapshot_unchanged = previous_snapshot
                        .as_ref()
                        .is_some_and(|previous| previous == &current_snapshot);
                    if snapshot_unchanged && !blocked_grew {
                        break iteration_minimums;
                    }

                    if reruns >= solver.fixpoint_iteration_limit {
                        return Err(SolveError::FixpointIterationLimitReached);
                    }
                    reruns += 1;
                    inlay_event!(
                        name: "solver.fixpoint_rerun",
                        dfn = dfn.index() as u64,
                        ?result_ref,
                        rerun = reruns,
                        blocked_grew,
                        snapshot_unchanged
                    );
                    previous_snapshot = Some(current_snapshot);

                    solver.search_graph.rollback_to(dfn + 1);
                };

                Ok(final_minimums)
            })?;

        // check if every child does not depend on any nodes higher than current in search graph
        if final_minimums.ancestor() >= dfn {
            for entry in self.search_graph.take_cacheable_entries(dfn) {
                self.cache.insert_entry(entry);
            }
        }

        Ok((GoalSolveResult::Resolved { result_ref }, final_minimums))
    }

    fn try_close_active_cycle(
        &mut self,
        goal: &GoalKey<R>,
    ) -> Result<Option<(GoalSolveResult<R>, Minimums)>, SolveError> {
        let Some(ancestor_dfn) =
            self.search_graph
                .closest_goal(&goal.query, goal.state_id, &goal.env)
        else {
            return Ok(None);
        };

        let ancestor_node = &self.search_graph[ancestor_dfn];
        let (ancestor_lazy_depth, result_ref, stack_depth) = (
            ancestor_node.goal.lazy_depth,
            ancestor_node.answer.result_ref,
            ancestor_node
                .stack_depth
                .expect("closest active goal must still be on stack"),
        );

        if ancestor_lazy_depth >= goal.lazy_depth {
            // do not flag cycle on stack: same depth cycles are not interesting for fixpoint
            // iterations since they do not depend on provisional result
            return Err(SolveError::SameDepthCycle);
        }

        self.stack[stack_depth].flag_cycle();
        Ok(Some((
            GoalSolveResult::Lazy { result_ref },
            Minimums::from_self(ancestor_dfn),
        )))
    }

    fn try_close_cross_env_active_cycle(
        &mut self,
        goal: &GoalKey<R>,
    ) -> Option<(GoalSolveResult<R>, Minimums)> {
        let ancestor_dfn = self
            .search_graph
            .closest_goal_any_env(&goal.query, goal.state_id)?;
        let (ancestor_lazy_depth, result_ref, ancestor_env, stack_depth) = {
            let ancestor_node = &self.search_graph[ancestor_dfn];
            (
                ancestor_node.goal.lazy_depth,
                ancestor_node.answer.result_ref,
                Arc::clone(&ancestor_node.goal.env),
                ancestor_node
                    .stack_depth
                    .expect("closest active goal must still be on stack"),
            )
        };

        if ancestor_env == goal.env || ancestor_lazy_depth >= goal.lazy_depth {
            return None;
        }

        let blocked_key = (result_ref, Arc::clone(&goal.env));
        if self.blocked_cross_env_reuses.contains(&blocked_key) {
            return None;
        }

        self.stack[stack_depth].flag_cycle();
        Some((
            GoalSolveResult::LazyCrossEnv { result_ref },
            Minimums::from_self(ancestor_dfn),
        ))
    }

    #[instrumented(
        name = "solver.try_cache_reuse",
        target = "inlay",
        level = "trace",
        ret,
        skip(self),
        fields(
            query_hash = hash_value(&goal.query),
            env_hash = debug_env_hash::<R>(Arc::as_ref(&goal.env)),
            state_hash = hash_value(&goal.state_id),
            lazy_depth = goal.lazy_depth.0 as u64
        )
    )]
    fn try_cache_reuse(&mut self, goal: &GoalKey<R>) -> Option<(GoalSolveResult<R>, Minimums)> {
        if let Some(result_ref) = self.cache.get_same_env_result(goal) {
            inlay_event!(
                name: "solver.cache_probe",
                exact_env = true,
                bucket_len = 1_u64
            );
            return Some((
                GoalSolveResult::Resolved {
                    result_ref: result_ref.result_ref(),
                },
                Minimums::new(),
            ));
        }

        let candidates = self.cache.get_result_candidates(goal);
        if candidates.is_empty() {
            return None;
        }

        inlay_event!(
            name: "solver.cache_probe",
            exact_env = false,
            bucket_len = candidates.len() as u64
        );
        for result_ref in candidates {
            let matched = self.answer_matches_env(result_ref, &goal.env);
            if matched {
                return Some((
                    GoalSolveResult::Resolved {
                        result_ref: result_ref.result_ref(),
                    },
                    Minimums::new(),
                ));
            }
        }

        None
    }

    fn try_graph_goal_reuse(&self, goal: &GoalKey<R>) -> Option<(GoalSolveResult<R>, Minimums)> {
        let dfn = self.search_graph.lookup(goal)?;
        let node = &self.search_graph[dfn];
        debug_assert!(
            node.stack_depth.is_none(),
            "on-stack goal from search grapth is never be reused by try_graph_goal_reuse"
        );
        Some((
            GoalSolveResult::Resolved {
                result_ref: node.answer.result_ref,
            },
            node.links,
        ))
    }
}
