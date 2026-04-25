#![cfg_attr(not(feature = "tracing"), allow(unused_variables, unused_assignments))]

use std::collections::HashMap;
#[cfg(feature = "tracing")]
use std::collections::hash_map::DefaultHasher;
#[cfg(feature = "tracing")]
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use derive_where::derive_where;
use inlay_instrument_macros::instrumented;
use thiserror::Error;

use crate::{
    cache::cache_key,
    context::{AnswerMatchMemo, Context},
    instrument::{solver_event, solver_in_span, solver_span_record},
    lookup_support::{answer_support_matches_env, compact_lookup_supports},
    rule::{
        RuleEnvSharedState, RuleQuery, RuleResult, RuleResultRef, RuleResultsArena,
    },
    search_graph::{Dependency, GoalKey, LazyDepth, Minimums},
    stack::StackError,
    traits::{Arena, Rule},
};

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

pub struct SolveOutcome<R: Rule> {
    pub shared_state: RuleEnvSharedState<R>,
    pub result: Result<(RuleResultRef<R>, RuleResultsArena<R>), SolveError>,
}

impl<R: Rule> std::fmt::Debug for SolveOutcome<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.result {
            Ok((result_ref, results_arena)) => f
                .debug_struct("SolveOutcome")
                .field("ok", &true)
                .field("result_ref", result_ref)
                .field("results", &results_arena.len())
                .finish(),
            Err(error) => f
                .debug_struct("SolveOutcome")
                .field("ok", &false)
                .field("error", error)
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
}

impl From<StackError> for SolveError {
    fn from(error: StackError) -> Self {
        match error {
            StackError::Overflow => Self::StackOverflowDepthReached,
        }
    }
}

fn replace_result<R: Rule>(
    ctx: &mut Context<R>,
    result_ref: RuleResultRef<R>,
    result: RuleResult<R>,
) {
    ctx.results_arena
        .replace(result_ref, result)
        .expect("solver-managed result ref must remain valid");
}

#[cfg(feature = "tracing")]
pub(crate) fn debug_lookup_query_label<R: Rule>(
    rule: &R,
    query: &crate::rule::RuleLookupQuery<R>,
) -> String {
    rule.debug_lookup_query_label(query)
        .unwrap_or_else(|| format!("lookup={:x}", hash_value(query)))
}

#[cfg(feature = "tracing")]
pub(crate) fn debug_lookup_result_label<R: Rule>(
    rule: &R,
    result: &crate::rule::RuleLookupResult<R>,
) -> String {
    rule.debug_lookup_result_label(result)
        .unwrap_or_else(|| format!("result={:x}", hash_value(result)))
}

#[cfg(feature = "tracing")]
fn debug_result_query_label<R: Rule>(
    rule: &R,
    result_ref: RuleResultRef<R>,
    ctx: &Context<R>,
) -> String {
    ctx.search_graph
        .goal_for_result_ref(result_ref)
        .map(|goal| debug_cache_key_label::<R>(rule, &goal.query, goal.state_id))
        .unwrap_or_else(|| format!("result_ref={:?}", result_ref))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActiveAnswerMatch {
    Matches,
    Mismatch,
    Unknown,
}

#[instrumented(
    name = "solver.answer_match",
    target = "context_solver",
    level = "trace",
    ret,
    fields(
        result_ref = ?result_ref,
        env_hash = debug_env_hash::<R>(Arc::as_ref(env)),
        result_query = %trace_result_query_label::<R>(rule, result_ref, ctx),
        env_label = %trace_env_label::<R>(rule, Arc::as_ref(env))
    )
)]
fn answer_matches_env<R: Rule>(
    rule: &R,
    result_ref: RuleResultRef<R>,
    env: &Arc<R::Env>,
    ctx: &mut Context<R>,
) -> bool {
    match ctx.answer_match_memo(result_ref, env) {
        Some(AnswerMatchMemo::Resolved(matches)) => {
            solver_event!(
                name: "solver.answer_match_memo",
                matched = matches
            );
            return matches;
        }
        Some(AnswerMatchMemo::InProgress) => {
            solver_event!(name: "solver.answer_match_in_progress");
            return true;
        }
        None => {}
    }

    ctx.insert_answer_match_memo(result_ref, env, AnswerMatchMemo::InProgress);

    let matches = match ctx.cached_answer_supports(result_ref) {
        Some(supports) => {
            let mut matched = false;
            for support in supports {
                if answer_support_matches_env(rule, result_ref, &support, env, ctx) {
                    matched = true;
                    break;
                }
            }
            matched
        }
        None => {
            solver_event!(name: "solver.cache_missing_answer");
            false
        }
    };
    ctx.insert_answer_match_memo(result_ref, env, AnswerMatchMemo::Resolved(matches));
    matches
}

#[instrumented(
    name = "solver.answer_matches_env_for_backref",
    target = "context_solver",
    level = "trace",
    ret,
    fields(
        result_ref = ?result_ref,
        env_hash = debug_env_hash::<R>(Arc::as_ref(env)),
        result_query = %trace_result_query_label::<R>(rule, result_ref, ctx),
        env_label = %trace_env_label::<R>(rule, Arc::as_ref(env))
    )
)]
fn answer_matches_env_for_backref<R: Rule>(
    rule: &R,
    result_ref: RuleResultRef<R>,
    env: &Arc<R::Env>,
    ctx: &mut Context<R>,
    resolved_memo: &mut HashMap<(RuleResultRef<R>, Arc<R::Env>), ActiveAnswerMatch>,
) -> ActiveAnswerMatch {
    let key = (result_ref, Arc::clone(env));
    if let Some(result) = resolved_memo.get(&key).copied() {
        return result;
    }

    let result = match ctx.graph_answer_support(result_ref) {
        Some(support) => {
            if answer_support_matches_env(rule, result_ref, &support, env, ctx) {
                ActiveAnswerMatch::Matches
            } else {
                ActiveAnswerMatch::Mismatch
            }
        }
        None => ActiveAnswerMatch::Unknown,
    };

    resolved_memo.insert(key, result);
    result
}

#[instrumented(
    name = "solver.update_blocked_cross_env_reuses",
    target = "context_solver",
    level = "trace",
    ret,
    fields(
        dfn = dfn.index() as u64,
        cross_env_reuses,
        blocked_total
    )
)]
fn update_blocked_cross_env_reuses_in_suffix<R: Rule>(
    rule: &R,
    dfn: crate::search_graph::DepthFirstNumber,
    ctx: &mut Context<R>,
) -> bool {
    let mut resolved_memo = HashMap::new();
    let mut blocked_grew = false;
    let cross_env_reuses = ctx.search_graph.suffix_cross_env_reuses(dfn);
    solver_span_record!(cross_env_reuses = cross_env_reuses.len() as u64);

    for (result_ref, env) in cross_env_reuses {
        let blocked_key = (result_ref, Arc::clone(&env));
        if ctx.blocked_cross_env_reuses.contains(&blocked_key) {
            continue;
        }

        if answer_matches_env_for_backref(rule, result_ref, &env, ctx, &mut resolved_memo)
            == ActiveAnswerMatch::Mismatch
        {
            blocked_grew |= ctx.blocked_cross_env_reuses.insert(blocked_key);
        }
    }

    solver_span_record!(blocked_total = ctx.blocked_cross_env_reuses.len() as u64);
    blocked_grew
}

type SuffixSnapshot<R> = HashMap<RuleResultRef<R>, RuleResult<R>>;

#[cfg(feature = "tracing")]
pub(crate) fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[cfg(feature = "tracing")]
pub(crate) fn debug_env_hash<R: Rule>(env: &R::Env) -> u64 {
    hash_value(env)
}

#[cfg(feature = "tracing")]
pub(crate) fn debug_env_label<R: Rule>(rule: &R, env: &R::Env) -> String {
    rule.debug_env_label(env)
        .unwrap_or_else(|| format!("env={:x}", debug_env_hash::<R>(env)))
}

#[cfg(feature = "tracing")]
fn debug_cache_key_label<R: Rule>(
    rule: &R,
    query: &RuleQuery<R>,
    state_id: R::RuleStateId,
) -> String {
    if let Some(label) = rule.debug_query_label(query, state_id) {
        return label;
    }

    let mut hasher = DefaultHasher::new();
    query.hash(&mut hasher);
    let query_hash = hasher.finish();

    let mut hasher = DefaultHasher::new();
    state_id.hash(&mut hasher);
    let state_hash = hasher.finish();

    format!("query={query_hash:x}#state={state_hash:x}")
}

#[cfg(feature = "tracing")]
fn trace_query_label<R: Rule>(rule: &R, query: &RuleQuery<R>, state_id: R::RuleStateId) -> String {
    debug_cache_key_label::<R>(rule, query, state_id)
}

#[cfg(feature = "tracing")]
pub(crate) fn trace_env_label<R: Rule>(rule: &R, env: &R::Env) -> String {
    debug_env_label(rule, env)
}

#[cfg(feature = "tracing")]
pub(crate) fn trace_result_query_label<R: Rule>(
    rule: &R,
    result_ref: RuleResultRef<R>,
    ctx: &Context<R>,
) -> String {
    debug_result_query_label(rule, result_ref, ctx)
}

fn snapshot_suffix<R: Rule>(
    ctx: &Context<R>,
    dfn: crate::search_graph::DepthFirstNumber,
) -> SuffixSnapshot<R> {
    ctx.search_graph
        .suffix_result_refs(dfn)
        .into_iter()
        .map(|result_ref| {
            let result = ctx
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
    target = "context_solver",
    level = "trace",
    fields(
        dfn = dfn.index() as u64,
        query_hash = hash_value(&ctx.search_graph[dfn].goal.query),
        env_hash = debug_env_hash::<R>(Arc::as_ref(&ctx.search_graph[dfn].goal.env)),
        lazy_depth = ctx.search_graph[dfn].goal.lazy_depth.0 as u64,
        query_label = %trace_query_label::<R>(
            rule,
            &ctx.search_graph[dfn].goal.query,
            ctx.search_graph[dfn].goal.state_id,
        ),
        env_label = %trace_env_label::<R>(rule, Arc::as_ref(&ctx.search_graph[dfn].goal.env))
    )
)]
fn evaluate_goal_once<R: Rule>(
    rule: &R,
    dfn: crate::search_graph::DepthFirstNumber,
    ctx: &mut Context<R>,
) -> Result<Minimums, SolveError> {
    let goal = ctx.search_graph[dfn].goal.clone();

    let mut minimums = Minimums::new();
    let (direct_supports, raw_lookup_support_count, dependencies, cross_env_reuses, result_ref) = {
        let mut rule_ctx =
            crate::rule::RuleContext::new(rule, goal.state_id, goal.env, ctx, dfn, &mut minimums);

        let result = solver_in_span!("solver.rule_run", {}, {
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
        let result_ref = rule_ctx.ctx.search_graph[dfn].answer.result_ref;

        replace_result(rule_ctx.ctx, result_ref, result);
        (
            direct_supports,
            raw_lookup_support_count,
            dependencies,
            cross_env_reuses,
            result_ref,
        )
    };

    ctx.search_graph[dfn].cross_env_reuses = cross_env_reuses;
    let direct_support_count = direct_supports.len() as u64;
    let dependency_count = dependencies.len() as u64;
    let cross_env_reuse_count = ctx.search_graph[dfn].cross_env_reuses.len() as u64;
    ctx.store_graph_answer(
        dfn,
        crate::search_graph::Answer {
            result_ref,
            direct_supports,
            dependencies,
        },
    );
    let result_kind = match ctx
        .results_arena
        .get(&result_ref)
        .expect("result just stored")
    {
        Ok(_) => "ok",
        Err(_) => "err",
    };
    solver_event!(
        name: "solver.goal_eval",
        ?result_ref,
        lookup_supports_raw = raw_lookup_support_count,
        direct_supports = direct_support_count,
        dependencies = dependency_count,
        cross_env_reuses = cross_env_reuse_count,
        result_kind
    );
    ctx.search_graph[dfn].links = minimums;
    Ok(minimums)
}

#[instrumented(
    name = "solver.new_goal",
    target = "context_solver",
    level = "trace",
    fields(
        query_hash = hash_value(&goal.query),
        env_hash = debug_env_hash::<R>(Arc::as_ref(&goal.env)),
        lazy_depth = goal.lazy_depth.0 as u64,
        query_label = %trace_query_label::<R>(rule, &goal.query, goal.state_id),
        env_label = %trace_env_label::<R>(rule, Arc::as_ref(&goal.env))
    )
)]
fn solve_new_goal<R: Rule>(
    rule: &R,
    goal: GoalKey<R>,
    ctx: &mut Context<R>,
) -> Result<(GoalSolveResult<R>, Minimums), SolveError> {
    let (dfn, result_ref, final_minimums) = ctx.call_on_stack(&goal, |ctx, dfn, stack_depth, result_ref| {
        let mut reruns: usize = 0;
        let mut previous_snapshot = None;
        let final_minimums = loop {
            let iteration_minimums = evaluate_goal_once(rule, dfn, ctx)?;

            if !ctx.stack[stack_depth].read_and_reset_cycle_flag() {
                break iteration_minimums;
            }

            let blocked_grew = update_blocked_cross_env_reuses_in_suffix(rule, dfn, ctx);
            let current_snapshot = snapshot_suffix(ctx, dfn);
            let snapshot_unchanged = previous_snapshot
                .as_ref()
                .is_some_and(|previous| previous == &current_snapshot);
            if snapshot_unchanged && !blocked_grew {
                break iteration_minimums;
            }

            if reruns >= ctx.fixpoint_iteration_limit {
                return Err(SolveError::FixpointIterationLimitReached);
            }
            reruns += 1;
            solver_event!(
                name: "solver.fixpoint_rerun",
                dfn = dfn.index() as u64,
                ?result_ref,
                rerun = reruns,
                blocked_grew,
                snapshot_unchanged
            );
            previous_snapshot = Some(current_snapshot);

            ctx.search_graph.rollback_to(dfn + 1);
        };

        Ok(final_minimums)
    })?;

    // check if every child does not depend on any nodes higher than current in search graph
    if final_minimums.ancestor() >= dfn {
        for entry in ctx.search_graph.take_cacheable_entries(dfn) {
            ctx.cache.insert_entry(entry);
        }
    }

    Ok((GoalSolveResult::Resolved { result_ref }, final_minimums))
}

fn try_close_active_cycle<R: Rule>(
    goal: &GoalKey<R>,
    ctx: &mut Context<R>,
) -> Result<Option<(GoalSolveResult<R>, Minimums)>, SolveError> {
    let Some(ancestor_dfn) = ctx
        .search_graph
        .closest_goal(&goal.query, goal.state_id, &goal.env)
    else {
        return Ok(None);
    };

    let ancestor_node = &ctx.search_graph[ancestor_dfn];
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

    ctx.stack[stack_depth].flag_cycle();
    Ok(Some((
        GoalSolveResult::Lazy { result_ref },
        Minimums::from_self(ancestor_dfn),
    )))
}

fn try_close_cross_env_active_cycle<R: Rule>(
    goal: &GoalKey<R>,
    ctx: &mut Context<R>,
) -> Option<(GoalSolveResult<R>, Minimums)> {
    let ancestor_dfn = ctx
        .search_graph
        .closest_goal_any_env(&goal.query, goal.state_id)?;
    let (ancestor_lazy_depth, result_ref, ancestor_env, stack_depth) = {
        let ancestor_node = &ctx.search_graph[ancestor_dfn];
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
    if ctx.blocked_cross_env_reuses.contains(&blocked_key) {
        return None;
    }

    ctx.stack[stack_depth].flag_cycle();
    Some((
        GoalSolveResult::LazyCrossEnv { result_ref },
        Minimums::from_self(ancestor_dfn),
    ))
}

#[instrumented(
    name = "solver.try_cache_reuse",
    target = "context_solver",
    level = "trace",
    ret,
    fields(
        query_hash = hash_value(&goal.query),
        env_hash = debug_env_hash::<R>(Arc::as_ref(&goal.env)),
        state_hash = hash_value(&goal.state_id),
        lazy_depth = goal.lazy_depth.0 as u64,
        query_label = %trace_query_label::<R>(rule, &goal.query, goal.state_id),
        env_label = %trace_env_label::<R>(rule, Arc::as_ref(&goal.env))
    )
)]
fn try_cache_reuse<R: Rule>(
    rule: &R,
    goal: &GoalKey<R>,
    ctx: &mut Context<R>,
) -> Option<(GoalSolveResult<R>, Minimums)> {
    let key = cache_key(goal);
    let (exact_result_ref, bucket_len, result_refs) = ctx.cache.get(&key).map(|bucket| {
        (
            bucket.get(&goal.env).copied(),
            bucket.len(),
            bucket.values().copied().collect::<Vec<_>>(),
        )
    })?;

    if let Some(result_ref) = exact_result_ref {
        solver_event!(
            name: "solver.cache_probe",
            exact_env = true,
            bucket_len = 1_u64
        );
        let matched = answer_matches_env(rule, result_ref, &goal.env, ctx);
        if matched {
            return Some((GoalSolveResult::Resolved { result_ref }, Minimums::new()));
        }
    }

    solver_event!(
        name: "solver.cache_probe",
        exact_env = false,
        bucket_len = bucket_len.saturating_sub(usize::from(exact_result_ref.is_some())) as u64
    );
    for result_ref in result_refs {
        if Some(result_ref) == exact_result_ref {
            continue;
        }
        let matched = answer_matches_env(rule, result_ref, &goal.env, ctx);
        if matched {
            return Some((GoalSolveResult::Resolved { result_ref }, Minimums::new()));
        }
    }

    None
}

fn try_graph_goal_reuse<R: Rule>(
    goal: &GoalKey<R>,
    ctx: &mut Context<R>,
) -> Option<(GoalSolveResult<R>, Minimums)> {
    let dfn = ctx.search_graph.lookup(goal)?;
    let node = &ctx.search_graph[dfn];
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

#[instrumented(
    name = "solver.solve_goal",
    target = "context_solver",
    level = "trace",
    ret,
    err,
    fields(
        query_hash = hash_value(&goal.query),
        env_hash = debug_env_hash::<R>(Arc::as_ref(&goal.env)),
        state_hash = hash_value(&goal.state_id),
        lazy_depth = goal.lazy_depth.0 as u64,
        query_label = %trace_query_label::<R>(rule, &goal.query, goal.state_id),
        env_label = %trace_env_label::<R>(rule, Arc::as_ref(&goal.env))
    )
)]
pub(crate) fn solve_goal<R: Rule>(
    rule: &R,
    goal: GoalKey<R>,
    ctx: &mut Context<R>,
) -> Result<(GoalSolveResult<R>, Minimums), SolveError> {
    if let Some(result) = try_close_active_cycle(&goal, ctx)? {
        return Ok(result);
    }

    if let Some(result) = try_close_cross_env_active_cycle(&goal, ctx) {
        return Ok(result);
    }

    if let Some(result) = try_cache_reuse(rule, &goal, ctx) {
        return Ok(result);
    }

    if let Some(result) = try_graph_goal_reuse(&goal, ctx) {
        return Ok(result);
    }

    solve_new_goal(rule, goal, ctx)
}

#[instrumented(
    name = "solver.solve",
    target = "context_solver",
    level = "trace",
    ret,
    fields(
        query_hash = hash_value(&query),
        env_hash = debug_env_hash::<R>(Arc::as_ref(&env)),
        state_hash = hash_value(&initial_rule),
        lazy_depth = 0_u64,
        query_label = %trace_query_label::<R>(rule, &query, initial_rule),
        env_label = %trace_env_label::<R>(rule, Arc::as_ref(&env))
    )
)]
pub fn solve<R: Rule>(
    rule: &R,
    query: RuleQuery<R>,
    initial_rule: R::RuleStateId,
    env: Arc<R::Env>,
    shared_state: RuleEnvSharedState<R>,
    fixpoint_iteration_limit: usize,
    stack_depth_limit: usize,
) -> SolveOutcome<R> {
    let root_goal = GoalKey {
        query,
        state_id: initial_rule,
        env,
        lazy_depth: LazyDepth(0),
    };

    let mut ctx = Context::new(shared_state, fixpoint_iteration_limit, stack_depth_limit);
    let result = solve_goal(rule, root_goal, &mut ctx)
        .map(|(solve_result, _)| match solve_result {
            GoalSolveResult::Resolved { result_ref } => result_ref,
            GoalSolveResult::Lazy { .. } | GoalSolveResult::LazyCrossEnv { .. } => {
                unreachable!(
                    "root solve_goal cannot resolve lazily because lazy results require an active ancestor"
                )
            }
        });

    let Context {
        results_arena,
        shared_state,
        ..
    } = ctx;
    SolveOutcome {
        shared_state,
        result: result.map(|root_result_ref| (root_result_ref, results_arena)),
    }
}
