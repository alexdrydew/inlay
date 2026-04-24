#![cfg_attr(not(feature = "tracing"), allow(unused_variables, unused_assignments))]

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use inlay_instrument_macros::instrumented;
use thiserror::Error;

use crate::{
    arena::Arena,
    cache::{cache_key, insert_cache_entries},
    context::{AnswerMatchMemo, Context},
    instrument::{solver_event, solver_in_span, solver_span_record},
    rule::{
        Lookups, ResolutionEnv, Rule, RuleEnvSharedState, RuleLookupQuery, RuleLookupResult,
        RuleQuery, RuleResult, RuleResultRef, RuleResultsArena,
    },
    search_graph::{Dependency, GoalKey, LazyDepth, Minimums},
    stack::StackError,
};

#[cfg(feature = "tracing")]
use crate::instrument::solver_trace_enabled;

pub(crate) enum GoalSolveResult<R: Rule> {
    Resolved { result_ref: RuleResultRef<R> },
    Lazy { result_ref: RuleResultRef<R> },
    LazyCrossEnv { result_ref: RuleResultRef<R> },
}

impl<R: Rule> std::fmt::Debug for GoalSolveResult<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resolved { result_ref } => f
                .debug_struct("Resolved")
                .field("result_ref", result_ref)
                .finish(),
            Self::Lazy { result_ref } => f
                .debug_struct("Lazy")
                .field("result_ref", result_ref)
                .finish(),
            Self::LazyCrossEnv { result_ref } => f
                .debug_struct("LazyCrossEnv")
                .field("result_ref", result_ref)
                .finish(),
        }
    }
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

pub(crate) fn debug_lookup_query_label<R: Rule>(rule: &R, query: &RuleLookupQuery<R>) -> String {
    rule.debug_lookup_query_label(query)
        .unwrap_or_else(|| format!("lookup={:x}", hash_value(query)))
}

pub(crate) fn debug_lookup_result_label<R: Rule>(rule: &R, result: &RuleLookupResult<R>) -> String {
    rule.debug_lookup_result_label(result)
        .unwrap_or_else(|| format!("result={:x}", hash_value(result)))
}

fn debug_result_query_label<R: Rule>(
    rule: &R,
    result_ref: RuleResultRef<R>,
    ctx: &Context<R>,
) -> String {
    ctx.goal_for_result_ref(result_ref)
        .map(|goal| debug_cache_key_label::<R>(rule, &goal.query, goal.state_id))
        .unwrap_or_else(|| format!("result_ref={:?}", result_ref))
}

#[instrumented(
    name = "solver.lookups_match_env",
    target = "context_solver",
    level = "trace",
    ret,
    fields(
        lookups = lookups.len() as u64,
        env_hash = debug_env_hash::<R>(Arc::as_ref(env)),
        env_label = %trace_env_label::<R>(rule, Arc::as_ref(env))
    )
)]
fn lookups_match_env<R: Rule>(
    rule: &R,
    lookups: &Lookups<R>,
    env: &Arc<R::Env>,
    ctx: &mut Context<R>,
) -> bool {
    let mut mismatch = None;
    for (query, expected_result) in lookups {
        let actual_result = env.lookup(&mut ctx.shared_state, query);
        if actual_result != *expected_result {
            mismatch = Some((query.clone(), expected_result.clone(), actual_result));
            break;
        }
    }
    let Some((query, expected_result, actual_result)) = mismatch else {
        return true;
    };

    #[cfg(feature = "tracing")]
    {
        let trace_enabled = solver_trace_enabled!();
        let lookup_label = trace_enabled
            .then(|| debug_lookup_query_label(rule, &query))
            .unwrap_or_default();
        let expected_label = trace_enabled
            .then(|| debug_lookup_result_label(rule, &expected_result))
            .unwrap_or_default();
        let actual_label = trace_enabled
            .then(|| debug_lookup_result_label(rule, &actual_result))
            .unwrap_or_default();
        solver_event!(
            name: "solver.cache_lookup_miss",
            lookup_hash = hash_value(&query),
            lookup_label = lookup_label.as_str(),
            expected_hash = hash_value(&expected_result),
            expected_label = expected_label.as_str(),
            actual_hash = hash_value(&actual_result),
            actual_label = actual_label.as_str()
        );
    }

    false
}

#[instrumented(
    name = "solver.answer_match",
    target = "context_solver",
    level = "trace",
    ret,
    fields(
        result_ref = ?result_ref,
        env_hash = debug_env_hash::<R>(Arc::as_ref(env)),
        depth = depth as u64,
        result_query = %trace_result_query_label::<R>(rule, result_ref, ctx),
        env_label = %trace_env_label::<R>(rule, Arc::as_ref(env))
    )
)]
fn answer_matches_env<R: Rule>(
    rule: &R,
    result_ref: RuleResultRef<R>,
    env: &Arc<R::Env>,
    ctx: &mut Context<R>,
    depth: usize,
) -> bool {
    match ctx
        .answer_match_memo
        .get(&(result_ref, Arc::clone(env)))
        .copied()
    {
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

    ctx.answer_match_memo
        .insert((result_ref, Arc::clone(env)), AnswerMatchMemo::InProgress);

    let Some(answer) = ctx.answer_for(result_ref).cloned() else {
        solver_event!(name: "solver.cache_missing_answer");
        ctx.answer_match_memo.insert(
            (result_ref, Arc::clone(env)),
            AnswerMatchMemo::Resolved(false),
        );
        return false;
    };

    let matches = if !lookups_match_env(rule, &answer.lookups, env, ctx) {
        false
    } else {
        let mut matches = true;
        for dependency in &answer.dependencies {
            let dependency_env = ctx.rebase_env_for_dependency(env, &dependency.env_delta);
            if !answer_matches_env(rule, dependency.result_ref, &dependency_env, ctx, depth + 1) {
                solver_event!(
                    name: "solver.cache_dependency_miss",
                    dependency_result_ref = ?dependency.result_ref
                );
                matches = false;
                break;
            }
        }
        matches
    };
    ctx.answer_match_memo.insert(
        (result_ref, Arc::clone(env)),
        AnswerMatchMemo::Resolved(matches),
    );
    matches
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActiveAnswerMatch {
    Matches,
    Mismatch,
    Unknown,
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
    let mut in_progress = HashSet::new();
    answer_matches_env_for_backref_inner(
        rule,
        result_ref,
        env,
        ctx,
        resolved_memo,
        &mut in_progress,
    )
}

fn answer_matches_env_for_backref_inner<R: Rule>(
    rule: &R,
    result_ref: RuleResultRef<R>,
    env: &Arc<R::Env>,
    ctx: &mut Context<R>,
    resolved_memo: &mut HashMap<(RuleResultRef<R>, Arc<R::Env>), ActiveAnswerMatch>,
    in_progress: &mut HashSet<(RuleResultRef<R>, Arc<R::Env>)>,
) -> ActiveAnswerMatch {
    let key = (result_ref, Arc::clone(env));
    if let Some(result) = resolved_memo.get(&key).copied() {
        return result;
    }

    if !in_progress.insert(key.clone()) {
        return ActiveAnswerMatch::Matches;
    }

    let result = if let Some(answer) = ctx.answer_for(result_ref).cloned() {
        if !lookups_match_env(rule, &answer.lookups, env, ctx) {
            ActiveAnswerMatch::Mismatch
        } else {
            let mut saw_unknown = false;
            let mut result = ActiveAnswerMatch::Matches;
            for dependency in answer.dependencies {
                let dependency_env = ctx.rebase_env_for_dependency(env, &dependency.env_delta);
                match answer_matches_env_for_backref_inner(
                    rule,
                    dependency.result_ref,
                    &dependency_env,
                    ctx,
                    resolved_memo,
                    in_progress,
                ) {
                    ActiveAnswerMatch::Matches => {}
                    ActiveAnswerMatch::Mismatch => {
                        result = ActiveAnswerMatch::Mismatch;
                        break;
                    }
                    ActiveAnswerMatch::Unknown => saw_unknown = true,
                }
            }

            if result == ActiveAnswerMatch::Mismatch {
                ActiveAnswerMatch::Mismatch
            } else if saw_unknown {
                ActiveAnswerMatch::Unknown
            } else {
                ActiveAnswerMatch::Matches
            }
        }
    } else {
        ActiveAnswerMatch::Unknown
    };

    in_progress.remove(&key);
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

pub(crate) fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

pub(crate) fn debug_env_hash<R: Rule>(env: &R::Env) -> u64 {
    hash_value(env)
}

pub(crate) fn debug_env_label<R: Rule>(rule: &R, env: &R::Env) -> String {
    rule.debug_env_label(env)
        .unwrap_or_else(|| format!("env={:x}", debug_env_hash::<R>(env)))
}

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
    solver_trace_enabled!()
        .then(|| debug_cache_key_label::<R>(rule, query, state_id))
        .unwrap_or_default()
}

#[cfg(feature = "tracing")]
fn trace_env_label<R: Rule>(rule: &R, env: &R::Env) -> String {
    solver_trace_enabled!()
        .then(|| debug_env_label(rule, env))
        .unwrap_or_default()
}

#[cfg(feature = "tracing")]
fn trace_result_query_label<R: Rule>(
    rule: &R,
    result_ref: RuleResultRef<R>,
    ctx: &Context<R>,
) -> String {
    solver_trace_enabled!()
        .then(|| debug_result_query_label(rule, result_ref, ctx))
        .unwrap_or_default()
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
    let (lookups, dependencies, cross_env_reuses, result_ref) = {
        let mut rule_ctx =
            crate::rule::RuleContext::new(rule, goal.state_id, goal.env, ctx, dfn, &mut minimums);

        let result = solver_in_span!("solver.rule_run", {}, {
            match rule.run(goal.query, &mut rule_ctx) {
                Ok(output) => Ok(output),
                Err(crate::rule::RunError::Rule(err)) => Err(err),
                Err(crate::rule::RunError::Solve(error)) => return Err(error),
            }
        });
        let lookups = rule_ctx.lookups.clone();
        let dependencies: Vec<Dependency<R>> =
            rule_ctx.child_dependencies.iter().cloned().collect();
        let cross_env_reuses: Vec<(RuleResultRef<R>, Arc<R::Env>)> =
            rule_ctx.cross_env_reuses.iter().cloned().collect();
        let result_ref = rule_ctx.ctx.search_graph[dfn].answer.result_ref;

        replace_result(rule_ctx.ctx, result_ref, result);
        (lookups, dependencies, cross_env_reuses, result_ref)
    };

    ctx.search_graph[dfn].answer.lookups = lookups.clone();
    ctx.search_graph[dfn].answer.dependencies = dependencies.clone();
    ctx.search_graph[dfn].cross_env_reuses = cross_env_reuses;
    let lookup_count = lookups.len() as u64;
    let dependency_count = dependencies.len() as u64;
    let cross_env_reuse_count = ctx.search_graph[dfn].cross_env_reuses.len() as u64;
    ctx.replace_answer(
        result_ref,
        crate::search_graph::Answer {
            result_ref,
            lookups,
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
        lookups = lookup_count,
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
    let result_ref = ctx.result_ref_for(&goal);
    let (dfn, final_minimums) = ctx.call_on_stack(&goal, result_ref, |ctx, dfn, stack_depth| {
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
        let cacheable_entries = ctx.search_graph.take_cacheable_entries(dfn);
        insert_cache_entries(ctx, cacheable_entries);
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

#[cfg(feature = "cross-env-active-reuse")]
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
    if !ctx.cache_reuse_enabled {
        return None;
    }

    let key = cache_key(goal);
    let (exact_result_refs, bucket_len, entries) = ctx.cache.get(&key).map(|bucket| {
        (
            bucket.cloned_result_refs_for_env(&goal.env),
            bucket.len(),
            bucket.cloned_entries(),
        )
    })?;

    if let Some(result_refs) = exact_result_refs.as_ref() {
        solver_event!(
            name: "solver.cache_probe",
            exact_env = true,
            bucket_len = result_refs.len() as u64
        );
        for result_ref in result_refs.iter().rev() {
            let matched = answer_matches_env(rule, *result_ref, &goal.env, ctx, 0);
            if matched {
                return Some((
                    GoalSolveResult::Resolved {
                        result_ref: *result_ref,
                    },
                    Minimums::new(),
                ));
            }
        }
    }

    solver_event!(
        name: "solver.cache_probe",
        exact_env = false,
        bucket_len = bucket_len as u64
    );
    for entry in entries.iter().rev() {
        if exact_result_refs.is_some() && entry.env == goal.env {
            continue;
        }
        let matched = answer_matches_env(rule, entry.result_ref, &goal.env, ctx, 0);
        if matched {
            return Some((
                GoalSolveResult::Resolved {
                    result_ref: entry.result_ref,
                },
                Minimums::new(),
            ));
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

    #[cfg(feature = "cross-env-active-reuse")]
    {
        if let Some(result) = try_close_cross_env_active_cycle(&goal, ctx) {
            return Ok(result);
        }
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
