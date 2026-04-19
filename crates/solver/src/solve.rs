use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use thiserror::Error;

use crate::{
    arena::Arena,
    context::{AnswerMatchMemo, Context},
    rule::{
        Lookups, ResolutionEnv, Rule, RuleEnvSharedState, RuleQuery, RuleResult, RuleResultRef,
        RuleResultsArena,
    },
    search_graph::{CacheKey, GoalKey, LazyDepth, Minimums},
    stack::StackError,
};

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq, Hash)]
#[error("same depth cycle")]
pub struct SameDepthCycleError;

#[derive(Debug, Error)]
pub enum SolveQueryError {
    #[error(transparent)]
    SameDepthCycle(#[from] SameDepthCycleError),
    #[error(transparent)]
    Solve(#[from] SolveError),
}

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

pub struct SolveOutcome<R: Rule> {
    pub shared_state: RuleEnvSharedState<R>,
    pub result: Result<(RuleResultRef<R>, RuleResultsArena<R>), SolveError>,
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum SolveError {
    #[error("fixpoint iteration limit reached")]
    FixpointIterationLimitReached,
    #[error("stack overflow depth reached")]
    StackOverflowDepthReached,
    #[error("same depth cycle escaped to the root solve")]
    UnexpectedSameDepthCycle,
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

fn lookups_match_env<R: Rule>(
    lookups: &Lookups<R>,
    env: &Arc<R::Env>,
    ctx: &mut Context<R>,
) -> bool {
    let started = Instant::now();
    let matches = lookups.iter().all(|(query, expected_result)| {
        env.lookup(&mut ctx.shared_state, query) == *expected_result
    });
    ctx.record_lookup_match(lookups.len(), started.elapsed());
    matches
}

fn answer_matches_env<R: Rule>(
    result_ref: RuleResultRef<R>,
    env: &Arc<R::Env>,
    ctx: &mut Context<R>,
    depth: usize,
) -> bool {
    ctx.record_answer_match_call(depth);

    match ctx
        .answer_match_memo
        .get(&(result_ref, Arc::clone(env)))
        .copied()
    {
        Some(AnswerMatchMemo::Resolved(matches)) => {
            ctx.record_answer_match_memo_hit();
            return matches;
        }
        Some(AnswerMatchMemo::InProgress) => {
            ctx.record_answer_match_in_progress_hit();
            return true;
        }
        None => {}
    }

    ctx.answer_match_memo
        .insert((result_ref, Arc::clone(env)), AnswerMatchMemo::InProgress);

    let started = Instant::now();

    let Some(answer) = ctx.answer_for(result_ref).cloned() else {
        ctx.record_answer_match_missing_answer();
        ctx.record_answer_match_evaluation(0, started.elapsed());
        ctx.answer_match_memo.insert(
            (result_ref, Arc::clone(env)),
            AnswerMatchMemo::Resolved(false),
        );
        return false;
    };

    let matches = lookups_match_env(&answer.lookups, env, ctx)
        && answer
            .dependencies
            .iter()
            .all(|dependency| answer_matches_env(*dependency, env, ctx, depth + 1));
    ctx.record_answer_match_evaluation(answer.dependencies.len(), started.elapsed());
    ctx.answer_match_memo.insert(
        (result_ref, Arc::clone(env)),
        AnswerMatchMemo::Resolved(matches),
    );
    matches
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ActiveAnswerMatch {
    Matches,
    Mismatch,
    Unknown,
}

#[derive(Clone, Copy)]
enum ActiveAnswerMatchMemo {
    InProgress,
    Resolved(ActiveAnswerMatch),
}

// This memo is valid only within a single backreference validation pass.
// The current answer graph is read-only while this function runs, and the memo
// is discarded before any fixpoint rerun can replace results. `InProgress`
// therefore only guards recursive walks over the current graph snapshot.
fn answer_matches_env_for_backref<R: Rule>(
    result_ref: RuleResultRef<R>,
    env: &Arc<R::Env>,
    ctx: &mut Context<R>,
    memo: &mut HashMap<(RuleResultRef<R>, Arc<R::Env>), ActiveAnswerMatchMemo>,
) -> ActiveAnswerMatch {
    let key = (result_ref, Arc::clone(env));
    match memo.get(&key).copied() {
        Some(ActiveAnswerMatchMemo::Resolved(result)) => return result,
        Some(ActiveAnswerMatchMemo::InProgress) => return ActiveAnswerMatch::Matches,
        None => {}
    }

    memo.insert(key.clone(), ActiveAnswerMatchMemo::InProgress);

    let Some(answer) = ctx.answer_for(result_ref).cloned() else {
        memo.insert(
            key,
            ActiveAnswerMatchMemo::Resolved(ActiveAnswerMatch::Unknown),
        );
        return ActiveAnswerMatch::Unknown;
    };

    if !lookups_match_env(&answer.lookups, env, ctx) {
        memo.insert(
            key,
            ActiveAnswerMatchMemo::Resolved(ActiveAnswerMatch::Mismatch),
        );
        return ActiveAnswerMatch::Mismatch;
    }

    let mut saw_unknown = false;
    for dependency in answer.dependencies {
        match answer_matches_env_for_backref(dependency, env, ctx, memo) {
            ActiveAnswerMatch::Matches => {}
            ActiveAnswerMatch::Mismatch => {
                memo.insert(
                    key,
                    ActiveAnswerMatchMemo::Resolved(ActiveAnswerMatch::Mismatch),
                );
                return ActiveAnswerMatch::Mismatch;
            }
            ActiveAnswerMatch::Unknown => saw_unknown = true,
        }
    }

    let result = if saw_unknown {
        ActiveAnswerMatch::Unknown
    } else {
        ActiveAnswerMatch::Matches
    };
    memo.insert(key, ActiveAnswerMatchMemo::Resolved(result));
    result
}

fn validate_cross_env_reuses_in_suffix<R: Rule>(
    dfn: crate::search_graph::DepthFirstNumber,
    ctx: &mut Context<R>,
) -> bool {
    let mut validation_memo = HashMap::new();
    let mut blocked_grew = false;

    for (result_ref, env) in ctx.search_graph.suffix_cross_env_reuses(dfn) {
        let blocked_key = (result_ref, Arc::clone(&env));
        if ctx.blocked_cross_env_reuses.contains(&blocked_key) {
            continue;
        }

        if answer_matches_env_for_backref(result_ref, &env, ctx, &mut validation_memo)
            == ActiveAnswerMatch::Mismatch
        {
            blocked_grew |= ctx.blocked_cross_env_reuses.insert(blocked_key);
        }
    }

    blocked_grew
}

fn cache_key<R: Rule>(goal: &GoalKey<R>) -> CacheKey<R> {
    (goal.query.clone(), goal.state_id)
}

type SuffixSnapshot<R> = HashMap<RuleResultRef<R>, RuleResult<R>>;

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

fn evaluate_goal_once<R: Rule>(
    rule: &R,
    dfn: crate::search_graph::DepthFirstNumber,
    ctx: &mut Context<R>,
) -> Result<Minimums, SolveQueryError> {
    let goal = ctx.search_graph[dfn].goal.clone();
    let mut minimums = Minimums::new();
    let (lookups, dependencies, cross_env_reuses, result_ref) = {
        let mut rule_ctx =
            crate::rule::RuleContext::new(rule, goal.state_id, goal.env, ctx, dfn, &mut minimums);

        let result = match rule.run(goal.query, &mut rule_ctx) {
            Ok(output) => Ok(output),
            Err(crate::rule::RunError::Rule(err)) => Err(err),
            Err(crate::rule::RunError::Solve(error)) => return Err(error),
        };
        let lookups = rule_ctx.lookups.clone();
        let dependencies: Vec<RuleResultRef<R>> =
            rule_ctx.child_result_refs.iter().copied().collect();
        let cross_env_reuses: Vec<(RuleResultRef<R>, Arc<R::Env>)> =
            rule_ctx.cross_env_reuses.iter().cloned().collect();
        let result_ref = rule_ctx.ctx.search_graph[dfn].answer.result_ref;

        replace_result(rule_ctx.ctx, result_ref, result);
        (lookups, dependencies, cross_env_reuses, result_ref)
    };

    ctx.search_graph[dfn].answer.lookups = lookups.clone();
    ctx.search_graph[dfn].answer.dependencies = dependencies.clone();
    ctx.search_graph[dfn].cross_env_reuses = cross_env_reuses;
    ctx.record_answer(lookups.len(), dependencies.len());
    ctx.result_answers.insert(
        result_ref,
        crate::search_graph::Answer {
            result_ref,
            lookups,
            dependencies,
        },
    );
    ctx.search_graph[dfn].links = minimums;
    Ok(minimums)
}

fn solve_new_goal<R: Rule>(
    rule: &R,
    goal: GoalKey<R>,
    ctx: &mut Context<R>,
) -> Result<(GoalSolveResult<R>, Minimums), SolveQueryError> {
    ctx.record_new_goal();
    let result_ref = ctx.result_ref_for(&goal);
    let stack_depth = ctx.stack.push().map_err(|error| match error {
        StackError::Overflow => SolveError::StackOverflowDepthReached,
    })?;
    let dfn = ctx.search_graph.insert(&goal, stack_depth, result_ref);

    let mut reruns = 0;
    let mut previous_snapshot = None;
    let final_minimums = loop {
        let iteration_minimums = evaluate_goal_once(rule, dfn, ctx)?;
        let blocked_grew = validate_cross_env_reuses_in_suffix(dfn, ctx);

        if !ctx.stack[stack_depth].read_and_reset_cycle_flag() {
            break iteration_minimums;
        }

        let current_snapshot = snapshot_suffix(ctx, dfn);
        if previous_snapshot
            .as_ref()
            .is_some_and(|previous| previous == &current_snapshot)
            && !blocked_grew
        {
            break iteration_minimums;
        }

        if reruns >= ctx.fixpoint_iteration_limit {
            return Err(SolveError::FixpointIterationLimitReached.into());
        }
        reruns += 1;
        ctx.record_fixpoint_rerun();
        previous_snapshot = Some(current_snapshot);

        ctx.search_graph.rollback_to(dfn + 1);
    };

    ctx.search_graph.pop_stack_goal(dfn);
    ctx.stack.pop(stack_depth);

    if final_minimums.ancestor() >= dfn {
        ctx.search_graph.move_to_cache(dfn, &mut ctx.cache);
    }

    Ok((GoalSolveResult::Resolved { result_ref }, final_minimums))
}

pub(crate) fn solve_goal<R: Rule>(
    rule: &R,
    goal: GoalKey<R>,
    ctx: &mut Context<R>,
) -> Result<(GoalSolveResult<R>, Minimums), SolveQueryError> {
    ctx.record_goal_attempt();
    if let Some(ancestor_dfn) = ctx
        .search_graph
        .closest_goal(&goal.query, goal.state_id, &goal.env)
    {
        let (ancestor_lazy_depth, result_ref, stack_depth) = {
            let ancestor_node = &ctx.search_graph[ancestor_dfn];
            (
                ancestor_node.goal.lazy_depth,
                ancestor_node.answer.result_ref,
                ancestor_node
                    .stack_depth
                    .expect("closest active goal must still be on stack"),
            )
        };

        if ancestor_lazy_depth >= goal.lazy_depth {
            ctx.record_active_ancestor_same_depth_cycle();
            ctx.stack[stack_depth].flag_cycle();
            return Err(SameDepthCycleError.into());
        }

        ctx.stack[stack_depth].flag_cycle();
        ctx.record_active_ancestor_lazy_hit();
        return Ok((
            GoalSolveResult::Lazy { result_ref },
            Minimums::from_self(ancestor_dfn),
        ));
    }

    if ctx.cross_env_active_reuse_enabled {
        if let Some(ancestor_dfn) = ctx
            .search_graph
            .closest_goal_any_env(&goal.query, goal.state_id)
        {
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

            if ancestor_env != goal.env && ancestor_lazy_depth < goal.lazy_depth {
                let blocked_key = (result_ref, Arc::clone(&goal.env));
                if !ctx.blocked_cross_env_reuses.contains(&blocked_key) {
                    ctx.stack[stack_depth].flag_cycle();
                    ctx.record_active_ancestor_lazy_hit();
                    return Ok((
                        GoalSolveResult::LazyCrossEnv { result_ref },
                        Minimums::from_self(ancestor_dfn),
                    ));
                }
            }
        }
    }

    if let Some(bucket) = ctx.cache.get(&cache_key(&goal)).cloned() {
        ctx.record_cache_bucket_probe();
        for answer in bucket.iter().rev() {
            if answer_matches_env(answer.result_ref, &goal.env, ctx, 0) {
                ctx.record_cache_candidate_hit();
                return Ok((
                    GoalSolveResult::Resolved {
                        result_ref: answer.result_ref,
                    },
                    Minimums::new(),
                ));
            }
            ctx.record_cache_candidate_miss();
        }
    }

    if let Some(dfn) = ctx.search_graph.lookup(&goal) {
        ctx.record_graph_goal_hit();
        let node = &ctx.search_graph[dfn];
        if let Some(stack_depth) = node.stack_depth {
            ctx.stack[stack_depth].flag_cycle();
        }
        return Ok((
            GoalSolveResult::Resolved {
                result_ref: node.answer.result_ref,
            },
            node.links,
        ));
    }

    solve_new_goal(rule, goal, ctx)
}

pub fn solve<R: Rule>(
    rule: &R,
    query: RuleQuery<R>,
    initial_rule: R::RuleStateId,
    env: Arc<R::Env>,
    shared_state: RuleEnvSharedState<R>,
    fixpoint_iteration_limit: usize,
    stack_overflow_depth: usize,
) -> SolveOutcome<R> {
    let mut ctx = Context::new(shared_state, fixpoint_iteration_limit, stack_overflow_depth);
    let root_goal = GoalKey {
        query,
        state_id: initial_rule,
        env,
        lazy_depth: LazyDepth(0),
    };
    let result = solve_goal(rule, root_goal, &mut ctx)
        .map(|(solve_result, _minimums)| match solve_result {
            GoalSolveResult::Resolved { result_ref } => result_ref,
            GoalSolveResult::Lazy { .. } | GoalSolveResult::LazyCrossEnv { .. } => {
                unreachable!("root solve_goal must not resolve lazily")
            }
        })
        .map_err(|error| match error {
            SolveQueryError::SameDepthCycle(_) => SolveError::UnexpectedSameDepthCycle,
            SolveQueryError::Solve(error) => error,
        });

    ctx.emit_stats();

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
