use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;

use crate::{
    arena::Arena,
    context::Context,
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
    lookups.iter().all(|(query, expected_result)| {
        env.lookup(&mut ctx.shared_state, query) == *expected_result
    })
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
    let lookups = {
        let mut rule_ctx =
            crate::rule::RuleContext::new(rule, goal.state_id, goal.env, ctx, dfn, &mut minimums);

        let result = match rule.run(goal.query, &mut rule_ctx) {
            Ok(output) => Ok(output),
            Err(crate::rule::RunError::Rule(err)) => Err(err),
            Err(crate::rule::RunError::Solve(error)) => return Err(error),
        };
        let lookups = rule_ctx.lookups.clone();
        let result_ref = rule_ctx.ctx.search_graph[dfn].answer.result_ref;

        replace_result(rule_ctx.ctx, result_ref, result);
        lookups
    };

    ctx.search_graph[dfn].answer.lookups = lookups;
    ctx.search_graph[dfn].links = minimums;
    Ok(minimums)
}

fn solve_new_goal<R: Rule>(
    rule: &R,
    goal: GoalKey<R>,
    ctx: &mut Context<R>,
) -> Result<(GoalSolveResult<R>, Minimums), SolveQueryError> {
    let result_ref = ctx.result_ref_for(&goal);
    let stack_depth = ctx.stack.push().map_err(|error| match error {
        StackError::Overflow => SolveError::StackOverflowDepthReached,
    })?;
    let dfn = ctx.search_graph.insert(&goal, stack_depth, result_ref);

    let mut reruns = 0;
    let mut previous_snapshot = None;
    let final_minimums = loop {
        let iteration_minimums = evaluate_goal_once(rule, dfn, ctx)?;

        if !ctx.stack[stack_depth].read_and_reset_cycle_flag() {
            break iteration_minimums;
        }

        let current_snapshot = snapshot_suffix(ctx, dfn);
        if previous_snapshot
            .as_ref()
            .is_some_and(|previous| previous == &current_snapshot)
        {
            break iteration_minimums;
        }

        if reruns >= ctx.fixpoint_iteration_limit {
            return Err(SolveError::FixpointIterationLimitReached.into());
        }
        reruns += 1;
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
    if let Some(ancestor_dfn) = ctx
        .search_graph
        .closest_goal(&goal.query, goal.state_id, &goal.env)
    {
        let ancestor_node = &ctx.search_graph[ancestor_dfn];
        let ancestor_lazy_depth = ancestor_node.goal.lazy_depth;

        if ancestor_lazy_depth >= goal.lazy_depth {
            let stack_depth = ancestor_node
                .stack_depth
                .expect("closest active goal must still be on stack");
            ctx.stack[stack_depth].flag_cycle();
            return Err(SameDepthCycleError.into());
        }

        let (result_ref, stack_depth) = (
            ancestor_node.answer.result_ref,
            ancestor_node
                .stack_depth
                .expect("closest active goal must still be on stack"),
        );

        ctx.stack[stack_depth].flag_cycle();
        return Ok((
            GoalSolveResult::Lazy { result_ref },
            Minimums::from_self(ancestor_dfn),
        ));
    }

    if let Some(bucket) = ctx.cache.get(&cache_key(&goal)).cloned() {
        if let Some(answer) = bucket
            .iter()
            .rev()
            .find(|answer| lookups_match_env(&answer.lookups, &goal.env, ctx))
        {
            return Ok((
                GoalSolveResult::Resolved {
                    result_ref: answer.result_ref,
                },
                Minimums::new(),
            ));
        }
    }

    if let Some(dfn) = ctx.search_graph.lookup(&goal) {
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
            GoalSolveResult::Lazy { .. } => {
                unreachable!("root solve_goal must not resolve lazily")
            }
        })
        .map_err(|error| match error {
            SolveQueryError::SameDepthCycle(_) => SolveError::UnexpectedSameDepthCycle,
            SolveQueryError::Solve(error) => error,
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
