use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use crate::{
    arena::Arena,
    context::Context,
    search_graph::{DepthFirstNumber, GoalKey, LazyDepth, Minimums},
    solve::{GoalSolveResult, SolveQueryError, SolveResult, solve_goal},
};

pub trait ResolutionEnv: Hash + Eq {
    type SharedState;
    type Query: Hash + Eq + Clone;
    type QueryResult: Eq + Clone;

    fn lookup(
        self: &Arc<Self>,
        shared_state: &mut Self::SharedState,
        query: &Self::Query,
    ) -> Self::QueryResult;
}

pub type RuleResultsArena<R> = <R as Rule>::ResultsArena;
pub type RuleEnv<R> = <R as Rule>::Env;
pub type RuleEnvSharedState<R> = <RuleEnv<R> as ResolutionEnv>::SharedState;
pub type RuleQuery<R> = <R as Rule>::Query;
pub type RuleLookupQuery<R> = <RuleEnv<R> as ResolutionEnv>::Query;
pub type RuleLookupResult<R> = <RuleEnv<R> as ResolutionEnv>::QueryResult;
pub type Lookups<R> = Vec<(RuleLookupQuery<R>, RuleLookupResult<R>)>;
pub type RuleResult<R> = Result<<R as Rule>::Output, <R as Rule>::Err>;
pub type RuleResultRef<R> = <RuleResultsArena<R> as Arena<RuleResult<R>>>::Key;

pub enum RunError<R: Rule> {
    Rule(R::Err),
    Solve(SolveQueryError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LazyDepthMode {
    Keep,
    Increment,
}

impl<R: Rule> From<SolveQueryError> for RunError<R> {
    fn from(value: SolveQueryError) -> Self {
        Self::Solve(value)
    }
}

pub struct RuleContext<'a, R: Rule> {
    env: Arc<R::Env>,
    state_id: R::RuleStateId,
    dfn: DepthFirstNumber,
    pub(crate) lookups: Lookups<R>,
    pub(crate) minimums: &'a mut Minimums,
    pub(crate) ctx: &'a mut Context<R>,
    pub(crate) rule: &'a R,
}

impl<R: Rule> RuleContext<'_, R> {
    pub(crate) fn new<'a>(
        rule: &'a R,
        state_id: R::RuleStateId,
        env: Arc<R::Env>,
        ctx: &'a mut Context<R>,
        dfn: DepthFirstNumber,
        minimums: &'a mut Minimums,
    ) -> RuleContext<'a, R> {
        RuleContext {
            env,
            state_id,
            dfn,
            lookups: vec![],
            minimums,
            ctx,
            rule,
        }
    }

    pub fn state_id(&self) -> R::RuleStateId {
        self.state_id
    }

    pub fn env(&self) -> &R::Env {
        self.env.as_ref()
    }

    pub fn shared(&mut self) -> &mut RuleEnvSharedState<R> {
        &mut self.ctx.shared_state
    }

    pub fn solve(
        &mut self,
        query: RuleQuery<R>,
        state_id: R::RuleStateId,
        lazy_depth_mode: LazyDepthMode,
        env: Arc<R::Env>,
    ) -> Result<SolveResult<'_, R>, SolveQueryError> {
        let current_lazy_depth = self.ctx.search_graph[self.dfn].goal.lazy_depth;
        let lazy_depth = match lazy_depth_mode {
            LazyDepthMode::Keep => current_lazy_depth,
            LazyDepthMode::Increment => LazyDepth(current_lazy_depth.0 + 1),
        };
        let goal = GoalKey {
            query,
            state_id,
            env,
            lazy_depth,
        };
        let (solve_result, child_minimums) = solve_goal(self.rule, goal, self.ctx)?;
        self.minimums.update_from(child_minimums);

        match solve_result {
            GoalSolveResult::Resolved { result_ref } => Ok(SolveResult::Resolved {
                result: self
                    .ctx
                    .results_arena
                    .get(&result_ref)
                    .expect("resolved result must exist"),
                result_ref,
            }),
            GoalSolveResult::Lazy { result_ref } => Ok(SolveResult::Lazy { result_ref }),
        }
    }

    pub fn lookup(&mut self, query: &RuleLookupQuery<R>) -> RuleLookupResult<R> {
        let result = self.env.lookup(&mut self.ctx.shared_state, query);
        self.lookups.push((query.clone(), result.clone()));
        result
    }
}

pub trait Rule: Sized + Debug {
    type Query: Hash + Eq + Clone;
    type Output: 'static + Hash + Eq + Clone;
    type Err: 'static + Hash + Eq + Clone + std::error::Error;
    type Env: ResolutionEnv + Debug;
    type ResultsArena: Arena<RuleResult<Self>> + Default;
    type RuleStateId: Hash + Eq + Copy;

    fn run(
        &self,
        query: RuleQuery<Self>,
        ctx: &mut RuleContext<Self>,
    ) -> Result<Self::Output, RunError<Self>>;
}
