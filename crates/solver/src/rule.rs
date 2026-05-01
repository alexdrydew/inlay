#![cfg_attr(not(feature = "tracing"), allow(unused_variables))]

use std::fmt;
use std::sync::Arc;

use derive_where::derive_where;
use inlay_instrument::{inlay_event, inlay_span_record, instrumented};
use rustc_hash::FxHashSet as HashSet;
use thiserror::Error;

use crate::{
    context::Context,
    lookup_support::LookupSupports,
    search_graph::{DepthFirstNumber, GoalKey, LazyDepth, Minimums},
    solve::{GoalSolveResult, SolveError, SolveResult, solve_goal},
    traits::Arena,
};

pub use crate::traits::{ResolutionEnv, Rule};

pub type RuleResultsArena<R> = <R as Rule>::ResultsArena;
pub type RuleEnv<R> = <R as Rule>::Env;
pub type RuleEnvSharedState<R> = <RuleEnv<R> as ResolutionEnv>::SharedState;
pub type RuleDependencyEnvDelta<R> = <RuleEnv<R> as ResolutionEnv>::DependencyEnvDelta;
pub type RuleQuery<R> = <R as Rule>::Query;
pub type RuleLookupQuery<R> = <RuleEnv<R> as ResolutionEnv>::Query;
pub type RuleLookupResult<R> = <RuleEnv<R> as ResolutionEnv>::QueryResult;
pub type RuleResult<R> = Result<<R as Rule>::Output, <R as Rule>::Err>;
pub type RuleResultRef<R> = <RuleResultsArena<R> as Arena<RuleResult<R>>>::Key;

#[derive(Error)]
#[derive_where(Debug)]
pub enum RunError<R: Rule> {
    #[error("{0}")]
    Rule(R::Err),
    #[error("{0}")]
    Solve(#[from] SolveError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LazyDepthMode {
    Keep,
    Increment,
}

pub struct RuleContext<'a, R: Rule> {
    env: Arc<R::Env>,
    state_id: R::RuleStateId,
    dfn: DepthFirstNumber,
    pub(crate) lookup_supports: LookupSupports<R>,
    pub(crate) child_dependencies: HashSet<crate::search_graph::Dependency<R>>,
    pub(crate) cross_env_reuses: HashSet<(RuleResultRef<R>, Arc<R::Env>)>,
    pub(crate) minimums: &'a mut Minimums,
    pub(crate) ctx: &'a mut Context<R>,
    pub(crate) rule: &'a R,
}

impl<R: Rule> fmt::Debug for RuleContext<'_, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuleContext")
            .field("state_id", &self.state_id)
            .field("dfn", &self.dfn)
            .field("env", &self.env)
            .field("lookup_supports", &self.lookup_supports.len())
            .field("child_dependencies", &self.child_dependencies.len())
            .field("cross_env_reuses", &self.cross_env_reuses.len())
            .finish()
    }
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
            lookup_supports: vec![],
            child_dependencies: HashSet::default(),
            cross_env_reuses: HashSet::default(),
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

    pub fn env_arc(&self) -> Arc<R::Env> {
        Arc::clone(&self.env)
    }

    pub fn shared(&mut self) -> &mut RuleEnvSharedState<R> {
        &mut self.ctx.shared_state
    }

    #[instrumented(
        name = "solver.solve_child",
        target = "inlay",
        level = "trace",
        ret,
        err,
        fields(
            parent_dfn,
            parent_query_hash,
            child_query_hash,
            parent_env_hash,
            child_env_hash,
            child_lazy_depth,
            lazy_mode,
            parent_state_hash,
            child_state_hash
        )
    )]
    pub fn solve(
        &mut self,
        query: RuleQuery<R>,
        state_id: R::RuleStateId,
        lazy_depth_mode: LazyDepthMode,
        env: Arc<R::Env>,
    ) -> Result<SolveResult<'_, R>, SolveError> {
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
        let parent_goal = self.ctx.search_graph[self.dfn].goal.clone();
        inlay_span_record!(
            parent_dfn = self.dfn.index() as u64,
            parent_query_hash = crate::solve::hash_value(&parent_goal.query),
            child_query_hash = crate::solve::hash_value(&goal.query),
            parent_env_hash = crate::solve::debug_env_hash::<R>(Arc::as_ref(&parent_goal.env)),
            child_env_hash = crate::solve::debug_env_hash::<R>(Arc::as_ref(&goal.env)),
            child_lazy_depth = goal.lazy_depth.0 as u64,
            lazy_mode = ::tracing::field::debug(lazy_depth_mode),
            parent_state_hash = crate::solve::hash_value(&parent_goal.state_id),
            child_state_hash = crate::solve::hash_value(&goal.state_id)
        );
        let child_env = Arc::clone(&goal.env);
        let (solve_result, child_minimums) = solve_goal(self.rule, goal, self.ctx)?;
        self.minimums.update_from(child_minimums);

        match solve_result {
            GoalSolveResult::Resolved { result_ref } => {
                let env_delta = R::Env::dependency_env_delta(&self.env, &child_env);
                inlay_event!(
                    name: "solver.dependency_edge",
                    ?result_ref,
                    delta_hash = crate::solve::hash_value(&env_delta),
                    outcome = "resolved"
                );
                self.child_dependencies
                    .insert(crate::search_graph::Dependency {
                        result_ref,
                        env_delta,
                    });
                Ok(SolveResult::Resolved {
                    result: self
                        .ctx
                        .results_arena
                        .get(&result_ref)
                        .expect("resolved result must exist"),
                    result_ref,
                })
            }
            GoalSolveResult::Lazy { result_ref } => {
                let env_delta = R::Env::dependency_env_delta(&self.env, &child_env);
                inlay_event!(
                    name: "solver.dependency_edge",
                    ?result_ref,
                    delta_hash = crate::solve::hash_value(&env_delta),
                    outcome = "lazy"
                );
                self.child_dependencies
                    .insert(crate::search_graph::Dependency {
                        result_ref,
                        env_delta,
                    });
                Ok(SolveResult::Lazy { result_ref })
            }
            GoalSolveResult::LazyCrossEnv { result_ref } => {
                let env_delta = R::Env::dependency_env_delta(&self.env, &child_env);
                inlay_event!(
                    name: "solver.dependency_edge",
                    ?result_ref,
                    delta_hash = crate::solve::hash_value(&env_delta),
                    outcome = "lazy_cross_env"
                );
                self.child_dependencies
                    .insert(crate::search_graph::Dependency {
                        result_ref,
                        env_delta,
                    });
                self.cross_env_reuses.insert((result_ref, child_env));
                Ok(SolveResult::Lazy { result_ref })
            }
        }
    }

    #[instrumented(
        name = "solver.lookup",
        target = "inlay",
        level = "trace",
        ret,
        fields(query_hash, result_hash, env_hash)
    )]
    pub fn lookup(&mut self, query: &RuleLookupQuery<R>) -> RuleLookupResult<R> {
        let result = self.env.lookup(&mut self.ctx.shared_state, query);
        let support = self
            .env
            .lookup_support(&mut self.ctx.shared_state, query, &result);
        inlay_span_record!(
            query_hash = crate::solve::hash_value(query),
            result_hash = crate::solve::hash_value(&result),
            env_hash = crate::solve::debug_env_hash::<R>(self.env.as_ref())
        );
        self.lookup_supports.push(support);
        result
    }
}
