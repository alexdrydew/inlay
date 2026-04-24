#![cfg_attr(not(feature = "tracing"), allow(unused_variables))]

use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use crate::{
    arena::Arena,
    context::Context,
    instrument::{solver_trace, solver_trace_enabled},
    search_graph::{DepthFirstNumber, GoalKey, LazyDepth, Minimums},
    solve::{
        debug_env_hash, debug_env_label, debug_lookup_query_label, debug_lookup_result_label,
        hash_value, solve_goal, GoalSolveResult, SolveError, SolveResult,
    },
};

pub trait ResolutionEnv: Hash + Eq {
    type SharedState;
    type Query: Hash + Eq + Clone;
    type QueryResult: Hash + Eq + Clone;
    type DependencyEnvDelta: Hash + Eq + Clone;

    fn lookup(
        self: &Arc<Self>,
        shared_state: &mut Self::SharedState,
        query: &Self::Query,
    ) -> Self::QueryResult;

    fn dependency_env_delta(parent: &Arc<Self>, child: &Arc<Self>) -> Self::DependencyEnvDelta;

    fn apply_dependency_env_delta(
        parent: &Arc<Self>,
        delta: &Self::DependencyEnvDelta,
    ) -> Arc<Self>;

    fn env_item_count(_env: &Self) -> usize {
        0
    }

    fn dependency_env_delta_item_count(_delta: &Self::DependencyEnvDelta) -> usize {
        0
    }
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
    Solve(SolveError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LazyDepthMode {
    Keep,
    Increment,
}

impl<R: Rule> From<SolveError> for RunError<R> {
    fn from(value: SolveError) -> Self {
        Self::Solve(value)
    }
}

pub struct RuleContext<'a, R: Rule> {
    env: Arc<R::Env>,
    state_id: R::RuleStateId,
    dfn: DepthFirstNumber,
    pub(crate) lookups: Lookups<R>,
    pub(crate) child_dependencies: HashSet<crate::search_graph::Dependency<R>>,
    pub(crate) cross_env_reuses: HashSet<(RuleResultRef<R>, Arc<R::Env>)>,
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
            child_dependencies: HashSet::new(),
            cross_env_reuses: HashSet::new(),
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
        let parent_query_hash = hash_value(&parent_goal.query);
        let child_query_hash = hash_value(&goal.query);
        let parent_env_hash = debug_env_hash::<R>(Arc::as_ref(&parent_goal.env));
        let child_env_hash = debug_env_hash::<R>(Arc::as_ref(&goal.env));
        let trace_enabled = solver_trace_enabled!();
        let parent_query_label = trace_enabled
            .then(|| {
                self.rule
                    .debug_query_label(&parent_goal.query, parent_goal.state_id)
                    .unwrap_or_else(|| format!("query={parent_query_hash:x}"))
            })
            .unwrap_or_default();
        let child_query_label = trace_enabled
            .then(|| {
                self.rule
                    .debug_query_label(&goal.query, goal.state_id)
                    .unwrap_or_else(|| format!("query={child_query_hash:x}"))
            })
            .unwrap_or_default();
        let parent_env_label = trace_enabled
            .then(|| debug_env_label(self.rule, Arc::as_ref(&parent_goal.env)))
            .unwrap_or_default();
        let child_env_label = trace_enabled
            .then(|| debug_env_label(self.rule, Arc::as_ref(&goal.env)))
            .unwrap_or_default();
        solver_trace!(
            name: "solver.solve_edge",
            parent_dfn = self.dfn.index() as u64,
            parent_query_hash,
            child_query_hash,
            parent_env_hash,
            child_env_hash,
            child_lazy_depth = goal.lazy_depth.0 as u64,
            lazy_mode = match lazy_depth_mode {
                LazyDepthMode::Keep => "keep",
                LazyDepthMode::Increment => "increment",
            },
            parent_query_label = parent_query_label.as_str(),
            child_query_label = child_query_label.as_str(),
            parent_env_label = parent_env_label.as_str(),
            child_env_label = child_env_label.as_str(),
            parent_state_hash = hash_value(&parent_goal.state_id),
            child_state_hash = hash_value(&goal.state_id)
        );
        let child_env = Arc::clone(&goal.env);
        let (solve_result, child_minimums) = solve_goal(self.rule, goal, self.ctx)?;
        self.minimums.update_from(child_minimums);

        match solve_result {
            GoalSolveResult::Resolved { result_ref } => {
                let env_delta = R::Env::dependency_env_delta(&self.env, &child_env);
                let delta_items = R::Env::dependency_env_delta_item_count(&env_delta) as u64;
                solver_trace!(
                    name: "solver.dependency_edge",
                    ?result_ref,
                    delta_items,
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
                let delta_items = R::Env::dependency_env_delta_item_count(&env_delta) as u64;
                solver_trace!(
                    name: "solver.dependency_edge",
                    ?result_ref,
                    delta_items,
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
                let delta_items = R::Env::dependency_env_delta_item_count(&env_delta) as u64;
                solver_trace!(
                    name: "solver.dependency_edge",
                    ?result_ref,
                    delta_items,
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

    pub fn lookup(&mut self, query: &RuleLookupQuery<R>) -> RuleLookupResult<R> {
        let result = self.env.lookup(&mut self.ctx.shared_state, query);
        let query_hash = hash_value(query);
        let result_hash = hash_value(&result);
        let env_hash = debug_env_hash::<R>(self.env.as_ref());
        let trace_enabled = solver_trace_enabled!();
        let query_label = trace_enabled
            .then(|| debug_lookup_query_label(self.rule, query))
            .unwrap_or_default();
        let result_label = trace_enabled
            .then(|| debug_lookup_result_label(self.rule, &result))
            .unwrap_or_default();
        solver_trace!(
            name: "solver.lookup",
            query_hash,
            result_hash,
            env_hash,
            query_label = query_label.as_str(),
            result_label = result_label.as_str()
        );
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

    fn debug_query_label(
        &self,
        _query: &RuleQuery<Self>,
        _state_id: Self::RuleStateId,
    ) -> Option<String> {
        None
    }

    fn debug_env_label(&self, _env: &Self::Env) -> Option<String> {
        None
    }

    fn debug_lookup_query_label(&self, _query: &RuleLookupQuery<Self>) -> Option<String> {
        None
    }

    fn debug_lookup_result_label(&self, _result: &RuleLookupResult<Self>) -> Option<String> {
        None
    }
}
