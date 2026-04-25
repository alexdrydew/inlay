#![cfg_attr(not(feature = "tracing"), allow(unused_variables))]

use std::collections::HashSet;
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::sync::Arc;

use derive_where::derive_where;
use inlay_instrument_macros::instrumented;

use crate::{
    arena::Arena,
    context::Context,
    instrument::{solver_event, solver_span_record},
    search_graph::{DepthFirstNumber, GoalKey, LazyDepth, Minimums},
    solve::{GoalSolveResult, SolveError, SolveResult, debug_env_hash, hash_value, solve_goal},
};

pub trait ResolutionEnv: Hash + Eq {
    type SharedState: Debug;
    type Query: Hash + Eq + Clone + Debug;
    type QueryResult: Hash + Eq + Clone + Debug;
    type DependencyEnvDelta: Hash + Eq + Clone + Debug;
    type LookupSupport: Hash + Eq + Clone + Debug;

    fn lookup(
        self: &Arc<Self>,
        shared_state: &mut Self::SharedState,
        query: &Self::Query,
    ) -> Self::QueryResult;

    fn lookup_support(
        self: &Arc<Self>,
        shared_state: &mut Self::SharedState,
        query: &Self::Query,
        result: &Self::QueryResult,
    ) -> Self::LookupSupport;

    fn lookup_support_matches(
        self: &Arc<Self>,
        candidate: &Arc<Self>,
        shared_state: &mut Self::SharedState,
        support: &Self::LookupSupport,
    ) -> bool;

    fn merge_lookup_support(
        left: &Self::LookupSupport,
        right: &Self::LookupSupport,
    ) -> Option<Self::LookupSupport> {
        if left == right {
            Some(left.clone())
        } else {
            None
        }
    }

    fn pullback_lookup_support(
        _support: &Self::LookupSupport,
        _delta: &Self::DependencyEnvDelta,
    ) -> Option<Self::LookupSupport> {
        None
    }

    fn dependency_env_delta(parent: &Arc<Self>, child: &Arc<Self>) -> Self::DependencyEnvDelta;

    fn compose_dependency_env_delta(
        first: &Self::DependencyEnvDelta,
        second: &Self::DependencyEnvDelta,
    ) -> Self::DependencyEnvDelta;

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
pub type RuleLookupSupport<R> = <RuleEnv<R> as ResolutionEnv>::LookupSupport;

pub type LookupSupports<R> = Vec<RuleLookupSupport<R>>;
pub type RuleResult<R> = Result<<R as Rule>::Output, <R as Rule>::Err>;
pub type RuleResultRef<R> = <RuleResultsArena<R> as Arena<RuleResult<R>>>::Key;

fn merged_lookup_support<R: Rule>(
    left: &RuleLookupSupport<R>,
    right: &RuleLookupSupport<R>,
) -> Option<RuleLookupSupport<R>> {
    RuleEnv::<R>::merge_lookup_support(left, right)
        .or_else(|| RuleEnv::<R>::merge_lookup_support(right, left))
}

fn insert_compact_lookup_support<R: Rule>(
    compacted: &mut LookupSupports<R>,
    mut support: RuleLookupSupport<R>,
) {
    let mut index = 0;
    while index < compacted.len() {
        let Some(merged) = merged_lookup_support::<R>(&compacted[index], &support) else {
            index += 1;
            continue;
        };

        if merged == compacted[index] {
            return;
        }

        if merged == support {
            compacted.swap_remove(index);
            continue;
        }

        compacted.swap_remove(index);
        support = merged;
        index = 0;
    }

    compacted.push(support);
}

pub(crate) fn compact_lookup_supports<R: Rule>(supports: LookupSupports<R>) -> LookupSupports<R> {
    let mut compacted = Vec::new();
    for support in supports {
        insert_compact_lookup_support::<R>(&mut compacted, support);
    }
    compacted
}

#[derive_where(Debug)]
pub enum RunError<R: Rule> {
    Rule(R::Err),
    Solve(SolveError),
}

impl<R: Rule> fmt::Display for RunError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rule(error) => fmt::Display::fmt(error, f),
            Self::Solve(error) => fmt::Display::fmt(error, f),
        }
    }
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

    #[instrumented(
        name = "solver.solve_child",
        target = "context_solver",
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
            parent_query_label,
            child_query_label,
            parent_env_label,
            child_env_label,
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
        let parent_query_hash = hash_value(&parent_goal.query);
        let child_query_hash = hash_value(&goal.query);
        let parent_env_hash = debug_env_hash::<R>(Arc::as_ref(&parent_goal.env));
        let child_env_hash = debug_env_hash::<R>(Arc::as_ref(&goal.env));
        solver_span_record!(
            parent_dfn = self.dfn.index() as u64,
            parent_query_hash,
            child_query_hash,
            parent_env_hash,
            child_env_hash,
            child_lazy_depth = goal.lazy_depth.0 as u64,
            lazy_mode = ::tracing::field::debug(lazy_depth_mode),
            parent_query_label = self
                .rule
                .debug_query_label(&parent_goal.query, parent_goal.state_id)
                .unwrap_or_else(|| format!("query={parent_query_hash:x}"))
                .as_str(),
            child_query_label = self
                .rule
                .debug_query_label(&goal.query, goal.state_id)
                .unwrap_or_else(|| format!("query={child_query_hash:x}"))
                .as_str(),
            parent_env_label =
                crate::solve::debug_env_label(self.rule, Arc::as_ref(&parent_goal.env),).as_str(),
            child_env_label =
                crate::solve::debug_env_label(self.rule, Arc::as_ref(&goal.env)).as_str(),
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
                solver_event!(
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
                solver_event!(
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
                solver_event!(
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

    #[instrumented(
        name = "solver.lookup",
        target = "context_solver",
        level = "trace",
        ret,
        fields(query_hash, result_hash, env_hash, query_label, result_label)
    )]
    pub fn lookup(&mut self, query: &RuleLookupQuery<R>) -> RuleLookupResult<R> {
        let result = self.env.lookup(&mut self.ctx.shared_state, query);
        let support = self
            .env
            .lookup_support(&mut self.ctx.shared_state, query, &result);
        let query_hash = hash_value(query);
        let result_hash = hash_value(&result);
        let env_hash = debug_env_hash::<R>(self.env.as_ref());
        #[cfg(feature = "tracing")]
        let query_label = crate::solve::debug_lookup_query_label(self.rule, query);
        #[cfg(feature = "tracing")]
        let result_label = crate::solve::debug_lookup_result_label(self.rule, &result);
        solver_span_record!(
            query_hash,
            result_hash,
            env_hash,
            query_label = query_label.as_str(),
            result_label = result_label.as_str()
        );
        self.lookup_supports.push(support);
        result
    }
}

pub trait Rule: Sized + Debug {
    type Query: Hash + Eq + Clone + Debug;
    type Output: 'static + Hash + Eq + Clone;
    type Err: 'static + Hash + Eq + Clone + std::error::Error;
    type Env: ResolutionEnv + Debug;
    type ResultsArena: Arena<RuleResult<Self>> + Default;
    type RuleStateId: Hash + Eq + Copy + Debug;

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
