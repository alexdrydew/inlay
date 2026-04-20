use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use std::time::Instant;

use crate::{
    arena::Arena,
    context::Context,
    search_graph::{DepthFirstNumber, GoalKey, LazyDepth, Minimums},
    solve::{
        debug_env_hash, debug_env_label, json_escape, solve_goal, GoalSolveResult, SolveQueryError,
        SolveResult,
    },
};

pub trait ResolutionEnv: Hash + Eq {
    type SharedState;
    type Query: Hash + Eq + Clone;
    type QueryResult: Hash + Eq + Clone;

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
    pub(crate) child_result_refs: HashSet<RuleResultRef<R>>,
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
            child_result_refs: HashSet::new(),
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
        if let Some(seq) = self.ctx.next_trace_seq() {
            let parent_goal = self.ctx.search_graph[self.dfn].goal.clone();
            let parent_query = self
                .rule
                .debug_query_label(&parent_goal.query, parent_goal.state_id)
                .unwrap_or_else(|| "parent".to_string());
            let child_query = self
                .rule
                .debug_query_label(&goal.query, goal.state_id)
                .unwrap_or_else(|| "child".to_string());
            let parent_env = debug_env_label(self.rule, Arc::as_ref(&parent_goal.env));
            let child_env = debug_env_label(self.rule, Arc::as_ref(&goal.env));
            let line = format!(
                concat!(
                    "{{\"seq\":{},\"event\":\"solve_edge\",",
                    "\"parent_dfn\":{},\"parent_query\":\"{}\",",
                    "\"parent_env\":\"{}\",\"parent_env_hash\":\"{:x}\",",
                    "\"child_query\":\"{}\",\"child_env\":\"{}\",",
                    "\"child_env_hash\":\"{:x}\",\"child_lazy_depth\":{},",
                    "\"lazy_mode\":\"{}\"}}"
                ),
                seq,
                self.dfn.index(),
                json_escape(&parent_query),
                json_escape(&parent_env),
                debug_env_hash::<R>(Arc::as_ref(&parent_goal.env)),
                json_escape(&child_query),
                json_escape(&child_env),
                debug_env_hash::<R>(Arc::as_ref(&goal.env)),
                goal.lazy_depth.0 as u64,
                match lazy_depth_mode {
                    LazyDepthMode::Keep => "keep",
                    LazyDepthMode::Increment => "increment",
                }
            );
            self.ctx
                .trace_line(&format!("{parent_query} {child_query}"), line);
        }
        let child_env = Arc::clone(&goal.env);
        let (solve_result, child_minimums) = solve_goal(self.rule, goal, self.ctx)?;
        self.minimums.update_from(child_minimums);

        match solve_result {
            GoalSolveResult::Resolved { result_ref } => {
                self.child_result_refs.insert(result_ref);
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
                self.child_result_refs.insert(result_ref);
                Ok(SolveResult::Lazy { result_ref })
            }
            GoalSolveResult::LazyCrossEnv { result_ref } => {
                self.child_result_refs.insert(result_ref);
                self.cross_env_reuses.insert((result_ref, child_env));
                Ok(SolveResult::Lazy { result_ref })
            }
        }
    }

    pub fn lookup(&mut self, query: &RuleLookupQuery<R>) -> RuleLookupResult<R> {
        let started = Instant::now();
        let result = self.env.lookup(&mut self.ctx.shared_state, query);
        self.ctx.record_lookup_call(started.elapsed());
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
}
