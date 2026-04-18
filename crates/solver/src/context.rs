use std::collections::HashMap;

use crate::{
    arena::Arena,
    rule::{Rule, RuleEnvSharedState, RuleResultRef, RuleResultsArena},
    search_graph::{Answer, CacheKey, GoalKey, SearchGraph},
    stack::Stack,
};

pub(crate) struct Context<R: Rule> {
    pub(crate) results_arena: RuleResultsArena<R>,
    pub(crate) result_refs: HashMap<GoalKey<R>, RuleResultRef<R>>,
    pub(crate) search_graph: SearchGraph<R>,
    pub(crate) cache: HashMap<CacheKey<R>, Vec<Answer<R>>>,
    pub(crate) stack: Stack,
    pub(crate) fixpoint_iteration_limit: usize,
    pub(crate) shared_state: RuleEnvSharedState<R>,
}

impl<R: Rule> Context<R> {
    pub(crate) fn new(
        env_shared_state: RuleEnvSharedState<R>,
        fixpoint_iteration_limit: usize,
        stack_overflow_depth: usize,
    ) -> Self {
        Self {
            results_arena: RuleResultsArena::<R>::default(),
            result_refs: HashMap::new(),
            search_graph: SearchGraph::new(),
            cache: HashMap::new(),
            stack: Stack::new(stack_overflow_depth),
            fixpoint_iteration_limit,
            shared_state: env_shared_state,
        }
    }

    pub(crate) fn result_ref_for(&mut self, goal: &GoalKey<R>) -> RuleResultRef<R> {
        if let Some(result_ref) = self.result_refs.get(goal).copied() {
            return result_ref;
        }

        let result_ref = self.results_arena.insert_placeholder();
        self.result_refs.insert(goal.clone(), result_ref);
        result_ref
    }
}
