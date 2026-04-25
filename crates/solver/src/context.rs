#![cfg_attr(not(feature = "tracing"), allow(unused_variables))]

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use derive_where::derive_where;
use inlay_instrument_macros::instrumented;

use crate::{
    instrument::solver_span_record,
    rule::{
        RuleEnv, RuleEnvSharedState, RuleLookupSupport, RuleResultRef, RuleResultsArena,
    },
    search_graph::{Answer, CacheBucket, CacheKey, DepthFirstNumber, GoalKey, SearchGraph},
    stack::{Stack, StackDepth, StackError},
    traits::{Arena, ResolutionEnv, Rule},
};

#[derive(Clone, Copy)]
pub(crate) enum AnswerMatchMemo {
    InProgress,
    Resolved(bool),
}

type AnswerMatchMemoKey<R> = (RuleResultRef<R>, Arc<RuleEnv<R>>);
type BlockedCrossEnvReuse<R> = (RuleResultRef<R>, Arc<RuleEnv<R>>);
type RebasedEnvCacheKey<R> = (
    Arc<RuleEnv<R>>,
    <RuleEnv<R> as ResolutionEnv>::DependencyEnvDelta,
);
pub(crate) type SupportCheck<R> = (
    <RuleEnv<R> as ResolutionEnv>::DependencyEnvDelta,
    RuleLookupSupport<R>,
);

#[derive_where(Clone)]
pub(crate) struct AnswerSupport<R: Rule> {
    pub(crate) checks: Vec<SupportCheck<R>>,
}

fn merged_lookup_support<R: Rule>(
    left: &RuleLookupSupport<R>,
    right: &RuleLookupSupport<R>,
) -> Option<RuleLookupSupport<R>> {
    crate::traits::RuleLookupSupport::merge_lookup_support(left, right)
        .or_else(|| crate::traits::RuleLookupSupport::merge_lookup_support(right, left))
}

fn insert_support_check<R: Rule>(checks: &mut Vec<SupportCheck<R>>, mut check: SupportCheck<R>) {
    if checks.contains(&check) {
        return;
    }

    let mut index = 0;
    while index < checks.len() {
        if checks[index].0 != check.0 {
            index += 1;
            continue;
        }

        let Some(merged_support) = merged_lookup_support::<R>(&checks[index].1, &check.1) else {
            index += 1;
            continue;
        };
        let merged_check = (check.0.clone(), merged_support);

        if merged_check == checks[index] {
            return;
        }

        if merged_check == check {
            checks.swap_remove(index);
            continue;
        }

        checks.swap_remove(index);
        check = merged_check;
        index = 0;
    }

    checks.push(check);
}

fn insert_transported_support_check<R: Rule>(
    checks: &mut Vec<SupportCheck<R>>,
    root_delta: &<RuleEnv<R> as ResolutionEnv>::DependencyEnvDelta,
    delta_from_root: &<RuleEnv<R> as ResolutionEnv>::DependencyEnvDelta,
    support: &RuleLookupSupport<R>,
) {
    insert_support_check::<R>(
        checks,
        RuleEnv::<R>::pullback_lookup_support(support, delta_from_root)
            .map(|support| (root_delta.clone(), support))
            .unwrap_or_else(|| (delta_from_root.clone(), support.clone())),
    );
}

pub(crate) struct Context<R: Rule> {
    pub(crate) results_arena: RuleResultsArena<R>,
    pub(crate) result_refs: HashMap<GoalKey<R>, RuleResultRef<R>>,
    pub(crate) result_goals: HashMap<RuleResultRef<R>, GoalKey<R>>,
    pub(crate) result_answers: HashMap<RuleResultRef<R>, Answer<R>>,
    pub(crate) answer_fingerprints: HashMap<RuleResultRef<R>, u64>,
    answer_supports: HashMap<RuleResultRef<R>, AnswerSupport<R>>,
    pub(crate) answer_dependents: HashMap<RuleResultRef<R>, HashSet<RuleResultRef<R>>>,
    answer_match_memo: HashMap<AnswerMatchMemoKey<R>, AnswerMatchMemo>,
    answer_match_memo_envs: HashMap<RuleResultRef<R>, HashSet<Arc<RuleEnv<R>>>>,
    rebased_env_cache: HashMap<RebasedEnvCacheKey<R>, Arc<RuleEnv<R>>>,
    pub(crate) blocked_cross_env_reuses: HashSet<BlockedCrossEnvReuse<R>>,
    pub(crate) search_graph: SearchGraph<R>,
    pub(crate) cache: HashMap<CacheKey<R>, CacheBucket<R>>,
    pub(crate) stack: Stack,
    pub(crate) fixpoint_iteration_limit: usize,
    pub(crate) shared_state: RuleEnvSharedState<R>,
    pub(crate) cache_reuse_enabled: bool,
    pub(crate) cache_dedup_enabled: bool,
}

impl<R: Rule> fmt::Debug for Context<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context")
            .field("results", &self.results_arena.len())
            .field("result_refs", &self.result_refs.len())
            .field("result_goals", &self.result_goals.len())
            .field("result_answers", &self.result_answers.len())
            .field("answer_fingerprints", &self.answer_fingerprints.len())
            .field("answer_supports", &self.answer_supports.len())
            .field("answer_dependents", &self.answer_dependents.len())
            .field("answer_match_memo", &self.answer_match_memo.len())
            .field("answer_match_memo_envs", &self.answer_match_memo_envs.len())
            .field("rebased_env_cache", &self.rebased_env_cache.len())
            .field(
                "blocked_cross_env_reuses",
                &self.blocked_cross_env_reuses.len(),
            )
            .field("cache", &self.cache.len())
            .field("fixpoint_iteration_limit", &self.fixpoint_iteration_limit)
            .field("cache_reuse_enabled", &self.cache_reuse_enabled)
            .field("cache_dedup_enabled", &self.cache_dedup_enabled)
            .finish()
    }
}

impl<R: Rule> Context<R> {
    pub(crate) fn new(
        env_shared_state: RuleEnvSharedState<R>,
        fixpoint_iteration_limit: usize,
        stack_depth_limit: usize,
    ) -> Self {
        Self {
            results_arena: RuleResultsArena::<R>::default(),
            result_refs: HashMap::new(),
            result_goals: HashMap::new(),
            result_answers: HashMap::new(),
            answer_fingerprints: HashMap::new(),
            answer_supports: HashMap::new(),
            answer_dependents: HashMap::new(),
            answer_match_memo: HashMap::new(),
            answer_match_memo_envs: HashMap::new(),
            rebased_env_cache: HashMap::new(),
            blocked_cross_env_reuses: HashSet::new(),
            search_graph: SearchGraph::new(),
            cache: HashMap::new(),
            stack: Stack::new(stack_depth_limit),
            fixpoint_iteration_limit,
            shared_state: env_shared_state,
            cache_reuse_enabled: std::env::var_os("INLAY_DISABLE_CACHE_REUSE").is_none(),
            cache_dedup_enabled: std::env::var_os("INLAY_DISABLE_CACHE_DEDUP").is_none(),
        }
    }

    pub(crate) fn result_ref_for(&mut self, goal: &GoalKey<R>) -> RuleResultRef<R> {
        if let Some(result_ref) = self.result_refs.get(goal).copied() {
            return result_ref;
        }

        let result_ref = self.results_arena.insert_placeholder();
        self.result_refs.insert(goal.clone(), result_ref);
        self.result_goals.insert(result_ref, goal.clone());
        result_ref
    }

    pub(crate) fn call_on_stack<T, E>(
        &mut self,
        goal: &GoalKey<R>,
        result_ref: RuleResultRef<R>,
        f: impl FnOnce(&mut Self, DepthFirstNumber, StackDepth) -> Result<T, E>,
    ) -> Result<(DepthFirstNumber, T), E>
    where
        E: From<StackError>,
    {
        let stack_depth = self.stack.push().map_err(E::from)?;
        let dfn = self.search_graph.insert(goal, stack_depth, result_ref);
        let result = f(self, dfn, stack_depth);
        self.search_graph.pop_stack_goal(dfn);
        self.stack.pop(stack_depth);
        result.map(|value| (dfn, value))
    }

    pub(crate) fn goal_for_result_ref(&self, result_ref: RuleResultRef<R>) -> Option<&GoalKey<R>> {
        self.result_goals.get(&result_ref)
    }

    #[instrumented(
        name = "solver.replace_answer",
        target = "context_solver",
        level = "trace",
        fields(
            result_ref = ?result_ref,
            changed,
            dependency_count,
            memo_entries_cleared,
            support_entries_cleared
        )
    )]
    pub(crate) fn replace_answer(&mut self, result_ref: RuleResultRef<R>, answer: Answer<R>) {
        let changed = self
            .result_answers
            .get(&result_ref)
            .is_none_or(|old| old != &answer);
        let dependency_count = answer.dependencies.len() as u64;

        if !changed {
            solver_span_record!(
                changed,
                dependency_count,
                memo_entries_cleared = 0_u64,
                support_entries_cleared = 0_u64
            );
            return;
        }

        let old_dependencies = self
            .result_answers
            .insert(result_ref, answer)
            .map(|old| old.dependencies)
            .unwrap_or_default();

        for dependency in old_dependencies {
            if let Some(dependents) = self.answer_dependents.get_mut(&dependency.result_ref) {
                dependents.remove(&result_ref);
                if dependents.is_empty() {
                    self.answer_dependents.remove(&dependency.result_ref);
                }
            }
        }

        let dependencies = self
            .result_answers
            .get(&result_ref)
            .expect("inserted answer must exist")
            .dependencies
            .clone();
        for dependency in dependencies {
            self.answer_dependents
                .entry(dependency.result_ref)
                .or_default()
                .insert(result_ref);
        }

        let memo_entries_cleared = self.invalidate_answer_match_memo_closure(result_ref);
        let support_entries_cleared = self.invalidate_answer_support_closure(result_ref);
        solver_span_record!(
            changed,
            dependency_count,
            memo_entries_cleared,
            support_entries_cleared
        );
        self.invalidate_fingerprint_closure(result_ref);
    }

    fn dependent_closure(&self, result_ref: RuleResultRef<R>) -> HashSet<RuleResultRef<R>> {
        let mut stack = vec![result_ref];
        let mut visited = HashSet::new();

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            if let Some(dependents) = self.answer_dependents.get(&current) {
                stack.extend(dependents.iter().copied());
            }
        }

        visited
    }

    fn invalidate_answer_match_memo_closure(&mut self, result_ref: RuleResultRef<R>) -> u64 {
        let mut removed = 0_u64;
        for affected_result_ref in self.dependent_closure(result_ref) {
            let Some(envs) = self.answer_match_memo_envs.remove(&affected_result_ref) else {
                continue;
            };
            for env in envs {
                if self
                    .answer_match_memo
                    .remove(&(affected_result_ref, env))
                    .is_some()
                {
                    removed += 1;
                }
            }
        }
        removed
    }

    fn invalidate_fingerprint_closure(&mut self, result_ref: RuleResultRef<R>) {
        for current in self.dependent_closure(result_ref) {
            self.answer_fingerprints.remove(&current);
        }
    }

    fn invalidate_answer_support_closure(&mut self, result_ref: RuleResultRef<R>) -> u64 {
        let mut removed = 0_u64;
        for current in self.dependent_closure(result_ref) {
            if self.answer_supports.remove(&current).is_some() {
                removed += 1;
            }
        }
        removed
    }

    pub(crate) fn answer_for(&self, result_ref: RuleResultRef<R>) -> Option<&Answer<R>> {
        self.result_answers.get(&result_ref)
    }

    pub(crate) fn answer_support(
        &mut self,
        result_ref: RuleResultRef<R>,
    ) -> Option<AnswerSupport<R>> {
        if let Some(support) = self.answer_supports.get(&result_ref).cloned() {
            return Some(support);
        }

        let support = self.build_answer_support(result_ref)?;
        self.answer_supports.insert(result_ref, support.clone());
        Some(support)
    }

    #[instrumented(
        name = "solver.build_answer_support",
        target = "context_solver",
        level = "trace",
        skip(self),
        fields(result_ref = ?result_ref, answer_nodes, checks, missing_answer)
    )]
    fn build_answer_support(&mut self, result_ref: RuleResultRef<R>) -> Option<AnswerSupport<R>> {
        let root_env = Arc::clone(&self.goal_for_result_ref(result_ref)?.env);
        let root_delta = RuleEnv::<R>::dependency_env_delta(&root_env, &root_env);
        let mut stack = vec![(result_ref, root_delta.clone())];
        let mut visited = HashSet::new();
        let mut checks = Vec::new();
        let mut answer_nodes = 0_u64;

        while let Some((current, delta_from_root)) = stack.pop() {
            if !visited.insert((current, delta_from_root.clone())) {
                continue;
            }
            answer_nodes += 1;

            let Some(answer) = self.answer_for(current).cloned() else {
                solver_span_record!(
                    answer_nodes,
                    checks = checks.len() as u64,
                    missing_answer = true
                );
                return None;
            };

            for support in &answer.direct_supports {
                insert_transported_support_check::<R>(
                    &mut checks,
                    &root_delta,
                    &delta_from_root,
                    support,
                );
            }

            for dependency in answer.dependencies.iter().rev() {
                let dependency_delta_from_root = RuleEnv::<R>::compose_dependency_env_delta(
                    &delta_from_root,
                    &dependency.env_delta,
                );
                if let Some(child_support) = self.answer_supports.get(&dependency.result_ref) {
                    if !visited.insert((dependency.result_ref, dependency_delta_from_root.clone()))
                    {
                        continue;
                    }
                    for (child_delta, support) in &child_support.checks {
                        insert_transported_support_check::<R>(
                            &mut checks,
                            &root_delta,
                            &RuleEnv::<R>::compose_dependency_env_delta(
                                &dependency_delta_from_root,
                                child_delta,
                            ),
                            support,
                        );
                    }
                } else {
                    stack.push((dependency.result_ref, dependency_delta_from_root));
                }
            }
        }

        solver_span_record!(
            answer_nodes,
            checks = checks.len() as u64,
            missing_answer = false
        );
        Some(AnswerSupport { checks })
    }

    pub(crate) fn answer_match_memo(
        &self,
        result_ref: RuleResultRef<R>,
        env: &Arc<RuleEnv<R>>,
    ) -> Option<AnswerMatchMemo> {
        self.answer_match_memo
            .get(&(result_ref, Arc::clone(env)))
            .copied()
    }

    pub(crate) fn insert_answer_match_memo(
        &mut self,
        result_ref: RuleResultRef<R>,
        env: &Arc<RuleEnv<R>>,
        memo: AnswerMatchMemo,
    ) {
        self.answer_match_memo
            .insert((result_ref, Arc::clone(env)), memo);
        self.answer_match_memo_envs
            .entry(result_ref)
            .or_default()
            .insert(Arc::clone(env));
    }

    #[instrumented(
        name = "solver.rebase_env_for_dependency",
        target = "context_solver",
        level = "trace",
        fields(parent_items, delta_items, child_items, cache_hit)
    )]
    pub(crate) fn rebase_env_for_dependency(
        &mut self,
        parent: &Arc<RuleEnv<R>>,
        delta: &<RuleEnv<R> as ResolutionEnv>::DependencyEnvDelta,
    ) -> Arc<RuleEnv<R>> {
        let key = (Arc::clone(parent), delta.clone());
        let parent_items = R::Env::env_item_count(parent.as_ref()) as u64;
        let delta_items = R::Env::dependency_env_delta_item_count(delta) as u64;
        solver_span_record!(parent_items, delta_items);

        if let Some(env) = self.rebased_env_cache.get(&key).cloned() {
            let child_items = R::Env::env_item_count(env.as_ref()) as u64;
            solver_span_record!(cache_hit = true, child_items);
            return env;
        }

        let env = RuleEnv::<R>::apply_dependency_env_delta(parent, delta);
        self.rebased_env_cache.insert(key, Arc::clone(&env));
        let child_items = R::Env::env_item_count(env.as_ref()) as u64;
        solver_span_record!(cache_hit = false, child_items);
        env
    }
}
