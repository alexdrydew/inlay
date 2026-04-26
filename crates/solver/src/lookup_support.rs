#![cfg_attr(not(feature = "tracing"), allow(unused_variables))]

use std::sync::Arc;

use derive_where::derive_where;
use inlay_instrument_macros::instrumented;
use rustc_hash::FxHashSet as HashSet;

use crate::{
    cache::Cache,
    context::Context,
    instrument::{solver_event, solver_span_record},
    rule::{RuleDependencyEnvDelta, RuleEnv, RuleResultRef},
    search_graph::{Answer, SearchGraph},
    traits::{ResolutionEnv, Rule},
};

pub(crate) type RuleLookupSupport<R> = <RuleEnv<R> as ResolutionEnv>::LookupSupport;
pub(crate) type LookupSupports<R> = Vec<RuleLookupSupport<R>>;
#[derive_where(Clone, PartialEq, Eq)]
pub(crate) struct AnswerSupport<R: Rule> {
    pub(crate) checks: Vec<RuleLookupSupport<R>>,
}

fn merged_lookup_support<R: Rule>(
    left: &RuleLookupSupport<R>,
    right: &RuleLookupSupport<R>,
) -> Option<RuleLookupSupport<R>> {
    crate::traits::RuleLookupSupport::merge_lookup_support(left, right)
        .or_else(|| crate::traits::RuleLookupSupport::merge_lookup_support(right, left))
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

fn insert_transported_support_check<R: Rule>(
    checks: &mut Vec<RuleLookupSupport<R>>,
    delta_from_root: &RuleDependencyEnvDelta<R>,
    support: &RuleLookupSupport<R>,
) {
    insert_compact_lookup_support::<R>(
        checks,
        RuleEnv::<R>::pullback_lookup_support(support, delta_from_root),
    );
}

fn stored_graph_or_cache_answer_support<'a, R: Rule>(
    search_graph: &'a SearchGraph<R>,
    cache: &'a Cache<R>,
    result_ref: RuleResultRef<R>,
) -> Option<&'a AnswerSupport<R>> {
    search_graph
        .stored_answer_support(result_ref)
        .or_else(|| cache.stored_answer_support(result_ref))
}

fn graph_or_cache_answer<'a, R: Rule>(
    search_graph: &'a SearchGraph<R>,
    cache: &'a Cache<R>,
    result_ref: RuleResultRef<R>,
) -> Option<&'a Answer<R>> {
    search_graph
        .answer_for(result_ref)
        .or_else(|| cache.answer_for(result_ref))
}

#[instrumented(
    name = "solver.build_graph_answer_support",
    target = "context_solver",
    level = "trace",
    skip(search_graph, cache),
    fields(result_ref = ?result_ref, answer_nodes, checks, missing_answer)
)]
fn build_graph_answer_support<R: Rule>(
    search_graph: &SearchGraph<R>,
    cache: &Cache<R>,
    result_ref: RuleResultRef<R>,
) -> Option<AnswerSupport<R>> {
    let root_env = Arc::clone(&search_graph.goal_for_result_ref(result_ref)?.env);
    let root_delta = RuleEnv::<R>::dependency_env_delta(&root_env, &root_env);
    let mut stack = vec![(result_ref, root_delta)];
    let mut visited = HashSet::default();
    let mut checks = Vec::new();
    let mut answer_nodes = 0_u64;

    while let Some((current, delta_from_root)) = stack.pop() {
        if !visited.insert((current, delta_from_root.clone())) {
            continue;
        }
        answer_nodes += 1;

        let Some(answer) = graph_or_cache_answer(search_graph, cache, current).cloned() else {
            solver_span_record!(
                answer_nodes,
                checks = checks.len() as u64,
                missing_answer = true
            );
            return None;
        };

        for support in &answer.direct_supports {
            insert_transported_support_check::<R>(&mut checks, &delta_from_root, support);
        }

        for dependency in answer.dependencies.iter().rev() {
            let dependency_delta_from_root =
                RuleEnv::<R>::compose_dependency_env_delta(&delta_from_root, &dependency.env_delta);
            if let Some(child_support) =
                stored_graph_or_cache_answer_support(search_graph, cache, dependency.result_ref)
                    .cloned()
            {
                if !visited.insert((dependency.result_ref, dependency_delta_from_root.clone())) {
                    continue;
                }
                for support in &child_support.checks {
                    insert_transported_support_check::<R>(
                        &mut checks,
                        &dependency_delta_from_root,
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

#[instrumented(
    name = "solver.build_cached_answer_support",
    target = "context_solver",
    level = "trace",
    skip(cache),
    fields(result_ref = ?result_ref, answer_nodes, checks, missing_answer)
)]
fn build_cached_answer_support<R: Rule>(
    cache: &Cache<R>,
    result_ref: RuleResultRef<R>,
) -> Option<AnswerSupport<R>> {
    let root_env = Arc::clone(&cache.goal_for_result_ref(result_ref)?.env);
    let root_delta = RuleEnv::<R>::dependency_env_delta(&root_env, &root_env);
    let mut stack = vec![(result_ref, root_delta)];
    let mut visited = HashSet::default();
    let mut checks = Vec::new();
    let mut answer_nodes = 0_u64;

    while let Some((current, delta_from_root)) = stack.pop() {
        if !visited.insert((current, delta_from_root.clone())) {
            continue;
        }
        answer_nodes += 1;

        let Some(answer) = cache.answer_for(current).cloned() else {
            solver_span_record!(
                answer_nodes,
                checks = checks.len() as u64,
                missing_answer = true
            );
            return None;
        };

        for support in &answer.direct_supports {
            insert_transported_support_check::<R>(&mut checks, &delta_from_root, support);
        }

        for dependency in answer.dependencies.iter().rev() {
            let dependency_delta_from_root =
                RuleEnv::<R>::compose_dependency_env_delta(&delta_from_root, &dependency.env_delta);
            if let Some(child_support) = cache.stored_answer_support(dependency.result_ref).cloned() {
                if !visited.insert((dependency.result_ref, dependency_delta_from_root.clone())) {
                    continue;
                }
                for support in &child_support.checks {
                    insert_transported_support_check::<R>(
                        &mut checks,
                        &dependency_delta_from_root,
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

impl<R: Rule> Context<R> {
    pub(crate) fn cached_answer_support(
        &mut self,
        result_ref: RuleResultRef<R>,
    ) -> Option<AnswerSupport<R>> {
        if let Some(support) = self.cache.stored_answer_support(result_ref) {
            return Some(support.clone());
        }

        let support = build_cached_answer_support(&self.cache, result_ref)?;
        if !self.cache.store_answer_support(result_ref, support.clone()) {
            return None;
        }
        Some(support)
    }

    pub(crate) fn graph_answer_support(
        &mut self,
        result_ref: RuleResultRef<R>,
    ) -> Option<AnswerSupport<R>> {
        if let Some(support) = self.search_graph.stored_answer_support(result_ref).cloned() {
            return Some(support);
        }

        let support = build_graph_answer_support(&self.search_graph, &self.cache, result_ref)?;
        if !self.search_graph.store_answer_support(result_ref, support.clone()) {
            return None;
        }
        Some(support)
    }
}

#[instrumented(
    name = "solver.answer_support_match",
    target = "context_solver",
    level = "trace",
    skip(support),
    ret,
    fields(
        result_ref = ?result_ref,
        checks = support.checks.len() as u64,
        env_hash = crate::solve::debug_env_hash::<R>(Arc::as_ref(env)),
        result_query = %crate::solve::trace_result_query_label::<R>(rule, result_ref, ctx),
        env_label = %crate::solve::trace_env_label::<R>(rule, Arc::as_ref(env))
    )
)]
pub(crate) fn answer_support_matches_env<R: Rule>(
    rule: &R,
    result_ref: RuleResultRef<R>,
    support: &AnswerSupport<R>,
    env: &Arc<R::Env>,
    ctx: &mut Context<R>,
) -> bool {
    for lookup_support in &support.checks {
        if env.lookup_support_matches(&mut ctx.shared_state, lookup_support) {
            continue;
        }

        solver_event!(
            name: "solver.cache_support_miss",
            support_hash = crate::solve::hash_value(lookup_support),
            support_label = format!("{lookup_support:?}").as_str()
        );

        return false;
    }

    true
}
