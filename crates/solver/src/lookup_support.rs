#![cfg_attr(not(feature = "tracing"), allow(unused_variables))]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use derive_where::derive_where;
use inlay_instrument_macros::instrumented;

use crate::{
    context::Context,
    instrument::{solver_event, solver_span_record},
    rule::{RuleEnv, RuleResultRef},
    traits::{ResolutionEnv, Rule},
};

pub(crate) type RuleLookupSupport<R> = <RuleEnv<R> as ResolutionEnv>::LookupSupport;
pub(crate) type LookupSupports<R> = Vec<RuleLookupSupport<R>>;
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

pub(crate) fn lookup_support_bags_equal<R: Rule>(
    left: &LookupSupports<R>,
    right: &LookupSupports<R>,
) -> bool {
    if left.len() != right.len() {
        return false;
    }

    let mut counts: HashMap<RuleLookupSupport<R>, usize> = HashMap::new();
    for pair in left {
        *counts.entry(pair.clone()).or_default() += 1;
    }
    for pair in right {
        let Some(count) = counts.get_mut(pair) else {
            return false;
        };
        if *count == 1 {
            counts.remove(pair);
        } else {
            *count -= 1;
        }
    }
    counts.is_empty()
}

impl<R: Rule> Context<R> {
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
    let Some(original_env) = ctx
        .goal_for_result_ref(result_ref)
        .map(|goal| Arc::clone(&goal.env))
    else {
        solver_event!(name: "solver.cache_missing_answer_goal");
        return false;
    };
    let mut original_rebased_envs = HashMap::new();
    let mut rebased_envs = HashMap::new();
    for (delta, lookup_support) in &support.checks {
        let original_check_env = match original_rebased_envs.get(delta).cloned() {
            Some(check_env) => check_env,
            None => {
                let check_env = ctx.rebase_env_for_dependency(&original_env, delta);
                original_rebased_envs.insert(delta.clone(), Arc::clone(&check_env));
                check_env
            }
        };
        let check_env = match rebased_envs.get(delta).cloned() {
            Some(check_env) => check_env,
            None => {
                let check_env = ctx.rebase_env_for_dependency(env, delta);
                rebased_envs.insert(delta.clone(), Arc::clone(&check_env));
                check_env
            }
        };
        if original_check_env.lookup_support_matches(
            &check_env,
            &mut ctx.shared_state,
            lookup_support,
        ) {
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
