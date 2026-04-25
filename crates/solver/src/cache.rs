use std::collections::HashMap;
use std::sync::Arc;

use inlay_instrument_macros::instrumented;

use crate::{
    context::Context,
    instrument::solver_span_record,
    rule::{LookupSupports, RuleLookupSupport, RuleResultRef, RuleResultsArena},
    search_graph::{Answer, CacheBucket, CacheKey, Dependency, GoalKey},
    solve::hash_value,
    traits::{Arena, Rule},
};

#[derive(Clone, Copy)]
enum FingerprintMemoEntry {
    InProgress(u64),
    Resolved(u64),
}

#[derive(Clone, Copy)]
enum StructuralEqMemoEntry {
    InProgress,
    Resolved(bool),
}

struct ContextView<'a, R: Rule> {
    results_arena: &'a RuleResultsArena<R>,
    result_answers: &'a HashMap<RuleResultRef<R>, Answer<R>>,
    persistent_fingerprints: &'a mut HashMap<RuleResultRef<R>, u64>,
}

impl<R: Rule> ContextView<'_, R> {
    fn answer_for(&self, result_ref: RuleResultRef<R>) -> Option<&Answer<R>> {
        self.result_answers.get(&result_ref)
    }
}

struct CacheDedupState<R: Rule> {
    fingerprint_memo: HashMap<RuleResultRef<R>, FingerprintMemoEntry>,
    structural_eq_memo: HashMap<(RuleResultRef<R>, RuleResultRef<R>), StructuralEqMemoEntry>,
    next_cycle_id: u64,
}

impl<R: Rule> CacheDedupState<R> {
    fn new() -> Self {
        Self {
            fingerprint_memo: HashMap::new(),
            structural_eq_memo: HashMap::new(),
            next_cycle_id: 0,
        }
    }

    fn fingerprint(&mut self, result_ref: RuleResultRef<R>, ctx: &mut ContextView<'_, R>) -> u64 {
        match self.fingerprint_memo.get(&result_ref).copied() {
            Some(FingerprintMemoEntry::Resolved(fingerprint)) => return fingerprint,
            Some(FingerprintMemoEntry::InProgress(cycle_id)) => {
                return hash_value(&("cycle", cycle_id));
            }
            None => {}
        }

        if let Some(fingerprint) = ctx.persistent_fingerprints.get(&result_ref).copied() {
            self.fingerprint_memo
                .insert(result_ref, FingerprintMemoEntry::Resolved(fingerprint));
            return fingerprint;
        }

        self.next_cycle_id += 1;
        let cycle_id = self.next_cycle_id;
        self.fingerprint_memo
            .insert(result_ref, FingerprintMemoEntry::InProgress(cycle_id));

        let Some(result) = ctx.results_arena.get(&result_ref) else {
            let fingerprint = hash_value(&("missing-result", result_ref));
            ctx.persistent_fingerprints.insert(result_ref, fingerprint);
            self.fingerprint_memo
                .insert(result_ref, FingerprintMemoEntry::Resolved(fingerprint));
            return fingerprint;
        };
        let Some(answer) = ctx.answer_for(result_ref) else {
            let fingerprint = hash_value(&("missing-answer", result_ref));
            ctx.persistent_fingerprints.insert(result_ref, fingerprint);
            self.fingerprint_memo
                .insert(result_ref, FingerprintMemoEntry::Resolved(fingerprint));
            return fingerprint;
        };
        let direct_supports = answer.direct_supports.clone();
        let dependencies = answer.dependencies.clone();

        let direct_support_hashes = direct_supports.iter().map(hash_value).collect::<Vec<_>>();
        let dependency_hashes = dependencies
            .iter()
            .map(|dependency| {
                hash_value(&(
                    self.fingerprint(dependency.result_ref, ctx),
                    &dependency.env_delta,
                ))
            })
            .collect::<Vec<_>>();

        let fingerprint = hash_value(&(
            hash_value(result),
            hash_sorted_hashes(direct_support_hashes),
            hash_sorted_hashes(dependency_hashes),
        ));
        ctx.persistent_fingerprints.insert(result_ref, fingerprint);
        self.fingerprint_memo
            .insert(result_ref, FingerprintMemoEntry::Resolved(fingerprint));
        fingerprint
    }

    fn structurally_equal(
        &mut self,
        left: RuleResultRef<R>,
        right: RuleResultRef<R>,
        ctx: &mut ContextView<'_, R>,
    ) -> bool {
        if left == right {
            return true;
        }

        let key = (left, right);
        match self.structural_eq_memo.get(&key).copied() {
            Some(StructuralEqMemoEntry::Resolved(equal)) => return equal,
            Some(StructuralEqMemoEntry::InProgress) => return true,
            None => {}
        }

        self.structural_eq_memo
            .insert(key, StructuralEqMemoEntry::InProgress);
        self.structural_eq_memo
            .insert((right, left), StructuralEqMemoEntry::InProgress);

        let result = self.structurally_equal_impl(left, right, ctx);
        self.structural_eq_memo
            .insert(key, StructuralEqMemoEntry::Resolved(result));
        self.structural_eq_memo
            .insert((right, left), StructuralEqMemoEntry::Resolved(result));
        result
    }

    fn structurally_equal_impl(
        &mut self,
        left: RuleResultRef<R>,
        right: RuleResultRef<R>,
        ctx: &mut ContextView<'_, R>,
    ) -> bool {
        let Some(left_result) = ctx.results_arena.get(&left) else {
            return false;
        };
        let Some(right_result) = ctx.results_arena.get(&right) else {
            return false;
        };
        if left_result != right_result {
            return false;
        }

        let Some(left_answer) = ctx.answer_for(left) else {
            return false;
        };
        let Some(right_answer) = ctx.answer_for(right) else {
            return false;
        };
        let left_direct_supports = left_answer.direct_supports.clone();
        let right_direct_supports = right_answer.direct_supports.clone();
        let left_dependencies = left_answer.dependencies.clone();
        let right_dependencies = right_answer.dependencies.clone();

        if !lookup_support_bags_equal::<R>(&left_direct_supports, &right_direct_supports) {
            return false;
        }

        dependencies_bag_equal(&left_dependencies, &right_dependencies, ctx, self)
    }
}

pub(crate) fn cache_key<R: Rule>(goal: &GoalKey<R>) -> CacheKey<R> {
    (goal.query.clone(), goal.state_id)
}

#[instrumented(
    name = "solver.insert_cache_entries",
    target = "context_solver",
    level = "trace",
    skip(ctx, entries),
    fields(entries, inserted, dedup_skipped)
)]
pub(crate) fn insert_cache_entries<R: Rule>(
    ctx: &mut Context<R>,
    entries: Vec<(CacheKey<R>, Arc<R::Env>, RuleResultRef<R>)>,
) {
    solver_span_record!(entries = entries.len() as u64);
    let mut dedup = CacheDedupState::new();
    let results_arena = &ctx.results_arena;
    let result_answers = &ctx.result_answers;
    let persistent_fingerprints = &mut ctx.answer_fingerprints;
    let dedup_enabled = ctx.cache_dedup_enabled;
    #[cfg(feature = "tracing")]
    let mut inserted = 0_u64;
    #[cfg(feature = "tracing")]
    let mut dedup_skipped = 0_u64;

    for (cache_key, env, result_ref) in entries {
        let cache_inserted = insert_cache_entry(
            &mut ctx.cache,
            cache_key,
            env,
            result_ref,
            results_arena,
            result_answers,
            persistent_fingerprints,
            &mut dedup,
            dedup_enabled,
        );
        #[cfg(feature = "tracing")]
        if cache_inserted {
            inserted += 1;
        } else {
            dedup_skipped += 1;
        }
        #[cfg(not(feature = "tracing"))]
        let _ = cache_inserted;
    }
    solver_span_record!(inserted, dedup_skipped);
}

fn hash_sorted_hashes(mut values: Vec<u64>) -> u64 {
    values.sort_unstable();
    hash_value(&values)
}

fn lookup_support_bags_equal<R: Rule>(
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

fn dependencies_bag_equal<R: Rule>(
    left: &[Dependency<R>],
    right: &[Dependency<R>],
    ctx: &mut ContextView<'_, R>,
    dedup: &mut CacheDedupState<R>,
) -> bool {
    if left.len() != right.len() {
        return false;
    }

    let mut right_groups: HashMap<u64, Vec<Dependency<R>>> = HashMap::new();
    for dependency in right {
        let dependency_hash = hash_value(&(
            dedup.fingerprint(dependency.result_ref, ctx),
            &dependency.env_delta,
        ));
        right_groups
            .entry(dependency_hash)
            .or_default()
            .push(dependency.clone());
    }

    for dependency in left {
        let dependency_hash = hash_value(&(
            dedup.fingerprint(dependency.result_ref, ctx),
            &dependency.env_delta,
        ));
        let Some(group) = right_groups.get_mut(&dependency_hash) else {
            return false;
        };

        let Some(index) = group.iter().position(|candidate| {
            dependency.env_delta == candidate.env_delta
                && dedup.structurally_equal(dependency.result_ref, candidate.result_ref, ctx)
        }) else {
            return false;
        };
        group.swap_remove(index);
        if group.is_empty() {
            right_groups.remove(&dependency_hash);
        }
    }

    right_groups.is_empty()
}

fn insert_cache_entry<R: Rule>(
    cache: &mut HashMap<CacheKey<R>, CacheBucket<R>>,
    key: CacheKey<R>,
    env: Arc<R::Env>,
    result_ref: RuleResultRef<R>,
    results_arena: &RuleResultsArena<R>,
    result_answers: &HashMap<RuleResultRef<R>, Answer<R>>,
    persistent_fingerprints: &mut HashMap<RuleResultRef<R>, u64>,
    dedup: &mut CacheDedupState<R>,
    dedup_enabled: bool,
) -> bool {
    let mut ctx = ContextView {
        results_arena,
        result_answers,
        persistent_fingerprints,
    };
    let fingerprint = if dedup_enabled {
        dedup.fingerprint(result_ref, &mut ctx)
    } else {
        0
    };
    let bucket = cache.entry(key).or_default();
    if dedup_enabled {
        if let Some(indices) = bucket.cloned_indices_for_env_fingerprint(&env, fingerprint) {
            let entries = bucket.cloned_entries();
            for index in indices {
                let candidate = entries[index].result_ref;
                if dedup.structurally_equal(candidate, result_ref, &mut ctx) {
                    return false;
                }
            }
        }
    }
    bucket.insert(env, result_ref, fingerprint);
    true
}
