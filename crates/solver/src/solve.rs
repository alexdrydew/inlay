#![cfg_attr(not(feature = "tracing"), allow(unused_variables, unused_assignments))]

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use inlay_instrument_macros::instrumented;
use thiserror::Error;

use crate::{
    arena::Arena,
    context::{AnswerMatchMemo, Context},
    instrument::{solver_in_span, solver_trace, solver_trace_enabled},
    rule::{
        Lookups, ResolutionEnv, Rule, RuleEnvSharedState, RuleLookupQuery, RuleLookupResult,
        RuleQuery, RuleResult, RuleResultRef, RuleResultsArena,
    },
    search_graph::{CacheKey, Dependency, GoalKey, LazyDepth, Minimums},
    stack::StackError,
};

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq, Hash)]
#[error("same depth cycle")]
pub struct SameDepthCycleError;

#[derive(Debug, Error)]
pub enum SolveQueryError {
    #[error(transparent)]
    SameDepthCycle(#[from] SameDepthCycleError),
    #[error(transparent)]
    Solve(#[from] SolveError),
}

pub(crate) enum GoalSolveResult<R: Rule> {
    Resolved { result_ref: RuleResultRef<R> },
    Lazy { result_ref: RuleResultRef<R> },
    LazyCrossEnv { result_ref: RuleResultRef<R> },
}

pub enum SolveResult<'a, R: Rule> {
    Resolved {
        result: &'a RuleResult<R>,
        result_ref: RuleResultRef<R>,
    },
    Lazy {
        result_ref: RuleResultRef<R>,
    },
}

pub struct SolveOutcome<R: Rule> {
    pub shared_state: RuleEnvSharedState<R>,
    pub result: Result<(RuleResultRef<R>, RuleResultsArena<R>), SolveError>,
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum SolveError {
    #[error("fixpoint iteration limit reached")]
    FixpointIterationLimitReached,
    #[error("stack overflow depth reached")]
    StackOverflowDepthReached,
    #[error("same depth cycle escaped to the root solve")]
    UnexpectedSameDepthCycle,
}

fn replace_result<R: Rule>(
    ctx: &mut Context<R>,
    result_ref: RuleResultRef<R>,
    result: RuleResult<R>,
) {
    ctx.results_arena
        .replace(result_ref, result)
        .expect("solver-managed result ref must remain valid");
}

struct CacheValidationContext<R: Rule> {
    cache_key_label: String,
    candidate_result_ref: RuleResultRef<R>,
    candidate_env_label: String,
    candidate_env_hash: u64,
    exact_env: bool,
}

impl<R: Rule> CacheValidationContext<R> {
    fn new(
        rule: &R,
        cache_key_label: &str,
        candidate_result_ref: RuleResultRef<R>,
        candidate_env: &Arc<R::Env>,
        exact_env: bool,
    ) -> Self {
        Self {
            cache_key_label: cache_key_label.to_string(),
            candidate_result_ref,
            candidate_env_label: debug_env_label(rule, candidate_env.as_ref()),
            candidate_env_hash: debug_env_hash::<R>(candidate_env.as_ref()),
            exact_env,
        }
    }

    fn emit_lookup_miss(
        &self,
        rule: &R,
        result_ref: RuleResultRef<R>,
        env: &Arc<R::Env>,
        depth: usize,
        query: &RuleLookupQuery<R>,
        expected_result: &RuleLookupResult<R>,
        actual_result: &RuleLookupResult<R>,
    ) {
        let env_label = debug_env_label(rule, env.as_ref());
        let lookup_label = debug_lookup_query_label(rule, query);
        let expected_label = debug_lookup_result_label(rule, expected_result);
        let actual_label = debug_lookup_result_label(rule, actual_result);
        solver_trace!(
            name: "solver.cache_lookup_miss",
            cache_key_label = self.cache_key_label.as_str(),
            ?result_ref,
            candidate_result_ref = ?self.candidate_result_ref,
            candidate_env_hash = self.candidate_env_hash,
            candidate_env_label = self.candidate_env_label.as_str(),
            exact_env = self.exact_env,
            env_hash = debug_env_hash::<R>(env.as_ref()),
            env_label = env_label.as_str(),
            depth = depth as u64,
            lookup_hash = hash_value(query),
            lookup_label = lookup_label.as_str(),
            expected_hash = hash_value(expected_result),
            expected_label = expected_label.as_str(),
            actual_hash = hash_value(actual_result),
            actual_label = actual_label.as_str()
        );
    }

    fn emit_missing_answer(&self, env: &Arc<R::Env>, depth: usize) {
        solver_trace!(
            name: "solver.cache_missing_answer",
            cache_key_label = self.cache_key_label.as_str(),
            candidate_result_ref = ?self.candidate_result_ref,
            candidate_env_hash = self.candidate_env_hash,
            candidate_env_label = self.candidate_env_label.as_str(),
            exact_env = self.exact_env,
            env_hash = debug_env_hash::<R>(env.as_ref()),
            depth = depth as u64
        );
    }

    fn emit_dependency_miss(
        &self,
        env: &Arc<R::Env>,
        depth: usize,
        dependency_result_ref: RuleResultRef<R>,
    ) {
        solver_trace!(
            name: "solver.cache_dependency_miss",
            cache_key_label = self.cache_key_label.as_str(),
            candidate_result_ref = ?self.candidate_result_ref,
            candidate_env_hash = self.candidate_env_hash,
            candidate_env_label = self.candidate_env_label.as_str(),
            exact_env = self.exact_env,
            env_hash = debug_env_hash::<R>(env.as_ref()),
            depth = depth as u64,
            dependency_result_ref = ?dependency_result_ref
        );
    }
}

pub(crate) fn debug_lookup_query_label<R: Rule>(rule: &R, query: &RuleLookupQuery<R>) -> String {
    rule.debug_lookup_query_label(query)
        .unwrap_or_else(|| format!("lookup={:x}", hash_value(query)))
}

pub(crate) fn debug_lookup_result_label<R: Rule>(rule: &R, result: &RuleLookupResult<R>) -> String {
    rule.debug_lookup_result_label(result)
        .unwrap_or_else(|| format!("result={:x}", hash_value(result)))
}

fn debug_result_query_label<R: Rule>(
    rule: &R,
    result_ref: RuleResultRef<R>,
    ctx: &Context<R>,
) -> String {
    ctx.goal_for_result_ref(result_ref)
        .map(|goal| debug_cache_key_label::<R>(rule, &goal.query, goal.state_id))
        .unwrap_or_else(|| format!("result_ref={:?}", result_ref))
}

fn lookups_match_env<R: Rule>(
    rule: &R,
    lookups: &Lookups<R>,
    env: &Arc<R::Env>,
    ctx: &mut Context<R>,
    result_ref: RuleResultRef<R>,
    depth: usize,
    trace: Option<&CacheValidationContext<R>>,
) -> bool {
    let mut mismatch = None;
    for (query, expected_result) in lookups {
        let actual_result = env.lookup(&mut ctx.shared_state, query);
        if actual_result != *expected_result {
            mismatch = Some((query.clone(), expected_result.clone(), actual_result));
            break;
        }
    }
    let Some((query, expected_result, actual_result)) = mismatch else {
        return true;
    };

    if let Some(trace) = trace {
        trace.emit_lookup_miss(
            rule,
            result_ref,
            env,
            depth,
            &query,
            &expected_result,
            &actual_result,
        );
    }

    false
}

fn answer_matches_env<R: Rule>(
    rule: &R,
    result_ref: RuleResultRef<R>,
    env: &Arc<R::Env>,
    ctx: &mut Context<R>,
    depth: usize,
    trace: Option<&CacheValidationContext<R>>,
) -> bool {
    let env_hash = debug_env_hash::<R>(env.as_ref());
    let result_query = solver_trace_enabled!()
        .then(|| debug_result_query_label(rule, result_ref, ctx))
        .unwrap_or_default();

    solver_in_span!(
        "solver.answer_match",
        {
            ?result_ref,
            env_hash,
            depth = depth as u64,
            result_query = result_query.as_str()
        },
        {
            match ctx
                .answer_match_memo
                .get(&(result_ref, Arc::clone(env)))
                .copied()
            {
                Some(AnswerMatchMemo::Resolved(matches)) => {
                    solver_trace!(
                        name: "solver.answer_match_memo",
                        ?result_ref,
                        env_hash,
                        depth = depth as u64,
                        matched = matches
                    );
                    return matches;
                }
                Some(AnswerMatchMemo::InProgress) => {
                    solver_trace!(
                        name: "solver.answer_match_in_progress",
                        ?result_ref,
                        env_hash,
                        depth = depth as u64
                    );
                    return true;
                }
                None => {}
            }

            ctx.answer_match_memo
                .insert((result_ref, Arc::clone(env)), AnswerMatchMemo::InProgress);

            let Some(answer) = ctx.answer_for(result_ref).cloned() else {
                if let Some(trace) = trace {
                    trace.emit_missing_answer(env, depth);
                }
                solver_trace!(
                    name: "solver.answer_match_result",
                    ?result_ref,
                    env_hash,
                    depth = depth as u64,
                    dependency_count = 0_u64,
                    matched = false,
                    reason = "missing_answer"
                );
                ctx.answer_match_memo.insert(
                    (result_ref, Arc::clone(env)),
                    AnswerMatchMemo::Resolved(false),
                );
                return false;
            };

            let mut failure_reason = None;
            let matches = if !lookups_match_env(
                rule,
                &answer.lookups,
                env,
                ctx,
                result_ref,
                depth,
                trace,
            ) {
                failure_reason = Some("lookup_mismatch");
                false
            } else {
                let mut matches = true;
                for dependency in &answer.dependencies {
                    let dependency_env = ctx.rebased_env_for_dependency(env, &dependency.env_delta);
                    if !answer_matches_env(
                        rule,
                        dependency.result_ref,
                        &dependency_env,
                        ctx,
                        depth + 1,
                        trace,
                    ) {
                        if let Some(trace) = trace {
                            trace.emit_dependency_miss(env, depth, dependency.result_ref);
                        }
                        failure_reason = Some("dependency_mismatch");
                        matches = false;
                        break;
                    }
                }
                matches
            };
            solver_trace!(
                name: "solver.answer_match_result",
                ?result_ref,
                env_hash,
                depth = depth as u64,
                dependency_count = answer.dependencies.len() as u64,
                matched = matches,
                reason = failure_reason.unwrap_or("matched")
            );
            ctx.answer_match_memo.insert(
                (result_ref, Arc::clone(env)),
                AnswerMatchMemo::Resolved(matches),
            );
            matches
        }
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ActiveAnswerMatch {
    Matches,
    Mismatch,
    Unknown,
}

#[derive(Clone, Copy)]
enum ActiveAnswerMatchMemo {
    InProgress,
    Resolved(ActiveAnswerMatch),
}

// This memo is valid only within a single backreference validation pass.
// The current answer graph is read-only while this function runs, and the memo
// is discarded before any fixpoint rerun can replace results. `InProgress`
// therefore only guards recursive walks over the current graph snapshot.
fn answer_matches_env_for_backref<R: Rule>(
    rule: &R,
    result_ref: RuleResultRef<R>,
    env: &Arc<R::Env>,
    ctx: &mut Context<R>,
    memo: &mut HashMap<(RuleResultRef<R>, Arc<R::Env>), ActiveAnswerMatchMemo>,
) -> ActiveAnswerMatch {
    let key = (result_ref, Arc::clone(env));
    match memo.get(&key).copied() {
        Some(ActiveAnswerMatchMemo::Resolved(result)) => return result,
        Some(ActiveAnswerMatchMemo::InProgress) => return ActiveAnswerMatch::Matches,
        None => {}
    }

    memo.insert(key.clone(), ActiveAnswerMatchMemo::InProgress);

    let Some(answer) = ctx.answer_for(result_ref).cloned() else {
        memo.insert(
            key,
            ActiveAnswerMatchMemo::Resolved(ActiveAnswerMatch::Unknown),
        );
        return ActiveAnswerMatch::Unknown;
    };

    if !lookups_match_env(rule, &answer.lookups, env, ctx, result_ref, 0, None) {
        memo.insert(
            key,
            ActiveAnswerMatchMemo::Resolved(ActiveAnswerMatch::Mismatch),
        );
        return ActiveAnswerMatch::Mismatch;
    }

    let mut saw_unknown = false;
    for dependency in answer.dependencies {
        let dependency_env = ctx.rebased_env_for_dependency(env, &dependency.env_delta);
        match answer_matches_env_for_backref(
            rule,
            dependency.result_ref,
            &dependency_env,
            ctx,
            memo,
        ) {
            ActiveAnswerMatch::Matches => {}
            ActiveAnswerMatch::Mismatch => {
                memo.insert(
                    key,
                    ActiveAnswerMatchMemo::Resolved(ActiveAnswerMatch::Mismatch),
                );
                return ActiveAnswerMatch::Mismatch;
            }
            ActiveAnswerMatch::Unknown => saw_unknown = true,
        }
    }

    let result = if saw_unknown {
        ActiveAnswerMatch::Unknown
    } else {
        ActiveAnswerMatch::Matches
    };
    memo.insert(key, ActiveAnswerMatchMemo::Resolved(result));
    result
}

fn validate_cross_env_reuses_in_suffix<R: Rule>(
    rule: &R,
    dfn: crate::search_graph::DepthFirstNumber,
    ctx: &mut Context<R>,
) -> bool {
    let mut validation_memo = HashMap::new();
    let mut blocked_grew = false;

    for (result_ref, env) in ctx.search_graph.suffix_cross_env_reuses(dfn) {
        let blocked_key = (result_ref, Arc::clone(&env));
        if ctx.blocked_cross_env_reuses.contains(&blocked_key) {
            continue;
        }

        if answer_matches_env_for_backref(rule, result_ref, &env, ctx, &mut validation_memo)
            == ActiveAnswerMatch::Mismatch
        {
            blocked_grew |= ctx.blocked_cross_env_reuses.insert(blocked_key);
        }
    }

    blocked_grew
}

fn cache_key<R: Rule>(goal: &GoalKey<R>) -> CacheKey<R> {
    (goal.query.clone(), goal.state_id)
}

type SuffixSnapshot<R> = HashMap<RuleResultRef<R>, RuleResult<R>>;

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
    result_answers: &'a HashMap<RuleResultRef<R>, crate::search_graph::Answer<R>>,
    persistent_fingerprints: &'a mut HashMap<RuleResultRef<R>, u64>,
}

impl<R: Rule> ContextView<'_, R> {
    fn answer_for(&self, result_ref: RuleResultRef<R>) -> Option<&crate::search_graph::Answer<R>> {
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
        let lookups = answer.lookups.clone();
        let dependencies = answer.dependencies.clone();

        let lookup_hashes = lookups.iter().map(hash_value).collect::<Vec<_>>();
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
            hash_sorted_hashes(lookup_hashes),
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
        let left_lookups = left_answer.lookups.clone();
        let right_lookups = right_answer.lookups.clone();
        let left_dependencies = left_answer.dependencies.clone();
        let right_dependencies = right_answer.dependencies.clone();

        if !lookup_bags_equal::<R>(&left_lookups, &right_lookups) {
            return false;
        }

        dependencies_bag_equal(&left_dependencies, &right_dependencies, ctx, self)
    }
}

pub(crate) fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

pub(crate) fn debug_env_hash<R: Rule>(env: &R::Env) -> u64 {
    hash_value(env)
}

pub(crate) fn debug_env_label<R: Rule>(rule: &R, env: &R::Env) -> String {
    rule.debug_env_label(env)
        .unwrap_or_else(|| format!("env={:x}", debug_env_hash::<R>(env)))
}

fn hash_sorted_hashes(mut values: Vec<u64>) -> u64 {
    values.sort_unstable();
    hash_value(&values)
}

fn lookup_bags_equal<R: Rule>(left: &Lookups<R>, right: &Lookups<R>) -> bool {
    if left.len() != right.len() {
        return false;
    }

    let mut counts: HashMap<(RuleLookupQuery<R>, RuleLookupResult<R>), usize> = HashMap::new();
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
    cache: &mut HashMap<CacheKey<R>, crate::search_graph::CacheBucket<R>>,
    key: CacheKey<R>,
    env: Arc<R::Env>,
    result_ref: RuleResultRef<R>,
    results_arena: &RuleResultsArena<R>,
    result_answers: &HashMap<RuleResultRef<R>, crate::search_graph::Answer<R>>,
    persistent_fingerprints: &mut HashMap<RuleResultRef<R>, u64>,
    dedup: &mut CacheDedupState<R>,
    dedup_enabled: bool,
) {
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
                    return;
                }
            }
        }
    }
    bucket.insert(env, result_ref, fingerprint);
}

fn debug_cache_key_label<R: Rule>(
    rule: &R,
    query: &RuleQuery<R>,
    state_id: R::RuleStateId,
) -> String {
    if let Some(label) = rule.debug_query_label(query, state_id) {
        return label;
    }

    let mut hasher = DefaultHasher::new();
    query.hash(&mut hasher);
    let query_hash = hasher.finish();

    let mut hasher = DefaultHasher::new();
    state_id.hash(&mut hasher);
    let state_hash = hasher.finish();

    format!("query={query_hash:x}#state={state_hash:x}")
}

fn snapshot_suffix<R: Rule>(
    ctx: &Context<R>,
    dfn: crate::search_graph::DepthFirstNumber,
) -> SuffixSnapshot<R> {
    ctx.search_graph
        .suffix_result_refs(dfn)
        .into_iter()
        .map(|result_ref| {
            let result = ctx
                .results_arena
                .get(&result_ref)
                .cloned()
                .expect("solver-managed suffix node must have a stored result");
            (result_ref, result)
        })
        .collect()
}

#[instrumented(
    name = "solver.evaluate_goal",
    level = "trace",
    setup = {
        let goal = ctx.search_graph[dfn].goal.clone();
    },
    trace_setup = {
        let query_hash = hash_value(&goal.query);
        let env_hash = debug_env_hash::<R>(Arc::as_ref(&goal.env));
        let trace_enabled = solver_trace_enabled!();
        let query_label = trace_enabled
            .then(|| debug_cache_key_label::<R>(rule, &goal.query, goal.state_id))
            .unwrap_or_default();
        let env_label = trace_enabled
            .then(|| debug_env_label(rule, Arc::as_ref(&goal.env)))
            .unwrap_or_default();
    },
    fields(
        dfn = dfn.index() as u64,
        query_hash,
        env_hash,
        lazy_depth = goal.lazy_depth.0 as u64,
        query_label = query_label.as_str(),
        env_label = env_label.as_str()
    )
)]
fn evaluate_goal_once<R: Rule>(
    rule: &R,
    dfn: crate::search_graph::DepthFirstNumber,
    ctx: &mut Context<R>,
) -> Result<Minimums, SolveQueryError> {
    let mut minimums = Minimums::new();
    let (lookups, dependencies, cross_env_reuses, result_ref) = {
        let mut rule_ctx =
            crate::rule::RuleContext::new(rule, goal.state_id, goal.env, ctx, dfn, &mut minimums);

        let result = solver_in_span!(
            "solver.rule_run",
            {
                dfn = dfn.index() as u64,
                query_hash,
                env_hash,
                lazy_depth = goal.lazy_depth.0 as u64,
                query_label = query_label.as_str(),
                env_label = env_label.as_str()
            },
            {
                match rule.run(goal.query, &mut rule_ctx) {
                    Ok(output) => Ok(output),
                    Err(crate::rule::RunError::Rule(err)) => Err(err),
                    Err(crate::rule::RunError::Solve(error)) => return Err(error),
                }
            }
        );
        let lookups = rule_ctx.lookups.clone();
        let dependencies: Vec<Dependency<R>> =
            rule_ctx.child_dependencies.iter().cloned().collect();
        let cross_env_reuses: Vec<(RuleResultRef<R>, Arc<R::Env>)> =
            rule_ctx.cross_env_reuses.iter().cloned().collect();
        let result_ref = rule_ctx.ctx.search_graph[dfn].answer.result_ref;

        replace_result(rule_ctx.ctx, result_ref, result);
        (lookups, dependencies, cross_env_reuses, result_ref)
    };

    ctx.search_graph[dfn].answer.lookups = lookups.clone();
    ctx.search_graph[dfn].answer.dependencies = dependencies.clone();
    ctx.search_graph[dfn].cross_env_reuses = cross_env_reuses;
    let lookup_count = lookups.len() as u64;
    let dependency_count = dependencies.len() as u64;
    let cross_env_reuse_count = ctx.search_graph[dfn].cross_env_reuses.len() as u64;
    ctx.replace_answer(
        result_ref,
        crate::search_graph::Answer {
            result_ref,
            lookups,
            dependencies,
        },
    );
    let result_kind = match ctx
        .results_arena
        .get(&result_ref)
        .expect("result just stored")
    {
        Ok(_) => "ok",
        Err(_) => "err",
    };
    solver_trace!(
        name: "solver.goal_eval",
        dfn = dfn.index() as u64,
        ?result_ref,
        query_hash,
        env_hash,
        lazy_depth = goal.lazy_depth.0 as u64,
        query_label = query_label.as_str(),
        env_label = env_label.as_str(),
        lookups = lookup_count,
        dependencies = dependency_count,
        cross_env_reuses = cross_env_reuse_count,
        result_kind
    );
    ctx.search_graph[dfn].links = minimums;
    Ok(minimums)
}

#[instrumented(
    name = "solver.new_goal",
    level = "trace",
    trace_setup = {
        let query_hash = hash_value(&goal.query);
        let env_hash = debug_env_hash::<R>(Arc::as_ref(&goal.env));
        let trace_enabled = solver_trace_enabled!();
        let query_label = trace_enabled
            .then(|| debug_cache_key_label::<R>(rule, &goal.query, goal.state_id))
            .unwrap_or_default();
        let env_label = trace_enabled
            .then(|| debug_env_label(rule, goal.env.as_ref()))
            .unwrap_or_default();
    },
    fields(
        query_hash,
        env_hash,
        lazy_depth = goal.lazy_depth.0 as u64,
        query_label = query_label.as_str(),
        env_label = env_label.as_str()
    )
)]
fn solve_new_goal<R: Rule>(
    rule: &R,
    goal: GoalKey<R>,
    ctx: &mut Context<R>,
) -> Result<(GoalSolveResult<R>, Minimums), SolveQueryError> {
    let result_ref = ctx.result_ref_for(&goal);
    let stack_depth = ctx.stack.push().map_err(|error| match error {
        StackError::Overflow => SolveError::StackOverflowDepthReached,
    })?;
    let dfn = ctx.search_graph.insert(&goal, stack_depth, result_ref);
    solver_trace!(
        name: "solver.new_goal",
        dfn = dfn.index() as u64,
        ?result_ref,
        query_hash,
        env_hash,
        lazy_depth = goal.lazy_depth.0 as u64,
        query_label = query_label.as_str(),
        env_label = env_label.as_str()
    );

    let mut reruns = 0_u64;
    let mut previous_snapshot = None;
    let final_minimums = loop {
        let iteration_minimums = evaluate_goal_once(rule, dfn, ctx)?;
        let blocked_grew = validate_cross_env_reuses_in_suffix(rule, dfn, ctx);

        if !ctx.stack[stack_depth].read_and_reset_cycle_flag() {
            break iteration_minimums;
        }

        let current_snapshot = snapshot_suffix(ctx, dfn);
        let snapshot_unchanged = previous_snapshot
            .as_ref()
            .is_some_and(|previous| previous == &current_snapshot);
        if snapshot_unchanged && !blocked_grew {
            break iteration_minimums;
        }

        if reruns as usize >= ctx.fixpoint_iteration_limit {
            return Err(SolveError::FixpointIterationLimitReached.into());
        }
        reruns += 1;
        solver_trace!(
            name: "solver.fixpoint_rerun",
            dfn = dfn.index() as u64,
            ?result_ref,
            rerun = reruns,
            blocked_grew,
            snapshot_unchanged,
            query_hash,
            env_hash,
            lazy_depth = goal.lazy_depth.0 as u64,
            query_label = query_label.as_str(),
            env_label = env_label.as_str()
        );
        previous_snapshot = Some(current_snapshot);

        ctx.search_graph.rollback_to(dfn + 1);
    };

    ctx.search_graph.pop_stack_goal(dfn);
    ctx.stack.pop(stack_depth);

    if final_minimums.ancestor() >= dfn {
        let cacheable_entries = ctx.search_graph.take_cacheable_entries(dfn);
        let mut dedup = CacheDedupState::new();
        let results_arena = &ctx.results_arena;
        let result_answers = &ctx.result_answers;
        let persistent_fingerprints = &mut ctx.answer_fingerprints;
        let dedup_enabled = ctx.cache_dedup_enabled;
        for (cache_key, env, result_ref) in cacheable_entries {
            insert_cache_entry(
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
        }
    }

    solver_trace!(
        name: "solver.goal_outcome",
        outcome = "new_goal_resolved",
        dfn = dfn.index() as u64,
        ?result_ref,
        reruns,
        query_hash,
        env_hash,
        lazy_depth = goal.lazy_depth.0 as u64,
        query_label = query_label.as_str(),
        env_label = env_label.as_str()
    );

    Ok((GoalSolveResult::Resolved { result_ref }, final_minimums))
}

#[instrumented(
    name = "solver.solve_goal",
    level = "trace",
    setup = {
        let trace_enabled = solver_trace_enabled!();
        let query_label = trace_enabled
            .then(|| debug_cache_key_label::<R>(rule, &goal.query, goal.state_id))
            .unwrap_or_default();
    },
    trace_setup = {
        let query_hash = hash_value(&goal.query);
        let env_hash = debug_env_hash::<R>(Arc::as_ref(&goal.env));
        let env_label = trace_enabled
            .then(|| debug_env_label(rule, goal.env.as_ref()))
            .unwrap_or_default();
    },
    fields(
        query_hash,
        env_hash,
        state_hash = hash_value(&goal.state_id),
        lazy_depth = goal.lazy_depth.0 as u64,
        query_label = query_label.as_str(),
        env_label = env_label.as_str()
    )
)]
pub(crate) fn solve_goal<R: Rule>(
    rule: &R,
    goal: GoalKey<R>,
    ctx: &mut Context<R>,
) -> Result<(GoalSolveResult<R>, Minimums), SolveQueryError> {
    if let Some(ancestor_dfn) = ctx
        .search_graph
        .closest_goal(&goal.query, goal.state_id, &goal.env)
    {
        let (ancestor_lazy_depth, result_ref, stack_depth) = {
            let ancestor_node = &ctx.search_graph[ancestor_dfn];
            (
                ancestor_node.goal.lazy_depth,
                ancestor_node.answer.result_ref,
                ancestor_node
                    .stack_depth
                    .expect("closest active goal must still be on stack"),
            )
        };

        if ancestor_lazy_depth >= goal.lazy_depth {
            solver_trace!(
                name: "solver.goal_outcome",
                outcome = "same_depth_cycle",
                ancestor_dfn = ancestor_dfn.index() as u64,
                ?result_ref,
                query_hash,
                env_hash,
                lazy_depth = goal.lazy_depth.0 as u64,
                query_label = query_label.as_str(),
                env_label = env_label.as_str()
            );
            ctx.stack[stack_depth].flag_cycle();
            return Err(SameDepthCycleError.into());
        }

        ctx.stack[stack_depth].flag_cycle();
        solver_trace!(
            name: "solver.goal_outcome",
            outcome = "active_lazy_hit",
            ancestor_dfn = ancestor_dfn.index() as u64,
            ?result_ref,
            query_hash,
            env_hash,
            lazy_depth = goal.lazy_depth.0 as u64,
            query_label = query_label.as_str(),
            env_label = env_label.as_str()
        );
        return Ok((
            GoalSolveResult::Lazy { result_ref },
            Minimums::from_self(ancestor_dfn),
        ));
    }

    if ctx.cross_env_active_reuse_enabled {
        if let Some(ancestor_dfn) = ctx
            .search_graph
            .closest_goal_any_env(&goal.query, goal.state_id)
        {
            let (ancestor_lazy_depth, result_ref, ancestor_env, stack_depth) = {
                let ancestor_node = &ctx.search_graph[ancestor_dfn];
                (
                    ancestor_node.goal.lazy_depth,
                    ancestor_node.answer.result_ref,
                    Arc::clone(&ancestor_node.goal.env),
                    ancestor_node
                        .stack_depth
                        .expect("closest active goal must still be on stack"),
                )
            };

            if ancestor_env != goal.env && ancestor_lazy_depth < goal.lazy_depth {
                let blocked_key = (result_ref, Arc::clone(&goal.env));
                if !ctx.blocked_cross_env_reuses.contains(&blocked_key) {
                    ctx.stack[stack_depth].flag_cycle();
                    solver_trace!(
                        name: "solver.goal_outcome",
                        outcome = "cross_env_reuse",
                        ancestor_dfn = ancestor_dfn.index() as u64,
                        ?result_ref,
                        ancestor_env_hash = debug_env_hash::<R>(Arc::as_ref(&ancestor_env)),
                        query_hash,
                        env_hash,
                        lazy_depth = goal.lazy_depth.0 as u64,
                        query_label = query_label.as_str(),
                        env_label = env_label.as_str()
                    );
                    return Ok((
                        GoalSolveResult::LazyCrossEnv { result_ref },
                        Minimums::from_self(ancestor_dfn),
                    ));
                }
            }
        }
    }

    if ctx.cache_reuse_enabled {
        let key = cache_key(&goal);
        if let Some((exact_result_refs, bucket_len, entries)) = ctx.cache.get(&key).map(|bucket| {
            (
                bucket.cloned_result_refs_for_env(&goal.env),
                bucket.len(),
                bucket.cloned_entries(),
            )
        }) {
            if let Some(result_refs) = exact_result_refs.as_ref() {
                solver_trace!(
                    name: "solver.cache_probe",
                    query_hash,
                    env_hash,
                    exact_env = true,
                    bucket_len = result_refs.len() as u64,
                    lazy_depth = goal.lazy_depth.0 as u64,
                    query_label = query_label.as_str(),
                    env_label = env_label.as_str()
                );
                for result_ref in result_refs.iter().rev() {
                    let trace = trace_enabled.then(|| {
                        CacheValidationContext::new(
                            rule,
                            &query_label,
                            *result_ref,
                            &goal.env,
                            true,
                        )
                    });
                    let matched =
                        answer_matches_env(rule, *result_ref, &goal.env, ctx, 0, trace.as_ref());
                    solver_trace!(
                        name: "solver.cache_candidate",
                        query_hash,
                        env_hash,
                        exact_env = true,
                        bucket_len = result_refs.len() as u64,
                        matched,
                        result_ref = ?result_ref,
                        candidate_env_hash = env_hash
                    );
                    if matched {
                        solver_trace!(
                            name: "solver.goal_outcome",
                            outcome = "cache_exact_hit",
                            ?result_ref,
                            query_hash,
                            env_hash,
                            lazy_depth = goal.lazy_depth.0 as u64,
                            query_label = query_label.as_str(),
                            env_label = env_label.as_str()
                        );
                        return Ok((
                            GoalSolveResult::Resolved {
                                result_ref: *result_ref,
                            },
                            Minimums::new(),
                        ));
                    }
                }
            }

            solver_trace!(
                name: "solver.cache_probe",
                query_hash,
                env_hash,
                exact_env = false,
                bucket_len = bucket_len as u64,
                lazy_depth = goal.lazy_depth.0 as u64,
                query_label = query_label.as_str(),
                env_label = env_label.as_str()
            );
            for entry in entries.iter().rev() {
                if exact_result_refs.is_some() && entry.env == goal.env {
                    continue;
                }
                let trace = trace_enabled.then(|| {
                    CacheValidationContext::new(
                        rule,
                        &query_label,
                        entry.result_ref,
                        &entry.env,
                        false,
                    )
                });
                let matched =
                    answer_matches_env(rule, entry.result_ref, &goal.env, ctx, 0, trace.as_ref());
                let candidate_env_hash = debug_env_hash::<R>(Arc::as_ref(&entry.env));
                solver_trace!(
                    name: "solver.cache_candidate",
                    query_hash,
                    env_hash,
                    exact_env = false,
                    bucket_len = bucket_len as u64,
                    matched,
                    result_ref = ?entry.result_ref,
                    candidate_env_hash
                );
                if matched {
                    solver_trace!(
                        name: "solver.goal_outcome",
                        outcome = "cache_hit",
                        result_ref = ?entry.result_ref,
                        candidate_env_hash,
                        query_hash,
                        env_hash,
                        lazy_depth = goal.lazy_depth.0 as u64,
                        query_label = query_label.as_str(),
                        env_label = env_label.as_str()
                    );
                    return Ok((
                        GoalSolveResult::Resolved {
                            result_ref: entry.result_ref,
                        },
                        Minimums::new(),
                    ));
                }
            }
        }
    }

    if let Some(dfn) = ctx.search_graph.lookup(&goal) {
        let node = &ctx.search_graph[dfn];
        if let Some(stack_depth) = node.stack_depth {
            ctx.stack[stack_depth].flag_cycle();
        }
        solver_trace!(
            name: "solver.goal_outcome",
            outcome = "graph_goal_hit",
            dfn = dfn.index() as u64,
            result_ref = ?node.answer.result_ref,
            query_hash,
            env_hash,
            lazy_depth = goal.lazy_depth.0 as u64,
            query_label = query_label.as_str(),
            env_label = env_label.as_str()
        );
        return Ok((
            GoalSolveResult::Resolved {
                result_ref: node.answer.result_ref,
            },
            node.links,
        ));
    }

    solve_new_goal(rule, goal, ctx)
}

#[instrumented(
    name = "solver.solve",
    level = "trace",
    setup = {
        let root_goal = GoalKey {
            query,
            state_id: initial_rule,
            env,
            lazy_depth: LazyDepth(0),
        };
    },
    trace_setup = {
        let root_query_hash = hash_value(&root_goal.query);
        let root_env_hash = debug_env_hash::<R>(Arc::as_ref(&root_goal.env));
        let trace_enabled = solver_trace_enabled!();
        let root_query_label = trace_enabled
            .then(|| debug_cache_key_label::<R>(rule, &root_goal.query, root_goal.state_id))
            .unwrap_or_default();
        let root_env_label = trace_enabled
            .then(|| debug_env_label(rule, Arc::as_ref(&root_goal.env)))
            .unwrap_or_default();
    },
    fields(
        query_hash = root_query_hash,
        env_hash = root_env_hash,
        state_hash = hash_value(&root_goal.state_id),
        lazy_depth = 0,
        query_label = root_query_label.as_str(),
        env_label = root_env_label.as_str()
    )
)]
pub fn solve<R: Rule>(
    rule: &R,
    query: RuleQuery<R>,
    initial_rule: R::RuleStateId,
    env: Arc<R::Env>,
    shared_state: RuleEnvSharedState<R>,
    fixpoint_iteration_limit: usize,
    stack_depth_limit: usize,
) -> SolveOutcome<R> {
    let mut ctx = Context::new(shared_state, fixpoint_iteration_limit, stack_depth_limit);
    let result = solve_goal(rule, root_goal, &mut ctx)
        .map(|(solve_result, _minimums)| match solve_result {
            GoalSolveResult::Resolved { result_ref } => result_ref,
            GoalSolveResult::Lazy { .. } | GoalSolveResult::LazyCrossEnv { .. } => {
                unreachable!(
                    "root solve_goal cannot resolve lazily because lazy results require an active ancestor"
                )
            }
        })
        .map_err(|error| match error {
            SolveQueryError::SameDepthCycle(_) => SolveError::UnexpectedSameDepthCycle,
            SolveQueryError::Solve(error) => error,
        });

    solver_trace!(
        name: "solver.solve_result",
        query_hash = root_query_hash,
        env_hash = root_env_hash,
        query_label = root_query_label.as_str(),
        env_label = root_env_label.as_str(),
        ok = result.is_ok(),
        error = if result.is_err() { "solve_error" } else { "" }
    );

    let Context {
        results_arena,
        shared_state,
        ..
    } = ctx;
    SolveOutcome {
        shared_state,
        result: result.map(|root_result_ref| (root_result_ref, results_arena)),
    }
}
