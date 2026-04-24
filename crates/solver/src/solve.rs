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
    instrument::{solver_event, solver_in_span},
    rule::{
        Lookups, ResolutionEnv, Rule, RuleEnvSharedState, RuleLookupQuery, RuleLookupResult,
        RuleQuery, RuleResult, RuleResultRef, RuleResultsArena,
    },
    search_graph::{CacheKey, Dependency, GoalKey, LazyDepth, Minimums},
    stack::StackError,
};

#[cfg(feature = "tracing")]
use crate::instrument::solver_trace_enabled;

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

impl<R: Rule> std::fmt::Debug for SolveOutcome<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.result {
            Ok((result_ref, results_arena)) => f
                .debug_struct("SolveOutcome")
                .field("ok", &true)
                .field("result_ref", result_ref)
                .field("results", &results_arena.len())
                .finish(),
            Err(error) => f
                .debug_struct("SolveOutcome")
                .field("ok", &false)
                .field("error", error)
                .finish(),
        }
    }
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum SolveError {
    #[error("fixpoint iteration limit reached")]
    FixpointIterationLimitReached,
    #[error("stack overflow depth reached")]
    StackOverflowDepthReached,
    #[error("same depth cycle")]
    SameDepthCycle,
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

    #[cfg(feature = "tracing")]
    {
        let trace_enabled = solver_trace_enabled!();
        let lookup_label = trace_enabled
            .then(|| debug_lookup_query_label(rule, &query))
            .unwrap_or_default();
        let expected_label = trace_enabled
            .then(|| debug_lookup_result_label(rule, &expected_result))
            .unwrap_or_default();
        let actual_label = trace_enabled
            .then(|| debug_lookup_result_label(rule, &actual_result))
            .unwrap_or_default();
        solver_event!(
            name: "solver.cache_lookup_miss",
            lookup_hash = hash_value(&query),
            lookup_label = lookup_label.as_str(),
            expected_hash = hash_value(&expected_result),
            expected_label = expected_label.as_str(),
            actual_hash = hash_value(&actual_result),
            actual_label = actual_label.as_str()
        );
    }

    false
}

#[instrumented(
    name = "solver.answer_match",
    target = "context_solver",
    level = "trace",
    ret,
    fields(
        result_ref = ?result_ref,
        env_hash = debug_env_hash::<R>(Arc::as_ref(env)),
        depth = depth as u64,
        result_query = %trace_result_query_label::<R>(rule, result_ref, ctx),
        env_label = %trace_env_label::<R>(rule, Arc::as_ref(env))
    )
)]
fn answer_matches_env<R: Rule>(
    rule: &R,
    result_ref: RuleResultRef<R>,
    env: &Arc<R::Env>,
    ctx: &mut Context<R>,
    depth: usize,
) -> bool {
    match ctx
        .answer_match_memo
        .get(&(result_ref, Arc::clone(env)))
        .copied()
    {
        Some(AnswerMatchMemo::Resolved(matches)) => {
            solver_event!(
                name: "solver.answer_match_memo",
                matched = matches
            );
            return matches;
        }
        Some(AnswerMatchMemo::InProgress) => {
            solver_event!(name: "solver.answer_match_in_progress");
            return true;
        }
        None => {}
    }

    ctx.answer_match_memo
        .insert((result_ref, Arc::clone(env)), AnswerMatchMemo::InProgress);

    let Some(answer) = ctx.answer_for(result_ref).cloned() else {
        solver_event!(name: "solver.cache_missing_answer");
        ctx.answer_match_memo.insert(
            (result_ref, Arc::clone(env)),
            AnswerMatchMemo::Resolved(false),
        );
        return false;
    };

    let matches = if !lookups_match_env(rule, &answer.lookups, env, ctx) {
        false
    } else {
        let mut matches = true;
        for dependency in &answer.dependencies {
            let dependency_env = ctx.rebased_env_for_dependency(env, &dependency.env_delta);
            if !answer_matches_env(rule, dependency.result_ref, &dependency_env, ctx, depth + 1) {
                solver_event!(
                    name: "solver.cache_dependency_miss",
                    dependency_result_ref = ?dependency.result_ref
                );
                matches = false;
                break;
            }
        }
        matches
    };
    ctx.answer_match_memo.insert(
        (result_ref, Arc::clone(env)),
        AnswerMatchMemo::Resolved(matches),
    );
    matches
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

    if !lookups_match_env(rule, &answer.lookups, env, ctx) {
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

#[cfg(feature = "tracing")]
fn trace_query_label<R: Rule>(rule: &R, query: &RuleQuery<R>, state_id: R::RuleStateId) -> String {
    solver_trace_enabled!()
        .then(|| debug_cache_key_label::<R>(rule, query, state_id))
        .unwrap_or_default()
}

#[cfg(feature = "tracing")]
fn trace_env_label<R: Rule>(rule: &R, env: &R::Env) -> String {
    solver_trace_enabled!()
        .then(|| debug_env_label(rule, env))
        .unwrap_or_default()
}

#[cfg(feature = "tracing")]
fn trace_result_query_label<R: Rule>(
    rule: &R,
    result_ref: RuleResultRef<R>,
    ctx: &Context<R>,
) -> String {
    solver_trace_enabled!()
        .then(|| debug_result_query_label(rule, result_ref, ctx))
        .unwrap_or_default()
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
    target = "context_solver",
    level = "trace",
    fields(
        dfn = dfn.index() as u64,
        query_hash = hash_value(&ctx.search_graph[dfn].goal.query),
        env_hash = debug_env_hash::<R>(Arc::as_ref(&ctx.search_graph[dfn].goal.env)),
        lazy_depth = ctx.search_graph[dfn].goal.lazy_depth.0 as u64,
        query_label = %trace_query_label::<R>(
            rule,
            &ctx.search_graph[dfn].goal.query,
            ctx.search_graph[dfn].goal.state_id,
        ),
        env_label = %trace_env_label::<R>(rule, Arc::as_ref(&ctx.search_graph[dfn].goal.env))
    )
)]
fn evaluate_goal_once<R: Rule>(
    rule: &R,
    dfn: crate::search_graph::DepthFirstNumber,
    ctx: &mut Context<R>,
) -> Result<Minimums, SolveError> {
    let goal = ctx.search_graph[dfn].goal.clone();

    let mut minimums = Minimums::new();
    let (lookups, dependencies, cross_env_reuses, result_ref) = {
        let mut rule_ctx =
            crate::rule::RuleContext::new(rule, goal.state_id, goal.env, ctx, dfn, &mut minimums);

        let result = solver_in_span!("solver.rule_run", {}, {
            match rule.run(goal.query, &mut rule_ctx) {
                Ok(output) => Ok(output),
                Err(crate::rule::RunError::Rule(err)) => Err(err),
                Err(crate::rule::RunError::Solve(error)) => return Err(error),
            }
        });
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
    solver_event!(
        name: "solver.goal_eval",
        ?result_ref,
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
    target = "context_solver",
    level = "trace",
    fields(
        query_hash = hash_value(&goal.query),
        env_hash = debug_env_hash::<R>(Arc::as_ref(&goal.env)),
        lazy_depth = goal.lazy_depth.0 as u64,
        query_label = %trace_query_label::<R>(rule, &goal.query, goal.state_id),
        env_label = %trace_env_label::<R>(rule, Arc::as_ref(&goal.env))
    )
)]
fn solve_new_goal<R: Rule>(
    rule: &R,
    goal: GoalKey<R>,
    ctx: &mut Context<R>,
) -> Result<(GoalSolveResult<R>, Minimums), SolveError> {
    let result_ref = ctx.result_ref_for(&goal);
    let stack_depth = ctx.stack.push().map_err(|error| match error {
        StackError::Overflow => SolveError::StackOverflowDepthReached,
    })?;
    let dfn = ctx.search_graph.insert(&goal, stack_depth, result_ref);
    solver_event!(
        name: "solver.new_goal",
        dfn = dfn.index() as u64,
        ?result_ref
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
            return Err(SolveError::FixpointIterationLimitReached);
        }
        reruns += 1;
        solver_event!(
            name: "solver.fixpoint_rerun",
            dfn = dfn.index() as u64,
            ?result_ref,
            rerun = reruns,
            blocked_grew,
            snapshot_unchanged
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

    solver_event!(
        name: "solver.goal_outcome",
        outcome = "new_goal_resolved",
        dfn = dfn.index() as u64,
        ?result_ref,
        reruns
    );

    Ok((GoalSolveResult::Resolved { result_ref }, final_minimums))
}

fn try_close_active_cycle<R: Rule>(
    goal: &GoalKey<R>,
    ctx: &mut Context<R>,
) -> Result<Option<(GoalSolveResult<R>, Minimums)>, SolveError> {
    let Some(ancestor_dfn) = ctx
        .search_graph
        .closest_goal(&goal.query, goal.state_id, &goal.env)
    else {
        return Ok(None);
    };

    let ancestor_node = &ctx.search_graph[ancestor_dfn];
    let (ancestor_lazy_depth, result_ref, stack_depth) = (
        ancestor_node.goal.lazy_depth,
        ancestor_node.answer.result_ref,
        ancestor_node
            .stack_depth
            .expect("closest active goal must still be on stack"),
    );

    if ancestor_lazy_depth >= goal.lazy_depth {
        solver_event!(
            name: "solver.goal_outcome",
        outcome = "same_depth_cycle",
        ancestor_dfn = ancestor_dfn.index() as u64,
        ?result_ref
        );
        ctx.stack[stack_depth].flag_cycle();
        return Err(SolveError::SameDepthCycle);
    }

    ctx.stack[stack_depth].flag_cycle();
    solver_event!(
        name: "solver.goal_outcome",
        outcome = "active_lazy_hit",
        ancestor_dfn = ancestor_dfn.index() as u64,
        ?result_ref
    );
    Ok(Some((
        GoalSolveResult::Lazy { result_ref },
        Minimums::from_self(ancestor_dfn),
    )))
}

fn try_close_cross_env_active_cycle<R: Rule>(
    goal: &GoalKey<R>,
    ctx: &mut Context<R>,
) -> Option<(GoalSolveResult<R>, Minimums)> {
    if !ctx.cross_env_active_reuse_enabled {
        return None;
    }

    let ancestor_dfn = ctx
        .search_graph
        .closest_goal_any_env(&goal.query, goal.state_id)?;
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

    if ancestor_env == goal.env || ancestor_lazy_depth >= goal.lazy_depth {
        return None;
    }

    let blocked_key = (result_ref, Arc::clone(&goal.env));
    if ctx.blocked_cross_env_reuses.contains(&blocked_key) {
        return None;
    }

    ctx.stack[stack_depth].flag_cycle();
    solver_event!(
        name: "solver.goal_outcome",
        outcome = "cross_env_reuse",
        ancestor_dfn = ancestor_dfn.index() as u64,
        ?result_ref,
        ancestor_env_hash = debug_env_hash::<R>(Arc::as_ref(&ancestor_env))
    );
    Some((
        GoalSolveResult::LazyCrossEnv { result_ref },
        Minimums::from_self(ancestor_dfn),
    ))
}

#[instrumented(
    name = "solver.cache_candidate",
    target = "context_solver",
    level = "trace",
    ret,
    fields(
        result_ref = ?result_ref,
        exact_env,
        bucket_len = bucket_len as u64,
        candidate_env_hash = debug_env_hash::<R>(Arc::as_ref(candidate_env)),
        candidate_env_label = %trace_env_label::<R>(rule, Arc::as_ref(candidate_env))
    )
)]
fn cache_candidate_matches<R: Rule>(
    rule: &R,
    result_ref: RuleResultRef<R>,
    candidate_env: &Arc<R::Env>,
    goal_env: &Arc<R::Env>,
    ctx: &mut Context<R>,
    exact_env: bool,
    bucket_len: usize,
) -> bool {
    let matched = answer_matches_env(rule, result_ref, goal_env, ctx, 0);
    if matched {
        solver_event!(
            name: "solver.goal_outcome",
            outcome = if exact_env {
                "cache_exact_hit"
            } else {
                "cache_hit"
            }
        );
    }
    matched
}

fn try_cache_reuse<R: Rule>(
    rule: &R,
    goal: &GoalKey<R>,
    ctx: &mut Context<R>,
) -> Option<(GoalSolveResult<R>, Minimums)> {
    if !ctx.cache_reuse_enabled {
        return None;
    }

    let key = cache_key(goal);
    let (exact_result_refs, bucket_len, entries) = ctx.cache.get(&key).map(|bucket| {
        (
            bucket.cloned_result_refs_for_env(&goal.env),
            bucket.len(),
            bucket.cloned_entries(),
        )
    })?;

    if let Some(result_refs) = exact_result_refs.as_ref() {
        solver_event!(
            name: "solver.cache_probe",
            exact_env = true,
            bucket_len = result_refs.len() as u64
        );
        for result_ref in result_refs.iter().rev() {
            let matched = cache_candidate_matches(
                rule,
                *result_ref,
                &goal.env,
                &goal.env,
                ctx,
                true,
                result_refs.len(),
            );
            if matched {
                return Some((
                    GoalSolveResult::Resolved {
                        result_ref: *result_ref,
                    },
                    Minimums::new(),
                ));
            }
        }
    }

    solver_event!(
        name: "solver.cache_probe",
        exact_env = false,
        bucket_len = bucket_len as u64
    );
    for entry in entries.iter().rev() {
        if exact_result_refs.is_some() && entry.env == goal.env {
            continue;
        }
        let matched = cache_candidate_matches(
            rule,
            entry.result_ref,
            &entry.env,
            &goal.env,
            ctx,
            false,
            bucket_len,
        );
        if matched {
            return Some((
                GoalSolveResult::Resolved {
                    result_ref: entry.result_ref,
                },
                Minimums::new(),
            ));
        }
    }

    None
}

fn try_graph_goal_reuse<R: Rule>(
    goal: &GoalKey<R>,
    ctx: &mut Context<R>,
) -> Option<(GoalSolveResult<R>, Minimums)> {
    let dfn = ctx.search_graph.lookup(goal)?;
    let node = &ctx.search_graph[dfn];
    if let Some(stack_depth) = node.stack_depth {
        ctx.stack[stack_depth].flag_cycle();
    }
    solver_event!(
        name: "solver.goal_outcome",
        outcome = "graph_goal_hit",
        dfn = dfn.index() as u64,
        result_ref = ?node.answer.result_ref
    );
    Some((
        GoalSolveResult::Resolved {
            result_ref: node.answer.result_ref,
        },
        node.links,
    ))
}

#[instrumented(
    name = "solver.solve_goal",
    target = "context_solver",
    level = "trace",
    fields(
        query_hash = hash_value(&goal.query),
        env_hash = debug_env_hash::<R>(Arc::as_ref(&goal.env)),
        state_hash = hash_value(&goal.state_id),
        lazy_depth = goal.lazy_depth.0 as u64,
        query_label = %trace_query_label::<R>(rule, &goal.query, goal.state_id),
        env_label = %trace_env_label::<R>(rule, Arc::as_ref(&goal.env))
    )
)]
pub(crate) fn solve_goal<R: Rule>(
    rule: &R,
    goal: GoalKey<R>,
    ctx: &mut Context<R>,
) -> Result<(GoalSolveResult<R>, Minimums), SolveError> {
    if let Some(result) = try_close_active_cycle(&goal, ctx)? {
        return Ok(result);
    }

    if let Some(result) = try_close_cross_env_active_cycle(&goal, ctx) {
        return Ok(result);
    }

    if let Some(result) = try_cache_reuse(rule, &goal, ctx) {
        return Ok(result);
    }

    if let Some(result) = try_graph_goal_reuse(&goal, ctx) {
        return Ok(result);
    }

    solve_new_goal(rule, goal, ctx)
}

#[instrumented(
    name = "solver.solve",
    target = "context_solver",
    level = "trace",
    ret,
    fields(
        query_hash = hash_value(&query),
        env_hash = debug_env_hash::<R>(Arc::as_ref(&env)),
        state_hash = hash_value(&initial_rule),
        lazy_depth = 0_u64,
        query_label = %trace_query_label::<R>(rule, &query, initial_rule),
        env_label = %trace_env_label::<R>(rule, Arc::as_ref(&env))
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
    let root_goal = GoalKey {
        query,
        state_id: initial_rule,
        env,
        lazy_depth: LazyDepth(0),
    };

    let mut ctx = Context::new(shared_state, fixpoint_iteration_limit, stack_depth_limit);
    let result = solve_goal(rule, root_goal, &mut ctx)
        .map(|(solve_result, _)| match solve_result {
            GoalSolveResult::Resolved { result_ref } => result_ref,
            GoalSolveResult::Lazy { .. } | GoalSolveResult::LazyCrossEnv { .. } => {
                unreachable!(
                    "root solve_goal cannot resolve lazily because lazy results require an active ancestor"
                )
            }
        });

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
