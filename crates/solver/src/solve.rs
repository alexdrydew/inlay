use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use thiserror::Error;

use crate::{
    arena::Arena,
    context::{AnswerMatchMemo, Context},
    rule::{
        Lookups, ResolutionEnv, Rule, RuleEnvSharedState, RuleLookupQuery, RuleLookupResult,
        RuleQuery, RuleResult, RuleResultRef, RuleResultsArena,
    },
    search_graph::{CacheKey, Dependency, GoalKey, LazyDepth, Minimums},
    stack::StackError,
};

const CACHE_MISS_TRACE_EVENT_LIMIT: usize = 12;

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

struct CacheValidationTrace<R: Rule> {
    cache_key_label: String,
    candidate_result_ref: RuleResultRef<R>,
    candidate_env_label: String,
    candidate_env_hash: u64,
    exact_env: bool,
    remaining_events: usize,
    emitted: bool,
}

impl<R: Rule> CacheValidationTrace<R> {
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
            remaining_events: CACHE_MISS_TRACE_EVENT_LIMIT,
            emitted: false,
        }
    }

    fn emit(
        &mut self,
        rule: &R,
        ctx: &mut Context<R>,
        event: &str,
        env: &Arc<R::Env>,
        depth: usize,
        extra_fields: &[(&str, String)],
    ) {
        if self.remaining_events == 0 {
            return;
        }
        self.remaining_events -= 1;
        self.emitted = true;

        let mut fields = vec![
            (
                "candidate_result_ref",
                format!(
                    "\"{}\"",
                    json_escape(&format!("{:?}", self.candidate_result_ref))
                ),
            ),
            (
                "candidate_env",
                format!("\"{}\"", json_escape(&self.candidate_env_label)),
            ),
            (
                "candidate_env_hash",
                format!("\"{:x}\"", self.candidate_env_hash),
            ),
            ("exact_env", self.exact_env.to_string()),
        ];
        fields.extend(
            extra_fields
                .iter()
                .map(|(key, value)| (*key, value.clone())),
        );
        trace_cache_validation_event(
            ctx,
            event,
            &self.cache_key_label,
            &self.cache_key_label,
            &debug_env_label(rule, env.as_ref()),
            debug_env_hash::<R>(env.as_ref()),
            depth,
            &fields,
        );
    }
}

fn trace_cache_validation_event<R: Rule>(
    ctx: &mut Context<R>,
    event: &str,
    filter_text: &str,
    cache_key_label: &str,
    env_label: &str,
    env_hash: u64,
    depth: usize,
    extra_fields: &[(&str, String)],
) {
    let Some(seq) = ctx.next_trace_seq() else {
        return;
    };
    let mut line = format!(
        "{{\"seq\":{seq},\"event\":\"{}\",\"cache_key\":\"{}\",\"env\":\"{}\",\"env_hash\":\"{:x}\",\"depth\":{}}}",
        json_escape(event),
        json_escape(cache_key_label),
        json_escape(env_label),
        env_hash,
        depth,
    );
    line.pop();
    for (key, value) in extra_fields {
        line.push_str(&format!(",\"{}\":{}", json_escape(key), value));
    }
    line.push('}');
    ctx.trace_line(filter_text, line);
}

fn debug_lookup_query_label<R: Rule>(rule: &R, query: &RuleLookupQuery<R>) -> String {
    rule.debug_lookup_query_label(query)
        .unwrap_or_else(|| format!("lookup={:x}", hash_value(query)))
}

fn debug_lookup_result_label<R: Rule>(rule: &R, result: &RuleLookupResult<R>) -> String {
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

fn answer_match_input_label<R: Rule>(result_query: &str, env: &Arc<R::Env>) -> String {
    format!(
        "{}@env_hash={:x}@env_items={}",
        result_query,
        hash_value(env.as_ref()),
        R::Env::env_item_count(env.as_ref())
    )
}

fn lookups_match_env<R: Rule>(
    rule: &R,
    lookups: &Lookups<R>,
    env: &Arc<R::Env>,
    ctx: &mut Context<R>,
    result_ref: RuleResultRef<R>,
    depth: usize,
    trace: Option<&mut CacheValidationTrace<R>>,
) -> bool {
    let started = Instant::now();
    let mut mismatch = None;
    for (query, expected_result) in lookups {
        let actual_result = env.lookup(&mut ctx.shared_state, query);
        if actual_result != *expected_result {
            mismatch = Some((query.clone(), expected_result.clone(), actual_result));
            break;
        }
    }
    ctx.record_lookup_match(lookups.len(), started.elapsed());
    let Some((query, expected_result, actual_result)) = mismatch else {
        return true;
    };

    if let Some(trace) = trace {
        let result_query = debug_result_query_label(rule, result_ref, ctx);
        trace.emit(
            rule,
            ctx,
            "cache_lookup_miss",
            env,
            depth,
            &[
                (
                    "result_ref",
                    format!("\"{}\"", json_escape(&format!("{:?}", result_ref))),
                ),
                (
                    "result_query",
                    format!("\"{}\"", json_escape(&result_query)),
                ),
                (
                    "lookup",
                    format!(
                        "\"{}\"",
                        json_escape(&debug_lookup_query_label(rule, &query))
                    ),
                ),
                (
                    "expected",
                    format!(
                        "\"{}\"",
                        json_escape(&debug_lookup_result_label(rule, &expected_result))
                    ),
                ),
                (
                    "actual",
                    format!(
                        "\"{}\"",
                        json_escape(&debug_lookup_result_label(rule, &actual_result))
                    ),
                ),
            ],
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
    trace: Option<&mut CacheValidationTrace<R>>,
) -> bool {
    ctx.record_answer_match_call(depth);
    let result_query = debug_result_query_label(rule, result_ref, ctx);
    let input_key = ctx
        .stats
        .as_ref()
        .map(|_| answer_match_input_label::<R>(&result_query, env));
    if let Some(key) = &input_key {
        ctx.record_answer_match_input_call(key);
    }

    match ctx
        .answer_match_memo
        .get(&(result_ref, Arc::clone(env)))
        .copied()
    {
        Some(AnswerMatchMemo::Resolved(matches)) => {
            ctx.record_answer_match_memo_hit();
            if let Some(key) = &input_key {
                ctx.record_answer_match_input_memo_hit(key);
            }
            return matches;
        }
        Some(AnswerMatchMemo::InProgress) => {
            ctx.record_answer_match_in_progress_hit();
            if let Some(key) = &input_key {
                ctx.record_answer_match_input_in_progress_hit(key);
            }
            return true;
        }
        None => {}
    }

    ctx.answer_match_memo
        .insert((result_ref, Arc::clone(env)), AnswerMatchMemo::InProgress);

    let started = Instant::now();
    let Some(answer) = ctx.answer_for(result_ref).cloned() else {
        if let Some(trace) = trace {
            trace.emit(
                rule,
                ctx,
                "cache_missing_answer",
                env,
                depth,
                &[
                    (
                        "result_ref",
                        format!("\"{}\"", json_escape(&format!("{:?}", result_ref))),
                    ),
                    (
                        "result_query",
                        format!("\"{}\"", json_escape(&result_query)),
                    ),
                ],
            );
        }
        ctx.record_answer_match_missing_answer();
        ctx.record_answer_match_evaluation(
            input_key.as_deref().unwrap_or(&result_query),
            0,
            started.elapsed(),
            false,
        );
        ctx.answer_match_memo.insert(
            (result_ref, Arc::clone(env)),
            AnswerMatchMemo::Resolved(false),
        );
        return false;
    };

    let matches = if let Some(trace) = trace {
        if !lookups_match_env(
            rule,
            &answer.lookups,
            env,
            ctx,
            result_ref,
            depth,
            Some(trace),
        ) {
            false
        } else {
            let mut matches = true;
            for dependency in &answer.dependencies {
                let parent_item_count = R::Env::env_item_count(env.as_ref());
                let delta_item_count =
                    R::Env::dependency_env_delta_item_count(&dependency.env_delta);
                let dependency_env = ctx.rebased_env_for_dependency(env, &dependency.env_delta);
                ctx.record_dependency_env_rebase(
                    parent_item_count,
                    delta_item_count,
                    R::Env::env_item_count(dependency_env.as_ref()),
                );
                if !answer_matches_env(
                    rule,
                    dependency.result_ref,
                    &dependency_env,
                    ctx,
                    depth + 1,
                    Some(trace),
                ) {
                    trace.emit(
                        rule,
                        ctx,
                        "cache_dependency_miss",
                        env,
                        depth,
                        &[
                            (
                                "result_ref",
                                format!("\"{}\"", json_escape(&format!("{:?}", result_ref))),
                            ),
                            (
                                "result_query",
                                format!("\"{}\"", json_escape(&result_query)),
                            ),
                            (
                                "dependency_result_ref",
                                format!(
                                    "\"{}\"",
                                    json_escape(&format!("{:?}", dependency.result_ref))
                                ),
                            ),
                        ],
                    );
                    matches = false;
                    break;
                }
            }
            matches
        }
    } else {
        lookups_match_env(rule, &answer.lookups, env, ctx, result_ref, depth, None)
            && answer.dependencies.iter().all(|dependency| {
                let parent_item_count = R::Env::env_item_count(env.as_ref());
                let delta_item_count =
                    R::Env::dependency_env_delta_item_count(&dependency.env_delta);
                let dependency_env = ctx.rebased_env_for_dependency(env, &dependency.env_delta);
                ctx.record_dependency_env_rebase(
                    parent_item_count,
                    delta_item_count,
                    R::Env::env_item_count(dependency_env.as_ref()),
                );
                answer_matches_env(
                    rule,
                    dependency.result_ref,
                    &dependency_env,
                    ctx,
                    depth + 1,
                    None,
                )
            })
    };
    ctx.record_answer_match_evaluation(
        input_key.as_deref().unwrap_or(&result_query),
        answer.dependencies.len(),
        started.elapsed(),
        matches,
    );
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

    if !lookups_match_env(rule, &answer.lookups, env, ctx, result_ref, 0, None) {
        memo.insert(
            key,
            ActiveAnswerMatchMemo::Resolved(ActiveAnswerMatch::Mismatch),
        );
        return ActiveAnswerMatch::Mismatch;
    }

    let mut saw_unknown = false;
    for dependency in answer.dependencies {
        let parent_item_count = R::Env::env_item_count(env.as_ref());
        let delta_item_count = R::Env::dependency_env_delta_item_count(&dependency.env_delta);
        let dependency_env = ctx.rebased_env_for_dependency(env, &dependency.env_delta);
        ctx.record_dependency_env_rebase(
            parent_item_count,
            delta_item_count,
            R::Env::env_item_count(dependency_env.as_ref()),
        );
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

fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

pub(crate) fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out
}

pub(crate) fn debug_env_hash<R: Rule>(env: &R::Env) -> u64 {
    hash_value(env)
}

pub(crate) fn debug_env_label<R: Rule>(rule: &R, env: &R::Env) -> String {
    rule.debug_env_label(env)
        .unwrap_or_else(|| format!("env={:x}", debug_env_hash::<R>(env)))
}

pub(crate) fn trace_goal_event<R: Rule>(
    ctx: &mut Context<R>,
    event: &str,
    filter_text: &str,
    query_label: &str,
    env_label: &str,
    env_hash: u64,
    lazy_depth: u64,
    extra_fields: &[(&str, String)],
) {
    let Some(seq) = ctx.next_trace_seq() else {
        return;
    };
    let mut line = format!(
        "{{\"seq\":{seq},\"event\":\"{}\",\"query\":\"{}\",\"env\":\"{}\",\"env_hash\":\"{:x}\",\"lazy_depth\":{}",
        json_escape(event),
        json_escape(query_label),
        json_escape(env_label),
        env_hash,
        lazy_depth,
    );
    for (key, value) in extra_fields {
        line.push_str(&format!(",\"{}\":{}", json_escape(key), value,));
    }
    line.push('}');
    ctx.trace_line(filter_text, line);
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

fn evaluate_goal_once<R: Rule>(
    rule: &R,
    dfn: crate::search_graph::DepthFirstNumber,
    ctx: &mut Context<R>,
) -> Result<Minimums, SolveQueryError> {
    let goal = ctx.search_graph[dfn].goal.clone();
    let query_label = debug_cache_key_label::<R>(rule, &goal.query, goal.state_id);
    let env_label = debug_env_label(rule, Arc::as_ref(&goal.env));
    let env_hash = debug_env_hash::<R>(Arc::as_ref(&goal.env));
    let mut minimums = Minimums::new();
    let (lookups, dependencies, cross_env_reuses, result_ref) = {
        let mut rule_ctx =
            crate::rule::RuleContext::new(rule, goal.state_id, goal.env, ctx, dfn, &mut minimums);

        let result = match rule.run(goal.query, &mut rule_ctx) {
            Ok(output) => Ok(output),
            Err(crate::rule::RunError::Rule(err)) => Err(err),
            Err(crate::rule::RunError::Solve(error)) => return Err(error),
        };
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
    let lookup_count = lookups.len();
    let dependency_count = dependencies.len();
    let cross_env_reuse_count = ctx.search_graph[dfn].cross_env_reuses.len();
    ctx.record_answer(lookup_count, dependency_count);
    ctx.replace_answer(
        &query_label,
        result_ref,
        crate::search_graph::Answer {
            result_ref,
            lookups,
            dependencies,
        },
    );
    trace_goal_event(
        ctx,
        "goal_eval",
        &query_label,
        &query_label,
        &env_label,
        env_hash,
        goal.lazy_depth.0 as u64,
        &[
            ("dfn", dfn.index().to_string()),
            (
                "result_ref",
                format!("\"{}\"", json_escape(&format!("{:?}", result_ref))),
            ),
            ("lookups", lookup_count.to_string()),
            ("dependencies", dependency_count.to_string()),
            ("cross_env_reuses", cross_env_reuse_count.to_string()),
            (
                "result_kind",
                format!(
                    "\"{}\"",
                    match ctx
                        .results_arena
                        .get(&result_ref)
                        .expect("result just stored")
                    {
                        Ok(_) => "ok",
                        Err(_) => "err",
                    }
                ),
            ),
        ],
    );
    ctx.search_graph[dfn].links = minimums;
    Ok(minimums)
}

fn solve_new_goal<R: Rule>(
    rule: &R,
    goal: GoalKey<R>,
    ctx: &mut Context<R>,
) -> Result<(GoalSolveResult<R>, Minimums), SolveQueryError> {
    ctx.record_new_goal();
    let result_ref = ctx.result_ref_for(&goal);
    let stack_depth = ctx.stack.push().map_err(|error| match error {
        StackError::Overflow => SolveError::StackOverflowDepthReached,
    })?;
    let dfn = ctx.search_graph.insert(&goal, stack_depth, result_ref);
    let query_label = debug_cache_key_label::<R>(rule, &goal.query, goal.state_id);
    let env_label = debug_env_label(rule, goal.env.as_ref());
    trace_goal_event(
        ctx,
        "new_goal",
        &query_label,
        &query_label,
        &env_label,
        debug_env_hash::<R>(Arc::as_ref(&goal.env)),
        goal.lazy_depth.0 as u64,
        &[
            ("dfn", dfn.index().to_string()),
            (
                "result_ref",
                format!("\"{}\"", json_escape(&format!("{:?}", result_ref))),
            ),
        ],
    );

    let mut reruns = 0;
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

        if reruns >= ctx.fixpoint_iteration_limit {
            return Err(SolveError::FixpointIterationLimitReached.into());
        }
        reruns += 1;
        ctx.record_fixpoint_rerun();
        trace_goal_event(
            ctx,
            "fixpoint_rerun",
            &query_label,
            &query_label,
            &env_label,
            debug_env_hash::<R>(Arc::as_ref(&goal.env)),
            goal.lazy_depth.0 as u64,
            &[
                ("dfn", dfn.index().to_string()),
                ("rerun", reruns.to_string()),
                ("blocked_grew", blocked_grew.to_string()),
                ("snapshot_unchanged", snapshot_unchanged.to_string()),
            ],
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

    Ok((GoalSolveResult::Resolved { result_ref }, final_minimums))
}

pub(crate) fn solve_goal<R: Rule>(
    rule: &R,
    goal: GoalKey<R>,
    ctx: &mut Context<R>,
) -> Result<(GoalSolveResult<R>, Minimums), SolveQueryError> {
    ctx.record_goal_attempt();
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
            ctx.record_active_ancestor_same_depth_cycle();
            ctx.stack[stack_depth].flag_cycle();
            return Err(SameDepthCycleError.into());
        }

        ctx.stack[stack_depth].flag_cycle();
        ctx.record_active_ancestor_lazy_hit();
        let query_label = debug_cache_key_label::<R>(rule, &goal.query, goal.state_id);
        let env_label = debug_env_label(rule, goal.env.as_ref());
        trace_goal_event(
            ctx,
            "active_lazy_hit",
            &query_label,
            &query_label,
            &env_label,
            debug_env_hash::<R>(Arc::as_ref(&goal.env)),
            goal.lazy_depth.0 as u64,
            &[
                ("ancestor_dfn", ancestor_dfn.index().to_string()),
                (
                    "result_ref",
                    format!("\"{}\"", json_escape(&format!("{:?}", result_ref))),
                ),
            ],
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
                    ctx.record_active_ancestor_lazy_hit();
                    let query_label = debug_cache_key_label::<R>(rule, &goal.query, goal.state_id);
                    let env_label = debug_env_label(rule, goal.env.as_ref());
                    trace_goal_event(
                        ctx,
                        "cross_env_reuse",
                        &query_label,
                        &query_label,
                        &env_label,
                        debug_env_hash::<R>(Arc::as_ref(&goal.env)),
                        goal.lazy_depth.0 as u64,
                        &[
                            ("ancestor_dfn", ancestor_dfn.index().to_string()),
                            (
                                "result_ref",
                                format!("\"{}\"", json_escape(&format!("{:?}", result_ref))),
                            ),
                            (
                                "ancestor_env_hash",
                                format!(
                                    "\"{:x}\"",
                                    debug_env_hash::<R>(Arc::as_ref(&ancestor_env))
                                ),
                            ),
                        ],
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
            let cache_key_label = debug_cache_key_label::<R>(rule, &goal.query, goal.state_id);

            if let Some(result_refs) = exact_result_refs.as_ref() {
                ctx.record_cache_exact_env_bucket_probe();
                for result_ref in result_refs.iter().rev() {
                    let mut trace = ctx.should_trace_cache_miss(&cache_key_label).then(|| {
                        CacheValidationTrace::new(
                            rule,
                            &cache_key_label,
                            *result_ref,
                            &goal.env,
                            true,
                        )
                    });
                    if answer_matches_env(rule, *result_ref, &goal.env, ctx, 0, trace.as_mut()) {
                        ctx.record_cache_exact_env_candidate_hit();
                        let query_label =
                            debug_cache_key_label::<R>(rule, &goal.query, goal.state_id);
                        let env_label = debug_env_label(rule, goal.env.as_ref());
                        trace_goal_event(
                            ctx,
                            "cache_exact_hit",
                            &query_label,
                            &query_label,
                            &env_label,
                            debug_env_hash::<R>(Arc::as_ref(&goal.env)),
                            goal.lazy_depth.0 as u64,
                            &[(
                                "result_ref",
                                format!("\"{}\"", json_escape(&format!("{:?}", result_ref))),
                            )],
                        );
                        return Ok((
                            GoalSolveResult::Resolved {
                                result_ref: *result_ref,
                            },
                            Minimums::new(),
                        ));
                    }
                    if trace.as_ref().is_some_and(|trace| trace.emitted) {
                        ctx.record_cache_miss_trace();
                    }
                    ctx.record_cache_exact_env_candidate_miss();
                }
            }

            ctx.record_cache_bucket_probe();
            ctx.record_cache_bucket_probe_for(&cache_key_label, bucket_len);
            for entry in entries.iter().rev() {
                if exact_result_refs.is_some() && entry.env == goal.env {
                    continue;
                }
                let mut trace = ctx.should_trace_cache_miss(&cache_key_label).then(|| {
                    CacheValidationTrace::new(
                        rule,
                        &cache_key_label,
                        entry.result_ref,
                        &entry.env,
                        false,
                    )
                });
                if answer_matches_env(rule, entry.result_ref, &goal.env, ctx, 0, trace.as_mut()) {
                    ctx.record_cache_candidate_hit();
                    ctx.record_cache_candidate_hit_for(&cache_key_label);
                    let env_label = debug_env_label(rule, goal.env.as_ref());
                    trace_goal_event(
                        ctx,
                        "cache_hit",
                        &cache_key_label,
                        &cache_key_label,
                        &env_label,
                        debug_env_hash::<R>(Arc::as_ref(&goal.env)),
                        goal.lazy_depth.0 as u64,
                        &[(
                            "result_ref",
                            format!("\"{}\"", json_escape(&format!("{:?}", entry.result_ref))),
                        )],
                    );
                    return Ok((
                        GoalSolveResult::Resolved {
                            result_ref: entry.result_ref,
                        },
                        Minimums::new(),
                    ));
                }
                if trace.as_ref().is_some_and(|trace| trace.emitted) {
                    ctx.record_cache_miss_trace();
                }
                ctx.record_cache_candidate_miss();
                ctx.record_cache_candidate_miss_for(&cache_key_label);
            }
        }
    }

    if let Some(dfn) = ctx.search_graph.lookup(&goal) {
        ctx.record_graph_goal_hit();
        let node = &ctx.search_graph[dfn];
        if let Some(stack_depth) = node.stack_depth {
            ctx.stack[stack_depth].flag_cycle();
        }
        return Ok((
            GoalSolveResult::Resolved {
                result_ref: node.answer.result_ref,
            },
            node.links,
        ));
    }

    solve_new_goal(rule, goal, ctx)
}

pub fn solve<R: Rule>(
    rule: &R,
    query: RuleQuery<R>,
    initial_rule: R::RuleStateId,
    env: Arc<R::Env>,
    shared_state: RuleEnvSharedState<R>,
    fixpoint_iteration_limit: usize,
    stack_overflow_depth: usize,
) -> SolveOutcome<R> {
    let mut ctx = Context::new(shared_state, fixpoint_iteration_limit, stack_overflow_depth);
    let root_goal = GoalKey {
        query,
        state_id: initial_rule,
        env,
        lazy_depth: LazyDepth(0),
    };
    let result = solve_goal(rule, root_goal, &mut ctx)
        .map(|(solve_result, _minimums)| match solve_result {
            GoalSolveResult::Resolved { result_ref } => result_ref,
            GoalSolveResult::Lazy { .. } | GoalSolveResult::LazyCrossEnv { .. } => {
                unreachable!("root solve_goal must not resolve lazily")
            }
        })
        .map_err(|error| match error {
            SolveQueryError::SameDepthCycle(_) => SolveError::UnexpectedSameDepthCycle,
            SolveQueryError::Solve(error) => error,
        });

    ctx.emit_stats();

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
