use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::Duration;

use crate::{
    arena::Arena,
    rule::{ResolutionEnv, Rule, RuleEnv, RuleEnvSharedState, RuleResultRef, RuleResultsArena},
    search_graph::{Answer, CacheBucket, CacheKey, GoalKey, SearchGraph},
    stack::Stack,
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
    <RuleEnv<R> as crate::rule::ResolutionEnv>::DependencyEnvDelta,
);

pub(crate) struct Context<R: Rule> {
    pub(crate) results_arena: RuleResultsArena<R>,
    pub(crate) result_refs: HashMap<GoalKey<R>, RuleResultRef<R>>,
    pub(crate) result_goals: HashMap<RuleResultRef<R>, GoalKey<R>>,
    pub(crate) result_answers: HashMap<RuleResultRef<R>, Answer<R>>,
    pub(crate) answer_fingerprints: HashMap<RuleResultRef<R>, u64>,
    pub(crate) answer_dependents: HashMap<RuleResultRef<R>, HashSet<RuleResultRef<R>>>,
    pub(crate) answer_match_memo: HashMap<AnswerMatchMemoKey<R>, AnswerMatchMemo>,
    rebased_env_cache: HashMap<RebasedEnvCacheKey<R>, Arc<RuleEnv<R>>>,
    pub(crate) blocked_cross_env_reuses: HashSet<BlockedCrossEnvReuse<R>>,
    pub(crate) search_graph: SearchGraph<R>,
    pub(crate) cache: HashMap<CacheKey<R>, CacheBucket<R>>,
    pub(crate) stack: Stack,
    pub(crate) fixpoint_iteration_limit: usize,
    pub(crate) shared_state: RuleEnvSharedState<R>,
    pub(crate) stats: Option<SolverStats>,
    progress_interval: Option<u64>,
    pub(crate) cross_env_active_reuse_enabled: bool,
    pub(crate) cache_reuse_enabled: bool,
    pub(crate) cache_dedup_enabled: bool,
    trace: Option<SolverTrace>,
    cache_miss_trace_limit: usize,
    cache_miss_traces_emitted: usize,
}

struct SolverTrace {
    writer: BufWriter<std::fs::File>,
    filter: Option<String>,
    next_seq: u64,
}

#[derive(Default)]
pub(crate) struct SolverStats {
    pub(crate) goal_attempts: u64,
    pub(crate) new_goals: u64,
    pub(crate) active_ancestor_lazy_hits: u64,
    pub(crate) active_ancestor_same_depth_cycles: u64,
    pub(crate) graph_goal_hits: u64,
    pub(crate) fixpoint_reruns: u64,
    pub(crate) lookup_calls: u64,
    pub(crate) lookup_time: Duration,
    pub(crate) answers_recorded: u64,
    pub(crate) answer_lookups_recorded: u64,
    pub(crate) answer_dependencies_recorded: u64,
    pub(crate) max_answer_lookups: u64,
    pub(crate) max_answer_dependencies: u64,
    pub(crate) cache_bucket_probes: u64,
    pub(crate) cache_exact_env_bucket_probes: u64,
    pub(crate) cache_candidates_checked: u64,
    pub(crate) cache_exact_env_candidates_checked: u64,
    pub(crate) cache_candidate_hits: u64,
    pub(crate) cache_exact_env_candidate_hits: u64,
    pub(crate) cache_candidate_misses: u64,
    pub(crate) cache_exact_env_candidate_misses: u64,
    pub(crate) lookup_match_calls: u64,
    pub(crate) lookup_match_entries: u64,
    pub(crate) lookup_match_time: Duration,
    pub(crate) answer_match_calls: u64,
    pub(crate) answer_match_evaluations: u64,
    pub(crate) answer_match_edges: u64,
    pub(crate) answer_match_memo_hits: u64,
    pub(crate) answer_match_in_progress_hits: u64,
    pub(crate) answer_match_missing_answers: u64,
    pub(crate) answer_match_max_depth: u64,
    pub(crate) answer_match_time: Duration,
    pub(crate) dependency_env_delta_items_recorded: u64,
    pub(crate) max_dependency_env_delta_items: u64,
    pub(crate) dependency_env_rebases: u64,
    pub(crate) dependency_env_rebase_parent_items: u64,
    pub(crate) dependency_env_rebase_delta_items: u64,
    pub(crate) dependency_env_rebase_child_items: u64,
    pub(crate) max_dependency_env_rebase_parent_items: u64,
    pub(crate) max_dependency_env_rebase_delta_items: u64,
    pub(crate) max_dependency_env_rebase_child_items: u64,
    pub(crate) max_answer_match_memo_entries: u64,
    pub(crate) rebased_env_cache_hits: u64,
    pub(crate) rebased_env_cache_misses: u64,
    pub(crate) max_rebased_env_cache_entries: u64,
    pub(crate) replace_answer_calls: u64,
    pub(crate) replace_answer_changed: u64,
    pub(crate) replace_answer_unchanged: u64,
    pub(crate) replace_answer_memo_entries_cleared: u64,
    pub(crate) max_replace_answer_memo_entries_cleared: u64,
    pub(crate) replace_answer_changed_with_memo_entries: u64,
    pub(crate) replace_answer_changed_memo_entries_cleared: u64,
    pub(crate) cache_key_stats: BTreeMap<String, CacheKeyStats>,
    pub(crate) answer_match_key_stats: BTreeMap<String, AnswerMatchKeyStats>,
    pub(crate) answer_match_input_stats: BTreeMap<String, AnswerMatchInputStats>,
    pub(crate) replace_answer_key_stats: BTreeMap<String, ReplaceAnswerKeyStats>,
}

#[derive(Default)]
pub(crate) struct CacheKeyStats {
    pub(crate) probes: u64,
    pub(crate) candidates_checked: u64,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) total_bucket_len: u64,
    pub(crate) max_bucket_len: u64,
}

#[derive(Default)]
pub(crate) struct AnswerMatchKeyStats {
    pub(crate) evaluations: u64,
    pub(crate) edges: u64,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) time: Duration,
}

#[derive(Default)]
pub(crate) struct AnswerMatchInputStats {
    pub(crate) calls: u64,
    pub(crate) evaluations: u64,
    pub(crate) memo_hits: u64,
    pub(crate) in_progress_hits: u64,
}

#[derive(Default)]
pub(crate) struct ReplaceAnswerKeyStats {
    pub(crate) calls: u64,
    pub(crate) changed: u64,
    pub(crate) unchanged: u64,
    pub(crate) memo_entries_cleared: u64,
    pub(crate) changed_with_memo_entries: u64,
    pub(crate) changed_memo_entries_cleared: u64,
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
            result_goals: HashMap::new(),
            result_answers: HashMap::new(),
            answer_fingerprints: HashMap::new(),
            answer_dependents: HashMap::new(),
            answer_match_memo: HashMap::new(),
            rebased_env_cache: HashMap::new(),
            blocked_cross_env_reuses: HashSet::new(),
            search_graph: SearchGraph::new(),
            cache: HashMap::new(),
            stack: Stack::new(stack_overflow_depth),
            fixpoint_iteration_limit,
            shared_state: env_shared_state,
            stats: std::env::var_os("INLAY_SOLVER_STATS").map(|_| SolverStats::default()),
            progress_interval: std::env::var("INLAY_SOLVER_PROGRESS_INTERVAL")
                .ok()
                .and_then(|value| value.parse().ok()),
            cross_env_active_reuse_enabled: std::env::var_os(
                "INLAY_DISABLE_CROSS_ENV_ACTIVE_REUSE",
            )
            .is_none(),
            cache_reuse_enabled: std::env::var_os("INLAY_DISABLE_CACHE_REUSE").is_none(),
            cache_dedup_enabled: std::env::var_os("INLAY_DISABLE_CACHE_DEDUP").is_none(),
            trace: SolverTrace::new(),
            cache_miss_trace_limit: std::env::var("INLAY_SOLVER_CACHE_MISS_TRACE_LIMIT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0),
            cache_miss_traces_emitted: 0,
        }
    }

    pub(crate) fn trace_line(&mut self, filter_text: &str, line: String) {
        let Some(trace) = &mut self.trace else {
            return;
        };
        if trace
            .filter
            .as_ref()
            .is_some_and(|filter| !filter_text.contains(filter))
        {
            return;
        }
        if writeln!(trace.writer, "{line}").is_err() {
            self.trace = None;
            return;
        }
        if trace.writer.flush().is_err() {
            self.trace = None;
        }
    }

    pub(crate) fn next_trace_seq(&mut self) -> Option<u64> {
        let trace = self.trace.as_mut()?;
        let seq = trace.next_seq;
        trace.next_seq += 1;
        Some(seq)
    }

    pub(crate) fn trace_matches_filter(&self, filter_text: &str) -> bool {
        let Some(trace) = &self.trace else {
            return false;
        };
        trace
            .filter
            .as_ref()
            .is_none_or(|filter| filter_text.contains(filter))
    }

    pub(crate) fn should_trace_cache_miss(&self, filter_text: &str) -> bool {
        self.cache_miss_traces_emitted < self.cache_miss_trace_limit
            && self.trace_matches_filter(filter_text)
    }

    pub(crate) fn record_cache_miss_trace(&mut self) {
        self.cache_miss_traces_emitted += 1;
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

    pub(crate) fn goal_for_result_ref(&self, result_ref: RuleResultRef<R>) -> Option<&GoalKey<R>> {
        self.result_goals.get(&result_ref)
    }

    pub(crate) fn replace_answer(
        &mut self,
        key: &str,
        result_ref: RuleResultRef<R>,
        answer: Answer<R>,
    ) {
        let changed = self
            .result_answers
            .get(&result_ref)
            .is_none_or(|old| old != &answer);
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

        let memo_entries_cleared = self.answer_match_memo.len() as u64;
        self.record_replace_answer(key, changed, memo_entries_cleared);
        self.answer_match_memo.clear();
        self.invalidate_fingerprint_closure(result_ref);
    }

    fn invalidate_fingerprint_closure(&mut self, result_ref: RuleResultRef<R>) {
        let mut stack = vec![result_ref];
        let mut visited = HashSet::new();

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            self.answer_fingerprints.remove(&current);
            if let Some(dependents) = self.answer_dependents.get(&current) {
                stack.extend(dependents.iter().copied());
            }
        }
    }

    pub(crate) fn answer_for(&self, result_ref: RuleResultRef<R>) -> Option<&Answer<R>> {
        self.result_answers.get(&result_ref)
    }

    pub(crate) fn record_lookup_call(&mut self, elapsed: Duration) {
        if let Some(stats) = &mut self.stats {
            stats.lookup_calls += 1;
            stats.lookup_time += elapsed;
        }
    }

    pub(crate) fn record_goal_attempt(&mut self) {
        if let Some(stats) = &mut self.stats {
            stats.goal_attempts += 1;
        }
    }

    pub(crate) fn record_new_goal(&mut self) {
        if let Some(stats) = &mut self.stats {
            stats.new_goals += 1;
            if let Some(interval) = self.progress_interval {
                if stats.new_goals % interval == 0 {
                    let cache_entries: usize =
                        self.cache.values().map(|bucket| bucket.entry_count()).sum();
                    eprintln!(
                        concat!(
                            "[context-solver-progress] ",
                            "new_goals={} ",
                            "goal_attempts={} ",
                            "active_lazy_hits={} ",
                            "graph_goal_hits={} ",
                            "fixpoint_reruns={} ",
                            "lookup_ms={:.3} ",
                            "lookup_match_ms={:.3} ",
                            "answer_match_ms={:.3} ",
                            "blocked_cross_env_reuses={} ",
                            "result_answers={} ",
                            "answer_lookups_recorded={} ",
                            "answer_dependencies_recorded={} ",
                            "dependency_env_delta_items_recorded={} ",
                            "cache_keys={} ",
                            "cache_entries={} ",
                            "answer_match_memo_entries={} ",
                            "dependency_env_rebases={} ",
                            "dependency_env_rebase_parent_items={} ",
                            "dependency_env_rebase_delta_items={} ",
                            "dependency_env_rebase_child_items={} ",
                            "rebased_env_cache_entries={} ",
                            "rebased_env_cache_hits={} ",
                            "rebased_env_cache_misses={} ",
                            "replace_answer_calls={} ",
                            "replace_answer_changed={} ",
                            "replace_answer_unchanged={} ",
                            "replace_answer_memo_entries_cleared={} ",
                            "replace_answer_changed_with_memo_entries={} ",
                            "replace_answer_changed_memo_entries_cleared={} ",
                            "cache_candidate_hits={} ",
                            "cache_candidate_misses={}"
                        ),
                        stats.new_goals,
                        stats.goal_attempts,
                        stats.active_ancestor_lazy_hits,
                        stats.graph_goal_hits,
                        stats.fixpoint_reruns,
                        stats.lookup_time.as_secs_f64() * 1000.0,
                        stats.lookup_match_time.as_secs_f64() * 1000.0,
                        stats.answer_match_time.as_secs_f64() * 1000.0,
                        self.blocked_cross_env_reuses.len(),
                        self.result_answers.len(),
                        stats.answer_lookups_recorded,
                        stats.answer_dependencies_recorded,
                        stats.dependency_env_delta_items_recorded,
                        self.cache.len(),
                        cache_entries,
                        self.answer_match_memo.len(),
                        stats.dependency_env_rebases,
                        stats.dependency_env_rebase_parent_items,
                        stats.dependency_env_rebase_delta_items,
                        stats.dependency_env_rebase_child_items,
                        self.rebased_env_cache.len(),
                        stats.rebased_env_cache_hits,
                        stats.rebased_env_cache_misses,
                        stats.replace_answer_calls,
                        stats.replace_answer_changed,
                        stats.replace_answer_unchanged,
                        stats.replace_answer_memo_entries_cleared,
                        stats.replace_answer_changed_with_memo_entries,
                        stats.replace_answer_changed_memo_entries_cleared,
                        stats.cache_candidate_hits,
                        stats.cache_candidate_misses,
                    );

                    let mut answer_match_entries =
                        stats.answer_match_key_stats.iter().collect::<Vec<_>>();
                    answer_match_entries.sort_by(|left, right| {
                        right
                            .1
                            .time
                            .cmp(&left.1.time)
                            .then(right.1.evaluations.cmp(&left.1.evaluations))
                            .then_with(|| left.0.cmp(right.0))
                    });
                    for (key, entry) in answer_match_entries.into_iter().take(8) {
                        eprintln!(
                            concat!(
                                "[context-answer-match-time] ",
                                "ms={:.3} ",
                                "evals={} ",
                                "hits={} ",
                                "misses={} ",
                                "edges={} ",
                                "key={}"
                            ),
                            entry.time.as_secs_f64() * 1000.0,
                            entry.evaluations,
                            entry.hits,
                            entry.misses,
                            entry.edges,
                            key,
                        );
                    }

                    let mut answer_match_inputs =
                        stats.answer_match_input_stats.iter().collect::<Vec<_>>();
                    answer_match_inputs.sort_by(|left, right| {
                        let left_re_evals = left.1.evaluations.saturating_sub(1);
                        let right_re_evals = right.1.evaluations.saturating_sub(1);
                        right_re_evals
                            .cmp(&left_re_evals)
                            .then(right.1.calls.cmp(&left.1.calls))
                            .then(right.1.memo_hits.cmp(&left.1.memo_hits))
                            .then_with(|| left.0.cmp(right.0))
                    });
                    for (key, entry) in answer_match_inputs.into_iter().take(8) {
                        eprintln!(
                            concat!(
                                "[context-answer-match-input] ",
                                "calls={} ",
                                "evals={} ",
                                "reevals={} ",
                                "memo_hits={} ",
                                "in_progress_hits={} ",
                                "key={}"
                            ),
                            entry.calls,
                            entry.evaluations,
                            entry.evaluations.saturating_sub(1),
                            entry.memo_hits,
                            entry.in_progress_hits,
                            key,
                        );
                    }

                    let mut replace_answer_entries =
                        stats.replace_answer_key_stats.iter().collect::<Vec<_>>();
                    replace_answer_entries.sort_by(|left, right| {
                        right
                            .1
                            .changed_memo_entries_cleared
                            .cmp(&left.1.changed_memo_entries_cleared)
                            .then(
                                right
                                    .1
                                    .changed_with_memo_entries
                                    .cmp(&left.1.changed_with_memo_entries),
                            )
                            .then(right.1.changed.cmp(&left.1.changed))
                            .then(
                                right
                                    .1
                                    .memo_entries_cleared
                                    .cmp(&left.1.memo_entries_cleared),
                            )
                            .then_with(|| left.0.cmp(right.0))
                    });
                    for (key, entry) in replace_answer_entries.into_iter().take(8) {
                        eprintln!(
                            concat!(
                                "[context-replace-answer] ",
                                "calls={} ",
                                "changed={} ",
                                "unchanged={} ",
                                "memo_entries_cleared={} ",
                                "changed_with_memo_entries={} ",
                                "changed_memo_entries_cleared={} ",
                                "key={}"
                            ),
                            entry.calls,
                            entry.changed,
                            entry.unchanged,
                            entry.memo_entries_cleared,
                            entry.changed_with_memo_entries,
                            entry.changed_memo_entries_cleared,
                            key,
                        );
                    }
                }
            }
        }
    }

    pub(crate) fn record_active_ancestor_lazy_hit(&mut self) {
        if let Some(stats) = &mut self.stats {
            stats.active_ancestor_lazy_hits += 1;
        }
    }

    pub(crate) fn record_active_ancestor_same_depth_cycle(&mut self) {
        if let Some(stats) = &mut self.stats {
            stats.active_ancestor_same_depth_cycles += 1;
        }
    }

    pub(crate) fn record_graph_goal_hit(&mut self) {
        if let Some(stats) = &mut self.stats {
            stats.graph_goal_hits += 1;
        }
    }

    pub(crate) fn record_fixpoint_rerun(&mut self) {
        if let Some(stats) = &mut self.stats {
            stats.fixpoint_reruns += 1;
        }
    }

    pub(crate) fn record_answer(&mut self, lookup_count: usize, dependency_count: usize) {
        if let Some(stats) = &mut self.stats {
            stats.answers_recorded += 1;
            stats.answer_lookups_recorded += lookup_count as u64;
            stats.answer_dependencies_recorded += dependency_count as u64;
            stats.max_answer_lookups = stats.max_answer_lookups.max(lookup_count as u64);
            stats.max_answer_dependencies =
                stats.max_answer_dependencies.max(dependency_count as u64);
        }
    }

    pub(crate) fn record_replace_answer(
        &mut self,
        key: &str,
        changed: bool,
        memo_entries_cleared: u64,
    ) {
        if let Some(stats) = &mut self.stats {
            stats.replace_answer_calls += 1;
            if changed {
                stats.replace_answer_changed += 1;
                if memo_entries_cleared > 0 {
                    stats.replace_answer_changed_with_memo_entries += 1;
                    stats.replace_answer_changed_memo_entries_cleared += memo_entries_cleared;
                }
            } else {
                stats.replace_answer_unchanged += 1;
            }
            stats.replace_answer_memo_entries_cleared += memo_entries_cleared;
            stats.max_replace_answer_memo_entries_cleared = stats
                .max_replace_answer_memo_entries_cleared
                .max(memo_entries_cleared);

            let entry = stats
                .replace_answer_key_stats
                .entry(key.to_string())
                .or_default();
            entry.calls += 1;
            if changed {
                entry.changed += 1;
                if memo_entries_cleared > 0 {
                    entry.changed_with_memo_entries += 1;
                    entry.changed_memo_entries_cleared += memo_entries_cleared;
                }
            } else {
                entry.unchanged += 1;
            }
            entry.memo_entries_cleared += memo_entries_cleared;
        }
    }

    pub(crate) fn record_cache_bucket_probe(&mut self) {
        if let Some(stats) = &mut self.stats {
            stats.cache_bucket_probes += 1;
        }
    }

    pub(crate) fn record_cache_exact_env_bucket_probe(&mut self) {
        if let Some(stats) = &mut self.stats {
            stats.cache_exact_env_bucket_probes += 1;
        }
    }

    pub(crate) fn record_cache_bucket_probe_for(&mut self, key: &str, bucket_len: usize) {
        if std::env::var_os("INLAY_SOLVER_CACHE_STATS").is_none() {
            return;
        }
        let Some(stats) = &mut self.stats else {
            return;
        };
        let entry = stats.cache_key_stats.entry(key.to_string()).or_default();
        entry.probes += 1;
        entry.total_bucket_len += bucket_len as u64;
        entry.max_bucket_len = entry.max_bucket_len.max(bucket_len as u64);
    }

    pub(crate) fn record_cache_candidate_hit(&mut self) {
        if let Some(stats) = &mut self.stats {
            stats.cache_candidates_checked += 1;
            stats.cache_candidate_hits += 1;
        }
    }

    pub(crate) fn record_cache_exact_env_candidate_hit(&mut self) {
        if let Some(stats) = &mut self.stats {
            stats.cache_exact_env_candidates_checked += 1;
            stats.cache_exact_env_candidate_hits += 1;
        }
    }

    pub(crate) fn record_cache_candidate_hit_for(&mut self, key: &str) {
        if std::env::var_os("INLAY_SOLVER_CACHE_STATS").is_none() {
            return;
        }
        let Some(stats) = &mut self.stats else {
            return;
        };
        let entry = stats.cache_key_stats.entry(key.to_string()).or_default();
        entry.candidates_checked += 1;
        entry.hits += 1;
    }

    pub(crate) fn record_cache_candidate_miss(&mut self) {
        if let Some(stats) = &mut self.stats {
            stats.cache_candidates_checked += 1;
            stats.cache_candidate_misses += 1;
        }
    }

    pub(crate) fn record_cache_exact_env_candidate_miss(&mut self) {
        if let Some(stats) = &mut self.stats {
            stats.cache_exact_env_candidates_checked += 1;
            stats.cache_exact_env_candidate_misses += 1;
        }
    }

    pub(crate) fn record_cache_candidate_miss_for(&mut self, key: &str) {
        if std::env::var_os("INLAY_SOLVER_CACHE_STATS").is_none() {
            return;
        }
        let Some(stats) = &mut self.stats else {
            return;
        };
        let entry = stats.cache_key_stats.entry(key.to_string()).or_default();
        entry.candidates_checked += 1;
        entry.misses += 1;
    }

    pub(crate) fn record_lookup_match(&mut self, entry_count: usize, elapsed: Duration) {
        if let Some(stats) = &mut self.stats {
            stats.lookup_match_calls += 1;
            stats.lookup_match_entries += entry_count as u64;
            stats.lookup_match_time += elapsed;
        }
    }

    pub(crate) fn record_answer_match_call(&mut self, depth: usize) {
        if let Some(stats) = &mut self.stats {
            stats.answer_match_calls += 1;
            stats.answer_match_max_depth = stats.answer_match_max_depth.max(depth as u64);
        }
    }

    pub(crate) fn record_answer_match_input_call(&mut self, key: &str) {
        if let Some(stats) = &mut self.stats {
            stats
                .answer_match_input_stats
                .entry(key.to_string())
                .or_default()
                .calls += 1;
        }
    }

    pub(crate) fn record_answer_match_input_memo_hit(&mut self, key: &str) {
        if let Some(stats) = &mut self.stats {
            stats
                .answer_match_input_stats
                .entry(key.to_string())
                .or_default()
                .memo_hits += 1;
        }
    }

    pub(crate) fn record_answer_match_input_in_progress_hit(&mut self, key: &str) {
        if let Some(stats) = &mut self.stats {
            stats
                .answer_match_input_stats
                .entry(key.to_string())
                .or_default()
                .in_progress_hits += 1;
        }
    }

    pub(crate) fn record_answer_match_evaluation(
        &mut self,
        key: &str,
        dependency_count: usize,
        elapsed: Duration,
        matches: bool,
    ) {
        if let Some(stats) = &mut self.stats {
            stats.answer_match_evaluations += 1;
            stats.answer_match_edges += dependency_count as u64;
            stats.answer_match_time += elapsed;
            let entry = stats
                .answer_match_key_stats
                .entry(key.to_string())
                .or_default();
            entry.evaluations += 1;
            entry.edges += dependency_count as u64;
            entry.time += elapsed;
            if matches {
                entry.hits += 1;
            } else {
                entry.misses += 1;
            }
            stats
                .answer_match_input_stats
                .entry(key.to_string())
                .or_default()
                .evaluations += 1;
        }
    }

    pub(crate) fn record_dependency_env_delta(&mut self, delta_item_count: usize) {
        if let Some(stats) = &mut self.stats {
            stats.dependency_env_delta_items_recorded += delta_item_count as u64;
            stats.max_dependency_env_delta_items = stats
                .max_dependency_env_delta_items
                .max(delta_item_count as u64);
        }
    }

    pub(crate) fn record_dependency_env_rebase(
        &mut self,
        parent_item_count: usize,
        delta_item_count: usize,
        child_item_count: usize,
    ) {
        if let Some(stats) = &mut self.stats {
            stats.dependency_env_rebases += 1;
            stats.dependency_env_rebase_parent_items += parent_item_count as u64;
            stats.dependency_env_rebase_delta_items += delta_item_count as u64;
            stats.dependency_env_rebase_child_items += child_item_count as u64;
            stats.max_dependency_env_rebase_parent_items = stats
                .max_dependency_env_rebase_parent_items
                .max(parent_item_count as u64);
            stats.max_dependency_env_rebase_delta_items = stats
                .max_dependency_env_rebase_delta_items
                .max(delta_item_count as u64);
            stats.max_dependency_env_rebase_child_items = stats
                .max_dependency_env_rebase_child_items
                .max(child_item_count as u64);
            stats.max_answer_match_memo_entries = stats
                .max_answer_match_memo_entries
                .max(self.answer_match_memo.len() as u64);
        }
    }

    pub(crate) fn rebased_env_for_dependency(
        &mut self,
        parent: &Arc<RuleEnv<R>>,
        delta: &<RuleEnv<R> as crate::rule::ResolutionEnv>::DependencyEnvDelta,
    ) -> Arc<RuleEnv<R>> {
        let key = (Arc::clone(parent), delta.clone());
        if let Some(env) = self.rebased_env_cache.get(&key).cloned() {
            if let Some(stats) = &mut self.stats {
                stats.rebased_env_cache_hits += 1;
            }
            return env;
        }

        let env = RuleEnv::<R>::apply_dependency_env_delta(parent, delta);
        self.rebased_env_cache.insert(key, Arc::clone(&env));
        if let Some(stats) = &mut self.stats {
            stats.rebased_env_cache_misses += 1;
            stats.max_rebased_env_cache_entries = stats
                .max_rebased_env_cache_entries
                .max(self.rebased_env_cache.len() as u64);
        }
        env
    }

    pub(crate) fn record_answer_match_memo_hit(&mut self) {
        if let Some(stats) = &mut self.stats {
            stats.answer_match_memo_hits += 1;
        }
    }

    pub(crate) fn record_answer_match_in_progress_hit(&mut self) {
        if let Some(stats) = &mut self.stats {
            stats.answer_match_in_progress_hits += 1;
        }
    }

    pub(crate) fn record_answer_match_missing_answer(&mut self) {
        if let Some(stats) = &mut self.stats {
            stats.answer_match_missing_answers += 1;
        }
    }

    pub(crate) fn emit_stats(&self) {
        let Some(stats) = &self.stats else {
            return;
        };

        eprintln!(
            concat!(
                "[context-solver-stats] ",
                "goal_attempts={} ",
                "new_goals={} ",
                "active_ancestor_lazy_hits={} ",
                "active_ancestor_same_depth_cycles={} ",
                "graph_goal_hits={} ",
                "fixpoint_reruns={} ",
                "lookup_calls={} ",
                "lookup_ms={:.3} ",
                "answers_recorded={} ",
                "answer_lookups_recorded={} ",
                "answer_dependencies_recorded={} ",
                "max_answer_lookups={} ",
                "max_answer_dependencies={} ",
                "cache_bucket_probes={} ",
                "cache_exact_env_bucket_probes={} ",
                "cache_candidates_checked={} ",
                "cache_exact_env_candidates_checked={} ",
                "cache_candidate_hits={} ",
                "cache_exact_env_candidate_hits={} ",
                "cache_candidate_misses={} ",
                "cache_exact_env_candidate_misses={} ",
                "lookup_match_calls={} ",
                "lookup_match_entries={} ",
                "lookup_match_ms={:.3} ",
                "answer_match_calls={} ",
                "answer_match_evaluations={} ",
                "answer_match_edges={} ",
                "answer_match_memo_hits={} ",
                "answer_match_in_progress_hits={} ",
                "answer_match_missing_answers={} ",
                "answer_match_max_depth={} ",
                "answer_match_ms={:.3} ",
                "dependency_env_delta_items_recorded={} ",
                "max_dependency_env_delta_items={} ",
                "dependency_env_rebases={} ",
                "dependency_env_rebase_parent_items={} ",
                "dependency_env_rebase_delta_items={} ",
                "dependency_env_rebase_child_items={} ",
                "max_dependency_env_rebase_parent_items={} ",
                "max_dependency_env_rebase_delta_items={} ",
                "max_dependency_env_rebase_child_items={} ",
                "max_answer_match_memo_entries={} ",
                "rebased_env_cache_hits={} ",
                "rebased_env_cache_misses={} ",
                "max_rebased_env_cache_entries={} ",
                "replace_answer_calls={} ",
                "replace_answer_changed={} ",
                "replace_answer_unchanged={} ",
                "replace_answer_memo_entries_cleared={} ",
                "max_replace_answer_memo_entries_cleared={} ",
                "replace_answer_changed_with_memo_entries={} ",
                "replace_answer_changed_memo_entries_cleared={} ",
                "result_answers={} ",
                "cache_keys={}"
            ),
            stats.goal_attempts,
            stats.new_goals,
            stats.active_ancestor_lazy_hits,
            stats.active_ancestor_same_depth_cycles,
            stats.graph_goal_hits,
            stats.fixpoint_reruns,
            stats.lookup_calls,
            stats.lookup_time.as_secs_f64() * 1000.0,
            stats.answers_recorded,
            stats.answer_lookups_recorded,
            stats.answer_dependencies_recorded,
            stats.max_answer_lookups,
            stats.max_answer_dependencies,
            stats.cache_bucket_probes,
            stats.cache_exact_env_bucket_probes,
            stats.cache_candidates_checked,
            stats.cache_exact_env_candidates_checked,
            stats.cache_candidate_hits,
            stats.cache_exact_env_candidate_hits,
            stats.cache_candidate_misses,
            stats.cache_exact_env_candidate_misses,
            stats.lookup_match_calls,
            stats.lookup_match_entries,
            stats.lookup_match_time.as_secs_f64() * 1000.0,
            stats.answer_match_calls,
            stats.answer_match_evaluations,
            stats.answer_match_edges,
            stats.answer_match_memo_hits,
            stats.answer_match_in_progress_hits,
            stats.answer_match_missing_answers,
            stats.answer_match_max_depth,
            stats.answer_match_time.as_secs_f64() * 1000.0,
            stats.dependency_env_delta_items_recorded,
            stats.max_dependency_env_delta_items,
            stats.dependency_env_rebases,
            stats.dependency_env_rebase_parent_items,
            stats.dependency_env_rebase_delta_items,
            stats.dependency_env_rebase_child_items,
            stats.max_dependency_env_rebase_parent_items,
            stats.max_dependency_env_rebase_delta_items,
            stats.max_dependency_env_rebase_child_items,
            stats.max_answer_match_memo_entries,
            stats.rebased_env_cache_hits,
            stats.rebased_env_cache_misses,
            stats.max_rebased_env_cache_entries,
            stats.replace_answer_calls,
            stats.replace_answer_changed,
            stats.replace_answer_unchanged,
            stats.replace_answer_memo_entries_cleared,
            stats.max_replace_answer_memo_entries_cleared,
            stats.replace_answer_changed_with_memo_entries,
            stats.replace_answer_changed_memo_entries_cleared,
            self.result_answers.len(),
            self.cache.len(),
        );

        if std::env::var_os("INLAY_SOLVER_CACHE_STATS").is_some() {
            let mut entries = stats.cache_key_stats.iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| {
                right
                    .1
                    .misses
                    .cmp(&left.1.misses)
                    .then(right.1.candidates_checked.cmp(&left.1.candidates_checked))
            });
            for (key, entry) in entries.into_iter().take(25) {
                let avg_bucket_len = if entry.probes == 0 {
                    0.0
                } else {
                    entry.total_bucket_len as f64 / entry.probes as f64
                };
                eprintln!(
                    concat!(
                        "[context-solver-cache-stats] ",
                        "key={} ",
                        "probes={} ",
                        "checked={} ",
                        "hits={} ",
                        "misses={} ",
                        "avg_bucket_len={:.2} ",
                        "max_bucket_len={}"
                    ),
                    key,
                    entry.probes,
                    entry.candidates_checked,
                    entry.hits,
                    entry.misses,
                    avg_bucket_len,
                    entry.max_bucket_len,
                );
            }
        }

        let mut answer_match_entries = stats.answer_match_key_stats.iter().collect::<Vec<_>>();
        answer_match_entries.sort_by(|left, right| {
            right
                .1
                .time
                .cmp(&left.1.time)
                .then(right.1.evaluations.cmp(&left.1.evaluations))
                .then_with(|| left.0.cmp(right.0))
        });
        for (key, entry) in answer_match_entries.into_iter().take(25) {
            eprintln!(
                concat!(
                    "[context-answer-match-stats] ",
                    "key={} ",
                    "evals={} ",
                    "hits={} ",
                    "misses={} ",
                    "edges={} ",
                    "ms={:.3}"
                ),
                key,
                entry.evaluations,
                entry.hits,
                entry.misses,
                entry.edges,
                entry.time.as_secs_f64() * 1000.0,
            );
        }

        let mut replace_answer_entries = stats.replace_answer_key_stats.iter().collect::<Vec<_>>();
        replace_answer_entries.sort_by(|left, right| {
            right
                .1
                .changed_memo_entries_cleared
                .cmp(&left.1.changed_memo_entries_cleared)
                .then(
                    right
                        .1
                        .changed_with_memo_entries
                        .cmp(&left.1.changed_with_memo_entries),
                )
                .then(right.1.changed.cmp(&left.1.changed))
                .then(
                    right
                        .1
                        .memo_entries_cleared
                        .cmp(&left.1.memo_entries_cleared),
                )
                .then_with(|| left.0.cmp(right.0))
        });
        for (key, entry) in replace_answer_entries.into_iter().take(25) {
            eprintln!(
                concat!(
                    "[context-replace-answer-stats] ",
                    "key={} ",
                    "calls={} ",
                    "changed={} ",
                    "unchanged={} ",
                    "memo_entries_cleared={} ",
                    "changed_with_memo_entries={} ",
                    "changed_memo_entries_cleared={}"
                ),
                key,
                entry.calls,
                entry.changed,
                entry.unchanged,
                entry.memo_entries_cleared,
                entry.changed_with_memo_entries,
                entry.changed_memo_entries_cleared,
            );
        }
    }
}

impl SolverTrace {
    fn new() -> Option<Self> {
        let path = std::env::var_os("INLAY_SOLVER_TRACE_PATH")?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)
            .ok()?;
        Some(Self {
            writer: BufWriter::new(file),
            filter: std::env::var("INLAY_SOLVER_TRACE_FILTER").ok(),
            next_seq: 0,
        })
    }
}
