use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::Duration;

use crate::{
    arena::Arena,
    rule::{Rule, RuleEnv, RuleEnvSharedState, RuleResultRef, RuleResultsArena},
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

pub(crate) struct Context<R: Rule> {
    pub(crate) results_arena: RuleResultsArena<R>,
    pub(crate) result_refs: HashMap<GoalKey<R>, RuleResultRef<R>>,
    pub(crate) result_answers: HashMap<RuleResultRef<R>, Answer<R>>,
    pub(crate) answer_fingerprints: HashMap<RuleResultRef<R>, u64>,
    pub(crate) answer_dependents: HashMap<RuleResultRef<R>, HashSet<RuleResultRef<R>>>,
    pub(crate) answer_match_memo: HashMap<AnswerMatchMemoKey<R>, AnswerMatchMemo>,
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
    pub(crate) cache_key_stats: BTreeMap<String, CacheKeyStats>,
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

impl<R: Rule> Context<R> {
    pub(crate) fn new(
        env_shared_state: RuleEnvSharedState<R>,
        fixpoint_iteration_limit: usize,
        stack_overflow_depth: usize,
    ) -> Self {
        Self {
            results_arena: RuleResultsArena::<R>::default(),
            result_refs: HashMap::new(),
            result_answers: HashMap::new(),
            answer_fingerprints: HashMap::new(),
            answer_dependents: HashMap::new(),
            answer_match_memo: HashMap::new(),
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

    pub(crate) fn result_ref_for(&mut self, goal: &GoalKey<R>) -> RuleResultRef<R> {
        if let Some(result_ref) = self.result_refs.get(goal).copied() {
            return result_ref;
        }

        let result_ref = self.results_arena.insert_placeholder();
        self.result_refs.insert(goal.clone(), result_ref);
        result_ref
    }

    pub(crate) fn replace_answer(&mut self, result_ref: RuleResultRef<R>, answer: Answer<R>) {
        let old_dependencies = self
            .result_answers
            .insert(result_ref, answer)
            .map(|old| old.dependencies)
            .unwrap_or_default();

        for dependency in old_dependencies {
            if let Some(dependents) = self.answer_dependents.get_mut(&dependency) {
                dependents.remove(&result_ref);
                if dependents.is_empty() {
                    self.answer_dependents.remove(&dependency);
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
                .entry(dependency)
                .or_default()
                .insert(result_ref);
        }

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
                    eprintln!(
                        concat!(
                            "[context-solver-progress] ",
                            "new_goals={} ",
                            "goal_attempts={} ",
                            "active_lazy_hits={} ",
                            "graph_goal_hits={} ",
                            "fixpoint_reruns={} ",
                            "blocked_cross_env_reuses={} ",
                            "result_answers={}"
                        ),
                        stats.new_goals,
                        stats.goal_attempts,
                        stats.active_ancestor_lazy_hits,
                        stats.graph_goal_hits,
                        stats.fixpoint_reruns,
                        self.blocked_cross_env_reuses.len(),
                        self.result_answers.len(),
                    );
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

    pub(crate) fn record_answer_match_evaluation(
        &mut self,
        dependency_count: usize,
        elapsed: Duration,
    ) {
        if let Some(stats) = &mut self.stats {
            stats.answer_match_evaluations += 1;
            stats.answer_match_edges += dependency_count as u64;
            stats.answer_match_time += elapsed;
        }
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
