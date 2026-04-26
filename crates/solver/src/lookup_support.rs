#![cfg_attr(not(feature = "tracing"), allow(unused_variables))]

use std::{convert::Infallible, sync::Arc};

use derive_where::derive_where;
use inlay_instrument_macros::instrumented;
use rustc_hash::FxHashSet as HashSet;

use crate::{
    cache::{Cache, CachedResultRef},
    context::Context,
    instrument::{solver_event, solver_span_record},
    rule::{RuleDependencyEnvDelta, RuleEnv, RuleResultRef},
    search_graph::{Answer, SearchGraph},
    traits::{ResolutionEnv, Rule},
};

pub(crate) type RuleLookupSupport<R> = <RuleEnv<R> as ResolutionEnv>::LookupSupport;
pub(crate) type LookupSupports<R> = Vec<RuleLookupSupport<R>>;
#[derive_where(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AnswerSupport<R: Rule> {
    pub(crate) checks: Vec<RuleLookupSupport<R>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AnswerSupportBuildError {
    MissingAnswer,
    MissingGraphSupportTarget,
}

fn insert_compact_lookup_support<R: Rule>(
    compacted: &mut LookupSupports<R>,
    mut support: RuleLookupSupport<R>,
) {
    let mut index = 0;
    while index < compacted.len() {
        let Some(merged) =
            crate::traits::RuleLookupSupport::merge_lookup_support(&compacted[index], &support)
        else {
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
    path_delta: Option<&RuleDependencyEnvDelta<R>>,
    support: &RuleLookupSupport<R>,
) {
    let support = match path_delta {
        Some(path_delta) => RuleEnv::<R>::pullback_lookup_support(support, path_delta),
        None => support.clone(),
    };
    insert_compact_lookup_support::<R>(checks, support);
}

trait AnswerSupportSource<R: Rule> {
    type Error;

    fn answer_for(&self, result_ref: RuleResultRef<R>) -> Result<&Answer<R>, Self::Error>;

    fn stored_answer_support(
        &self,
        result_ref: RuleResultRef<R>,
    ) -> Result<Option<&AnswerSupport<R>>, Self::Error>;
}

struct GraphAnswerSupportSource<'a, R: Rule> {
    search_graph: &'a SearchGraph<R>,
    cache: &'a Cache<R>,
}

impl<R: Rule> AnswerSupportSource<R> for GraphAnswerSupportSource<'_, R> {
    type Error = AnswerSupportBuildError;

    fn answer_for(&self, result_ref: RuleResultRef<R>) -> Result<&Answer<R>, Self::Error> {
        self.search_graph
            .answer_for(result_ref)
            .map(|answer| Ok(answer))
            .or_else(|| {
                self.cache
                    .cached_result_ref(result_ref)
                    .map(|cached_result_ref| Ok(self.cache.answer_for(cached_result_ref)))
            })
            .unwrap_or(Err(AnswerSupportBuildError::MissingAnswer))
    }

    fn stored_answer_support(
        &self,
        result_ref: RuleResultRef<R>,
    ) -> Result<Option<&AnswerSupport<R>>, Self::Error> {
        if let Some(support) = self.search_graph.stored_answer_support(result_ref) {
            return Ok(Some(support));
        }
        let Some(cached_result_ref) = self.cache.cached_result_ref(result_ref) else {
            return Ok(None);
        };
        Ok(self.cache.stored_answer_support(cached_result_ref))
    }
}

struct CacheAnswerSupportSource<'a, R: Rule> {
    cache: &'a Cache<R>,
}

impl<R: Rule> AnswerSupportSource<R> for CacheAnswerSupportSource<'_, R> {
    type Error = Infallible;

    fn answer_for(&self, result_ref: RuleResultRef<R>) -> Result<&Answer<R>, Self::Error> {
        Ok(self.cache.answer_for(self.cached_result_ref(result_ref)))
    }

    fn stored_answer_support(
        &self,
        result_ref: RuleResultRef<R>,
    ) -> Result<Option<&AnswerSupport<R>>, Self::Error> {
        Ok(self
            .cache
            .stored_answer_support(self.cached_result_ref(result_ref)))
    }
}

impl<R: Rule> CacheAnswerSupportSource<'_, R> {
    fn cached_result_ref(&self, result_ref: RuleResultRef<R>) -> CachedResultRef<R> {
        self.cache
            .cached_result_ref(result_ref)
            .expect("cached answer dependency must remain cached")
    }
}

fn collect_answer_support<R: Rule, SourceT: AnswerSupportSource<R>>(
    source: &SourceT,
    result_ref: RuleResultRef<R>,
    path_delta: Option<RuleDependencyEnvDelta<R>>,
    visited: &mut HashSet<(RuleResultRef<R>, Option<RuleDependencyEnvDelta<R>>)>,
    checks: &mut Vec<RuleLookupSupport<R>>,
    answer_nodes: &mut u64,
) -> Result<(), SourceT::Error> {
    if !visited.insert((result_ref, path_delta.clone())) {
        return Ok(());
    }
    *answer_nodes += 1;

    let answer = source.answer_for(result_ref)?.clone();
    for support in &answer.direct_supports {
        insert_transported_support_check::<R>(checks, path_delta.as_ref(), support);
    }

    for dependency in &answer.dependencies {
        let dependency_path_delta = path_delta.as_ref().map_or_else(
            || dependency.env_delta.clone(),
            |path_delta| {
                RuleEnv::<R>::compose_dependency_env_delta(path_delta, &dependency.env_delta)
            },
        );
        if let Some(child_support) = source
            .stored_answer_support(dependency.result_ref)?
            .cloned()
        {
            if !visited.insert((dependency.result_ref, Some(dependency_path_delta.clone()))) {
                continue;
            }
            for support in &child_support.checks {
                insert_transported_support_check::<R>(
                    checks,
                    Some(&dependency_path_delta),
                    support,
                );
            }
        } else {
            collect_answer_support(
                source,
                dependency.result_ref,
                Some(dependency_path_delta),
                visited,
                checks,
                answer_nodes,
            )?;
        }
    }

    Ok(())
}

fn build_answer_support<R: Rule, SourceT: AnswerSupportSource<R>>(
    source: &SourceT,
    result_ref: RuleResultRef<R>,
) -> Result<AnswerSupport<R>, SourceT::Error> {
    let mut visited = HashSet::default();
    let mut checks = Vec::new();
    let mut answer_nodes = 0_u64;

    if let Err(error) = collect_answer_support(
        source,
        result_ref,
        None,
        &mut visited,
        &mut checks,
        &mut answer_nodes,
    ) {
        solver_span_record!(
            answer_nodes,
            checks = checks.len() as u64,
            missing_answer = true
        );
        return Err(error);
    }

    solver_span_record!(
        answer_nodes,
        checks = checks.len() as u64,
        missing_answer = false
    );
    Ok(AnswerSupport { checks })
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
) -> Result<AnswerSupport<R>, AnswerSupportBuildError> {
    build_answer_support(
        &GraphAnswerSupportSource {
            search_graph,
            cache,
        },
        result_ref,
    )
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
    result_ref: CachedResultRef<R>,
) -> AnswerSupport<R> {
    match build_answer_support(&CacheAnswerSupportSource { cache }, result_ref.result_ref()) {
        Ok(support) => support,
        Err(error) => match error {},
    }
}

impl<R: Rule> Context<R> {
    pub(crate) fn cached_answer_support(
        &mut self,
        result_ref: CachedResultRef<R>,
    ) -> AnswerSupport<R> {
        if let Some(support) = self.cache.stored_answer_support(result_ref) {
            return support.clone();
        }

        let support = build_cached_answer_support(&self.cache, result_ref);
        self.cache.store_answer_support(result_ref, support.clone());
        support
    }

    pub(crate) fn graph_answer_support(
        &mut self,
        result_ref: RuleResultRef<R>,
    ) -> Result<AnswerSupport<R>, AnswerSupportBuildError> {
        if let Some(support) = self.search_graph.stored_answer_support(result_ref).cloned() {
            return Ok(support);
        }

        let support = build_graph_answer_support(&self.search_graph, &self.cache, result_ref)?;
        if !self
            .search_graph
            .store_answer_support(result_ref, support.clone())
        {
            return Err(AnswerSupportBuildError::MissingGraphSupportTarget);
        }
        Ok(support)
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
