#![cfg_attr(not(feature = "tracing"), allow(unused_variables))]

use std::{hash::Hash, sync::Arc};

use derive_where::derive_where;
use inlay_instrument::{instrumented, solver_event, span_record as solver_span_record};
use rustc_hash::FxHashSet as HashSet;

use crate::{
    cache::{Cache, CachedResultRef},
    context::Context,
    rule::{RuleDependencyEnvDelta, RuleEnv, RuleResultRef},
    search_graph::{Dependency, SearchGraph},
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

#[derive_where(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GraphDependencyRef<R: Rule> {
    Graph(RuleResultRef<R>),
    Cached(CachedResultRef<R>),
}

pub(crate) struct SupportAnswer<R: Rule, DependencyRef: Copy + Eq + Hash> {
    pub(crate) direct_supports: LookupSupports<R>,
    pub(crate) dependencies: Vec<Dependency<R, DependencyRef>>,
}

pub(crate) enum DependencyResolution<R: Rule, AnswerRef> {
    Precomputed(AnswerSupport<R>),
    Traverse(AnswerRef),
}

pub(crate) trait AnswerSupportSource<R: Rule> {
    type AnswerRef: Copy + Eq + Hash;
    type DependencyRef: Copy + Eq + Hash;
    type Error;

    fn answer_for(
        &mut self,
        result_ref: Self::AnswerRef,
    ) -> Result<SupportAnswer<R, Self::DependencyRef>, Self::Error>;

    fn resolve_dependency(
        &mut self,
        result_ref: Self::DependencyRef,
    ) -> Result<DependencyResolution<R, Self::AnswerRef>, Self::Error>;
}

struct GraphAnswerSupportSource<'a, R: Rule> {
    search_graph: &'a SearchGraph<R>,
    cache: &'a mut Cache<R>,
}

impl<R: Rule> AnswerSupportSource<R> for GraphAnswerSupportSource<'_, R> {
    type AnswerRef = RuleResultRef<R>;
    type DependencyRef = GraphDependencyRef<R>;
    type Error = AnswerSupportBuildError;

    fn answer_for(
        &mut self,
        result_ref: Self::AnswerRef,
    ) -> Result<SupportAnswer<R, Self::DependencyRef>, Self::Error> {
        let answer = self
            .search_graph
            .answer_for(result_ref)
            .cloned()
            .ok_or(AnswerSupportBuildError::MissingAnswer)?;
        Ok(SupportAnswer {
            direct_supports: answer.direct_supports,
            dependencies: answer
                .dependencies
                .into_iter()
                .map(|dependency| Dependency {
                    result_ref: self.dependency_ref(dependency.result_ref),
                    env_delta: dependency.env_delta,
                })
                .collect(),
        })
    }

    fn resolve_dependency(
        &mut self,
        result_ref: Self::DependencyRef,
    ) -> Result<DependencyResolution<R, Self::AnswerRef>, Self::Error> {
        match result_ref {
            GraphDependencyRef::Graph(result_ref) => {
                if let Some(support) = self.search_graph.stored_answer_support(result_ref) {
                    return Ok(DependencyResolution::Precomputed(support.clone()));
                }
                Ok(DependencyResolution::Traverse(result_ref))
            }
            GraphDependencyRef::Cached(result_ref) => Ok(DependencyResolution::Precomputed(
                self.cache.answer_support(result_ref),
            )),
        }
    }
}

impl<R: Rule> GraphAnswerSupportSource<'_, R> {
    fn dependency_ref(&self, result_ref: RuleResultRef<R>) -> GraphDependencyRef<R> {
        if self.search_graph.answer_for(result_ref).is_some() {
            return GraphDependencyRef::Graph(result_ref);
        }
        self.cache.cached_result_ref(result_ref).map_or(
            GraphDependencyRef::Graph(result_ref),
            GraphDependencyRef::Cached,
        )
    }
}

fn collect_answer_support<R: Rule, SourceT: AnswerSupportSource<R>>(
    source: &mut SourceT,
    result_ref: SourceT::AnswerRef,
    visit_ref: SourceT::DependencyRef,
    path_delta: Option<RuleDependencyEnvDelta<R>>,
    visited: &mut HashSet<(SourceT::DependencyRef, Option<RuleDependencyEnvDelta<R>>)>,
    checks: &mut Vec<RuleLookupSupport<R>>,
    answer_nodes: &mut u64,
) -> Result<(), SourceT::Error> {
    if !visited.insert((visit_ref, path_delta.clone())) {
        return Ok(());
    }
    *answer_nodes += 1;

    let answer = source.answer_for(result_ref)?;
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
        match source.resolve_dependency(dependency.result_ref)? {
            DependencyResolution::Precomputed(child_support) => {
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
            }
            DependencyResolution::Traverse(result_ref) => {
                collect_answer_support(
                    source,
                    result_ref,
                    dependency.result_ref,
                    Some(dependency_path_delta),
                    visited,
                    checks,
                    answer_nodes,
                )?;
            }
        }
    }

    Ok(())
}

pub(crate) fn build_answer_support<R: Rule, SourceT: AnswerSupportSource<R>>(
    source: &mut SourceT,
    result_ref: SourceT::AnswerRef,
    visit_ref: SourceT::DependencyRef,
) -> Result<AnswerSupport<R>, SourceT::Error> {
    let mut visited = HashSet::default();
    let mut checks = Vec::new();
    let mut answer_nodes = 0_u64;

    if let Err(error) = collect_answer_support(
        source,
        result_ref,
        visit_ref,
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
    cache: &mut Cache<R>,
    result_ref: RuleResultRef<R>,
) -> Result<AnswerSupport<R>, AnswerSupportBuildError> {
    let mut source = GraphAnswerSupportSource {
        search_graph,
        cache,
    };
    build_answer_support(
        &mut source,
        result_ref,
        GraphDependencyRef::Graph(result_ref),
    )
}

impl<R: Rule> Context<R> {
    pub(crate) fn cached_answer_support(
        &mut self,
        result_ref: CachedResultRef<R>,
    ) -> AnswerSupport<R> {
        self.cache.answer_support(result_ref)
    }

    pub(crate) fn graph_answer_support(
        &mut self,
        result_ref: RuleResultRef<R>,
    ) -> Result<AnswerSupport<R>, AnswerSupportBuildError> {
        if let Some(support) = self.search_graph.stored_answer_support(result_ref).cloned() {
            return Ok(support);
        }

        let support = build_graph_answer_support(&self.search_graph, &mut self.cache, result_ref)?;
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
