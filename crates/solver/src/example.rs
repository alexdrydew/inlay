use std::collections::BTreeMap;
use std::hash::Hash;
use std::sync::Arc;

use slotmap::{SlotMap, new_key_type};
use thiserror::Error;

use crate::{
    rule::{LazyDepthMode, RuleContext, RunError},
    solve::{SolveError, SolveResult, solve},
    traits::{Arena, ReplaceError, ResolutionEnv, Rule, RuleLookupSupport},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExampleEdgeKind {
    Eager,
    Lazy,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExampleEdge {
    pub kind: ExampleEdgeKind,
    pub target: String,
    pub scope: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExampleSpec {
    Leaf(String),
    Node(Vec<ExampleEdge>),
    MatchFirst(Vec<String>),
    DeferredSwitch {
        immediate: Box<ExampleSpec>,
        deferred: Box<ExampleSpec>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExampleDefinition {
    pub name: String,
    pub spec: ExampleSpec,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct ExampleEnv {
    is_deferred: bool,
    scope: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExampleEnvDelta {
    set_deferred: bool,
    scope: Option<String>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ExampleSharedState {
    definitions: Arc<BTreeMap<String, ExampleSpec>>,
    pub support_validations: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExampleLookupSupport {
    results: BTreeMap<(ExampleEnvDelta, String), ExampleSpec>,
}

impl RuleLookupSupport for ExampleLookupSupport {
    fn merge_lookup_support(&self, other: &Self) -> Option<Self> {
        let mut results = self.results.clone();

        for (query, result) in &other.results {
            match results.insert(query.clone(), result.clone()) {
                Some(existing) if existing != *result => return None,
                _ => {}
            }
        }

        Some(Self { results })
    }
}

impl ExampleSharedState {
    pub fn new(definitions: impl IntoIterator<Item = ExampleDefinition>) -> Self {
        let mut definitions_by_name = BTreeMap::new();

        for definition in definitions {
            assert!(
                definitions_by_name
                    .insert(definition.name, definition.spec)
                    .is_none(),
                "example definitions must have unique names"
            );
        }

        Self {
            definitions: Arc::new(definitions_by_name),
            support_validations: 0,
        }
    }
}

impl ExampleEnv {
    #[cfg(test)]
    pub(crate) fn scoped(scope: impl Into<String>) -> Self {
        Self {
            is_deferred: false,
            scope: Some(scope.into()),
        }
    }

    fn descend(&self, edge: &ExampleEdge) -> Self {
        Self {
            is_deferred: self.is_deferred || matches!(edge.kind, ExampleEdgeKind::Lazy),
            scope: edge.scope.clone().or_else(|| self.scope.clone()),
        }
    }

    fn resolve_spec(&self, spec: &ExampleSpec) -> ExampleSpec {
        match spec {
            ExampleSpec::DeferredSwitch {
                immediate,
                deferred,
            } => self.resolve_spec(if self.is_deferred {
                deferred.as_ref()
            } else {
                immediate.as_ref()
            }),
            _ => spec.clone(),
        }
    }

    fn resolve_spec_with_delta(&self, delta: &ExampleEnvDelta, spec: &ExampleSpec) -> ExampleSpec {
        Self {
            is_deferred: self.is_deferred || delta.set_deferred,
            scope: delta.scope.clone().or_else(|| self.scope.clone()),
        }
        .resolve_spec(spec)
    }
}

impl ResolutionEnv for ExampleEnv {
    type SharedState = ExampleSharedState;
    type Query = String;
    type QueryResult = ExampleSpec;
    type DependencyEnvDelta = ExampleEnvDelta;
    type LookupSupport = ExampleLookupSupport;

    fn lookup(
        self: &Arc<Self>,
        shared_state: &mut Self::SharedState,
        query: &Self::Query,
    ) -> Self::QueryResult {
        self.resolve_spec(
            shared_state
                .definitions
                .get(query)
                .unwrap_or_else(|| panic!("example definition '{query}' not found")),
        )
    }

    fn lookup_support(
        self: &Arc<Self>,
        _shared_state: &mut Self::SharedState,
        query: &Self::Query,
        result: &Self::QueryResult,
    ) -> Self::LookupSupport {
        ExampleLookupSupport {
            results: [(
                (Self::dependency_env_delta(self, self), query.clone()),
                result.clone(),
            )]
            .into(),
        }
    }

    fn lookup_support_matches(
        self: &Arc<Self>,
        shared_state: &mut Self::SharedState,
        support: &Self::LookupSupport,
    ) -> bool {
        shared_state.support_validations += 1;
        support.results.iter().all(|((delta, query), result)| {
            shared_state
                .definitions
                .get(query)
                .is_some_and(|spec| self.resolve_spec_with_delta(delta, spec) == *result)
        })
    }

    fn pullback_lookup_support(
        support: &Self::LookupSupport,
        delta: &Self::DependencyEnvDelta,
    ) -> Self::LookupSupport {
        Self::LookupSupport {
            results: support
                .results
                .iter()
                .map(|((support_delta, query), result)| {
                    (
                        (
                            Self::compose_dependency_env_delta(delta, support_delta),
                            query.clone(),
                        ),
                        result.clone(),
                    )
                })
                .collect(),
        }
    }

    fn dependency_env_delta(parent: &Arc<Self>, child: &Arc<Self>) -> Self::DependencyEnvDelta {
        Self::DependencyEnvDelta {
            set_deferred: child.is_deferred && !parent.is_deferred,
            scope: (child.scope != parent.scope)
                .then(|| child.scope.clone())
                .flatten(),
        }
    }

    fn compose_dependency_env_delta(
        first: &Self::DependencyEnvDelta,
        second: &Self::DependencyEnvDelta,
    ) -> Self::DependencyEnvDelta {
        Self::DependencyEnvDelta {
            set_deferred: first.set_deferred || second.set_deferred,
            scope: second.scope.clone().or_else(|| first.scope.clone()),
        }
    }

    fn dependency_env_delta_item_count(delta: &Self::DependencyEnvDelta) -> usize {
        usize::from(delta.set_deferred) + usize::from(delta.scope.is_some())
    }
}

new_key_type! {
    pub struct ExampleResultRef;
}

pub type ExampleResult = Result<ExampleOutput, ExampleRuleError>;

#[derive(Debug, Default)]
pub struct ExampleResultsArena {
    results: SlotMap<ExampleResultRef, Option<ExampleResult>>,
}

impl ExampleResultsArena {
    pub fn result(&self, key: ExampleResultRef) -> &ExampleResult {
        self.results
            .get(key)
            .and_then(Option::as_ref)
            .expect("example result ref must point to a stored result")
    }
}

impl Arena<ExampleResult> for ExampleResultsArena {
    type Key = ExampleResultRef;

    fn insert(&mut self, val: ExampleResult) -> Self::Key
    where
        ExampleResult: Hash + Eq,
    {
        self.results.insert(Some(val))
    }

    fn insert_placeholder(&mut self) -> Self::Key {
        self.results.insert(None)
    }

    fn replace(
        &mut self,
        key: Self::Key,
        val: ExampleResult,
    ) -> Result<Option<ExampleResult>, ReplaceError>
    where
        ExampleResult: Hash + Eq,
    {
        Ok(self
            .results
            .get_mut(key)
            .ok_or(ReplaceError::InvalidKey)?
            .replace(val))
    }

    fn get(&self, key: &Self::Key) -> Option<&ExampleResult> {
        self.results.get(*key)?.as_ref()
    }

    fn len(&self) -> usize {
        self.results.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResolvedExampleEdge {
    pub kind: ExampleEdgeKind,
    pub target: ExampleResultRef,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExampleDescription {
    Leaf {
        value: String,
    },
    MatchFirst {
        branch_count: usize,
    },
    Node {
        edge_count: usize,
        lazy_edge_count: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExampleOutput {
    Description(ExampleDescription),
    Delegate(ExampleResultRef),
    Leaf(String),
    Node(Vec<ResolvedExampleEdge>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExampleState {
    Describe,
    Resolve,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Error)]
pub enum ExampleRuleError {
    #[error("inductive cycle")]
    InductiveCycle,
    #[error("no matching branch")]
    NoMatchingBranch,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExampleRule;

impl ExampleRule {
    fn lazy_depth_mode(kind: ExampleEdgeKind) -> LazyDepthMode {
        match kind {
            ExampleEdgeKind::Eager => LazyDepthMode::Keep,
            ExampleEdgeKind::Lazy => LazyDepthMode::Increment,
        }
    }

    fn describe_edges(edges: &[ExampleEdge]) -> ExampleOutput {
        ExampleOutput::Description(ExampleDescription::Node {
            edge_count: edges.len(),
            lazy_edge_count: edges
                .iter()
                .filter(|edge| edge.kind == ExampleEdgeKind::Lazy)
                .count(),
        })
    }

    fn describe_match_first(branches: &[String]) -> ExampleOutput {
        ExampleOutput::Description(ExampleDescription::MatchFirst {
            branch_count: branches.len(),
        })
    }

    fn resolve_edge(
        edge: &ExampleEdge,
        ctx: &mut RuleContext<Self>,
    ) -> Result<ResolvedExampleEdge, RunError<Self>> {
        let child_env = Arc::new(ctx.env().descend(edge));

        match ctx.solve(
            edge.target.clone(),
            ExampleState::Resolve,
            Self::lazy_depth_mode(edge.kind),
            child_env,
        ) {
            Ok(SolveResult::Resolved { result, result_ref }) => match result {
                Ok(_) => Ok(ResolvedExampleEdge {
                    kind: edge.kind,
                    target: result_ref,
                }),
                Err(err) => Err(RunError::Rule(err.clone())),
            },
            Ok(SolveResult::Lazy { result_ref }) => Ok(ResolvedExampleEdge {
                kind: edge.kind,
                target: result_ref,
            }),
            Err(SolveError::SameDepthCycle) => {
                Err(RunError::Rule(ExampleRuleError::InductiveCycle))
            }
            Err(error) => Err(error.into()),
        }
    }

    fn resolve_node(
        edges: Vec<ExampleEdge>,
        ctx: &mut RuleContext<Self>,
    ) -> Result<ExampleOutput, RunError<Self>> {
        let mut resolved_edges = Vec::with_capacity(edges.len());

        for edge in &edges {
            resolved_edges.push(Self::resolve_edge(edge, ctx)?);
        }

        Ok(ExampleOutput::Node(resolved_edges))
    }

    fn resolve_match_first(
        branches: &[String],
        ctx: &mut RuleContext<Self>,
    ) -> Result<ExampleOutput, RunError<Self>> {
        for branch in branches {
            match ctx.solve(
                branch.clone(),
                ExampleState::Resolve,
                LazyDepthMode::Keep,
                Arc::new(ctx.env().clone()),
            ) {
                Ok(SolveResult::Resolved { result, result_ref }) => match result {
                    Ok(_) => return Ok(ExampleOutput::Delegate(result_ref)),
                    Err(_) => continue,
                },
                Ok(SolveResult::Lazy { result_ref }) => {
                    return Ok(ExampleOutput::Delegate(result_ref));
                }
                Err(SolveError::SameDepthCycle) => continue,
                Err(error) => return Err(error.into()),
            }
        }

        Err(RunError::Rule(ExampleRuleError::NoMatchingBranch))
    }
}

impl Rule for ExampleRule {
    type Query = String;
    type Output = ExampleOutput;
    type Err = ExampleRuleError;
    type Env = ExampleEnv;
    type ResultsArena = ExampleResultsArena;
    type RuleStateId = ExampleState;

    fn run(
        &self,
        query: String,
        ctx: &mut RuleContext<Self>,
    ) -> Result<Self::Output, RunError<Self>> {
        match ctx.state_id() {
            ExampleState::Describe => match ctx.lookup(&query) {
                ExampleSpec::Leaf(value) => {
                    Ok(ExampleOutput::Description(ExampleDescription::Leaf {
                        value,
                    }))
                }
                ExampleSpec::MatchFirst(branches) => Ok(Self::describe_match_first(&branches)),
                ExampleSpec::Node(edges) => Ok(Self::describe_edges(&edges)),
                ExampleSpec::DeferredSwitch { .. } => {
                    panic!("deferred switches must be resolved by example env lookup")
                }
            },
            ExampleState::Resolve => match ctx.lookup(&query) {
                ExampleSpec::Leaf(value) => Ok(ExampleOutput::Leaf(value)),
                ExampleSpec::MatchFirst(branches) => Self::resolve_match_first(&branches, ctx),
                ExampleSpec::Node(edges) => Self::resolve_node(edges, ctx),
                ExampleSpec::DeferredSwitch { .. } => {
                    panic!("deferred switches must be resolved by example env lookup")
                }
            },
        }
    }

    fn debug_query_label(
        &self,
        _query: &Self::Query,
        _state_id: Self::RuleStateId,
    ) -> Option<String> {
        None
    }

    fn debug_env_label(&self, _env: &Self::Env) -> Option<String> {
        None
    }

    fn debug_lookup_query_label(
        &self,
        _query: &<Self::Env as ResolutionEnv>::Query,
    ) -> Option<String> {
        None
    }

    fn debug_lookup_result_label(
        &self,
        _result: &<Self::Env as ResolutionEnv>::QueryResult,
    ) -> Option<String> {
        None
    }
}

#[derive(Debug, Clone)]
pub struct ExampleSystem {
    rule: ExampleRule,
    shared_state: ExampleSharedState,
    fixpoint_iteration_limit: usize,
    stack_depth_limit: usize,
}

impl ExampleSystem {
    pub fn new(definitions: impl IntoIterator<Item = ExampleDefinition>) -> Self {
        Self {
            rule: ExampleRule,
            shared_state: ExampleSharedState::new(definitions),
            fixpoint_iteration_limit: 32,
            stack_depth_limit: 512,
        }
    }

    pub fn with_fixpoint_iteration_limit(mut self, fixpoint_iteration_limit: usize) -> Self {
        self.fixpoint_iteration_limit = fixpoint_iteration_limit;
        self
    }

    pub fn with_stack_depth_limit(mut self, stack_overflow_depth: usize) -> Self {
        self.stack_depth_limit = stack_overflow_depth;
        self
    }

    pub fn solve(
        &self,
        root: impl Into<String>,
    ) -> Result<(ExampleResultsArena, ExampleResultRef), SolveError> {
        self.solve_in_state(root, ExampleState::Resolve)
    }

    pub fn solve_in_state(
        &self,
        root: impl Into<String>,
        state: ExampleState,
    ) -> Result<(ExampleResultsArena, ExampleResultRef), SolveError> {
        let outcome = solve(
            &self.rule,
            root.into(),
            state,
            self.shared_state.clone(),
            self.fixpoint_iteration_limit,
            self.stack_depth_limit,
        );

        outcome
            .result
            .map(|(result_ref, results_arena)| (results_arena, result_ref))
    }
}

pub fn definition(name: impl Into<String>, spec: ExampleSpec) -> ExampleDefinition {
    ExampleDefinition {
        name: name.into(),
        spec,
    }
}

pub fn leaf(value: impl Into<String>) -> ExampleSpec {
    ExampleSpec::Leaf(value.into())
}

pub fn node(edges: impl IntoIterator<Item = ExampleEdge>) -> ExampleSpec {
    ExampleSpec::Node(edges.into_iter().collect())
}

pub fn match_first<I, T>(branches: I) -> ExampleSpec
where
    I: IntoIterator<Item = T>,
    T: Into<String>,
{
    ExampleSpec::MatchFirst(branches.into_iter().map(Into::into).collect())
}

pub fn eager(target: impl Into<String>) -> ExampleEdge {
    ExampleEdge {
        kind: ExampleEdgeKind::Eager,
        target: target.into(),
        scope: None,
    }
}

pub fn lazy(target: impl Into<String>) -> ExampleEdge {
    ExampleEdge {
        kind: ExampleEdgeKind::Lazy,
        target: target.into(),
        scope: None,
    }
}

pub fn scoped_eager(scope: impl Into<String>, target: impl Into<String>) -> ExampleEdge {
    ExampleEdge {
        kind: ExampleEdgeKind::Eager,
        target: target.into(),
        scope: Some(scope.into()),
    }
}

pub fn scoped_lazy(scope: impl Into<String>, target: impl Into<String>) -> ExampleEdge {
    ExampleEdge {
        kind: ExampleEdgeKind::Lazy,
        target: target.into(),
        scope: Some(scope.into()),
    }
}

pub fn deferred_switch(immediate: ExampleSpec, deferred: ExampleSpec) -> ExampleSpec {
    ExampleSpec::DeferredSwitch {
        immediate: Box::new(immediate),
        deferred: Box::new(deferred),
    }
}
