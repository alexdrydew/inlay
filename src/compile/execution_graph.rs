use std::{
    collections::{BTreeMap, HashMap, HashSet},
    ops::{Index, IndexMut},
    sync::Arc,
};

use context_solver::Arena as ResultsArena;
use inlay_instrument::{inlay_event, instrumented};

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    python_identity::PythonIdentity,
    registry::{Source, SourceKind},
    rules::{
        ResolutionError, SolverResolutionArena, SolverResolutionNode, SolverResolutionRef,
        SolverResolvedNode, SolverResolvedTransition, SolverResolvedTransitionImplementation,
        SolverTransitionImplementationCallable, SolverWritableDependency, TransitionParam,
    },
    types::{
        MemberAccessKind, ParamKind, PyType, PyTypeConcreteKey, SentinelTypeKind, TypeArenas,
        WrapperKind,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ExecutionNodeId(u32);

impl ExecutionNodeId {
    pub(crate) fn from_index(index: usize) -> Self {
        Self(
            index
                .try_into()
                .expect("execution graph cannot exceed u32::MAX entries"),
        )
    }

    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ExecutionSourceNodeId(pub(crate) ExecutionNodeId);

impl ExecutionSourceNodeId {
    pub(crate) fn node_id(self) -> ExecutionNodeId {
        self.0
    }
}

#[derive(Default)]
struct SourceNodeInterner<'ty> {
    sources: HashMap<Source<'ty>, ExecutionSourceNodeId>,
}

impl<'ty> SourceNodeInterner<'ty> {
    fn intern(
        &mut self,
        source: &Source<'ty>,
        graph: &mut BuildExecutionGraph,
    ) -> ExecutionSourceNodeId {
        if let Some(source_node_id) = self.sources.get(source) {
            return *source_node_id;
        }

        let node = match &source.kind {
            SourceKind::ProviderResult(value) => computed_node(
                false,
                ExecutionComputedKind::StaticValue {
                    value: Arc::clone(value),
                },
            ),
            SourceKind::Transition { .. } => ExecutionNode::Variable,
        };
        let node_id = graph.insert(BuildExecutionEntry::ready(node));
        let source_node_id = ExecutionSourceNodeId(node_id);
        self.sources.insert(source.clone(), source_node_id);
        source_node_id
    }
}

#[derive(Clone)]
pub(crate) struct ConstructorParam {
    pub(crate) name: Arc<str>,
    pub(crate) kind: ParamKind,
    pub(crate) node: ExecutionNodeId,
}

#[derive(Clone)]
pub(crate) struct ExecutionParam {
    pub(crate) name: Arc<str>,
    pub(crate) kind: ParamKind,
    pub(crate) sources: Vec<ExecutionSourceNodeId>,
}

impl ExecutionParam {
    fn from_transition_param<'ty>(
        param: &TransitionParam<'ty>,
        source_interner: &mut SourceNodeInterner<'ty>,
        graph: &mut BuildExecutionGraph,
    ) -> Self {
        let sources = param
            .logical_sources
            .iter()
            .map(|source| source_interner.intern(source, graph))
            .collect();
        Self {
            name: Arc::clone(&param.name),
            kind: param.kind,
            sources,
        }
    }
}

#[derive(Clone)]
pub(crate) enum ExecutionTransitionImplementationCallable {
    Static(Arc<Py<PyAny>>),
    Source(ExecutionSourceNodeId),
}

#[derive(Clone)]
pub(crate) struct ExecutionTransitionImplementation {
    pub(crate) implementation: ExecutionTransitionImplementationCallable,
    pub(crate) bound_to: Option<ExecutionNodeId>,
    pub(crate) params: Vec<ConstructorParam>,
    pub(crate) return_wrapper: WrapperKind,
    pub(crate) result_source: Option<ExecutionSourceNodeId>,
}

#[derive(Clone)]
pub(crate) struct RuntimeCallableMatchParam {
    pub(crate) name: Arc<str>,
    pub(crate) kind: ParamKind,
    pub(crate) has_default: bool,
}

#[derive(Clone)]
pub(crate) enum RuntimeTypeMatcher {
    None,
    Class {
        origin: Arc<Py<PyAny>>,
        display_name: Arc<str>,
    },
    Callable {
        params: Vec<RuntimeCallableMatchParam>,
    },
}

#[derive(Clone)]
pub(crate) struct ExecutionRuntimeUnionBranch {
    pub(crate) matcher: RuntimeTypeMatcher,
    pub(crate) target: ExecutionNodeId,
    pub(crate) arm_source: ExecutionSourceNodeId,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct MemberSignature {
    name: Arc<str>,
    node: usize,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct ConstructorParamSignature {
    name: Arc<str>,
    kind: ParamKind,
    node: usize,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct ExecutionParamSignature {
    name: Arc<str>,
    kind: ParamKind,
    sources: Vec<usize>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum TransitionImplementationCallableSignature {
    Static(PythonIdentity),
    Source(usize),
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct TransitionImplementationSignature {
    implementation: TransitionImplementationCallableSignature,
    bound_to: Option<usize>,
    params: Vec<ConstructorParamSignature>,
    return_wrapper: WrapperKind,
    result_source: Option<usize>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum RuntimeTypeMatcherSignature {
    None,
    Class(PythonIdentity),
    Callable(Vec<(Arc<str>, ParamKind, bool)>),
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct RuntimeUnionBranchSignature {
    matcher: RuntimeTypeMatcherSignature,
    target: usize,
    arm_source: usize,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum ExecutionSignature {
    Variable {
        node_identity: usize,
    },
    Field {
        source: usize,
        name: Arc<str>,
        access_kind: MemberAccessKind,
    },
    Computed {
        dynamic: bool,
        cache: ExecutionCachePolicy,
        writable_dependencies: Vec<usize>,
        kind: ExecutionComputedKindSignature,
    },
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum ExecutionComputedKindSignature {
    Property {
        source: usize,
        property_name: Arc<str>,
    },
    ReadCell {
        target: usize,
    },
    Cell {
        target: usize,
    },
    None,
    StaticValue {
        value: PythonIdentity,
    },
    Protocol {
        members: Vec<MemberSignature>,
    },
    TypedDict {
        members: Vec<MemberSignature>,
    },
    Transition {
        return_wrapper: WrapperKind,
        accepts_varargs: bool,
        accepts_varkw: bool,
        params: Vec<ExecutionParamSignature>,
        implementations: Vec<TransitionImplementationSignature>,
        target: usize,
    },
    RuntimeUnionDispatch {
        source: usize,
        branches: Vec<RuntimeUnionBranchSignature>,
    },
    Constructor {
        implementation: PythonIdentity,
        params: Vec<ConstructorParamSignature>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExecutionCachePolicy {
    Never,
    Cached,
}

#[derive(Clone)]
pub(crate) struct ExecutionField {
    pub(crate) source: ExecutionNodeId,
    pub(crate) name: Arc<str>,
    pub(crate) access_kind: MemberAccessKind,
}

#[derive(Clone)]
pub(crate) struct ExecutionComputed {
    pub(crate) dynamic: bool,
    pub(crate) cache: ExecutionCachePolicy,
    pub(crate) writable_dependencies: Vec<ExecutionNodeId>,
    pub(crate) kind: ExecutionComputedKind,
}

#[derive(Clone)]
pub(crate) enum ExecutionComputedKind {
    Property {
        source: ExecutionNodeId,
        property_name: Arc<str>,
    },
    ReadCell {
        target: ExecutionNodeId,
    },
    Cell {
        target: ExecutionNodeId,
    },
    None,
    StaticValue {
        value: Arc<Py<PyAny>>,
    },
    Protocol {
        members: BTreeMap<Arc<str>, ExecutionNodeId>,
    },
    TypedDict {
        members: BTreeMap<Arc<str>, ExecutionNodeId>,
    },
    Transition {
        return_wrapper: WrapperKind,
        accepts_varargs: bool,
        accepts_varkw: bool,
        params: Vec<ExecutionParam>,
        implementations: Vec<ExecutionTransitionImplementation>,
        target: ExecutionNodeId,
    },
    RuntimeUnionDispatch {
        source: ExecutionSourceNodeId,
        branches: Vec<ExecutionRuntimeUnionBranch>,
    },
    Constructor {
        implementation: Arc<Py<PyAny>>,
        params: Vec<ConstructorParam>,
    },
}

#[derive(Clone)]
pub(crate) enum ExecutionNode {
    Variable,
    Field(ExecutionField),
    Computed(ExecutionComputed),
}

enum BuildExecutionNode {
    Pending,
    Ready(ExecutionNode),
}

struct BuildExecutionEntry {
    node: BuildExecutionNode,
}

impl BuildExecutionEntry {
    fn pending() -> Self {
        Self {
            node: BuildExecutionNode::Pending,
        }
    }

    fn ready(node: ExecutionNode) -> Self {
        Self {
            node: BuildExecutionNode::Ready(node),
        }
    }

    fn ready_node(&self) -> &ExecutionNode {
        match &self.node {
            BuildExecutionNode::Ready(node) => node,
            BuildExecutionNode::Pending => {
                unreachable!("pending execution node must be resolved before canonicalization")
            }
        }
    }
}

pub(crate) struct ExecutionEntry {
    pub(crate) node: ExecutionNode,
    pub(crate) resource_deps: HashSet<ExecutionSourceNodeId>,
}

impl std::fmt::Debug for ExecutionEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionEntry")
            .field("resource_deps", &self.resource_deps.len())
            .field("cached", &execution_node_cached(&self.node))
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct WritableExclusionId(usize);

#[derive(Clone, Copy)]
struct WritableExclusion {
    node_id: ExecutionNodeId,
    parent: Option<WritableExclusionId>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct WritableDependant {
    node_id: ExecutionNodeId,
    exclusions: Option<WritableExclusionId>,
}

#[derive(Default)]
struct BuildExecutionGraph {
    entries: Vec<BuildExecutionEntry>,
}

impl BuildExecutionGraph {
    fn insert(&mut self, entry: BuildExecutionEntry) -> ExecutionNodeId {
        let key = ExecutionNodeId::from_index(self.entries.len());
        self.entries.push(entry);
        key
    }

    fn keys(&self) -> impl Iterator<Item = ExecutionNodeId> + '_ {
        (0..self.entries.len()).map(ExecutionNodeId::from_index)
    }
}

impl Index<ExecutionNodeId> for BuildExecutionGraph {
    type Output = BuildExecutionEntry;

    fn index(&self, index: ExecutionNodeId) -> &Self::Output {
        &self.entries[index.index()]
    }
}

impl IndexMut<ExecutionNodeId> for BuildExecutionGraph {
    fn index_mut(&mut self, index: ExecutionNodeId) -> &mut Self::Output {
        &mut self.entries[index.index()]
    }
}

#[derive(Default)]
pub(crate) struct ExecutionGraph {
    entries: Vec<ExecutionEntry>,
    writable_dependants: Vec<Vec<WritableDependant>>,
    writable_exclusions: Vec<WritableExclusion>,
}

impl ExecutionGraph {
    fn from_entries(entries: Vec<ExecutionEntry>) -> Self {
        let writable_dependants = vec![Vec::new(); entries.len()];
        Self {
            entries,
            writable_dependants,
            writable_exclusions: Vec::new(),
        }
    }

    fn rebuild_dependencies(&mut self) {
        (self.writable_dependants, self.writable_exclusions) = compute_writable_dependants(self);
        let resource_deps = compute_resource_deps(self);
        for node_id in self.keys().collect::<Vec<_>>() {
            self[node_id].resource_deps = resource_deps[&node_id].clone();
        }
    }

    pub(crate) fn affected_dependants(
        &self,
        writable_node: ExecutionNodeId,
    ) -> HashSet<ExecutionNodeId> {
        let mut visited = HashSet::from([writable_node]);
        let mut pending = vec![writable_node];
        let mut affected = HashSet::new();
        while let Some(node_id) = pending.pop() {
            for dependant in &self.writable_dependants[node_id.index()] {
                if self.writable_dependency_excludes(dependant.exclusions, writable_node) {
                    continue;
                }
                if visited.insert(dependant.node_id) {
                    affected.insert(dependant.node_id);
                    pending.push(dependant.node_id);
                }
            }
        }
        affected
    }

    fn writable_dependency_excludes(
        &self,
        mut exclusions: Option<WritableExclusionId>,
        writable_node: ExecutionNodeId,
    ) -> bool {
        while let Some(exclusion_id) = exclusions {
            let exclusion = self.writable_exclusions[exclusion_id.0];
            if exclusion.node_id == writable_node {
                return true;
            }
            exclusions = exclusion.parent;
        }
        false
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn keys(&self) -> impl Iterator<Item = ExecutionNodeId> + '_ {
        (0..self.entries.len()).map(ExecutionNodeId::from_index)
    }
}

impl Index<ExecutionNodeId> for ExecutionGraph {
    type Output = ExecutionEntry;

    fn index(&self, index: ExecutionNodeId) -> &Self::Output {
        &self.entries[index.index()]
    }
}

impl IndexMut<ExecutionNodeId> for ExecutionGraph {
    fn index_mut(&mut self, index: ExecutionNodeId) -> &mut Self::Output {
        &mut self.entries[index.index()]
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct ExecutionGraphState {
    nodes: Vec<ExecutionNodeState>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ExecutionNodeState {
    Variable,
    Field {
        source: usize,
        name: String,
        access_kind: MemberAccessKind,
    },
    Computed {
        dynamic: bool,
        cache: ExecutionCachePolicy,
        writable_dependencies: Vec<usize>,
        computed: ExecutionComputedKindState,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ExecutionComputedKindState {
    #[serde(rename = "none")]
    NoneValue,
    StaticValue {
        value_ref: usize,
    },
    Property {
        source: usize,
        property_name: String,
    },
    ReadCell {
        target: usize,
    },
    Cell {
        target: usize,
    },
    Protocol {
        members: Vec<MemberState>,
    },
    TypedDict {
        members: Vec<MemberState>,
    },
    Transition {
        return_wrapper: WrapperKind,
        accepts_varargs: bool,
        accepts_varkw: bool,
        params: Vec<ExecutionParamState>,
        implementations: Vec<ExecutionTransitionImplementationState>,
        target: usize,
    },
    RuntimeUnionDispatch {
        source: usize,
        branches: Vec<ExecutionRuntimeUnionBranchState>,
    },
    Constructor {
        implementation_ref: usize,
        params: Vec<ConstructorParamState>,
    },
}

#[derive(Serialize, Deserialize)]
struct MemberState {
    name: String,
    node: usize,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct ConstructorParamState {
    name: String,
    kind: ParamKind,
    node: usize,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct ExecutionParamState {
    name: String,
    kind: ParamKind,
    sources: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct ExecutionTransitionImplementationState {
    implementation: ExecutionTransitionImplementationCallableState,
    bound_to: Option<usize>,
    params: Vec<ConstructorParamState>,
    return_wrapper: WrapperKind,
    result_source: Option<usize>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum ExecutionTransitionImplementationCallableState {
    Static { implementation_ref: usize },
    Source { source: usize },
}

#[derive(Serialize, Deserialize)]
struct ExecutionRuntimeUnionBranchState {
    matcher: RuntimeTypeMatcherState,
    target: usize,
    arm_source: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum RuntimeTypeMatcherState {
    #[serde(rename = "none")]
    NoneValue,
    Class {
        origin_ref: usize,
        display_name: String,
    },
    Callable {
        params: Vec<RuntimeCallableMatchParamState>,
    },
}

#[derive(Serialize, Deserialize)]
struct RuntimeCallableMatchParamState {
    name: String,
    kind: ParamKind,
    has_default: bool,
}

impl ExecutionGraph {
    pub(crate) fn to_state(
        &self,
        py: Python<'_>,
        refs: &mut crate::pickle::PyRefCollector,
    ) -> ExecutionGraphState {
        ExecutionGraphState {
            nodes: self
                .entries
                .iter()
                .map(|entry| execution_node_to_state(py, &entry.node, refs))
                .collect(),
        }
    }

    pub(crate) fn from_state(
        state: ExecutionGraphState,
        refs: &crate::pickle::PyRefResolver<'_>,
    ) -> PyResult<Self> {
        let entries = state
            .nodes
            .iter()
            .map(|node| {
                Ok(ExecutionEntry {
                    node: execution_node_from_state(node, refs)?,
                    resource_deps: HashSet::new(),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;

        let mut graph = ExecutionGraph::from_entries(entries);
        graph.rebuild_dependencies();
        Ok(graph)
    }
}

pub(crate) fn execution_params_to_state(params: &[ExecutionParam]) -> Vec<ExecutionParamState> {
    params
        .iter()
        .map(|param| ExecutionParamState {
            name: param.name.to_string(),
            kind: param.kind,
            sources: param
                .sources
                .iter()
                .map(|source| source.node_id().index())
                .collect(),
        })
        .collect()
}

pub(crate) fn execution_params_from_state(params: &[ExecutionParamState]) -> Vec<ExecutionParam> {
    params
        .iter()
        .map(|param| ExecutionParam {
            name: Arc::from(param.name.as_str()),
            kind: param.kind,
            sources: param
                .sources
                .iter()
                .map(|&source| ExecutionSourceNodeId(ExecutionNodeId::from_index(source)))
                .collect(),
        })
        .collect()
}

pub(crate) fn transition_implementations_to_state(
    py: Python<'_>,
    implementations: &[ExecutionTransitionImplementation],
    refs: &mut crate::pickle::PyRefCollector,
) -> Vec<ExecutionTransitionImplementationState> {
    implementations
        .iter()
        .map(|implementation| ExecutionTransitionImplementationState {
            implementation: transition_callable_to_state(py, &implementation.implementation, refs),
            bound_to: implementation.bound_to.map(ExecutionNodeId::index),
            params: constructor_params_to_state(&implementation.params),
            return_wrapper: implementation.return_wrapper,
            result_source: implementation
                .result_source
                .map(|source| source.node_id().index()),
        })
        .collect()
}

pub(crate) fn transition_implementations_from_state(
    implementations: &[ExecutionTransitionImplementationState],
    refs: &crate::pickle::PyRefResolver<'_>,
) -> PyResult<Vec<ExecutionTransitionImplementation>> {
    implementations
        .iter()
        .map(|implementation| {
            Ok(ExecutionTransitionImplementation {
                implementation: transition_callable_from_state(
                    &implementation.implementation,
                    refs,
                )?,
                bound_to: implementation.bound_to.map(ExecutionNodeId::from_index),
                params: constructor_params_from_state(&implementation.params),
                return_wrapper: implementation.return_wrapper,
                result_source: implementation
                    .result_source
                    .map(|source| ExecutionSourceNodeId(ExecutionNodeId::from_index(source))),
            })
        })
        .collect()
}

fn execution_node_to_state(
    py: Python<'_>,
    node: &ExecutionNode,
    refs: &mut crate::pickle::PyRefCollector,
) -> ExecutionNodeState {
    match node {
        ExecutionNode::Variable => ExecutionNodeState::Variable,
        ExecutionNode::Field(field) => ExecutionNodeState::Field {
            source: field.source.index(),
            name: field.name.to_string(),
            access_kind: field.access_kind,
        },
        ExecutionNode::Computed(computed) => ExecutionNodeState::Computed {
            dynamic: computed.dynamic,
            cache: computed.cache,
            writable_dependencies: computed
                .writable_dependencies
                .iter()
                .map(|node_id| node_id.index())
                .collect(),
            computed: computed_kind_to_state(py, &computed.kind, refs),
        },
    }
}

fn computed_kind_to_state(
    py: Python<'_>,
    kind: &ExecutionComputedKind,
    refs: &mut crate::pickle::PyRefCollector,
) -> ExecutionComputedKindState {
    match kind {
        ExecutionComputedKind::None => ExecutionComputedKindState::NoneValue,
        ExecutionComputedKind::StaticValue { value } => ExecutionComputedKindState::StaticValue {
            value_ref: refs.push(py, value.as_ref()),
        },
        ExecutionComputedKind::Property {
            source,
            property_name,
        } => ExecutionComputedKindState::Property {
            source: source.index(),
            property_name: property_name.to_string(),
        },
        ExecutionComputedKind::ReadCell { target } => ExecutionComputedKindState::ReadCell {
            target: target.index(),
        },
        ExecutionComputedKind::Cell { target } => ExecutionComputedKindState::Cell {
            target: target.index(),
        },
        ExecutionComputedKind::Protocol { members } => ExecutionComputedKindState::Protocol {
            members: members_to_state(members),
        },
        ExecutionComputedKind::TypedDict { members } => ExecutionComputedKindState::TypedDict {
            members: members_to_state(members),
        },
        ExecutionComputedKind::Transition {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            implementations,
            target,
        } => ExecutionComputedKindState::Transition {
            return_wrapper: *return_wrapper,
            accepts_varargs: *accepts_varargs,
            accepts_varkw: *accepts_varkw,
            params: execution_params_to_state(params),
            implementations: transition_implementations_to_state(py, implementations, refs),
            target: target.index(),
        },
        ExecutionComputedKind::RuntimeUnionDispatch { source, branches } => {
            ExecutionComputedKindState::RuntimeUnionDispatch {
                source: source.node_id().index(),
                branches: branches
                    .iter()
                    .map(|branch| ExecutionRuntimeUnionBranchState {
                        matcher: runtime_type_matcher_to_state(py, &branch.matcher, refs),
                        target: branch.target.index(),
                        arm_source: branch.arm_source.node_id().index(),
                    })
                    .collect(),
            }
        }
        ExecutionComputedKind::Constructor {
            implementation,
            params,
        } => ExecutionComputedKindState::Constructor {
            implementation_ref: refs.push(py, implementation.as_ref()),
            params: constructor_params_to_state(params),
        },
    }
}

fn execution_node_from_state(
    state: &ExecutionNodeState,
    refs: &crate::pickle::PyRefResolver<'_>,
) -> PyResult<ExecutionNode> {
    match state {
        ExecutionNodeState::Variable => Ok(ExecutionNode::Variable),
        ExecutionNodeState::Field {
            source,
            name,
            access_kind,
        } => Ok(ExecutionNode::Field(ExecutionField {
            source: ExecutionNodeId::from_index(*source),
            name: Arc::from(name.as_str()),
            access_kind: *access_kind,
        })),
        ExecutionNodeState::Computed {
            dynamic,
            cache,
            writable_dependencies,
            computed,
        } => {
            if *cache != cache_policy(*dynamic) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "computed cache policy does not match dynamicness",
                ));
            }
            Ok(computed_node_with_dependencies(
                *dynamic,
                writable_dependencies
                    .iter()
                    .map(|&node_id| ExecutionNodeId::from_index(node_id))
                    .collect(),
                computed_kind_from_state(computed, refs)?,
            ))
        }
    }
}

fn computed_kind_from_state(
    state: &ExecutionComputedKindState,
    refs: &crate::pickle::PyRefResolver<'_>,
) -> PyResult<ExecutionComputedKind> {
    match state {
        ExecutionComputedKindState::NoneValue => Ok(ExecutionComputedKind::None),
        ExecutionComputedKindState::StaticValue { value_ref } => {
            Ok(ExecutionComputedKind::StaticValue {
                value: Arc::new(refs.get(*value_ref)?),
            })
        }
        ExecutionComputedKindState::Property {
            source,
            property_name,
        } => Ok(ExecutionComputedKind::Property {
            source: ExecutionNodeId::from_index(*source),
            property_name: Arc::from(property_name.as_str()),
        }),
        ExecutionComputedKindState::ReadCell { target } => Ok(ExecutionComputedKind::ReadCell {
            target: ExecutionNodeId::from_index(*target),
        }),
        ExecutionComputedKindState::Cell { target } => Ok(ExecutionComputedKind::Cell {
            target: ExecutionNodeId::from_index(*target),
        }),
        ExecutionComputedKindState::Protocol { members } => Ok(ExecutionComputedKind::Protocol {
            members: members_from_state(members),
        }),
        ExecutionComputedKindState::TypedDict { members } => Ok(ExecutionComputedKind::TypedDict {
            members: members_from_state(members),
        }),
        ExecutionComputedKindState::Transition {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            implementations,
            target,
        } => Ok(ExecutionComputedKind::Transition {
            return_wrapper: *return_wrapper,
            accepts_varargs: *accepts_varargs,
            accepts_varkw: *accepts_varkw,
            params: execution_params_from_state(params),
            implementations: transition_implementations_from_state(implementations, refs)?,
            target: ExecutionNodeId::from_index(*target),
        }),
        ExecutionComputedKindState::RuntimeUnionDispatch { source, branches } => {
            Ok(ExecutionComputedKind::RuntimeUnionDispatch {
                source: ExecutionSourceNodeId(ExecutionNodeId::from_index(*source)),
                branches: branches
                    .iter()
                    .map(|branch| {
                        Ok(ExecutionRuntimeUnionBranch {
                            matcher: runtime_type_matcher_from_state(&branch.matcher, refs)?,
                            target: ExecutionNodeId::from_index(branch.target),
                            arm_source: ExecutionSourceNodeId(ExecutionNodeId::from_index(
                                branch.arm_source,
                            )),
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            })
        }
        ExecutionComputedKindState::Constructor {
            implementation_ref,
            params,
        } => Ok(ExecutionComputedKind::Constructor {
            implementation: Arc::new(refs.get(*implementation_ref)?),
            params: constructor_params_from_state(params),
        }),
    }
}

fn members_to_state(members: &BTreeMap<Arc<str>, ExecutionNodeId>) -> Vec<MemberState> {
    members
        .iter()
        .map(|(name, node)| MemberState {
            name: name.to_string(),
            node: node.index(),
        })
        .collect()
}

fn members_from_state(members: &[MemberState]) -> BTreeMap<Arc<str>, ExecutionNodeId> {
    members
        .iter()
        .map(|member| {
            (
                Arc::from(member.name.as_str()),
                ExecutionNodeId::from_index(member.node),
            )
        })
        .collect()
}

fn constructor_params_to_state(params: &[ConstructorParam]) -> Vec<ConstructorParamState> {
    params
        .iter()
        .map(|param| ConstructorParamState {
            name: param.name.to_string(),
            kind: param.kind,
            node: param.node.index(),
        })
        .collect()
}

fn constructor_params_from_state(params: &[ConstructorParamState]) -> Vec<ConstructorParam> {
    params
        .iter()
        .map(|param| ConstructorParam {
            name: Arc::from(param.name.as_str()),
            kind: param.kind,
            node: ExecutionNodeId::from_index(param.node),
        })
        .collect()
}

fn transition_callable_to_state(
    py: Python<'_>,
    implementation: &ExecutionTransitionImplementationCallable,
    refs: &mut crate::pickle::PyRefCollector,
) -> ExecutionTransitionImplementationCallableState {
    match implementation {
        ExecutionTransitionImplementationCallable::Static(implementation) => {
            ExecutionTransitionImplementationCallableState::Static {
                implementation_ref: refs.push(py, implementation.as_ref()),
            }
        }
        ExecutionTransitionImplementationCallable::Source(source) => {
            ExecutionTransitionImplementationCallableState::Source {
                source: source.node_id().index(),
            }
        }
    }
}

fn transition_callable_from_state(
    state: &ExecutionTransitionImplementationCallableState,
    refs: &crate::pickle::PyRefResolver<'_>,
) -> PyResult<ExecutionTransitionImplementationCallable> {
    match state {
        ExecutionTransitionImplementationCallableState::Static { implementation_ref } => {
            Ok(ExecutionTransitionImplementationCallable::Static(Arc::new(
                refs.get(*implementation_ref)?,
            )))
        }
        ExecutionTransitionImplementationCallableState::Source { source } => {
            Ok(ExecutionTransitionImplementationCallable::Source(
                ExecutionSourceNodeId(ExecutionNodeId::from_index(*source)),
            ))
        }
    }
}

fn runtime_type_matcher_to_state(
    py: Python<'_>,
    matcher: &RuntimeTypeMatcher,
    refs: &mut crate::pickle::PyRefCollector,
) -> RuntimeTypeMatcherState {
    match matcher {
        RuntimeTypeMatcher::None => RuntimeTypeMatcherState::NoneValue,
        RuntimeTypeMatcher::Class {
            origin,
            display_name,
        } => RuntimeTypeMatcherState::Class {
            origin_ref: refs.push(py, origin.as_ref()),
            display_name: display_name.to_string(),
        },
        RuntimeTypeMatcher::Callable { params } => RuntimeTypeMatcherState::Callable {
            params: params
                .iter()
                .map(|param| RuntimeCallableMatchParamState {
                    name: param.name.to_string(),
                    kind: param.kind,
                    has_default: param.has_default,
                })
                .collect(),
        },
    }
}

fn runtime_type_matcher_from_state(
    state: &RuntimeTypeMatcherState,
    refs: &crate::pickle::PyRefResolver<'_>,
) -> PyResult<RuntimeTypeMatcher> {
    match state {
        RuntimeTypeMatcherState::NoneValue => Ok(RuntimeTypeMatcher::None),
        RuntimeTypeMatcherState::Class {
            origin_ref,
            display_name,
        } => Ok(RuntimeTypeMatcher::Class {
            origin: Arc::new(refs.get(*origin_ref)?),
            display_name: Arc::from(display_name.as_str()),
        }),
        RuntimeTypeMatcherState::Callable { params } => Ok(RuntimeTypeMatcher::Callable {
            params: params
                .iter()
                .map(|param| RuntimeCallableMatchParam {
                    name: Arc::from(param.name.as_str()),
                    kind: param.kind,
                    has_default: param.has_default,
                })
                .collect(),
        }),
    }
}

#[instrumented(
    name = "inlay.create_execution_graph",
    target = "inlay",
    level = "trace",
    skip(results, types)
)]
pub(crate) fn create_execution_graph<'ty>(
    results: &SolverResolutionArena<'ty>,
    root: SolverResolutionRef,
    types: &TypeArenas<'ty>,
) -> Result<(ExecutionGraph, ExecutionNodeId), ResolutionError<'ty>> {
    let mut graph = BuildExecutionGraph::default();
    let mut refs = HashMap::new();
    let mut source_interner = SourceNodeInterner::default();
    let root = resolve_ref(
        results,
        root,
        types,
        &mut graph,
        &mut refs,
        &mut source_interner,
    )?;
    apply_solver_writable_dependencies(
        results,
        types,
        &mut graph,
        &mut refs,
        &mut source_interner,
    )?;
    inlay_event!(
        name: "inlay.create_execution_graph.reachable_result_refs",
        reachable_result_refs = refs.len() as u64,
    );
    let (graph, root) = canonicalize_execution_graph(graph, root);
    Ok((graph, root))
}

fn apply_solver_writable_dependencies<'ty>(
    results: &SolverResolutionArena<'ty>,
    types: &TypeArenas<'ty>,
    graph: &mut BuildExecutionGraph,
    refs: &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
    source_interner: &mut SourceNodeInterner<'ty>,
) -> Result<(), ResolutionError<'ty>> {
    let mut processed = HashSet::new();
    loop {
        let Some(result_ref) = refs
            .keys()
            .find(|&&result_ref| !processed.contains(&result_ref))
            .copied()
        else {
            return Ok(());
        };
        processed.insert(result_ref);
        let node_id = refs[&result_ref];
        let dependencies = get_resolved_node(results, result_ref)?
            .writable_dependencies
            .clone();
        let mut node_dependencies = Vec::with_capacity(dependencies.len());
        for dependency in dependencies {
            let dependency = match dependency {
                SolverWritableDependency::Result(result_ref) => {
                    resolve_ref(results, result_ref, types, graph, refs, source_interner)?
                }
                SolverWritableDependency::Source(source) => {
                    source_interner.intern(&source, graph).node_id()
                }
            };
            node_dependencies.push(dependency);
        }
        if let BuildExecutionNode::Ready(ExecutionNode::Computed(computed)) =
            &mut graph[node_id].node
        {
            computed.writable_dependencies.extend(node_dependencies);
            computed.writable_dependencies =
                normalize_node_ids(std::mem::take(&mut computed.writable_dependencies));
        }
    }
}

fn cache_policy(dynamic: bool) -> ExecutionCachePolicy {
    if dynamic {
        ExecutionCachePolicy::Never
    } else {
        ExecutionCachePolicy::Cached
    }
}

fn computed_node(dynamic: bool, kind: ExecutionComputedKind) -> ExecutionNode {
    computed_node_with_dependencies(dynamic, Vec::new(), kind)
}

fn computed_node_with_dependencies(
    dynamic: bool,
    mut writable_dependencies: Vec<ExecutionNodeId>,
    kind: ExecutionComputedKind,
) -> ExecutionNode {
    writable_dependencies.sort_unstable();
    writable_dependencies.dedup();
    ExecutionNode::Computed(ExecutionComputed {
        dynamic,
        cache: cache_policy(dynamic),
        writable_dependencies,
        kind,
    })
}

fn normalize_node_ids(mut node_ids: Vec<ExecutionNodeId>) -> Vec<ExecutionNodeId> {
    node_ids.sort_unstable();
    node_ids.dedup();
    node_ids
}

fn execution_node_cached(node: &ExecutionNode) -> bool {
    matches!(
        node,
        ExecutionNode::Computed(ExecutionComputed {
            cache: ExecutionCachePolicy::Cached,
            ..
        })
    )
}

fn resolve_ref<'ty>(
    results: &SolverResolutionArena<'ty>,
    node_ref: SolverResolutionRef,
    types: &TypeArenas<'ty>,
    graph: &mut BuildExecutionGraph,
    refs: &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
    source_interner: &mut SourceNodeInterner<'ty>,
) -> Result<ExecutionNodeId, ResolutionError<'ty>> {
    if let Some(&node_id) = refs.get(&node_ref) {
        return Ok(node_id);
    }

    let resolved = get_resolved_node(results, node_ref)?;
    match &resolved.resolution {
        SolverResolutionNode::Delegate(target) | SolverResolutionNode::UnionVariant { target } => {
            let node_id = resolve_ref(results, *target, types, graph, refs, source_interner)?;
            refs.insert(node_ref, node_id);
            Ok(node_id)
        }
        SolverResolutionNode::None => {
            materialize_node(node_ref, graph, refs, source_interner, |_, _, _| {
                Ok(computed_node(resolved.dynamic, ExecutionComputedKind::None))
            })
        }
        SolverResolutionNode::Constant { source } => {
            let source_node_id = source_interner.intern(source, graph);
            refs.insert(node_ref, source_node_id.node_id());
            Ok(source_node_id.node_id())
        }
        SolverResolutionNode::Property {
            source,
            property_name,
        } => materialize_node(
            node_ref,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner| {
                Ok(computed_node(
                    resolved.dynamic,
                    ExecutionComputedKind::Property {
                        source: resolve_ref(results, *source, types, graph, refs, source_interner)?,
                        property_name: property_name.clone(),
                    },
                ))
            },
        ),
        SolverResolutionNode::ReadCell { target } => materialize_node(
            node_ref,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner| {
                Ok(computed_node(
                    resolved.dynamic,
                    ExecutionComputedKind::ReadCell {
                        target: resolve_ref(results, *target, types, graph, refs, source_interner)?,
                    },
                ))
            },
        ),
        SolverResolutionNode::Cell { target } => materialize_node(
            node_ref,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner| {
                Ok(computed_node(
                    resolved.dynamic,
                    ExecutionComputedKind::Cell {
                        target: resolve_ref(results, *target, types, graph, refs, source_interner)?,
                    },
                ))
            },
        ),
        SolverResolutionNode::Protocol { members } => materialize_node(
            node_ref,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner| {
                Ok(computed_node(
                    resolved.dynamic,
                    ExecutionComputedKind::Protocol {
                        members: members
                            .iter()
                            .map(|(name, &member_ref)| {
                                resolve_ref(
                                    results,
                                    member_ref,
                                    types,
                                    graph,
                                    refs,
                                    source_interner,
                                )
                                .map(|node_id| (name.clone(), node_id))
                            })
                            .collect::<Result<_, _>>()?,
                    },
                ))
            },
        ),
        SolverResolutionNode::TypedDict { members } => materialize_node(
            node_ref,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner| {
                Ok(computed_node(
                    resolved.dynamic,
                    ExecutionComputedKind::TypedDict {
                        members: members
                            .iter()
                            .map(|(name, &member_ref)| {
                                resolve_ref(
                                    results,
                                    member_ref,
                                    types,
                                    graph,
                                    refs,
                                    source_interner,
                                )
                                .map(|node_id| (name.clone(), node_id))
                            })
                            .collect::<Result<_, _>>()?,
                    },
                ))
            },
        ),
        SolverResolutionNode::Transition(transition) => materialize_node(
            node_ref,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner| {
                build_transition_node(
                    resolved.dynamic,
                    transition,
                    results,
                    types,
                    graph,
                    refs,
                    source_interner,
                )
            },
        ),
        SolverResolutionNode::RuntimeUnionDispatch { .. } => materialize_node(
            node_ref,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner| {
                build_runtime_union_dispatch_node(
                    resolved,
                    results,
                    types,
                    graph,
                    refs,
                    source_interner,
                )
            },
        ),
        SolverResolutionNode::Attribute {
            source,
            attribute_name,
            access_kind,
        } => materialize_node(
            node_ref,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner| {
                Ok(ExecutionNode::Field(ExecutionField {
                    source: resolve_ref(results, *source, types, graph, refs, source_interner)?,
                    name: attribute_name.clone(),
                    access_kind: *access_kind,
                }))
            },
        ),
        SolverResolutionNode::Constructor {
            implementation,
            params,
        } => materialize_node(
            node_ref,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner| {
                Ok(computed_node(
                    resolved.dynamic,
                    ExecutionComputedKind::Constructor {
                        implementation: Arc::clone(&implementation.implementation),
                        params: params
                            .iter()
                            .map(|(param_ref, name, kind)| {
                                resolve_ref(
                                    results,
                                    *param_ref,
                                    types,
                                    graph,
                                    refs,
                                    source_interner,
                                )
                                .map(|node_id| ConstructorParam {
                                    name: name.clone(),
                                    kind: *kind,
                                    node: node_id,
                                })
                            })
                            .collect::<Result<_, _>>()?,
                    },
                ))
            },
        ),
        SolverResolutionNode::Init {
            implementation,
            params,
        } => materialize_node(
            node_ref,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner| {
                Ok(computed_node(
                    resolved.dynamic,
                    ExecutionComputedKind::Constructor {
                        implementation: Arc::clone(&implementation.implementation),
                        params: params
                            .iter()
                            .map(|(param_ref, name, kind)| {
                                resolve_ref(
                                    results,
                                    *param_ref,
                                    types,
                                    graph,
                                    refs,
                                    source_interner,
                                )
                                .map(|node_id| ConstructorParam {
                                    name: name.clone(),
                                    kind: *kind,
                                    node: node_id,
                                })
                            })
                            .collect::<Result<_, _>>()?,
                    },
                ))
            },
        ),
    }
}

fn build_transition_node<'ty>(
    dynamic: bool,
    transition: &SolverResolvedTransition<'ty>,
    results: &SolverResolutionArena<'ty>,
    types: &TypeArenas<'ty>,
    graph: &mut BuildExecutionGraph,
    refs: &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
    source_interner: &mut SourceNodeInterner<'ty>,
) -> Result<ExecutionNode, ResolutionError<'ty>> {
    let execution_params = transition
        .params
        .iter()
        .map(|param| ExecutionParam::from_transition_param(param, source_interner, graph))
        .collect();
    let implementations = convert_transition_implementations(
        results,
        &transition.implementations,
        types,
        graph,
        refs,
        source_interner,
    )?;
    let target = resolve_ref(
        results,
        transition.target,
        types,
        graph,
        refs,
        source_interner,
    )?;
    Ok(computed_node(
        dynamic,
        ExecutionComputedKind::Transition {
            return_wrapper: transition.return_wrapper,
            accepts_varargs: transition.accepts_varargs,
            accepts_varkw: transition.accepts_varkw,
            params: execution_params,
            implementations,
            target,
        },
    ))
}

fn build_runtime_union_dispatch_node<'ty>(
    resolved: &SolverResolvedNode<'ty>,
    results: &SolverResolutionArena<'ty>,
    types: &TypeArenas<'ty>,
    graph: &mut BuildExecutionGraph,
    refs: &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
    source_interner: &mut SourceNodeInterner<'ty>,
) -> Result<ExecutionNode, ResolutionError<'ty>> {
    let SolverResolutionNode::RuntimeUnionDispatch { source, branches } = &resolved.resolution
    else {
        unreachable!()
    };
    let source = source_interner.intern(source, graph);
    let branches = branches
        .iter()
        .map(|branch| {
            Ok(ExecutionRuntimeUnionBranch {
                matcher: runtime_matcher_for_type(branch.implementation_variant, types)?,
                target: resolve_ref(results, branch.target, types, graph, refs, source_interner)?,
                arm_source: source_interner.intern(&branch.arm_source, graph),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(computed_node(
        resolved.dynamic,
        ExecutionComputedKind::RuntimeUnionDispatch { source, branches },
    ))
}

fn origin_matcher<'ty>(
    type_ref: PyTypeConcreteKey<'ty>,
    origin: &Option<Arc<Py<PyAny>>>,
    display_name: &Arc<str>,
) -> Result<RuntimeTypeMatcher, ResolutionError<'ty>> {
    let Some(origin) = origin else {
        return Err(ResolutionError::UnsupportedRuntimeUnionMatcher(type_ref));
    };
    Ok(RuntimeTypeMatcher::Class {
        origin: Arc::clone(origin),
        display_name: Arc::clone(display_name),
    })
}

fn runtime_matcher_for_type<'ty>(
    type_ref: PyTypeConcreteKey<'ty>,
    types: &TypeArenas<'ty>,
) -> Result<RuntimeTypeMatcher, ResolutionError<'ty>> {
    match type_ref {
        PyType::Sentinel(key) => match types.sentinels.get(key).inner.value {
            SentinelTypeKind::None => Ok(RuntimeTypeMatcher::None),
            SentinelTypeKind::Ellipsis => {
                Err(ResolutionError::UnsupportedRuntimeUnionMatcher(type_ref))
            }
        },
        PyType::Plain(key) => {
            let plain = types.concrete.plains.get(key);
            origin_matcher(
                type_ref,
                &plain.inner.descriptor.origin,
                &plain.inner.descriptor.display_name,
            )
        }
        PyType::Class(key) => {
            let class_type = types.concrete.classes.get(key);
            origin_matcher(
                type_ref,
                &class_type.inner.descriptor.origin,
                &class_type.inner.descriptor.display_name,
            )
        }
        PyType::Callable(key) => {
            let callable = types.concrete.callables.get(key);
            let params = callable
                .inner
                .params
                .iter()
                .zip(callable.inner.param_kinds.iter())
                .zip(callable.inner.param_has_default.iter())
                .map(
                    |(((name, _), &kind), &has_default)| RuntimeCallableMatchParam {
                        name: Arc::clone(name),
                        kind,
                        has_default,
                    },
                )
                .collect();
            Ok(RuntimeTypeMatcher::Callable { params })
        }
        _ => Err(ResolutionError::UnsupportedRuntimeUnionMatcher(type_ref)),
    }
}

fn materialize_node<'ty>(
    node_ref: SolverResolutionRef,
    graph: &mut BuildExecutionGraph,
    refs: &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
    source_interner: &mut SourceNodeInterner<'ty>,
    build_node: impl FnOnce(
        &mut BuildExecutionGraph,
        &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
        &mut SourceNodeInterner<'ty>,
    ) -> Result<ExecutionNode, ResolutionError<'ty>>,
) -> Result<ExecutionNodeId, ResolutionError<'ty>> {
    let node_id = graph.insert(BuildExecutionEntry::pending());
    refs.insert(node_ref, node_id);
    graph[node_id].node = BuildExecutionNode::Ready(build_node(graph, refs, source_interner)?);
    Ok(node_id)
}

fn get_resolved_node<'a, 'ty>(
    results: &'a SolverResolutionArena<'ty>,
    node_ref: SolverResolutionRef,
) -> Result<&'a SolverResolvedNode<'ty>, ResolutionError<'ty>> {
    match results
        .get(&node_ref)
        .expect("solver result ref must point to a stored result")
    {
        Ok(node) => Ok(node),
        Err(err) => Err(err.clone()),
    }
}

fn convert_transition_implementations<'ty>(
    results: &SolverResolutionArena<'ty>,
    implementations: &[SolverResolvedTransitionImplementation<'ty>],
    types: &TypeArenas<'ty>,
    graph: &mut BuildExecutionGraph,
    refs: &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
    source_interner: &mut SourceNodeInterner<'ty>,
) -> Result<Vec<ExecutionTransitionImplementation>, ResolutionError<'ty>> {
    let mut converted = Vec::with_capacity(implementations.len());
    for implementation in implementations {
        let bound_to = implementation
            .bound_to
            .map(|node_ref| resolve_ref(results, node_ref, types, graph, refs, source_interner))
            .transpose()?;
        let params = implementation
            .params
            .iter()
            .map(|(node_ref, name, kind)| {
                resolve_ref(results, *node_ref, types, graph, refs, source_interner).map(
                    |node_id| ConstructorParam {
                        name: name.clone(),
                        kind: *kind,
                        node: node_id,
                    },
                )
            })
            .collect::<Result<_, _>>()?;
        let result_source = implementation
            .result_source
            .as_ref()
            .map(|source| source_interner.intern(source, graph));
        let implementation_callable = match &implementation.implementation {
            SolverTransitionImplementationCallable::Static(implementation) => {
                ExecutionTransitionImplementationCallable::Static(Arc::clone(implementation))
            }
            SolverTransitionImplementationCallable::Source(source) => {
                ExecutionTransitionImplementationCallable::Source(
                    source_interner.intern(source, graph),
                )
            }
        };

        converted.push(ExecutionTransitionImplementation {
            implementation: implementation_callable,
            bound_to,
            params,
            return_wrapper: implementation.return_wrapper,
            result_source,
        });
    }
    Ok(converted)
}

fn canonicalize_execution_graph(
    graph: BuildExecutionGraph,
    root: ExecutionNodeId,
) -> (ExecutionGraph, ExecutionNodeId) {
    let node_classes = compute_node_classes(&graph);
    let representatives = class_representatives(&node_classes);

    let canonical_node_ids_by_class: Vec<ExecutionNodeId> = (0..representatives.len())
        .map(ExecutionNodeId::from_index)
        .collect();
    let entries = representatives
        .iter()
        .map(|&representative| ExecutionEntry {
            node: remap_node_refs_to_canonical_ids(
                graph[representative].ready_node(),
                &node_classes,
                &canonical_node_ids_by_class,
            ),
            resource_deps: HashSet::new(),
        })
        .collect();
    let mut canonical = ExecutionGraph::from_entries(entries);
    canonical.rebuild_dependencies();

    let root = canonical_id(root, &node_classes, &canonical_node_ids_by_class);
    (canonical, root)
}

fn node_class(node_id: ExecutionNodeId, classes: &[usize]) -> usize {
    classes[node_id.index()]
}

fn source_class(source: ExecutionSourceNodeId, classes: &[usize]) -> usize {
    node_class(source.node_id(), classes)
}

fn execution_signature(
    node: &ExecutionNode,
    node_identity: usize,
    classes: &[usize],
) -> ExecutionSignature {
    match node {
        ExecutionNode::Variable => ExecutionSignature::Variable { node_identity },
        ExecutionNode::Field(field) => ExecutionSignature::Field {
            source: node_class(field.source, classes),
            name: Arc::clone(&field.name),
            access_kind: field.access_kind,
        },
        ExecutionNode::Computed(computed) => ExecutionSignature::Computed {
            dynamic: computed.dynamic,
            cache: computed.cache,
            writable_dependencies: dependency_classes(&computed.writable_dependencies, classes),
            kind: computed_kind_signature(&computed.kind, classes),
        },
    }
}

fn dependency_classes(dependencies: &[ExecutionNodeId], classes: &[usize]) -> Vec<usize> {
    let mut dependencies: Vec<_> = dependencies
        .iter()
        .map(|&node_id| node_class(node_id, classes))
        .collect();
    dependencies.sort_unstable();
    dependencies.dedup();
    dependencies
}

fn computed_kind_signature(
    kind: &ExecutionComputedKind,
    classes: &[usize],
) -> ExecutionComputedKindSignature {
    match kind {
        ExecutionComputedKind::Property {
            source,
            property_name,
        } => ExecutionComputedKindSignature::Property {
            source: node_class(*source, classes),
            property_name: Arc::clone(property_name),
        },
        ExecutionComputedKind::ReadCell { target } => ExecutionComputedKindSignature::ReadCell {
            target: node_class(*target, classes),
        },
        ExecutionComputedKind::Cell { target } => ExecutionComputedKindSignature::Cell {
            target: node_class(*target, classes),
        },
        ExecutionComputedKind::None => ExecutionComputedKindSignature::None,
        ExecutionComputedKind::StaticValue { value } => {
            ExecutionComputedKindSignature::StaticValue {
                value: py_identity(value),
            }
        }
        ExecutionComputedKind::Protocol { members } => ExecutionComputedKindSignature::Protocol {
            members: member_signatures(members, classes),
        },
        ExecutionComputedKind::TypedDict { members } => ExecutionComputedKindSignature::TypedDict {
            members: member_signatures(members, classes),
        },
        ExecutionComputedKind::Transition {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            implementations,
            target,
        } => ExecutionComputedKindSignature::Transition {
            return_wrapper: *return_wrapper,
            accepts_varargs: *accepts_varargs,
            accepts_varkw: *accepts_varkw,
            params: execution_param_signatures(params, classes),
            implementations: transition_implementation_signatures(implementations, classes),
            target: node_class(*target, classes),
        },
        ExecutionComputedKind::RuntimeUnionDispatch { source, branches } => {
            ExecutionComputedKindSignature::RuntimeUnionDispatch {
                source: source_class(*source, classes),
                branches: branches
                    .iter()
                    .map(|branch| RuntimeUnionBranchSignature {
                        matcher: runtime_type_matcher_signature(&branch.matcher),
                        target: node_class(branch.target, classes),
                        arm_source: source_class(branch.arm_source, classes),
                    })
                    .collect(),
            }
        }
        ExecutionComputedKind::Constructor {
            implementation,
            params,
        } => ExecutionComputedKindSignature::Constructor {
            implementation: py_identity(implementation),
            params: constructor_param_signatures(params, classes),
        },
    }
}

fn compute_node_classes(graph: &BuildExecutionGraph) -> Vec<usize> {
    let mut classes = vec![0; graph.entries.len()];

    loop {
        let mut signatures = HashMap::new();
        let mut next_classes = Vec::with_capacity(graph.entries.len());

        for (node_identity, node_id) in graph.keys().enumerate() {
            let signature =
                execution_signature(graph[node_id].ready_node(), node_identity, &classes);
            let next_class_id = signatures.len();
            let class_id = *signatures.entry(signature).or_insert(next_class_id);
            next_classes.push(class_id);
        }

        if next_classes == classes {
            return classes;
        }
        classes = next_classes;
    }
}

fn class_representatives(node_classes: &[usize]) -> Vec<ExecutionNodeId> {
    let mut representatives = Vec::new();
    for (index, &class_id) in node_classes.iter().enumerate() {
        if class_id == representatives.len() {
            debug_assert_eq!(class_id, representatives.len());
            representatives.push(ExecutionNodeId::from_index(index));
        }
    }
    representatives
}

fn remap_node_refs_to_canonical_ids(
    node: &ExecutionNode,
    node_classes: &[usize],
    canonical_node_ids_by_class: &[ExecutionNodeId],
) -> ExecutionNode {
    match node {
        ExecutionNode::Variable => ExecutionNode::Variable,
        ExecutionNode::Field(field) => ExecutionNode::Field(ExecutionField {
            source: canonical_id(field.source, node_classes, canonical_node_ids_by_class),
            name: Arc::clone(&field.name),
            access_kind: field.access_kind,
        }),
        ExecutionNode::Computed(computed) => ExecutionNode::Computed(ExecutionComputed {
            dynamic: computed.dynamic,
            cache: computed.cache,
            writable_dependencies: normalize_node_ids(
                computed
                    .writable_dependencies
                    .iter()
                    .map(|&node_id| {
                        canonical_id(node_id, node_classes, canonical_node_ids_by_class)
                    })
                    .collect(),
            ),
            kind: remap_computed_kind_refs_to_canonical_ids(
                &computed.kind,
                node_classes,
                canonical_node_ids_by_class,
            ),
        }),
    }
}

fn remap_computed_kind_refs_to_canonical_ids(
    kind: &ExecutionComputedKind,
    node_classes: &[usize],
    canonical_node_ids_by_class: &[ExecutionNodeId],
) -> ExecutionComputedKind {
    match kind {
        ExecutionComputedKind::Property {
            source,
            property_name,
        } => ExecutionComputedKind::Property {
            source: canonical_id(*source, node_classes, canonical_node_ids_by_class),
            property_name: Arc::clone(property_name),
        },
        ExecutionComputedKind::ReadCell { target } => ExecutionComputedKind::ReadCell {
            target: canonical_id(*target, node_classes, canonical_node_ids_by_class),
        },
        ExecutionComputedKind::Cell { target } => ExecutionComputedKind::Cell {
            target: canonical_id(*target, node_classes, canonical_node_ids_by_class),
        },
        ExecutionComputedKind::None => ExecutionComputedKind::None,
        ExecutionComputedKind::StaticValue { value } => ExecutionComputedKind::StaticValue {
            value: Arc::clone(value),
        },
        ExecutionComputedKind::Protocol { members } => ExecutionComputedKind::Protocol {
            members: members
                .iter()
                .map(|(name, &node_id)| {
                    (
                        Arc::clone(name),
                        canonical_id(node_id, node_classes, canonical_node_ids_by_class),
                    )
                })
                .collect(),
        },
        ExecutionComputedKind::TypedDict { members } => ExecutionComputedKind::TypedDict {
            members: members
                .iter()
                .map(|(name, &node_id)| {
                    (
                        Arc::clone(name),
                        canonical_id(node_id, node_classes, canonical_node_ids_by_class),
                    )
                })
                .collect(),
        },
        ExecutionComputedKind::Transition {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            implementations,
            target,
        } => ExecutionComputedKind::Transition {
            return_wrapper: *return_wrapper,
            accepts_varargs: *accepts_varargs,
            accepts_varkw: *accepts_varkw,
            params: remap_execution_params(params, node_classes, canonical_node_ids_by_class),
            implementations: remap_transition_implementations(
                implementations,
                node_classes,
                canonical_node_ids_by_class,
            ),
            target: canonical_id(*target, node_classes, canonical_node_ids_by_class),
        },
        ExecutionComputedKind::RuntimeUnionDispatch { source, branches } => {
            ExecutionComputedKind::RuntimeUnionDispatch {
                source: canonical_source_node_id(
                    *source,
                    node_classes,
                    canonical_node_ids_by_class,
                ),
                branches: branches
                    .iter()
                    .map(|branch| ExecutionRuntimeUnionBranch {
                        matcher: branch.matcher.clone(),
                        target: canonical_id(
                            branch.target,
                            node_classes,
                            canonical_node_ids_by_class,
                        ),
                        arm_source: canonical_source_node_id(
                            branch.arm_source,
                            node_classes,
                            canonical_node_ids_by_class,
                        ),
                    })
                    .collect(),
            }
        }
        ExecutionComputedKind::Constructor {
            implementation,
            params,
        } => ExecutionComputedKind::Constructor {
            implementation: Arc::clone(implementation),
            params: params
                .iter()
                .map(|param| ConstructorParam {
                    name: Arc::clone(&param.name),
                    kind: param.kind,
                    node: canonical_id(param.node, node_classes, canonical_node_ids_by_class),
                })
                .collect(),
        },
    }
}

fn remap_execution_params(
    params: &[ExecutionParam],
    node_classes: &[usize],
    canonical_node_ids_by_class: &[ExecutionNodeId],
) -> Vec<ExecutionParam> {
    params
        .iter()
        .map(|param| ExecutionParam {
            name: Arc::clone(&param.name),
            kind: param.kind,
            sources: param
                .sources
                .iter()
                .map(|&source| {
                    canonical_source_node_id(source, node_classes, canonical_node_ids_by_class)
                })
                .collect(),
        })
        .collect()
}

fn canonical_source_node_id(
    source: ExecutionSourceNodeId,
    node_classes: &[usize],
    canonical_node_ids_by_class: &[ExecutionNodeId],
) -> ExecutionSourceNodeId {
    ExecutionSourceNodeId(canonical_id(
        source.node_id(),
        node_classes,
        canonical_node_ids_by_class,
    ))
}

fn remap_transition_implementations(
    implementations: &[ExecutionTransitionImplementation],
    node_classes: &[usize],
    canonical_node_ids_by_class: &[ExecutionNodeId],
) -> Vec<ExecutionTransitionImplementation> {
    implementations
        .iter()
        .map(|implementation| ExecutionTransitionImplementation {
            implementation: remap_transition_implementation_callable(
                &implementation.implementation,
                node_classes,
                canonical_node_ids_by_class,
            ),
            bound_to: implementation
                .bound_to
                .map(|node_id| canonical_id(node_id, node_classes, canonical_node_ids_by_class)),
            params: implementation
                .params
                .iter()
                .map(|param| ConstructorParam {
                    name: Arc::clone(&param.name),
                    kind: param.kind,
                    node: canonical_id(param.node, node_classes, canonical_node_ids_by_class),
                })
                .collect(),
            return_wrapper: implementation.return_wrapper,
            result_source: implementation.result_source.map(|source| {
                canonical_source_node_id(source, node_classes, canonical_node_ids_by_class)
            }),
        })
        .collect()
}

fn remap_transition_implementation_callable(
    implementation: &ExecutionTransitionImplementationCallable,
    node_classes: &[usize],
    canonical_node_ids_by_class: &[ExecutionNodeId],
) -> ExecutionTransitionImplementationCallable {
    match implementation {
        ExecutionTransitionImplementationCallable::Static(implementation) => {
            ExecutionTransitionImplementationCallable::Static(Arc::clone(implementation))
        }
        ExecutionTransitionImplementationCallable::Source(source) => {
            ExecutionTransitionImplementationCallable::Source(canonical_source_node_id(
                *source,
                node_classes,
                canonical_node_ids_by_class,
            ))
        }
    }
}

fn canonical_id(
    node_id: ExecutionNodeId,
    node_classes: &[usize],
    canonical_node_ids_by_class: &[ExecutionNodeId],
) -> ExecutionNodeId {
    canonical_node_ids_by_class[node_classes[node_id.index()]]
}

fn compute_resource_deps(
    graph: &ExecutionGraph,
) -> HashMap<ExecutionNodeId, HashSet<ExecutionSourceNodeId>> {
    let node_ids: Vec<_> = graph.keys().collect();
    let mut deps: HashMap<_, _> = node_ids
        .iter()
        .map(|&node_id| (node_id, HashSet::new()))
        .collect();
    loop {
        let mut changed = false;
        for &node_id in &node_ids {
            let next = resource_deps_for_node(graph, node_id, &deps);
            if next != deps[&node_id] {
                deps.insert(node_id, next);
                changed = true;
            }
        }
        if !changed {
            return deps;
        }
    }
}

fn resource_deps_for_node(
    graph: &ExecutionGraph,
    node_id: ExecutionNodeId,
    deps: &HashMap<ExecutionNodeId, HashSet<ExecutionSourceNodeId>>,
) -> HashSet<ExecutionSourceNodeId> {
    match &graph[node_id].node {
        ExecutionNode::Variable => HashSet::from([ExecutionSourceNodeId(node_id)]),
        ExecutionNode::Field(field) => deps[&field.source].clone(),
        ExecutionNode::Computed(computed) => match &computed.kind {
            ExecutionComputedKind::None | ExecutionComputedKind::StaticValue { .. } => {
                HashSet::new()
            }
            ExecutionComputedKind::Property { source, .. } => deps[source].clone(),
            ExecutionComputedKind::ReadCell { target } | ExecutionComputedKind::Cell { target } => {
                deps[target].clone()
            }
            ExecutionComputedKind::Protocol { members }
            | ExecutionComputedKind::TypedDict { members } => members
                .values()
                .flat_map(|member| deps[member].iter().copied())
                .collect(),
            ExecutionComputedKind::Constructor { params, .. } => params
                .iter()
                .flat_map(|param| deps[&param.node].iter().copied())
                .collect(),
            ExecutionComputedKind::Transition {
                params,
                implementations,
                ..
            } => transition_resource_deps(params, implementations, deps),
            ExecutionComputedKind::RuntimeUnionDispatch { source, branches } => {
                let mut result = deps[&source.node_id()].clone();
                for branch in branches {
                    extend_available_resource_deps(
                        &mut result,
                        deps,
                        branch.target,
                        &HashSet::from([branch.arm_source]),
                    );
                }
                result
            }
        },
    }
}

fn transition_resource_deps(
    params: &[ExecutionParam],
    implementations: &[ExecutionTransitionImplementation],
    deps: &HashMap<ExecutionNodeId, HashSet<ExecutionSourceNodeId>>,
) -> HashSet<ExecutionSourceNodeId> {
    let mut result = HashSet::new();
    let mut unavailable = transition_param_sources(params);
    for implementation in implementations {
        if let ExecutionTransitionImplementationCallable::Source(source) =
            &implementation.implementation
        {
            extend_available_resource_deps(&mut result, deps, source.node_id(), &unavailable);
        }
        if let Some(bound_to) = implementation.bound_to {
            extend_available_resource_deps(&mut result, deps, bound_to, &unavailable);
        }
        for param in &implementation.params {
            extend_available_resource_deps(&mut result, deps, param.node, &unavailable);
        }
        if let Some(result_source) = implementation.result_source {
            unavailable.insert(result_source);
        }
    }
    result
}

fn extend_available_resource_deps(
    result: &mut HashSet<ExecutionSourceNodeId>,
    deps: &HashMap<ExecutionNodeId, HashSet<ExecutionSourceNodeId>>,
    node_id: ExecutionNodeId,
    unavailable: &HashSet<ExecutionSourceNodeId>,
) {
    result.extend(
        deps[&node_id]
            .iter()
            .copied()
            .filter(|source| !unavailable.contains(source)),
    );
}

struct WritableDependantsBuilder {
    dependants: Vec<Vec<WritableDependant>>,
    exclusions: Vec<WritableExclusion>,
}

impl WritableDependantsBuilder {
    fn new(node_count: usize) -> Self {
        Self {
            dependants: vec![Vec::new(); node_count],
            exclusions: Vec::new(),
        }
    }

    fn push_exclusion(
        &mut self,
        node_id: ExecutionNodeId,
        parent: Option<WritableExclusionId>,
    ) -> WritableExclusionId {
        let id = WritableExclusionId(self.exclusions.len());
        self.exclusions.push(WritableExclusion { node_id, parent });
        id
    }

    fn push_dependency(
        &mut self,
        dependency: ExecutionNodeId,
        dependant: ExecutionNodeId,
        exclusions: Option<WritableExclusionId>,
    ) {
        self.dependants[dependency.index()].push(WritableDependant {
            node_id: dependant,
            exclusions,
        });
    }

    fn finish(mut self) -> (Vec<Vec<WritableDependant>>, Vec<WritableExclusion>) {
        for dependants in &mut self.dependants {
            dependants.sort();
            dependants.dedup();
        }
        (self.dependants, self.exclusions)
    }
}

fn compute_writable_dependants(
    graph: &ExecutionGraph,
) -> (Vec<Vec<WritableDependant>>, Vec<WritableExclusion>) {
    let mut builder = WritableDependantsBuilder::new(graph.entries.len());
    for node_id in graph.keys() {
        match &graph[node_id].node {
            ExecutionNode::Variable => {}
            ExecutionNode::Field(field) => {
                builder.push_dependency(field.source, node_id, None);
            }
            ExecutionNode::Computed(computed) => {
                for &dependency in &computed.writable_dependencies {
                    builder.push_dependency(dependency, node_id, None);
                }
                match &computed.kind {
                    ExecutionComputedKind::Transition {
                        params,
                        implementations,
                        ..
                    } => add_transition_writable_dependencies(
                        &mut builder,
                        node_id,
                        params,
                        implementations,
                    ),
                    ExecutionComputedKind::RuntimeUnionDispatch { branches, .. } => {
                        for branch in branches {
                            let exclusions =
                                Some(builder.push_exclusion(branch.arm_source.node_id(), None));
                            builder.push_dependency(branch.target, node_id, exclusions);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    builder.finish()
}

fn add_transition_writable_dependencies(
    builder: &mut WritableDependantsBuilder,
    dependant: ExecutionNodeId,
    params: &[ExecutionParam],
    implementations: &[ExecutionTransitionImplementation],
) {
    let mut initial = transition_param_sources(params)
        .into_iter()
        .map(ExecutionSourceNodeId::node_id)
        .collect::<Vec<_>>();
    initial.sort_unstable();
    initial.dedup();

    let mut unavailable = initial.iter().copied().collect::<HashSet<_>>();
    let mut exclusions = None;
    for node_id in initial {
        exclusions = Some(builder.push_exclusion(node_id, exclusions));
    }

    for implementation in implementations {
        if let ExecutionTransitionImplementationCallable::Source(source) =
            &implementation.implementation
        {
            builder.push_dependency(source.node_id(), dependant, exclusions);
        }
        if let Some(bound_to) = implementation.bound_to {
            builder.push_dependency(bound_to, dependant, exclusions);
        }
        for param in &implementation.params {
            builder.push_dependency(param.node, dependant, exclusions);
        }
        if let Some(result_source) = implementation.result_source
            && unavailable.insert(result_source.node_id())
        {
            exclusions = Some(builder.push_exclusion(result_source.node_id(), exclusions));
        }
    }
}

fn transition_param_sources(params: &[ExecutionParam]) -> HashSet<ExecutionSourceNodeId> {
    params
        .iter()
        .flat_map(|param| param.sources.iter().copied())
        .collect()
}

fn member_signatures(
    members: &BTreeMap<Arc<str>, ExecutionNodeId>,
    classes: &[usize],
) -> Vec<MemberSignature> {
    members
        .iter()
        .map(|(name, &node)| MemberSignature {
            name: Arc::clone(name),
            node: node_class(node, classes),
        })
        .collect()
}

fn constructor_param_signatures(
    params: &[ConstructorParam],
    classes: &[usize],
) -> Vec<ConstructorParamSignature> {
    params
        .iter()
        .map(|param| ConstructorParamSignature {
            name: Arc::clone(&param.name),
            kind: param.kind,
            node: node_class(param.node, classes),
        })
        .collect()
}

fn execution_param_signatures(
    params: &[ExecutionParam],
    classes: &[usize],
) -> Vec<ExecutionParamSignature> {
    params
        .iter()
        .map(|param| ExecutionParamSignature {
            name: Arc::clone(&param.name),
            kind: param.kind,
            sources: param
                .sources
                .iter()
                .map(|&source| source_class(source, classes))
                .collect(),
        })
        .collect()
}

fn transition_implementation_signatures(
    implementations: &[ExecutionTransitionImplementation],
    classes: &[usize],
) -> Vec<TransitionImplementationSignature> {
    implementations
        .iter()
        .map(|implementation| TransitionImplementationSignature {
            implementation: transition_implementation_callable_signature(
                &implementation.implementation,
                classes,
            ),
            bound_to: implementation
                .bound_to
                .map(|node_id| node_class(node_id, classes)),
            params: constructor_param_signatures(&implementation.params, classes),
            return_wrapper: implementation.return_wrapper,
            result_source: implementation
                .result_source
                .map(|source| source_class(source, classes)),
        })
        .collect()
}

fn transition_implementation_callable_signature(
    implementation: &ExecutionTransitionImplementationCallable,
    classes: &[usize],
) -> TransitionImplementationCallableSignature {
    match implementation {
        ExecutionTransitionImplementationCallable::Static(implementation) => {
            TransitionImplementationCallableSignature::Static(py_identity(implementation))
        }
        ExecutionTransitionImplementationCallable::Source(source) => {
            TransitionImplementationCallableSignature::Source(source_class(*source, classes))
        }
    }
}

fn runtime_type_matcher_signature(matcher: &RuntimeTypeMatcher) -> RuntimeTypeMatcherSignature {
    match matcher {
        RuntimeTypeMatcher::None => RuntimeTypeMatcherSignature::None,
        RuntimeTypeMatcher::Class { origin, .. } => {
            RuntimeTypeMatcherSignature::Class(py_identity(origin))
        }
        RuntimeTypeMatcher::Callable { params } => RuntimeTypeMatcherSignature::Callable(
            params
                .iter()
                .map(|param| (Arc::clone(&param.name), param.kind, param.has_default))
                .collect(),
        ),
    }
}

fn py_identity(value: &Arc<Py<PyAny>>) -> PythonIdentity {
    PythonIdentity::from_arc_py_any(value)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use context_solver::Arena as _;
    use pyo3::Python;
    use pyo3::types::PyDict;

    use super::*;
    use crate::qualifier::Qualifier;
    use crate::rules::SolverResolvedNode;
    use crate::types::{
        Concrete, Keyed, PlainType, PyType, PyTypeConcreteKey, PyTypeDescriptor, PyTypeId, Qual,
        Qualified, TypeArenas,
    };

    pub(crate) fn execution_node_id(index: usize) -> ExecutionNodeId {
        ExecutionNodeId::from_index(index)
    }

    pub(crate) fn execution_source_node_id(index: usize) -> ExecutionSourceNodeId {
        ExecutionSourceNodeId(ExecutionNodeId::from_index(index))
    }

    pub(crate) fn execution_graph(nodes: Vec<ExecutionNode>) -> ExecutionGraph {
        let mut graph = ExecutionGraph::from_entries(
            nodes
                .into_iter()
                .map(|node| ExecutionEntry {
                    node,
                    resource_deps: HashSet::new(),
                })
                .collect(),
        );
        graph.rebuild_dependencies();
        graph
    }

    fn with_target_type<R>(
        run: impl for<'ty> FnOnce(&TypeArenas<'ty>, PyTypeConcreteKey<'ty>) -> R,
    ) -> R {
        let mut arenas = TypeArenas::default();
        let key = arenas.concrete.plains.insert(Qualified {
            inner: PlainType::<Qual<Keyed>, Concrete> {
                descriptor: PyTypeDescriptor {
                    id: PyTypeId::new("Target".to_string()),
                    display_name: Arc::from("Target"),
                    origin: None,
                },
                args: Vec::new(),
            },
            qualifier: Qualifier::any(),
        });
        run(&arenas, PyType::Plain(key))
    }

    fn py_object() -> Arc<Py<PyAny>> {
        Python::initialize();
        Python::attach(|py| Arc::new(PyDict::new(py).into_any().unbind()))
    }

    fn entry(node: ExecutionNode) -> BuildExecutionEntry {
        BuildExecutionEntry::ready(node)
    }

    fn constructor_param(name: &str, node: ExecutionNodeId) -> ConstructorParam {
        ConstructorParam {
            name: Arc::from(name),
            kind: ParamKind::PositionalOrKeyword,
            node,
        }
    }

    fn execution_param(name: &str, source: ExecutionSourceNodeId) -> ExecutionParam {
        ExecutionParam {
            name: Arc::from(name),
            kind: ParamKind::PositionalOrKeyword,
            sources: vec![source],
        }
    }

    fn variable() -> ExecutionNode {
        ExecutionNode::Variable
    }

    fn none() -> ExecutionNode {
        computed_node(false, ExecutionComputedKind::None)
    }

    fn read_cell(target: ExecutionNodeId) -> ExecutionNode {
        computed_node(false, ExecutionComputedKind::ReadCell { target })
    }

    fn transition(
        params: Vec<ExecutionParam>,
        implementations: Vec<ExecutionTransitionImplementation>,
        target: ExecutionNodeId,
    ) -> ExecutionNode {
        computed_node(
            false,
            ExecutionComputedKind::Transition {
                return_wrapper: WrapperKind::None,
                accepts_varargs: false,
                accepts_varkw: false,
                params,
                implementations,
                target,
            },
        )
    }

    fn constructor(implementation: Arc<Py<PyAny>>, params: Vec<ConstructorParam>) -> ExecutionNode {
        computed_node(
            false,
            ExecutionComputedKind::Constructor {
                implementation,
                params,
            },
        )
    }

    fn is_constructor(node: &ExecutionNode) -> bool {
        matches!(
            node,
            ExecutionNode::Computed(ExecutionComputed {
                kind: ExecutionComputedKind::Constructor { .. },
                ..
            })
        )
    }

    fn is_none(node: &ExecutionNode) -> bool {
        matches!(
            node,
            ExecutionNode::Computed(ExecutionComputed {
                kind: ExecutionComputedKind::None,
                ..
            })
        )
    }

    #[test]
    fn delegate_alias_does_not_materialize_execution_node() {
        with_target_type(|arenas, target_type| {
            let mut results = SolverResolutionArena::default();
            let target = results.insert(Ok(SolverResolvedNode {
                target_type,
                dynamic: false,
                writable_dependencies: Vec::new(),
                resolution: SolverResolutionNode::None,
            }));
            let root = results.insert(Ok(SolverResolvedNode {
                target_type,
                dynamic: false,
                writable_dependencies: Vec::new(),
                resolution: SolverResolutionNode::Delegate(target),
            }));

            let (graph, root_node) =
                create_execution_graph(&results, root, arenas).expect("create_execution_graph");

            assert_eq!(graph.len(), 1);
            assert!(is_none(&graph[root_node].node));
        });
    }

    #[test]
    fn union_variant_alias_does_not_materialize_execution_node() {
        with_target_type(|arenas, target_type| {
            let mut results = SolverResolutionArena::default();
            let target = results.insert(Ok(SolverResolvedNode {
                target_type,
                dynamic: false,
                writable_dependencies: Vec::new(),
                resolution: SolverResolutionNode::None,
            }));
            let root = results.insert(Ok(SolverResolvedNode {
                target_type,
                dynamic: false,
                writable_dependencies: Vec::new(),
                resolution: SolverResolutionNode::UnionVariant { target },
            }));

            let (graph, root_node) =
                create_execution_graph(&results, root, arenas).expect("create_execution_graph");

            assert_eq!(graph.len(), 1);
            assert!(is_none(&graph[root_node].node));
        });
    }

    #[test]
    fn equivalent_constructor_nodes_are_canonicalized() {
        let mut graph = BuildExecutionGraph::default();
        let implementation = py_object();
        let left = graph.insert(entry(constructor(Arc::clone(&implementation), Vec::new())));
        graph.insert(entry(constructor(implementation, Vec::new())));

        let (graph, root) = canonicalize_execution_graph(graph, left);

        assert_eq!(graph.len(), 1);
        assert!(is_constructor(&graph[root].node));
    }

    #[test]
    fn dag_sharing_is_not_part_of_execution_identity() {
        let mut graph = BuildExecutionGraph::default();
        let dep_impl = py_object();
        let pair_impl = py_object();
        let shared_dep = graph.insert(entry(constructor(Arc::clone(&dep_impl), Vec::new())));
        let shared_pair = graph.insert(entry(constructor(
            Arc::clone(&pair_impl),
            vec![
                constructor_param("left", shared_dep),
                constructor_param("right", shared_dep),
            ],
        )));
        let left_dep = graph.insert(entry(constructor(Arc::clone(&dep_impl), Vec::new())));
        let right_dep = graph.insert(entry(constructor(dep_impl, Vec::new())));
        graph.insert(entry(constructor(
            pair_impl,
            vec![
                constructor_param("left", left_dep),
                constructor_param("right", right_dep),
            ],
        )));

        let (graph, root) = canonicalize_execution_graph(graph, shared_pair);

        assert_eq!(graph.len(), 2);
        assert!(is_constructor(&graph[root].node));
    }

    #[test]
    fn equivalent_regular_cycle_is_canonicalized() {
        let mut graph = BuildExecutionGraph::default();
        let a_impl = py_object();
        let b_impl = py_object();
        let a_outer = graph.insert(BuildExecutionEntry::pending());
        let read_b_outer = graph.insert(BuildExecutionEntry::pending());
        let b = graph.insert(BuildExecutionEntry::pending());
        let a_inner = graph.insert(BuildExecutionEntry::pending());
        let read_b_inner = graph.insert(BuildExecutionEntry::pending());

        graph[a_outer].node = BuildExecutionNode::Ready(constructor(
            Arc::clone(&a_impl),
            vec![constructor_param("b", read_b_outer)],
        ));
        graph[read_b_outer].node = BuildExecutionNode::Ready(read_cell(b));
        graph[b].node = BuildExecutionNode::Ready(constructor(
            Arc::clone(&b_impl),
            vec![constructor_param("a", a_inner)],
        ));
        graph[a_inner].node = BuildExecutionNode::Ready(constructor(
            a_impl,
            vec![constructor_param("b", read_b_inner)],
        ));
        graph[read_b_inner].node = BuildExecutionNode::Ready(read_cell(b));

        let (graph, _) = canonicalize_execution_graph(graph, a_outer);

        assert_eq!(graph.len(), 3);
    }

    #[test]
    fn transition_target_is_part_of_execution_identity() {
        let mut graph = BuildExecutionGraph::default();
        let left_target = graph.insert(entry(variable()));
        let right_target = graph.insert(entry(variable()));
        let left = graph.insert(entry(transition(Vec::new(), Vec::new(), left_target)));
        graph.insert(entry(transition(Vec::new(), Vec::new(), right_target)));

        let (graph, _) = canonicalize_execution_graph(graph, left);

        assert_eq!(graph.len(), 4);
    }

    #[test]
    fn transition_implementation_params_are_part_of_execution_identity() {
        let mut graph = BuildExecutionGraph::default();
        let target = graph.insert(entry(none()));
        let left_param = graph.insert(entry(variable()));
        let right_param = graph.insert(entry(variable()));
        let implementation = py_object();
        let transition_impl = |node| ExecutionTransitionImplementation {
            implementation: ExecutionTransitionImplementationCallable::Static(Arc::clone(
                &implementation,
            )),
            bound_to: None,
            params: vec![constructor_param("audit", node)],
            return_wrapper: WrapperKind::None,
            result_source: None,
        };
        let left = graph.insert(entry(transition(
            Vec::new(),
            vec![transition_impl(left_param)],
            target,
        )));
        graph.insert(entry(transition(
            Vec::new(),
            vec![transition_impl(right_param)],
            target,
        )));

        let (graph, _) = canonicalize_execution_graph(graph, left);

        assert_eq!(graph.len(), 5);
    }

    #[test]
    fn computed_cache_policy_follows_dynamicness() {
        let target = execution_node_id(0);
        let source = execution_source_node_id(0);
        let static_nodes = [
            none(),
            computed_node(
                false,
                ExecutionComputedKind::StaticValue { value: py_object() },
            ),
            read_cell(target),
            computed_node(false, ExecutionComputedKind::Cell { target }),
            transition(Vec::new(), Vec::new(), target),
            computed_node(
                false,
                ExecutionComputedKind::RuntimeUnionDispatch {
                    source,
                    branches: Vec::new(),
                },
            ),
        ];
        assert!(static_nodes.iter().all(execution_node_cached));

        let dynamic = computed_node(
            true,
            ExecutionComputedKind::Property {
                source: target,
                property_name: Arc::from("value"),
            },
        );
        assert!(!execution_node_cached(&dynamic));
    }

    #[test]
    fn read_cell_and_cell_keep_distinct_serialized_kinds() {
        Python::attach(|py| {
            let graph = execution_graph(vec![
                variable(),
                read_cell(execution_node_id(0)),
                computed_node(
                    false,
                    ExecutionComputedKind::Cell {
                        target: execution_node_id(0),
                    },
                ),
            ]);
            let mut refs = crate::pickle::PyRefCollector::default();
            let state = graph.to_state(py, &mut refs);
            assert!(matches!(
                state.nodes[1],
                ExecutionNodeState::Computed {
                    computed: ExecutionComputedKindState::ReadCell { .. },
                    ..
                }
            ));
            assert!(matches!(
                state.nodes[2],
                ExecutionNodeState::Computed {
                    computed: ExecutionComputedKindState::Cell { .. },
                    ..
                }
            ));

            let refs = refs.into_tuple(py).expect("refs tuple");
            let resolver = crate::pickle::PyRefResolver::new(refs.bind(py)).expect("resolver");
            let graph = ExecutionGraph::from_state(state, &resolver).expect("graph state");
            assert!(matches!(
                graph[execution_node_id(1)].node,
                ExecutionNode::Computed(ExecutionComputed {
                    kind: ExecutionComputedKind::ReadCell { .. },
                    ..
                })
            ));
            assert!(matches!(
                graph[execution_node_id(2)].node,
                ExecutionNode::Computed(ExecutionComputed {
                    kind: ExecutionComputedKind::Cell { .. },
                    ..
                })
            ));
        });
    }

    #[test]
    fn equivalent_solver_results_share_execution_node() {
        with_target_type(|arenas, target_type| {
            let mut results = SolverResolutionArena::default();
            let left = results.insert(Ok(SolverResolvedNode {
                target_type,
                dynamic: false,
                writable_dependencies: Vec::new(),
                resolution: SolverResolutionNode::None,
            }));
            let right = results.insert(Ok(SolverResolvedNode {
                target_type,
                dynamic: false,
                writable_dependencies: Vec::new(),
                resolution: SolverResolutionNode::None,
            }));
            let root = results.insert(Ok(SolverResolvedNode {
                target_type,
                dynamic: true,
                writable_dependencies: Vec::new(),
                resolution: SolverResolutionNode::Protocol {
                    members: [(Arc::from("left"), left), (Arc::from("right"), right)].into(),
                },
            }));

            let (graph, root) = create_execution_graph(&results, root, arenas).expect("graph");
            let ExecutionNode::Computed(ExecutionComputed {
                kind: ExecutionComputedKind::Protocol { members },
                ..
            }) = &graph[root].node
            else {
                panic!("protocol root expected")
            };
            assert_eq!(members["left"], members["right"]);
            assert_eq!(graph.len(), 2);
        });
    }

    #[test]
    fn solver_writable_dependencies_lower_for_all_computed_kinds() {
        with_target_type(|arenas, target_type| {
            let mut results = SolverResolutionArena::default();
            let dependency = results.insert(Ok(SolverResolvedNode {
                target_type,
                dynamic: false,
                writable_dependencies: Vec::new(),
                resolution: SolverResolutionNode::Constant {
                    source: Source::transition(None, target_type),
                },
            }));
            let root = results.insert(Ok(SolverResolvedNode {
                target_type,
                dynamic: false,
                writable_dependencies: vec![
                    SolverWritableDependency::Result(dependency),
                    SolverWritableDependency::Result(dependency),
                ],
                resolution: SolverResolutionNode::None,
            }));

            let (graph, root) = create_execution_graph(&results, root, arenas).expect("graph");
            let ExecutionNode::Computed(computed) = &graph[root].node else {
                panic!("computed root expected")
            };
            assert_eq!(computed.writable_dependencies.len(), 1);
            let dependency = computed.writable_dependencies[0];
            assert_eq!(graph.affected_dependants(dependency), HashSet::from([root]));
        });
    }

    #[test]
    fn static_computed_tracks_direct_value_dependencies() {
        let mut graph = BuildExecutionGraph::default();
        let source = graph.insert(entry(variable()));
        let computed = graph.insert(entry(computed_node_with_dependencies(
            false,
            vec![source],
            ExecutionComputedKind::Constructor {
                implementation: py_object(),
                params: vec![constructor_param("value", source)],
            },
        )));

        let (graph, computed) = canonicalize_execution_graph(graph, computed);
        let source = match &graph[computed].node {
            ExecutionNode::Computed(ExecutionComputed {
                kind: ExecutionComputedKind::Constructor { params, .. },
                ..
            }) => params[0].node,
            _ => panic!("constructor expected"),
        };
        assert_eq!(graph.affected_dependants(source), HashSet::from([computed]));
        assert!(graph.affected_dependants(computed).is_empty());
    }

    #[test]
    fn writable_dependency_storage_is_linear_for_a_chain() {
        let count = 128;
        let mut nodes = Vec::with_capacity(count);
        nodes.push(variable());
        for index in 1..count {
            nodes.push(computed_node_with_dependencies(
                false,
                vec![ExecutionNodeId::from_index(index - 1)],
                ExecutionComputedKind::None,
            ));
        }

        let graph = execution_graph(nodes);
        let edge_count = graph
            .writable_dependants
            .iter()
            .map(Vec::len)
            .sum::<usize>();

        assert_eq!(edge_count, count - 1);
        assert_eq!(
            graph
                .affected_dependants(ExecutionNodeId::from_index(0))
                .len(),
            count - 1
        );
    }

    #[test]
    fn transition_exclusion_storage_is_linear() {
        let param_count = 64;
        let implementation_count = 128;
        let param_sources = (0..param_count)
            .map(ExecutionNodeId::from_index)
            .collect::<Vec<_>>();
        let dependency_offset = param_sources.len();
        let result_offset = dependency_offset + implementation_count;
        let result_count = implementation_count / 2;
        let mut nodes = (0..result_offset + result_count)
            .map(|_| variable())
            .collect::<Vec<_>>();
        let params = param_sources
            .iter()
            .flat_map(|&node_id| {
                [
                    execution_param("param", ExecutionSourceNodeId(node_id)),
                    execution_param("duplicate", ExecutionSourceNodeId(node_id)),
                ]
            })
            .collect();
        let implementations = (0..implementation_count)
            .map(|index| ExecutionTransitionImplementation {
                implementation: ExecutionTransitionImplementationCallable::Static(py_object()),
                bound_to: (index > 0)
                    .then(|| ExecutionNodeId::from_index(result_offset + (index - 1) / 2)),
                params: vec![constructor_param(
                    "dependency",
                    ExecutionNodeId::from_index(dependency_offset + index),
                )],
                return_wrapper: WrapperKind::None,
                result_source: Some(ExecutionSourceNodeId(ExecutionNodeId::from_index(
                    result_offset + index / 2,
                ))),
            })
            .collect();
        let target = ExecutionNodeId::from_index(0);
        nodes.push(transition(params, implementations, target));

        let graph = execution_graph(nodes);
        let transition = ExecutionNodeId::from_index(result_offset + result_count);
        let edge_count = graph
            .writable_dependants
            .iter()
            .map(Vec::len)
            .sum::<usize>();

        assert_eq!(graph.writable_exclusions.len(), param_count + result_count);
        assert_eq!(edge_count, implementation_count + result_count);
        assert!(graph.affected_dependants(param_sources[0]).is_empty());
        assert!(
            graph
                .affected_dependants(ExecutionNodeId::from_index(dependency_offset))
                .contains(&transition)
        );
        assert!(
            graph
                .affected_dependants(ExecutionNodeId::from_index(result_offset))
                .is_empty()
        );
    }

    #[test]
    fn writable_dependency_cycles_do_not_include_the_written_node() {
        let left = ExecutionNodeId::from_index(0);
        let right = ExecutionNodeId::from_index(1);
        let graph = execution_graph(vec![
            computed_node_with_dependencies(true, vec![right], ExecutionComputedKind::None),
            computed_node_with_dependencies(true, vec![left], ExecutionComputedKind::None),
        ]);

        assert_eq!(graph.affected_dependants(left), HashSet::from([right]));
        assert_eq!(graph.affected_dependants(right), HashSet::from([left]));
    }

    #[test]
    fn static_runtime_union_tracks_selector_dependency() {
        let mut graph = BuildExecutionGraph::default();
        let selector = graph.insert(entry(variable()));
        let dispatch = graph.insert(entry(computed_node_with_dependencies(
            false,
            vec![selector],
            ExecutionComputedKind::RuntimeUnionDispatch {
                source: ExecutionSourceNodeId(selector),
                branches: Vec::new(),
            },
        )));

        let (graph, dispatch) = canonicalize_execution_graph(graph, dispatch);
        let selector = match &graph[dispatch].node {
            ExecutionNode::Computed(ExecutionComputed {
                writable_dependencies,
                kind: ExecutionComputedKind::RuntimeUnionDispatch { source, .. },
                ..
            }) => {
                assert_eq!(writable_dependencies, &[source.node_id()]);
                source.node_id()
            }
            _ => panic!("runtime union dispatch expected"),
        };
        assert_eq!(
            graph.affected_dependants(selector),
            HashSet::from([dispatch])
        );
    }

    #[test]
    fn runtime_union_arm_source_does_not_invalidate_dispatch() {
        let selector = ExecutionNodeId::from_index(0);
        let arm_source = ExecutionSourceNodeId(ExecutionNodeId::from_index(1));
        let target = ExecutionNodeId::from_index(2);
        let dispatch = ExecutionNodeId::from_index(3);
        let graph = execution_graph(vec![
            variable(),
            variable(),
            computed_node_with_dependencies(
                false,
                vec![arm_source.node_id()],
                ExecutionComputedKind::None,
            ),
            computed_node_with_dependencies(
                false,
                vec![selector],
                ExecutionComputedKind::RuntimeUnionDispatch {
                    source: ExecutionSourceNodeId(selector),
                    branches: vec![ExecutionRuntimeUnionBranch {
                        matcher: RuntimeTypeMatcher::None,
                        target,
                        arm_source,
                    }],
                },
            ),
        ]);

        assert_eq!(
            graph.affected_dependants(arm_source.node_id()),
            HashSet::from([target])
        );
        assert_eq!(
            graph.affected_dependants(selector),
            HashSet::from([dispatch])
        );
        assert_eq!(graph.affected_dependants(target), HashSet::from([dispatch]));
    }

    #[test]
    fn transition_implementation_bound_instance_is_dependency_but_transition_target_is_not() {
        let mut graph = BuildExecutionGraph::default();
        let bound = graph.insert(entry(variable()));
        let target = graph.insert(entry(variable()));
        let transition = graph.insert(entry(transition(
            Vec::new(),
            vec![ExecutionTransitionImplementation {
                implementation: ExecutionTransitionImplementationCallable::Static(py_object()),
                bound_to: Some(bound),
                params: Vec::new(),
                return_wrapper: WrapperKind::None,
                result_source: None,
            }],
            target,
        )));

        let (graph, transition) = canonicalize_execution_graph(graph, transition);
        let bound_source = match &graph[transition].node {
            ExecutionNode::Computed(ExecutionComputed {
                kind:
                    ExecutionComputedKind::Transition {
                        implementations, ..
                    },
                ..
            }) if implementations.len() == 1 => match implementations[0].bound_to {
                Some(bound) => ExecutionSourceNodeId(bound),
                None => panic!("expected transition implementation with bound source"),
            },
            _ => panic!("expected transition with one implementation"),
        };

        assert_eq!(
            graph.affected_dependants(bound_source.node_id()),
            HashSet::from([transition])
        );
    }

    #[test]
    fn transition_implementation_context_params_are_dependencies_but_call_params_are_not() {
        let mut graph = BuildExecutionGraph::default();
        let context = graph.insert(entry(variable()));
        let call_arg = graph.insert(entry(variable()));
        let target = graph.insert(entry(none()));
        let transition = graph.insert(entry(transition(
            vec![execution_param("call_arg", ExecutionSourceNodeId(call_arg))],
            vec![ExecutionTransitionImplementation {
                implementation: ExecutionTransitionImplementationCallable::Static(py_object()),
                bound_to: None,
                params: vec![
                    constructor_param("context", context),
                    constructor_param("call_arg", call_arg),
                ],
                return_wrapper: WrapperKind::None,
                result_source: None,
            }],
            target,
        )));

        let (graph, transition) = canonicalize_execution_graph(graph, transition);
        let (context_source, call_source) = match &graph[transition].node {
            ExecutionNode::Computed(ExecutionComputed {
                kind:
                    ExecutionComputedKind::Transition {
                        implementations, ..
                    },
                ..
            }) if implementations.len() == 1 => {
                let params = &implementations[0].params;
                (
                    ExecutionSourceNodeId(params[0].node),
                    ExecutionSourceNodeId(params[1].node),
                )
            }
            _ => panic!("expected transition with one implementation"),
        };

        assert!(
            graph
                .affected_dependants(context_source.node_id())
                .contains(&transition)
        );
        assert!(
            !graph
                .affected_dependants(call_source.node_id())
                .contains(&transition)
        );
        assert!(graph.affected_dependants(transition).is_empty());
    }

    #[test]
    fn zero_implementation_transition_target_is_not_a_source_dependency() {
        let mut graph = BuildExecutionGraph::default();
        let target = graph.insert(entry(variable()));
        let transition = graph.insert(entry(transition(Vec::new(), Vec::new(), target)));

        let (graph, transition) = canonicalize_execution_graph(graph, transition);

        assert!(graph.affected_dependants(target).is_empty());
        assert!(graph.affected_dependants(transition).is_empty());
    }
}
