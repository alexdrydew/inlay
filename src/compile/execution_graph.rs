use std::{
    collections::{BTreeMap, HashMap, HashSet},
    ops::{Index, IndexMut},
    sync::Arc,
};

use context_solver::Arena as ResultsArena;
use inlay_instrument::{inlay_event, instrumented};

use pyo3::prelude::*;

use crate::{
    python_identity::PythonIdentity,
    registry::{Source, SourceKind},
    rules::{
        ResolutionError, SolverResolutionArena, SolverResolutionNode, SolverResolutionRef,
        SolverResolvedNode, SolverResolvedTransition, SolverResolvedTransitionImplementation,
        SolverRuntimeUnionBranch, SolverTransitionImplementationCallable, TransitionParam,
    },
    types::{
        MemberAccessKind, ParamKind, PyType, PyTypeConcreteKey, SentinelTypeKind, TypeArenas,
        WrapperKind,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ExecutionNodeId(u32);

impl ExecutionNodeId {
    fn from_index(index: usize) -> Self {
        Self(
            index
                .try_into()
                .expect("execution graph cannot exceed u32::MAX entries"),
        )
    }

    fn index(self) -> usize {
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
            SourceKind::ProviderResult(value) => ExecutionNode::StaticValue {
                value: Arc::clone(value),
            },
            SourceKind::Transition { .. } => ExecutionNode::Constant,
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

#[derive(Clone, PartialEq, Eq)]
struct MemberSignature {
    name: Arc<str>,
    node: usize,
}

#[derive(Clone, PartialEq, Eq)]
struct ConstructorParamSignature {
    name: Arc<str>,
    kind: ParamKind,
    node: usize,
}

#[derive(Clone, PartialEq, Eq)]
struct ExecutionParamSignature {
    name: Arc<str>,
    kind: ParamKind,
    sources: Vec<usize>,
}

#[derive(Clone, PartialEq, Eq)]
enum TransitionImplementationCallableSignature {
    Static(PythonIdentity),
    Source(usize),
}

#[derive(Clone, PartialEq, Eq)]
struct TransitionImplementationSignature {
    implementation: TransitionImplementationCallableSignature,
    bound_to: Option<usize>,
    params: Vec<ConstructorParamSignature>,
    return_wrapper: WrapperKind,
    result_source: Option<usize>,
}

#[derive(Clone, PartialEq, Eq)]
enum RuntimeTypeMatcherSignature {
    None,
    Class(PythonIdentity),
    Callable(Vec<(Arc<str>, ParamKind, bool)>),
}

#[derive(Clone, PartialEq, Eq)]
struct RuntimeUnionBranchSignature {
    matcher: RuntimeTypeMatcherSignature,
    target: usize,
    arm_source: usize,
}

#[derive(Clone, PartialEq, Eq)]
enum ExecutionSignature {
    Constant {
        node_identity: usize,
    },
    Property {
        source: usize,
        property_name: Arc<str>,
    },
    LazyRef {
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
    Attribute {
        source: usize,
        name: Arc<str>,
        access_kind: MemberAccessKind,
    },
    Constructor {
        implementation: PythonIdentity,
        params: Vec<ConstructorParamSignature>,
    },
}

#[derive(Clone)]
pub(crate) enum ExecutionNode {
    Constant,
    Property {
        source: ExecutionNodeId,
        property_name: Arc<str>,
    },
    LazyRef {
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
    Attribute {
        source: ExecutionNodeId,
        attribute_name: Arc<str>,
        access_kind: MemberAccessKind,
    },
    Constructor {
        implementation: Arc<Py<PyAny>>,
        params: Vec<ConstructorParam>,
    },
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
    pub(crate) source_deps: HashSet<ExecutionSourceNodeId>,
}

impl std::fmt::Debug for ExecutionEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionEntry")
            .field("source_deps", &self.source_deps.len())
            .field(
                "cached",
                &matches!(&self.node, ExecutionNode::Constructor { .. }),
            )
            .finish()
    }
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
}

impl ExecutionGraph {
    fn from_entries(entries: Vec<ExecutionEntry>) -> Self {
        Self { entries }
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
    inlay_event!(
        name: "inlay.create_execution_graph.reachable_result_refs",
        reachable_result_refs = refs.len() as u64,
    );
    let (graph, root) = canonicalize_execution_graph(graph, root);
    Ok((graph, root))
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
                Ok(ExecutionNode::None)
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
                Ok(ExecutionNode::Property {
                    source: resolve_ref(results, *source, types, graph, refs, source_interner)?,
                    property_name: property_name.clone(),
                })
            },
        ),
        SolverResolutionNode::LazyRef { target } => materialize_node(
            node_ref,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner| {
                Ok(ExecutionNode::LazyRef {
                    target: resolve_ref(results, *target, types, graph, refs, source_interner)?,
                })
            },
        ),
        SolverResolutionNode::Protocol { members } => materialize_node(
            node_ref,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner| {
                Ok(ExecutionNode::Protocol {
                    members: members
                        .iter()
                        .map(|(name, &member_ref)| {
                            resolve_ref(results, member_ref, types, graph, refs, source_interner)
                                .map(|node_id| (name.clone(), node_id))
                        })
                        .collect::<Result<_, _>>()?,
                })
            },
        ),
        SolverResolutionNode::TypedDict { members } => materialize_node(
            node_ref,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner| {
                Ok(ExecutionNode::TypedDict {
                    members: members
                        .iter()
                        .map(|(name, &member_ref)| {
                            resolve_ref(results, member_ref, types, graph, refs, source_interner)
                                .map(|node_id| (name.clone(), node_id))
                        })
                        .collect::<Result<_, _>>()?,
                })
            },
        ),
        SolverResolutionNode::Transition(transition) => materialize_node(
            node_ref,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner| {
                build_transition_node(transition, results, types, graph, refs, source_interner)
            },
        ),
        SolverResolutionNode::RuntimeUnionDispatch { source, branches } => materialize_node(
            node_ref,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner| {
                build_runtime_union_dispatch_node(
                    source,
                    branches,
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
                Ok(ExecutionNode::Attribute {
                    source: resolve_ref(results, *source, types, graph, refs, source_interner)?,
                    attribute_name: attribute_name.clone(),
                    access_kind: *access_kind,
                })
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
                Ok(ExecutionNode::Constructor {
                    implementation: Arc::clone(&implementation.implementation),
                    params: params
                        .iter()
                        .map(|(param_ref, name, kind)| {
                            resolve_ref(results, *param_ref, types, graph, refs, source_interner)
                                .map(|node_id| ConstructorParam {
                                    name: name.clone(),
                                    kind: *kind,
                                    node: node_id,
                                })
                        })
                        .collect::<Result<_, _>>()?,
                })
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
                Ok(ExecutionNode::Constructor {
                    implementation: Arc::clone(&implementation.implementation),
                    params: params
                        .iter()
                        .map(|(param_ref, name, kind)| {
                            resolve_ref(results, *param_ref, types, graph, refs, source_interner)
                                .map(|node_id| ConstructorParam {
                                    name: name.clone(),
                                    kind: *kind,
                                    node: node_id,
                                })
                        })
                        .collect::<Result<_, _>>()?,
                })
            },
        ),
    }
}

fn build_transition_node<'ty>(
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
    Ok(ExecutionNode::Transition {
        return_wrapper: transition.return_wrapper,
        accepts_varargs: transition.accepts_varargs,
        accepts_varkw: transition.accepts_varkw,
        params: execution_params,
        implementations,
        target,
    })
}

fn build_runtime_union_dispatch_node<'ty>(
    source: &Source<'ty>,
    branches: &[SolverRuntimeUnionBranch<'ty>],
    results: &SolverResolutionArena<'ty>,
    types: &TypeArenas<'ty>,
    graph: &mut BuildExecutionGraph,
    refs: &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
    source_interner: &mut SourceNodeInterner<'ty>,
) -> Result<ExecutionNode, ResolutionError<'ty>> {
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

    Ok(ExecutionNode::RuntimeUnionDispatch { source, branches })
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
            source_deps: HashSet::new(),
        })
        .collect();
    let mut canonical = ExecutionGraph::from_entries(entries);

    let source_deps = compute_source_deps(&canonical);
    for (node_id, deps) in source_deps {
        canonical[node_id].source_deps = deps;
    }

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
        ExecutionNode::Constant => ExecutionSignature::Constant { node_identity },
        ExecutionNode::StaticValue { value } => ExecutionSignature::StaticValue {
            value: py_identity(value),
        },
        ExecutionNode::Property {
            source,
            property_name,
        } => ExecutionSignature::Property {
            source: node_class(*source, classes),
            property_name: Arc::clone(property_name),
        },
        ExecutionNode::LazyRef { target } => ExecutionSignature::LazyRef {
            target: node_class(*target, classes),
        },
        ExecutionNode::None => ExecutionSignature::None,
        ExecutionNode::Protocol { members } => ExecutionSignature::Protocol {
            members: member_signatures(members, classes),
        },
        ExecutionNode::TypedDict { members } => ExecutionSignature::TypedDict {
            members: member_signatures(members, classes),
        },
        ExecutionNode::Transition {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            implementations,
            target,
        } => ExecutionSignature::Transition {
            return_wrapper: *return_wrapper,
            accepts_varargs: *accepts_varargs,
            accepts_varkw: *accepts_varkw,
            params: execution_param_signatures(params, classes),
            implementations: transition_implementation_signatures(implementations, classes),
            target: node_class(*target, classes),
        },
        ExecutionNode::RuntimeUnionDispatch { source, branches } => {
            ExecutionSignature::RuntimeUnionDispatch {
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
        ExecutionNode::Attribute {
            source,
            attribute_name,
            access_kind,
        } => ExecutionSignature::Attribute {
            source: node_class(*source, classes),
            name: Arc::clone(attribute_name),
            access_kind: *access_kind,
        },
        ExecutionNode::Constructor {
            implementation,
            params,
        } => ExecutionSignature::Constructor {
            implementation: py_identity(implementation),
            params: constructor_param_signatures(params, classes),
        },
    }
}

fn compute_node_classes(graph: &BuildExecutionGraph) -> Vec<usize> {
    let mut classes = vec![0; graph.entries.len()];

    loop {
        let mut signatures = Vec::new();
        let mut next_classes = Vec::with_capacity(graph.entries.len());

        for (node_identity, node_id) in graph.keys().enumerate() {
            let signature =
                execution_signature(graph[node_id].ready_node(), node_identity, &classes);
            let class_id = signatures
                .iter()
                .position(|existing| existing == &signature)
                .unwrap_or_else(|| {
                    signatures.push(signature);
                    signatures.len() - 1
                });
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
        ExecutionNode::Constant => ExecutionNode::Constant,
        ExecutionNode::StaticValue { value } => ExecutionNode::StaticValue {
            value: Arc::clone(value),
        },
        ExecutionNode::Property {
            source,
            property_name,
        } => ExecutionNode::Property {
            source: canonical_id(*source, node_classes, canonical_node_ids_by_class),
            property_name: Arc::clone(property_name),
        },
        ExecutionNode::LazyRef { target } => ExecutionNode::LazyRef {
            target: canonical_id(*target, node_classes, canonical_node_ids_by_class),
        },
        ExecutionNode::None => ExecutionNode::None,
        ExecutionNode::Protocol { members } => ExecutionNode::Protocol {
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
        ExecutionNode::TypedDict { members } => ExecutionNode::TypedDict {
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
        ExecutionNode::Transition {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            implementations,
            target,
        } => ExecutionNode::Transition {
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
        ExecutionNode::RuntimeUnionDispatch { source, branches } => {
            ExecutionNode::RuntimeUnionDispatch {
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
        ExecutionNode::Attribute {
            source,
            attribute_name,
            access_kind,
        } => ExecutionNode::Attribute {
            source: canonical_id(*source, node_classes, canonical_node_ids_by_class),
            attribute_name: Arc::clone(attribute_name),
            access_kind: *access_kind,
        },
        ExecutionNode::Constructor {
            implementation,
            params,
        } => ExecutionNode::Constructor {
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

fn compute_source_deps(
    graph: &ExecutionGraph,
) -> HashMap<ExecutionNodeId, HashSet<ExecutionSourceNodeId>> {
    let node_ids: Vec<ExecutionNodeId> = graph.keys().collect();
    let mut deps: HashMap<ExecutionNodeId, HashSet<ExecutionSourceNodeId>> = node_ids
        .iter()
        .map(|&node_id| (node_id, HashSet::new()))
        .collect();

    let mut changed = true;
    while changed {
        changed = false;
        for &node_id in &node_ids {
            let next = source_deps_for_node(graph, node_id, &deps);
            if next != deps[&node_id] {
                deps.insert(node_id, next);
                changed = true;
            }
        }
    }

    deps
}

fn source_deps_for_node(
    graph: &ExecutionGraph,
    node_id: ExecutionNodeId,
    deps: &HashMap<ExecutionNodeId, HashSet<ExecutionSourceNodeId>>,
) -> HashSet<ExecutionSourceNodeId> {
    match &graph[node_id].node {
        ExecutionNode::Constant => HashSet::from([ExecutionSourceNodeId(node_id)]),
        ExecutionNode::StaticValue { .. } => HashSet::new(),
        ExecutionNode::Transition {
            params,
            implementations,
            ..
        } => transition_source_deps(params, implementations, deps),
        ExecutionNode::RuntimeUnionDispatch { source, branches } => {
            let mut result = HashSet::new();
            extend_available_source_deps(&mut result, deps, source.node_id(), &HashSet::new());
            for branch in branches {
                extend_available_source_deps(
                    &mut result,
                    deps,
                    branch.target,
                    &HashSet::from([branch.arm_source]),
                );
            }
            result
        }
        node => source_dep_children(node)
            .into_iter()
            .flat_map(|child| deps[&child].iter().copied())
            .collect(),
    }
}

/// special cases parameters introduced by the transition itself: propagates only parameters that are
/// bound to parent context sources forcing parents to invalidate when these sources change.
fn transition_source_deps(
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
            extend_available_source_deps(&mut result, deps, source.node_id(), &unavailable);
        }
        if let Some(bound_to) = implementation.bound_to {
            extend_available_source_deps(&mut result, deps, bound_to, &unavailable);
        }
        for param in &implementation.params {
            extend_available_source_deps(&mut result, deps, param.node, &unavailable);
        }
        if let Some(result_source) = implementation.result_source {
            unavailable.insert(result_source);
        }
    }

    result
}

fn transition_param_sources(params: &[ExecutionParam]) -> HashSet<ExecutionSourceNodeId> {
    params
        .iter()
        .flat_map(|param| param.sources.iter().copied())
        .collect()
}

fn extend_available_source_deps(
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

fn source_dep_children(node: &ExecutionNode) -> Vec<ExecutionNodeId> {
    match node {
        ExecutionNode::Constant | ExecutionNode::StaticValue { .. } | ExecutionNode::None => {
            Vec::new()
        }
        ExecutionNode::Property { source, .. } | ExecutionNode::Attribute { source, .. } => {
            vec![*source]
        }
        ExecutionNode::LazyRef { target } => vec![*target],
        ExecutionNode::Protocol { members } | ExecutionNode::TypedDict { members } => {
            members.values().copied().collect()
        }
        ExecutionNode::Transition { .. } | ExecutionNode::RuntimeUnionDispatch { .. } => Vec::new(),
        ExecutionNode::Constructor { params, .. } => {
            params.iter().map(|param| param.node).collect()
        }
    }
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
        ExecutionGraph::from_entries(
            nodes
                .into_iter()
                .map(|node| ExecutionEntry {
                    node,
                    source_deps: HashSet::new(),
                })
                .collect(),
        )
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

    fn constructor(implementation: Arc<Py<PyAny>>, params: Vec<ConstructorParam>) -> ExecutionNode {
        ExecutionNode::Constructor {
            implementation,
            params,
        }
    }

    #[test]
    fn delegate_alias_does_not_materialize_execution_node() {
        with_target_type(|arenas, target_type| {
            let mut results = SolverResolutionArena::default();
            let target = results.insert(Ok(SolverResolvedNode {
                target_type,
                resolution: SolverResolutionNode::None,
            }));
            let root = results.insert(Ok(SolverResolvedNode {
                target_type,
                resolution: SolverResolutionNode::Delegate(target),
            }));

            let (graph, root_node) =
                create_execution_graph(&results, root, arenas).expect("create_execution_graph");

            assert_eq!(graph.len(), 1);
            assert!(matches!(&graph[root_node].node, ExecutionNode::None));
        });
    }

    #[test]
    fn union_variant_alias_does_not_materialize_execution_node() {
        with_target_type(|arenas, target_type| {
            let mut results = SolverResolutionArena::default();
            let target = results.insert(Ok(SolverResolvedNode {
                target_type,
                resolution: SolverResolutionNode::None,
            }));
            let root = results.insert(Ok(SolverResolvedNode {
                target_type,
                resolution: SolverResolutionNode::UnionVariant { target },
            }));

            let (graph, root_node) =
                create_execution_graph(&results, root, arenas).expect("create_execution_graph");

            assert_eq!(graph.len(), 1);
            assert!(matches!(&graph[root_node].node, ExecutionNode::None));
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
        assert!(matches!(
            &graph[root].node,
            ExecutionNode::Constructor { .. }
        ));
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
        assert!(matches!(
            &graph[root].node,
            ExecutionNode::Constructor { .. }
        ));
    }

    #[test]
    fn equivalent_regular_cycle_is_canonicalized() {
        let mut graph = BuildExecutionGraph::default();
        let a_impl = py_object();
        let b_impl = py_object();
        let a_outer = graph.insert(BuildExecutionEntry::pending());
        let lazy_b_outer = graph.insert(BuildExecutionEntry::pending());
        let b = graph.insert(BuildExecutionEntry::pending());
        let a_inner = graph.insert(BuildExecutionEntry::pending());
        let lazy_b_inner = graph.insert(BuildExecutionEntry::pending());

        graph[a_outer].node = BuildExecutionNode::Ready(constructor(
            Arc::clone(&a_impl),
            vec![constructor_param("b", lazy_b_outer)],
        ));
        graph[lazy_b_outer].node = BuildExecutionNode::Ready(ExecutionNode::LazyRef { target: b });
        graph[b].node = BuildExecutionNode::Ready(constructor(
            Arc::clone(&b_impl),
            vec![constructor_param("a", a_inner)],
        ));
        graph[a_inner].node = BuildExecutionNode::Ready(constructor(
            a_impl,
            vec![constructor_param("b", lazy_b_inner)],
        ));
        graph[lazy_b_inner].node = BuildExecutionNode::Ready(ExecutionNode::LazyRef { target: b });

        let (graph, _) = canonicalize_execution_graph(graph, a_outer);

        assert_eq!(graph.len(), 3);
    }

    #[test]
    fn transition_target_is_part_of_execution_identity() {
        let mut graph = BuildExecutionGraph::default();
        let left_target = graph.insert(entry(ExecutionNode::Constant));
        let right_target = graph.insert(entry(ExecutionNode::Constant));
        let left = graph.insert(entry(ExecutionNode::Transition {
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            params: Vec::new(),
            implementations: Vec::new(),
            target: left_target,
        }));
        graph.insert(entry(ExecutionNode::Transition {
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            params: Vec::new(),
            implementations: Vec::new(),
            target: right_target,
        }));

        let (graph, _) = canonicalize_execution_graph(graph, left);

        assert_eq!(graph.len(), 4);
    }

    #[test]
    fn transition_implementation_params_are_part_of_execution_identity() {
        let mut graph = BuildExecutionGraph::default();
        let target = graph.insert(entry(ExecutionNode::None));
        let left_param = graph.insert(entry(ExecutionNode::Constant));
        let right_param = graph.insert(entry(ExecutionNode::Constant));
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
        let left = graph.insert(entry(ExecutionNode::Transition {
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            params: Vec::new(),
            implementations: vec![transition_impl(left_param)],
            target,
        }));
        graph.insert(entry(ExecutionNode::Transition {
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            params: Vec::new(),
            implementations: vec![transition_impl(right_param)],
            target,
        }));

        let (graph, _) = canonicalize_execution_graph(graph, left);

        assert_eq!(graph.len(), 5);
    }

    #[test]
    fn transition_implementation_bound_instance_is_dependency_but_transition_target_is_not() {
        let mut graph = BuildExecutionGraph::default();
        let bound = graph.insert(entry(ExecutionNode::Constant));
        let target = graph.insert(entry(ExecutionNode::Constant));
        let transition = graph.insert(entry(ExecutionNode::Transition {
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            params: Vec::new(),
            implementations: vec![ExecutionTransitionImplementation {
                implementation: ExecutionTransitionImplementationCallable::Static(py_object()),
                bound_to: Some(bound),
                params: Vec::new(),
                return_wrapper: WrapperKind::None,
                result_source: None,
            }],
            target,
        }));

        let (graph, transition) = canonicalize_execution_graph(graph, transition);
        let bound_source = match &graph[transition].node {
            ExecutionNode::Transition {
                implementations, ..
            } if implementations.len() == 1 => match implementations[0].bound_to {
                Some(bound) => ExecutionSourceNodeId(bound),
                None => panic!("expected transition implementation with bound source"),
            },
            _ => panic!("expected transition with one implementation"),
        };

        assert_eq!(graph[transition].source_deps, HashSet::from([bound_source]));
    }

    #[test]
    fn transition_implementation_context_params_are_dependencies_but_call_params_are_not() {
        let mut graph = BuildExecutionGraph::default();
        let context = graph.insert(entry(ExecutionNode::Constant));
        let call_arg = graph.insert(entry(ExecutionNode::Constant));
        let target = graph.insert(entry(ExecutionNode::None));
        let transition = graph.insert(entry(ExecutionNode::Transition {
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            params: vec![execution_param("call_arg", ExecutionSourceNodeId(call_arg))],
            implementations: vec![ExecutionTransitionImplementation {
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
        }));

        let (graph, transition) = canonicalize_execution_graph(graph, transition);
        let (context_source, call_source) = match &graph[transition].node {
            ExecutionNode::Transition {
                implementations, ..
            } if implementations.len() == 1 => {
                let params = &implementations[0].params;
                (
                    ExecutionSourceNodeId(params[0].node),
                    ExecutionSourceNodeId(params[1].node),
                )
            }
            _ => panic!("expected transition with one implementation"),
        };

        assert!(graph[transition].source_deps.contains(&context_source));
        assert!(!graph[transition].source_deps.contains(&call_source));
    }

    #[test]
    fn zero_implementation_transition_target_is_not_a_source_dependency() {
        let mut graph = BuildExecutionGraph::default();
        let target = graph.insert(entry(ExecutionNode::Constant));
        let transition = graph.insert(entry(ExecutionNode::Transition {
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            params: Vec::new(),
            implementations: Vec::new(),
            target,
        }));

        let (graph, transition) = canonicalize_execution_graph(graph, transition);

        assert!(graph[transition].source_deps.is_empty());
    }
}
