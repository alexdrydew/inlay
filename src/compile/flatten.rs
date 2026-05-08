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
    registry::Source,
    rules::{
        MethodParam, ResolutionError, SolverResolutionArena, SolverResolutionNode,
        SolverResolutionRef, SolverResolvedMethodImplementation, SolverResolvedNode,
    },
    types::{MemberAccessKind, ParamKind, WrapperKind},
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

#[derive(Clone, Default, PartialEq, Eq, Hash)]
struct SourceContext<'ty> {
    transitions: BTreeMap<Source<'ty>, ExactSourceKey<'ty>>,
}

impl<'ty> SourceContext<'ty> {
    fn resolve_or_allocate(
        &mut self,
        source: &Source<'ty>,
        source_interner: &mut SourceNodeInterner<'ty>,
    ) -> ExactSourceKey<'ty> {
        if let Some(exact) = self.transitions.get(source) {
            return exact.clone();
        }
        let exact = source_interner.allocate_exact(source);
        self.transitions.insert(source.clone(), exact.clone());
        exact
    }

    fn allocate_transition(
        &mut self,
        source: &Source<'ty>,
        source_interner: &mut SourceNodeInterner<'ty>,
    ) -> ExactSourceKey<'ty> {
        let exact = source_interner.allocate_exact(source);
        self.transitions.insert(source.clone(), exact.clone());
        exact
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum ExactSourceKey<'ty> {
    Provider(Source<'ty>),
    Transition { source: Source<'ty>, epoch: u32 },
}

#[derive(Default)]
struct SourceNodeInterner<'ty> {
    sources: HashMap<ExactSourceKey<'ty>, ExecutionSourceNodeId>,
    next_epoch: u32,
}

impl<'ty> SourceNodeInterner<'ty> {
    fn allocate_exact(&mut self, source: &Source<'ty>) -> ExactSourceKey<'ty> {
        if source.transition_type_ref().is_none() {
            return ExactSourceKey::Provider(source.clone());
        }
        let epoch = self.next_epoch;
        self.next_epoch = self
            .next_epoch
            .checked_add(1)
            .expect("source epoch cannot exceed u32::MAX");
        ExactSourceKey::Transition {
            source: source.clone(),
            epoch,
        }
    }

    fn intern(
        &mut self,
        source: &ExactSourceKey<'ty>,
        graph: &mut BuildExecutionGraph,
    ) -> ExecutionSourceNodeId {
        if let Some(source_node_id) = self.sources.get(source) {
            return *source_node_id;
        }

        let node_id = graph.insert(BuildExecutionEntry::ready(ExecutionNode::Constant));
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
    fn from_method_param<'ty>(
        param: &MethodParam<'ty>,
        source_context: &mut SourceContext<'ty>,
        source_interner: &mut SourceNodeInterner<'ty>,
        graph: &mut BuildExecutionGraph,
    ) -> Self {
        let sources = param
            .logical_sources
            .iter()
            .map(|source| {
                let exact = source_context.allocate_transition(source, source_interner);
                source_interner.intern(&exact, graph)
            })
            .collect();
        Self {
            name: Arc::clone(&param.name),
            kind: param.kind,
            sources,
        }
    }
}

#[derive(Clone)]
pub(crate) struct ExecutionMethodImplementation {
    pub(crate) implementation: Arc<Py<PyAny>>,
    pub(crate) bound_to: Option<ExecutionNodeId>,
    pub(crate) params: Vec<ConstructorParam>,
    pub(crate) return_wrapper: WrapperKind,
    pub(crate) result_source: Option<ExecutionSourceNodeId>,
}

#[derive(Clone, PartialEq, Eq)]
struct ConstructorParamLabel {
    name: Arc<str>,
    kind: ParamKind,
}

#[derive(Clone, PartialEq, Eq)]
struct ExecutionParamLabel {
    name: Arc<str>,
    kind: ParamKind,
    sources: Vec<ExecutionSourceNodeId>,
}

#[derive(Clone, PartialEq, Eq)]
struct MethodImplementationLabel {
    implementation: PythonIdentity,
    has_bound_to: bool,
    params: Vec<ConstructorParamLabel>,
    return_wrapper: WrapperKind,
    result_source: Option<ExecutionSourceNodeId>,
}

#[derive(Clone, PartialEq, Eq)]
enum ExecutionIdentityLabel {
    Constant(usize),
    Property(Arc<str>),
    LazyRef,
    None,
    Protocol {
        members: Vec<Arc<str>>,
    },
    TypedDict {
        members: Vec<Arc<str>>,
    },
    Method {
        return_wrapper: WrapperKind,
        accepts_varargs: bool,
        accepts_varkw: bool,
        params: Vec<ExecutionParamLabel>,
        implementations: Vec<MethodImplementationLabel>,
    },
    AutoMethod {
        return_wrapper: WrapperKind,
        accepts_varargs: bool,
        accepts_varkw: bool,
        params: Vec<ExecutionParamLabel>,
    },
    Attribute {
        name: Arc<str>,
        access_kind: MemberAccessKind,
    },
    Constructor {
        implementation: PythonIdentity,
        params: Vec<ConstructorParamLabel>,
    },
}

struct IdentityNode {
    label: ExecutionIdentityLabel,
    children: Vec<ExecutionNodeId>,
}

#[derive(Clone, PartialEq, Eq)]
struct IdentitySignature {
    label: ExecutionIdentityLabel,
    children: Vec<usize>,
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
    Protocol {
        members: BTreeMap<Arc<str>, ExecutionNodeId>,
    },
    TypedDict {
        members: BTreeMap<Arc<str>, ExecutionNodeId>,
    },
    Method {
        return_wrapper: WrapperKind,
        accepts_varargs: bool,
        accepts_varkw: bool,
        params: Vec<ExecutionParam>,
        implementations: Vec<ExecutionMethodImplementation>,
        target: ExecutionNodeId,
    },
    AutoMethod {
        return_wrapper: WrapperKind,
        accepts_varargs: bool,
        accepts_varkw: bool,
        params: Vec<ExecutionParam>,
        target: ExecutionNodeId,
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

#[derive(Clone, Default)]
pub(crate) struct ResourcePlan {
    pub(crate) sources: HashSet<ExecutionSourceNodeId>,
    pub(crate) caches: HashSet<ExecutionNodeId>,
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

#[instrumented(name = "inlay.flatten", target = "inlay", level = "trace")]
pub(crate) fn flatten<'ty>(
    results: SolverResolutionArena<'ty>,
    root: SolverResolutionRef,
) -> Result<(ExecutionGraph, ExecutionNodeId), ResolutionError<'ty>> {
    let mut graph = BuildExecutionGraph::default();
    let mut refs = HashMap::new();
    let mut source_interner = SourceNodeInterner::default();
    let root = resolve_ref(
        &results,
        root,
        &mut SourceContext::default(),
        &mut graph,
        &mut refs,
        &mut source_interner,
    )?;
    inlay_event!(
        name: "inlay.flatten.reachable_result_refs",
        reachable_result_refs = refs.len() as u64,
    );
    let (graph, root) = canonicalize_execution_graph(graph, root);
    Ok((graph, root))
}

fn resolve_ref<'ty>(
    results: &SolverResolutionArena<'ty>,
    node_ref: SolverResolutionRef,
    source_context: &mut SourceContext<'ty>,
    graph: &mut BuildExecutionGraph,
    refs: &mut HashMap<(SolverResolutionRef, SourceContext<'ty>), ExecutionNodeId>,
    source_interner: &mut SourceNodeInterner<'ty>,
) -> Result<ExecutionNodeId, ResolutionError<'ty>> {
    let ref_key = (node_ref, source_context.clone());
    if let Some(&node_id) = refs.get(&ref_key) {
        return Ok(node_id);
    }

    let resolved = get_resolved_node(results, node_ref)?;
    match &resolved.resolution {
        SolverResolutionNode::Delegate(target) | SolverResolutionNode::UnionVariant { target } => {
            let node_id = resolve_ref(
                results,
                *target,
                source_context,
                graph,
                refs,
                source_interner,
            )?;
            refs.insert(ref_key, node_id);
            Ok(node_id)
        }
        SolverResolutionNode::None => materialize_node(
            node_ref,
            source_context,
            graph,
            refs,
            source_interner,
            |_, _, _, _| Ok(ExecutionNode::None),
        ),
        SolverResolutionNode::Constant { source } => {
            let exact = source_context.resolve_or_allocate(source, source_interner);
            let source_node_id = source_interner.intern(&exact, graph);
            refs.insert(ref_key, source_node_id.node_id());
            Ok(source_node_id.node_id())
        }
        SolverResolutionNode::Property {
            source,
            property_name,
        } => materialize_node(
            node_ref,
            source_context,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner, source_context| {
                Ok(ExecutionNode::Property {
                    source: resolve_ref(
                        results,
                        *source,
                        source_context,
                        graph,
                        refs,
                        source_interner,
                    )?,
                    property_name: property_name.clone(),
                })
            },
        ),
        SolverResolutionNode::LazyRef { target } => materialize_node(
            node_ref,
            source_context,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner, source_context| {
                Ok(ExecutionNode::LazyRef {
                    target: resolve_ref(
                        results,
                        *target,
                        source_context,
                        graph,
                        refs,
                        source_interner,
                    )?,
                })
            },
        ),
        SolverResolutionNode::Protocol { members } => materialize_node(
            node_ref,
            source_context,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner, source_context| {
                Ok(ExecutionNode::Protocol {
                    members: members
                        .iter()
                        .map(|(name, &member_ref)| {
                            resolve_ref(
                                results,
                                member_ref,
                                source_context,
                                graph,
                                refs,
                                source_interner,
                            )
                            .map(|node_id| (name.clone(), node_id))
                        })
                        .collect::<Result<_, _>>()?,
                })
            },
        ),
        SolverResolutionNode::TypedDict { members } => materialize_node(
            node_ref,
            source_context,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner, source_context| {
                Ok(ExecutionNode::TypedDict {
                    members: members
                        .iter()
                        .map(|(name, &member_ref)| {
                            resolve_ref(
                                results,
                                member_ref,
                                source_context,
                                graph,
                                refs,
                                source_interner,
                            )
                            .map(|node_id| (name.clone(), node_id))
                        })
                        .collect::<Result<_, _>>()?,
                })
            },
        ),
        SolverResolutionNode::Method {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            implementations,
            target,
        } => materialize_node(
            node_ref,
            source_context,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner, source_context| {
                let mut body_context = source_context.clone();
                let execution_params = params
                    .iter()
                    .map(|param| {
                        ExecutionParam::from_method_param(
                            param,
                            &mut body_context,
                            source_interner,
                            graph,
                        )
                    })
                    .collect();
                let implementations = convert_method_implementations(
                    results,
                    implementations,
                    &mut body_context,
                    graph,
                    refs,
                    source_interner,
                )?;
                let target = resolve_ref(
                    results,
                    *target,
                    &mut body_context,
                    graph,
                    refs,
                    source_interner,
                )?;
                Ok(ExecutionNode::Method {
                    return_wrapper: *return_wrapper,
                    accepts_varargs: *accepts_varargs,
                    accepts_varkw: *accepts_varkw,
                    params: execution_params,
                    implementations,
                    target,
                })
            },
        ),
        SolverResolutionNode::AutoMethod {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            target,
        } => materialize_node(
            node_ref,
            source_context,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner, source_context| {
                let mut body_context = source_context.clone();
                let execution_params = params
                    .iter()
                    .map(|param| {
                        ExecutionParam::from_method_param(
                            param,
                            &mut body_context,
                            source_interner,
                            graph,
                        )
                    })
                    .collect();
                let target = resolve_ref(
                    results,
                    *target,
                    &mut body_context,
                    graph,
                    refs,
                    source_interner,
                )?;
                Ok(ExecutionNode::AutoMethod {
                    return_wrapper: *return_wrapper,
                    accepts_varargs: *accepts_varargs,
                    accepts_varkw: *accepts_varkw,
                    params: execution_params,
                    target,
                })
            },
        ),
        SolverResolutionNode::Attribute {
            source,
            attribute_name,
            access_kind,
        } => materialize_node(
            node_ref,
            source_context,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner, source_context| {
                Ok(ExecutionNode::Attribute {
                    source: resolve_ref(
                        results,
                        *source,
                        source_context,
                        graph,
                        refs,
                        source_interner,
                    )?,
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
            source_context,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner, source_context| {
                Ok(ExecutionNode::Constructor {
                    implementation: Arc::clone(&implementation.implementation),
                    params: params
                        .iter()
                        .map(|(param_ref, name, kind)| {
                            resolve_ref(
                                results,
                                *param_ref,
                                source_context,
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
                })
            },
        ),
    }
}

fn materialize_node<'ty>(
    node_ref: SolverResolutionRef,
    source_context: &mut SourceContext<'ty>,
    graph: &mut BuildExecutionGraph,
    refs: &mut HashMap<(SolverResolutionRef, SourceContext<'ty>), ExecutionNodeId>,
    source_interner: &mut SourceNodeInterner<'ty>,
    build_node: impl FnOnce(
        &mut BuildExecutionGraph,
        &mut HashMap<(SolverResolutionRef, SourceContext<'ty>), ExecutionNodeId>,
        &mut SourceNodeInterner<'ty>,
        &mut SourceContext<'ty>,
    ) -> Result<ExecutionNode, ResolutionError<'ty>>,
) -> Result<ExecutionNodeId, ResolutionError<'ty>> {
    let ref_key = (node_ref, source_context.clone());
    let node_id = graph.insert(BuildExecutionEntry::pending());
    refs.insert(ref_key, node_id);
    graph[node_id].node =
        BuildExecutionNode::Ready(build_node(graph, refs, source_interner, source_context)?);
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

fn convert_method_implementations<'ty>(
    results: &SolverResolutionArena<'ty>,
    implementations: &[SolverResolvedMethodImplementation<'ty>],
    source_context: &mut SourceContext<'ty>,
    graph: &mut BuildExecutionGraph,
    refs: &mut HashMap<(SolverResolutionRef, SourceContext<'ty>), ExecutionNodeId>,
    source_interner: &mut SourceNodeInterner<'ty>,
) -> Result<Vec<ExecutionMethodImplementation>, ResolutionError<'ty>> {
    let mut converted = Vec::with_capacity(implementations.len());
    for implementation in implementations {
        let bound_to = implementation
            .bound_to
            .map(|node_ref| {
                resolve_ref(
                    results,
                    node_ref,
                    source_context,
                    graph,
                    refs,
                    source_interner,
                )
            })
            .transpose()?;
        let params = implementation
            .params
            .iter()
            .map(|(node_ref, name, kind)| {
                resolve_ref(
                    results,
                    *node_ref,
                    source_context,
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
            .collect::<Result<_, _>>()?;
        let result_source = implementation.result_source.as_ref().map(|source| {
            let exact = source_context.allocate_transition(source, source_interner);
            source_interner.intern(&exact, graph)
        });

        converted.push(ExecutionMethodImplementation {
            implementation: Arc::clone(&implementation.implementation.implementation),
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
    let node_ids: Vec<ExecutionNodeId> = graph.keys().collect();
    let node_index: HashMap<ExecutionNodeId, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(index, &node_id)| (node_id, index))
        .collect();
    let identity_nodes: Vec<IdentityNode> = node_ids
        .iter()
        .enumerate()
        .map(|(node_identity, &node_id)| identity_node(graph[node_id].ready_node(), node_identity))
        .collect();
    let node_classes = compute_node_classes(&identity_nodes, &node_index);
    let representatives = class_representatives(&node_ids, &node_classes);

    let canonical_node_ids_by_class: Vec<ExecutionNodeId> = (0..representatives.len())
        .map(ExecutionNodeId::from_index)
        .collect();
    let entries = representatives
        .iter()
        .map(|&representative| ExecutionEntry {
            node: remap_node_refs_to_canonical_ids(
                graph[representative].ready_node(),
                &node_index,
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

    let root = canonical_id(
        root,
        &node_index,
        &node_classes,
        &canonical_node_ids_by_class,
    );
    (canonical, root)
}

fn identity_node(node: &ExecutionNode, node_identity: usize) -> IdentityNode {
    match node {
        ExecutionNode::Constant => IdentityNode {
            label: ExecutionIdentityLabel::Constant(node_identity),
            children: Vec::new(),
        },
        ExecutionNode::Property {
            source,
            property_name,
        } => IdentityNode {
            label: ExecutionIdentityLabel::Property(Arc::clone(property_name)),
            children: vec![*source],
        },
        ExecutionNode::LazyRef { target } => IdentityNode {
            label: ExecutionIdentityLabel::LazyRef,
            children: vec![*target],
        },
        ExecutionNode::None => IdentityNode {
            label: ExecutionIdentityLabel::None,
            children: Vec::new(),
        },
        ExecutionNode::Protocol { members } => IdentityNode {
            label: ExecutionIdentityLabel::Protocol {
                members: member_names(members),
            },
            children: members.values().copied().collect(),
        },
        ExecutionNode::TypedDict { members } => IdentityNode {
            label: ExecutionIdentityLabel::TypedDict {
                members: member_names(members),
            },
            children: members.values().copied().collect(),
        },
        ExecutionNode::Method {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            implementations,
            target,
        } => IdentityNode {
            label: ExecutionIdentityLabel::Method {
                return_wrapper: *return_wrapper,
                accepts_varargs: *accepts_varargs,
                accepts_varkw: *accepts_varkw,
                params: execution_param_labels(params),
                implementations: method_implementation_labels(implementations),
            },
            children: transition_children(*target, implementations),
        },
        ExecutionNode::AutoMethod {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            target,
        } => IdentityNode {
            label: ExecutionIdentityLabel::AutoMethod {
                return_wrapper: *return_wrapper,
                accepts_varargs: *accepts_varargs,
                accepts_varkw: *accepts_varkw,
                params: execution_param_labels(params),
            },
            children: vec![*target],
        },
        ExecutionNode::Attribute {
            source,
            attribute_name,
            access_kind,
        } => IdentityNode {
            label: ExecutionIdentityLabel::Attribute {
                name: Arc::clone(attribute_name),
                access_kind: *access_kind,
            },
            children: vec![*source],
        },
        ExecutionNode::Constructor {
            implementation,
            params,
        } => IdentityNode {
            label: ExecutionIdentityLabel::Constructor {
                implementation: py_identity(implementation),
                params: constructor_param_labels(params),
            },
            children: params.iter().map(|param| param.node).collect(),
        },
    }
}

fn compute_node_classes(
    identity_nodes: &[IdentityNode],
    node_index: &HashMap<ExecutionNodeId, usize>,
) -> Vec<usize> {
    let mut classes = vec![0; identity_nodes.len()];

    loop {
        let mut signatures = Vec::new();
        let mut next_classes = Vec::with_capacity(identity_nodes.len());

        for node in identity_nodes {
            let signature = IdentitySignature {
                label: node.label.clone(),
                children: node
                    .children
                    .iter()
                    .map(|child| classes[*node_index.get(child).expect("child node must exist")])
                    .collect(),
            };
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

fn class_representatives(
    node_ids: &[ExecutionNodeId],
    node_classes: &[usize],
) -> Vec<ExecutionNodeId> {
    let mut representatives = Vec::new();
    for (index, &node_id) in node_ids.iter().enumerate() {
        if node_classes[index] == representatives.len() {
            representatives.push(node_id);
        }
    }
    representatives
}

fn remap_node_refs_to_canonical_ids(
    node: &ExecutionNode,
    node_index: &HashMap<ExecutionNodeId, usize>,
    node_classes: &[usize],
    canonical_node_ids_by_class: &[ExecutionNodeId],
) -> ExecutionNode {
    match node {
        ExecutionNode::Constant => ExecutionNode::Constant,
        ExecutionNode::Property {
            source,
            property_name,
        } => ExecutionNode::Property {
            source: canonical_id(
                *source,
                node_index,
                node_classes,
                canonical_node_ids_by_class,
            ),
            property_name: Arc::clone(property_name),
        },
        ExecutionNode::LazyRef { target } => ExecutionNode::LazyRef {
            target: canonical_id(
                *target,
                node_index,
                node_classes,
                canonical_node_ids_by_class,
            ),
        },
        ExecutionNode::None => ExecutionNode::None,
        ExecutionNode::Protocol { members } => ExecutionNode::Protocol {
            members: members
                .iter()
                .map(|(name, &node_id)| {
                    (
                        Arc::clone(name),
                        canonical_id(
                            node_id,
                            node_index,
                            node_classes,
                            canonical_node_ids_by_class,
                        ),
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
                        canonical_id(
                            node_id,
                            node_index,
                            node_classes,
                            canonical_node_ids_by_class,
                        ),
                    )
                })
                .collect(),
        },
        ExecutionNode::Method {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            implementations,
            target,
        } => ExecutionNode::Method {
            return_wrapper: *return_wrapper,
            accepts_varargs: *accepts_varargs,
            accepts_varkw: *accepts_varkw,
            params: remap_execution_params(
                params,
                node_index,
                node_classes,
                canonical_node_ids_by_class,
            ),
            implementations: remap_method_implementations(
                implementations,
                node_index,
                node_classes,
                canonical_node_ids_by_class,
            ),
            target: canonical_id(
                *target,
                node_index,
                node_classes,
                canonical_node_ids_by_class,
            ),
        },
        ExecutionNode::AutoMethod {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            target,
        } => ExecutionNode::AutoMethod {
            return_wrapper: *return_wrapper,
            accepts_varargs: *accepts_varargs,
            accepts_varkw: *accepts_varkw,
            params: remap_execution_params(
                params,
                node_index,
                node_classes,
                canonical_node_ids_by_class,
            ),
            target: canonical_id(
                *target,
                node_index,
                node_classes,
                canonical_node_ids_by_class,
            ),
        },
        ExecutionNode::Attribute {
            source,
            attribute_name,
            access_kind,
        } => ExecutionNode::Attribute {
            source: canonical_id(
                *source,
                node_index,
                node_classes,
                canonical_node_ids_by_class,
            ),
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
                    node: canonical_id(
                        param.node,
                        node_index,
                        node_classes,
                        canonical_node_ids_by_class,
                    ),
                })
                .collect(),
        },
    }
}

fn remap_execution_params(
    params: &[ExecutionParam],
    node_index: &HashMap<ExecutionNodeId, usize>,
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
                    canonical_source_node_id(
                        source,
                        node_index,
                        node_classes,
                        canonical_node_ids_by_class,
                    )
                })
                .collect(),
        })
        .collect()
}

fn canonical_source_node_id(
    source: ExecutionSourceNodeId,
    node_index: &HashMap<ExecutionNodeId, usize>,
    node_classes: &[usize],
    canonical_node_ids_by_class: &[ExecutionNodeId],
) -> ExecutionSourceNodeId {
    ExecutionSourceNodeId(canonical_id(
        source.node_id(),
        node_index,
        node_classes,
        canonical_node_ids_by_class,
    ))
}

fn remap_method_implementations(
    implementations: &[ExecutionMethodImplementation],
    node_index: &HashMap<ExecutionNodeId, usize>,
    node_classes: &[usize],
    canonical_node_ids_by_class: &[ExecutionNodeId],
) -> Vec<ExecutionMethodImplementation> {
    implementations
        .iter()
        .map(|implementation| ExecutionMethodImplementation {
            implementation: Arc::clone(&implementation.implementation),
            bound_to: implementation.bound_to.map(|node_id| {
                canonical_id(
                    node_id,
                    node_index,
                    node_classes,
                    canonical_node_ids_by_class,
                )
            }),
            params: implementation
                .params
                .iter()
                .map(|param| ConstructorParam {
                    name: Arc::clone(&param.name),
                    kind: param.kind,
                    node: canonical_id(
                        param.node,
                        node_index,
                        node_classes,
                        canonical_node_ids_by_class,
                    ),
                })
                .collect(),
            return_wrapper: implementation.return_wrapper,
            result_source: implementation.result_source.map(|source| {
                canonical_source_node_id(
                    source,
                    node_index,
                    node_classes,
                    canonical_node_ids_by_class,
                )
            }),
        })
        .collect()
}

fn canonical_id(
    node_id: ExecutionNodeId,
    node_index: &HashMap<ExecutionNodeId, usize>,
    node_classes: &[usize],
    canonical_node_ids_by_class: &[ExecutionNodeId],
) -> ExecutionNodeId {
    canonical_node_ids_by_class
        [node_classes[*node_index.get(&node_id).expect("child node must exist")]]
}

fn compute_source_deps(
    graph: &ExecutionGraph,
) -> HashMap<ExecutionNodeId, HashSet<ExecutionSourceNodeId>> {
    let node_ids: Vec<ExecutionNodeId> = graph.keys().collect();
    let mut deps: HashMap<ExecutionNodeId, HashSet<ExecutionSourceNodeId>> = node_ids
        .iter()
        .map(|&node_id| {
            let deps = match &graph[node_id].node {
                ExecutionNode::Constant => HashSet::from([ExecutionSourceNodeId(node_id)]),
                ExecutionNode::Property { .. }
                | ExecutionNode::LazyRef { .. }
                | ExecutionNode::None
                | ExecutionNode::Protocol { .. }
                | ExecutionNode::TypedDict { .. }
                | ExecutionNode::Method { .. }
                | ExecutionNode::AutoMethod { .. }
                | ExecutionNode::Attribute { .. }
                | ExecutionNode::Constructor { .. } => HashSet::new(),
            };
            (node_id, deps)
        })
        .collect();

    let mut changed = true;
    while changed {
        changed = false;
        for &node_id in &node_ids {
            let mut next = deps[&node_id].clone();
            for child in source_dep_children(&graph[node_id].node) {
                next.extend(deps[&child].iter().copied());
            }
            if next != deps[&node_id] {
                deps.insert(node_id, next);
                changed = true;
            }
        }
    }

    deps
}

fn source_dep_children(node: &ExecutionNode) -> Vec<ExecutionNodeId> {
    match node {
        ExecutionNode::Constant | ExecutionNode::None | ExecutionNode::AutoMethod { .. } => {
            Vec::new()
        }
        ExecutionNode::Property { source, .. } | ExecutionNode::Attribute { source, .. } => {
            vec![*source]
        }
        ExecutionNode::LazyRef { target } => vec![*target],
        ExecutionNode::Protocol { members } | ExecutionNode::TypedDict { members } => {
            members.values().copied().collect()
        }
        ExecutionNode::Method {
            implementations, ..
        } => implementations
            .iter()
            .flat_map(|implementation| implementation.bound_to.iter().copied())
            .collect(),
        ExecutionNode::Constructor { params, .. } => {
            params.iter().map(|param| param.node).collect()
        }
    }
}

pub(crate) fn resource_plan_for_node(
    graph: &ExecutionGraph,
    node_id: ExecutionNodeId,
    unavailable_sources: &HashSet<ExecutionSourceNodeId>,
) -> ResourcePlan {
    let mut plan = ResourcePlan::default();
    collect_resource_plan(
        graph,
        node_id,
        unavailable_sources,
        &mut HashSet::new(),
        &mut plan,
    );
    plan
}

pub(crate) fn resource_plan_for_roots(
    graph: &ExecutionGraph,
    roots: impl IntoIterator<Item = ExecutionNodeId>,
    unavailable_sources: &HashSet<ExecutionSourceNodeId>,
) -> ResourcePlan {
    let mut plan = ResourcePlan::default();
    let mut stack = HashSet::new();
    for root in roots {
        collect_resource_plan(graph, root, unavailable_sources, &mut stack, &mut plan);
    }
    plan
}

pub(crate) fn transition_introduced_sources(
    params: &[ExecutionParam],
    implementations: &[ExecutionMethodImplementation],
) -> HashSet<ExecutionSourceNodeId> {
    params
        .iter()
        .flat_map(|param| param.sources.iter().copied())
        .chain(
            implementations
                .iter()
                .filter_map(|implementation| implementation.result_source),
        )
        .collect()
}

pub(crate) fn transition_body_roots(
    target: ExecutionNodeId,
    implementations: &[ExecutionMethodImplementation],
) -> impl Iterator<Item = ExecutionNodeId> + '_ {
    implementations
        .iter()
        .flat_map(|implementation| {
            implementation
                .bound_to
                .into_iter()
                .chain(implementation.params.iter().map(|param| param.node))
        })
        .chain(std::iter::once(target))
}

fn collect_resource_plan(
    graph: &ExecutionGraph,
    node_id: ExecutionNodeId,
    unavailable_sources: &HashSet<ExecutionSourceNodeId>,
    stack: &mut HashSet<ExecutionNodeId>,
    plan: &mut ResourcePlan,
) {
    if !stack.insert(node_id) {
        return;
    }

    match &graph[node_id].node {
        ExecutionNode::Constant => {
            let source = ExecutionSourceNodeId(node_id);
            if !unavailable_sources.contains(&source) {
                plan.sources.insert(source);
            }
        }
        ExecutionNode::None => {}
        ExecutionNode::Property { source, .. } | ExecutionNode::Attribute { source, .. } => {
            collect_resource_plan(graph, *source, unavailable_sources, stack, plan);
        }
        ExecutionNode::LazyRef { target } => {
            collect_resource_plan(graph, *target, unavailable_sources, stack, plan);
        }
        ExecutionNode::Protocol { members } | ExecutionNode::TypedDict { members } => {
            for &member in members.values() {
                collect_resource_plan(graph, member, unavailable_sources, stack, plan);
            }
        }
        ExecutionNode::Method {
            params,
            implementations,
            target,
            ..
        } => {
            let introduced = transition_introduced_sources(params, implementations);
            let mut call_unavailable = unavailable_sources.clone();
            call_unavailable.extend(introduced);
            for root in transition_body_roots(*target, implementations) {
                collect_resource_plan(graph, root, &call_unavailable, stack, plan);
            }
        }
        ExecutionNode::AutoMethod { params, target, .. } => {
            let introduced = params
                .iter()
                .flat_map(|param| param.sources.iter().copied())
                .collect::<HashSet<_>>();
            let mut call_unavailable = unavailable_sources.clone();
            call_unavailable.extend(introduced);
            collect_resource_plan(graph, *target, &call_unavailable, stack, plan);
        }
        ExecutionNode::Constructor { params, .. } => {
            if graph[node_id].source_deps.is_disjoint(unavailable_sources) {
                plan.caches.insert(node_id);
            }
            for param in params {
                collect_resource_plan(graph, param.node, unavailable_sources, stack, plan);
            }
        }
    }

    stack.remove(&node_id);
}

fn constructor_param_labels(params: &[ConstructorParam]) -> Vec<ConstructorParamLabel> {
    params
        .iter()
        .map(|param| ConstructorParamLabel {
            name: Arc::clone(&param.name),
            kind: param.kind,
        })
        .collect()
}

fn execution_param_labels(params: &[ExecutionParam]) -> Vec<ExecutionParamLabel> {
    params
        .iter()
        .map(|param| ExecutionParamLabel {
            name: Arc::clone(&param.name),
            kind: param.kind,
            sources: param.sources.clone(),
        })
        .collect()
}

fn method_implementation_labels(
    implementations: &[ExecutionMethodImplementation],
) -> Vec<MethodImplementationLabel> {
    implementations
        .iter()
        .map(|implementation| MethodImplementationLabel {
            implementation: py_identity(&implementation.implementation),
            has_bound_to: implementation.bound_to.is_some(),
            params: constructor_param_labels(&implementation.params),
            return_wrapper: implementation.return_wrapper,
            result_source: implementation.result_source,
        })
        .collect()
}

fn transition_children(
    target: ExecutionNodeId,
    implementations: &[ExecutionMethodImplementation],
) -> Vec<ExecutionNodeId> {
    transition_body_roots(target, implementations).collect()
}

fn member_names(members: &BTreeMap<Arc<str>, ExecutionNodeId>) -> Vec<Arc<str>> {
    members.keys().cloned().collect()
}

fn py_identity(value: &Arc<Py<PyAny>>) -> PythonIdentity {
    PythonIdentity::from_arc_py_any(value)
}

#[cfg(test)]
mod tests {
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

    fn with_target_type<R>(run: impl for<'ty> FnOnce(PyTypeConcreteKey<'ty>) -> R) -> R {
        let mut arenas = TypeArenas::default();
        let key = arenas.concrete.plains.insert(Qualified {
            inner: PlainType::<Qual<Keyed>, Concrete> {
                descriptor: PyTypeDescriptor {
                    id: PyTypeId::new("Target".to_string()),
                    display_name: Arc::from("Target"),
                },
                args: Vec::new(),
            },
            qualifier: Qualifier::any(),
        });
        run(PyType::Plain(key))
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

    fn constructor(implementation: Arc<Py<PyAny>>, params: Vec<ConstructorParam>) -> ExecutionNode {
        ExecutionNode::Constructor {
            implementation,
            params,
        }
    }

    #[test]
    fn delegate_alias_does_not_materialize_execution_node() {
        with_target_type(|target_type| {
            let mut results = SolverResolutionArena::default();
            let target = results.insert(Ok(SolverResolvedNode {
                target_type,
                resolution: SolverResolutionNode::None,
            }));
            let root = results.insert(Ok(SolverResolvedNode {
                target_type,
                resolution: SolverResolutionNode::Delegate(target),
            }));

            let (graph, root_node) = flatten(results, root).expect("flatten");

            assert_eq!(graph.len(), 1);
            assert!(matches!(&graph[root_node].node, ExecutionNode::None));
        });
    }

    #[test]
    fn union_variant_alias_does_not_materialize_execution_node() {
        with_target_type(|target_type| {
            let mut results = SolverResolutionArena::default();
            let target = results.insert(Ok(SolverResolvedNode {
                target_type,
                resolution: SolverResolutionNode::None,
            }));
            let root = results.insert(Ok(SolverResolvedNode {
                target_type,
                resolution: SolverResolutionNode::UnionVariant { target },
            }));

            let (graph, root_node) = flatten(results, root).expect("flatten");

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
        let left = graph.insert(entry(ExecutionNode::AutoMethod {
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            params: Vec::new(),
            target: left_target,
        }));
        graph.insert(entry(ExecutionNode::AutoMethod {
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            params: Vec::new(),
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
        let method_impl = |node| ExecutionMethodImplementation {
            implementation: Arc::clone(&implementation),
            bound_to: None,
            params: vec![constructor_param("audit", node)],
            return_wrapper: WrapperKind::None,
            result_source: None,
        };
        let left = graph.insert(entry(ExecutionNode::Method {
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            params: Vec::new(),
            implementations: vec![method_impl(left_param)],
            target,
        }));
        graph.insert(entry(ExecutionNode::Method {
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            params: Vec::new(),
            implementations: vec![method_impl(right_param)],
            target,
        }));

        let (graph, _) = canonicalize_execution_graph(graph, left);

        assert_eq!(graph.len(), 5);
    }

    #[test]
    fn method_implementation_bound_instance_is_dependency_but_transition_target_is_not() {
        let mut graph = BuildExecutionGraph::default();
        let bound = graph.insert(entry(ExecutionNode::Constant));
        let target = graph.insert(entry(ExecutionNode::Constant));
        let method = graph.insert(entry(ExecutionNode::Method {
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            params: Vec::new(),
            implementations: vec![ExecutionMethodImplementation {
                implementation: py_object(),
                bound_to: Some(bound),
                params: Vec::new(),
                return_wrapper: WrapperKind::None,
                result_source: None,
            }],
            target,
        }));

        let (graph, method) = canonicalize_execution_graph(graph, method);
        let bound_source = match &graph[method].node {
            ExecutionNode::Method {
                implementations, ..
            } if implementations.len() == 1 => match implementations[0].bound_to {
                Some(bound) => ExecutionSourceNodeId(bound),
                None => panic!("expected method implementation with bound source"),
            },
            _ => panic!("expected method with one implementation"),
        };

        assert_eq!(graph[method].source_deps, HashSet::from([bound_source]));
    }

    #[test]
    fn auto_method_target_is_not_a_source_dependency() {
        let mut graph = BuildExecutionGraph::default();
        let target = graph.insert(entry(ExecutionNode::Constant));
        let auto_method = graph.insert(entry(ExecutionNode::AutoMethod {
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            params: Vec::new(),
            target,
        }));

        let (graph, auto_method) = canonicalize_execution_graph(graph, auto_method);

        assert!(graph[auto_method].source_deps.is_empty());
    }
}
