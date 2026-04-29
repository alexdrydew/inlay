use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

use context_solver::Arena as ResultsArena;
use inlay_instrument_macros::instrumented;
use slotmap::{SlotMap, new_key_type};

use pyo3::PyTraverseError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;

use crate::{
    registry::Source,
    rules::{
        MethodParam, ResolutionError, SolverResolutionArena, SolverResolutionNode,
        SolverResolutionRef, SolverResolvedHook, SolverResolvedNode,
    },
    types::{ParamKind, WrapperKind},
};

new_key_type! {
    pub(crate) struct ExecutionNodeId;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ExecutionSourceNodeId(ExecutionNodeId);

impl ExecutionSourceNodeId {
    pub(crate) fn node_id(self) -> ExecutionNodeId {
        self.0
    }
}

#[derive(Default)]
struct SourceNodeInterner {
    sources: HashMap<Source, ExecutionSourceNodeId>,
}

impl SourceNodeInterner {
    fn intern(&mut self, source: &Source, graph: &mut ExecutionGraph) -> ExecutionSourceNodeId {
        if let Some(source_node_id) = self.sources.get(source) {
            return *source_node_id;
        }

        let node_id = graph.insert(ExecutionEntry {
            node: ExecutionNode::Constant,
            source_deps: HashSet::new(),
        });
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
    pub(crate) source: ExecutionSourceNodeId,
}

impl ExecutionParam {
    fn from_method_param(
        param: &MethodParam,
        source_interner: &mut SourceNodeInterner,
        graph: &mut ExecutionGraph,
    ) -> Self {
        Self {
            name: Arc::clone(&param.name),
            kind: param.kind,
            source: source_interner.intern(&param.source, graph),
        }
    }
}

#[derive(Clone)]
pub(crate) struct ExecutionResultBinding {
    pub(crate) name: Arc<str>,
    pub(crate) source: ExecutionSourceNodeId,
}

impl ExecutionResultBinding {
    fn from_transition_binding(
        binding: &crate::rules::TransitionResultBinding,
        source_interner: &mut SourceNodeInterner,
        graph: &mut ExecutionGraph,
    ) -> Self {
        Self {
            name: Arc::clone(&binding.name),
            source: source_interner.intern(&binding.source, graph),
        }
    }
}

#[derive(Clone)]
pub(crate) struct ExecutionHook {
    pub(crate) implementation: Arc<Py<PyAny>>,
    pub(crate) params: Vec<ConstructorParam>,
}

impl ExecutionHook {
    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&*self.implementation)
    }
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
    source: ExecutionSourceNodeId,
}

#[derive(Clone, PartialEq, Eq)]
struct ResultBindingLabel {
    name: Arc<str>,
    source: ExecutionSourceNodeId,
}

#[derive(Clone, PartialEq, Eq)]
struct HookParamLabel {
    name: Arc<str>,
    kind: ParamKind,
}

#[derive(Clone, PartialEq, Eq)]
struct HookLabel {
    implementation: usize,
    params: Vec<HookParamLabel>,
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
        implementation: usize,
        return_wrapper: WrapperKind,
        accepts_varargs: bool,
        accepts_varkw: bool,
        has_bound_to: bool,
        params: Vec<ExecutionParamLabel>,
        result_source: ExecutionSourceNodeId,
        result_bindings: Vec<ResultBindingLabel>,
        hooks: Vec<HookLabel>,
    },
    AutoMethod {
        return_wrapper: WrapperKind,
        accepts_varargs: bool,
        accepts_varkw: bool,
        params: Vec<ExecutionParamLabel>,
        hooks: Vec<HookLabel>,
    },
    Attribute(Arc<str>),
    Constructor {
        implementation: usize,
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
        implementation: Arc<Py<PyAny>>,
        return_wrapper: WrapperKind,
        accepts_varargs: bool,
        accepts_varkw: bool,
        bound_to: Option<ExecutionNodeId>,
        params: Vec<ExecutionParam>,
        result_source: ExecutionSourceNodeId,
        result_bindings: Vec<ExecutionResultBinding>,
        target: ExecutionNodeId,
        hooks: Vec<ExecutionHook>,
    },
    AutoMethod {
        return_wrapper: WrapperKind,
        accepts_varargs: bool,
        accepts_varkw: bool,
        params: Vec<ExecutionParam>,
        target: ExecutionNodeId,
        hooks: Vec<ExecutionHook>,
    },
    Attribute {
        source: ExecutionNodeId,
        attribute_name: Arc<str>,
    },
    Constructor {
        implementation: Arc<Py<PyAny>>,
        params: Vec<ConstructorParam>,
    },
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

pub(crate) type ExecutionGraph = SlotMap<ExecutionNodeId, ExecutionEntry>;

#[instrumented(name = "inlay.flatten", target = "inlay", level = "trace")]
pub(crate) fn flatten(
    results: SolverResolutionArena,
    root: SolverResolutionRef,
) -> Result<(ExecutionGraph, ExecutionNodeId, usize), ResolutionError> {
    let mut graph: ExecutionGraph = SlotMap::with_key();
    let mut refs = HashMap::new();
    let mut source_interner = SourceNodeInterner::default();
    let root = resolve_ref(&results, root, &mut graph, &mut refs, &mut source_interner)?;
    let reachable_result_refs = refs.len();
    let (graph, root) = canonicalize_execution_graph(graph, root);
    Ok((graph, root, reachable_result_refs))
}

fn resolve_ref(
    results: &SolverResolutionArena,
    node_ref: SolverResolutionRef,
    graph: &mut ExecutionGraph,
    refs: &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
    source_interner: &mut SourceNodeInterner,
) -> Result<ExecutionNodeId, ResolutionError> {
    if let Some(&node_id) = refs.get(&node_ref) {
        return Ok(node_id);
    }

    let resolved = get_resolved_node(results, node_ref)?;
    match &resolved.resolution {
        SolverResolutionNode::Delegate(target) | SolverResolutionNode::UnionVariant { target } => {
            let node_id = resolve_ref(results, *target, graph, refs, source_interner)?;
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
                    source: resolve_ref(results, *source, graph, refs, source_interner)?,
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
                    target: resolve_ref(results, *target, graph, refs, source_interner)?,
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
                            resolve_ref(results, member_ref, graph, refs, source_interner)
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
                            resolve_ref(results, member_ref, graph, refs, source_interner)
                                .map(|node_id| (name.clone(), node_id))
                        })
                        .collect::<Result<_, _>>()?,
                })
            },
        ),
        SolverResolutionNode::Method {
            implementation,
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            bound_to,
            params,
            result_source,
            result_bindings,
            target,
            hooks,
        } => materialize_node(
            node_ref,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner| {
                Ok(ExecutionNode::Method {
                    implementation: Arc::clone(&implementation.implementation),
                    return_wrapper: *return_wrapper,
                    accepts_varargs: *accepts_varargs,
                    accepts_varkw: *accepts_varkw,
                    bound_to: bound_to
                        .map(|node_ref| {
                            resolve_ref(results, node_ref, graph, refs, source_interner)
                        })
                        .transpose()?,
                    params: params
                        .iter()
                        .map(|param| {
                            ExecutionParam::from_method_param(param, source_interner, graph)
                        })
                        .collect(),
                    result_source: source_interner.intern(result_source, graph),
                    result_bindings: result_bindings
                        .iter()
                        .map(|binding| {
                            ExecutionResultBinding::from_transition_binding(
                                binding,
                                source_interner,
                                graph,
                            )
                        })
                        .collect(),
                    target: resolve_ref(results, *target, graph, refs, source_interner)?,
                    hooks: convert_hooks(results, hooks, graph, refs, source_interner)?,
                })
            },
        ),
        SolverResolutionNode::AutoMethod {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            target,
            hooks,
        } => materialize_node(
            node_ref,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner| {
                Ok(ExecutionNode::AutoMethod {
                    return_wrapper: *return_wrapper,
                    accepts_varargs: *accepts_varargs,
                    accepts_varkw: *accepts_varkw,
                    params: params
                        .iter()
                        .map(|param| {
                            ExecutionParam::from_method_param(param, source_interner, graph)
                        })
                        .collect(),
                    target: resolve_ref(results, *target, graph, refs, source_interner)?,
                    hooks: convert_hooks(results, hooks, graph, refs, source_interner)?,
                })
            },
        ),
        SolverResolutionNode::Attribute {
            source,
            attribute_name,
        } => materialize_node(
            node_ref,
            graph,
            refs,
            source_interner,
            |graph, refs, source_interner| {
                Ok(ExecutionNode::Attribute {
                    source: resolve_ref(results, *source, graph, refs, source_interner)?,
                    attribute_name: attribute_name.clone(),
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
                            resolve_ref(results, *param_ref, graph, refs, source_interner).map(
                                |node_id| ConstructorParam {
                                    name: name.clone(),
                                    kind: *kind,
                                    node: node_id,
                                },
                            )
                        })
                        .collect::<Result<_, _>>()?,
                })
            },
        ),
    }
}

fn materialize_node(
    node_ref: SolverResolutionRef,
    graph: &mut ExecutionGraph,
    refs: &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
    source_interner: &mut SourceNodeInterner,
    build_node: impl FnOnce(
        &mut ExecutionGraph,
        &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
        &mut SourceNodeInterner,
    ) -> Result<ExecutionNode, ResolutionError>,
) -> Result<ExecutionNodeId, ResolutionError> {
    let node_id = graph.insert(ExecutionEntry {
        node: ExecutionNode::None,
        source_deps: HashSet::new(),
    });
    refs.insert(node_ref, node_id);
    let node = build_node(graph, refs, source_interner)?;
    graph[node_id].node = node;
    Ok(node_id)
}

fn get_resolved_node(
    results: &SolverResolutionArena,
    node_ref: SolverResolutionRef,
) -> Result<&SolverResolvedNode, ResolutionError> {
    match results
        .get(&node_ref)
        .expect("solver result ref must point to a stored result")
    {
        Ok(node) => Ok(node),
        Err(err) => Err(err.clone()),
    }
}

fn convert_hooks(
    results: &SolverResolutionArena,
    hooks: &[SolverResolvedHook],
    graph: &mut ExecutionGraph,
    refs: &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
    source_interner: &mut SourceNodeInterner,
) -> Result<Vec<ExecutionHook>, ResolutionError> {
    hooks
        .iter()
        .map(|hook| {
            Ok(ExecutionHook {
                implementation: Arc::clone(&hook.hook.implementation),
                params: hook
                    .params
                    .iter()
                    .map(|(node_ref, name, kind)| {
                        resolve_ref(results, *node_ref, graph, refs, source_interner).map(
                            |node_id| ConstructorParam {
                                name: name.clone(),
                                kind: *kind,
                                node: node_id,
                            },
                        )
                    })
                    .collect::<Result<_, _>>()?,
            })
        })
        .collect()
}

fn canonicalize_execution_graph(
    graph: ExecutionGraph,
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
        .map(|(node_identity, &node_id)| identity_node(&graph[node_id].node, node_identity))
        .collect();
    let node_classes = compute_node_classes(&identity_nodes, &node_index);
    let representatives = class_representatives(&node_ids, &node_classes);

    let mut canonical = ExecutionGraph::with_key();
    let class_node_ids: Vec<ExecutionNodeId> = representatives
        .iter()
        .map(|_| {
            canonical.insert(ExecutionEntry {
                node: ExecutionNode::None,
                source_deps: HashSet::new(),
            })
        })
        .collect();

    for (class_id, &representative) in representatives.iter().enumerate() {
        canonical[class_node_ids[class_id]].node = remap_node(
            &graph[representative].node,
            &node_index,
            &node_classes,
            &class_node_ids,
        );
    }

    let source_deps = compute_source_deps(&canonical);
    for (node_id, deps) in source_deps {
        canonical[node_id].source_deps = deps;
    }

    let root = canonical_id(root, &node_index, &node_classes, &class_node_ids);
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
            implementation,
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            bound_to,
            params,
            result_source,
            result_bindings,
            target,
            hooks,
        } => IdentityNode {
            label: ExecutionIdentityLabel::Method {
                implementation: py_identity(implementation),
                return_wrapper: *return_wrapper,
                accepts_varargs: *accepts_varargs,
                accepts_varkw: *accepts_varkw,
                has_bound_to: bound_to.is_some(),
                params: execution_param_labels(params),
                result_source: *result_source,
                result_bindings: result_binding_labels(result_bindings),
                hooks: hook_labels(hooks),
            },
            children: transition_children(bound_to.iter().copied(), *target, hooks),
        },
        ExecutionNode::AutoMethod {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            target,
            hooks,
        } => IdentityNode {
            label: ExecutionIdentityLabel::AutoMethod {
                return_wrapper: *return_wrapper,
                accepts_varargs: *accepts_varargs,
                accepts_varkw: *accepts_varkw,
                params: execution_param_labels(params),
                hooks: hook_labels(hooks),
            },
            children: transition_children(std::iter::empty(), *target, hooks),
        },
        ExecutionNode::Attribute {
            source,
            attribute_name,
        } => IdentityNode {
            label: ExecutionIdentityLabel::Attribute(Arc::clone(attribute_name)),
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

fn remap_node(
    node: &ExecutionNode,
    node_index: &HashMap<ExecutionNodeId, usize>,
    node_classes: &[usize],
    class_node_ids: &[ExecutionNodeId],
) -> ExecutionNode {
    match node {
        ExecutionNode::Constant => ExecutionNode::Constant,
        ExecutionNode::Property {
            source,
            property_name,
        } => ExecutionNode::Property {
            source: canonical_id(*source, node_index, node_classes, class_node_ids),
            property_name: Arc::clone(property_name),
        },
        ExecutionNode::LazyRef { target } => ExecutionNode::LazyRef {
            target: canonical_id(*target, node_index, node_classes, class_node_ids),
        },
        ExecutionNode::None => ExecutionNode::None,
        ExecutionNode::Protocol { members } => ExecutionNode::Protocol {
            members: members
                .iter()
                .map(|(name, &node_id)| {
                    (
                        Arc::clone(name),
                        canonical_id(node_id, node_index, node_classes, class_node_ids),
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
                        canonical_id(node_id, node_index, node_classes, class_node_ids),
                    )
                })
                .collect(),
        },
        ExecutionNode::Method {
            implementation,
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            bound_to,
            params,
            result_source,
            result_bindings,
            target,
            hooks,
        } => ExecutionNode::Method {
            implementation: Arc::clone(implementation),
            return_wrapper: *return_wrapper,
            accepts_varargs: *accepts_varargs,
            accepts_varkw: *accepts_varkw,
            bound_to: bound_to
                .map(|node_id| canonical_id(node_id, node_index, node_classes, class_node_ids)),
            params: remap_execution_params(params, node_index, node_classes, class_node_ids),
            result_source: canonical_source_node_id(
                *result_source,
                node_index,
                node_classes,
                class_node_ids,
            ),
            result_bindings: remap_result_bindings(
                result_bindings,
                node_index,
                node_classes,
                class_node_ids,
            ),
            target: canonical_id(*target, node_index, node_classes, class_node_ids),
            hooks: remap_hooks(hooks, node_index, node_classes, class_node_ids),
        },
        ExecutionNode::AutoMethod {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            target,
            hooks,
        } => ExecutionNode::AutoMethod {
            return_wrapper: *return_wrapper,
            accepts_varargs: *accepts_varargs,
            accepts_varkw: *accepts_varkw,
            params: remap_execution_params(params, node_index, node_classes, class_node_ids),
            target: canonical_id(*target, node_index, node_classes, class_node_ids),
            hooks: remap_hooks(hooks, node_index, node_classes, class_node_ids),
        },
        ExecutionNode::Attribute {
            source,
            attribute_name,
        } => ExecutionNode::Attribute {
            source: canonical_id(*source, node_index, node_classes, class_node_ids),
            attribute_name: Arc::clone(attribute_name),
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
                    node: canonical_id(param.node, node_index, node_classes, class_node_ids),
                })
                .collect(),
        },
    }
}

fn remap_execution_params(
    params: &[ExecutionParam],
    node_index: &HashMap<ExecutionNodeId, usize>,
    node_classes: &[usize],
    class_node_ids: &[ExecutionNodeId],
) -> Vec<ExecutionParam> {
    params
        .iter()
        .map(|param| ExecutionParam {
            name: Arc::clone(&param.name),
            kind: param.kind,
            source: canonical_source_node_id(
                param.source,
                node_index,
                node_classes,
                class_node_ids,
            ),
        })
        .collect()
}

fn remap_result_bindings(
    bindings: &[ExecutionResultBinding],
    node_index: &HashMap<ExecutionNodeId, usize>,
    node_classes: &[usize],
    class_node_ids: &[ExecutionNodeId],
) -> Vec<ExecutionResultBinding> {
    bindings
        .iter()
        .map(|binding| ExecutionResultBinding {
            name: Arc::clone(&binding.name),
            source: canonical_source_node_id(
                binding.source,
                node_index,
                node_classes,
                class_node_ids,
            ),
        })
        .collect()
}

fn canonical_source_node_id(
    source: ExecutionSourceNodeId,
    node_index: &HashMap<ExecutionNodeId, usize>,
    node_classes: &[usize],
    class_node_ids: &[ExecutionNodeId],
) -> ExecutionSourceNodeId {
    ExecutionSourceNodeId(canonical_id(
        source.node_id(),
        node_index,
        node_classes,
        class_node_ids,
    ))
}

fn remap_hooks(
    hooks: &[ExecutionHook],
    node_index: &HashMap<ExecutionNodeId, usize>,
    node_classes: &[usize],
    class_node_ids: &[ExecutionNodeId],
) -> Vec<ExecutionHook> {
    hooks
        .iter()
        .map(|hook| ExecutionHook {
            implementation: Arc::clone(&hook.implementation),
            params: hook
                .params
                .iter()
                .map(|param| ConstructorParam {
                    name: Arc::clone(&param.name),
                    kind: param.kind,
                    node: canonical_id(param.node, node_index, node_classes, class_node_ids),
                })
                .collect(),
        })
        .collect()
}

fn canonical_id(
    node_id: ExecutionNodeId,
    node_index: &HashMap<ExecutionNodeId, usize>,
    node_classes: &[usize],
    class_node_ids: &[ExecutionNodeId],
) -> ExecutionNodeId {
    class_node_ids[node_classes[*node_index.get(&node_id).expect("child node must exist")]]
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
        ExecutionNode::Method { bound_to, .. } => bound_to.iter().copied().collect(),
        ExecutionNode::Constructor { params, .. } => {
            params.iter().map(|param| param.node).collect()
        }
    }
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
            source: param.source,
        })
        .collect()
}

fn result_binding_labels(bindings: &[ExecutionResultBinding]) -> Vec<ResultBindingLabel> {
    bindings
        .iter()
        .map(|binding| ResultBindingLabel {
            name: Arc::clone(&binding.name),
            source: binding.source,
        })
        .collect()
}

fn hook_labels(hooks: &[ExecutionHook]) -> Vec<HookLabel> {
    hooks
        .iter()
        .map(|hook| HookLabel {
            implementation: py_identity(&hook.implementation),
            params: hook
                .params
                .iter()
                .map(|param| HookParamLabel {
                    name: Arc::clone(&param.name),
                    kind: param.kind,
                })
                .collect(),
        })
        .collect()
}

fn transition_children(
    prefix: impl IntoIterator<Item = ExecutionNodeId>,
    target: ExecutionNodeId,
    hooks: &[ExecutionHook],
) -> Vec<ExecutionNodeId> {
    prefix
        .into_iter()
        .chain(std::iter::once(target))
        .chain(
            hooks
                .iter()
                .flat_map(|hook| hook.params.iter().map(|param| param.node)),
        )
        .collect()
}

fn member_names(members: &BTreeMap<Arc<str>, ExecutionNodeId>) -> Vec<Arc<str>> {
    members.keys().cloned().collect()
}

fn py_identity(value: &Arc<Py<PyAny>>) -> usize {
    value.as_ref().as_ptr() as usize
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
        Arena as _, Concrete, Keyed, PlainType, PyType, PyTypeConcreteKey, PyTypeDescriptor,
        PyTypeId, Qual, Qualified, TypeArenas,
    };

    fn target_type() -> PyTypeConcreteKey {
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
        PyType::Plain(key)
    }

    fn py_object() -> Arc<Py<PyAny>> {
        Python::initialize();
        Python::attach(|py| Arc::new(PyDict::new(py).into_any().unbind()))
    }

    fn entry(node: ExecutionNode) -> ExecutionEntry {
        ExecutionEntry {
            node,
            source_deps: HashSet::new(),
        }
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
        let target_type = target_type();
        let mut results = SolverResolutionArena::default();
        let target = results.insert(Ok(SolverResolvedNode {
            target_type,
            resolution: SolverResolutionNode::None,
        }));
        let root = results.insert(Ok(SolverResolvedNode {
            target_type,
            resolution: SolverResolutionNode::Delegate(target),
        }));

        let (graph, root_node, reachable_result_refs) = flatten(results, root).expect("flatten");

        assert_eq!(graph.len(), 1);
        assert_eq!(reachable_result_refs, 2);
        assert!(matches!(&graph[root_node].node, ExecutionNode::None));
    }

    #[test]
    fn union_variant_alias_does_not_materialize_execution_node() {
        let target_type = target_type();
        let mut results = SolverResolutionArena::default();
        let target = results.insert(Ok(SolverResolvedNode {
            target_type,
            resolution: SolverResolutionNode::None,
        }));
        let root = results.insert(Ok(SolverResolvedNode {
            target_type,
            resolution: SolverResolutionNode::UnionVariant { target },
        }));

        let (graph, root_node, reachable_result_refs) = flatten(results, root).expect("flatten");

        assert_eq!(graph.len(), 1);
        assert_eq!(reachable_result_refs, 2);
        assert!(matches!(&graph[root_node].node, ExecutionNode::None));
    }

    #[test]
    fn equivalent_constructor_nodes_are_canonicalized() {
        let mut graph = ExecutionGraph::with_key();
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
        let mut graph = ExecutionGraph::with_key();
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
        let mut graph = ExecutionGraph::with_key();
        let a_impl = py_object();
        let b_impl = py_object();
        let a_outer = graph.insert(entry(ExecutionNode::None));
        let lazy_b_outer = graph.insert(entry(ExecutionNode::None));
        let b = graph.insert(entry(ExecutionNode::None));
        let a_inner = graph.insert(entry(ExecutionNode::None));
        let lazy_b_inner = graph.insert(entry(ExecutionNode::None));

        graph[a_outer].node = constructor(
            Arc::clone(&a_impl),
            vec![constructor_param("b", lazy_b_outer)],
        );
        graph[lazy_b_outer].node = ExecutionNode::LazyRef { target: b };
        graph[b].node = constructor(Arc::clone(&b_impl), vec![constructor_param("a", a_inner)]);
        graph[a_inner].node = constructor(a_impl, vec![constructor_param("b", lazy_b_inner)]);
        graph[lazy_b_inner].node = ExecutionNode::LazyRef { target: b };

        let (graph, _) = canonicalize_execution_graph(graph, a_outer);

        assert_eq!(graph.len(), 3);
    }

    #[test]
    fn transition_target_is_part_of_execution_identity() {
        let mut graph = ExecutionGraph::with_key();
        let left_target = graph.insert(entry(ExecutionNode::Constant));
        let right_target = graph.insert(entry(ExecutionNode::Constant));
        let left = graph.insert(entry(ExecutionNode::AutoMethod {
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            params: Vec::new(),
            target: left_target,
            hooks: Vec::new(),
        }));
        graph.insert(entry(ExecutionNode::AutoMethod {
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            params: Vec::new(),
            target: right_target,
            hooks: Vec::new(),
        }));

        let (graph, _) = canonicalize_execution_graph(graph, left);

        assert_eq!(graph.len(), 4);
    }

    #[test]
    fn transition_hook_params_are_part_of_execution_identity() {
        let mut graph = ExecutionGraph::with_key();
        let target = graph.insert(entry(ExecutionNode::None));
        let left_param = graph.insert(entry(ExecutionNode::Constant));
        let right_param = graph.insert(entry(ExecutionNode::Constant));
        let hook_impl = py_object();
        let hook = |node| ExecutionHook {
            implementation: Arc::clone(&hook_impl),
            params: vec![constructor_param("audit", node)],
        };
        let left = graph.insert(entry(ExecutionNode::AutoMethod {
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            params: Vec::new(),
            target,
            hooks: vec![hook(left_param)],
        }));
        graph.insert(entry(ExecutionNode::AutoMethod {
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            params: Vec::new(),
            target,
            hooks: vec![hook(right_param)],
        }));

        let (graph, _) = canonicalize_execution_graph(graph, left);

        assert_eq!(graph.len(), 5);
    }

    #[test]
    fn method_bound_instance_is_dependency_but_transition_target_is_not() {
        let mut graph = ExecutionGraph::with_key();
        let bound = graph.insert(entry(ExecutionNode::Constant));
        let target = graph.insert(entry(ExecutionNode::Constant));
        let result_source = ExecutionSourceNodeId(graph.insert(entry(ExecutionNode::Constant)));
        let method = graph.insert(entry(ExecutionNode::Method {
            implementation: py_object(),
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            bound_to: Some(bound),
            params: Vec::new(),
            result_source,
            result_bindings: Vec::new(),
            target,
            hooks: Vec::new(),
        }));

        let (graph, method) = canonicalize_execution_graph(graph, method);
        let bound_source = match &graph[method].node {
            ExecutionNode::Method {
                bound_to: Some(bound),
                ..
            } => ExecutionSourceNodeId(*bound),
            _ => panic!("expected method with bound source"),
        };

        assert_eq!(graph[method].source_deps, HashSet::from([bound_source]));
    }

    #[test]
    fn auto_method_target_is_not_a_source_dependency() {
        let mut graph = ExecutionGraph::with_key();
        let target = graph.insert(entry(ExecutionNode::Constant));
        let auto_method = graph.insert(entry(ExecutionNode::AutoMethod {
            return_wrapper: WrapperKind::None,
            accepts_varargs: false,
            accepts_varkw: false,
            params: Vec::new(),
            target,
            hooks: Vec::new(),
        }));

        let (graph, auto_method) = canonicalize_execution_graph(graph, auto_method);

        assert!(graph[auto_method].source_deps.is_empty());
    }
}
