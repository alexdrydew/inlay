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
    registry::{Constructor, MethodImplementation, Source},
    rules::{
        MethodParam, ResolutionError, SolverResolutionArena, SolverResolutionNode,
        SolverResolutionRef, SolverResolvedHook, SolverResolvedNode,
    },
    types::{ParamKind, PyTypeConcreteKey, WrapperKind},
};

new_key_type! {
    pub(crate) struct ExecutionNodeId;
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum ExecutionCacheKey {
    Target(PyTypeConcreteKey),
    Source(Source),
    Property {
        source: Box<ExecutionCacheKey>,
        property_name: Arc<str>,
    },
    Attribute {
        source: Box<ExecutionCacheKey>,
        attribute_name: Arc<str>,
    },
    LazyRef {
        target: Box<ExecutionCacheKey>,
    },
    Constructor {
        implementation: Arc<Constructor>,
        params: Vec<ExecutionCacheKey>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExecutionCacheMode {
    Computed,
    Live,
}

#[derive(Clone)]
pub(crate) struct ConstructorParam {
    pub(crate) name: Arc<str>,
    pub(crate) kind: ParamKind,
    pub(crate) node: ExecutionNodeId,
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

pub(crate) enum ExecutionNode {
    Constant(Source),
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
        implementation: Arc<MethodImplementation>,
        return_wrapper: WrapperKind,
        accepts_varargs: bool,
        accepts_varkw: bool,
        bound_to: Option<ExecutionNodeId>,
        params: Vec<MethodParam>,
        result_source: Option<Source>,
        result_bindings: Vec<crate::rules::TransitionResultBinding>,
        target: ExecutionNodeId,
        hooks: Vec<ExecutionHook>,
    },
    AutoMethod {
        return_wrapper: WrapperKind,
        accepts_varargs: bool,
        accepts_varkw: bool,
        params: Vec<MethodParam>,
        target: ExecutionNodeId,
        hooks: Vec<ExecutionHook>,
    },
    Attribute {
        source: ExecutionNodeId,
        attribute_name: Arc<str>,
    },
    Constructor {
        implementation: Arc<Constructor>,
        params: Vec<ConstructorParam>,
    },
}

pub(crate) struct ExecutionEntry {
    pub(crate) target_type: PyTypeConcreteKey,
    pub(crate) cache_key: ExecutionCacheKey,
    pub(crate) node: ExecutionNode,
    pub(crate) source_deps: HashSet<Source>,
    pub(crate) cache_mode: ExecutionCacheMode,
}

impl std::fmt::Debug for ExecutionEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionEntry")
            .field("source_deps", &self.source_deps.len())
            .field("cache_mode", &self.cache_mode)
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
    let root = resolve_ref(&results, root, &mut graph, &mut refs)?;
    let reachable_result_refs = refs.len();
    compute_cache_keys(&mut graph);
    Ok((graph, root, reachable_result_refs))
}

fn resolve_ref(
    results: &SolverResolutionArena,
    node_ref: SolverResolutionRef,
    graph: &mut ExecutionGraph,
    refs: &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
) -> Result<ExecutionNodeId, ResolutionError> {
    if let Some(&node_id) = refs.get(&node_ref) {
        return Ok(node_id);
    }

    let resolved = get_resolved_node(results, node_ref)?;
    match &resolved.resolution {
        SolverResolutionNode::Delegate(target) | SolverResolutionNode::UnionVariant { target } => {
            let node_id = resolve_ref(results, *target, graph, refs)?;
            refs.insert(node_ref, node_id);
            Ok(node_id)
        }
        resolution => {
            let node_id = graph.insert(ExecutionEntry {
                target_type: resolved.target_type,
                cache_key: ExecutionCacheKey::Target(resolved.target_type),
                node: ExecutionNode::None,
                source_deps: HashSet::new(),
                cache_mode: ExecutionCacheMode::Live,
            });
            refs.insert(node_ref, node_id);
            graph[node_id].node = convert_node(results, resolution, graph, refs)?;
            graph[node_id].cache_mode = cache_mode_for(&graph[node_id].node);
            Ok(node_id)
        }
    }
}

fn cache_mode_for(node: &ExecutionNode) -> ExecutionCacheMode {
    match node {
        ExecutionNode::Constructor { .. } => ExecutionCacheMode::Computed,
        ExecutionNode::None
        | ExecutionNode::Constant(_)
        | ExecutionNode::Property { .. }
        | ExecutionNode::LazyRef { .. }
        | ExecutionNode::Protocol { .. }
        | ExecutionNode::TypedDict { .. }
        | ExecutionNode::Method { .. }
        | ExecutionNode::AutoMethod { .. }
        | ExecutionNode::Attribute { .. } => ExecutionCacheMode::Live,
    }
}

fn compute_cache_keys(graph: &mut ExecutionGraph) {
    let node_ids: Vec<_> = graph.keys().collect();
    let mut cache_keys: HashMap<ExecutionNodeId, ExecutionCacheKey> = node_ids
        .iter()
        .map(|&node_id| {
            (
                node_id,
                ExecutionCacheKey::Target(graph[node_id].target_type),
            )
        })
        .collect();

    let mut changed = true;
    while changed {
        changed = false;
        for &node_id in &node_ids {
            let new_key = match &graph[node_id].node {
                ExecutionNode::Constant(source) => ExecutionCacheKey::Source(source.clone()),
                ExecutionNode::Property {
                    source,
                    property_name,
                } => ExecutionCacheKey::Property {
                    source: Box::new(
                        cache_keys
                            .get(source)
                            .expect("property source cache key must exist")
                            .clone(),
                    ),
                    property_name: Arc::clone(property_name),
                },
                ExecutionNode::Attribute {
                    source,
                    attribute_name,
                } => ExecutionCacheKey::Attribute {
                    source: Box::new(
                        cache_keys
                            .get(source)
                            .expect("attribute source cache key must exist")
                            .clone(),
                    ),
                    attribute_name: Arc::clone(attribute_name),
                },
                ExecutionNode::LazyRef { target } => ExecutionCacheKey::LazyRef {
                    target: Box::new(
                        cache_keys
                            .get(target)
                            .expect("lazy ref target cache key must exist")
                            .clone(),
                    ),
                },
                ExecutionNode::Constructor {
                    implementation,
                    params,
                } => ExecutionCacheKey::Constructor {
                    implementation: Arc::clone(implementation),
                    params: params
                        .iter()
                        .map(|param| {
                            cache_keys
                                .get(&param.node)
                                .expect("constructor param cache key must exist")
                                .clone()
                        })
                        .collect(),
                },
                ExecutionNode::None
                | ExecutionNode::Protocol { .. }
                | ExecutionNode::TypedDict { .. }
                | ExecutionNode::Method { .. }
                | ExecutionNode::AutoMethod { .. } => {
                    ExecutionCacheKey::Target(graph[node_id].target_type)
                }
            };

            let current = cache_keys
                .get_mut(&node_id)
                .expect("execution cache key must exist");
            if *current != new_key {
                *current = new_key;
                changed = true;
            }
        }
    }

    for (node_id, cache_key) in cache_keys {
        graph[node_id].cache_key = cache_key;
    }
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

fn convert_node(
    results: &SolverResolutionArena,
    node: &SolverResolutionNode,
    graph: &mut ExecutionGraph,
    refs: &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
) -> Result<ExecutionNode, ResolutionError> {
    match node {
        SolverResolutionNode::Delegate(_) | SolverResolutionNode::UnionVariant { .. } => {
            unreachable!("indirection nodes must be resolved before convert_node")
        }
        SolverResolutionNode::None => Ok(ExecutionNode::None),
        SolverResolutionNode::Constant { source } => Ok(ExecutionNode::Constant(source.clone())),
        SolverResolutionNode::Property {
            source,
            property_name,
        } => Ok(ExecutionNode::Property {
            source: resolve_ref(results, *source, graph, refs)?,
            property_name: property_name.clone(),
        }),
        SolverResolutionNode::LazyRef { target } => Ok(ExecutionNode::LazyRef {
            target: resolve_ref(results, *target, graph, refs)?,
        }),
        SolverResolutionNode::Protocol { members } => Ok(ExecutionNode::Protocol {
            members: members
                .iter()
                .map(|(name, &node_ref)| {
                    resolve_ref(results, node_ref, graph, refs)
                        .map(|node_id| (name.clone(), node_id))
                })
                .collect::<Result<_, _>>()?,
        }),
        SolverResolutionNode::TypedDict { members } => Ok(ExecutionNode::TypedDict {
            members: members
                .iter()
                .map(|(name, &node_ref)| {
                    resolve_ref(results, node_ref, graph, refs)
                        .map(|node_id| (name.clone(), node_id))
                })
                .collect::<Result<_, _>>()?,
        }),
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
        } => Ok(ExecutionNode::Method {
            implementation: Arc::clone(implementation),
            return_wrapper: *return_wrapper,
            accepts_varargs: *accepts_varargs,
            accepts_varkw: *accepts_varkw,
            bound_to: bound_to
                .map(|node_ref| resolve_ref(results, node_ref, graph, refs))
                .transpose()?,
            params: params.clone(),
            result_source: result_source.clone(),
            result_bindings: result_bindings.clone(),
            target: resolve_ref(results, *target, graph, refs)?,
            hooks: convert_hooks(results, hooks, graph, refs)?,
        }),
        SolverResolutionNode::AutoMethod {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            target,
            hooks,
        } => Ok(ExecutionNode::AutoMethod {
            return_wrapper: *return_wrapper,
            accepts_varargs: *accepts_varargs,
            accepts_varkw: *accepts_varkw,
            params: params.clone(),
            target: resolve_ref(results, *target, graph, refs)?,
            hooks: convert_hooks(results, hooks, graph, refs)?,
        }),
        SolverResolutionNode::Attribute {
            source,
            attribute_name,
        } => Ok(ExecutionNode::Attribute {
            source: resolve_ref(results, *source, graph, refs)?,
            attribute_name: attribute_name.clone(),
        }),
        SolverResolutionNode::Constructor {
            implementation,
            params,
        } => Ok(ExecutionNode::Constructor {
            implementation: Arc::clone(implementation),
            params: params
                .iter()
                .map(|(node_ref, name, kind)| {
                    resolve_ref(results, *node_ref, graph, refs).map(|node_id| ConstructorParam {
                        name: name.clone(),
                        kind: *kind,
                        node: node_id,
                    })
                })
                .collect::<Result<_, _>>()?,
        }),
    }
}

fn convert_hooks(
    results: &SolverResolutionArena,
    hooks: &[SolverResolvedHook],
    graph: &mut ExecutionGraph,
    refs: &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
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
                        resolve_ref(results, *node_ref, graph, refs).map(|node_id| {
                            ConstructorParam {
                                name: name.clone(),
                                kind: *kind,
                                node: node_id,
                            }
                        })
                    })
                    .collect::<Result<_, _>>()?,
            })
        })
        .collect()
}
