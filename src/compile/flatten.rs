use std::{
    collections::{BTreeMap, HashMap, HashSet},
    hash::{Hash, Hasher},
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
    types::{ParamKind, PyTypeConcreteKey, WrapperKind},
};

new_key_type! {
    pub(crate) struct ExecutionNodeId;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ExecutionSourceId(usize);

#[derive(Default)]
struct SourceInterner {
    sources: HashMap<Source, ExecutionSourceId>,
}

impl SourceInterner {
    fn intern(&mut self, source: &Source) -> ExecutionSourceId {
        if let Some(source_id) = self.sources.get(source) {
            return *source_id;
        }

        let source_id = ExecutionSourceId(self.sources.len());
        self.sources.insert(source.clone(), source_id);
        source_id
    }
}

#[derive(Clone)]
pub(crate) enum ExecutionCacheKey {
    Target(PyTypeConcreteKey),
    Source(ExecutionSourceId),
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
        implementation: Arc<Py<PyAny>>,
        params: Vec<ExecutionCacheKey>,
    },
}

impl PartialEq for ExecutionCacheKey {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Target(a), Self::Target(b)) => a == b,
            (Self::Source(a), Self::Source(b)) => a == b,
            (
                Self::Property {
                    source: a_source,
                    property_name: a_name,
                },
                Self::Property {
                    source: b_source,
                    property_name: b_name,
                },
            ) => a_source == b_source && a_name == b_name,
            (
                Self::Attribute {
                    source: a_source,
                    attribute_name: a_name,
                },
                Self::Attribute {
                    source: b_source,
                    attribute_name: b_name,
                },
            ) => a_source == b_source && a_name == b_name,
            (Self::LazyRef { target: a }, Self::LazyRef { target: b }) => a == b,
            (
                Self::Constructor {
                    implementation: a_implementation,
                    params: a_params,
                },
                Self::Constructor {
                    implementation: b_implementation,
                    params: b_params,
                },
            ) => {
                a_implementation.as_ref().as_ptr() == b_implementation.as_ref().as_ptr()
                    && a_params == b_params
            }
            _ => false,
        }
    }
}

impl Eq for ExecutionCacheKey {}

impl Hash for ExecutionCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Target(target) => target.hash(state),
            Self::Source(source) => source.hash(state),
            Self::Property {
                source,
                property_name,
            } => {
                source.hash(state);
                property_name.hash(state);
            }
            Self::Attribute {
                source,
                attribute_name,
            } => {
                source.hash(state);
                attribute_name.hash(state);
            }
            Self::LazyRef { target } => target.hash(state),
            Self::Constructor {
                implementation,
                params,
            } => {
                implementation.as_ref().as_ptr().hash(state);
                params.hash(state);
            }
        }
    }
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
pub(crate) struct ExecutionParam {
    pub(crate) name: Arc<str>,
    pub(crate) kind: ParamKind,
    pub(crate) source: ExecutionSourceId,
}

impl ExecutionParam {
    fn from_method_param(param: &MethodParam, source_interner: &mut SourceInterner) -> Self {
        Self {
            name: Arc::clone(&param.name),
            kind: param.kind,
            source: source_interner.intern(&param.source),
        }
    }
}

#[derive(Clone)]
pub(crate) struct ExecutionResultBinding {
    pub(crate) name: Arc<str>,
    pub(crate) source: ExecutionSourceId,
}

impl ExecutionResultBinding {
    fn from_transition_binding(
        binding: &crate::rules::TransitionResultBinding,
        source_interner: &mut SourceInterner,
    ) -> Self {
        Self {
            name: Arc::clone(&binding.name),
            source: source_interner.intern(&binding.source),
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

pub(crate) enum ExecutionNode {
    Constant(ExecutionSourceId),
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
        result_source: ExecutionSourceId,
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

impl ExecutionNode {
    fn cache_mode(&self) -> ExecutionCacheMode {
        match self {
            Self::Constructor { .. } => ExecutionCacheMode::Computed,
            Self::None
            | Self::Constant(_)
            | Self::Property { .. }
            | Self::LazyRef { .. }
            | Self::Protocol { .. }
            | Self::TypedDict { .. }
            | Self::Method { .. }
            | Self::AutoMethod { .. }
            | Self::Attribute { .. } => ExecutionCacheMode::Live,
        }
    }
}

pub(crate) struct ExecutionEntry {
    pub(crate) target_type: PyTypeConcreteKey,
    pub(crate) cache_key: ExecutionCacheKey,
    pub(crate) node: ExecutionNode,
    pub(crate) source_deps: HashSet<ExecutionSourceId>,
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
    let mut source_interner = SourceInterner::default();
    let root = resolve_ref(&results, root, &mut graph, &mut refs, &mut source_interner)?;
    let reachable_result_refs = refs.len();
    compute_cache_keys(&mut graph);
    Ok((graph, root, reachable_result_refs))
}

fn resolve_ref(
    results: &SolverResolutionArena,
    node_ref: SolverResolutionRef,
    graph: &mut ExecutionGraph,
    refs: &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
    source_interner: &mut SourceInterner,
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
        SolverResolutionNode::None => materialize_node(
            resolved.target_type,
            node_ref,
            graph,
            refs,
            source_interner,
            |_, _, _| Ok(ExecutionNode::None),
        ),
        SolverResolutionNode::Constant { source } => materialize_node(
            resolved.target_type,
            node_ref,
            graph,
            refs,
            source_interner,
            |_, _, source_interner| Ok(ExecutionNode::Constant(source_interner.intern(source))),
        ),
        SolverResolutionNode::Property {
            source,
            property_name,
        } => materialize_node(
            resolved.target_type,
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
            resolved.target_type,
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
            resolved.target_type,
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
            resolved.target_type,
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
            resolved.target_type,
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
                        .map(|param| ExecutionParam::from_method_param(param, source_interner))
                        .collect(),
                    result_source: source_interner.intern(result_source),
                    result_bindings: result_bindings
                        .iter()
                        .map(|binding| {
                            ExecutionResultBinding::from_transition_binding(
                                binding,
                                source_interner,
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
            resolved.target_type,
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
                        .map(|param| ExecutionParam::from_method_param(param, source_interner))
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
            resolved.target_type,
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
            resolved.target_type,
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
    target_type: PyTypeConcreteKey,
    node_ref: SolverResolutionRef,
    graph: &mut ExecutionGraph,
    refs: &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
    source_interner: &mut SourceInterner,
    build_node: impl FnOnce(
        &mut ExecutionGraph,
        &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
        &mut SourceInterner,
    ) -> Result<ExecutionNode, ResolutionError>,
) -> Result<ExecutionNodeId, ResolutionError> {
    let node_id = graph.insert(ExecutionEntry {
        target_type,
        cache_key: ExecutionCacheKey::Target(target_type),
        node: ExecutionNode::None,
        source_deps: HashSet::new(),
        cache_mode: ExecutionCacheMode::Live,
    });
    refs.insert(node_ref, node_id);
    let node = build_node(graph, refs, source_interner)?;
    graph[node_id].node = node;
    graph[node_id].cache_mode = graph[node_id].node.cache_mode();
    Ok(node_id)
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
                ExecutionNode::Constant(source) => ExecutionCacheKey::Source(*source),
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

fn convert_hooks(
    results: &SolverResolutionArena,
    hooks: &[SolverResolvedHook],
    graph: &mut ExecutionGraph,
    refs: &mut HashMap<SolverResolutionRef, ExecutionNodeId>,
    source_interner: &mut SourceInterner,
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use context_solver::Arena as _;

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
}
