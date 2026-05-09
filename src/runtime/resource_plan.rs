use std::collections::HashSet;

use crate::compile::flatten::{
    ExecutionGraph, ExecutionMethodImplementation, ExecutionNode, ExecutionNodeId, ExecutionParam,
    ExecutionSourceNodeId,
};

#[derive(Clone, Default)]
pub(crate) struct ResourcePlan {
    pub(crate) sources: HashSet<ExecutionSourceNodeId>,
    pub(crate) caches: HashSet<ExecutionNodeId>,
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
