use std::collections::HashSet;

use crate::compile::flatten::{
    ExecutionGraph, ExecutionMethodImplementation, ExecutionNode, ExecutionNodeId, ExecutionParam,
    ExecutionSourceNodeId,
};

/// Minimal descriptor of runtime resources needed to execute a graph node later.
#[derive(Clone, Default)]
pub(crate) struct ResourcePlan {
    /// source slots whose current Python values must be retained
    pub(crate) sources: HashSet<ExecutionSourceNodeId>,
    /// cache cells that should be shared with the capturing runtime scope
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

pub(crate) fn resource_plan_for_transition(
    graph: &ExecutionGraph,
    params: &[ExecutionParam],
    implementations: &[ExecutionMethodImplementation],
    target: ExecutionNodeId,
) -> ResourcePlan {
    let mut plan = ResourcePlan::default();
    let mut stack = HashSet::new();
    let unavailable_sources = HashSet::new();
    collect_transition_resource_plan(
        graph,
        params,
        implementations,
        target,
        &unavailable_sources,
        &mut stack,
        &mut plan,
    );
    plan
}

fn transition_param_sources(params: &[ExecutionParam]) -> HashSet<ExecutionSourceNodeId> {
    params
        .iter()
        .flat_map(|param| param.sources.iter().copied())
        .collect()
}

fn collect_transition_resource_plan(
    graph: &ExecutionGraph,
    params: &[ExecutionParam],
    implementations: &[ExecutionMethodImplementation],
    target: ExecutionNodeId,
    unavailable_sources: &HashSet<ExecutionSourceNodeId>,
    stack: &mut HashSet<ExecutionNodeId>,
    plan: &mut ResourcePlan,
) {
    let mut local_unavailable = unavailable_sources.clone();
    local_unavailable.extend(transition_param_sources(params));

    for implementation in implementations {
        if let Some(bound_to) = implementation.bound_to {
            collect_resource_plan(graph, bound_to, &local_unavailable, stack, plan);
        }
        for param in &implementation.params {
            collect_resource_plan(graph, param.node, &local_unavailable, stack, plan);
        }
        if let Some(result_source) = implementation.result_source {
            local_unavailable.insert(result_source);
        }
    }

    collect_resource_plan(graph, target, &local_unavailable, stack, plan);
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
            collect_transition_resource_plan(
                graph,
                params,
                implementations,
                *target,
                unavailable_sources,
                stack,
                plan,
            );
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
