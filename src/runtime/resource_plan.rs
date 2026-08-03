use std::collections::HashSet;

use crate::compile::execution_graph::{
    ExecutionCachePolicy, ExecutionComputedKind, ExecutionGraph, ExecutionNode, ExecutionNodeId,
    ExecutionParam, ExecutionSourceNodeId, ExecutionTransitionImplementation,
    ExecutionTransitionImplementationCallable,
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
    implementations: &[ExecutionTransitionImplementation],
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
    implementations: &[ExecutionTransitionImplementation],
    target: ExecutionNodeId,
    unavailable_sources: &HashSet<ExecutionSourceNodeId>,
    stack: &mut HashSet<ExecutionNodeId>,
    plan: &mut ResourcePlan,
) {
    let mut local_unavailable = unavailable_sources.clone();
    local_unavailable.extend(transition_param_sources(params));

    for implementation in implementations {
        if let ExecutionTransitionImplementationCallable::Source(source) =
            &implementation.implementation
        {
            collect_resource_plan(graph, source.node_id(), &local_unavailable, stack, plan);
        }
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
        ExecutionNode::Variable => {
            let source = ExecutionSourceNodeId(node_id);
            if !unavailable_sources.contains(&source) {
                plan.sources.insert(source);
            }
        }
        ExecutionNode::Field(field) => {
            collect_resource_plan(graph, field.source, unavailable_sources, stack, plan);
        }
        ExecutionNode::Computed(computed) => {
            if !computed.dynamic
                && matches!(computed.cache, ExecutionCachePolicy::Cached)
                && graph[node_id]
                    .resource_deps
                    .is_disjoint(unavailable_sources)
            {
                plan.caches.insert(node_id);
            }
            collect_computed_resource_plan(graph, &computed.kind, unavailable_sources, stack, plan);
        }
    }

    stack.remove(&node_id);
}

fn collect_computed_resource_plan(
    graph: &ExecutionGraph,
    kind: &ExecutionComputedKind,
    unavailable_sources: &HashSet<ExecutionSourceNodeId>,
    stack: &mut HashSet<ExecutionNodeId>,
    plan: &mut ResourcePlan,
) {
    match kind {
        ExecutionComputedKind::None | ExecutionComputedKind::StaticValue { .. } => {}
        ExecutionComputedKind::Property { source, .. } => {
            collect_resource_plan(graph, *source, unavailable_sources, stack, plan);
        }
        ExecutionComputedKind::ReadCell { target } | ExecutionComputedKind::Cell { target } => {
            collect_resource_plan(graph, *target, unavailable_sources, stack, plan);
        }
        ExecutionComputedKind::Protocol { members }
        | ExecutionComputedKind::TypedDict { members } => {
            for &member in members.values() {
                collect_resource_plan(graph, member, unavailable_sources, stack, plan);
            }
        }
        ExecutionComputedKind::Transition {
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
        ExecutionComputedKind::RuntimeUnionDispatch { source, branches } => {
            collect_resource_plan(graph, source.node_id(), unavailable_sources, stack, plan);
            for branch in branches {
                let mut local_unavailable = unavailable_sources.clone();
                local_unavailable.insert(branch.arm_source);
                collect_resource_plan(graph, branch.target, &local_unavailable, stack, plan);
            }
        }
        ExecutionComputedKind::Constructor { params, .. } => {
            for param in params {
                collect_resource_plan(graph, param.node, unavailable_sources, stack, plan);
            }
        }
    }
}
