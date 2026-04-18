use std::collections::{HashMap, HashSet};

use crate::{registry::Source, types::ArenaFamily};

use super::flatten::{ExecutionGraph, ExecutionNode, ExecutionNodeId};

/// Compute transitive source dependencies for every node in the execution graph.
///
/// After this pass, each `ExecutionEntry.source_deps` contains the set of
/// exact external sources the node transitively depends on.
pub(crate) fn compute_source_deps<S: ArenaFamily>(graph: &mut ExecutionGraph<S>) {
    let node_ids: Vec<ExecutionNodeId> = graph.keys().collect();

    // Iterative fixed-point: the graph is a DAG (Method/AutoMethod break cycles).
    let mut deps: HashMap<ExecutionNodeId, HashSet<Source>> = HashMap::new();
    for &nid in &node_ids {
        deps.insert(nid, HashSet::new());
    }

    let mut changed = true;
    while changed {
        changed = false;
        for &nid in &node_ids {
            let new_deps = compute_node_deps(nid, graph, &deps);
            let entry = deps.get_mut(&nid).expect("pre-populated");
            if *entry != new_deps {
                *entry = new_deps;
                changed = true;
            }
        }
    }

    for (nid, dep_set) in deps {
        graph[nid].source_deps = dep_set;
    }
}

fn compute_node_deps<S: ArenaFamily>(
    nid: ExecutionNodeId,
    graph: &ExecutionGraph<S>,
    deps: &HashMap<ExecutionNodeId, HashSet<Source>>,
) -> HashSet<Source> {
    let entry = &graph[nid];
    let mut result = HashSet::new();

    match &entry.node {
        ExecutionNode::None => {}

        ExecutionNode::Constant(source) => {
            result.insert(source.clone());
        }

        ExecutionNode::Constructor { params, .. } => {
            for p in params {
                result.extend(deps[&p.node].iter().cloned());
            }
        }

        ExecutionNode::Property { source, .. } => {
            result.extend(deps[source].iter().cloned());
        }

        ExecutionNode::Attribute { source, .. } => {
            result.extend(deps[source].iter().cloned());
        }

        ExecutionNode::Protocol { members } => {
            for child in members.values() {
                result.extend(deps[child].iter().cloned());
            }
        }

        ExecutionNode::TypedDict { members } => {
            for child in members.values() {
                result.extend(deps[child].iter().cloned());
            }
        }

        ExecutionNode::LazyRef { target } => {
            result.extend(deps[target].iter().cloned());
        }

        // Method/AutoMethod are closures — they don't eagerly evaluate their target.
        // Their deps are empty (breaks cycles in the dependency graph).
        ExecutionNode::Method { .. } | ExecutionNode::AutoMethod { .. } => {}
    }

    result
}
