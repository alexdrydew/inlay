use std::collections::HashMap;
use std::mem;
use std::sync::Arc;

use context_solver::solve::solve;
use pyo3::prelude::*;
use tracing::{info, info_span};

use crate::ingest::ingest_parametric;
use crate::normalized::NormalizedTypeRef;
use crate::registry::converter::Registry;
use crate::rules::{
    RegistryEnv, RegistryResolutionRule, RegistrySharedState, ResolutionError, builder::RuleGraph,
};
use crate::runtime::deps::compute_source_deps;
use crate::runtime::executor::{ContextData, attach_scope, execute};
use crate::runtime::flatten::flatten;
use crate::runtime::scope::Scope;
use crate::types::Bindings;

const SOLVER_FIXPOINT_ITERATION_LIMIT: usize = 1024;
const SOLVER_DIRTY_FRAME_REEVALUATION_LIMIT: usize = 1024;

pub(crate) fn compile(
    py: Python<'_>,
    registry: &mut Registry,
    rules: &RuleGraph,
    target: NormalizedTypeRef,
) -> PyResult<Py<PyAny>> {
    let _compile_span = info_span!("compile").entered();

    let parametric = ingest_parametric(&mut registry.arenas, py, &target)?;
    info!(msg = "ingestion complete");

    let data = py.detach(|| {
        let concrete = registry
            .arenas
            .apply_bindings(parametric, &Bindings::default());
        let shared_state = RegistrySharedState::new(
            &registry.constructors,
            &registry.methods,
            &registry.hooks,
            mem::take(&mut registry.arenas),
        );
        let outcome = solve(
            &RegistryResolutionRule::new(Arc::new(rules.arena.clone())),
            concrete,
            rules.root,
            Arc::new(RegistryEnv::root()),
            shared_state,
            SOLVER_FIXPOINT_ITERATION_LIMIT,
            SOLVER_DIRTY_FRAME_REEVALUATION_LIMIT,
        );
        registry.arenas = outcome.shared_state.into_types();
        let (root, results) = outcome.result.map_err(|_| {
            ResolutionError::FixpointLimitReached(concrete).into_py_err(&registry.arenas)
        })?;

        let (mut exec_graph, exec_root) =
            flatten(results, root).map_err(|e| e.into_py_err(&registry.arenas))?;
        info!(graph_nodes = exec_graph.len(), msg = "flatten complete");
        compute_source_deps(&mut exec_graph);

        Ok::<_, PyErr>(ContextData {
            graph: Arc::new(exec_graph),
            root_node: exec_root,
        })
    })?;
    let (result, scope_handle) = execute(py, &data, Scope::root(HashMap::new()), &[])?;
    attach_scope(py, result, scope_handle)
}
