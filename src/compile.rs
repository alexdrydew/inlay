use std::collections::HashMap;
use std::mem;
use std::sync::Arc;
use std::time::Instant;

use context_solver::solve::{SolveError, solve};
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
use crate::types::{Bindings, PyTypeConcreteKey, SlotBackend};

fn solver_error_to_resolution_error(
    error: SolveError,
    target: PyTypeConcreteKey<SlotBackend>,
) -> ResolutionError<SlotBackend> {
    match error {
        SolveError::FixpointIterationLimitReached => ResolutionError::FixpointLimitReached(target),
        SolveError::StackOverflowDepthReached => ResolutionError::StackOverflowDepthReached(target),
        SolveError::UnexpectedSameDepthCycle => ResolutionError::UnexpectedSameDepthCycle(target),
    }
}

const SOLVER_FIXPOINT_ITERATION_LIMIT: usize = 1024;
const SOLVER_DIRTY_FRAME_REEVALUATION_LIMIT: usize = 1024;

fn solver_fixpoint_iteration_limit() -> usize {
    std::env::var("INLAY_SOLVER_FIXPOINT_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(SOLVER_FIXPOINT_ITERATION_LIMIT)
}

fn solver_stack_overflow_depth() -> usize {
    std::env::var("INLAY_SOLVER_STACK_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(SOLVER_DIRTY_FRAME_REEVALUATION_LIMIT)
}

pub(crate) fn compile(
    py: Python<'_>,
    registry: &mut Registry,
    rules: &RuleGraph,
    target: NormalizedTypeRef,
) -> PyResult<Py<PyAny>> {
    let _compile_span = info_span!("compile").entered();
    let emit_stats = std::env::var_os("INLAY_COMPILE_STATS").is_some();

    let ingest_started = Instant::now();
    let parametric = ingest_parametric(&mut registry.arenas, py, &target)?;
    let ingest_elapsed = ingest_started.elapsed();
    info!(msg = "ingestion complete");

    let detached_started = Instant::now();
    let data = py.detach(|| {
        let apply_bindings_started = Instant::now();
        let concrete = registry
            .arenas
            .apply_bindings(parametric, &Bindings::default());
        let apply_bindings_elapsed = apply_bindings_started.elapsed();

        let shared_state_started = Instant::now();
        let shared_state = RegistrySharedState::new(
            &registry.constructors,
            &registry.methods,
            &registry.hooks,
            mem::take(&mut registry.arenas),
        );
        let shared_state_elapsed = shared_state_started.elapsed();

        let solve_started = Instant::now();
        let outcome = solve(
            &RegistryResolutionRule::new(Arc::new(rules.arena.clone())),
            concrete,
            rules.root,
            Arc::new(RegistryEnv::root()),
            shared_state,
            solver_fixpoint_iteration_limit(),
            solver_stack_overflow_depth(),
        );
        let solve_elapsed = solve_started.elapsed();
        if emit_stats {
            outcome.shared_state.emit_stats();
        }

        let restore_types_started = Instant::now();
        registry.arenas = outcome.shared_state.into_types();
        let restore_types_elapsed = restore_types_started.elapsed();
        let (root, results) = outcome.result.map_err(|error| {
            solver_error_to_resolution_error(error, concrete).into_py_err(&registry.arenas)
        })?;
        let solver_results_total = results.len();

        let flatten_started = Instant::now();
        let (mut exec_graph, exec_root, reachable_result_refs) =
            flatten(results, root).map_err(|e| e.into_py_err(&registry.arenas))?;
        let flatten_elapsed = flatten_started.elapsed();
        info!(graph_nodes = exec_graph.len(), msg = "flatten complete");

        let source_deps_started = Instant::now();
        compute_source_deps(&mut exec_graph);
        let source_deps_elapsed = source_deps_started.elapsed();

        if emit_stats {
            eprintln!(
                concat!(
                    "[inlay-compile-stats] ",
                    "ingest_ms={:.3} ",
                    "apply_bindings_ms={:.3} ",
                    "shared_state_ms={:.3} ",
                    "solve_ms={:.3} ",
                    "restore_types_ms={:.3} ",
                    "flatten_ms={:.3} ",
                    "source_deps_ms={:.3} ",
                    "detached_ms={:.3} ",
                    "solver_results_total={} ",
                    "reachable_result_refs={} ",
                    "graph_nodes={}"
                ),
                ingest_elapsed.as_secs_f64() * 1000.0,
                apply_bindings_elapsed.as_secs_f64() * 1000.0,
                shared_state_elapsed.as_secs_f64() * 1000.0,
                solve_elapsed.as_secs_f64() * 1000.0,
                restore_types_elapsed.as_secs_f64() * 1000.0,
                flatten_elapsed.as_secs_f64() * 1000.0,
                source_deps_elapsed.as_secs_f64() * 1000.0,
                detached_started.elapsed().as_secs_f64() * 1000.0,
                solver_results_total,
                reachable_result_refs,
                exec_graph.len(),
            );
        }

        Ok::<_, PyErr>(ContextData {
            graph: Arc::new(exec_graph),
            root_node: exec_root,
        })
    })?;
    let execute_started = Instant::now();
    let (result, scope_handle) = execute(py, &data, Scope::root(HashMap::new()), &[])?;
    let execute_elapsed = execute_started.elapsed();
    let attach_scope_started = Instant::now();
    let attached = attach_scope(py, result, scope_handle)?;
    if emit_stats {
        eprintln!(
            concat!(
                "[inlay-compile-stats] ",
                "execute_ms={:.3} ",
                "attach_scope_ms={:.3} ",
                "compile_total_ms={:.3}"
            ),
            execute_elapsed.as_secs_f64() * 1000.0,
            attach_scope_started.elapsed().as_secs_f64() * 1000.0,
            ingest_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(attached)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pyo3::{PyErr, Python};

    use super::solver_error_to_resolution_error;
    use crate::qualifier::Qualifier;
    use crate::types::{
        Arena, Concrete, Keyed, PlainType, PyType, PyTypeDescriptor, PyTypeId, Qual, Qualified,
        SlotBackend, TypeArenas,
    };
    use context_solver::solve::SolveError;

    fn target_type() -> (
        TypeArenas<SlotBackend>,
        crate::types::PyTypeConcreteKey<SlotBackend>,
    ) {
        let mut arenas = TypeArenas::<SlotBackend>::default();
        let key = arenas.concrete.plains.insert(Qualified {
            inner: PlainType::<Qual<Keyed<SlotBackend>>, Concrete> {
                descriptor: PyTypeDescriptor {
                    id: PyTypeId::new("BenchmarkRoot".to_string()),
                    display_name: Arc::from("BenchmarkRoot"),
                },
                args: Vec::new(),
            },
            qualifier: Qualifier::any(),
        });
        (arenas, PyType::Plain(key))
    }

    fn error_message(error: PyErr) -> String {
        Python::initialize();
        Python::attach(|py| error.value(py).to_string())
    }

    #[test]
    fn python_error_preserves_fixpoint_limit_failure() {
        let (arenas, target) = target_type();

        let error =
            solver_error_to_resolution_error(SolveError::FixpointIterationLimitReached, target)
                .into_py_err(&arenas);

        assert!(
            error_message(error)
                .contains("solver fixpoint limit reached resolving type 'BenchmarkRoot<ANY>'")
        );
    }

    #[test]
    fn python_error_preserves_stack_overflow_failure() {
        let (arenas, target) = target_type();

        let error = solver_error_to_resolution_error(SolveError::StackOverflowDepthReached, target)
            .into_py_err(&arenas);

        assert!(
            error_message(error).contains(
                "solver stack overflow depth reached resolving type 'BenchmarkRoot<ANY>'"
            )
        );
    }

    #[test]
    fn python_error_preserves_unexpected_same_depth_cycle_failure() {
        let (arenas, target) = target_type();

        let error = solver_error_to_resolution_error(SolveError::UnexpectedSameDepthCycle, target)
            .into_py_err(&arenas);

        assert!(error_message(error).contains(
            "unexpected same depth cycle escaped to root solve resolving type 'BenchmarkRoot<ANY>'"
        ));
    }
}
