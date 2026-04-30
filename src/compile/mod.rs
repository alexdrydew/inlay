use std::collections::HashMap;
use std::mem;
use std::sync::Arc;

pub(crate) mod flatten;
pub(crate) mod ingest;

use context_solver::solve::{SolveError, solve};
use inlay_instrument::instrumented;
use pyo3::prelude::*;

use self::flatten::flatten;
use self::ingest::ingest_parametric;
use crate::normalized::NormalizedTypeRef;
use crate::registry::{Constructor, Hook, MethodImplementation};
use crate::rules::{
    RegistryResolutionRule, RegistrySharedState, ResolutionError, ResolutionQuery,
    builder::RuleGraph,
};
use crate::runtime::executor::{ContextData, attach_scope, execute};
use crate::runtime::scope::Scope;
use crate::types::{Bindings, PyTypeConcreteKey, TypeArenas};

fn solver_error_to_resolution_error(
    error: SolveError,
    target: PyTypeConcreteKey,
) -> ResolutionError {
    match error {
        SolveError::FixpointIterationLimitReached => ResolutionError::FixpointLimitReached(target),
        SolveError::StackOverflowDepthReached => ResolutionError::StackOverflowDepthReached(target),
        SolveError::SameDepthCycle => ResolutionError::UnexpectedSameDepthCycle(target),
        SolveError::AnswerSupportClosureIncomplete => {
            ResolutionError::AnswerSupportClosureIncomplete(target)
        }
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

fn solver_stack_depth_limit() -> usize {
    std::env::var("INLAY_SOLVER_STACK_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(SOLVER_DIRTY_FRAME_REEVALUATION_LIMIT)
}

#[instrumented(
    name = "inlay.compile",
    target = "inlay",
    level = "info",
    skip(py, arenas, constructors, methods, hooks)
)]
pub(crate) fn compile(
    py: Python<'_>,
    arenas: &mut TypeArenas,
    constructors: &[Constructor],
    methods: &[MethodImplementation],
    hooks: &[Hook],
    rules: &RuleGraph,
    target: NormalizedTypeRef,
) -> PyResult<Py<PyAny>> {
    let parametric = ingest_parametric(arenas, py, &target)?;

    let data = py.detach(|| {
        let concrete = arenas.apply_bindings(parametric, &Bindings::default());

        let shared_state =
            RegistrySharedState::new(constructors, methods, hooks, mem::take(arenas));

        let outcome = solve(
            &RegistryResolutionRule::new(Arc::new(rules.arena.clone())),
            ResolutionQuery::unnamed(concrete),
            rules.root,
            shared_state,
            solver_fixpoint_iteration_limit(),
            solver_stack_depth_limit(),
        );

        *arenas = outcome.shared_state.types;
        let (root, results) = outcome.result.map_err(|error| {
            solver_error_to_resolution_error(error, concrete).into_py_err(arenas)
        })?;

        let (exec_graph, exec_root, _reachable_result_refs) =
            flatten(results, root).map_err(|e| e.into_py_err(arenas))?;

        Ok::<_, PyErr>(ContextData {
            graph: Arc::new(exec_graph),
            root_node: exec_root,
        })
    })?;
    let (result, scope_handle) = execute(py, &data, Scope::root(HashMap::new()), &[])?;
    let attached = attach_scope(py, result, scope_handle)?;
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
        TypeArenas,
    };
    use context_solver::solve::SolveError;

    fn target_type() -> (TypeArenas, crate::types::PyTypeConcreteKey) {
        let mut arenas = TypeArenas::default();
        let key = arenas.concrete.plains.insert(Qualified {
            inner: PlainType::<Qual<Keyed>, Concrete> {
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

        let error = solver_error_to_resolution_error(SolveError::SameDepthCycle, target)
            .into_py_err(&arenas);

        assert!(error_message(error).contains(
            "unexpected same depth cycle escaped to root solve resolving type 'BenchmarkRoot<ANY>'"
        ));
    }

    #[test]
    fn python_error_preserves_answer_support_closure_failure() {
        let (arenas, target) = target_type();

        let error =
            solver_error_to_resolution_error(SolveError::AnswerSupportClosureIncomplete, target)
                .into_py_err(&arenas);

        assert!(
            error_message(error).contains(
                "answer support closure is incomplete resolving type 'BenchmarkRoot<ANY>'"
            )
        );
    }
}
