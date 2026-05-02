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
use crate::registry::{Constructor, MethodImplementation};
use crate::rules::{
    RegistryResolutionRule, RegistrySharedState, ResolutionError, ResolutionQuery,
    builder::RuleGraph,
};
use crate::runtime::executor::{ContextData, execute};
use crate::runtime::resources::RuntimeResources;
use crate::types::{Bindings, PyTypeConcreteKey, TypeArenas};

fn solver_error_to_resolution_error(
    error: SolveError,
    target: PyTypeConcreteKey<'_>,
) -> ResolutionError<'_> {
    match error {
        SolveError::FixpointIterationLimitReached => ResolutionError::FixpointLimitReached(target),
        SolveError::StackOverflowDepthReached => ResolutionError::StackOverflowDepthReached(target),
        SolveError::SameDepthCycle => ResolutionError::UnexpectedSameDepthCycle(target),
        SolveError::AnswerSupportClosureIncomplete => {
            ResolutionError::AnswerSupportClosureIncomplete(target)
        }
    }
}

pub(crate) const SOLVER_FIXPOINT_ITERATION_LIMIT: usize = 1024;
pub(crate) const SOLVER_STACK_DEPTH_LIMIT: usize = 1024;

#[derive(Clone, Copy)]
pub(crate) struct CompileRegistry<'types, 'a> {
    pub(crate) constructors: &'a [Constructor<'types>],
    pub(crate) methods: &'a [MethodImplementation<'types>],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SolverLimits {
    pub(crate) fixpoint_iteration: usize,
    pub(crate) stack_depth: usize,
}

#[instrumented(
    name = "inlay.compile",
    target = "inlay",
    level = "info",
    skip(py, arenas, registry)
)]
pub(crate) fn compile<'types>(
    py: Python<'_>,
    arenas: &mut TypeArenas<'types>,
    registry: CompileRegistry<'types, '_>,
    rules: &RuleGraph,
    target: NormalizedTypeRef,
    solver_limits: SolverLimits,
) -> PyResult<Py<PyAny>> {
    let parametric = ingest_parametric(arenas, py, &target)?;

    let data = py.detach(|| {
        let concrete = arenas.apply_bindings(parametric, &Bindings::default());

        let shared_state =
            RegistrySharedState::new(registry.constructors, registry.methods, mem::take(arenas));

        let outcome = solve(
            &RegistryResolutionRule::new(Arc::new(rules.arena.clone())),
            ResolutionQuery::unnamed(concrete),
            rules.root,
            shared_state,
            solver_limits.fixpoint_iteration,
            solver_limits.stack_depth,
        );

        *arenas = outcome.shared_state.types;
        let (root, results) = outcome.result.map_err(|error| {
            solver_error_to_resolution_error(error, concrete).into_py_err(arenas)
        })?;

        let (exec_graph, exec_root) = flatten(results, root).map_err(|e| e.into_py_err(arenas))?;

        Ok::<_, PyErr>(ContextData {
            graph: Arc::new(exec_graph),
            root_node: exec_root,
        })
    })?;
    execute(py, &data, RuntimeResources::empty(), false)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pyo3::{PyErr, Python};

    use super::solver_error_to_resolution_error;
    use crate::qualifier::Qualifier;
    use crate::types::{
        Concrete, Keyed, PlainType, PyType, PyTypeDescriptor, PyTypeId, Qual, Qualified, TypeArenas,
    };
    use context_solver::solve::SolveError;

    fn with_target_type<R>(
        run: impl for<'types> FnOnce(&TypeArenas<'types>, crate::types::PyTypeConcreteKey<'types>) -> R,
    ) -> R {
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
        run(&arenas, PyType::Plain(key))
    }

    fn error_message(error: PyErr) -> String {
        Python::initialize();
        Python::attach(|py| error.value(py).to_string())
    }

    #[test]
    fn python_error_preserves_fixpoint_limit_failure() {
        let error = with_target_type(|arenas, target| {
            solver_error_to_resolution_error(SolveError::FixpointIterationLimitReached, target)
                .into_py_err(arenas)
        });

        assert!(
            error_message(error)
                .contains("solver fixpoint limit reached resolving type 'BenchmarkRoot<ANY>'")
        );
    }

    #[test]
    fn python_error_preserves_stack_overflow_failure() {
        let error = with_target_type(|arenas, target| {
            solver_error_to_resolution_error(SolveError::StackOverflowDepthReached, target)
                .into_py_err(arenas)
        });

        assert!(
            error_message(error).contains(
                "solver stack overflow depth reached resolving type 'BenchmarkRoot<ANY>'"
            )
        );
    }

    #[test]
    fn python_error_preserves_unexpected_same_depth_cycle_failure() {
        let error = with_target_type(|arenas, target| {
            solver_error_to_resolution_error(SolveError::SameDepthCycle, target).into_py_err(arenas)
        });

        assert!(error_message(error).contains(
            "unexpected same depth cycle escaped to root solve resolving type 'BenchmarkRoot<ANY>'"
        ));
    }

    #[test]
    fn python_error_preserves_answer_support_closure_failure() {
        let error = with_target_type(|arenas, target| {
            solver_error_to_resolution_error(SolveError::AnswerSupportClosureIncomplete, target)
                .into_py_err(arenas)
        });

        assert!(
            error_message(error).contains(
                "answer support closure is incomplete resolving type 'BenchmarkRoot<ANY>'"
            )
        );
    }
}
