mod compiler;
pub(crate) mod execution_graph;
pub(crate) mod ingest;

use context_solver::solve::{SolveError, Solver};

pub(crate) use compiler::Compiler;

use crate::rules::{RegistryResolutionRule, ResolutionError};
use crate::types::PyTypeConcreteKey;

pub(crate) type RegistrySolver = Solver<RegistryResolutionRule<'static>>;

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
        run: impl for<'ty> FnOnce(&TypeArenas<'ty>, crate::types::PyTypeConcreteKey<'ty>) -> R,
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
