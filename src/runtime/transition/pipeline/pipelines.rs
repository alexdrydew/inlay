use pyo3::PyTraverseError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;

use crate::compile::flatten::{ExecutionMethodImplementation, ExecutionSourceNodeId};
use crate::runtime::executor::{
    ContextData, ExecutionState, bind_lazy_refs, execute_method_implementation, execute_node,
};
use crate::runtime::resources::RuntimeResources;
use crate::types::WrapperKind;

use super::super::wrap_transition_leaf_result;
use super::{ExceptionTriple, ExitItem, ExitOutcome, MixedExitStack};

pub(crate) struct PipelineCommon {
    data: ContextData,
    state: ExecutionState,
    implementations: Vec<ExecutionMethodImplementation>,
    next_index: usize,
}

impl PipelineCommon {
    pub(crate) fn new(
        data: ContextData,
        resources: RuntimeResources,
        implementations: Vec<ExecutionMethodImplementation>,
    ) -> Self {
        Self {
            data,
            state: ExecutionState::new(resources, true),
            implementations,
            next_index: 0,
        }
    }

    fn has_more_implementations(&self) -> bool {
        self.next_index < self.implementations.len()
    }

    fn call_next_implementation(
        &mut self,
        py: Python<'_>,
    ) -> PyResult<(ExecutionMethodImplementation, Py<PyAny>)> {
        let implementation = self.implementations[self.next_index].clone();
        let result =
            execute_method_implementation(py, &self.data, &mut self.state, &implementation)?;
        Ok((implementation, result))
    }

    fn finish_implementation(
        &mut self,
        py: Python<'_>,
        result_source: Option<ExecutionSourceNodeId>,
        result: Py<PyAny>,
    ) {
        if let Some(source) = result_source {
            self.state
                .resources
                .insert_source(&self.data.graph, source, result.clone_ref(py));
        }
        self.next_index += 1;
    }

    fn execute_target(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let result = execute_node(py, &self.data, &mut self.state, self.data.root_node)?;
        bind_lazy_refs(py, &self.data, &mut self.state)?;
        wrap_transition_leaf_result(py, self.data.graph.clone(), result)
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.state.traverse(visit)?;
        for implementation in &self.implementations {
            visit.call(&*implementation.implementation)?;
        }
        Ok(())
    }
}

pub(crate) struct EnterProgram {
    common: PipelineCommon,
}

pub(crate) struct EnterContinuation {
    program: EnterProgram,
    result_source: Option<ExecutionSourceNodeId>,
}

pub(crate) enum EnterEffect {
    Await {
        awaitable: Py<PyAny>,
        continuation: EnterContinuation,
    },
    EnterSync {
        context: Py<PyAny>,
        continuation: EnterContinuation,
    },
    EnterAsync {
        context: Py<PyAny>,
        continuation: EnterContinuation,
    },
}

pub(crate) enum EnterPoll {
    Effect(EnterEffect),
    Completed(Py<PyAny>),
}

impl EnterProgram {
    pub(crate) fn new(common: PipelineCommon) -> Self {
        Self { common }
    }

    pub(crate) fn poll(mut self, py: Python<'_>) -> PyResult<EnterPoll> {
        while self.common.has_more_implementations() {
            let (implementation, result) = self.common.call_next_implementation(py)?;
            match implementation.return_wrapper {
                WrapperKind::None => {
                    self.common
                        .finish_implementation(py, implementation.result_source, result);
                }
                WrapperKind::ContextManager => {
                    return Ok(EnterPoll::Effect(EnterEffect::EnterSync {
                        context: result,
                        continuation: EnterContinuation {
                            program: self,
                            result_source: implementation.result_source,
                        },
                    }));
                }
                WrapperKind::Awaitable => {
                    return Ok(EnterPoll::Effect(EnterEffect::Await {
                        awaitable: result,
                        continuation: EnterContinuation {
                            program: self,
                            result_source: implementation.result_source,
                        },
                    }));
                }
                WrapperKind::AsyncContextManager => {
                    return Ok(EnterPoll::Effect(EnterEffect::EnterAsync {
                        context: result,
                        continuation: EnterContinuation {
                            program: self,
                            result_source: implementation.result_source,
                        },
                    }));
                }
            }
        }

        Ok(EnterPoll::Completed(self.common.execute_target(py)?))
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.common.traverse(visit)
    }
}

impl EnterContinuation {
    pub(crate) fn resume(mut self, py: Python<'_>, value: Py<PyAny>) -> EnterProgram {
        self.program
            .common
            .finish_implementation(py, self.result_source, value);
        self.program
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.program.traverse(visit)
    }
}

pub(crate) struct ExitProgram {
    exits: MixedExitStack,
    active: ExceptionTriple,
    received_exception: bool,
    suppressed_exception: bool,
    pending_error: Option<PyErr>,
}

pub(crate) struct ExitContinuation {
    program: ExitProgram,
}

pub(crate) enum ExitEffect {
    ExitSync {
        context: Py<PyAny>,
        exc: ExceptionTriple,
        continuation: ExitContinuation,
    },
    ExitAsync {
        context: Py<PyAny>,
        exc: ExceptionTriple,
        continuation: ExitContinuation,
    },
}

pub(crate) enum ExitPoll {
    Effect(ExitEffect),
    Completed { suppressed: bool },
    Failed(PyErr),
}

impl ExitProgram {
    pub(crate) fn new(py: Python<'_>, exits: MixedExitStack, active: ExceptionTriple) -> Self {
        let received_exception = active.is_some(py);
        Self {
            exits,
            active,
            received_exception,
            suppressed_exception: false,
            pending_error: None,
        }
    }

    pub(crate) fn poll(mut self, py: Python<'_>) -> ExitPoll {
        if let Some(item) = self.exits.pop() {
            let exc = self.active.clone_ref(py);
            let continuation = ExitContinuation { program: self };
            return match item {
                ExitItem::Sync(context) => ExitPoll::Effect(ExitEffect::ExitSync {
                    context,
                    exc,
                    continuation,
                }),
                ExitItem::Async(context) => ExitPoll::Effect(ExitEffect::ExitAsync {
                    context,
                    exc,
                    continuation,
                }),
            };
        }

        self.completion()
    }

    fn consume_outcome(&mut self, py: Python<'_>, outcome: ExitOutcome) {
        match outcome {
            ExitOutcome::Returned(value) => match value.bind(py).is_truthy() {
                Ok(true) => {
                    if self.active.is_some(py) {
                        self.active = ExceptionTriple::none(py);
                        self.pending_error = None;
                        self.suppressed_exception = true;
                    }
                }
                Ok(false) => {}
                Err(error) => self.set_error(py, error),
            },
            ExitOutcome::Raised(error) => self.set_error(py, error),
        }
    }

    fn set_error(&mut self, py: Python<'_>, error: PyErr) {
        self.active = ExceptionTriple::from_error(py, &error);
        self.pending_error = Some(error);
    }

    fn completion(&mut self) -> ExitPoll {
        if let Some(error) = self.pending_error.take() {
            ExitPoll::Failed(error)
        } else {
            ExitPoll::Completed {
                suppressed: self.received_exception && self.suppressed_exception,
            }
        }
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.exits.traverse(visit)?;
        self.active.traverse(visit)
    }
}

impl ExitContinuation {
    pub(crate) fn resume(mut self, py: Python<'_>, outcome: ExitOutcome) -> ExitProgram {
        self.program.consume_outcome(py, outcome);
        self.program
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.program.traverse(visit)
    }
}
