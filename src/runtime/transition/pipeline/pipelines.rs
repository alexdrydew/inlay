use pyo3::PyTraverseError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;

use crate::compile::flatten::{ExecutionMethodImplementation, ExecutionSourceNodeId};
use crate::runtime::executor::{
    ContextData, ExecutionState, bind_lazy_refs, execute_method_implementation, execute_node,
};
use crate::runtime::resources::RuntimeResources;
use crate::types::WrapperKind;

use super::super::wrap_transition_leaf_result;
use super::exits::{ExceptionTriple, ExitItem, MixedExitStack, SyncExitStack};
use super::step::{
    AsyncContextManagerEnterStep, AsyncExitStep, AwaitableMethodStep, ContextManagerEnterStep,
    ExitDrainCompletion, ExitOutcome, SyncExitStep,
};

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

    pub(crate) fn run_plain(mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        while self.has_more_implementations() {
            let (implementation, result) = self.call_next_implementation(py)?;
            if implementation.return_wrapper != WrapperKind::None {
                return Err(PyRuntimeError::new_err(
                    "wrapped method implementation cannot run in a plain sync transition",
                ));
            }
            self.finish_implementation(py, implementation.result_source, result);
        }

        self.execute_target(py)
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
                .insert_source(source, result.clone_ref(py));
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

enum InlayEnterState {
    Idle,
    Awaiting {
        result_source: Option<ExecutionSourceNodeId>,
    },
}

pub(crate) struct ContextManagerEnterPipeline {
    common: PipelineCommon,
    state: InlayEnterState,
}

impl ContextManagerEnterPipeline {
    pub(crate) fn new(common: PipelineCommon) -> Self {
        Self {
            common,
            state: InlayEnterState::Idle,
        }
    }

    pub(crate) fn next(
        &mut self,
        py: Python<'_>,
        input: Option<Py<PyAny>>,
    ) -> PyResult<ContextManagerEnterStep> {
        self.consume_pending_input(py, input);

        while self.common.has_more_implementations() {
            let (implementation, result) = self.common.call_next_implementation(py)?;
            match implementation.return_wrapper {
                WrapperKind::None => {
                    self.common
                        .finish_implementation(py, implementation.result_source, result);
                }
                WrapperKind::ContextManager => {
                    self.state = InlayEnterState::Awaiting {
                        result_source: implementation.result_source,
                    };
                    return Ok(ContextManagerEnterStep::EnterSync(result));
                }
                WrapperKind::Awaitable | WrapperKind::AsyncContextManager => {
                    return Err(PyRuntimeError::new_err(
                        "incompatible wrapped implementation for context manager transition",
                    ));
                }
            }
        }

        Ok(ContextManagerEnterStep::Done(
            self.common.execute_target(py)?,
        ))
    }

    fn consume_pending_input(&mut self, py: Python<'_>, input: Option<Py<PyAny>>) {
        if let InlayEnterState::Awaiting { result_source } =
            std::mem::replace(&mut self.state, InlayEnterState::Idle)
        {
            self.common.finish_implementation(
                py,
                result_source,
                input.unwrap_or_else(|| py.None()),
            );
        }
    }
}

pub(crate) struct AwaitableMethodPipeline {
    common: PipelineCommon,
    state: InlayEnterState,
}

impl AwaitableMethodPipeline {
    pub(crate) fn new(common: PipelineCommon) -> Self {
        Self {
            common,
            state: InlayEnterState::Idle,
        }
    }

    pub(crate) fn next(
        &mut self,
        py: Python<'_>,
        input: Option<Py<PyAny>>,
    ) -> PyResult<AwaitableMethodStep> {
        self.consume_pending_input(py, input);

        while self.common.has_more_implementations() {
            let (implementation, result) = self.common.call_next_implementation(py)?;
            match implementation.return_wrapper {
                WrapperKind::None => {
                    self.common
                        .finish_implementation(py, implementation.result_source, result);
                }
                WrapperKind::Awaitable => {
                    self.state = InlayEnterState::Awaiting {
                        result_source: implementation.result_source,
                    };
                    return Ok(AwaitableMethodStep::Await(result));
                }
                WrapperKind::ContextManager | WrapperKind::AsyncContextManager => {
                    return Err(PyRuntimeError::new_err(
                        "incompatible wrapped implementation for awaitable transition",
                    ));
                }
            }
        }

        Ok(AwaitableMethodStep::Done(self.common.execute_target(py)?))
    }

    fn consume_pending_input(&mut self, py: Python<'_>, input: Option<Py<PyAny>>) {
        if let InlayEnterState::Awaiting { result_source } =
            std::mem::replace(&mut self.state, InlayEnterState::Idle)
        {
            self.common.finish_implementation(
                py,
                result_source,
                input.unwrap_or_else(|| py.None()),
            );
        }
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.common.traverse(visit)
    }
}

pub(crate) struct AsyncContextManagerEnterPipeline {
    common: PipelineCommon,
    state: InlayEnterState,
}

impl AsyncContextManagerEnterPipeline {
    pub(crate) fn new(common: PipelineCommon) -> Self {
        Self {
            common,
            state: InlayEnterState::Idle,
        }
    }

    pub(crate) fn next(
        &mut self,
        py: Python<'_>,
        input: Option<Py<PyAny>>,
    ) -> PyResult<AsyncContextManagerEnterStep> {
        self.consume_pending_input(py, input);

        while self.common.has_more_implementations() {
            let (implementation, result) = self.common.call_next_implementation(py)?;
            match implementation.return_wrapper {
                WrapperKind::None => {
                    self.common
                        .finish_implementation(py, implementation.result_source, result);
                }
                WrapperKind::ContextManager => {
                    self.state = InlayEnterState::Awaiting {
                        result_source: implementation.result_source,
                    };
                    return Ok(AsyncContextManagerEnterStep::EnterSync(result));
                }
                WrapperKind::Awaitable => {
                    self.state = InlayEnterState::Awaiting {
                        result_source: implementation.result_source,
                    };
                    return Ok(AsyncContextManagerEnterStep::Await(result));
                }
                WrapperKind::AsyncContextManager => {
                    self.state = InlayEnterState::Awaiting {
                        result_source: implementation.result_source,
                    };
                    return Ok(AsyncContextManagerEnterStep::EnterAsync(result));
                }
            }
        }

        Ok(AsyncContextManagerEnterStep::Done(
            self.common.execute_target(py)?,
        ))
    }

    fn consume_pending_input(&mut self, py: Python<'_>, input: Option<Py<PyAny>>) {
        if let InlayEnterState::Awaiting { result_source } =
            std::mem::replace(&mut self.state, InlayEnterState::Idle)
        {
            self.common.finish_implementation(
                py,
                result_source,
                input.unwrap_or_else(|| py.None()),
            );
        }
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.common.traverse(visit)
    }
}

pub(crate) struct SyncExitDrainerPipeline {
    exits: SyncExitStack,
    active: ExceptionTriple,
    received_exception: bool,
    suppressed_exception: bool,
    pending_error: Option<PyErr>,
    awaiting_exit: bool,
}

impl SyncExitDrainerPipeline {
    pub(crate) fn new(py: Python<'_>, exits: SyncExitStack, active: ExceptionTriple) -> Self {
        let received_exception = active.is_some(py);
        Self {
            exits,
            active,
            received_exception,
            suppressed_exception: false,
            pending_error: None,
            awaiting_exit: false,
        }
    }

    pub(crate) fn next(&mut self, py: Python<'_>, outcome: Option<ExitOutcome>) -> SyncExitStep {
        if self.awaiting_exit {
            self.consume_outcome(py, outcome.expect("exit outcome expected"));
            self.awaiting_exit = false;
        }

        if let Some(context) = self.exits.pop() {
            self.awaiting_exit = true;
            return SyncExitStep::ExitSync {
                context,
                exc: self.active.clone_ref(py),
            };
        }

        SyncExitStep::Done(self.completion())
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

    fn completion(&mut self) -> ExitDrainCompletion {
        if let Some(error) = self.pending_error.take() {
            ExitDrainCompletion::Raise(error)
        } else {
            ExitDrainCompletion::Return(self.received_exception && self.suppressed_exception)
        }
    }
}

pub(crate) struct AsyncExitDrainerPipeline {
    exits: MixedExitStack,
    active: ExceptionTriple,
    received_exception: bool,
    suppressed_exception: bool,
    pending_error: Option<PyErr>,
    awaiting_exit: bool,
}

impl AsyncExitDrainerPipeline {
    pub(crate) fn new(py: Python<'_>, exits: MixedExitStack, active: ExceptionTriple) -> Self {
        let received_exception = active.is_some(py);
        Self {
            exits,
            active,
            received_exception,
            suppressed_exception: false,
            pending_error: None,
            awaiting_exit: false,
        }
    }

    pub(crate) fn next(&mut self, py: Python<'_>, outcome: Option<ExitOutcome>) -> AsyncExitStep {
        if self.awaiting_exit {
            self.consume_outcome(py, outcome.expect("exit outcome expected"));
            self.awaiting_exit = false;
        }

        if let Some(item) = self.exits.pop() {
            self.awaiting_exit = true;
            let exc = self.active.clone_ref(py);
            return match item {
                ExitItem::Sync(context) => AsyncExitStep::ExitSync { context, exc },
                ExitItem::Async(context) => AsyncExitStep::ExitAsync { context, exc },
            };
        }

        AsyncExitStep::Done(self.completion())
    }

    fn consume_outcome(&mut self, py: Python<'_>, outcome: ExitOutcome) {
        match outcome {
            ExitOutcome::Returned(value) => match value.bind(py).is_truthy() {
                Ok(true) => {
                    // A truthy return only suppresses when an exception is active, but
                    // truthiness is still evaluated so __bool__ errors become cleanup errors.
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

    fn completion(&mut self) -> ExitDrainCompletion {
        if let Some(error) = self.pending_error.take() {
            ExitDrainCompletion::Raise(error)
        } else {
            ExitDrainCompletion::Return(self.received_exception && self.suppressed_exception)
        }
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.exits.traverse(visit)?;
        self.active.traverse(visit)
    }
}
