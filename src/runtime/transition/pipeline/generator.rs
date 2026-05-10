use std::sync::{Arc, Mutex};

use pyo3::PyTraverseError;
use pyo3::exceptions::{PyGeneratorExit, PyRuntimeError, PyStopIteration};
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::PyBool;

use super::pipelines::{
    EnterContinuation, EnterEffect, EnterPoll, EnterProgram, ExitContinuation, ExitEffect,
    ExitPoll, ExitProgram,
};
use super::{ExceptionTriple, ExitItem, ExitOutcome, MixedExitStack};

pub(crate) enum AsyncContextManagerState {
    NotEntered,
    Entering,
    Entered(MixedExitStack),
    Exited,
}

impl AsyncContextManagerState {
    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Self::Entered(exits) = self {
            exits.traverse(visit)?;
        }
        Ok(())
    }
}

pub(crate) enum GeneratorResult {
    Suspended {
        generator: Box<dyn PyGeneratorLike>,
        yielded: Py<PyAny>,
    },
    Completed(Py<PyAny>),
    Failed(PyErr),
}

pub(crate) enum GeneratorCloseResult {
    Closed,
    IgnoredGeneratorExit {
        generator: Box<dyn PyGeneratorLike>,
        error: PyErr,
    },
    Failed(PyErr),
}

enum GeneratorCloseError {
    IgnoredGeneratorExit(PyErr),
    Failed(PyErr),
}

pub(crate) trait PyGeneratorLike: Send {
    fn is_initial(&self) -> bool;

    fn send(self: Box<Self>, py: Python<'_>, value: Py<PyAny>) -> GeneratorResult;

    fn throw(
        self: Box<Self>,
        py: Python<'_>,
        typ: Py<PyAny>,
        val: Option<Py<PyAny>>,
        tb: Option<Py<PyAny>>,
    ) -> GeneratorResult;

    fn close(self: Box<Self>, py: Python<'_>) -> GeneratorCloseResult;

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError>;
}

enum CoroutineResume {
    Send(Py<PyAny>),
    Throw {
        typ: Py<PyAny>,
        val: Option<Py<PyAny>>,
        tb: Option<Py<PyAny>>,
    },
}

enum CoroutineReply {
    Yielded(Py<PyAny>),
    Returned(Py<PyAny>),
    Raised(PyErr),
}

fn stop_iteration_value(py: Python<'_>, err: &PyErr) -> Py<PyAny> {
    err.value(py)
        .getattr("value")
        .map(|v| v.unbind())
        .unwrap_or_else(|_| py.None())
}

fn resume_coroutine(py: Python<'_>, coro: &Py<PyAny>, resume: CoroutineResume) -> CoroutineReply {
    let result = match resume {
        CoroutineResume::Send(value) => coro.call_method1(py, "send", (&value,)),
        CoroutineResume::Throw { typ, val, tb } => {
            let none = py.None();
            let val_ref = val.as_ref().unwrap_or(&none);
            let tb_ref = tb.as_ref().unwrap_or(&none);
            coro.call_method1(py, "throw", (&typ, val_ref, tb_ref))
        }
    };

    match result {
        Ok(yielded) => CoroutineReply::Yielded(yielded),
        Err(error) if error.is_instance_of::<PyStopIteration>(py) => {
            CoroutineReply::Returned(stop_iteration_value(py, &error))
        }
        Err(error) => CoroutineReply::Raised(error),
    }
}

fn classify_coroutine_close_error(py: Python<'_>, error: PyErr) -> GeneratorCloseError {
    // CPython exposes "yielded while handling GeneratorExit" from coroutine.close()
    // only as this RuntimeError; the structural yielded/raised distinction has
    // already been collapsed by the time the public close() call returns.
    if error.is_instance_of::<PyRuntimeError>(py)
        && error
            .value(py)
            .str()
            .and_then(|message| Ok(message.to_str()? == "coroutine ignored GeneratorExit"))
            .unwrap_or(false)
    {
        GeneratorCloseError::IgnoredGeneratorExit(error)
    } else {
        GeneratorCloseError::Failed(error)
    }
}

fn call_sync_exit(py: Python<'_>, context: Py<PyAny>, exc: ExceptionTriple) -> ExitOutcome {
    match context.call_method1(py, "__exit__", (&exc.exc_type, &exc.exc_val, &exc.exc_tb)) {
        Ok(value) => ExitOutcome::Returned(value),
        Err(error) => ExitOutcome::Raised(error),
    }
}

fn call_async_exit(
    py: Python<'_>,
    context: Py<PyAny>,
    exc: ExceptionTriple,
) -> Result<Py<PyAny>, ExitOutcome> {
    context
        .call_method1(py, "__aexit__", (&exc.exc_type, &exc.exc_val, &exc.exc_tb))
        .map_err(ExitOutcome::Raised)
}

fn throw_error(py: Python<'_>, typ: Py<PyAny>, val: Option<Py<PyAny>>) -> PyErr {
    if let Some(value) = val
        && !value.bind(py).is_none()
    {
        return PyErr::from_value(value.into_bound(py));
    }
    PyErr::from_value(typ.into_bound(py))
}

fn generator_exit() -> PyErr {
    PyErr::new::<PyGeneratorExit, _>(())
}

const ENTER_CLEANUP_SUPPRESSED_ERROR_MESSAGE: &str =
    "context manager enter did not produce a value";

pub(crate) fn run_sync_context_enter(
    py: Python<'_>,
    program: EnterProgram,
) -> PyResult<(Py<PyAny>, MixedExitStack)> {
    finish_sync_context_enter_result(py, EnterInterpreter::start(py, program))
}

fn finish_sync_context_enter_result(
    py: Python<'_>,
    result: EnterResult,
) -> PyResult<(Py<PyAny>, MixedExitStack)> {
    match result {
        EnterResult::Suspended { .. } => Err(PyRuntimeError::new_err(
            "sync context manager transition unexpectedly suspended",
        )),
        EnterResult::Completed { result, exits } => Ok((result, exits)),
        EnterResult::Failed(failure) => match failure {
            EnterFailure::Raise(error) => Err(error),
            EnterFailure::RequestExit {
                program,
                original_error,
            } => finish_sync_context_cleanup_result(
                ExitInterpreter::start(py, program),
                original_error,
            ),
        },
    }
}

fn finish_sync_context_cleanup_result(
    result: ExitResult,
    original_error: PyErr,
) -> PyResult<(Py<PyAny>, MixedExitStack)> {
    match result {
        ExitResult::Suspended { .. } => Err(PyRuntimeError::new_err(
            "sync context manager cleanup unexpectedly suspended",
        )),
        ExitResult::Completed { suppressed } => {
            sync_enter_cleanup_completed(suppressed, original_error)
        }
        ExitResult::Failed(error) => Err(error),
    }
}

fn sync_enter_cleanup_completed(
    suppressed: bool,
    original_error: PyErr,
) -> PyResult<(Py<PyAny>, MixedExitStack)> {
    if suppressed {
        Err(PyRuntimeError::new_err(
            ENTER_CLEANUP_SUPPRESSED_ERROR_MESSAGE,
        ))
    } else {
        Err(original_error)
    }
}

pub(crate) fn run_sync_context_exit(py: Python<'_>, program: ExitProgram) -> PyResult<bool> {
    match ExitInterpreter::start(py, program) {
        ExitResult::Suspended { .. } => Err(PyRuntimeError::new_err(
            "sync context manager exit unexpectedly suspended",
        )),
        ExitResult::Completed { suppressed } => Ok(suppressed),
        ExitResult::Failed(error) => Err(error),
    }
}

pub(crate) enum TransitionGenerator {
    Initial(GeneratorProgram),
    Enter {
        interpreter: EnterInterpreter,
        on_completion: EnterOnCompletion,
    },
    Exit {
        interpreter: ExitInterpreter,
        on_completion: ExitOnCompletion,
    },
}

pub(crate) enum EnterOnCompletion {
    ReturnResult,
    StoreAsyncContext {
        wrapper_state: Arc<Mutex<AsyncContextManagerState>>,
    },
}

pub(crate) enum ExitOnCompletion {
    ReturnSuppressed,
    FailedEnterCleanup {
        original_error: PyErr,
        wrapper_state: Option<Arc<Mutex<AsyncContextManagerState>>>,
    },
}

pub(crate) enum GeneratorProgram {
    AwaitableMethod(EnterProgram),
    AsyncContextEnter {
        program: EnterProgram,
        wrapper_state: Arc<Mutex<AsyncContextManagerState>>,
    },
    AsyncExit(ExitProgram),
}

pub(crate) struct EnterInterpreter {
    exits: MixedExitStack,
    suspension: EnterSuspension,
}

enum EnterSuspension {
    AwaitImplementation {
        continuation: EnterContinuation,
        coro: Py<PyAny>,
    },
    EnterAsync {
        context: Py<PyAny>,
        continuation: EnterContinuation,
        coro: Py<PyAny>,
    },
}

enum EnterResult {
    Suspended {
        interpreter: EnterInterpreter,
        yielded: Py<PyAny>,
    },
    Completed {
        result: Py<PyAny>,
        exits: MixedExitStack,
    },
    Failed(EnterFailure),
}

enum EnterFailure {
    RequestExit {
        program: ExitProgram,
        original_error: PyErr,
    },
    Raise(PyErr),
}

pub(crate) struct ExitInterpreter {
    suspension: ExitSuspension,
}

enum ExitSuspension {
    ExitAsync {
        continuation: ExitContinuation,
        coro: Py<PyAny>,
    },
}

enum ExitResult {
    Suspended {
        interpreter: ExitInterpreter,
        yielded: Py<PyAny>,
    },
    Completed {
        suppressed: bool,
    },
    Failed(PyErr),
}

enum CloseOutcome {
    Closed,
    Suspended {
        generator: Box<dyn PyGeneratorLike>,
        error: PyErr,
    },
    Failed(PyErr),
}

enum CloseDrainOutcome {
    Completed,
    Suspended {
        interpreter: Box<ExitInterpreter>,
        error: PyErr,
    },
    Raised(PyErr),
}

impl TransitionGenerator {
    pub(crate) fn awaitable_method(program: EnterProgram) -> Self {
        Self::Initial(GeneratorProgram::AwaitableMethod(program))
    }

    pub(crate) fn async_context_enter(
        program: EnterProgram,
        wrapper_state: Arc<Mutex<AsyncContextManagerState>>,
    ) -> Self {
        Self::Initial(GeneratorProgram::AsyncContextEnter {
            program,
            wrapper_state,
        })
    }

    pub(crate) fn async_exit(program: ExitProgram) -> Self {
        Self::Initial(GeneratorProgram::AsyncExit(program))
    }

    fn start_program(py: Python<'_>, program: GeneratorProgram) -> GeneratorResult {
        match program {
            GeneratorProgram::AwaitableMethod(program) => Self::finish_enter_result(
                py,
                EnterInterpreter::start(py, program),
                EnterOnCompletion::ReturnResult,
            ),
            GeneratorProgram::AsyncContextEnter {
                program,
                wrapper_state,
            } => Self::finish_enter_result(
                py,
                EnterInterpreter::start(py, program),
                EnterOnCompletion::StoreAsyncContext { wrapper_state },
            ),
            GeneratorProgram::AsyncExit(program) => Self::finish_exit_result(
                py,
                ExitInterpreter::start(py, program),
                ExitOnCompletion::ReturnSuppressed,
            ),
        }
    }

    fn finish_enter_result(
        py: Python<'_>,
        result: EnterResult,
        on_completion: EnterOnCompletion,
    ) -> GeneratorResult {
        match result {
            EnterResult::Suspended {
                interpreter,
                yielded,
            } => GeneratorResult::Suspended {
                generator: Box::new(TransitionGenerator::Enter {
                    interpreter,
                    on_completion,
                }),
                yielded,
            },
            EnterResult::Completed { result, exits } => on_completion.completed(result, exits),
            EnterResult::Failed(failure) => match failure {
                EnterFailure::Raise(error) => on_completion.failed(error),
                EnterFailure::RequestExit {
                    program,
                    original_error,
                } => {
                    let exit_on_completion =
                        on_completion.into_failed_enter_cleanup(original_error);
                    Self::finish_exit_result(
                        py,
                        ExitInterpreter::start(py, program),
                        exit_on_completion,
                    )
                }
            },
        }
    }

    fn finish_exit_result(
        py: Python<'_>,
        result: ExitResult,
        on_completion: ExitOnCompletion,
    ) -> GeneratorResult {
        match result {
            ExitResult::Suspended {
                interpreter,
                yielded,
            } => GeneratorResult::Suspended {
                generator: Box::new(TransitionGenerator::Exit {
                    interpreter,
                    on_completion,
                }),
                yielded,
            },
            ExitResult::Completed { suppressed } => on_completion.completed(py, suppressed),
            ExitResult::Failed(error) => on_completion.failed(error),
        }
    }

    fn finish_close_result(result: CloseOutcome) -> GeneratorCloseResult {
        match result {
            CloseOutcome::Closed => GeneratorCloseResult::Closed,
            CloseOutcome::Suspended { generator, error } => {
                GeneratorCloseResult::IgnoredGeneratorExit { generator, error }
            }
            CloseOutcome::Failed(error) => GeneratorCloseResult::Failed(error),
        }
    }

    fn close_enter_interpreter(
        py: Python<'_>,
        interpreter: EnterInterpreter,
        on_completion: EnterOnCompletion,
    ) -> CloseOutcome {
        let EnterInterpreter { exits, suspension } = interpreter;
        match suspension {
            EnterSuspension::AwaitImplementation { continuation, coro } => {
                match coro.call_method0(py, "close") {
                    Ok(_) => Self::close_enter_without_active(py, exits, on_completion),
                    Err(error) => match classify_coroutine_close_error(py, error) {
                        GeneratorCloseError::IgnoredGeneratorExit(error) => {
                            CloseOutcome::Suspended {
                                generator: Box::new(TransitionGenerator::Enter {
                                    interpreter: EnterInterpreter {
                                        exits,
                                        suspension: EnterSuspension::AwaitImplementation {
                                            continuation,
                                            coro,
                                        },
                                    },
                                    on_completion,
                                }),
                                error,
                            }
                        }
                        GeneratorCloseError::Failed(error) => {
                            on_completion.mark_exited();
                            CloseOutcome::Failed(error)
                        }
                    },
                }
            }
            EnterSuspension::EnterAsync {
                context,
                continuation,
                coro,
            } => match coro.call_method0(py, "close") {
                Ok(_) => Self::close_enter_without_active(py, exits, on_completion),
                Err(error) => match classify_coroutine_close_error(py, error) {
                    GeneratorCloseError::IgnoredGeneratorExit(error) => CloseOutcome::Suspended {
                        generator: Box::new(TransitionGenerator::Enter {
                            interpreter: EnterInterpreter {
                                exits,
                                suspension: EnterSuspension::EnterAsync {
                                    context,
                                    continuation,
                                    coro,
                                },
                            },
                            on_completion,
                        }),
                        error,
                    },
                    GeneratorCloseError::Failed(error) => {
                        on_completion.mark_exited();
                        CloseOutcome::Failed(error)
                    }
                },
            },
        }
    }

    fn close_enter_without_active(
        py: Python<'_>,
        exits: MixedExitStack,
        on_completion: EnterOnCompletion,
    ) -> CloseOutcome {
        if exits.is_empty() {
            on_completion.mark_exited();
            return CloseOutcome::Closed;
        }

        let generator_exit = generator_exit();
        let active = ExceptionTriple::from_error(py, &generator_exit);
        let exit_on_completion = on_completion.into_failed_enter_cleanup(generator_exit);
        Self::finish_close_drain_result(
            close_exit_without_suspension(py, ExitProgram::new(py, exits, active)),
            exit_on_completion,
        )
    }

    fn close_exit_interpreter(
        py: Python<'_>,
        interpreter: ExitInterpreter,
        on_completion: ExitOnCompletion,
    ) -> CloseOutcome {
        Self::finish_close_drain_result(interpreter.close(py), on_completion)
    }

    fn finish_close_drain_result(
        result: CloseDrainOutcome,
        on_completion: ExitOnCompletion,
    ) -> CloseOutcome {
        match result {
            CloseDrainOutcome::Completed => {
                on_completion.mark_exited();
                CloseOutcome::Closed
            }
            CloseDrainOutcome::Suspended { interpreter, error } => CloseOutcome::Suspended {
                generator: Box::new(TransitionGenerator::Exit {
                    interpreter: *interpreter,
                    on_completion,
                }),
                error,
            },
            CloseDrainOutcome::Raised(error) => {
                on_completion.mark_exited();
                CloseOutcome::Failed(error)
            }
        }
    }
}

impl PyGeneratorLike for TransitionGenerator {
    fn is_initial(&self) -> bool {
        matches!(self, Self::Initial(_))
    }

    fn send(self: Box<Self>, py: Python<'_>, value: Py<PyAny>) -> GeneratorResult {
        match *self {
            TransitionGenerator::Initial(program) => {
                TransitionGenerator::start_program(py, program)
            }
            TransitionGenerator::Enter {
                interpreter,
                on_completion,
            } => TransitionGenerator::finish_enter_result(
                py,
                interpreter.resume(py, CoroutineResume::Send(value)),
                on_completion,
            ),
            TransitionGenerator::Exit {
                interpreter,
                on_completion,
            } => TransitionGenerator::finish_exit_result(
                py,
                interpreter.resume(py, CoroutineResume::Send(value)),
                on_completion,
            ),
        }
    }

    fn throw(
        self: Box<Self>,
        py: Python<'_>,
        typ: Py<PyAny>,
        val: Option<Py<PyAny>>,
        tb: Option<Py<PyAny>>,
    ) -> GeneratorResult {
        match *self {
            TransitionGenerator::Initial(program) => program.throw_before_start(py, typ, val),
            TransitionGenerator::Enter {
                interpreter,
                on_completion,
            } => TransitionGenerator::finish_enter_result(
                py,
                interpreter.resume(py, CoroutineResume::Throw { typ, val, tb }),
                on_completion,
            ),
            TransitionGenerator::Exit {
                interpreter,
                on_completion,
            } => TransitionGenerator::finish_exit_result(
                py,
                interpreter.resume(py, CoroutineResume::Throw { typ, val, tb }),
                on_completion,
            ),
        }
    }

    fn close(self: Box<Self>, py: Python<'_>) -> GeneratorCloseResult {
        let result = match *self {
            TransitionGenerator::Initial(program) => {
                program.close_before_start();
                CloseOutcome::Closed
            }
            TransitionGenerator::Enter {
                interpreter,
                on_completion,
            } => TransitionGenerator::close_enter_interpreter(py, interpreter, on_completion),
            TransitionGenerator::Exit {
                interpreter,
                on_completion,
            } => TransitionGenerator::close_exit_interpreter(py, interpreter, on_completion),
        };
        TransitionGenerator::finish_close_result(result)
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match self {
            TransitionGenerator::Initial(program) => program.traverse(visit)?,
            TransitionGenerator::Enter { interpreter, .. } => interpreter.traverse(visit)?,
            TransitionGenerator::Exit { interpreter, .. } => interpreter.traverse(visit)?,
        }
        Ok(())
    }
}

impl GeneratorProgram {
    fn throw_before_start(
        self,
        py: Python<'_>,
        typ: Py<PyAny>,
        val: Option<Py<PyAny>>,
    ) -> GeneratorResult {
        if let Self::AsyncContextEnter { wrapper_state, .. } = self {
            *wrapper_state.lock().expect("poisoned") = AsyncContextManagerState::Exited;
        }
        GeneratorResult::Failed(throw_error(py, typ, val))
    }

    fn close_before_start(self) {
        if let Self::AsyncContextEnter { wrapper_state, .. } = self {
            *wrapper_state.lock().expect("poisoned") = AsyncContextManagerState::Exited;
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match self {
            Self::AwaitableMethod(program) | Self::AsyncContextEnter { program, .. } => {
                program.traverse(visit)
            }
            Self::AsyncExit(program) => program.traverse(visit),
        }
    }
}

impl EnterOnCompletion {
    fn completed(self, result: Py<PyAny>, exits: MixedExitStack) -> GeneratorResult {
        match self {
            Self::ReturnResult => GeneratorResult::Completed(result),
            Self::StoreAsyncContext { wrapper_state } => {
                *wrapper_state.lock().expect("poisoned") = AsyncContextManagerState::Entered(exits);
                GeneratorResult::Completed(result)
            }
        }
    }

    fn failed(self, error: PyErr) -> GeneratorResult {
        self.mark_exited();
        GeneratorResult::Failed(error)
    }

    fn mark_exited(&self) {
        if let Self::StoreAsyncContext { wrapper_state } = self {
            *wrapper_state.lock().expect("poisoned") = AsyncContextManagerState::Exited;
        }
    }

    fn into_failed_enter_cleanup(self, original_error: PyErr) -> ExitOnCompletion {
        let wrapper_state = match self {
            Self::ReturnResult => None,
            Self::StoreAsyncContext { wrapper_state } => Some(wrapper_state),
        };
        ExitOnCompletion::FailedEnterCleanup {
            original_error,
            wrapper_state,
        }
    }
}

impl ExitOnCompletion {
    fn completed(self, py: Python<'_>, suppressed: bool) -> GeneratorResult {
        match self {
            Self::ReturnSuppressed => {
                let value = PyBool::new(py, suppressed).to_owned().into_any().unbind();
                GeneratorResult::Completed(value)
            }
            Self::FailedEnterCleanup {
                original_error,
                wrapper_state,
            } => {
                if let Some(wrapper_state) = wrapper_state {
                    *wrapper_state.lock().expect("poisoned") = AsyncContextManagerState::Exited;
                }
                if suppressed {
                    GeneratorResult::Failed(PyRuntimeError::new_err(
                        ENTER_CLEANUP_SUPPRESSED_ERROR_MESSAGE,
                    ))
                } else {
                    GeneratorResult::Failed(original_error)
                }
            }
        }
    }

    fn failed(self, error: PyErr) -> GeneratorResult {
        self.mark_exited();
        GeneratorResult::Failed(error)
    }

    fn mark_exited(&self) {
        if let Self::FailedEnterCleanup {
            wrapper_state: Some(wrapper_state),
            ..
        } = self
        {
            *wrapper_state.lock().expect("poisoned") = AsyncContextManagerState::Exited;
        }
    }
}

impl EnterInterpreter {
    fn start(py: Python<'_>, program: EnterProgram) -> EnterResult {
        Self::run_program(py, MixedExitStack::new(), program)
    }

    fn resume(self, py: Python<'_>, resume: CoroutineResume) -> EnterResult {
        let Self { exits, suspension } = self;
        match suspension {
            EnterSuspension::AwaitImplementation { continuation, coro } => {
                Self::resume_await_implementation(py, exits, continuation, coro, resume)
            }
            EnterSuspension::EnterAsync {
                context,
                continuation,
                coro,
            } => Self::resume_enter_async(py, exits, context, continuation, coro, resume),
        }
    }

    fn run_program(
        py: Python<'_>,
        mut exits: MixedExitStack,
        mut program: EnterProgram,
    ) -> EnterResult {
        loop {
            match program.poll(py) {
                Ok(EnterPoll::Effect(EnterEffect::EnterSync {
                    context,
                    continuation,
                })) => match context.call_method0(py, "__enter__") {
                    Ok(entered) => {
                        exits.push_sync(context);
                        program = continuation.resume(py, entered);
                    }
                    Err(error) => return Self::failed(py, exits, error),
                },
                Ok(EnterPoll::Effect(EnterEffect::Await {
                    awaitable,
                    continuation,
                })) => {
                    return Self::resume_await_implementation(
                        py,
                        exits,
                        continuation,
                        awaitable,
                        CoroutineResume::Send(py.None()),
                    );
                }
                Ok(EnterPoll::Effect(EnterEffect::EnterAsync {
                    context,
                    continuation,
                })) => match context.call_method0(py, "__aenter__") {
                    Ok(coro) => {
                        return Self::resume_enter_async(
                            py,
                            exits,
                            context,
                            continuation,
                            coro,
                            CoroutineResume::Send(py.None()),
                        );
                    }
                    Err(error) => return Self::failed(py, exits, error),
                },
                Ok(EnterPoll::Completed(result)) => {
                    return EnterResult::Completed { result, exits };
                }
                Err(error) => return Self::failed(py, exits, error),
            }
        }
    }

    fn resume_await_implementation(
        py: Python<'_>,
        exits: MixedExitStack,
        continuation: EnterContinuation,
        coro: Py<PyAny>,
        resume: CoroutineResume,
    ) -> EnterResult {
        match resume_coroutine(py, &coro, resume) {
            CoroutineReply::Yielded(yielded) => EnterResult::Suspended {
                interpreter: EnterInterpreter {
                    exits,
                    suspension: EnterSuspension::AwaitImplementation { continuation, coro },
                },
                yielded,
            },
            CoroutineReply::Returned(value) => {
                Self::run_program(py, exits, continuation.resume(py, value))
            }
            CoroutineReply::Raised(error) => Self::failed(py, exits, error),
        }
    }

    fn resume_enter_async(
        py: Python<'_>,
        mut exits: MixedExitStack,
        context: Py<PyAny>,
        continuation: EnterContinuation,
        coro: Py<PyAny>,
        resume: CoroutineResume,
    ) -> EnterResult {
        match resume_coroutine(py, &coro, resume) {
            CoroutineReply::Yielded(yielded) => EnterResult::Suspended {
                interpreter: EnterInterpreter {
                    exits,
                    suspension: EnterSuspension::EnterAsync {
                        context,
                        continuation,
                        coro,
                    },
                },
                yielded,
            },
            CoroutineReply::Returned(value) => {
                exits.push_item(ExitItem::Async(context));
                Self::run_program(py, exits, continuation.resume(py, value))
            }
            CoroutineReply::Raised(error) => Self::failed(py, exits, error),
        }
    }

    fn failed(py: Python<'_>, exits: MixedExitStack, error: PyErr) -> EnterResult {
        if exits.is_empty() {
            return EnterResult::Failed(EnterFailure::Raise(error));
        }

        let active = ExceptionTriple::from_error(py, &error);
        EnterResult::Failed(EnterFailure::RequestExit {
            program: ExitProgram::new(py, exits, active),
            original_error: error,
        })
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.exits.traverse(visit)?;
        match &self.suspension {
            EnterSuspension::AwaitImplementation { continuation, coro } => {
                continuation.traverse(visit)?;
                visit.call(coro)?;
            }
            EnterSuspension::EnterAsync {
                context,
                continuation,
                coro,
            } => {
                continuation.traverse(visit)?;
                visit.call(context)?;
                visit.call(coro)?;
            }
        }
        Ok(())
    }
}

impl ExitInterpreter {
    fn start(py: Python<'_>, program: ExitProgram) -> ExitResult {
        Self::run_program(py, program)
    }

    fn resume(self, py: Python<'_>, resume: CoroutineResume) -> ExitResult {
        let Self { suspension } = self;
        let ExitSuspension::ExitAsync { continuation, coro } = suspension;
        Self::resume_exit_async(py, continuation, coro, resume)
    }

    fn run_program(py: Python<'_>, mut program: ExitProgram) -> ExitResult {
        loop {
            match program.poll(py) {
                ExitPoll::Effect(ExitEffect::ExitSync {
                    context,
                    exc,
                    continuation,
                }) => {
                    let outcome = call_sync_exit(py, context, exc);
                    program = continuation.resume(py, outcome);
                }
                ExitPoll::Effect(ExitEffect::ExitAsync {
                    context,
                    exc,
                    continuation,
                }) => match call_async_exit(py, context, exc) {
                    Ok(coro) => {
                        return Self::resume_exit_async(
                            py,
                            continuation,
                            coro,
                            CoroutineResume::Send(py.None()),
                        );
                    }
                    Err(outcome) => program = continuation.resume(py, outcome),
                },
                ExitPoll::Completed { suppressed } => {
                    return ExitResult::Completed { suppressed };
                }
                ExitPoll::Failed(error) => return ExitResult::Failed(error),
            }
        }
    }

    fn resume_exit_async(
        py: Python<'_>,
        continuation: ExitContinuation,
        coro: Py<PyAny>,
        resume: CoroutineResume,
    ) -> ExitResult {
        match resume_coroutine(py, &coro, resume) {
            CoroutineReply::Yielded(yielded) => ExitResult::Suspended {
                interpreter: ExitInterpreter {
                    suspension: ExitSuspension::ExitAsync { continuation, coro },
                },
                yielded,
            },
            CoroutineReply::Returned(value) => {
                Self::run_program(py, continuation.resume(py, ExitOutcome::Returned(value)))
            }
            CoroutineReply::Raised(error) => {
                Self::run_program(py, continuation.resume(py, ExitOutcome::Raised(error)))
            }
        }
    }

    fn close(self, py: Python<'_>) -> CloseDrainOutcome {
        let Self { suspension } = self;
        let ExitSuspension::ExitAsync { continuation, coro } = suspension;
        close_active_async_exit_and_resume_drainer(py, continuation, coro)
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        let ExitSuspension::ExitAsync { continuation, coro } = &self.suspension;
        continuation.traverse(visit)?;
        visit.call(coro)
    }
}

fn close_active_async_exit_and_resume_drainer(
    py: Python<'_>,
    continuation: ExitContinuation,
    active: Py<PyAny>,
) -> CloseDrainOutcome {
    let outcome = match active.call_method0(py, "close") {
        Ok(_) => ExitOutcome::Raised(generator_exit()),
        Err(error) => match classify_coroutine_close_error(py, error) {
            GeneratorCloseError::IgnoredGeneratorExit(error) => {
                return CloseDrainOutcome::Suspended {
                    interpreter: Box::new(ExitInterpreter {
                        suspension: ExitSuspension::ExitAsync {
                            continuation,
                            coro: active,
                        },
                    }),
                    error,
                };
            }
            GeneratorCloseError::Failed(error) => ExitOutcome::Raised(error),
        },
    };

    close_exit_without_suspension(py, continuation.resume(py, outcome))
}

fn close_exit_without_suspension(py: Python<'_>, mut program: ExitProgram) -> CloseDrainOutcome {
    loop {
        match program.poll(py) {
            ExitPoll::Effect(ExitEffect::ExitSync {
                context,
                exc,
                continuation,
            }) => {
                let outcome = call_sync_exit(py, context, exc);
                program = continuation.resume(py, outcome);
            }
            ExitPoll::Effect(ExitEffect::ExitAsync {
                context,
                exc,
                continuation,
            }) => match call_async_exit(py, context, exc) {
                Ok(coro) => match resume_coroutine(py, &coro, CoroutineResume::Send(py.None())) {
                    CoroutineReply::Yielded(_) => {
                        return CloseDrainOutcome::Suspended {
                            interpreter: Box::new(ExitInterpreter {
                                suspension: ExitSuspension::ExitAsync { continuation, coro },
                            }),
                            error: PyRuntimeError::new_err("coroutine ignored GeneratorExit"),
                        };
                    }
                    CoroutineReply::Returned(value) => {
                        program = continuation.resume(py, ExitOutcome::Returned(value));
                    }
                    CoroutineReply::Raised(error) => {
                        program = continuation.resume(py, ExitOutcome::Raised(error));
                    }
                },
                Err(outcome) => {
                    program = continuation.resume(py, outcome);
                }
            },
            ExitPoll::Completed { .. } => {
                return CloseDrainOutcome::Completed;
            }
            ExitPoll::Failed(error) => {
                if error.is_instance_of::<PyGeneratorExit>(py) {
                    return CloseDrainOutcome::Completed;
                }
                return CloseDrainOutcome::Raised(error);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use pyo3::exceptions::PyValueError;
    use pyo3::types::{PyModule, PyString};

    use super::*;
    use crate::compile::flatten::tests as flatten_tests;
    use crate::compile::flatten::{ExecutionMethodImplementation, ExecutionNode};
    use crate::runtime::executor::ContextData;
    use crate::runtime::resources::RuntimeResources;
    use crate::runtime::transition::pipeline::pipelines::PipelineCommon;
    use crate::types::WrapperKind;

    const TEST_MODULE_NAME: &str = "_inlay_generator_protocol_tests";
    const TESTDATA_DIR: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/runtime/transition/pipeline/testdata"
    );

    fn with_python(test: impl FnOnce(Python<'_>)) {
        Python::initialize();
        Python::attach(test);
    }

    fn test_module(py: Python<'_>) -> Bound<'_, PyModule> {
        let sys = PyModule::import(py, "sys").expect("sys should import");
        sys.getattr("path")
            .expect("sys.path should exist")
            .call_method1("insert", (0, TESTDATA_DIR))
            .expect("testdata path should be importable");

        let module = PyModule::import(py, TEST_MODULE_NAME).expect("test module should import");
        module
            .getattr("reset_log")
            .expect("reset_log should exist")
            .call0()
            .expect("reset_log should succeed");
        module
    }

    fn function(module: &Bound<'_, PyModule>, name: &str) -> Py<PyAny> {
        module
            .getattr(name)
            .unwrap_or_else(|_| panic!("missing test function {name}"))
            .unbind()
    }

    fn call_context(module: &Bound<'_, PyModule>, name: &str) -> Py<PyAny> {
        module
            .getattr(name)
            .unwrap_or_else(|_| panic!("missing test context factory {name}"))
            .call0()
            .expect("context factory should succeed")
            .unbind()
    }

    fn string(py: Python<'_>, value: &str) -> Py<PyAny> {
        PyString::new(py, value).into_any().unbind()
    }

    fn value_error_type(py: Python<'_>) -> Py<PyAny> {
        py.get_type::<PyValueError>().into_any().unbind()
    }

    fn value_error_value(py: Python<'_>, message: &str) -> Py<PyAny> {
        PyValueError::new_err(message.to_string())
            .value(py)
            .clone()
            .into_any()
            .unbind()
    }

    fn extract_string(py: Python<'_>, value: &Py<PyAny>) -> String {
        value.bind(py).extract().expect("value should be a string")
    }

    fn extract_bool(py: Python<'_>, value: &Py<PyAny>) -> bool {
        value.bind(py).extract().expect("value should be a bool")
    }

    fn log(module: &Bound<'_, PyModule>) -> Vec<String> {
        module
            .getattr("LOG")
            .expect("LOG should exist")
            .extract()
            .expect("LOG should be a string list")
    }

    fn expect_suspended(
        py: Python<'_>,
        result: GeneratorResult,
        expected: &str,
    ) -> Box<dyn PyGeneratorLike> {
        match result {
            GeneratorResult::Suspended { generator, yielded } => {
                assert_eq!(extract_string(py, &yielded), expected);
                generator
            }
            GeneratorResult::Completed(_) => panic!("expected suspension, got completion"),
            GeneratorResult::Failed(error) => panic!("expected suspension, got {error:?}"),
        }
    }

    fn expect_completed(result: GeneratorResult) -> Py<PyAny> {
        match result {
            GeneratorResult::Completed(value) => value,
            GeneratorResult::Suspended { .. } => panic!("expected completion, got suspension"),
            GeneratorResult::Failed(error) => panic!("expected completion, got {error:?}"),
        }
    }

    fn expect_failed(result: GeneratorResult) -> PyErr {
        match result {
            GeneratorResult::Failed(error) => error,
            GeneratorResult::Suspended { .. } => panic!("expected failure, got suspension"),
            GeneratorResult::Completed(_) => panic!("expected failure, got completion"),
        }
    }

    fn assert_closed(result: GeneratorCloseResult) {
        match result {
            GeneratorCloseResult::Closed => {}
            GeneratorCloseResult::IgnoredGeneratorExit { error, .. } => {
                panic!("expected closed, got ignored GeneratorExit: {error:?}")
            }
            GeneratorCloseResult::Failed(error) => panic!("expected closed, got {error:?}"),
        }
    }

    fn method_implementation(
        implementation: Py<PyAny>,
        return_wrapper: WrapperKind,
        result_source_index: usize,
    ) -> ExecutionMethodImplementation {
        ExecutionMethodImplementation {
            implementation: Arc::new(implementation),
            bound_to: None,
            params: Vec::new(),
            return_wrapper,
            result_source: Some(flatten_tests::execution_source_node_id(result_source_index)),
        }
    }

    fn pipeline_common(
        root_index: usize,
        node_count: usize,
        implementations: Vec<ExecutionMethodImplementation>,
    ) -> PipelineCommon {
        PipelineCommon::new(
            ContextData {
                graph: Arc::new(flatten_tests::execution_graph(
                    (0..node_count).map(|_| ExecutionNode::Constant).collect(),
                )),
                root_node: flatten_tests::execution_node_id(root_index),
            },
            RuntimeResources::empty(),
            implementations,
        )
    }

    fn awaitable_pipeline(implementation: Py<PyAny>, result_source_index: usize) -> EnterProgram {
        EnterProgram::new(pipeline_common(
            result_source_index,
            result_source_index + 1,
            vec![method_implementation(
                implementation,
                WrapperKind::Awaitable,
                result_source_index,
            )],
        ))
    }

    fn async_context_enter_pipeline(
        root_index: usize,
        node_count: usize,
        implementations: Vec<ExecutionMethodImplementation>,
    ) -> EnterProgram {
        EnterProgram::new(pipeline_common(root_index, node_count, implementations))
    }

    fn active_value_error(py: Python<'_>) -> ExceptionTriple {
        let error = PyValueError::new_err("active");
        ExceptionTriple::from_error(py, &error)
    }

    fn assert_async_context_entered(state: &Arc<Mutex<AsyncContextManagerState>>) {
        match &*state.lock().expect("poisoned") {
            AsyncContextManagerState::Entered(_) => {}
            AsyncContextManagerState::NotEntered
            | AsyncContextManagerState::Entering
            | AsyncContextManagerState::Exited => panic!("async context should be entered"),
        }
    }

    fn assert_async_context_exited(state: &Arc<Mutex<AsyncContextManagerState>>) {
        match &*state.lock().expect("poisoned") {
            AsyncContextManagerState::Exited => {}
            AsyncContextManagerState::NotEntered
            | AsyncContextManagerState::Entering
            | AsyncContextManagerState::Entered(_) => panic!("async context should be exited"),
        }
    }

    #[test]
    fn async_method_generator_send_resumes_awaitable() {
        with_python(|py| {
            let module = test_module(py);
            let pipeline = awaitable_pipeline(function(&module, "async_method_send_impl"), 0);
            let generator: Box<dyn PyGeneratorLike> =
                Box::new(TransitionGenerator::awaitable_method(pipeline));

            let generator =
                expect_suspended(py, generator.send(py, py.None()), "method-send-pause");
            let value = expect_completed(generator.send(py, string(py, "resume")));

            assert_eq!(extract_string(py, &value), "method-send-done:resume");
        });
    }

    #[test]
    fn async_method_generator_throw_resumes_awaitable() {
        with_python(|py| {
            let module = test_module(py);
            let pipeline = awaitable_pipeline(function(&module, "async_method_throw_impl"), 0);
            let generator: Box<dyn PyGeneratorLike> =
                Box::new(TransitionGenerator::awaitable_method(pipeline));

            let generator =
                expect_suspended(py, generator.send(py, py.None()), "method-throw-pause");
            let value = expect_completed(generator.throw(
                py,
                value_error_type(py),
                Some(value_error_value(py, "method boom")),
                None,
            ));

            assert_eq!(extract_string(py, &value), "method-throw-done:method boom");
        });
    }

    #[test]
    fn async_method_generator_close_closes_awaitable() {
        with_python(|py| {
            let module = test_module(py);
            let pipeline = awaitable_pipeline(function(&module, "async_method_close_impl"), 0);
            let generator: Box<dyn PyGeneratorLike> =
                Box::new(TransitionGenerator::awaitable_method(pipeline));

            let generator =
                expect_suspended(py, generator.send(py, py.None()), "method-close-pause");
            assert_closed(generator.close(py));

            assert_eq!(log(&module), vec!["method-close"]);
        });
    }

    #[test]
    fn async_context_enter_generator_send_resumes_aenter() {
        with_python(|py| {
            let module = test_module(py);
            let state = Arc::new(Mutex::new(AsyncContextManagerState::Entering));
            let pipeline = async_context_enter_pipeline(
                0,
                1,
                vec![method_implementation(
                    function(&module, "async_context_send_impl"),
                    WrapperKind::AsyncContextManager,
                    0,
                )],
            );
            let generator: Box<dyn PyGeneratorLike> = Box::new(
                TransitionGenerator::async_context_enter(pipeline, Arc::clone(&state)),
            );

            let generator =
                expect_suspended(py, generator.send(py, py.None()), "context-enter-pause");
            let value = expect_completed(generator.send(py, string(py, "resume")));

            assert_eq!(extract_string(py, &value), "context-entered");
            assert_async_context_entered(&state);
            assert_eq!(log(&module), vec!["context-enter-resume:resume"]);
        });
    }

    #[test]
    fn async_context_enter_generator_throw_drains_registered_exits() {
        with_python(|py| {
            let module = test_module(py);
            let state = Arc::new(Mutex::new(AsyncContextManagerState::Entering));
            let pipeline = async_context_enter_pipeline(
                1,
                2,
                vec![
                    method_implementation(
                        function(&module, "sync_context_impl"),
                        WrapperKind::ContextManager,
                        0,
                    ),
                    method_implementation(
                        function(&module, "context_throw_awaitable_impl"),
                        WrapperKind::Awaitable,
                        1,
                    ),
                ],
            );
            let generator: Box<dyn PyGeneratorLike> = Box::new(
                TransitionGenerator::async_context_enter(pipeline, Arc::clone(&state)),
            );

            let generator =
                expect_suspended(py, generator.send(py, py.None()), "context-throw-pause");
            let error = expect_failed(generator.throw(
                py,
                value_error_type(py),
                Some(value_error_value(py, "context boom")),
                None,
            ));

            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(error.value(py).to_string(), "context boom");
            assert_async_context_exited(&state);
            assert_eq!(log(&module), vec!["sync-enter", "sync-exit:ValueError"]);
        });
    }

    #[test]
    fn async_context_enter_generator_close_drains_registered_exits() {
        with_python(|py| {
            let module = test_module(py);
            let state = Arc::new(Mutex::new(AsyncContextManagerState::Entering));
            let pipeline = async_context_enter_pipeline(
                1,
                2,
                vec![
                    method_implementation(
                        function(&module, "sync_context_impl"),
                        WrapperKind::ContextManager,
                        0,
                    ),
                    method_implementation(
                        function(&module, "context_close_awaitable_impl"),
                        WrapperKind::Awaitable,
                        1,
                    ),
                ],
            );
            let generator: Box<dyn PyGeneratorLike> = Box::new(
                TransitionGenerator::async_context_enter(pipeline, Arc::clone(&state)),
            );

            let generator =
                expect_suspended(py, generator.send(py, py.None()), "context-close-pause");
            assert_closed(generator.close(py));

            assert_async_context_exited(&state);
            assert_eq!(
                log(&module),
                vec![
                    "sync-enter",
                    "context-await-close",
                    "sync-exit:GeneratorExit"
                ]
            );
        });
    }

    #[test]
    fn async_exit_generator_send_resumes_aexit() {
        with_python(|py| {
            let module = test_module(py);
            let mut exits = MixedExitStack::new();
            exits.push_item(ExitItem::Async(call_context(
                &module,
                "async_exit_send_context",
            )));
            let pipeline = ExitProgram::new(py, exits, active_value_error(py));
            let generator: Box<dyn PyGeneratorLike> =
                Box::new(TransitionGenerator::async_exit(pipeline));

            let generator =
                expect_suspended(py, generator.send(py, py.None()), "async-exit-send-pause");
            let value = expect_completed(generator.send(py, string(py, "resume")));

            assert!(extract_bool(py, &value));
            assert_eq!(
                log(&module),
                vec![
                    "async-exit-send-start:ValueError",
                    "async-exit-send-resume:resume"
                ]
            );
        });
    }

    #[test]
    fn async_exit_generator_throw_resumes_aexit() {
        with_python(|py| {
            let module = test_module(py);
            let mut exits = MixedExitStack::new();
            exits.push_item(ExitItem::Async(call_context(
                &module,
                "async_exit_throw_context",
            )));
            let pipeline = ExitProgram::new(py, exits, active_value_error(py));
            let generator: Box<dyn PyGeneratorLike> =
                Box::new(TransitionGenerator::async_exit(pipeline));

            let generator =
                expect_suspended(py, generator.send(py, py.None()), "async-exit-throw-pause");
            let value = expect_completed(generator.throw(
                py,
                value_error_type(py),
                Some(value_error_value(py, "exit boom")),
                None,
            ));

            assert!(extract_bool(py, &value));
            assert_eq!(
                log(&module),
                vec![
                    "async-exit-throw-start:ValueError",
                    "async-exit-throw:exit boom"
                ]
            );
        });
    }

    #[test]
    fn async_exit_generator_close_closes_active_aexit_and_drains_outer_exits() {
        with_python(|py| {
            let module = test_module(py);
            let mut exits = MixedExitStack::new();
            exits.push_sync(call_context(&module, "sync_exit_outer_context"));
            exits.push_item(ExitItem::Async(call_context(
                &module,
                "async_exit_close_inner_context",
            )));
            let pipeline = ExitProgram::new(py, exits, active_value_error(py));
            let generator: Box<dyn PyGeneratorLike> =
                Box::new(TransitionGenerator::async_exit(pipeline));

            let generator =
                expect_suspended(py, generator.send(py, py.None()), "async-exit-close-pause");
            assert_closed(generator.close(py));

            assert_eq!(
                log(&module),
                vec![
                    "async-exit-close-start:ValueError",
                    "async-exit-close-inner",
                    "sync-exit-outer:GeneratorExit"
                ]
            );
        });
    }
}
