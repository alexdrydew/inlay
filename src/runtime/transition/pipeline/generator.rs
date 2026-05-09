use std::sync::{Arc, Mutex};

use pyo3::PyTraverseError;
use pyo3::exceptions::{PyGeneratorExit, PyRuntimeError, PyStopIteration};
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::PyBool;

use super::exits::{ExceptionTriple, ExitItem, MixedExitStack, SyncExitStack};
use super::pipelines::{
    AsyncContextManagerEnterPipeline, AsyncExitDrainerPipeline, AwaitableMethodPipeline,
    ContextManagerEnterPipeline, SyncExitDrainerPipeline,
};
use super::step::{
    AsyncContextManagerEnterStep, AsyncExitStep, AwaitableMethodStep, ContextManagerEnterStep,
    ExitDrainCompletion, ExitOutcome, SyncExitStep,
};

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
    Suspended(Py<PyAny>),
    Completed(Py<PyAny>),
    Failed(PyErr),
}

pub(crate) enum GeneratorCloseResult {
    Closed,
    IgnoredGeneratorExit(PyErr),
    Failed(PyErr),
}

enum GeneratorCloseError {
    IgnoredGeneratorExit(PyErr),
    Failed(PyErr),
}

pub(crate) trait PyGeneratorLike: Send {
    fn is_initial(&self) -> bool;

    fn send(&mut self, py: Python<'_>, value: Py<PyAny>) -> GeneratorResult;

    fn throw(
        &mut self,
        py: Python<'_>,
        typ: Py<PyAny>,
        val: Option<Py<PyAny>>,
        tb: Option<Py<PyAny>>,
    ) -> GeneratorResult;

    fn close(&mut self, py: Python<'_>) -> GeneratorCloseResult;

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

enum ResumeAction {
    Step,
    Coroutine {
        coro: Py<PyAny>,
        resume: CoroutineResume,
    },
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

pub(crate) struct SyncContextEnterRunner {
    pipeline: ContextManagerEnterPipeline,
    exits: SyncExitStack,
}

impl SyncContextEnterRunner {
    pub(crate) fn new(pipeline: ContextManagerEnterPipeline) -> Self {
        Self {
            pipeline,
            exits: SyncExitStack::new(),
        }
    }

    pub(crate) fn run(mut self, py: Python<'_>) -> PyResult<(Py<PyAny>, SyncExitStack)> {
        let mut input = None;
        loop {
            match self.pipeline.next(py, input.take()) {
                Ok(ContextManagerEnterStep::EnterSync(context)) => {
                    match context.call_method0(py, "__enter__") {
                        Ok(entered) => {
                            self.exits.push(context);
                            input = Some(entered);
                        }
                        Err(error) => return self.cleanup_after_enter_error(py, error),
                    }
                }
                Ok(ContextManagerEnterStep::Done(value)) => return Ok((value, self.exits)),
                Err(error) => return self.cleanup_after_enter_error(py, error),
            }
        }
    }

    fn cleanup_after_enter_error(
        mut self,
        py: Python<'_>,
        error: PyErr,
    ) -> PyResult<(Py<PyAny>, SyncExitStack)> {
        let active = ExceptionTriple::from_error(py, &error);
        let mut drainer = SyncExitDrainerPipeline::new(py, std::mem::take(&mut self.exits), active);
        let mut outcome = None;

        loop {
            match drainer.next(py, outcome.take()) {
                SyncExitStep::ExitSync { context, exc } => {
                    outcome = Some(call_sync_exit(py, context, exc));
                }
                SyncExitStep::Done(ExitDrainCompletion::Raise(cleanup_error)) => {
                    return Err(cleanup_error);
                }
                SyncExitStep::Done(ExitDrainCompletion::Return(false)) => return Err(error),
                SyncExitStep::Done(ExitDrainCompletion::Return(true)) => {
                    return Err(PyRuntimeError::new_err(
                        "context manager enter did not produce a value",
                    ));
                }
            }
        }
    }
}

pub(crate) struct SyncContextExitRunner {
    pipeline: SyncExitDrainerPipeline,
}

impl SyncContextExitRunner {
    pub(crate) fn new(pipeline: SyncExitDrainerPipeline) -> Self {
        Self { pipeline }
    }

    pub(crate) fn run(mut self, py: Python<'_>) -> PyResult<bool> {
        let mut outcome = None;
        loop {
            match self.pipeline.next(py, outcome.take()) {
                SyncExitStep::ExitSync { context, exc } => {
                    outcome = Some(call_sync_exit(py, context, exc));
                }
                SyncExitStep::Done(ExitDrainCompletion::Raise(error)) => return Err(error),
                SyncExitStep::Done(ExitDrainCompletion::Return(suppressed)) => {
                    return Ok(suppressed);
                }
            }
        }
    }
}

enum GeneratorState {
    Initial,
    Executing,
    Done,
}

struct CleanupState {
    drainer: AsyncExitDrainerPipeline,
    original_error: PyErr,
}

enum AsyncContextEnterState {
    Initial,
    SuspendedAwait(Py<PyAny>),
    SuspendedEnterAsync {
        context: Py<PyAny>,
        coro: Py<PyAny>,
    },
    SuspendedCleanup {
        cleanup: CleanupState,
        coro: Py<PyAny>,
    },
    Done,
}

enum AsyncContextEnterAction {
    Complete(GeneratorResult),
    StepEnter,
    StepCleanup(CleanupState),
    Await {
        coro: Py<PyAny>,
        resume: CoroutineResume,
    },
    EnterAsync {
        context: Py<PyAny>,
        coro: Py<PyAny>,
        resume: CoroutineResume,
    },
    ExitAsync {
        cleanup: CleanupState,
        coro: Py<PyAny>,
        resume: CoroutineResume,
    },
}

pub(crate) struct AsyncMethodGenerator {
    pipeline: AwaitableMethodPipeline,
    state: GeneratorState,
    active: Option<Py<PyAny>>,
}

impl AsyncMethodGenerator {
    pub(crate) fn new(pipeline: AwaitableMethodPipeline) -> Self {
        Self {
            pipeline,
            state: GeneratorState::Initial,
            active: None,
        }
    }

    fn resume(&mut self, py: Python<'_>, resume: CoroutineResume) -> GeneratorResult {
        let mut action = match self.active.take() {
            Some(coro) => ResumeAction::Coroutine { coro, resume },
            None => ResumeAction::Step,
        };
        let mut input = None;

        loop {
            match action {
                ResumeAction::Step => {}
                ResumeAction::Coroutine { coro, resume } => {
                    match resume_coroutine(py, &coro, resume) {
                        CoroutineReply::Yielded(value) => {
                            self.active = Some(coro);
                            return GeneratorResult::Suspended(value);
                        }
                        CoroutineReply::Returned(value) => input = Some(value),
                        CoroutineReply::Raised(error) => {
                            self.state = GeneratorState::Done;
                            return GeneratorResult::Failed(error);
                        }
                    }
                }
            }

            action = match self.pipeline.next(py, input.take()) {
                Ok(AwaitableMethodStep::Await(coro)) => ResumeAction::Coroutine {
                    coro,
                    resume: CoroutineResume::Send(py.None()),
                },
                Ok(AwaitableMethodStep::Done(value)) => {
                    self.state = GeneratorState::Done;
                    return GeneratorResult::Completed(value);
                }
                Err(error) => {
                    self.state = GeneratorState::Done;
                    return GeneratorResult::Failed(error);
                }
            };
        }
    }
}

impl PyGeneratorLike for AsyncMethodGenerator {
    fn is_initial(&self) -> bool {
        matches!(self.state, GeneratorState::Initial)
    }

    fn send(&mut self, py: Python<'_>, value: Py<PyAny>) -> GeneratorResult {
        if matches!(self.state, GeneratorState::Initial) {
            self.state = GeneratorState::Executing;
        }
        self.resume(py, CoroutineResume::Send(value))
    }

    fn throw(
        &mut self,
        py: Python<'_>,
        typ: Py<PyAny>,
        val: Option<Py<PyAny>>,
        tb: Option<Py<PyAny>>,
    ) -> GeneratorResult {
        if matches!(self.state, GeneratorState::Initial) {
            self.state = GeneratorState::Done;
            return GeneratorResult::Failed(throw_error(py, typ, val));
        }
        self.resume(py, CoroutineResume::Throw { typ, val, tb })
    }

    fn close(&mut self, py: Python<'_>) -> GeneratorCloseResult {
        if let Some(coro) = self.active.take()
            && let Err(error) = coro.call_method0(py, "close")
        {
            return match classify_coroutine_close_error(py, error) {
                GeneratorCloseError::IgnoredGeneratorExit(error) => {
                    self.active = Some(coro);
                    GeneratorCloseResult::IgnoredGeneratorExit(error)
                }
                GeneratorCloseError::Failed(error) => {
                    self.state = GeneratorState::Done;
                    GeneratorCloseResult::Failed(error)
                }
            };
        }
        self.state = GeneratorState::Done;
        GeneratorCloseResult::Closed
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.pipeline.traverse(visit)?;
        if let Some(coro) = &self.active {
            visit.call(coro)?;
        }
        Ok(())
    }
}

pub(crate) struct AsyncContextEnterGenerator {
    pipeline: AsyncContextManagerEnterPipeline,
    exits: MixedExitStack,
    state: AsyncContextEnterState,
    wrapper_state: Arc<Mutex<AsyncContextManagerState>>,
}

impl AsyncContextEnterGenerator {
    pub(crate) fn new(
        pipeline: AsyncContextManagerEnterPipeline,
        wrapper_state: Arc<Mutex<AsyncContextManagerState>>,
    ) -> Self {
        Self {
            pipeline,
            exits: MixedExitStack::new(),
            state: AsyncContextEnterState::Initial,
            wrapper_state,
        }
    }

    fn resume(&mut self, py: Python<'_>, resume: CoroutineResume) -> GeneratorResult {
        let mut action = self.take_resume_action(py, resume);
        let mut input = None;
        let mut outcome = None;

        loop {
            action = match action {
                AsyncContextEnterAction::Complete(result) => return result,
                AsyncContextEnterAction::StepEnter => self.step_enter_pipeline(py, &mut input),
                AsyncContextEnterAction::StepCleanup(cleanup) => {
                    self.step_cleanup_drainer(py, cleanup, &mut outcome)
                }
                AsyncContextEnterAction::Await { coro, resume } => {
                    match resume_coroutine(py, &coro, resume) {
                        CoroutineReply::Yielded(value) => {
                            self.state = AsyncContextEnterState::SuspendedAwait(coro);
                            return GeneratorResult::Suspended(value);
                        }
                        CoroutineReply::Returned(value) => {
                            input = Some(value);
                            AsyncContextEnterAction::StepEnter
                        }
                        CoroutineReply::Raised(error) => self.start_cleanup(py, error),
                    }
                }
                AsyncContextEnterAction::EnterAsync {
                    context,
                    coro,
                    resume,
                } => match resume_coroutine(py, &coro, resume) {
                    CoroutineReply::Yielded(value) => {
                        self.state = AsyncContextEnterState::SuspendedEnterAsync { context, coro };
                        return GeneratorResult::Suspended(value);
                    }
                    CoroutineReply::Returned(value) => {
                        // Async contexts are registered only after __aenter__ completes;
                        // a raised __aenter__ must not get __aexit__.
                        self.exits.push_item(ExitItem::Async(context));
                        input = Some(value);
                        AsyncContextEnterAction::StepEnter
                    }
                    CoroutineReply::Raised(error) => self.start_cleanup(py, error),
                },
                AsyncContextEnterAction::ExitAsync {
                    cleanup,
                    coro,
                    resume,
                } => match resume_coroutine(py, &coro, resume) {
                    CoroutineReply::Yielded(value) => {
                        self.state = AsyncContextEnterState::SuspendedCleanup { cleanup, coro };
                        return GeneratorResult::Suspended(value);
                    }
                    CoroutineReply::Returned(value) => {
                        outcome = Some(ExitOutcome::Returned(value));
                        AsyncContextEnterAction::StepCleanup(cleanup)
                    }
                    CoroutineReply::Raised(error) => {
                        outcome = Some(ExitOutcome::Raised(error));
                        AsyncContextEnterAction::StepCleanup(cleanup)
                    }
                },
            };
        }
    }

    fn take_resume_action(
        &mut self,
        py: Python<'_>,
        resume: CoroutineResume,
    ) -> AsyncContextEnterAction {
        match std::mem::replace(&mut self.state, AsyncContextEnterState::Initial) {
            AsyncContextEnterState::Initial => AsyncContextEnterAction::StepEnter,
            AsyncContextEnterState::SuspendedAwait(coro) => {
                AsyncContextEnterAction::Await { coro, resume }
            }
            AsyncContextEnterState::SuspendedEnterAsync { context, coro } => {
                AsyncContextEnterAction::EnterAsync {
                    context,
                    coro,
                    resume,
                }
            }
            AsyncContextEnterState::SuspendedCleanup { cleanup, coro } => {
                AsyncContextEnterAction::ExitAsync {
                    cleanup,
                    coro,
                    resume,
                }
            }
            AsyncContextEnterState::Done => {
                self.state = AsyncContextEnterState::Done;
                AsyncContextEnterAction::Complete(GeneratorResult::Completed(py.None()))
            }
        }
    }

    fn step_enter_pipeline(
        &mut self,
        py: Python<'_>,
        input: &mut Option<Py<PyAny>>,
    ) -> AsyncContextEnterAction {
        match self.pipeline.next(py, input.take()) {
            Ok(AsyncContextManagerEnterStep::EnterSync(context)) => {
                match context.call_method0(py, "__enter__") {
                    Ok(entered) => {
                        self.exits.push_sync(context);
                        *input = Some(entered);
                        AsyncContextEnterAction::StepEnter
                    }
                    Err(error) => self.start_cleanup(py, error),
                }
            }
            Ok(AsyncContextManagerEnterStep::EnterAsync(context)) => {
                match context.call_method0(py, "__aenter__") {
                    Ok(coro) => AsyncContextEnterAction::EnterAsync {
                        context,
                        coro,
                        resume: CoroutineResume::Send(py.None()),
                    },
                    Err(error) => self.start_cleanup(py, error),
                }
            }
            Ok(AsyncContextManagerEnterStep::Await(coro)) => AsyncContextEnterAction::Await {
                coro,
                resume: CoroutineResume::Send(py.None()),
            },
            Ok(AsyncContextManagerEnterStep::Done(value)) => {
                self.state = AsyncContextEnterState::Done;
                let exits = std::mem::take(&mut self.exits);
                *self.wrapper_state.lock().expect("poisoned") =
                    AsyncContextManagerState::Entered(exits);
                AsyncContextEnterAction::Complete(GeneratorResult::Completed(value))
            }
            Err(error) => self.start_cleanup(py, error),
        }
    }

    fn step_cleanup_drainer(
        &mut self,
        py: Python<'_>,
        mut cleanup: CleanupState,
        outcome: &mut Option<ExitOutcome>,
    ) -> AsyncContextEnterAction {
        match cleanup.drainer.next(py, outcome.take()) {
            AsyncExitStep::ExitSync { context, exc } => {
                *outcome = Some(call_sync_exit(py, context, exc));
                AsyncContextEnterAction::StepCleanup(cleanup)
            }
            AsyncExitStep::ExitAsync { context, exc } => match call_async_exit(py, context, exc) {
                Ok(coro) => AsyncContextEnterAction::ExitAsync {
                    cleanup,
                    coro,
                    resume: CoroutineResume::Send(py.None()),
                },
                Err(exit_outcome) => {
                    *outcome = Some(exit_outcome);
                    AsyncContextEnterAction::StepCleanup(cleanup)
                }
            },
            AsyncExitStep::Done(ExitDrainCompletion::Raise(error)) => {
                self.finish_cleanup_failed(error)
            }
            AsyncExitStep::Done(ExitDrainCompletion::Return(false)) => {
                self.finish_cleanup_failed(cleanup.original_error)
            }
            AsyncExitStep::Done(ExitDrainCompletion::Return(true)) => {
                self.finish_cleanup_without_enter_value()
            }
        }
    }

    fn finish_cleanup_failed(&mut self, error: PyErr) -> AsyncContextEnterAction {
        self.state = AsyncContextEnterState::Done;
        *self.wrapper_state.lock().expect("poisoned") = AsyncContextManagerState::Exited;
        AsyncContextEnterAction::Complete(GeneratorResult::Failed(error))
    }

    fn finish_cleanup_without_enter_value(&mut self) -> AsyncContextEnterAction {
        self.state = AsyncContextEnterState::Done;
        *self.wrapper_state.lock().expect("poisoned") = AsyncContextManagerState::Exited;
        AsyncContextEnterAction::Complete(GeneratorResult::Failed(PyRuntimeError::new_err(
            "async context manager enter did not produce a value",
        )))
    }

    fn start_cleanup(&mut self, py: Python<'_>, error: PyErr) -> AsyncContextEnterAction {
        let active = ExceptionTriple::from_error(py, &error);
        let exits = std::mem::take(&mut self.exits);
        let drainer = AsyncExitDrainerPipeline::new(py, exits, active);
        AsyncContextEnterAction::StepCleanup(CleanupState {
            drainer,
            original_error: error,
        })
    }

    fn close_enter_without_active(&mut self, py: Python<'_>) -> GeneratorCloseResult {
        if self.exits.is_empty() {
            return self.finish_close_closed();
        }

        // close() cannot expose resumable async cleanup. Resume the drainer only
        // while every async __aexit__ completes without yielding; yielding means
        // the coroutine ignored GeneratorExit, matching CPython close semantics.
        let generator_exit = generator_exit();
        let active = ExceptionTriple::from_error(py, &generator_exit);
        let drainer = AsyncExitDrainerPipeline::new(py, std::mem::take(&mut self.exits), active);
        self.close_cleanup(
            py,
            CleanupState {
                drainer,
                original_error: generator_exit,
            },
            None,
        )
    }

    fn close_suspended_await(&mut self, py: Python<'_>, coro: Py<PyAny>) -> GeneratorCloseResult {
        match coro.call_method0(py, "close") {
            Ok(_) => self.close_enter_without_active(py),
            Err(error) => match classify_coroutine_close_error(py, error) {
                GeneratorCloseError::IgnoredGeneratorExit(error) => {
                    self.state = AsyncContextEnterState::SuspendedAwait(coro);
                    GeneratorCloseResult::IgnoredGeneratorExit(error)
                }
                GeneratorCloseError::Failed(error) => self.finish_close_failed(error),
            },
        }
    }

    fn close_suspended_enter_async(
        &mut self,
        py: Python<'_>,
        context: Py<PyAny>,
        coro: Py<PyAny>,
    ) -> GeneratorCloseResult {
        match coro.call_method0(py, "close") {
            Ok(_) => self.close_enter_without_active(py),
            Err(error) => match classify_coroutine_close_error(py, error) {
                GeneratorCloseError::IgnoredGeneratorExit(error) => {
                    self.state = AsyncContextEnterState::SuspendedEnterAsync { context, coro };
                    GeneratorCloseResult::IgnoredGeneratorExit(error)
                }
                GeneratorCloseError::Failed(error) => self.finish_close_failed(error),
            },
        }
    }

    fn close_suspended_cleanup(
        &mut self,
        py: Python<'_>,
        mut cleanup: CleanupState,
        coro: Py<PyAny>,
    ) -> GeneratorCloseResult {
        let result = close_active_async_exit_and_resume_drainer(py, &mut cleanup.drainer, coro);
        self.close_cleanup_result(cleanup, result)
    }

    fn close_cleanup(
        &mut self,
        py: Python<'_>,
        mut cleanup: CleanupState,
        outcome: Option<ExitOutcome>,
    ) -> GeneratorCloseResult {
        let result = resume_async_drainer_without_suspension(py, &mut cleanup.drainer, outcome);
        self.close_cleanup_result(cleanup, result)
    }

    fn close_cleanup_result(
        &mut self,
        cleanup: CleanupState,
        result: CloseDrainOutcome,
    ) -> GeneratorCloseResult {
        match result {
            CloseDrainOutcome::Completed => self.finish_close_closed(),
            CloseDrainOutcome::Suspended { active, error } => {
                self.state = AsyncContextEnterState::SuspendedCleanup {
                    cleanup,
                    coro: active,
                };
                GeneratorCloseResult::IgnoredGeneratorExit(error)
            }
            CloseDrainOutcome::Raised(error) => self.finish_close_failed(error),
        }
    }

    fn finish_close_failed(&mut self, error: PyErr) -> GeneratorCloseResult {
        self.state = AsyncContextEnterState::Done;
        *self.wrapper_state.lock().expect("poisoned") = AsyncContextManagerState::Exited;
        GeneratorCloseResult::Failed(error)
    }

    fn finish_close_closed(&mut self) -> GeneratorCloseResult {
        self.state = AsyncContextEnterState::Done;
        *self.wrapper_state.lock().expect("poisoned") = AsyncContextManagerState::Exited;
        GeneratorCloseResult::Closed
    }
}

impl PyGeneratorLike for AsyncContextEnterGenerator {
    fn is_initial(&self) -> bool {
        matches!(&self.state, AsyncContextEnterState::Initial)
    }

    fn send(&mut self, py: Python<'_>, value: Py<PyAny>) -> GeneratorResult {
        self.resume(py, CoroutineResume::Send(value))
    }

    fn throw(
        &mut self,
        py: Python<'_>,
        typ: Py<PyAny>,
        val: Option<Py<PyAny>>,
        tb: Option<Py<PyAny>>,
    ) -> GeneratorResult {
        if matches!(&self.state, AsyncContextEnterState::Initial) {
            self.state = AsyncContextEnterState::Done;
            *self.wrapper_state.lock().expect("poisoned") = AsyncContextManagerState::Exited;
            return GeneratorResult::Failed(throw_error(py, typ, val));
        }
        self.resume(py, CoroutineResume::Throw { typ, val, tb })
    }

    fn close(&mut self, py: Python<'_>) -> GeneratorCloseResult {
        match std::mem::replace(&mut self.state, AsyncContextEnterState::Done) {
            AsyncContextEnterState::Done => GeneratorCloseResult::Closed,
            AsyncContextEnterState::Initial => self.close_enter_without_active(py),
            AsyncContextEnterState::SuspendedAwait(coro) => self.close_suspended_await(py, coro),
            AsyncContextEnterState::SuspendedEnterAsync { context, coro } => {
                self.close_suspended_enter_async(py, context, coro)
            }
            AsyncContextEnterState::SuspendedCleanup { cleanup, coro } => {
                self.close_suspended_cleanup(py, cleanup, coro)
            }
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.pipeline.traverse(visit)?;
        self.exits.traverse(visit)?;
        match &self.state {
            AsyncContextEnterState::SuspendedAwait(coro) => {
                visit.call(coro)?;
            }
            AsyncContextEnterState::SuspendedEnterAsync { context, coro } => {
                visit.call(context)?;
                visit.call(coro)?;
            }
            AsyncContextEnterState::SuspendedCleanup { cleanup, coro } => {
                cleanup.drainer.traverse(visit)?;
                visit.call(coro)?;
            }
            AsyncContextEnterState::Initial | AsyncContextEnterState::Done => {}
        }
        Ok(())
    }
}

pub(crate) struct AsyncExitGenerator {
    pipeline: AsyncExitDrainerPipeline,
    state: GeneratorState,
    active: Option<Py<PyAny>>,
}

impl AsyncExitGenerator {
    pub(crate) fn new(pipeline: AsyncExitDrainerPipeline) -> Self {
        Self {
            pipeline,
            state: GeneratorState::Initial,
            active: None,
        }
    }

    fn resume(&mut self, py: Python<'_>, resume: CoroutineResume) -> GeneratorResult {
        let mut action = match self.active.take() {
            Some(coro) => ResumeAction::Coroutine { coro, resume },
            None => ResumeAction::Step,
        };
        let mut outcome = None;

        loop {
            match action {
                ResumeAction::Step => {}
                ResumeAction::Coroutine { coro, resume } => {
                    match resume_coroutine(py, &coro, resume) {
                        CoroutineReply::Yielded(value) => {
                            self.active = Some(coro);
                            return GeneratorResult::Suspended(value);
                        }
                        CoroutineReply::Returned(value) => {
                            outcome = Some(ExitOutcome::Returned(value))
                        }
                        CoroutineReply::Raised(error) => outcome = Some(ExitOutcome::Raised(error)),
                    }
                }
            }

            action = match self.pipeline.next(py, outcome.take()) {
                AsyncExitStep::ExitSync { context, exc } => {
                    outcome = Some(call_sync_exit(py, context, exc));
                    ResumeAction::Step
                }
                AsyncExitStep::ExitAsync { context, exc } => {
                    match call_async_exit(py, context, exc) {
                        Ok(coro) => ResumeAction::Coroutine {
                            coro,
                            resume: CoroutineResume::Send(py.None()),
                        },
                        Err(exit_outcome) => {
                            outcome = Some(exit_outcome);
                            ResumeAction::Step
                        }
                    }
                }
                AsyncExitStep::Done(ExitDrainCompletion::Raise(error)) => {
                    self.state = GeneratorState::Done;
                    return GeneratorResult::Failed(error);
                }
                AsyncExitStep::Done(ExitDrainCompletion::Return(suppressed)) => {
                    self.state = GeneratorState::Done;
                    let value = PyBool::new(py, suppressed).to_owned().into_any().unbind();
                    return GeneratorResult::Completed(value);
                }
            };
        }
    }
}

impl PyGeneratorLike for AsyncExitGenerator {
    fn is_initial(&self) -> bool {
        matches!(self.state, GeneratorState::Initial)
    }

    fn send(&mut self, py: Python<'_>, value: Py<PyAny>) -> GeneratorResult {
        if matches!(self.state, GeneratorState::Initial) {
            self.state = GeneratorState::Executing;
        }
        self.resume(py, CoroutineResume::Send(value))
    }

    fn throw(
        &mut self,
        py: Python<'_>,
        typ: Py<PyAny>,
        val: Option<Py<PyAny>>,
        tb: Option<Py<PyAny>>,
    ) -> GeneratorResult {
        if matches!(self.state, GeneratorState::Initial) {
            self.state = GeneratorState::Done;
            return GeneratorResult::Failed(throw_error(py, typ, val));
        }
        self.resume(py, CoroutineResume::Throw { typ, val, tb })
    }

    fn close(&mut self, py: Python<'_>) -> GeneratorCloseResult {
        if let Some(coro) = self.active.take() {
            return match close_active_async_exit_and_resume_drainer(py, &mut self.pipeline, coro) {
                CloseDrainOutcome::Completed => {
                    self.state = GeneratorState::Done;
                    GeneratorCloseResult::Closed
                }
                CloseDrainOutcome::Suspended { active, error } => {
                    self.active = Some(active);
                    GeneratorCloseResult::IgnoredGeneratorExit(error)
                }
                CloseDrainOutcome::Raised(error) => {
                    self.state = GeneratorState::Done;
                    GeneratorCloseResult::Failed(error)
                }
            };
        }
        self.state = GeneratorState::Done;
        GeneratorCloseResult::Closed
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.pipeline.traverse(visit)?;
        if let Some(coro) = &self.active {
            visit.call(coro)?;
        }
        Ok(())
    }
}

enum CloseDrainOutcome {
    Completed,
    Suspended { active: Py<PyAny>, error: PyErr },
    Raised(PyErr),
}

fn close_active_async_exit_and_resume_drainer(
    py: Python<'_>,
    drainer: &mut AsyncExitDrainerPipeline,
    active: Py<PyAny>,
) -> CloseDrainOutcome {
    let outcome = match active.call_method0(py, "close") {
        Ok(_) => ExitOutcome::Raised(generator_exit()),
        Err(error) => match classify_coroutine_close_error(py, error) {
            GeneratorCloseError::IgnoredGeneratorExit(error) => {
                return CloseDrainOutcome::Suspended { active, error };
            }
            GeneratorCloseError::Failed(error) => ExitOutcome::Raised(error),
        },
    };

    resume_async_drainer_without_suspension(py, drainer, Some(outcome))
}

fn resume_async_drainer_without_suspension(
    py: Python<'_>,
    drainer: &mut AsyncExitDrainerPipeline,
    mut outcome: Option<ExitOutcome>,
) -> CloseDrainOutcome {
    loop {
        match drainer.next(py, outcome.take()) {
            AsyncExitStep::ExitSync { context, exc } => {
                outcome = Some(call_sync_exit(py, context, exc));
            }
            AsyncExitStep::ExitAsync { context, exc } => match call_async_exit(py, context, exc) {
                Ok(coro) => match resume_coroutine(py, &coro, CoroutineResume::Send(py.None())) {
                    CoroutineReply::Yielded(_) => {
                        return CloseDrainOutcome::Suspended {
                            active: coro,
                            error: PyRuntimeError::new_err("coroutine ignored GeneratorExit"),
                        };
                    }
                    CoroutineReply::Returned(value) => {
                        outcome = Some(ExitOutcome::Returned(value));
                    }
                    CoroutineReply::Raised(error) => outcome = Some(ExitOutcome::Raised(error)),
                },
                Err(exit_outcome) => outcome = Some(exit_outcome),
            },
            AsyncExitStep::Done(ExitDrainCompletion::Raise(error)) => {
                if error.is_instance_of::<PyGeneratorExit>(py) {
                    return CloseDrainOutcome::Completed;
                }
                return CloseDrainOutcome::Raised(error);
            }
            AsyncExitStep::Done(ExitDrainCompletion::Return(_)) => {
                return CloseDrainOutcome::Completed;
            }
        }
    }
}
