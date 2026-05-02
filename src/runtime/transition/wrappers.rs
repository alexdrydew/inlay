use std::sync::{Arc, Mutex};

use pyo3::PyTraverseError;
use pyo3::exceptions::{PyBaseException, PyRuntimeError, PyStopIteration};
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::runtime::executor::{exit_sync_contexts, start_async_context_implementation};
use crate::types::WrapperKind;

use super::{
    ChildContext, ChildExecutionParams, TransitionShared, execute_child_context_with_sync_contexts,
    execute_child_from_params, prepare_child_execution, stop_iteration_value,
};

enum ContextManagerState {
    NotEntered,
    Running,
    Entered(Vec<Py<PyAny>>),
    Exited,
}

enum AsyncContextManagerState {
    NotEntered,
    Entering,
    Entered(Option<Py<PyAny>>),
    Exited,
}

#[pyclass(frozen, module = "inlay")]
pub(crate) struct ContextManagerWrapper {
    shared: TransitionShared,
    args: Py<PyTuple>,
    kwargs: Option<Py<PyDict>>,
    state: Mutex<ContextManagerState>,
}

impl ContextManagerWrapper {
    pub(super) fn new(
        shared: TransitionShared,
        args: Py<PyTuple>,
        kwargs: Option<Py<PyDict>>,
    ) -> Self {
        Self {
            shared,
            args,
            kwargs,
            state: Mutex::new(ContextManagerState::NotEntered),
        }
    }
}

struct CleanupError {
    error: PyErr,
    exc_type: Py<PyAny>,
    exc_val: Py<PyBaseException>,
    exc_tb: Py<PyAny>,
}

impl CleanupError {
    fn new(py: Python<'_>, error: PyErr) -> Self {
        let exc_type = error.get_type(py).into_any().unbind();
        let exc_val = error.value(py).clone().unbind();
        let exc_tb = error
            .traceback(py)
            .map(|tb| tb.into_any().unbind())
            .unwrap_or_else(|| py.None());
        Self {
            error,
            exc_type,
            exc_val,
            exc_tb,
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.exc_type)?;
        visit.call(&self.exc_val)?;
        visit.call(&self.exc_tb)?;
        Ok(())
    }
}

#[pymethods]
impl ContextManagerWrapper {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.shared.traverse(&visit)?;
        visit.call(&self.args)?;
        if let Some(kwargs) = &self.kwargs {
            visit.call(kwargs)?;
        }
        if let Ok(state) = self.state.try_lock()
            && let ContextManagerState::Entered(contexts) = &*state
        {
            for context in contexts {
                visit.call(context)?;
            }
        }
        Ok(())
    }

    fn __enter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        {
            let mut state = self.state.lock().expect("poisoned");
            match std::mem::replace(&mut *state, ContextManagerState::Running) {
                ContextManagerState::NotEntered => {}
                other => {
                    *state = other;
                    return Err(PyRuntimeError::new_err(
                        "context manager has already been entered",
                    ));
                }
            }
        }

        match execute_child_context_with_sync_contexts(ChildContext {
            py,
            shared: &self.shared,
            resources: self.shared.resources.clone_ref(py),
            args: self.args.bind(py),
            kwargs: self.kwargs.as_ref().map(|k| k.bind(py)),
        }) {
            Ok((result, contexts)) => {
                *self.state.lock().expect("poisoned") = ContextManagerState::Entered(contexts);
                Ok(result)
            }
            Err(error) => {
                *self.state.lock().expect("poisoned") = ContextManagerState::Exited;
                Err(error)
            }
        }
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        exc_type: Py<PyAny>,
        exc_val: Py<PyAny>,
        exc_tb: Py<PyAny>,
    ) -> PyResult<bool> {
        let contexts = {
            let mut state = self.state.lock().expect("poisoned");
            match std::mem::replace(&mut *state, ContextManagerState::Exited) {
                ContextManagerState::Entered(contexts) => contexts,
                other => {
                    *state = other;
                    return Err(PyRuntimeError::new_err(
                        "context manager has not been entered",
                    ));
                }
            }
        };
        exit_sync_contexts(py, contexts, &exc_type, &exc_val, &exc_tb)
    }
}

enum AwaitableState {
    Driving(Py<PyAny>),
    CleaningUp {
        coro: Py<PyAny>,
        error: CleanupError,
    },
    Immediate,
    Running,
    Done,
}

#[pyclass(frozen, module = "inlay")]
pub(crate) struct AwaitableWrapper {
    state: Mutex<AwaitableState>,
    child_execution: Option<ChildExecutionParams>,
    cleanup_context: Option<Py<PyAny>>,
    enter_state: Option<Arc<Mutex<AsyncContextManagerState>>>,
    enter_context: Option<Py<PyAny>>,
}

impl AwaitableWrapper {
    pub(super) fn new(
        inner_coro: Option<Py<PyAny>>,
        child_execution: Option<ChildExecutionParams>,
    ) -> Self {
        Self::new_with_cleanup(inner_coro, child_execution, None)
    }

    pub(super) fn new_with_cleanup(
        inner_coro: Option<Py<PyAny>>,
        child_execution: Option<ChildExecutionParams>,
        cleanup_context: Option<Py<PyAny>>,
    ) -> Self {
        Self::new_full(inner_coro, child_execution, cleanup_context, None, None)
    }

    fn new_with_enter_state(
        inner_coro: Option<Py<PyAny>>,
        child_execution: Option<ChildExecutionParams>,
        cleanup_context: Option<Py<PyAny>>,
        enter_state: Arc<Mutex<AsyncContextManagerState>>,
        enter_context: Option<Py<PyAny>>,
    ) -> Self {
        Self::new_full(
            inner_coro,
            child_execution,
            cleanup_context,
            Some(enter_state),
            enter_context,
        )
    }

    fn new_full(
        inner_coro: Option<Py<PyAny>>,
        child_execution: Option<ChildExecutionParams>,
        cleanup_context: Option<Py<PyAny>>,
        enter_state: Option<Arc<Mutex<AsyncContextManagerState>>>,
        enter_context: Option<Py<PyAny>>,
    ) -> Self {
        let state = match inner_coro {
            Some(coro) => AwaitableState::Driving(coro),
            None => AwaitableState::Immediate,
        };
        Self {
            state: Mutex::new(state),
            child_execution,
            cleanup_context,
            enter_state,
            enter_context,
        }
    }

    fn on_complete(&self, py: Python<'_>, coro_result: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        match &self.child_execution {
            Some(params) => match execute_child_from_params(py, params, coro_result) {
                Ok(child_result) => {
                    self.mark_entered(py)?;
                    Err(PyStopIteration::new_err(child_result))
                }
                Err(error) => self.start_cleanup(py, error),
            },
            None => Err(PyStopIteration::new_err(
                coro_result.unwrap_or_else(|| py.None()),
            )),
        }
    }

    fn mark_entered(&self, py: Python<'_>) -> PyResult<()> {
        let Some(enter_state) = &self.enter_state else {
            return Ok(());
        };
        let context = self
            .enter_context
            .as_ref()
            .map(|context| context.clone_ref(py));
        *enter_state.lock().expect("poisoned") = AsyncContextManagerState::Entered(context);
        Ok(())
    }

    fn start_cleanup(&self, py: Python<'_>, error: PyErr) -> PyResult<Py<PyAny>> {
        let Some(cleanup_context) = &self.cleanup_context else {
            return Err(error);
        };
        let cleanup_error = CleanupError::new(py, error);
        let cleanup_coro = cleanup_context.call_method1(
            py,
            "__aexit__",
            (
                &cleanup_error.exc_type,
                &cleanup_error.exc_val,
                &cleanup_error.exc_tb,
            ),
        )?;
        self.drive_cleanup(py, cleanup_coro, cleanup_error, py.None())
    }

    fn drive_cleanup(
        &self,
        py: Python<'_>,
        coro: Py<PyAny>,
        error: CleanupError,
        value: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        match coro.call_method1(py, "send", (&value,)) {
            Ok(yielded) => {
                self.replace_state(AwaitableState::CleaningUp { coro, error });
                Ok(yielded)
            }
            Err(cleanup_error) => {
                self.replace_state(AwaitableState::Done);
                if cleanup_error.is_instance_of::<PyStopIteration>(py) {
                    Err(error.error)
                } else {
                    Err(cleanup_error)
                }
            }
        }
    }

    fn take_state(&self) -> AwaitableState {
        std::mem::replace(
            &mut *self.state.lock().expect("poisoned"),
            AwaitableState::Running,
        )
    }

    fn replace_state(&self, state: AwaitableState) {
        *self.state.lock().expect("poisoned") = state;
    }
}

#[pymethods]
impl AwaitableWrapper {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Ok(state) = self.state.try_lock() {
            match &*state {
                AwaitableState::Driving(coro) => visit.call(coro)?,
                AwaitableState::CleaningUp { coro, error } => {
                    visit.call(coro)?;
                    error.traverse(&visit)?;
                }
                AwaitableState::Immediate | AwaitableState::Running | AwaitableState::Done => {}
            }
        }
        if let Some(params) = &self.child_execution {
            params.traverse(&visit)?;
        }
        if let Some(cleanup_context) = &self.cleanup_context {
            visit.call(cleanup_context)?;
        }
        if let Some(enter_context) = &self.enter_context {
            visit.call(enter_context)?;
        }
        Ok(())
    }

    fn __await__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.send(py, py.None())
    }

    fn send(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<Py<PyAny>> {
        match self.take_state() {
            AwaitableState::Driving(coro) => match coro.call_method1(py, "send", (&value,)) {
                Ok(yielded) => {
                    self.replace_state(AwaitableState::Driving(coro));
                    Ok(yielded)
                }
                Err(e) => {
                    if e.is_instance_of::<PyStopIteration>(py) {
                        let result = stop_iteration_value(py, &e);
                        self.replace_state(AwaitableState::Done);
                        self.on_complete(py, Some(result))
                    } else {
                        self.replace_state(AwaitableState::Done);
                        Err(e)
                    }
                }
            },
            AwaitableState::Immediate => {
                self.replace_state(AwaitableState::Done);
                self.on_complete(py, None)
            }
            AwaitableState::CleaningUp { coro, error } => {
                self.drive_cleanup(py, coro, error, value)
            }
            AwaitableState::Running => Err(PyRuntimeError::new_err("awaitable is already running")),
            AwaitableState::Done => Err(PyStopIteration::new_err(py.None())),
        }
    }

    #[pyo3(signature = (typ, val=None, tb=None))]
    fn throw(
        &self,
        py: Python<'_>,
        typ: Py<PyAny>,
        val: Option<Py<PyAny>>,
        tb: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        match self.take_state() {
            AwaitableState::Driving(coro) => {
                let none = py.None();
                let val_ref = val.as_ref().unwrap_or(&none);
                let tb_ref = tb.as_ref().unwrap_or(&none);
                match coro.call_method1(py, "throw", (&typ, val_ref, tb_ref)) {
                    Ok(yielded) => {
                        self.replace_state(AwaitableState::Driving(coro));
                        Ok(yielded)
                    }
                    Err(e) => {
                        self.replace_state(AwaitableState::Done);
                        Err(e)
                    }
                }
            }
            AwaitableState::CleaningUp { coro, error } => {
                let none = py.None();
                let val_ref = val.as_ref().unwrap_or(&none);
                let tb_ref = tb.as_ref().unwrap_or(&none);
                match coro.call_method1(py, "throw", (&typ, val_ref, tb_ref)) {
                    Ok(yielded) => {
                        self.replace_state(AwaitableState::CleaningUp { coro, error });
                        Ok(yielded)
                    }
                    Err(e) => {
                        self.replace_state(AwaitableState::Done);
                        if e.is_instance_of::<PyStopIteration>(py) {
                            Err(error.error)
                        } else {
                            Err(e)
                        }
                    }
                }
            }
            AwaitableState::Running => Err(PyRuntimeError::new_err("awaitable is already running")),
            AwaitableState::Immediate | AwaitableState::Done => {
                self.replace_state(AwaitableState::Done);
                Err(PyErr::from_value(typ.into_bound(py)))
            }
        }
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        match self.take_state() {
            AwaitableState::Driving(coro) => {
                let _ = coro.call_method0(py, "close");
                self.replace_state(AwaitableState::Done);
                Ok(())
            }
            AwaitableState::CleaningUp { coro, .. } => {
                let _ = coro.call_method0(py, "close");
                self.replace_state(AwaitableState::Done);
                Ok(())
            }
            AwaitableState::Running => Err(PyRuntimeError::new_err("awaitable is already running")),
            AwaitableState::Immediate | AwaitableState::Done => {
                self.replace_state(AwaitableState::Done);
                Ok(())
            }
        }
    }
}

#[pyclass(frozen, module = "inlay")]
pub(crate) struct AsyncContextManagerWrapper {
    shared: TransitionShared,
    args: Py<PyTuple>,
    kwargs: Option<Py<PyDict>>,
    state: Arc<Mutex<AsyncContextManagerState>>,
}

impl AsyncContextManagerWrapper {
    pub(super) fn new(
        shared: TransitionShared,
        args: Py<PyTuple>,
        kwargs: Option<Py<PyDict>>,
    ) -> Self {
        Self {
            shared,
            args,
            kwargs,
            state: Arc::new(Mutex::new(AsyncContextManagerState::NotEntered)),
        }
    }
}

#[pymethods]
impl AsyncContextManagerWrapper {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.shared.traverse(&visit)?;
        visit.call(&self.args)?;
        if let Some(kwargs) = &self.kwargs {
            visit.call(kwargs)?;
        }
        if let Ok(state) = self.state.try_lock()
            && let AsyncContextManagerState::Entered(Some(context)) = &*state
        {
            visit.call(context)?;
        }
        Ok(())
    }

    fn __aenter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        {
            let mut state = self.state.lock().expect("poisoned");
            match std::mem::replace(&mut *state, AsyncContextManagerState::Entering) {
                AsyncContextManagerState::NotEntered => {}
                other => {
                    *state = other;
                    return Err(PyRuntimeError::new_err(
                        "async context manager has already been entered",
                    ));
                }
            }
        }

        if matches!(
            self.shared
                .implementations
                .first()
                .map(|implementation| implementation.return_wrapper),
            Some(WrapperKind::AsyncContextManager)
        ) {
            let context = ChildContext {
                py,
                shared: &self.shared,
                resources: self.shared.resources.clone_ref(py),
                args: self.args.bind(py),
                kwargs: self.kwargs.as_ref().map(|k| k.bind(py)),
            };
            let (child_data, child_resources) = match prepare_child_execution(&context) {
                Ok(prepared) => prepared,
                Err(error) => {
                    *self.state.lock().expect("poisoned") = AsyncContextManagerState::Exited;
                    return Err(error);
                }
            };
            let first = self
                .shared
                .implementations
                .first()
                .expect("checked first implementation");
            let started =
                match start_async_context_implementation(py, &child_data, child_resources, first) {
                    Ok(started) => started,
                    Err(error) => {
                        *self.state.lock().expect("poisoned") = AsyncContextManagerState::Exited;
                        return Err(error);
                    }
                };
            let child_exec = ChildExecutionParams::new_prepared(
                &self.shared,
                py,
                self.args.clone_ref(py),
                self.kwargs.as_ref().map(|k| k.clone_ref(py)),
                started.resources,
                1,
                started.result_source,
            );
            let wrapper = AwaitableWrapper::new_with_enter_state(
                Some(started.enter_coro),
                Some(child_exec),
                Some(started.context.clone_ref(py)),
                Arc::clone(&self.state),
                Some(started.context),
            );
            return Ok(Py::new(py, wrapper)?.into_any());
        }

        let child_exec = ChildExecutionParams::new(
            &self.shared,
            py,
            self.args.clone_ref(py),
            self.kwargs.as_ref().map(|k| k.clone_ref(py)),
        );

        let wrapper = AwaitableWrapper::new_with_enter_state(
            None,
            Some(child_exec),
            None,
            Arc::clone(&self.state),
            None,
        );
        Ok(Py::new(py, wrapper)?.into_any())
    }

    fn __aexit__(
        &self,
        py: Python<'_>,
        exc_type: Py<PyAny>,
        exc_val: Py<PyAny>,
        exc_tb: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let context = {
            let mut state = self.state.lock().expect("poisoned");
            match std::mem::replace(&mut *state, AsyncContextManagerState::Exited) {
                AsyncContextManagerState::Entered(context) => context,
                other => {
                    *state = other;
                    return Err(PyRuntimeError::new_err(
                        "async context manager has not been entered",
                    ));
                }
            }
        };
        let inner_coro = match context {
            Some(context) => {
                Some(context.call_method1(py, "__aexit__", (&exc_type, &exc_val, &exc_tb))?)
            }
            None => None,
        };
        let wrapper = AwaitableWrapper::new(inner_coro, None);
        Ok(Py::new(py, wrapper)?.into_any())
    }
}
