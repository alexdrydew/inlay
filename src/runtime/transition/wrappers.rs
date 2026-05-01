use std::sync::{Mutex, OnceLock};

use pyo3::PyTraverseError;
use pyo3::exceptions::{PyBaseException, PyRuntimeError, PyStopIteration};
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::{
    ChildContext, ChildExecutionParams, TransitionKind, TransitionShared, call_implementation,
    execute_child_context, execute_child_from_params, stop_iteration_value,
};

#[pyclass(frozen, module = "inlay")]
pub(crate) struct ContextManagerWrapper {
    shared: TransitionShared,
    kind: TransitionKind,
    args: Py<PyTuple>,
    kwargs: Option<Py<PyDict>>,
    context_manager: OnceLock<Py<PyAny>>,
}

impl ContextManagerWrapper {
    pub(super) fn new(
        shared: TransitionShared,
        kind: TransitionKind,
        args: Py<PyTuple>,
        kwargs: Option<Py<PyDict>>,
    ) -> Self {
        Self {
            shared,
            kind,
            args,
            kwargs,
            context_manager: OnceLock::new(),
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
        self.kind.traverse(&visit)?;
        visit.call(&self.args)?;
        if let Some(kwargs) = &self.kwargs {
            visit.call(kwargs)?;
        }
        if let Some(context_manager) = self.context_manager.get() {
            visit.call(context_manager)?;
        }
        Ok(())
    }

    fn __enter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let method_result = match &self.kind {
            TransitionKind::Method {
                implementation,
                bound_instance,
                ..
            } => {
                let context_manager = call_implementation(
                    py,
                    implementation,
                    bound_instance,
                    self.args.bind(py),
                    self.kwargs.as_ref().map(|k| k.bind(py)),
                )?;
                let result = context_manager.call_method0(py, "__enter__")?;
                self.context_manager
                    .set(context_manager)
                    .map_err(|_| PyRuntimeError::new_err("__enter__ called twice"))?;
                Some(result)
            }
            TransitionKind::Auto => None,
        };
        match execute_child_context(ChildContext {
            py,
            shared: &self.shared,
            resources: self.shared.resources.clone_ref(py),
            args: self.args.bind(py),
            kwargs: self.kwargs.as_ref().map(|k| k.bind(py)),
            kind: &self.kind,
            method_result,
        }) {
            Ok(result) => Ok(result),
            Err(error) => {
                if let Some(context_manager) = self.context_manager.get() {
                    let cleanup_error = CleanupError::new(py, error);
                    context_manager.call_method1(
                        py,
                        "__exit__",
                        (
                            &cleanup_error.exc_type,
                            &cleanup_error.exc_val,
                            &cleanup_error.exc_tb,
                        ),
                    )?;
                    Err(cleanup_error.error)
                } else {
                    Err(error)
                }
            }
        }
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        exc_type: Py<PyAny>,
        exc_val: Py<PyAny>,
        exc_tb: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        match self.context_manager.get() {
            Some(context_manager) => {
                context_manager.call_method1(py, "__exit__", (exc_type, exc_val, exc_tb))
            }
            None => Ok(py.None()),
        }
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
        let state = match inner_coro {
            Some(coro) => AwaitableState::Driving(coro),
            None => AwaitableState::Immediate,
        };
        Self {
            state: Mutex::new(state),
            child_execution,
            cleanup_context,
        }
    }

    fn on_complete(&self, py: Python<'_>, coro_result: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        match &self.child_execution {
            Some(params) => match execute_child_from_params(py, params, coro_result) {
                Ok(child_result) => Err(PyStopIteration::new_err(child_result)),
                Err(error) => self.start_cleanup(py, error),
            },
            None => Err(PyStopIteration::new_err(
                coro_result.unwrap_or_else(|| py.None()),
            )),
        }
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
    kind: TransitionKind,
    args: Py<PyTuple>,
    kwargs: Option<Py<PyDict>>,
    async_context_manager: OnceLock<Py<PyAny>>,
}

impl AsyncContextManagerWrapper {
    pub(super) fn new(
        shared: TransitionShared,
        kind: TransitionKind,
        args: Py<PyTuple>,
        kwargs: Option<Py<PyDict>>,
    ) -> Self {
        Self {
            shared,
            kind,
            args,
            kwargs,
            async_context_manager: OnceLock::new(),
        }
    }
}

#[pymethods]
impl AsyncContextManagerWrapper {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.shared.traverse(&visit)?;
        self.kind.traverse(&visit)?;
        visit.call(&self.args)?;
        if let Some(kwargs) = &self.kwargs {
            visit.call(kwargs)?;
        }
        if let Some(async_context_manager) = self.async_context_manager.get() {
            visit.call(async_context_manager)?;
        }
        Ok(())
    }

    fn __aenter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let (inner_coro, cleanup_context) = match &self.kind {
            TransitionKind::Method {
                implementation,
                bound_instance,
                ..
            } => {
                let async_context_manager = call_implementation(
                    py,
                    implementation,
                    bound_instance,
                    self.args.bind(py),
                    self.kwargs.as_ref().map(|k| k.bind(py)),
                )?;
                let enter_coro = async_context_manager.call_method0(py, "__aenter__")?;
                let cleanup_context = async_context_manager.clone_ref(py);
                self.async_context_manager
                    .set(async_context_manager)
                    .map_err(|_| PyRuntimeError::new_err("__aenter__ called twice"))?;
                (Some(enter_coro), Some(cleanup_context))
            }
            TransitionKind::Auto => (None, None),
        };

        let child_exec = ChildExecutionParams::new(
            &self.shared,
            &self.kind,
            py,
            self.args.clone_ref(py),
            self.kwargs.as_ref().map(|k| k.clone_ref(py)),
        );

        let wrapper =
            AwaitableWrapper::new_with_cleanup(inner_coro, Some(child_exec), cleanup_context);
        Ok(Py::new(py, wrapper)?.into_any())
    }

    fn __aexit__(
        &self,
        py: Python<'_>,
        exc_type: Py<PyAny>,
        exc_val: Py<PyAny>,
        exc_tb: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let inner_coro = match self.async_context_manager.get() {
            Some(async_context_manager) => Some(async_context_manager.call_method1(
                py,
                "__aexit__",
                (&exc_type, &exc_val, &exc_tb),
            )?),
            None => None,
        };
        let wrapper = AwaitableWrapper::new(inner_coro, None);
        Ok(Py::new(py, wrapper)?.into_any())
    }
}
