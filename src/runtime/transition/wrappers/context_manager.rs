use std::sync::{Arc, Mutex};

use pyo3::PyTraverseError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::runtime::transition::pipeline::exits::{ExceptionTriple, SyncExitStack};
use crate::runtime::transition::pipeline::generator::{
    AsyncContextEnterGenerator, AsyncContextManagerState, AsyncExitGenerator,
    SyncContextEnterRunner, SyncContextExitRunner,
};
use crate::runtime::transition::pipeline::pipelines::{
    AsyncContextManagerEnterPipeline, AsyncExitDrainerPipeline, ContextManagerEnterPipeline,
    PipelineCommon, SyncExitDrainerPipeline,
};
use crate::runtime::transition::wrappers::awaitable::AwaitableWrapper;
use crate::runtime::transition::{ChildContext, TransitionShared, prepare_child_execution};

enum ContextManagerState {
    NotEntered,
    Running,
    Entered(SyncExitStack),
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
    pub(crate) fn new(
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

    fn run_enter(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, SyncExitStack)> {
        let (child_data, child_resources) = prepare_child_execution(ChildContext {
            py,
            shared: &self.shared,
            resources: self.shared.resources.clone_ref(py),
            args: self.args.bind(py),
            kwargs: self.kwargs.as_ref().map(|k| k.bind(py)),
        })?;
        let common = PipelineCommon::new(
            child_data,
            child_resources,
            self.shared.implementations.clone(),
        );
        let pipeline = ContextManagerEnterPipeline::new(common);
        SyncContextEnterRunner::new(pipeline).run(py)
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
            && let ContextManagerState::Entered(exits) = &*state
        {
            exits.traverse(&visit)?;
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

        match self.run_enter(py) {
            Ok((value, exits)) => {
                *self.state.lock().expect("poisoned") = ContextManagerState::Entered(exits);
                Ok(value)
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
        let exits = {
            let mut state = self.state.lock().expect("poisoned");
            match std::mem::replace(&mut *state, ContextManagerState::Exited) {
                ContextManagerState::Entered(exits) => exits,
                other => {
                    *state = other;
                    return Err(PyRuntimeError::new_err(
                        "context manager has not been entered",
                    ));
                }
            }
        };
        let active = ExceptionTriple {
            exc_type,
            exc_val,
            exc_tb,
        };
        let pipeline = SyncExitDrainerPipeline::new(py, exits, active);
        SyncContextExitRunner::new(pipeline).run(py)
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
    pub(crate) fn new(
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

    fn build_enter_awaitable(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let (child_data, child_resources) = prepare_child_execution(ChildContext {
            py,
            shared: &self.shared,
            resources: self.shared.resources.clone_ref(py),
            args: self.args.bind(py),
            kwargs: self.kwargs.as_ref().map(|k| k.bind(py)),
        })?;
        let common = PipelineCommon::new(
            child_data,
            child_resources,
            self.shared.implementations.clone(),
        );
        let pipeline = AsyncContextManagerEnterPipeline::new(common);
        let generator = AsyncContextEnterGenerator::new(pipeline, Arc::clone(&self.state));
        let wrapper = AwaitableWrapper::new(Box::new(generator));
        Ok(Py::new(py, wrapper)?.into_any())
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
        if let Ok(state) = self.state.try_lock() {
            state.traverse(&visit)?;
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

        match self.build_enter_awaitable(py) {
            Ok(wrapper) => Ok(wrapper),
            Err(error) => {
                *self.state.lock().expect("poisoned") = AsyncContextManagerState::Exited;
                Err(error)
            }
        }
    }

    fn __aexit__(
        &self,
        py: Python<'_>,
        exc_type: Py<PyAny>,
        exc_val: Py<PyAny>,
        exc_tb: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let exits = {
            let mut state = self.state.lock().expect("poisoned");
            match std::mem::replace(&mut *state, AsyncContextManagerState::Exited) {
                AsyncContextManagerState::Entered(exits) => exits,
                other => {
                    *state = other;
                    return Err(PyRuntimeError::new_err(
                        "async context manager has not been entered",
                    ));
                }
            }
        };

        let active = ExceptionTriple {
            exc_type,
            exc_val,
            exc_tb,
        };
        let pipeline = AsyncExitDrainerPipeline::new(py, exits, active);
        let generator = AsyncExitGenerator::new(pipeline);
        let wrapper = AwaitableWrapper::new(Box::new(generator));
        Ok(Py::new(py, wrapper)?.into_any())
    }
}
