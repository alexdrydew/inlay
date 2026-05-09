use std::sync::Mutex;

use pyo3::PyTraverseError;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyTypeError};
use pyo3::gc::PyVisit;
use pyo3::prelude::*;

use crate::runtime::transition::pipeline::generator::{
    GeneratorCloseResult, GeneratorResult, PyGeneratorLike,
};

enum AwaitableAdapterState {
    // AwaitableWrapper is a pyclass, so it cannot be generic over the generator type.
    Ready(Box<dyn PyGeneratorLike>),
    Running,
    Done,
}

#[pyclass(frozen, module = "inlay")]
pub(crate) struct AwaitableWrapper {
    state: Mutex<AwaitableAdapterState>,
}

impl AwaitableWrapper {
    pub(crate) fn new(generator: Box<dyn PyGeneratorLike>) -> Self {
        Self {
            state: Mutex::new(AwaitableAdapterState::Ready(generator)),
        }
    }

    fn take_ready_for_send(
        &self,
        py: Python<'_>,
        value: &Py<PyAny>,
    ) -> PyResult<Box<dyn PyGeneratorLike>> {
        let mut state = self.state.lock().expect("poisoned");
        match &*state {
            AwaitableAdapterState::Running => {
                return Err(PyRuntimeError::new_err("awaitable is already running"));
            }
            AwaitableAdapterState::Done => return Err(PyStopIteration::new_err(py.None())),
            AwaitableAdapterState::Ready(generator)
                if generator.is_initial() && !value.bind(py).is_none() =>
            {
                return Err(PyTypeError::new_err(
                    "can't send non-None value to a just-started awaitable",
                ));
            }
            AwaitableAdapterState::Ready(_) => {}
        }

        // Leave a visible Running marker while the generator may call back into Python and drop the
        // lock; reentrant protocol calls then fail immediately instead of deadlocking.
        let AwaitableAdapterState::Ready(generator) =
            std::mem::replace(&mut *state, AwaitableAdapterState::Running)
        else {
            unreachable!();
        };
        Ok(generator)
    }

    fn take_ready_for_throw(
        &self,
        py: Python<'_>,
        typ: &Py<PyAny>,
    ) -> PyResult<Box<dyn PyGeneratorLike>> {
        let mut state = self.state.lock().expect("poisoned");
        match &*state {
            AwaitableAdapterState::Ready(_) => {}
            AwaitableAdapterState::Running => {
                return Err(PyRuntimeError::new_err("awaitable is already running"));
            }
            AwaitableAdapterState::Done => {
                return Err(PyErr::from_value(typ.clone_ref(py).into_bound(py)));
            }
        }
        let AwaitableAdapterState::Ready(generator) =
            std::mem::replace(&mut *state, AwaitableAdapterState::Running)
        else {
            unreachable!();
        };
        Ok(generator)
    }

    fn take_ready_for_close(&self) -> PyResult<Option<Box<dyn PyGeneratorLike>>> {
        let mut state = self.state.lock().expect("poisoned");
        match &*state {
            AwaitableAdapterState::Done => return Ok(None),
            AwaitableAdapterState::Running => {
                return Err(PyRuntimeError::new_err("awaitable is already running"));
            }
            AwaitableAdapterState::Ready(_) => {}
        }
        let AwaitableAdapterState::Ready(generator) =
            std::mem::replace(&mut *state, AwaitableAdapterState::Running)
        else {
            unreachable!();
        };
        Ok(Some(generator))
    }

    fn store_ready(&self, generator: Box<dyn PyGeneratorLike>) {
        *self.state.lock().expect("poisoned") = AwaitableAdapterState::Ready(generator);
    }

    fn store_done(&self) {
        *self.state.lock().expect("poisoned") = AwaitableAdapterState::Done;
    }

    fn finish_generator_result(
        &self,
        generator: Box<dyn PyGeneratorLike>,
        result: GeneratorResult,
    ) -> PyResult<Py<PyAny>> {
        match result {
            GeneratorResult::Suspended(value) => {
                self.store_ready(generator);
                Ok(value)
            }
            GeneratorResult::Completed(value) => {
                self.store_done();
                Err(PyStopIteration::new_err(value))
            }
            GeneratorResult::Failed(error) => {
                self.store_done();
                Err(error)
            }
        }
    }
}

#[pymethods]
impl AwaitableWrapper {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Ok(state) = self.state.try_lock()
            && let AwaitableAdapterState::Ready(generator) = &*state
        {
            generator.traverse(&visit)?;
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
        let mut generator = self.take_ready_for_send(py, &value)?;
        let result = generator.send(py, value);
        self.finish_generator_result(generator, result)
    }

    #[pyo3(signature = (typ, val=None, tb=None))]
    fn throw(
        &self,
        py: Python<'_>,
        typ: Py<PyAny>,
        val: Option<Py<PyAny>>,
        tb: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let mut generator = self.take_ready_for_throw(py, &typ)?;
        let result = generator.throw(py, typ, val, tb);
        self.finish_generator_result(generator, result)
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let Some(mut generator) = self.take_ready_for_close()? else {
            return Ok(());
        };

        match generator.close(py) {
            GeneratorCloseResult::Closed => {
                self.store_done();
                Ok(())
            }
            GeneratorCloseResult::IgnoredGeneratorExit(error) => {
                // The coroutine yielded during close, we follow CPython behavior and keep
                // generator state while raising an error.
                self.store_ready(generator);
                Err(error)
            }
            GeneratorCloseResult::Failed(error) => {
                self.store_done();
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};

    use super::*;

    struct TestGeneratorLike {
        close_results: VecDeque<GeneratorCloseResult>,
    }

    impl TestGeneratorLike {
        fn with_close_results(close_results: Vec<GeneratorCloseResult>) -> Self {
            Self {
                close_results: close_results.into(),
            }
        }
    }

    impl PyGeneratorLike for TestGeneratorLike {
        fn is_initial(&self) -> bool {
            false
        }

        fn send(&mut self, py: Python<'_>, _value: Py<PyAny>) -> GeneratorResult {
            GeneratorResult::Completed(py.None())
        }

        fn throw(
            &mut self,
            py: Python<'_>,
            _typ: Py<PyAny>,
            _val: Option<Py<PyAny>>,
            _tb: Option<Py<PyAny>>,
        ) -> GeneratorResult {
            GeneratorResult::Completed(py.None())
        }

        fn close(&mut self, _py: Python<'_>) -> GeneratorCloseResult {
            self.close_results
                .pop_front()
                .unwrap_or(GeneratorCloseResult::Closed)
        }

        fn traverse(&self, _visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
            Ok(())
        }
    }

    fn wrapper_with_state(state: AwaitableAdapterState) -> AwaitableWrapper {
        AwaitableWrapper {
            state: Mutex::new(state),
        }
    }

    fn wrapper_with_generator(generator: impl PyGeneratorLike + 'static) -> AwaitableWrapper {
        AwaitableWrapper::new(Box::new(generator))
    }

    fn with_python(test: impl FnOnce(Python<'_>)) {
        Python::initialize();
        Python::attach(test);
    }

    #[test]
    fn completed_awaitable_close_is_noop() {
        with_python(|py| {
            let wrapper = wrapper_with_state(AwaitableAdapterState::Done);

            wrapper.close(py).expect("close should be a no-op");
        });
    }

    #[test]
    fn completed_awaitable_send_raises_stop_iteration() {
        with_python(|py| {
            let wrapper = wrapper_with_state(AwaitableAdapterState::Done);

            let error = wrapper.send(py, py.None()).expect_err("send should fail");

            assert!(error.is_instance_of::<PyStopIteration>(py));
        });
    }

    #[test]
    fn completed_awaitable_throw_raises_supplied_exception() {
        with_python(|py| {
            let wrapper = wrapper_with_state(AwaitableAdapterState::Done);
            let supplied = PyValueError::new_err("sentinel");
            let exception = supplied.value(py).clone().into_any().unbind();

            let error = wrapper
                .throw(py, exception, None, None)
                .expect_err("throw should fail");

            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(error.value(py).to_string(), "sentinel");
        });
    }

    #[test]
    fn running_awaitable_protocol_calls_raise_reentrancy_error() {
        with_python(|py| {
            let wrapper = wrapper_with_state(AwaitableAdapterState::Running);

            let send_error = wrapper.send(py, py.None()).expect_err("send should fail");
            assert!(send_error.is_instance_of::<PyRuntimeError>(py));
            assert_eq!(
                send_error.value(py).to_string(),
                "awaitable is already running"
            );

            let throw_error = wrapper
                .throw(py, py.None(), None, None)
                .expect_err("throw should fail");
            assert!(throw_error.is_instance_of::<PyRuntimeError>(py));
            assert_eq!(
                throw_error.value(py).to_string(),
                "awaitable is already running"
            );

            let close_error = wrapper.close(py).expect_err("close should fail");
            assert!(close_error.is_instance_of::<PyRuntimeError>(py));
            assert_eq!(
                close_error.value(py).to_string(),
                "awaitable is already running"
            );
        });
    }

    #[test]
    fn terminal_close_error_marks_awaitable_done() {
        with_python(|py| {
            let wrapper = wrapper_with_generator(TestGeneratorLike::with_close_results(vec![
                GeneratorCloseResult::Failed(PyValueError::new_err("close failed")),
            ]));

            let close_error = wrapper.close(py).expect_err("close should fail");
            assert!(close_error.is_instance_of::<PyValueError>(py));
            assert_eq!(close_error.value(py).to_string(), "close failed");

            let send_error = wrapper.send(py, py.None()).expect_err("send should fail");
            assert!(send_error.is_instance_of::<PyStopIteration>(py));
        });
    }

    #[test]
    fn ignored_generator_exit_preserves_generator() {
        with_python(|py| {
            let wrapper = wrapper_with_generator(TestGeneratorLike::with_close_results(vec![
                GeneratorCloseResult::IgnoredGeneratorExit(PyRuntimeError::new_err(
                    "coroutine ignored GeneratorExit",
                )),
                GeneratorCloseResult::Closed,
            ]));

            let close_error = wrapper.close(py).expect_err("close should fail");
            assert!(close_error.is_instance_of::<PyRuntimeError>(py));
            assert_eq!(
                close_error.value(py).to_string(),
                "coroutine ignored GeneratorExit"
            );

            wrapper
                .close(py)
                .expect("second close should reach preserved generator");
            let send_error = wrapper.send(py, py.None()).expect_err("send should fail");
            assert!(send_error.is_instance_of::<PyStopIteration>(py));
        });
    }
}
