use pyo3::prelude::*;

use super::exits::ExceptionTriple;

pub(crate) enum AwaitableMethodStep {
    Await(Py<PyAny>),
    Done(Py<PyAny>),
}

pub(crate) enum ContextManagerEnterStep {
    EnterSync(Py<PyAny>),
    Done(Py<PyAny>),
}

pub(crate) enum AsyncContextManagerEnterStep {
    EnterSync(Py<PyAny>),
    EnterAsync(Py<PyAny>),
    Await(Py<PyAny>),
    Done(Py<PyAny>),
}

pub(crate) enum ExitOutcome {
    Returned(Py<PyAny>),
    Raised(PyErr),
}

pub(crate) enum ExitDrainCompletion {
    Return(bool),
    Raise(PyErr),
}

pub(crate) enum SyncExitStep {
    ExitSync {
        context: Py<PyAny>,
        exc: ExceptionTriple,
    },
    Done(ExitDrainCompletion),
}

pub(crate) enum AsyncExitStep {
    ExitSync {
        context: Py<PyAny>,
        exc: ExceptionTriple,
    },
    ExitAsync {
        context: Py<PyAny>,
        exc: ExceptionTriple,
    },
    Done(ExitDrainCompletion),
}
