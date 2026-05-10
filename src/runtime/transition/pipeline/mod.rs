use pyo3::PyTraverseError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;

pub(crate) mod generator;
pub(crate) mod pipelines;

pub(crate) struct ExceptionTriple {
    pub(crate) exc_type: Py<PyAny>,
    pub(crate) exc_val: Py<PyAny>,
    pub(crate) exc_tb: Py<PyAny>,
}

impl ExceptionTriple {
    pub(crate) fn none(py: Python<'_>) -> Self {
        Self {
            exc_type: py.None(),
            exc_val: py.None(),
            exc_tb: py.None(),
        }
    }

    pub(crate) fn from_error(py: Python<'_>, error: &PyErr) -> Self {
        Self {
            exc_type: error.get_type(py).into_any().unbind(),
            exc_val: error.value(py).clone().into_any().unbind(),
            exc_tb: error
                .traceback(py)
                .map(|tb| tb.into_any().unbind())
                .unwrap_or_else(|| py.None()),
        }
    }

    pub(crate) fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            exc_type: self.exc_type.clone_ref(py),
            exc_val: self.exc_val.clone_ref(py),
            exc_tb: self.exc_tb.clone_ref(py),
        }
    }

    pub(crate) fn is_some(&self, py: Python<'_>) -> bool {
        !self.exc_type.bind(py).is_none()
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.exc_type)?;
        visit.call(&self.exc_val)?;
        visit.call(&self.exc_tb)?;
        Ok(())
    }
}

pub(crate) enum ExitItem {
    Sync(Py<PyAny>),
    Async(Py<PyAny>),
}

impl ExitItem {
    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match self {
            Self::Sync(context) | Self::Async(context) => visit.call(context),
        }
    }
}

#[derive(Default)]
pub(crate) struct MixedExitStack(Vec<ExitItem>);

impl MixedExitStack {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn push_sync(&mut self, context: Py<PyAny>) {
        self.0.push(ExitItem::Sync(context));
    }

    pub(crate) fn push_item(&mut self, item: ExitItem) {
        self.0.push(item);
    }

    pub(crate) fn pop(&mut self) -> Option<ExitItem> {
        self.0.pop()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for item in &self.0 {
            item.traverse(visit)?;
        }
        Ok(())
    }
}

pub(crate) enum ExitOutcome {
    Returned(Py<PyAny>),
    Raised(PyErr),
}
