use std::sync::Arc;

use pyo3::PyTraverseError;
use pyo3::exceptions::PyTypeError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::compile::flatten::{
    ExecutionGraph, ExecutionMethodImplementation, ExecutionNodeId, ExecutionParam,
    ExecutionSourceNodeId,
};
use crate::types::{ParamKind, WrapperKind};

use super::executor::ContextData;
use super::proxy::{ContextProxy, DelegatedMember};
use super::resource_plan::{
    resource_plan_for_roots, transition_body_roots, transition_introduced_sources,
};
use super::resources::RuntimeResources;

pub(crate) mod pipeline;
mod wrappers;

use pipeline::generator::TransitionGenerator;
use pipeline::pipelines::{EnterProgram, PipelineCommon};

pub(crate) use wrappers::{AsyncContextManagerWrapper, AwaitableWrapper, ContextManagerWrapper};

pub(crate) struct TransitionShared {
    pub(crate) graph: Arc<ExecutionGraph>,
    pub(crate) resources: RuntimeResources,
    pub(crate) target: ExecutionNodeId,
    pub(crate) params: Vec<ExecutionParam>,
    pub(crate) accepts_varargs: bool,
    pub(crate) accepts_varkw: bool,
    pub(crate) implementations: Vec<ExecutionMethodImplementation>,
}

impl TransitionShared {
    pub(crate) fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            graph: Arc::clone(&self.graph),
            resources: self.resources.clone_ref(py),
            target: self.target,
            params: self.params.clone(),
            accepts_varargs: self.accepts_varargs,
            accepts_varkw: self.accepts_varkw,
            implementations: self.implementations.clone(),
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.resources.traverse_py_refs(visit)?;
        for implementation in &self.implementations {
            visit.call(&*implementation.implementation)?;
        }
        Ok(())
    }
}

pub(crate) struct ChildContext<'a, 'py> {
    py: Python<'py>,
    shared: &'a TransitionShared,
    resources: RuntimeResources,
    args: &'a Bound<'py, PyTuple>,
    kwargs: Option<&'a Bound<'py, PyDict>>,
}

pub(crate) fn validate_param_signature(
    params: &[ExecutionParam],
    accepts_varargs: bool,
    accepts_varkw: bool,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let max_positional = params
        .iter()
        .filter(|param| !matches!(param.kind, ParamKind::KeywordOnly))
        .count();
    if !accepts_varargs && args.len() > max_positional {
        return Err(PyTypeError::new_err(format!(
            "takes {max_positional} positional arguments but {} were given",
            args.len()
        )));
    }

    if let Some(kw) = kwargs {
        for (key, _) in kw.iter() {
            let name = key.extract::<String>()?;
            let Some(param) = params
                .iter()
                .find(|param| param.name.as_ref() == name.as_str())
            else {
                if accepts_varkw {
                    continue;
                }
                return Err(PyTypeError::new_err(format!(
                    "got an unexpected keyword argument '{name}'"
                )));
            };
            if matches!(param.kind, ParamKind::PositionalOnly) {
                return Err(PyTypeError::new_err(format!(
                    "got positional-only argument '{}' passed as keyword argument",
                    param.name
                )));
            }
        }
    }

    let mut pos_index = 0usize;
    for param in params {
        match param.kind {
            ParamKind::PositionalOnly => {
                if pos_index < args.len() {
                    pos_index += 1;
                } else {
                    return Err(PyTypeError::new_err(format!(
                        "missing required argument '{}'",
                        param.name
                    )));
                }
            }
            ParamKind::PositionalOrKeyword => {
                if pos_index < args.len() {
                    if kwargs
                        .map(|kw| kw.get_item(&*param.name))
                        .transpose()?
                        .flatten()
                        .is_some()
                    {
                        return Err(PyTypeError::new_err(format!(
                            "got multiple values for argument '{}'",
                            param.name
                        )));
                    }
                    pos_index += 1;
                } else if kwargs
                    .map(|kw| kw.get_item(&*param.name))
                    .transpose()?
                    .flatten()
                    .is_none()
                {
                    return Err(PyTypeError::new_err(format!(
                        "missing required argument '{}'",
                        param.name
                    )));
                }
            }
            ParamKind::KeywordOnly => {
                if kwargs
                    .map(|kw| kw.get_item(&*param.name))
                    .transpose()?
                    .flatten()
                    .is_none()
                {
                    return Err(PyTypeError::new_err(format!(
                        "missing required keyword-only argument '{}'",
                        param.name
                    )));
                }
            }
        }
    }

    Ok(())
}

fn extract_param_sources(
    py: Python<'_>,
    params: &[ExecutionParam],
    accepts_varargs: bool,
    accepts_varkw: bool,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<(ExecutionSourceNodeId, Py<PyAny>)>> {
    validate_param_signature(params, accepts_varargs, accepts_varkw, args, kwargs)?;

    let mut result = Vec::with_capacity(params.len());
    let mut pos_index: usize = 0;

    for param in params {
        let value = match param.kind {
            ParamKind::PositionalOnly => {
                let val = args.get_item(pos_index)?;
                pos_index += 1;
                val.unbind()
            }
            ParamKind::PositionalOrKeyword => {
                if pos_index < args.len() {
                    let val = args.get_item(pos_index)?;
                    pos_index += 1;
                    val.unbind()
                } else if let Some(kw) = kwargs {
                    kw.get_item(&*param.name)?
                        .ok_or_else(|| {
                            PyTypeError::new_err(format!("missing argument '{}'", param.name))
                        })?
                        .unbind()
                } else {
                    return Err(PyTypeError::new_err(format!(
                        "missing argument '{}'",
                        param.name
                    )));
                }
            }
            ParamKind::KeywordOnly => {
                let kw = kwargs.ok_or_else(|| {
                    PyTypeError::new_err(format!("keyword-only argument '{}' missing", param.name))
                })?;
                kw.get_item(&*param.name)?
                    .ok_or_else(|| {
                        PyTypeError::new_err(format!(
                            "keyword-only argument '{}' missing",
                            param.name
                        ))
                    })?
                    .unbind()
            }
        };
        for &source in &param.sources {
            result.push((source, value.clone_ref(py)));
        }
    }

    Ok(result)
}

pub(crate) fn prepare_child_execution(
    mut context: ChildContext<'_, '_>,
) -> PyResult<(ContextData, RuntimeResources)> {
    let py = context.py;
    let shared = context.shared;
    let new_sources = extract_param_sources(
        py,
        &shared.params,
        shared.accepts_varargs,
        shared.accepts_varkw,
        context.args,
        context.kwargs,
    )?;
    let introduced_ids = transition_introduced_sources(&shared.params, &shared.implementations);

    let plan = resource_plan_for_roots(
        &shared.graph,
        transition_body_roots(shared.target, &shared.implementations),
        &introduced_ids,
    );
    let mut child_resources = context.resources.capture_plan(py, &plan)?;
    for (source, value) in new_sources {
        child_resources.insert_source(source, value);
    }
    let child_data = ContextData {
        graph: Arc::clone(&shared.graph),
        root_node: shared.target,
    };
    Ok((child_data, child_resources))
}

pub(crate) fn wrap_transition_leaf_result(
    py: Python<'_>,
    graph: Arc<ExecutionGraph>,
    result: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    let bound = result.bind(py);
    let Ok(member) = bound.cast::<DelegatedMember>() else {
        return Ok(result);
    };
    let member = member.borrow();
    let members = std::iter::once((member.name.clone(), result.clone_ref(py))).collect();
    Ok(Py::new(py, ContextProxy::from_materialized(graph, members))?.into_any())
}

#[pyclass(frozen, weakref, module = "inlay")]
pub(crate) struct Transition {
    shared: TransitionShared,
    return_wrapper: WrapperKind,
}

impl Transition {
    pub(crate) fn new(shared: TransitionShared, return_wrapper: WrapperKind) -> Self {
        Self {
            shared,
            return_wrapper,
        }
    }
}

#[pymethods]
impl Transition {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.shared.traverse(&visit)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        self.call_inner(py, args, kwargs)
    }
}

impl Transition {
    fn call_inner(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        validate_param_signature(
            &self.shared.params,
            self.shared.accepts_varargs,
            self.shared.accepts_varkw,
            args,
            kwargs,
        )?;

        match self.return_wrapper {
            WrapperKind::None => self.call_sync(py, args, kwargs),
            WrapperKind::ContextManager => self.call_context_manager(py, args, kwargs),
            WrapperKind::Awaitable => self.call_awaitable(py, args, kwargs),
            WrapperKind::AsyncContextManager => self.call_async_context_manager(py, args, kwargs),
        }
    }
}

impl Transition {
    fn call_sync(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let (child_data, child_resources) = prepare_child_execution(ChildContext {
            py,
            shared: &self.shared,
            resources: self.shared.resources.clone_ref(py),
            args,
            kwargs,
        })?;
        let common = PipelineCommon::new(
            child_data,
            child_resources,
            self.shared.implementations.clone(),
        );
        match EnterProgram::new(common).poll(py)? {
            pipeline::pipelines::EnterPoll::Completed(value) => Ok(value),
            pipeline::pipelines::EnterPoll::Effect(_) => {
                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "plain sync transition unexpectedly emitted an effect",
                ))
            }
        }
    }

    fn call_context_manager(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let wrapper = ContextManagerWrapper::new(
            self.shared.clone_ref(py),
            args.clone().unbind(),
            kwargs.map(|k| k.clone().unbind()),
        );
        Ok(Py::new(py, wrapper)?.into_any())
    }

    fn call_awaitable(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let (child_data, child_resources) = prepare_child_execution(ChildContext {
            py,
            shared: &self.shared,
            resources: self.shared.resources.clone_ref(py),
            args,
            kwargs,
        })?;
        let common = PipelineCommon::new(
            child_data,
            child_resources,
            self.shared.implementations.clone(),
        );
        let pipeline = EnterProgram::new(common);
        let generator = TransitionGenerator::awaitable_method(pipeline);
        let wrapper = AwaitableWrapper::new(Box::new(generator));
        Ok(Py::new(py, wrapper)?.into_any())
    }

    fn call_async_context_manager(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let wrapper = AsyncContextManagerWrapper::new(
            self.shared.clone_ref(py),
            args.clone().unbind(),
            kwargs.map(|k| k.clone().unbind()),
        );
        Ok(Py::new(py, wrapper)?.into_any())
    }
}
