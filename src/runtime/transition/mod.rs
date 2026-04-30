use std::sync::Arc;

use pyo3::PyTraverseError;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::compile::flatten::{
    ExecutionGraph, ExecutionHook, ExecutionNodeId, ExecutionParam, ExecutionResultBinding,
    ExecutionSourceNodeId,
};
use crate::types::{ParamKind, WrapperKind};

use super::executor::{ContextData, ScopeHandle, WeakScopeHandle, attach_scope, execute};
use super::proxy::{ContextProxy, DelegatedAttr};
use super::scope::Scope;

mod wrappers;

pub(crate) use wrappers::{AsyncContextManagerWrapper, AwaitableWrapper, ContextManagerWrapper};

// ---------------------------------------------------------------------------
// Shared transition types
// ---------------------------------------------------------------------------

pub(crate) enum TransitionKind {
    Method {
        implementation: Py<PyAny>,
        bound_instance: Option<Py<PyAny>>,
        result_source: ExecutionSourceNodeId,
        result_bindings: Vec<ExecutionResultBinding>,
    },
    Auto,
}

#[derive(Clone)]
pub(crate) struct TransitionShared {
    pub(crate) graph: Arc<ExecutionGraph>,
    pub(crate) parent_scope: WeakScopeHandle,
    pub(crate) target: ExecutionNodeId,
    pub(crate) params: Vec<ExecutionParam>,
    pub(crate) accepts_varargs: bool,
    pub(crate) accepts_varkw: bool,
    pub(crate) hooks: Vec<ExecutionHook>,
}

impl TransitionKind {
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match self {
            TransitionKind::Method {
                implementation,
                bound_instance,
                ..
            } => {
                visit.call(implementation)?;
                if let Some(inst) = bound_instance {
                    visit.call(inst)?;
                }
                Ok(())
            }
            TransitionKind::Auto => Ok(()),
        }
    }
}

impl TransitionShared {
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for hook in &self.hooks {
            hook.traverse(visit)?;
        }
        Ok(())
    }
}

fn traverse_scope_owner(
    scope_owner: &Option<ScopeHandle>,
    visit: &PyVisit<'_>,
) -> Result<(), PyTraverseError> {
    if let Some(handle) = scope_owner {
        if let Some(scope) = handle.get() {
            scope.traverse_py_refs(visit)?;
        }
    }
    Ok(())
}

/// Parameters captured by [`AwaitableWrapper`] for deferred child execution.
struct ChildExecutionParams {
    shared: TransitionShared,
    kind: TransitionKind,
    args: Py<PyTuple>,
    kwargs: Option<Py<PyDict>>,
}

impl ChildExecutionParams {
    fn new(
        shared: &TransitionShared,
        kind: &TransitionKind,
        py: Python<'_>,
        args: Py<PyTuple>,
        kwargs: Option<Py<PyDict>>,
    ) -> Self {
        Self {
            shared: shared.clone(),
            kind: clone_kind(kind, py),
            args,
            kwargs,
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.shared.traverse(visit)?;
        self.kind.traverse(visit)?;
        visit.call(&self.args)?;
        if let Some(kwargs) = &self.kwargs {
            visit.call(kwargs)?;
        }
        Ok(())
    }
}

struct ChildContext<'a, 'py> {
    py: Python<'py>,
    shared: &'a TransitionShared,
    parent_scope: &'a Arc<Scope>,
    args: &'a Bound<'py, PyTuple>,
    kwargs: Option<&'a Bound<'py, PyDict>>,
    kind: &'a TransitionKind,
    method_result: Option<Py<PyAny>>,
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn get_parent_scope(handle: &WeakScopeHandle) -> PyResult<Arc<Scope>> {
    let strong = handle
        .upgrade()
        .ok_or_else(|| PyRuntimeError::new_err("parent scope has been deallocated"))?;
    strong
        .get()
        .cloned()
        .ok_or_else(|| PyRuntimeError::new_err("transition called before parent scope was frozen"))
}

fn validate_param_signature(
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

/// Extract parameter values from Python args/kwargs and pair them with their
/// source identities for insertion into a child scope.
fn extract_param_sources(
    _py: Python<'_>,
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
        result.push((param.source, value));
    }

    Ok(result)
}

fn extract_result_bindings(
    result_bindings: &[ExecutionResultBinding],
    result_val: &Bound<'_, PyAny>,
) -> PyResult<Vec<(ExecutionSourceNodeId, Py<PyAny>)>> {
    if result_bindings.is_empty() {
        return Ok(Vec::new());
    }

    let dict = result_val.cast::<PyDict>().map_err(|_| {
        PyRuntimeError::new_err("transition result bindings require a TypedDict-backed dict")
    })?;
    let mut result = Vec::with_capacity(result_bindings.len());

    for binding in result_bindings {
        dict.get_item(&*binding.name)?.ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "missing transition result field '{}'",
                binding.name
            ))
        })?;
        let value = Py::new(
            result_val.py(),
            DelegatedAttr {
                source: result_val.clone().unbind(),
                name: Arc::clone(&binding.name),
            },
        )?
        .into_any();
        result.push((binding.source, value));
    }

    Ok(result)
}

/// Build a child scope, execute the target subtree, run hooks, freeze the
/// child scope, and return the result.
fn execute_child_context(context: ChildContext<'_, '_>) -> PyResult<Py<PyAny>> {
    let py = context.py;
    let shared = context.shared;
    let mut new_sources = extract_param_sources(
        py,
        &shared.params,
        shared.accepts_varargs,
        shared.accepts_varkw,
        context.args,
        context.kwargs,
    )?;

    // For Method transitions, add the implementation's return value and any
    // projected TypedDict field bindings to the child scope.
    if let (
        TransitionKind::Method {
            result_source,
            result_bindings,
            ..
        },
        Some(result_val),
    ) = (context.kind, context.method_result)
    {
        new_sources.push((*result_source, result_val.clone_ref(py)));

        let mut existing_sources = std::collections::HashSet::with_capacity(new_sources.len());
        for (source, _) in &new_sources {
            existing_sources.insert(*source);
        }

        for (source, value) in extract_result_bindings(result_bindings, result_val.bind(py))? {
            if existing_sources.insert(source) {
                new_sources.push((source, value));
            }
        }
    }

    let child_scope = Scope::child(Arc::clone(context.parent_scope), new_sources);
    let child_data = ContextData {
        graph: Arc::clone(&shared.graph),
        root_node: shared.target,
    };
    let (result, scope_handle) = execute(py, &child_data, child_scope, &shared.hooks)?;
    let result = wrap_transition_leaf_result(py, result)?;
    attach_scope(py, result, scope_handle)
}

fn wrap_transition_leaf_result(py: Python<'_>, result: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let bound = result.bind(py);
    let Ok(attribute) = bound.cast::<DelegatedAttr>() else {
        return Ok(result);
    };
    let attribute = attribute.borrow();
    let members = std::iter::once((attribute.name.clone(), result.clone_ref(py))).collect();
    Ok(Py::new(py, ContextProxy::new(members, Default::default()))?.into_any())
}

/// Variant of [`execute_child_context`] that takes owned [`ChildExecutionParams`].
fn execute_child_from_params(
    py: Python<'_>,
    cep: &ChildExecutionParams,
    method_result: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let parent = get_parent_scope(&cep.shared.parent_scope)?;
    execute_child_context(ChildContext {
        py,
        shared: &cep.shared,
        parent_scope: &parent,
        args: cep.args.bind(py),
        kwargs: cep.kwargs.as_ref().map(|k| k.bind(py)),
        kind: &cep.kind,
        method_result,
    })
}

/// Call the implementation with optional bound instance + caller's args/kwargs.
fn call_implementation(
    py: Python<'_>,
    implementation: &Py<PyAny>,
    bound_instance: &Option<Py<PyAny>>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    match bound_instance {
        Some(bound) => {
            let full_args = prepend_to_tuple(py, bound.bind(py), args)?;
            implementation.call(py, full_args, kwargs)
        }
        None => implementation.call(py, args, kwargs),
    }
}

/// Create a new tuple with `first` prepended to `rest`.
fn prepend_to_tuple<'py>(
    py: Python<'py>,
    first: &Bound<'py, PyAny>,
    rest: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyTuple>> {
    let mut items: Vec<Bound<'py, PyAny>> = Vec::with_capacity(rest.len() + 1);
    items.push(first.clone());
    for item in rest.iter() {
        items.push(item);
    }
    PyTuple::new(py, items)
}

/// Extract the value from a caught `StopIteration` exception.
fn stop_iteration_value(py: Python<'_>, err: &PyErr) -> Py<PyAny> {
    err.value(py)
        .getattr("value")
        .map(|v| v.unbind())
        .unwrap_or_else(|_| py.None())
}

// ---------------------------------------------------------------------------
// Transition — outer callable, dispatches on WrapperKind
// ---------------------------------------------------------------------------

#[pyclass(frozen, module = "inlay")]
pub(crate) struct Transition {
    shared: TransitionShared,
    kind: TransitionKind,
    return_wrapper: WrapperKind,
}

impl Transition {
    pub(crate) fn new(
        shared: TransitionShared,
        kind: TransitionKind,
        return_wrapper: WrapperKind,
    ) -> Self {
        Self {
            shared,
            kind,
            return_wrapper,
        }
    }
}

#[pymethods]
impl Transition {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.shared.traverse(&visit)?;
        self.kind.traverse(&visit)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        self.call_inner(py, args, kwargs, None)
    }
}

impl Transition {
    pub(crate) fn call_with_scope_owner(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
        scope_owner: ScopeHandle,
    ) -> PyResult<Py<PyAny>> {
        self.call_inner(py, args, kwargs, Some(scope_owner))
    }

    fn call_inner(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
        scope_owner: Option<ScopeHandle>,
    ) -> PyResult<Py<PyAny>> {
        if matches!(&self.kind, TransitionKind::Auto) {
            validate_param_signature(
                &self.shared.params,
                self.shared.accepts_varargs,
                self.shared.accepts_varkw,
                args,
                kwargs,
            )?;
        }

        match self.return_wrapper {
            WrapperKind::None => self.call_sync(py, args, kwargs),
            WrapperKind::ContextManager => self.call_context_manager(py, args, kwargs, scope_owner),
            WrapperKind::Awaitable => self.call_awaitable(py, args, kwargs, scope_owner),
            WrapperKind::AsyncContextManager => {
                self.call_async_context_manager(py, args, kwargs, scope_owner)
            }
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
        let parent = get_parent_scope(&self.shared.parent_scope)?;
        let method_result = match &self.kind {
            TransitionKind::Method {
                implementation,
                bound_instance,
                ..
            } => Some(call_implementation(
                py,
                implementation,
                bound_instance,
                args,
                kwargs,
            )?),
            TransitionKind::Auto => None,
        };
        execute_child_context(ChildContext {
            py,
            shared: &self.shared,
            parent_scope: &parent,
            args,
            kwargs,
            kind: &self.kind,
            method_result,
        })
    }

    fn call_context_manager(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
        scope_owner: Option<ScopeHandle>,
    ) -> PyResult<Py<PyAny>> {
        let wrapper = ContextManagerWrapper::new(
            self.shared.clone(),
            clone_kind(&self.kind, py),
            args.clone().unbind(),
            kwargs.map(|k| k.clone().unbind()),
            scope_owner,
        );
        Ok(Py::new(py, wrapper)?.into_any())
    }

    fn call_awaitable(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
        scope_owner: Option<ScopeHandle>,
    ) -> PyResult<Py<PyAny>> {
        // For Method: call implementation synchronously to get the coroutine object.
        // For Auto: no coroutine.
        let inner_coro = match &self.kind {
            TransitionKind::Method {
                implementation,
                bound_instance,
                ..
            } => Some(call_implementation(
                py,
                implementation,
                bound_instance,
                args,
                kwargs,
            )?),
            TransitionKind::Auto => None,
        };
        let child_exec = ChildExecutionParams::new(
            &self.shared,
            &self.kind,
            py,
            args.clone().unbind(),
            kwargs.map(|k| k.clone().unbind()),
        );
        let wrapper = AwaitableWrapper::new(inner_coro, Some(child_exec), scope_owner);
        Ok(Py::new(py, wrapper)?.into_any())
    }

    fn call_async_context_manager(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
        scope_owner: Option<ScopeHandle>,
    ) -> PyResult<Py<PyAny>> {
        let wrapper = AsyncContextManagerWrapper::new(
            self.shared.clone(),
            clone_kind(&self.kind, py),
            args.clone().unbind(),
            kwargs.map(|k| k.clone().unbind()),
            scope_owner,
        );
        Ok(Py::new(py, wrapper)?.into_any())
    }
}

/// Clone a `TransitionKind`, incrementing Python reference counts.
fn clone_kind(kind: &TransitionKind, py: Python<'_>) -> TransitionKind {
    match kind {
        TransitionKind::Method {
            implementation,
            bound_instance,
            result_source,
            result_bindings,
        } => TransitionKind::Method {
            implementation: implementation.clone_ref(py),
            bound_instance: bound_instance.as_ref().map(|b| b.clone_ref(py)),
            result_source: *result_source,
            result_bindings: result_bindings.clone(),
        },
        TransitionKind::Auto => TransitionKind::Auto,
    }
}
