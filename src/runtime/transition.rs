use std::sync::{Arc, Mutex, OnceLock};

use pyo3::PyTraverseError;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration};
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::registry::Source;
use crate::rules::{MethodParam, TransitionResultBinding};
use crate::types::{ParamKind, SlotBackend, WrapperKind};

use super::executor::{ContextData, WeakScopeHandle, attach_scope, execute};
use super::flatten::{ExecutionGraph, ExecutionHook, ExecutionNodeId};
use super::proxy::{ContextProxy, DelegatedAttr};
use super::scope::Scope;

// ---------------------------------------------------------------------------
// Shared transition types
// ---------------------------------------------------------------------------

pub(crate) enum TransitionKind {
    Method {
        implementation: Py<PyAny>,
        bound_instance: Option<Py<PyAny>>,
        result_source: Option<Source<SlotBackend>>,
        result_bindings: Vec<TransitionResultBinding<SlotBackend>>,
    },
    Auto,
}

pub(crate) struct TransitionShared {
    pub(crate) graph: Arc<ExecutionGraph<SlotBackend>>,
    pub(crate) parent_scope: WeakScopeHandle,
    pub(crate) target: ExecutionNodeId,
    pub(crate) params: Vec<MethodParam<SlotBackend>>,
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

/// Parameters captured by [`AwaitableWrapper`] for deferred child execution.
struct ChildExecutionParams {
    graph: Arc<ExecutionGraph<SlotBackend>>,
    parent_scope: WeakScopeHandle,
    target: ExecutionNodeId,
    kind: TransitionKind,
    params: Vec<MethodParam<SlotBackend>>,
    hooks: Vec<ExecutionHook>,
    args: Py<PyTuple>,
    kwargs: Option<Py<PyDict>>,
}

impl ChildExecutionParams {
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.kind.traverse(visit)?;
        for hook in &self.hooks {
            hook.traverse(visit)?;
        }
        visit.call(&self.args)?;
        if let Some(kwargs) = &self.kwargs {
            visit.call(kwargs)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn get_parent_scope(handle: &WeakScopeHandle) -> PyResult<Arc<Scope<SlotBackend>>> {
    let strong = handle
        .upgrade()
        .ok_or_else(|| PyRuntimeError::new_err("parent scope has been deallocated"))?;
    strong
        .get()
        .cloned()
        .ok_or_else(|| PyRuntimeError::new_err("transition called before parent scope was frozen"))
}

/// Extract parameter values from Python args/kwargs and pair them with their
/// source identities for insertion into a child scope.
fn extract_param_sources(
    _py: Python<'_>,
    params: &[MethodParam<SlotBackend>],
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<(Source<SlotBackend>, Py<PyAny>)>> {
    let mut result = Vec::with_capacity(params.len());
    let mut pos_index: usize = 0;

    for param in params {
        let value = match param.kind {
            ParamKind::PositionalOnly | ParamKind::PositionalOrKeyword => {
                if pos_index < args.len() {
                    let val = args.get_item(pos_index)?;
                    pos_index += 1;
                    val.unbind()
                } else if let Some(kw) = kwargs {
                    kw.get_item(&*param.name)?
                        .ok_or_else(|| {
                            PyRuntimeError::new_err(format!("missing argument '{}'", param.name))
                        })?
                        .unbind()
                } else {
                    return Err(PyRuntimeError::new_err(format!(
                        "missing argument '{}'",
                        param.name
                    )));
                }
            }
            ParamKind::KeywordOnly => {
                let kw = kwargs.ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "keyword-only argument '{}' missing",
                        param.name
                    ))
                })?;
                kw.get_item(&*param.name)?
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(format!(
                            "keyword-only argument '{}' missing",
                            param.name
                        ))
                    })?
                    .unbind()
            }
        };
        if let Some(source) = &param.source {
            result.push((source.clone(), value));
        }
    }

    Ok(result)
}

fn extract_result_bindings(
    result_bindings: &[TransitionResultBinding<SlotBackend>],
    result_val: &Bound<'_, PyAny>,
) -> PyResult<Vec<(Source<SlotBackend>, Py<PyAny>)>> {
    if result_bindings.is_empty() {
        return Ok(Vec::new());
    }

    let dict = result_val.cast::<PyDict>().map_err(|_| {
        PyRuntimeError::new_err("transition result bindings require a TypedDict-backed dict")
    })?;
    let mut result = Vec::with_capacity(result_bindings.len());

    for binding in result_bindings {
        let value = dict
            .get_item(&*binding.name)?
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "missing transition result field '{}'",
                    binding.name
                ))
            })?
            .unbind();
        result.push((binding.source.clone(), value));
    }

    Ok(result)
}

/// Build a child scope, execute the target subtree, run hooks, freeze the
/// child scope, and return the result.
fn execute_child_context(
    py: Python<'_>,
    graph: &Arc<ExecutionGraph<SlotBackend>>,
    parent_scope: &Arc<Scope<SlotBackend>>,
    target: ExecutionNodeId,
    params: &[MethodParam<SlotBackend>],
    hooks: &[ExecutionHook],
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    kind: &TransitionKind,
    method_result: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let mut new_sources = extract_param_sources(py, params, args, kwargs)?;

    // For Method transitions, add the implementation's return value and any
    // projected TypedDict field bindings to the child scope.
    if let (
        TransitionKind::Method {
            result_source,
            result_bindings,
            ..
        },
        Some(result_val),
    ) = (kind, method_result)
    {
        if let Some(source) = result_source {
            new_sources.push((source.clone(), result_val.clone_ref(py)));
        }

        let mut existing_sources = std::collections::HashSet::with_capacity(new_sources.len());
        for (source, _) in &new_sources {
            existing_sources.insert(source.clone());
        }

        for (source, value) in extract_result_bindings(result_bindings, result_val.bind(py))? {
            if existing_sources.insert(source.clone()) {
                new_sources.push((source, value));
            }
        }
    }

    let child_scope = Scope::child(Arc::clone(parent_scope), new_sources);
    let child_data = ContextData {
        graph: Arc::clone(graph),
        root_node: target,
    };
    let (result, scope_handle) = execute(py, &child_data, child_scope, hooks)?;
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
    let parent = get_parent_scope(&cep.parent_scope)?;
    execute_child_context(
        py,
        &cep.graph,
        &parent,
        cep.target,
        &cep.params,
        &cep.hooks,
        cep.args.bind(py),
        cep.kwargs.as_ref().map(|k| k.bind(py)),
        &cep.kind,
        method_result,
    )
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
        match self.return_wrapper {
            WrapperKind::None => self.call_sync(py, args, kwargs),
            WrapperKind::Cm => self.call_cm(py, args, kwargs),
            WrapperKind::Awaitable => self.call_awaitable(py, args, kwargs),
            WrapperKind::Acm => self.call_acm(py, args, kwargs),
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
        execute_child_context(
            py,
            &self.shared.graph,
            &parent,
            self.shared.target,
            &self.shared.params,
            &self.shared.hooks,
            args,
            kwargs,
            &self.kind,
            method_result,
        )
    }

    fn call_cm(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let wrapper = CmWrapper {
            shared: TransitionShared {
                graph: Arc::clone(&self.shared.graph),
                parent_scope: self.shared.parent_scope.clone(),
                target: self.shared.target,
                params: self.shared.params.clone(),
                hooks: self.shared.hooks.clone(),
            },
            kind: clone_kind(&self.kind, py),
            args: args.clone().unbind(),
            kwargs: kwargs.map(|k| k.clone().unbind()),
            cm: OnceLock::new(),
        };
        Ok(Py::new(py, wrapper)?.into_any())
    }

    fn call_awaitable(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
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
        let child_exec = ChildExecutionParams {
            graph: Arc::clone(&self.shared.graph),
            parent_scope: self.shared.parent_scope.clone(),
            target: self.shared.target,
            kind: clone_kind(&self.kind, py),
            params: self.shared.params.clone(),
            hooks: self.shared.hooks.clone(),
            args: args.clone().unbind(),
            kwargs: kwargs.map(|k| k.clone().unbind()),
        };
        let wrapper = AwaitableWrapper::new(inner_coro, Some(child_exec));
        Ok(Py::new(py, wrapper)?.into_any())
    }

    fn call_acm(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let wrapper = AcmWrapper {
            shared: TransitionShared {
                graph: Arc::clone(&self.shared.graph),
                parent_scope: self.shared.parent_scope.clone(),
                target: self.shared.target,
                params: self.shared.params.clone(),
                hooks: self.shared.hooks.clone(),
            },
            kind: clone_kind(&self.kind, py),
            args: args.clone().unbind(),
            kwargs: kwargs.map(|k| k.clone().unbind()),
            acm: OnceLock::new(),
        };
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
            result_source: result_source.clone(),
            result_bindings: result_bindings.clone(),
        },
        TransitionKind::Auto => TransitionKind::Auto,
    }
}

// ---------------------------------------------------------------------------
// CmWrapper — context manager for WrapperKind::Cm
// ---------------------------------------------------------------------------

#[pyclass(frozen, module = "inlay")]
pub(crate) struct CmWrapper {
    shared: TransitionShared,
    kind: TransitionKind,
    args: Py<PyTuple>,
    kwargs: Option<Py<PyDict>>,
    /// The underlying context manager returned by the implementation.
    /// Set in `__enter__`, read in `__exit__`. `None` for Auto transitions.
    cm: OnceLock<Py<PyAny>>,
}

#[pymethods]
impl CmWrapper {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.shared.traverse(&visit)?;
        self.kind.traverse(&visit)?;
        visit.call(&self.args)?;
        if let Some(kwargs) = &self.kwargs {
            visit.call(kwargs)?;
        }
        if let Some(cm) = self.cm.get() {
            visit.call(cm)?;
        }
        Ok(())
    }

    fn __enter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let parent = get_parent_scope(&self.shared.parent_scope)?;
        let method_result = match &self.kind {
            TransitionKind::Method {
                implementation,
                bound_instance,
                ..
            } => {
                let cm = call_implementation(
                    py,
                    implementation,
                    bound_instance,
                    self.args.bind(py),
                    self.kwargs.as_ref().map(|k| k.bind(py)),
                )?;
                let result = cm.call_method0(py, "__enter__")?;
                self.cm
                    .set(cm)
                    .map_err(|_| PyRuntimeError::new_err("__enter__ called twice"))?;
                Some(result)
            }
            TransitionKind::Auto => None,
        };
        execute_child_context(
            py,
            &self.shared.graph,
            &parent,
            self.shared.target,
            &self.shared.params,
            &self.shared.hooks,
            self.args.bind(py),
            self.kwargs.as_ref().map(|k| k.bind(py)),
            &self.kind,
            method_result,
        )
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        exc_type: Py<PyAny>,
        exc_val: Py<PyAny>,
        exc_tb: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        match self.cm.get() {
            Some(cm) => cm.call_method1(py, "__exit__", (exc_type, exc_val, exc_tb)),
            None => Ok(py.None()),
        }
    }
}

// ---------------------------------------------------------------------------
// AwaitableWrapper — coroutine protocol for WrapperKind::Awaitable
//
// Also reused by AcmWrapper for __aenter__ / __aexit__ awaitables.
//
// State machine:
//   Driving(inner_coro)  — forwarding send/throw to the inner coroutine
//   Immediate            — no coroutine; execute child on first send
//   Done                 — terminal
// ---------------------------------------------------------------------------

enum AwaitableState {
    Driving(Py<PyAny>),
    Immediate,
    Done,
}

#[pyclass(frozen, module = "inlay")]
pub(crate) struct AwaitableWrapper {
    state: Mutex<AwaitableState>,
    /// `Some` → execute child context on completion.
    /// `None` → just pass through the coroutine result (used by ACM __aexit__).
    child_execution: Option<ChildExecutionParams>,
}

impl AwaitableWrapper {
    fn new(inner_coro: Option<Py<PyAny>>, child_execution: Option<ChildExecutionParams>) -> Self {
        let state = match inner_coro {
            Some(coro) => AwaitableState::Driving(coro),
            None => AwaitableState::Immediate,
        };
        Self {
            state: Mutex::new(state),
            child_execution,
        }
    }

    /// Handle completion of the inner coroutine (or immediate mode).
    fn on_complete(&self, py: Python<'_>, coro_result: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        match &self.child_execution {
            Some(params) => {
                let child_result = execute_child_from_params(py, params, coro_result)?;
                Err(PyStopIteration::new_err(child_result))
            }
            None => {
                let val = coro_result.unwrap_or_else(|| py.None());
                Err(PyStopIteration::new_err(val))
            }
        }
    }
}

#[pymethods]
impl AwaitableWrapper {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        // GC always runs with the GIL held so the mutex should be available,
        // but use try_lock defensively.
        if let Ok(state) = self.state.try_lock() {
            if let AwaitableState::Driving(coro) = &*state {
                visit.call(coro)?;
            }
        }
        if let Some(params) = &self.child_execution {
            params.traverse(&visit)?;
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
        let mut state = self.state.lock().expect("poisoned");
        match &*state {
            AwaitableState::Driving(coro) => match coro.call_method1(py, "send", (&value,)) {
                Ok(yielded) => Ok(yielded),
                Err(e) => {
                    if e.is_instance_of::<PyStopIteration>(py) {
                        let result = stop_iteration_value(py, &e);
                        *state = AwaitableState::Done;
                        self.on_complete(py, Some(result))
                    } else {
                        *state = AwaitableState::Done;
                        Err(e)
                    }
                }
            },
            AwaitableState::Immediate => {
                *state = AwaitableState::Done;
                self.on_complete(py, None)
            }
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
        let mut state = self.state.lock().expect("poisoned");
        match &*state {
            AwaitableState::Driving(coro) => {
                let none = py.None();
                let val_ref = val.as_ref().unwrap_or(&none);
                let tb_ref = tb.as_ref().unwrap_or(&none);
                match coro.call_method1(py, "throw", (&typ, val_ref, tb_ref)) {
                    Ok(yielded) => Ok(yielded),
                    Err(e) => {
                        *state = AwaitableState::Done;
                        Err(e)
                    }
                }
            }
            _ => {
                *state = AwaitableState::Done;
                // Re-raise: instantiate the exception and raise it
                Err(PyErr::from_value(typ.into_bound(py)))
            }
        }
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let mut state = self.state.lock().expect("poisoned");
        match &*state {
            AwaitableState::Driving(coro) => {
                let _ = coro.call_method0(py, "close");
                *state = AwaitableState::Done;
                Ok(())
            }
            _ => {
                *state = AwaitableState::Done;
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AcmWrapper — async context manager for WrapperKind::Acm
// ---------------------------------------------------------------------------

#[pyclass(frozen, module = "inlay")]
pub(crate) struct AcmWrapper {
    shared: TransitionShared,
    kind: TransitionKind,
    args: Py<PyTuple>,
    kwargs: Option<Py<PyDict>>,
    /// The underlying async context manager returned by the implementation.
    /// Set in `__aenter__`, read in `__aexit__`. `None` for Auto transitions.
    acm: OnceLock<Py<PyAny>>,
}

#[pymethods]
impl AcmWrapper {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.shared.traverse(&visit)?;
        self.kind.traverse(&visit)?;
        visit.call(&self.args)?;
        if let Some(kwargs) = &self.kwargs {
            visit.call(kwargs)?;
        }
        if let Some(acm) = self.acm.get() {
            visit.call(acm)?;
        }
        Ok(())
    }

    fn __aenter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // For Method: call implementation to get the ACM, then start __aenter__.
        // For Auto: no coroutine, child execution happens immediately.
        let inner_coro = match &self.kind {
            TransitionKind::Method {
                implementation,
                bound_instance,
                ..
            } => {
                let acm = call_implementation(
                    py,
                    implementation,
                    bound_instance,
                    self.args.bind(py),
                    self.kwargs.as_ref().map(|k| k.bind(py)),
                )?;
                let enter_coro = acm.call_method0(py, "__aenter__")?;
                self.acm
                    .set(acm)
                    .map_err(|_| PyRuntimeError::new_err("__aenter__ called twice"))?;
                Some(enter_coro)
            }
            TransitionKind::Auto => None,
        };

        let child_exec = ChildExecutionParams {
            graph: Arc::clone(&self.shared.graph),
            parent_scope: self.shared.parent_scope.clone(),
            target: self.shared.target,
            kind: clone_kind(&self.kind, py),
            params: self.shared.params.clone(),
            hooks: self.shared.hooks.clone(),
            args: self.args.clone_ref(py),
            kwargs: self.kwargs.as_ref().map(|k| k.clone_ref(py)),
        };

        let wrapper = AwaitableWrapper::new(inner_coro, Some(child_exec));
        Ok(Py::new(py, wrapper)?.into_any())
    }

    fn __aexit__(
        &self,
        py: Python<'_>,
        exc_type: Py<PyAny>,
        exc_val: Py<PyAny>,
        exc_tb: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let inner_coro = match self.acm.get() {
            Some(acm) => Some(acm.call_method1(py, "__aexit__", (&exc_type, &exc_val, &exc_tb))?),
            None => None,
        };
        // No child execution for __aexit__ — just drive the coroutine and return.
        let wrapper = AwaitableWrapper::new(inner_coro, None);
        Ok(Py::new(py, wrapper)?.into_any())
    }
}
