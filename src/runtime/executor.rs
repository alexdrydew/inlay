use std::collections::HashMap;
use std::sync::{Arc, OnceLock, Weak};

use inlay_instrument_macros::instrumented;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::compile::flatten::{
    ConstructorParam, ExecutionCacheMode, ExecutionGraph, ExecutionHook, ExecutionNode,
    ExecutionNodeId,
};
use crate::types::ParamKind;

use super::lazy_ref::LazyRefImpl;
use super::proxy::{ContextProxy, DelegatedAttr, DelegatedDict, unwrap_delegated};
use super::scope::Scope;
use super::transition::{Transition, TransitionKind, TransitionShared};

pub(crate) type ScopeHandle = Arc<OnceLock<Arc<Scope>>>;
pub(crate) type WeakScopeHandle = Weak<OnceLock<Arc<Scope>>>;

pub(crate) struct ContextData {
    pub(crate) graph: Arc<ExecutionGraph>,
    pub(crate) root_node: ExecutionNodeId,
}

struct ExecutionState {
    scope: Scope,
    lazy_cells: Vec<(Py<LazyRefImpl>, ExecutionNodeId)>,
    scope_handle: ScopeHandle,
}

/// Execute the root node, run hooks, bind lazy refs, then freeze the scope
/// into the handle so transitions created during execution can access the
/// frozen parent scope.
#[instrumented(name = "inlay.execute", target = "inlay", level = "trace", skip_all)]
pub(crate) fn execute(
    py: Python<'_>,
    data: &ContextData,
    scope: Scope,
    hooks: &[ExecutionHook],
) -> PyResult<(Py<PyAny>, ScopeHandle)> {
    let mut state = ExecutionState {
        scope,
        lazy_cells: Vec::new(),
        scope_handle: Arc::new(OnceLock::new()),
    };

    let result = execute_node(py, data, &mut state, data.root_node)?;

    // Execute hooks (scope still mutable — hooks can materialize nodes)
    execute_hooks(py, data, &mut state, hooks)?;

    // Bind all lazy refs
    for (cell, target_id) in std::mem::take(&mut state.lazy_cells) {
        let val = execute_node(py, data, &mut state, target_id)?;
        cell.get().bind_value(val);
    }

    // Freeze scope and publish to transitions created during this execution.
    let frozen = Arc::new(std::mem::replace(
        &mut state.scope,
        Scope::root(HashMap::new()),
    ));
    if state.scope_handle.set(frozen).is_err() {
        panic!("scope handle already set — execute called twice?");
    }

    Ok((result, Arc::clone(&state.scope_handle)))
}

fn execute_hooks(
    py: Python<'_>,
    data: &ContextData,
    state: &mut ExecutionState,
    hooks: &[ExecutionHook],
) -> PyResult<()> {
    for hook in hooks {
        let mut values: Vec<Py<PyAny>> = Vec::with_capacity(hook.params.len());
        for p in &hook.params {
            let val = execute_node(py, data, state, p.node)?;
            values.push(unwrap_delegated(val.bind(py))?.unbind());
        }
        let param_refs: Vec<&ConstructorParam> = hook.params.iter().collect();
        let (args, kwargs) = build_call_args(py, &values, &param_refs)?;
        hook.implementation.call(py, args, kwargs.as_ref())?;
    }
    Ok(())
}

fn execute_node(
    py: Python<'_>,
    data: &ContextData,
    state: &mut ExecutionState,
    node_id: ExecutionNodeId,
) -> PyResult<Py<PyAny>> {
    let entry = &data.graph[node_id];

    // Cache check (only constructor results are cached in the execution scope)
    if entry.node.cache_mode() == ExecutionCacheMode::Computed {
        if let Some(cached) = state
            .scope
            .get_computed(&entry.cache_key, &entry.source_deps)
        {
            return Ok(cached.clone_ref(py));
        }
    }

    let result = dispatch_node(py, data, state, node_id)?;

    // Cache the result (only constructor results are cached)
    if data.graph[node_id].node.cache_mode() == ExecutionCacheMode::Computed {
        let cache_key = data.graph[node_id].cache_key.clone();
        state.scope.insert_computed(cache_key, result.clone_ref(py));
    }

    Ok(result)
}

fn dispatch_node(
    py: Python<'_>,
    data: &ContextData,
    state: &mut ExecutionState,
    node_id: ExecutionNodeId,
) -> PyResult<Py<PyAny>> {
    let entry = &data.graph[node_id];
    match &entry.node {
        ExecutionNode::None => Ok(py.None()),

        ExecutionNode::Constant(source) => state
            .scope
            .get_source(source)
            .map(|v| v.clone_ref(py))
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("source value not found in scope")
            }),

        ExecutionNode::Constructor {
            implementation,
            params,
        } => {
            let impl_ref = Arc::clone(implementation);
            let params_snapshot: Vec<&ConstructorParam> = params.iter().collect();
            let mut values: Vec<Py<PyAny>> = Vec::with_capacity(params_snapshot.len());
            for p in &params_snapshot {
                let val = execute_node(py, data, state, p.node)?;
                values.push(unwrap_delegated(val.bind(py))?.unbind());
            }
            let (args, kwargs) = build_call_args(py, &values, &params_snapshot)?;
            impl_ref.call(py, args, kwargs.as_ref())
        }

        ExecutionNode::Property {
            source,
            property_name,
        } => {
            let source_id = *source;
            let name = property_name.clone();
            let source_obj = execute_node(py, data, state, source_id)?;
            let bound = source_obj.bind(py);
            if let Ok(dict) = bound.cast::<PyDict>() {
                dict.get_item(&*name)?
                    .map(|v| v.unbind())
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(name.to_string()))
            } else {
                bound.getattr(&*name).map(|v| v.unbind())
            }
        }

        ExecutionNode::Attribute {
            source,
            attribute_name,
        } => {
            let source_id = *source;
            let name = attribute_name.clone();
            let source_obj = execute_node(py, data, state, source_id)?;
            let da = DelegatedAttr {
                source: source_obj,
                name,
            };
            Ok(Py::new(py, da)?.into_any())
        }

        ExecutionNode::Protocol { members } => {
            let member_entries: Vec<(Arc<str>, ExecutionNodeId)> =
                members.iter().map(|(k, &v)| (k.clone(), v)).collect();
            let mut resolved: HashMap<Arc<str>, Py<PyAny>> = HashMap::new();
            let mut writable = std::collections::HashSet::new();
            for (name, mid) in &member_entries {
                let val = execute_node(py, data, state, *mid)?;
                if val.bind(py).is_instance_of::<DelegatedAttr>() {
                    writable.insert(name.clone());
                }
                resolved.insert(name.clone(), val);
            }
            let proxy = ContextProxy::new(resolved, writable);
            Ok(Py::new(py, proxy)?.into_any())
        }

        ExecutionNode::TypedDict { members } => {
            let member_entries: Vec<(Arc<str>, ExecutionNodeId)> =
                members.iter().map(|(k, &v)| (k.clone(), v)).collect();
            let mut delegates: HashMap<Arc<str>, Py<DelegatedAttr>> = HashMap::new();
            for (name, mid) in &member_entries {
                let val = execute_node(py, data, state, *mid)?;
                let da = val.bind(py).cast::<DelegatedAttr>().map_err(|_| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "TypedDict member '{name}' did not resolve to DelegatedAttr"
                    ))
                })?;
                delegates.insert(name.clone(), da.clone().unbind());
            }
            let dict = DelegatedDict::new(delegates);
            Ok(Py::new(py, dict)?.into_any())
        }

        ExecutionNode::LazyRef { target } => {
            let cell = LazyRefImpl::new(String::new());
            let py_cell = Py::new(py, cell)?;
            state.lazy_cells.push((py_cell.clone_ref(py), *target));
            Ok(py_cell.into_any())
        }

        ExecutionNode::Method {
            implementation,
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            bound_to,
            params,
            result_source,
            result_bindings,
            target,
            hooks,
        } => {
            // Eagerly resolve bound instance from parent scope while it's still mutable.
            let bound_instance = match bound_to {
                Some(id) => Some(execute_node(py, data, state, *id)?),
                None => None,
            };
            let kind = TransitionKind::Method {
                implementation: implementation.clone_ref(py),
                bound_instance,
                result_source: *result_source,
                result_bindings: result_bindings.clone(),
            };
            let shared = TransitionShared {
                graph: Arc::clone(&data.graph),
                parent_scope: Arc::downgrade(&state.scope_handle),
                target: *target,
                params: params.clone(),
                accepts_varargs: *accepts_varargs,
                accepts_varkw: *accepts_varkw,
                hooks: hooks.clone(),
            };
            let transition = Transition::new(shared, kind, *return_wrapper);
            Ok(Py::new(py, transition)?.into_any())
        }

        ExecutionNode::AutoMethod {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            target,
            hooks,
        } => {
            let shared = TransitionShared {
                graph: Arc::clone(&data.graph),
                parent_scope: Arc::downgrade(&state.scope_handle),
                target: *target,
                params: params.clone(),
                accepts_varargs: *accepts_varargs,
                accepts_varkw: *accepts_varkw,
                hooks: hooks.clone(),
            };
            let transition = Transition::new(shared, TransitionKind::Auto, *return_wrapper);
            Ok(Py::new(py, transition)?.into_any())
        }
    }
}

/// Attach scope ownership to the result.
///
/// - `ContextProxy` — the scope handle is stored directly on it.
/// - `Transition` — wrapped in a [`ScopeOwningCallable`] that keeps the scope
///   alive and delegates `__call__`.
/// - Everything else (plain constructor results, etc.) — returned as-is; the
///   scope is not needed because these objects don't hold weak scope refs.
#[instrumented(
    name = "inlay.attach_scope",
    target = "inlay",
    level = "trace",
    skip_all
)]
pub(crate) fn attach_scope(
    py: Python<'_>,
    result: Py<PyAny>,
    scope_handle: ScopeHandle,
) -> PyResult<Py<PyAny>> {
    let bound = result.bind(py);
    if let Ok(proxy) = bound.cast::<ContextProxy>() {
        proxy.borrow_mut().set_scope_owner(scope_handle);
        Ok(result)
    } else if bound.is_instance_of::<Transition>() {
        Ok(Py::new(
            py,
            ScopeOwningCallable {
                inner: result,
                _scope: scope_handle,
            },
        )?
        .into_any())
    } else {
        Ok(result)
    }
}

/// Thin wrapper that keeps a scope alive while delegating calls to an inner
/// object (used when the root execution result is a Transition, not a
/// ContextProxy).
#[pyclass(frozen, weakref, module = "inlay")]
struct ScopeOwningCallable {
    inner: Py<PyAny>,
    _scope: ScopeHandle,
}

#[pymethods]
impl ScopeOwningCallable {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        self.inner.call(py, args, kwargs)
    }

    fn __traverse__(&self, visit: pyo3::gc::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        visit.call(&self.inner)?;
        if let Some(scope) = self._scope.get() {
            scope.traverse_py_refs(&visit)?;
        }
        Ok(())
    }
}

/// Split resolved values into positional args + keyword args based on param kinds.
///
/// - `PositionalOnly` and `PositionalOrKeyword` → positional tuple
/// - `KeywordOnly` → keyword dict
fn build_call_args<'py>(
    py: Python<'py>,
    values: &[Py<PyAny>],
    params: &[&ConstructorParam],
) -> PyResult<(Bound<'py, pyo3::types::PyTuple>, Option<Bound<'py, PyDict>>)> {
    let mut positional: Vec<&Py<PyAny>> = Vec::new();
    let mut keyword: Vec<(&str, &Py<PyAny>)> = Vec::new();

    for (val, param) in values.iter().zip(params.iter()) {
        match param.kind {
            ParamKind::PositionalOnly | ParamKind::PositionalOrKeyword => {
                positional.push(val);
            }
            ParamKind::KeywordOnly => {
                keyword.push((&param.name, val));
            }
        }
    }

    let args = pyo3::types::PyTuple::new(py, positional)?;
    let kwargs = if keyword.is_empty() {
        None
    } else {
        let dict = PyDict::new(py);
        for (name, val) in keyword {
            dict.set_item(name, val)?;
        }
        Some(dict)
    };

    Ok((args, kwargs))
}
