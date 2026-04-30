use std::collections::HashMap;
use std::sync::{Arc, OnceLock, Weak};

use inlay_instrument::instrumented;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::compile::flatten::{
    ConstructorParam, ExecutionGraph, ExecutionHook, ExecutionNode, ExecutionNodeId,
};
use crate::types::ParamKind;

use super::lazy_ref::LazyRefImpl;
use super::proxy::{ContextProxy, DelegatedDict, DelegatedMember, unwrap_delegated};
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

    execute_hooks(py, data, &mut state, hooks)?;

    // Binding a lazy target can create more lazy refs.
    while let Some((cell, target_id)) = state.lazy_cells.pop() {
        let val = execute_node(py, data, &mut state, target_id)?;
        cell.get().bind_value(val);
    }

    // Transitions created above hold weak refs until the scope is frozen here.
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
        let values = execute_constructor_params(py, data, state, &hook.params)?;
        let (args, kwargs) = build_call_args(py, &values, &hook.params)?;
        hook.implementation.call(py, args, kwargs.as_ref())?;
    }
    Ok(())
}

fn execute_constructor_params(
    py: Python<'_>,
    data: &ContextData,
    state: &mut ExecutionState,
    params: &[ConstructorParam],
) -> PyResult<Vec<Py<PyAny>>> {
    let mut values: Vec<Py<PyAny>> = Vec::with_capacity(params.len());
    for param in params {
        let val = execute_node(py, data, state, param.node)?;
        values.push(unwrap_delegated(val.bind(py))?.unbind());
    }
    Ok(values)
}

fn execute_node(
    py: Python<'_>,
    data: &ContextData,
    state: &mut ExecutionState,
    node_id: ExecutionNodeId,
) -> PyResult<Py<PyAny>> {
    let entry = &data.graph[node_id];
    let is_cached = matches!(&entry.node, ExecutionNode::Constructor { .. });

    if is_cached {
        if let Some(cached) = state.scope.get_cached(node_id, &entry.source_deps) {
            return Ok(cached.clone_ref(py));
        }
    }

    let result = dispatch_node(py, data, state, node_id)?;

    if is_cached {
        state.scope.insert_cached(node_id, result.clone_ref(py));
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

        ExecutionNode::Constant => state
            .scope
            .get_value(node_id)
            .map(|v| v.clone_ref(py))
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("source value not found in scope")
            }),

        ExecutionNode::Constructor {
            implementation,
            params,
        } => {
            let impl_ref = Arc::clone(implementation);
            let values = execute_constructor_params(py, data, state, params)?;
            let (args, kwargs) = build_call_args(py, &values, params)?;
            impl_ref.call(py, args, kwargs.as_ref())
        }

        ExecutionNode::Property {
            source,
            property_name,
        } => {
            let source_id = *source;
            let name = property_name.clone();
            let source_obj = execute_node(py, data, state, source_id)?;
            source_obj
                .bind(py)
                .getattr(name.as_ref())
                .map(|v| v.unbind())
        }

        ExecutionNode::Attribute {
            source,
            attribute_name,
            access_kind,
        } => {
            let source_id = *source;
            let name = attribute_name.clone();
            let source_obj = execute_node(py, data, state, source_id)?;
            let member = DelegatedMember {
                source: source_obj,
                name,
                access_kind: *access_kind,
            };
            Ok(Py::new(py, member)?.into_any())
        }

        ExecutionNode::Protocol { members } => {
            let member_entries: Vec<(Arc<str>, ExecutionNodeId)> =
                members.iter().map(|(k, &v)| (k.clone(), v)).collect();
            let mut resolved: HashMap<Arc<str>, Py<PyAny>> = HashMap::new();
            let mut writable = std::collections::HashSet::new();
            for (name, mid) in &member_entries {
                let val = execute_node(py, data, state, *mid)?;
                if val.bind(py).is_instance_of::<DelegatedMember>() {
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
            let mut delegates: HashMap<Arc<str>, Py<DelegatedMember>> = HashMap::new();
            for (name, mid) in &member_entries {
                let val = execute_node(py, data, state, *mid)?;
                let member = val.bind(py).cast::<DelegatedMember>().map_err(|_| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "TypedDict member '{name}' did not resolve to DelegatedMember"
                    ))
                })?;
                delegates.insert(name.clone(), member.clone().unbind());
            }
            let dict = DelegatedDict::new(delegates);
            Ok(Py::new(py, dict)?.into_any())
        }

        ExecutionNode::LazyRef { target } => {
            let cell = LazyRefImpl::new();
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
            // Method transitions may outlive this mutable execution scope.
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
        // Root transitions have no ContextProxy to own their frozen scope.
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
        if let Ok(transition) = self.inner.bind(py).cast::<Transition>() {
            return transition.borrow().call_with_scope_owner(
                py,
                args,
                kwargs,
                Arc::clone(&self._scope),
            );
        }
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

fn build_call_args<'py>(
    py: Python<'py>,
    values: &[Py<PyAny>],
    params: &[ConstructorParam],
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
