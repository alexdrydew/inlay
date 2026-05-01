use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use inlay_instrument::instrumented;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::compile::flatten::{
    ConstructorParam, ExecutionGraph, ExecutionHook, ExecutionNode, ExecutionNodeId,
    ExecutionSourceNodeId, hook_roots, resource_plan_for_node, resource_plan_for_roots,
    transition_introduced_sources,
};
use crate::types::ParamKind;

use super::lazy_ref::LazyRefImpl;
use super::proxy::{ContextProxy, DelegatedDict, DelegatedMember, unwrap_delegated};
use super::resources::RuntimeResources;
use super::transition::{Transition, TransitionKind, TransitionShared};

pub(crate) struct ContextData {
    pub(crate) graph: Arc<ExecutionGraph>,
    pub(crate) root_node: ExecutionNodeId,
}

struct ExecutionState {
    resources: RuntimeResources,
    lazy_cells: Vec<(Py<LazyRefImpl>, ExecutionNodeId)>,
    capture_root_transition: bool,
}

#[instrumented(name = "inlay.execute", target = "inlay", level = "trace", skip_all)]
pub(crate) fn execute(
    py: Python<'_>,
    data: &ContextData,
    resources: RuntimeResources,
    hooks: &[ExecutionHook],
    capture_root_transition: bool,
) -> PyResult<Py<PyAny>> {
    let mut state = ExecutionState {
        resources,
        lazy_cells: Vec::new(),
        capture_root_transition,
    };

    state.resources.ensure_caches(&resource_plan_for_node(
        &data.graph,
        data.root_node,
        &HashSet::new(),
    ));

    let result = execute_node(py, data, &mut state, data.root_node)?;
    bind_lazy_refs(py, data, &mut state)?;

    for hook in hooks {
        let values = execute_constructor_params(py, data, &mut state, &hook.params)?;
        bind_lazy_refs(py, data, &mut state)?;
        let (args, kwargs) = build_call_args(py, &values, &hook.params)?;
        hook.implementation.call(py, args, kwargs.as_ref())?;
    }

    Ok(result)
}

fn bind_lazy_refs(py: Python<'_>, data: &ContextData, state: &mut ExecutionState) -> PyResult<()> {
    // Binding a lazy target can create more lazy refs.
    while let Some((cell, target_id)) = state.lazy_cells.pop() {
        let val = execute_node(py, data, state, target_id)?;
        cell.get().bind_value(val);
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
    if matches!(&entry.node, ExecutionNode::Constructor { .. }) {
        let cache = state.resources.get_or_create_cache(node_id);
        if let Some(cached) = cache.get() {
            return Ok(cached.clone_ref(py));
        }

        let result = dispatch_node(py, data, state, node_id)?;
        if cache.set(result.clone_ref(py)).is_err()
            && let Some(cached) = cache.get()
        {
            return Ok(cached.clone_ref(py));
        }
        return Ok(result);
    }

    dispatch_node(py, data, state, node_id)
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
            .resources
            .get_source(py, ExecutionSourceNodeId(node_id)),

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
            let member_entries: HashMap<Arc<str>, ExecutionNodeId> = members
                .iter()
                .map(|(name, &node)| (name.clone(), node))
                .collect();
            let mut writable = std::collections::HashSet::new();
            for (name, node_id) in &member_entries {
                if matches!(&data.graph[*node_id].node, ExecutionNode::Attribute { .. }) {
                    writable.insert(name.clone());
                }
            }
            let plan = resource_plan_for_node(&data.graph, node_id, &HashSet::new());
            state.resources.ensure_caches(&plan);
            let proxy = ContextProxy::new(
                Arc::clone(&data.graph),
                member_entries,
                writable,
                state.resources.capture_plan(py, &plan)?,
            );
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
            let resources = if state.capture_root_transition || node_id != data.root_node {
                let introduced =
                    transition_introduced_sources(params, Some(*result_source), result_bindings);
                let plan = resource_plan_for_roots(
                    &data.graph,
                    std::iter::once(*target).chain(hook_roots(hooks)),
                    &introduced,
                );
                state.resources.ensure_caches(&plan);
                state.resources.capture_plan(py, &plan)?
            } else {
                RuntimeResources::empty()
            };
            let shared = TransitionShared {
                graph: Arc::clone(&data.graph),
                resources,
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
            let resources = if state.capture_root_transition || node_id != data.root_node {
                let introduced = transition_introduced_sources(params, None, &[]);
                let plan = resource_plan_for_roots(
                    &data.graph,
                    std::iter::once(*target).chain(hook_roots(hooks)),
                    &introduced,
                );
                state.resources.ensure_caches(&plan);
                state.resources.capture_plan(py, &plan)?
            } else {
                RuntimeResources::empty()
            };
            let shared = TransitionShared {
                graph: Arc::clone(&data.graph),
                resources,
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
