use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use inlay_instrument::instrumented;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::compile::flatten::{
    ConstructorParam, ExecutionGraph, ExecutionMethodImplementation, ExecutionNode,
    ExecutionNodeId, ExecutionSourceNodeId, resource_plan_for_node, resource_plan_for_roots,
    transition_body_roots, transition_introduced_sources,
};
use crate::types::{ParamKind, WrapperKind};

use super::lazy_ref::LazyRefImpl;
use super::proxy::{ContextProxy, DelegatedDict, DelegatedMember, unwrap_delegated};
use super::resources::RuntimeResources;
use super::transition::{Transition, TransitionShared};

pub(crate) struct ContextData {
    pub(crate) graph: Arc<ExecutionGraph>,
    pub(crate) root_node: ExecutionNodeId,
}

pub(crate) struct AsyncContextImplementationStart {
    pub(crate) resources: RuntimeResources,
    pub(crate) result_source: Option<ExecutionSourceNodeId>,
    pub(crate) context: Py<PyAny>,
    pub(crate) enter_coro: Py<PyAny>,
}

pub(crate) struct AwaitableImplementationStart {
    pub(crate) resources: RuntimeResources,
    pub(crate) result_source: Option<ExecutionSourceNodeId>,
    pub(crate) awaitable: Py<PyAny>,
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

    Ok(result)
}

pub(crate) fn execute_transition(
    py: Python<'_>,
    data: &ContextData,
    resources: RuntimeResources,
    implementations: &[ExecutionMethodImplementation],
) -> PyResult<Py<PyAny>> {
    execute_transition_from(py, data, resources, implementations, Vec::new())
}

pub(crate) fn execute_transition_from(
    py: Python<'_>,
    data: &ContextData,
    resources: RuntimeResources,
    implementations: &[ExecutionMethodImplementation],
    preset_sources: Vec<(ExecutionSourceNodeId, Py<PyAny>)>,
) -> PyResult<Py<PyAny>> {
    execute_transition_inner(py, data, resources, implementations, preset_sources, None)
}

pub(crate) fn execute_transition_with_sync_contexts(
    py: Python<'_>,
    data: &ContextData,
    resources: RuntimeResources,
    implementations: &[ExecutionMethodImplementation],
) -> PyResult<(Py<PyAny>, Vec<Py<PyAny>>)> {
    let mut contexts = Vec::new();
    match execute_transition_inner(
        py,
        data,
        resources,
        implementations,
        Vec::new(),
        Some(&mut contexts),
    ) {
        Ok(result) => Ok((result, contexts)),
        Err(error) => Err(cleanup_sync_contexts_after_error(py, contexts, error)),
    }
}

pub(crate) fn start_async_context_implementation(
    py: Python<'_>,
    data: &ContextData,
    resources: RuntimeResources,
    implementation: &ExecutionMethodImplementation,
) -> PyResult<AsyncContextImplementationStart> {
    if implementation.return_wrapper != WrapperKind::AsyncContextManager {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "method implementation is not an async context manager",
        ));
    }

    let mut state = ExecutionState {
        resources,
        lazy_cells: Vec::new(),
        capture_root_transition: true,
    };

    let context = execute_method_implementation(py, data, &mut state, implementation)?;
    let enter_coro = context.call_method0(py, "__aenter__")?;
    Ok(AsyncContextImplementationStart {
        resources: state.resources,
        result_source: implementation.result_source,
        context,
        enter_coro,
    })
}

pub(crate) fn start_awaitable_implementation(
    py: Python<'_>,
    data: &ContextData,
    resources: RuntimeResources,
    implementation: &ExecutionMethodImplementation,
) -> PyResult<AwaitableImplementationStart> {
    if implementation.return_wrapper != WrapperKind::Awaitable {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "method implementation is not awaitable",
        ));
    }

    let mut state = ExecutionState {
        resources,
        lazy_cells: Vec::new(),
        capture_root_transition: true,
    };

    let awaitable = execute_method_implementation(py, data, &mut state, implementation)?;
    Ok(AwaitableImplementationStart {
        resources: state.resources,
        result_source: implementation.result_source,
        awaitable,
    })
}

fn execute_transition_inner(
    py: Python<'_>,
    data: &ContextData,
    resources: RuntimeResources,
    implementations: &[ExecutionMethodImplementation],
    preset_sources: Vec<(ExecutionSourceNodeId, Py<PyAny>)>,
    mut sync_contexts: Option<&mut Vec<Py<PyAny>>>,
) -> PyResult<Py<PyAny>> {
    let mut state = ExecutionState {
        resources,
        lazy_cells: Vec::new(),
        capture_root_transition: true,
    };

    for (source, value) in preset_sources {
        state.resources.insert_source(source, value);
    }

    for implementation in implementations {
        let result = execute_method_implementation(py, data, &mut state, implementation)?;
        let result = unwrap_implementation_result(py, result, implementation, &mut sync_contexts)?;
        if let Some(source) = implementation.result_source {
            state.resources.insert_source(source, result.clone_ref(py));
        }
    }

    let result = execute_node(py, data, &mut state, data.root_node)?;
    bind_lazy_refs(py, data, &mut state)?;
    Ok(result)
}

fn execute_method_implementation(
    py: Python<'_>,
    data: &ContextData,
    state: &mut ExecutionState,
    implementation: &ExecutionMethodImplementation,
) -> PyResult<Py<PyAny>> {
    let impl_ref = Arc::clone(&implementation.implementation);
    let values = execute_constructor_params(py, data, state, &implementation.params)?;
    bind_lazy_refs(py, data, state)?;
    let (args, kwargs) = build_call_args(py, &values, &implementation.params)?;
    match implementation.bound_to {
        Some(bound_to) => {
            let bound_instance = execute_node(py, data, state, bound_to)?;
            bind_lazy_refs(py, data, state)?;
            let args = prepend_to_tuple(py, bound_instance.bind(py), &args)?;
            impl_ref.call(py, args, kwargs.as_ref())
        }
        None => impl_ref.call(py, args, kwargs.as_ref()),
    }
}

fn unwrap_implementation_result(
    py: Python<'_>,
    result: Py<PyAny>,
    implementation: &ExecutionMethodImplementation,
    sync_contexts: &mut Option<&mut Vec<Py<PyAny>>>,
) -> PyResult<Py<PyAny>> {
    match implementation.return_wrapper {
        WrapperKind::None => Ok(result),
        WrapperKind::ContextManager => {
            let Some(contexts) = sync_contexts.as_deref_mut() else {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "context manager method implementation requires a context manager transition",
                ));
            };
            let entered = result.call_method0(py, "__enter__")?;
            contexts.push(result);
            Ok(entered)
        }
        WrapperKind::Awaitable => {
            let _ = result.call_method0(py, "close");
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "awaitable method implementation cannot run in a synchronous transition",
            ))
        }
        WrapperKind::AsyncContextManager => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "async context manager method implementation cannot run in a synchronous transition",
        )),
    }
}

fn cleanup_sync_contexts_after_error(
    py: Python<'_>,
    contexts: Vec<Py<PyAny>>,
    error: PyErr,
) -> PyErr {
    if contexts.is_empty() {
        return error;
    }

    let exc_type = error.get_type(py).into_any().unbind();
    let exc_val = error.value(py).clone().into_any().unbind();
    let exc_tb = error
        .traceback(py)
        .map(|tb| tb.into_any().unbind())
        .unwrap_or_else(|| py.None());

    match exit_sync_contexts(py, contexts, &exc_type, &exc_val, &exc_tb) {
        Ok(_) => error,
        Err(cleanup_error) => cleanup_error,
    }
}

pub(crate) fn exit_sync_contexts(
    py: Python<'_>,
    mut contexts: Vec<Py<PyAny>>,
    exc_type: &Py<PyAny>,
    exc_val: &Py<PyAny>,
    exc_tb: &Py<PyAny>,
) -> PyResult<bool> {
    let mut active_type = exc_type.clone_ref(py);
    let mut active_val = exc_val.clone_ref(py);
    let mut active_tb = exc_tb.clone_ref(py);
    let mut suppressed = false;

    for context in contexts.drain(..).rev() {
        let had_exception = !active_type.bind(py).is_none();
        let result =
            context.call_method1(py, "__exit__", (&active_type, &active_val, &active_tb))?;
        if had_exception && result.bind(py).is_truthy()? {
            active_type = py.None();
            active_val = py.None();
            active_tb = py.None();
            suppressed = true;
        }
    }

    Ok(suppressed)
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
            let mut members: HashMap<Arc<str>, Py<PyAny>> = HashMap::new();
            for (name, mid) in &member_entries {
                let val = execute_node(py, data, state, *mid)?;
                members.insert(name.clone(), val);
            }
            let dict = DelegatedDict::new(members);
            Ok(Py::new(py, dict)?.into_any())
        }

        ExecutionNode::LazyRef { target } => {
            let cell = LazyRefImpl::new();
            let py_cell = Py::new(py, cell)?;
            state.lazy_cells.push((py_cell.clone_ref(py), *target));
            Ok(py_cell.into_any())
        }

        ExecutionNode::Method {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            implementations,
            target,
        } => {
            let resources = if state.capture_root_transition || node_id != data.root_node {
                let introduced = transition_introduced_sources(params, implementations);
                let plan = resource_plan_for_roots(
                    &data.graph,
                    transition_body_roots(*target, implementations),
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
                implementations: implementations.clone(),
            };
            let transition = Transition::new(shared, *return_wrapper);
            Ok(Py::new(py, transition)?.into_any())
        }

        ExecutionNode::AutoMethod {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            target,
        } => {
            let resources = if state.capture_root_transition || node_id != data.root_node {
                let introduced = params
                    .iter()
                    .flat_map(|param| param.sources.iter().copied())
                    .collect();
                let plan = resource_plan_for_node(&data.graph, *target, &introduced);
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
                implementations: Vec::new(),
            };
            let transition = Transition::new(shared, *return_wrapper);
            Ok(Py::new(py, transition)?.into_any())
        }
    }
}

fn prepend_to_tuple<'py>(
    py: Python<'py>,
    first: &Bound<'py, PyAny>,
    rest: &Bound<'py, pyo3::types::PyTuple>,
) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
    let mut items: Vec<Bound<'py, PyAny>> = Vec::with_capacity(rest.len() + 1);
    items.push(first.clone());
    for item in rest.iter() {
        items.push(item);
    }
    pyo3::types::PyTuple::new(py, items)
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
