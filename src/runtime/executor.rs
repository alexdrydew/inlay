use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use inlay_instrument::instrumented;
use pyo3::PyTraverseError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::compile::execution_graph::{
    ConstructorParam, ExecutionCachePolicy, ExecutionComputed, ExecutionComputedKind,
    ExecutionField, ExecutionGraph, ExecutionNode, ExecutionNodeId, ExecutionSourceNodeId,
    ExecutionTransitionImplementation, ExecutionTransitionImplementationCallable,
    RuntimeCallableMatchParam, RuntimeTypeMatcher,
};
use crate::types::ParamKind;

use super::lazy_ref::LazyRefImpl;
use super::proxy::{ContextProxy, DelegatedDict};
use super::resource_plan::{resource_plan_for_node, resource_plan_for_transition};
use super::resources::RuntimeResources;
use super::transition::{Transition, TransitionShared};

#[derive(Clone)]
pub(crate) struct ContextData {
    pub(crate) graph: Arc<ExecutionGraph>,
    pub(crate) root_node: ExecutionNodeId,
}

pub(crate) struct ExecutionState {
    pub(crate) resources: RuntimeResources,
    pub(crate) lazy_cells: Vec<(Py<LazyRefImpl>, ExecutionNodeId)>,
    pub(crate) capture_root_transition: bool,
}

impl ExecutionState {
    pub(crate) fn new(resources: RuntimeResources, capture_root_transition: bool) -> Self {
        Self {
            resources,
            lazy_cells: Vec::new(),
            capture_root_transition,
        }
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.resources.traverse_py_refs(visit)?;
        for (cell, _) in &self.lazy_cells {
            visit.call(cell)?;
        }
        Ok(())
    }
}

#[instrumented(name = "inlay.execute", target = "inlay", level = "trace", skip_all)]
pub(crate) fn execute(
    py: Python<'_>,
    data: &ContextData,
    resources: RuntimeResources,
    capture_root_transition: bool,
) -> PyResult<Py<PyAny>> {
    let mut state = ExecutionState::new(resources, capture_root_transition);

    let result = execute_node(py, data, &mut state, data.root_node)?;
    bind_lazy_refs(py, data, &mut state)?;

    Ok(result)
}

pub(crate) fn execute_transition_implementation(
    py: Python<'_>,
    data: &ContextData,
    state: &mut ExecutionState,
    implementation: &ExecutionTransitionImplementation,
) -> PyResult<Py<PyAny>> {
    let impl_ref = match &implementation.implementation {
        ExecutionTransitionImplementationCallable::Static(implementation) => {
            implementation.clone_ref(py)
        }
        ExecutionTransitionImplementationCallable::Source(source) => {
            get_source_value(py, data, state, *source)?
        }
    };
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

pub(crate) fn bind_lazy_refs(
    py: Python<'_>,
    data: &ContextData,
    state: &mut ExecutionState,
) -> PyResult<()> {
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
        values.push(val);
    }
    Ok(values)
}

pub(crate) fn execute_node(
    py: Python<'_>,
    data: &ContextData,
    state: &mut ExecutionState,
    node_id: ExecutionNodeId,
) -> PyResult<Py<PyAny>> {
    let node = data.graph[node_id].node.clone();
    if execution_node_uses_cache(&node) {
        let cache = state.resources.get_or_create_cache(node_id);
        if let Some(cached) = cache.get() {
            return Ok(cached.clone_ref(py));
        }

        let result = dispatch_node(py, data, state, node_id, &node)?;
        if cache.set(result.clone_ref(py)).is_err()
            && let Some(cached) = cache.get()
        {
            return Ok(cached.clone_ref(py));
        }
        return Ok(result);
    }

    dispatch_node(py, data, state, node_id, &node)
}

fn execution_node_uses_cache(node: &ExecutionNode) -> bool {
    matches!(
        node,
        ExecutionNode::Computed(ExecutionComputed {
            dynamic: false,
            cache: ExecutionCachePolicy::Cached,
            ..
        })
    )
}

fn dispatch_node(
    py: Python<'_>,
    data: &ContextData,
    state: &mut ExecutionState,
    node_id: ExecutionNodeId,
    node: &ExecutionNode,
) -> PyResult<Py<PyAny>> {
    match node {
        ExecutionNode::Variable(_) => state
            .resources
            .get_source(py, ExecutionSourceNodeId(node_id)),

        ExecutionNode::Field(field) => read_field_node(py, data, state, field),

        ExecutionNode::Computed(computed) => {
            execute_computed_node(py, data, state, node_id, computed)
        }
    }
}

fn read_field_node(
    py: Python<'_>,
    data: &ContextData,
    state: &mut ExecutionState,
    field: &ExecutionField,
) -> PyResult<Py<PyAny>> {
    let source_obj = execute_node(py, data, state, field.source)?;
    match field.access_kind {
        crate::types::MemberAccessKind::Attribute => source_obj
            .bind(py)
            .getattr(field.name.as_ref())
            .map(|v| v.unbind()),
        crate::types::MemberAccessKind::DictItem => source_obj
            .bind(py)
            .get_item(field.name.as_ref())
            .map(|v| v.unbind()),
    }
}

pub(crate) fn write_node(
    py: Python<'_>,
    data: &ContextData,
    state: &mut ExecutionState,
    node_id: ExecutionNodeId,
    value: Py<PyAny>,
) -> PyResult<()> {
    let node = data.graph[node_id].node.clone();
    match node {
        ExecutionNode::Variable(_) => {
            state
                .resources
                .insert_source(&data.graph, ExecutionSourceNodeId(node_id), value);
            Ok(())
        }
        ExecutionNode::Field(field) => {
            let source_obj = execute_node(py, data, state, field.source)?;
            match field.access_kind {
                crate::types::MemberAccessKind::Attribute => source_obj
                    .bind(py)
                    .setattr(field.name.as_ref(), value.bind(py))?,
                crate::types::MemberAccessKind::DictItem => source_obj
                    .bind(py)
                    .set_item(field.name.as_ref(), value.bind(py))?,
            }
            state
                .resources
                .invalidate_dependants(&data.graph, ExecutionSourceNodeId(node_id));
            Ok(())
        }
        ExecutionNode::Computed(_) => Err(pyo3::exceptions::PyAttributeError::new_err(
            "computed execution node is not writable",
        )),
    }
}

pub(crate) fn node_is_writable(graph: &ExecutionGraph, node_id: ExecutionNodeId) -> bool {
    matches!(
        graph[node_id].node,
        ExecutionNode::Variable(_) | ExecutionNode::Field(_)
    )
}

fn execute_computed_node(
    py: Python<'_>,
    data: &ContextData,
    state: &mut ExecutionState,
    node_id: ExecutionNodeId,
    computed: &ExecutionComputed,
) -> PyResult<Py<PyAny>> {
    match &computed.kind {
        ExecutionComputedKind::None => Ok(py.None()),

        ExecutionComputedKind::StaticValue { value } => Ok(value.clone_ref(py)),

        ExecutionComputedKind::Constructor {
            implementation,
            params,
        } => {
            let impl_ref = Arc::clone(implementation);
            let values = execute_constructor_params(py, data, state, params)?;
            let (args, kwargs) = build_call_args(py, &values, params)?;
            impl_ref.call(py, args, kwargs.as_ref())
        }

        ExecutionComputedKind::Property {
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

        ExecutionComputedKind::Protocol { members } => {
            let member_entries: HashMap<Arc<str>, ExecutionNodeId> = members
                .iter()
                .map(|(name, &node)| (name.clone(), node))
                .collect();
            let writable: HashSet<Arc<str>> = member_entries
                .iter()
                .filter_map(|(name, &node_id)| {
                    node_is_writable(&data.graph, node_id).then(|| name.clone())
                })
                .collect();
            let plan = resource_plan_for_node(&data.graph, node_id, &HashSet::new());
            let resources = state.resources.capture_plan(py, &plan)?;
            let proxy =
                ContextProxy::new(Arc::clone(&data.graph), member_entries, writable, resources);
            Ok(Py::new(py, proxy)?.into_any())
        }

        ExecutionComputedKind::TypedDict { members } => {
            let member_entries: HashMap<Arc<str>, ExecutionNodeId> =
                members.iter().map(|(k, &v)| (k.clone(), v)).collect();
            let plan = resource_plan_for_node(&data.graph, node_id, &HashSet::new());
            let resources = state.resources.capture_plan(py, &plan)?;
            let dict = DelegatedDict::new(Arc::clone(&data.graph), member_entries, resources);
            Ok(Py::new(py, dict)?.into_any())
        }

        ExecutionComputedKind::LazyRef { target } => {
            let cell = LazyRefImpl::new();
            let py_cell = Py::new(py, cell)?;
            state.lazy_cells.push((py_cell.clone_ref(py), *target));
            Ok(py_cell.into_any())
        }

        ExecutionComputedKind::Transition {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            implementations,
            target,
        } => {
            let resources = if state.capture_root_transition || node_id != data.root_node {
                let plan =
                    resource_plan_for_transition(&data.graph, params, implementations, *target);
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

        ExecutionComputedKind::RuntimeUnionDispatch { source, branches } => {
            let value = get_source_value(py, data, state, *source)?;
            for branch in branches {
                if runtime_matcher_matches(py, &branch.matcher, &value)? {
                    state.resources.insert_source(
                        &data.graph,
                        branch.arm_source,
                        value.clone_ref(py),
                    );
                    return execute_node(py, data, state, branch.target);
                }
            }
            let matchers = branches
                .iter()
                .map(|branch| runtime_matcher_summary(&branch.matcher))
                .collect::<Vec<_>>()
                .join(", ");
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "runtime union value did not match any implementation variant: {matchers}"
            )))
        }
    }
}

fn get_source_value(
    py: Python<'_>,
    data: &ContextData,
    state: &mut ExecutionState,
    source: ExecutionSourceNodeId,
) -> PyResult<Py<PyAny>> {
    match state.resources.get_source(py, source) {
        Ok(value) => Ok(value),
        Err(_) => execute_node(py, data, state, source.node_id()),
    }
}

fn runtime_matcher_summary(matcher: &RuntimeTypeMatcher) -> String {
    match matcher {
        RuntimeTypeMatcher::None => "None".to_string(),
        RuntimeTypeMatcher::Class { display_name, .. } => display_name.to_string(),
        RuntimeTypeMatcher::Callable { .. } => "callable".to_string(),
    }
}

fn runtime_matcher_matches(
    py: Python<'_>,
    matcher: &RuntimeTypeMatcher,
    value: &Py<PyAny>,
) -> PyResult<bool> {
    match matcher {
        RuntimeTypeMatcher::None => Ok(value.bind(py).is_none()),
        RuntimeTypeMatcher::Class { origin, .. } => value.bind(py).is_instance(origin.bind(py)),
        RuntimeTypeMatcher::Callable { params } => callable_matcher_matches(py, params, value),
    }
}

fn callable_matcher_matches(
    py: Python<'_>,
    params: &[RuntimeCallableMatchParam],
    value: &Py<PyAny>,
) -> PyResult<bool> {
    let bound = value.bind(py);
    if !bound.is_callable() {
        return Ok(false);
    }

    let inspect = py.import("inspect")?;
    let signature_fn = inspect.getattr("signature")?;
    let signature = match signature_fn.call1((bound,)) {
        Ok(signature) => signature,
        Err(_) => return Ok(false),
    };

    let mut positional_values: Vec<Py<PyAny>> = Vec::new();
    let kwargs = PyDict::new(py);
    let mut has_kwargs = false;
    for param in params {
        if param.has_default {
            continue;
        }
        match param.kind {
            ParamKind::PositionalOnly | ParamKind::PositionalOrKeyword => {
                positional_values.push(py.None());
            }
            ParamKind::KeywordOnly => {
                kwargs.set_item(param.name.as_ref(), py.None())?;
                has_kwargs = true;
            }
        }
    }
    let positional_refs = positional_values.iter().collect::<Vec<_>>();
    let args = pyo3::types::PyTuple::new(py, positional_refs)?;
    let kwargs = has_kwargs.then_some(kwargs);

    match signature.call_method("bind", args, kwargs.as_ref()) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
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
