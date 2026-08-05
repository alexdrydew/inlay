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

use super::cell::{CellHandle, CellImpl, ReadCellImpl};
use super::proxy::{ContextProxy, DelegatedDict};
use super::resource_plan::{resource_plan_for_node, resource_plan_for_transition};
use super::resources::{InProgressOwners, RuntimeResources};
use super::transition::{Transition, TransitionShared};

#[derive(Clone)]
pub(crate) struct ContextData {
    pub(crate) graph: Arc<ExecutionGraph>,
    pub(crate) root_node: ExecutionNodeId,
}

pub(crate) struct InProgressGuard {
    owners: InProgressOwners,
    node_id: ExecutionNodeId,
    thread_id: std::thread::ThreadId,
}

impl InProgressGuard {
    pub(crate) fn enter(owners: &InProgressOwners, node_id: ExecutionNodeId) -> PyResult<Self> {
        let thread_id = std::thread::current().id();
        if !owners
            .lock()
            .expect("poisoned")
            .entry(node_id)
            .or_default()
            .insert(thread_id)
        {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "execution node accessed while it was still being computed",
            ));
        }
        Ok(Self {
            owners: Arc::clone(owners),
            node_id,
            thread_id,
        })
    }
}

impl Drop for InProgressGuard {
    fn drop(&mut self) {
        let mut in_progress = self.owners.lock().expect("poisoned");
        let owners = in_progress
            .get_mut(&self.node_id)
            .expect("in-progress node missing");
        owners.remove(&self.thread_id);
        if owners.is_empty() {
            in_progress.remove(&self.node_id);
        }
    }
}

pub(crate) fn current_thread_owns_node(
    in_progress: &InProgressOwners,
    node_id: ExecutionNodeId,
) -> bool {
    let thread_id = std::thread::current().id();
    in_progress
        .lock()
        .expect("poisoned")
        .get(&node_id)
        .is_some_and(|owners| owners.contains(&thread_id))
}

pub(crate) fn current_thread_owns_in_progress_node(
    in_progress: &InProgressOwners,
    except: Option<ExecutionNodeId>,
) -> bool {
    let thread_id = std::thread::current().id();
    in_progress
        .lock()
        .expect("poisoned")
        .iter()
        .any(|(node_id, owners)| Some(*node_id) != except && owners.contains(&thread_id))
}

pub(crate) struct ExecutionState {
    pub(crate) resources: RuntimeResources,
    pub(crate) capture_root_transition: bool,
    pub(crate) in_progress: InProgressOwners,
}

impl ExecutionState {
    pub(crate) fn new(resources: RuntimeResources, capture_root_transition: bool) -> Self {
        let in_progress = resources.in_progress();
        Self {
            resources,
            capture_root_transition,
            in_progress,
        }
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.resources.traverse_py_refs(visit)
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

    execute_node(py, data, &mut state, data.root_node)
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
    let (args, kwargs) = build_call_args(py, &values, &implementation.params)?;
    match implementation.bound_to {
        Some(bound_to) => {
            let bound_instance = execute_node(py, data, state, bound_to)?;
            let args = prepend_to_tuple(py, bound_instance.bind(py), &args)?;
            impl_ref.call(py, args, kwargs.as_ref())
        }
        None => impl_ref.call(py, args, kwargs.as_ref()),
    }
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
        let start_generation = {
            let guard = cache.lock().expect("poisoned");
            if let Some(cached) = guard.value.as_ref() {
                return Ok(cached.value.clone_ref(py));
            }
            guard.generation
        };

        let result = guarded_dispatch_node(py, data, state, node_id, &node)?;
        let mut guard = cache.lock().expect("poisoned");
        if let Some(cached) = guard.value.as_ref() {
            return Ok(cached.value.clone_ref(py));
        }
        if guard.generation == start_generation {
            guard.value = Some(super::resources::CachedValue {
                value: result.clone_ref(py),
                origin: super::resources::CacheValueOrigin::Computed,
            });
        }
        return Ok(result);
    }

    guarded_dispatch_node(py, data, state, node_id, &node)
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

fn guarded_dispatch_node(
    py: Python<'_>,
    data: &ContextData,
    state: &mut ExecutionState,
    node_id: ExecutionNodeId,
    node: &ExecutionNode,
) -> PyResult<Py<PyAny>> {
    let guard = InProgressGuard::enter(&state.in_progress, node_id)?;
    let result = dispatch_node(py, data, state, node_id, node);
    drop(guard);
    result
}

fn dispatch_node(
    py: Python<'_>,
    data: &ContextData,
    state: &mut ExecutionState,
    node_id: ExecutionNodeId,
    node: &ExecutionNode,
) -> PyResult<Py<PyAny>> {
    match node {
        ExecutionNode::Variable => state
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
        ExecutionNode::Variable => {
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
            state.resources.invalidate_field_dependants(&data.graph);
            Ok(())
        }
        ExecutionNode::Computed(computed) if !computed.dynamic => {
            state.resources.write_override(&data.graph, node_id, value);
            Ok(())
        }
        ExecutionNode::Computed(_) => Err(pyo3::exceptions::PyAttributeError::new_err(
            "dynamic computed execution node is not writable",
        )),
    }
}

pub(crate) fn node_is_writable(graph: &ExecutionGraph, node_id: ExecutionNodeId) -> bool {
    matches!(
        graph[node_id].node,
        ExecutionNode::Variable
            | ExecutionNode::Field(_)
            | ExecutionNode::Computed(ExecutionComputed { dynamic: false, .. })
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
                .filter(|&(_, &node_id)| node_is_writable(&data.graph, node_id))
                .map(|(name, _)| name.clone())
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

        ExecutionComputedKind::ReadCell { target } => {
            let plan = resource_plan_for_node(&data.graph, *target, &HashSet::new());
            let resources = state.resources.capture_plan(py, &plan)?;
            let handle = CellHandle::new(Arc::clone(&data.graph), *target, resources);
            Ok(Py::new(py, ReadCellImpl::new(handle))?.into_any())
        }

        ExecutionComputedKind::Cell { target } => {
            let plan = resource_plan_for_node(&data.graph, *target, &HashSet::new());
            let resources = state.resources.capture_plan(py, &plan)?;
            let handle = CellHandle::new(Arc::clone(&data.graph), *target, resources);
            Ok(Py::new(py, CellImpl::new(handle))?.into_any())
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pyo3::types::{PyInt, PyString};

    use super::*;
    use crate::compile::execution_graph::tests::{
        execution_graph, execution_node_id, execution_source_node_id,
    };
    use crate::compile::execution_graph::{
        ExecutionComputed, ExecutionRuntimeUnionBranch, RuntimeTypeMatcher,
    };

    #[test]
    fn selector_write_invalidates_static_runtime_union_dispatch() {
        Python::initialize();
        Python::attach(|py| {
            let selector = execution_source_node_id(0);
            let none_arm = execution_source_node_id(1);
            let int_arm = execution_source_node_id(2);
            let none_target = execution_node_id(3);
            let int_target = execution_node_id(4);
            let dispatch = execution_node_id(5);
            let static_value = |value: &str| {
                ExecutionNode::Computed(ExecutionComputed {
                    dynamic: false,
                    cache: ExecutionCachePolicy::Cached,
                    writable_dependencies: Vec::new(),
                    kind: ExecutionComputedKind::StaticValue {
                        value: Arc::new(PyString::new(py, value).into_any().unbind()),
                    },
                })
            };
            let graph = Arc::new(execution_graph(vec![
                ExecutionNode::Variable,
                ExecutionNode::Variable,
                ExecutionNode::Variable,
                static_value("none"),
                static_value("int"),
                ExecutionNode::Computed(ExecutionComputed {
                    dynamic: false,
                    cache: ExecutionCachePolicy::Cached,
                    writable_dependencies: vec![selector.node_id()],
                    kind: ExecutionComputedKind::RuntimeUnionDispatch {
                        source: selector,
                        branches: vec![
                            ExecutionRuntimeUnionBranch {
                                matcher: RuntimeTypeMatcher::None,
                                target: none_target,
                                arm_source: none_arm,
                            },
                            ExecutionRuntimeUnionBranch {
                                matcher: RuntimeTypeMatcher::Class {
                                    origin: Arc::new(py.get_type::<PyInt>().into_any().unbind()),
                                    display_name: Arc::from("int"),
                                },
                                target: int_target,
                                arm_source: int_arm,
                            },
                        ],
                    },
                }),
            ]));
            let data = ContextData {
                graph: Arc::clone(&graph),
                root_node: dispatch,
            };
            let mut state = ExecutionState::new(RuntimeResources::empty(), false);

            state.resources.insert_source(&graph, selector, py.None());
            let first = execute_node(py, &data, &mut state, dispatch).expect("first dispatch");
            assert_eq!(first.extract::<String>(py).expect("first value"), "none");

            state.resources.insert_source(
                &graph,
                selector,
                1_i64.into_pyobject(py).expect("int").unbind().into_any(),
            );
            let second = execute_node(py, &data, &mut state, dispatch).expect("second dispatch");
            assert_eq!(second.extract::<String>(py).expect("second value"), "int");
        });
    }
}
