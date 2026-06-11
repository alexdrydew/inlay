use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use pyo3::PyTraverseError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::compile::execution_graph::{ExecutionGraph, ExecutionNodeId, ExecutionSourceNodeId};

use super::resource_plan::ResourcePlan;

pub(crate) type CacheRef = Arc<OnceLock<Py<PyAny>>>;

/// Runtime resource values retained by an executing graph scope.
#[derive(Default)]
pub(crate) struct RuntimeResources {
    /// source slot bindings to Python values
    sources: HashMap<ExecutionSourceNodeId, Py<PyAny>>,
    /// cache cells shared by this runtime scope
    caches: HashMap<ExecutionNodeId, CacheRef>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct RuntimeResourcesState {
    sources: Vec<ResourceValueState>,
    caches: Vec<ResourceValueState>,
}

#[derive(Serialize, Deserialize)]
struct ResourceValueState {
    id: usize,
    value_ref: usize,
}

impl RuntimeResources {
    pub(crate) fn empty() -> Self {
        Self::default()
    }

    pub(crate) fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            sources: self
                .sources
                .iter()
                .map(|(&source, value)| (source, value.clone_ref(py)))
                .collect(),
            caches: self
                .caches
                .iter()
                .map(|(&node_id, cache)| (node_id, Arc::clone(cache)))
                .collect(),
        }
    }

    pub(crate) fn get_source(
        &self,
        py: Python<'_>,
        source: ExecutionSourceNodeId,
    ) -> PyResult<Py<PyAny>> {
        self.sources
            .get(&source)
            .map(|value| value.clone_ref(py))
            .ok_or_else(|| PyRuntimeError::new_err("source value not found in resources"))
    }

    pub(crate) fn get_or_create_cache(&mut self, node_id: ExecutionNodeId) -> CacheRef {
        Arc::clone(
            self.caches
                .entry(node_id)
                .or_insert_with(|| Arc::new(OnceLock::new())),
        )
    }

    pub(crate) fn insert_source(
        &mut self,
        graph: &ExecutionGraph,
        source: ExecutionSourceNodeId,
        value: Py<PyAny>,
    ) {
        self.sources.insert(source, value);
        self.caches
            .retain(|node_id, _| !graph[*node_id].source_deps.contains(&source));
    }

    pub(crate) fn capture_plan(&mut self, py: Python<'_>, plan: &ResourcePlan) -> PyResult<Self> {
        let mut sources = HashMap::with_capacity(plan.sources.len());
        for &source in &plan.sources {
            sources.insert(source, self.get_source(py, source)?);
        }

        let mut caches = HashMap::with_capacity(plan.caches.len());
        for &node_id in &plan.caches {
            caches.insert(node_id, self.get_or_create_cache(node_id));
        }

        Ok(Self { sources, caches })
    }

    pub(crate) fn traverse_py_refs(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for value in self.sources.values() {
            visit.call(value)?;
        }
        for cache in self.caches.values() {
            if let Some(value) = cache.get() {
                visit.call(value)?;
            }
        }
        Ok(())
    }

    pub(crate) fn clear(&mut self) {
        self.sources.clear();
        self.caches.clear();
    }

    pub(crate) fn to_state(
        &self,
        py: Python<'_>,
        refs: &mut crate::pickle::PyRefCollector,
    ) -> RuntimeResourcesState {
        RuntimeResourcesState {
            sources: self
                .sources
                .iter()
                .map(|(source, value)| ResourceValueState {
                    id: source.node_id().index(),
                    value_ref: refs.push(py, value),
                })
                .collect(),
            caches: self
                .caches
                .iter()
                .filter_map(|(node_id, cache)| {
                    cache.get().map(|value| ResourceValueState {
                        id: node_id.index(),
                        value_ref: refs.push(py, value),
                    })
                })
                .collect(),
        }
    }

    pub(crate) fn from_state(
        state: RuntimeResourcesState,
        refs: &crate::pickle::PyRefResolver<'_>,
    ) -> PyResult<Self> {
        let sources = state
            .sources
            .iter()
            .map(|entry| {
                Ok((
                    ExecutionSourceNodeId(ExecutionNodeId::from_index(entry.id)),
                    refs.get(entry.value_ref)?,
                ))
            })
            .collect::<PyResult<HashMap<_, _>>>()?;

        let caches = state
            .caches
            .iter()
            .map(|entry| {
                let cell = Arc::new(OnceLock::new());
                assert!(
                    cell.set(refs.get(entry.value_ref)?).is_ok(),
                    "newly-created pickle cache cell must be empty"
                );
                Ok((ExecutionNodeId::from_index(entry.id), cell))
            })
            .collect::<PyResult<HashMap<_, _>>>()?;

        Ok(Self { sources, caches })
    }
}
