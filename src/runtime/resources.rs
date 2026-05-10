use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use pyo3::PyTraverseError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;

use crate::compile::flatten::{ExecutionGraph, ExecutionNodeId, ExecutionSourceNodeId};

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
}
