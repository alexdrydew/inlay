use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use pyo3::PyTraverseError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::compile::execution_graph::{ExecutionGraph, ExecutionNodeId, ExecutionSourceNodeId};

use super::resource_plan::ResourcePlan;

pub(crate) struct CacheCell {
    pub(crate) value: Option<Py<PyAny>>,
    pub(crate) generation: u64,
}

impl CacheCell {
    fn empty() -> Self {
        Self {
            value: None,
            generation: 0,
        }
    }

    fn with_value(value: Py<PyAny>) -> Self {
        Self {
            value: Some(value),
            generation: 0,
        }
    }

    fn invalidate(&mut self) {
        self.value = None;
        self.generation = self.generation.wrapping_add(1);
    }
}

pub(crate) type CacheRef = Arc<Mutex<CacheCell>>;
type CacheWeakRef = Weak<Mutex<CacheCell>>;
type SharedCacheRegistry = HashMap<ExecutionNodeId, Vec<CacheWeakRef>>;

/// Runtime resource values retained by an executing graph scope.
///
/// Cache ownership and invalidation are intentionally split: `owned_caches` keeps the Python
/// values alive for this scope and is the only cache set traversed/pickled, while
/// `shared_cache_registry` is a weak reachability index used to invalidate related graph-aware
/// views. Both precise source invalidation and conservative field-write invalidation flow through
/// the registry, and `CacheCell::generation` prevents in-flight computations from repopulating a
/// cache cell that was invalidated while they ran.
#[derive(Default)]
pub(crate) struct RuntimeResources {
    /// source bindings to Python values
    sources: HashMap<ExecutionSourceNodeId, Py<PyAny>>,
    /// cache cells owned by this runtime scope
    owned_caches: HashMap<ExecutionNodeId, CacheRef>,
    /// non-owning registry used to invalidate cache cells shared with graph-aware views captured
    /// from the same runtime scope. Cache ownership stays in `owned_caches`, so traversal, pickle
    /// state, and clear only need to account for local owning resources. The registry may contain
    /// multiple cells for the same node when separate transition calls bind different source values.
    shared_cache_registry: Arc<Mutex<SharedCacheRegistry>>,
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

fn weak_cache_registry_from_owned_caches(
    owned_caches: &HashMap<ExecutionNodeId, CacheRef>,
) -> Arc<Mutex<SharedCacheRegistry>> {
    let mut registry: SharedCacheRegistry = HashMap::new();
    for (&node_id, cache) in owned_caches {
        registry
            .entry(node_id)
            .or_default()
            .push(Arc::downgrade(cache));
    }
    Arc::new(Mutex::new(registry))
}

impl RuntimeResources {
    pub(crate) fn empty() -> Self {
        Self::default()
    }

    pub(crate) fn clone_ref(&self, py: Python<'_>) -> Self {
        let owned_caches: HashMap<_, _> = self
            .owned_caches
            .iter()
            .map(|(&node_id, cache)| (node_id, Arc::clone(cache)))
            .collect();
        Self {
            sources: self
                .sources
                .iter()
                .map(|(&source, value)| (source, value.clone_ref(py)))
                .collect(),
            owned_caches,
            shared_cache_registry: Arc::clone(&self.shared_cache_registry),
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
        if let Some(cache) = self.owned_caches.get(&node_id) {
            return Arc::clone(cache);
        }
        let cache = Arc::new(Mutex::new(CacheCell::empty()));
        {
            let mut store = self.shared_cache_registry.lock().expect("poisoned");
            let entries = store.entry(node_id).or_default();
            entries.retain(|weak_cache| weak_cache.strong_count() > 0);
            entries.push(Arc::downgrade(&cache));
        }
        self.owned_caches.insert(node_id, Arc::clone(&cache));
        cache
    }

    pub(crate) fn insert_source(
        &mut self,
        graph: &ExecutionGraph,
        source: ExecutionSourceNodeId,
        value: Py<PyAny>,
    ) {
        self.sources.insert(source, value);
        self.invalidate_dependants(graph, source);
    }

    pub(crate) fn invalidate_dependants(
        &mut self,
        graph: &ExecutionGraph,
        source: ExecutionSourceNodeId,
    ) {
        self.shared_cache_registry
            .lock()
            .expect("poisoned")
            .retain(|node_id, weak_caches| {
                let invalidate = graph[*node_id].source_deps.contains(&source);
                weak_caches.retain(|weak_cache| {
                    let Some(cache) = weak_cache.upgrade() else {
                        return false;
                    };
                    if invalidate {
                        cache.lock().expect("poisoned").invalidate();
                    }
                    true
                });
                !weak_caches.is_empty()
            });
        self.owned_caches
            .retain(|node_id, _| !graph[*node_id].source_deps.contains(&source));
    }

    pub(crate) fn invalidate_all_caches(&mut self) {
        self.shared_cache_registry
            .lock()
            .expect("poisoned")
            .retain(|_, weak_caches| {
                weak_caches.retain(|weak_cache| {
                    let Some(cache) = weak_cache.upgrade() else {
                        return false;
                    };
                    cache.lock().expect("poisoned").invalidate();
                    true
                });
                !weak_caches.is_empty()
            });
        self.owned_caches.clear();
    }

    pub(crate) fn capture_plan(&mut self, py: Python<'_>, plan: &ResourcePlan) -> PyResult<Self> {
        let mut sources = HashMap::with_capacity(plan.sources.len());
        for &source in &plan.sources {
            sources.insert(source, self.get_source(py, source)?);
        }

        let mut owned_caches = HashMap::with_capacity(plan.caches.len());
        for &node_id in &plan.caches {
            owned_caches.insert(node_id, self.get_or_create_cache(node_id));
        }

        Ok(Self {
            sources,
            owned_caches,
            shared_cache_registry: Arc::clone(&self.shared_cache_registry),
        })
    }

    pub(crate) fn traverse_py_refs(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for value in self.sources.values() {
            visit.call(value)?;
        }
        for cache in self.owned_caches.values() {
            if let Some(value) = cache.lock().expect("poisoned").value.as_ref() {
                visit.call(value)?;
            }
        }
        Ok(())
    }

    pub(crate) fn clear(&mut self) {
        self.sources.clear();
        self.owned_caches.clear();
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
                .owned_caches
                .iter()
                .filter_map(|(node_id, cache)| {
                    cache
                        .lock()
                        .expect("poisoned")
                        .value
                        .as_ref()
                        .map(|value| ResourceValueState {
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

        let owned_caches = state
            .caches
            .iter()
            .map(|entry| {
                Ok((
                    ExecutionNodeId::from_index(entry.id),
                    Arc::new(Mutex::new(CacheCell::with_value(
                        refs.get(entry.value_ref)?,
                    ))),
                ))
            })
            .collect::<PyResult<HashMap<_, _>>>()?;
        let shared_cache_registry = weak_cache_registry_from_owned_caches(&owned_caches);

        Ok(Self {
            sources,
            owned_caches,
            shared_cache_registry,
        })
    }
}
