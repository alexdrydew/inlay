use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, Weak};

use pyo3::PyTraverseError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::compile::execution_graph::{ExecutionGraph, ExecutionNodeId, ExecutionSourceNodeId};

use super::resource_plan::ResourcePlan;

#[derive(Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CacheValueOrigin {
    #[default]
    Computed,
    CellOverride,
}

pub(crate) struct CachedValue {
    pub(crate) value: Py<PyAny>,
    pub(crate) origin: CacheValueOrigin,
}

pub(crate) struct CacheCell {
    pub(crate) value: Option<CachedValue>,
    pub(crate) generation: u64,
}

impl CacheCell {
    fn empty() -> Self {
        Self {
            value: None,
            generation: 0,
        }
    }

    fn with_value(value: Py<PyAny>, origin: CacheValueOrigin) -> Self {
        Self {
            value: Some(CachedValue { value, origin }),
            generation: 0,
        }
    }

    fn invalidate_computed(&mut self) {
        if matches!(
            self.value.as_ref().map(|value| value.origin),
            Some(CacheValueOrigin::Computed)
        ) {
            self.value = None;
        }
        self.generation = self.generation.wrapping_add(1);
    }
}

pub(crate) type CacheRef = Arc<Mutex<CacheCell>>;
type CacheWeakRef = Weak<Mutex<CacheCell>>;
type SharedCacheRegistry = HashMap<ExecutionNodeId, Vec<CacheWeakRef>>;
type SourceRef = Arc<Mutex<Py<PyAny>>>;
pub(crate) type InProgressOwners =
    Arc<Mutex<HashMap<ExecutionNodeId, HashSet<std::thread::ThreadId>>>>;

#[derive(Default)]
pub(crate) struct RuntimeResources {
    sources: HashMap<ExecutionSourceNodeId, SourceRef>,
    owned_caches: HashMap<ExecutionNodeId, CacheRef>,
    shared_cache_registry: Arc<Mutex<SharedCacheRegistry>>,
    in_progress: InProgressOwners,
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
    #[serde(default)]
    origin: CacheValueOrigin,
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

    pub(crate) fn clone_ref(&self, _py: Python<'_>) -> Self {
        let owned_caches = self
            .owned_caches
            .iter()
            .map(|(&node_id, cache)| (node_id, Arc::clone(cache)))
            .collect();
        Self {
            sources: self
                .sources
                .iter()
                .map(|(&source, value)| (source, Arc::clone(value)))
                .collect(),
            owned_caches,
            shared_cache_registry: Arc::clone(&self.shared_cache_registry),
            in_progress: Arc::clone(&self.in_progress),
        }
    }

    pub(crate) fn in_progress(&self) -> InProgressOwners {
        Arc::clone(&self.in_progress)
    }

    pub(crate) fn get_source(
        &self,
        py: Python<'_>,
        source: ExecutionSourceNodeId,
    ) -> PyResult<Py<PyAny>> {
        self.sources
            .get(&source)
            .map(|value| value.lock().expect("poisoned").clone_ref(py))
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

    pub(crate) fn write_override(
        &mut self,
        graph: &ExecutionGraph,
        node_id: ExecutionNodeId,
        value: Py<PyAny>,
    ) {
        let cache = self.get_or_create_cache(node_id);
        self.invalidate_dependants(graph, node_id);
        self.owned_caches.insert(node_id, Arc::clone(&cache));
        let mut guard = cache.lock().expect("poisoned");
        guard.generation = guard.generation.wrapping_add(1);
        guard.value = Some(CachedValue {
            value,
            origin: CacheValueOrigin::CellOverride,
        });
    }

    pub(crate) fn insert_source(
        &mut self,
        graph: &ExecutionGraph,
        source: ExecutionSourceNodeId,
        value: Py<PyAny>,
    ) {
        match self.sources.get(&source) {
            Some(current) => *current.lock().expect("poisoned") = value,
            None => {
                self.sources.insert(source, Arc::new(Mutex::new(value)));
            }
        }
        self.invalidate_dependants(graph, source.node_id());
    }

    pub(crate) fn invalidate_dependants(
        &mut self,
        graph: &ExecutionGraph,
        writable_node: ExecutionNodeId,
    ) {
        let affected = graph.affected_dependants(writable_node);
        self.shared_cache_registry
            .lock()
            .expect("poisoned")
            .retain(|node_id, weak_caches| {
                let invalidate = affected.contains(node_id);
                weak_caches.retain(|weak_cache| {
                    let Some(cache) = weak_cache.upgrade() else {
                        return false;
                    };
                    if invalidate {
                        cache.lock().expect("poisoned").invalidate_computed();
                    }
                    true
                });
                !weak_caches.is_empty()
            });
        self.owned_caches.retain(|node_id, cache| {
            if !affected.contains(node_id) {
                return true;
            }
            matches!(
                cache
                    .lock()
                    .expect("poisoned")
                    .value
                    .as_ref()
                    .map(|v| v.origin),
                Some(CacheValueOrigin::CellOverride)
            )
        });
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
                    cache.lock().expect("poisoned").invalidate_computed();
                    true
                });
                !weak_caches.is_empty()
            });
        self.owned_caches.retain(|_, cache| {
            matches!(
                cache
                    .lock()
                    .expect("poisoned")
                    .value
                    .as_ref()
                    .map(|v| v.origin),
                Some(CacheValueOrigin::CellOverride)
            )
        });
    }

    pub(crate) fn capture_plan(&mut self, _py: Python<'_>, plan: &ResourcePlan) -> PyResult<Self> {
        let mut sources = HashMap::with_capacity(plan.sources.len());
        for &source in &plan.sources {
            let source_ref =
                self.sources.get(&source).map(Arc::clone).ok_or_else(|| {
                    PyRuntimeError::new_err("source value not found in resources")
                })?;
            sources.insert(source, source_ref);
        }

        let mut owned_caches = HashMap::with_capacity(plan.caches.len());
        for &node_id in &plan.caches {
            owned_caches.insert(node_id, self.get_or_create_cache(node_id));
        }

        Ok(Self {
            sources,
            owned_caches,
            shared_cache_registry: Arc::clone(&self.shared_cache_registry),
            in_progress: Arc::clone(&self.in_progress),
        })
    }

    pub(crate) fn traverse_py_refs(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for value in self.sources.values() {
            visit.call(&*value.lock().expect("poisoned"))?;
        }
        for cache in self.owned_caches.values() {
            if let Some(value) = cache.lock().expect("poisoned").value.as_ref() {
                visit.call(&value.value)?;
            }
        }
        Ok(())
    }

    pub(crate) fn cached_values(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.owned_caches
            .values()
            .filter_map(|cache| {
                cache
                    .lock()
                    .expect("poisoned")
                    .value
                    .as_ref()
                    .map(|value| value.value.clone_ref(py))
            })
            .collect()
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
                    value_ref: refs.push(py, &value.lock().expect("poisoned")),
                    origin: CacheValueOrigin::Computed,
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
                            value_ref: refs.push(py, &value.value),
                            origin: value.origin,
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
                    Arc::new(Mutex::new(refs.get(entry.value_ref)?)),
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
                        entry.origin,
                    ))),
                ))
            })
            .collect::<PyResult<HashMap<_, _>>>()?;
        let shared_cache_registry = weak_cache_registry_from_owned_caches(&owned_caches);

        Ok(Self {
            sources,
            owned_caches,
            shared_cache_registry,
            in_progress: Default::default(),
        })
    }
}
