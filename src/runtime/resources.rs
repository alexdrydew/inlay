use std::cell::Cell as ThreadCell;
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

    fn invalidate_computed(&mut self) -> Option<CachedValue> {
        let removed = matches!(
            self.value.as_ref().map(|value| value.origin),
            Some(CacheValueOrigin::Computed)
        )
        .then(|| self.value.take())
        .flatten();
        self.generation = self.generation.wrapping_add(1);
        removed
    }
}

pub(crate) type CacheRef = Arc<Mutex<CacheCell>>;
type CacheWeakRef = Weak<Mutex<CacheCell>>;
type SharedCacheRegistry = HashMap<ExecutionNodeId, Vec<CacheWeakRef>>;
type SourceRef = Arc<Mutex<Py<PyAny>>>;
pub(crate) type InProgressOwners =
    Arc<Mutex<HashMap<ExecutionNodeId, HashSet<std::thread::ThreadId>>>>;

thread_local! {
    static ACTIVE_RESOURCE_LEASES: ThreadCell<usize> = const { ThreadCell::new(0) };
}

pub(crate) struct ActiveResourceLease;

impl ActiveResourceLease {
    pub(crate) fn enter() -> Self {
        ACTIVE_RESOURCE_LEASES.with(|count| count.set(count.get() + 1));
        Self
    }

    pub(crate) fn current_thread_has_lease() -> bool {
        ACTIVE_RESOURCE_LEASES.with(|count| count.get() > 0)
    }
}

impl Drop for ActiveResourceLease {
    fn drop(&mut self) {
        ACTIVE_RESOURCE_LEASES.with(|count| {
            let current = count.get();
            debug_assert!(current > 0);
            count.set(current - 1);
        });
    }
}

#[derive(Default)]
pub(crate) struct RuntimeResources {
    sources: HashMap<ExecutionSourceNodeId, SourceRef>,
    owned_caches: HashMap<ExecutionNodeId, CacheRef>,
    shared_cache_registry: Arc<Mutex<SharedCacheRegistry>>,
    in_progress: InProgressOwners,
}

pub(crate) struct ClearedRuntimeResources {
    _sources: HashMap<ExecutionSourceNodeId, SourceRef>,
    _owned_caches: HashMap<ExecutionNodeId, CacheRef>,
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
        let removed = {
            let mut guard = cache.lock().expect("poisoned");
            guard.generation = guard.generation.wrapping_add(1);
            guard.value.replace(CachedValue {
                value,
                origin: CacheValueOrigin::CellOverride,
            })
        };
        self.owned_caches.insert(node_id, Arc::clone(&cache));
        self.invalidate_dependants(graph, node_id);
        drop(removed);
    }

    pub(crate) fn insert_source(
        &mut self,
        graph: &ExecutionGraph,
        source: ExecutionSourceNodeId,
        value: Py<PyAny>,
    ) {
        let removed = match self.sources.get(&source) {
            Some(current) => {
                let mut current = current.lock().expect("poisoned");
                Some(std::mem::replace(&mut *current, value))
            }
            None => {
                self.sources.insert(source, Arc::new(Mutex::new(value)));
                None
            }
        };
        self.invalidate_dependants(graph, source.node_id());
        drop(removed);
    }

    pub(crate) fn invalidate_dependants(
        &mut self,
        graph: &ExecutionGraph,
        writable_node: ExecutionNodeId,
    ) {
        let affected = graph.affected_dependants(writable_node);
        let mut removed_values = Vec::new();
        let mut upgraded_caches = Vec::new();
        {
            let mut registry = self.shared_cache_registry.lock().expect("poisoned");
            for node_id in &affected {
                let remove_entry = registry.get_mut(node_id).is_some_and(|weak_caches| {
                    weak_caches.retain(|weak_cache| {
                        let Some(cache) = weak_cache.upgrade() else {
                            return false;
                        };
                        let removed = cache.lock().expect("poisoned").invalidate_computed();
                        removed_values.extend(removed);
                        upgraded_caches.push(cache);
                        true
                    });
                    weak_caches.is_empty()
                });
                if remove_entry {
                    registry.remove(node_id);
                }
            }
        }
        drop(removed_values);
        drop(upgraded_caches);
        for node_id in affected {
            let retain = self.owned_caches.get(&node_id).is_some_and(|cache| {
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
            if !retain {
                self.owned_caches.remove(&node_id);
            }
        }
    }

    pub(crate) fn invalidate_all_caches(&mut self) {
        let mut removed_values = Vec::new();
        let mut upgraded_caches = Vec::new();
        {
            let mut registry = self.shared_cache_registry.lock().expect("poisoned");
            registry.retain(|_, weak_caches| {
                weak_caches.retain(|weak_cache| {
                    let Some(cache) = weak_cache.upgrade() else {
                        return false;
                    };
                    let removed = cache.lock().expect("poisoned").invalidate_computed();
                    removed_values.extend(removed);
                    upgraded_caches.push(cache);
                    true
                });
                !weak_caches.is_empty()
            });
        }
        drop(removed_values);
        drop(upgraded_caches);
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

    pub(crate) fn clear(&mut self) -> ClearedRuntimeResources {
        ClearedRuntimeResources {
            _sources: std::mem::take(&mut self.sources),
            _owned_caches: std::mem::take(&mut self.owned_caches),
        }
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
