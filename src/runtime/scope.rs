use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::PyTraverseError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;

use crate::compile::flatten::{ExecutionNodeId, ExecutionSourceNodeId};

pub(crate) struct Scope {
    parent: Option<Arc<Scope>>,
    source_values: HashMap<ExecutionNodeId, Py<PyAny>>,
    cached_values: HashMap<ExecutionNodeId, Py<PyAny>>,
    introduced_sources: HashSet<ExecutionSourceNodeId>,
}

impl Scope {
    pub(crate) fn root(source_values: HashMap<ExecutionNodeId, Py<PyAny>>) -> Self {
        Self {
            parent: None,
            source_values,
            cached_values: HashMap::new(),
            introduced_sources: HashSet::new(),
        }
    }

    pub(crate) fn child(
        parent: Arc<Scope>,
        new_sources: Vec<(ExecutionSourceNodeId, Py<PyAny>)>,
    ) -> Self {
        let mut introduced_sources = HashSet::with_capacity(new_sources.len());
        let mut source_values = HashMap::with_capacity(new_sources.len());

        for (source, value) in new_sources {
            introduced_sources.insert(source);
            source_values.insert(source.node_id(), value);
        }

        Self {
            parent: Some(parent),
            source_values,
            cached_values: HashMap::new(),
            introduced_sources,
        }
    }

    pub(crate) fn get_value(&self, node_id: ExecutionNodeId) -> Option<&Py<PyAny>> {
        self.source_values
            .get(&node_id)
            .or_else(|| self.parent.as_ref()?.get_value(node_id))
    }

    pub(crate) fn get_cached(
        &self,
        node_id: ExecutionNodeId,
        source_deps: &HashSet<ExecutionSourceNodeId>,
    ) -> Option<&Py<PyAny>> {
        if let Some(val) = self.cached_values.get(&node_id) {
            return Some(val);
        }
        let parent = self.parent.as_ref()?;
        // Parent constructor cache entries are invalid if this scope shadows
        // any source they depend on.
        if !self.introduced_sources.is_disjoint(source_deps) {
            return None;
        }
        parent.get_cached(node_id, source_deps)
    }

    pub(crate) fn insert_cached(&mut self, node_id: ExecutionNodeId, value: Py<PyAny>) {
        self.cached_values.insert(node_id, value);
    }

    pub(crate) fn traverse_py_refs(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for val in self.source_values.values() {
            visit.call(val)?;
        }
        for val in self.cached_values.values() {
            visit.call(val)?;
        }
        Ok(())
    }
}
