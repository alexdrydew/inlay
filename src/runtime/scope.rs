use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::PyTraverseError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;

use crate::compile::flatten::{ExecutionNodeId, ExecutionSourceNodeId};

/// Runtime execution scope with hierarchical cache.
///
/// Each scope holds:
/// - Source-bound values and cached constructor results keyed by canonical `ExecutionNodeId`
/// - A set of source nodes introduced in this scope
/// - An optional parent scope
///
/// Lookup for cached nodes checks the source set: if any of the node's
/// source dependencies were introduced in this scope, the parent's cached
/// value is NOT inherited — the node must be rebuilt.
pub(crate) struct Scope {
    parent: Option<Arc<Scope>>,
    values: HashMap<ExecutionNodeId, Py<PyAny>>,
    introduced_sources: HashSet<ExecutionSourceNodeId>,
}

impl Scope {
    /// Create a root scope with initial value bindings.
    pub(crate) fn root(values: HashMap<ExecutionNodeId, Py<PyAny>>) -> Self {
        Self {
            parent: None,
            values,
            introduced_sources: HashSet::new(),
        }
    }

    /// Create a child scope from a frozen parent, adding new source bindings.
    pub(crate) fn child(
        parent: Arc<Scope>,
        new_sources: Vec<(ExecutionSourceNodeId, Py<PyAny>)>,
    ) -> Self {
        let mut introduced_sources = HashSet::with_capacity(new_sources.len());
        let mut values = HashMap::with_capacity(new_sources.len());

        for (source, value) in new_sources {
            introduced_sources.insert(source);
            values.insert(source.node_id(), value);
        }

        Self {
            parent: Some(parent),
            values,
            introduced_sources,
        }
    }

    /// Look up a value by node id, walking up the scope chain.
    pub(crate) fn get_value(&self, node_id: ExecutionNodeId) -> Option<&Py<PyAny>> {
        self.values
            .get(&node_id)
            .or_else(|| self.parent.as_ref()?.get_value(node_id))
    }

    /// Look up a cached constructor result for a node.
    ///
    /// Returns `None` if:
    /// - The result is not cached locally AND
    ///   - Any of the node's source deps were introduced in this scope, OR
    ///   - The parent doesn't have it either
    pub(crate) fn get_cached(
        &self,
        node_id: ExecutionNodeId,
        source_deps: &HashSet<ExecutionSourceNodeId>,
    ) -> Option<&Py<PyAny>> {
        if let Some(val) = self.values.get(&node_id) {
            return Some(val);
        }
        let parent = self.parent.as_ref()?;
        if !self.introduced_sources.is_disjoint(source_deps) {
            return None;
        }
        parent.get_cached(node_id, source_deps)
    }

    /// Store a constructor result in this scope's local cache.
    pub(crate) fn insert_cached(&mut self, node_id: ExecutionNodeId, value: Py<PyAny>) {
        self.values.insert(node_id, value);
    }

    /// Visit all Python references held locally by this scope.
    ///
    /// Only traverses this scope's own value map —
    /// parent scopes are NOT traversed (each parent's own `ContextProxy`
    /// handles its own GC cycle independently).
    pub(crate) fn traverse_py_refs(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for val in self.values.values() {
            visit.call(val)?;
        }
        Ok(())
    }
}
