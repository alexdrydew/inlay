use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::PyTraverseError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;

use crate::compile::flatten::ExecutionCacheKey;
use crate::registry::Source;

/// Runtime execution scope with hierarchical cache.
///
/// Each scope holds:
/// - Source-bound values introduced in this scope (keyed by exact Source)
/// - Computed node results (keyed by ConcreteRef / target_type)
/// - A set of sources introduced in this scope
/// - An optional parent scope
///
/// Lookup for computed nodes checks the source set: if any of the node's
/// source dependencies were introduced in this scope, the parent's cached
/// value is NOT inherited — the node must be rebuilt.
pub(crate) struct Scope {
    parent: Option<Arc<Scope>>,
    sources: HashMap<Source, Py<PyAny>>,
    computed: HashMap<ExecutionCacheKey, Py<PyAny>>,
    introduced_sources: HashSet<Source>,
}

impl Scope {
    /// Create a root scope with initial source bindings.
    pub(crate) fn root(sources: HashMap<Source, Py<PyAny>>) -> Self {
        Self {
            parent: None,
            sources,
            computed: HashMap::new(),
            introduced_sources: HashSet::new(),
        }
    }

    /// Create a child scope from a frozen parent, adding new source bindings.
    pub(crate) fn child(parent: Arc<Scope>, new_sources: Vec<(Source, Py<PyAny>)>) -> Self {
        let mut introduced_sources = HashSet::with_capacity(new_sources.len());
        let mut sources = HashMap::with_capacity(new_sources.len());

        for (source, value) in new_sources {
            introduced_sources.insert(source.clone());
            sources.insert(source, value);
        }

        Self {
            parent: Some(parent),
            sources,
            computed: HashMap::new(),
            introduced_sources,
        }
    }

    /// Look up a source-bound value, walking up the scope chain.
    pub(crate) fn get_source(&self, source: &Source) -> Option<&Py<PyAny>> {
        self.sources
            .get(source)
            .or_else(|| self.parent.as_ref()?.get_source(source))
    }

    /// Look up a cached computed result for a node.
    ///
    /// Returns `None` if:
    /// - The result is not cached locally AND
    ///   - Any of the node's source deps were introduced in this scope, OR
    ///   - The parent doesn't have it either
    pub(crate) fn get_computed(
        &self,
        cache_key: &ExecutionCacheKey,
        source_deps: &HashSet<Source>,
    ) -> Option<&Py<PyAny>> {
        if let Some(val) = self.computed.get(cache_key) {
            return Some(val);
        }
        let parent = self.parent.as_ref()?;
        if !self.introduced_sources.is_disjoint(source_deps) {
            return None;
        }
        parent.get_computed(cache_key, source_deps)
    }

    /// Store a computed result in this scope's local cache.
    pub(crate) fn insert_computed(&mut self, cache_key: ExecutionCacheKey, value: Py<PyAny>) {
        self.computed.insert(cache_key, value);
    }

    /// Visit all Python references held locally by this scope.
    ///
    /// Only traverses this scope's own `computed` and `constants` maps —
    /// parent scopes are NOT traversed (each parent's own `ContextProxy`
    /// handles its own GC cycle independently).
    pub(crate) fn traverse_py_refs(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for val in self.computed.values() {
            visit.call(val)?;
        }
        for val in self.sources.values() {
            visit.call(val)?;
        }
        Ok(())
    }
}
