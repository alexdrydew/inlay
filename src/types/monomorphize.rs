use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::time::{Duration, Instant};

use crate::qualifier::Qualifier;

use super::{
    Arena, ArenaFamily, Bindings, MapChildren, OpaqueParamSpec, OpaqueTypeVar, PyType,
    PyTypeConcreteKey, PyTypeParametricKey, Qualified, TypeArenas, TypeChildren,
};

pub(crate) struct MonomorphizationStats {
    enabled: bool,
    pub(crate) apply_bindings_calls: u64,
    pub(crate) apply_bindings_repeat_calls: u64,
    pub(crate) apply_bindings_time: Duration,
    pub(crate) apply_bindings_canonical_reuse_hits: u64,
    apply_bindings_signatures: HashMap<u64, u64>,
    apply_bindings_labels: HashMap<u64, String>,
    pub(crate) requalify_calls: u64,
    pub(crate) requalify_repeat_calls: u64,
    pub(crate) requalify_time: Duration,
    pub(crate) requalify_canonical_reuse_hits: u64,
    requalify_signatures: HashMap<u64, u64>,
    requalify_labels: HashMap<u64, String>,
}

impl Default for MonomorphizationStats {
    fn default() -> Self {
        Self {
            enabled: std::env::var_os("INLAY_MONOMORPH_STATS").is_some(),
            apply_bindings_calls: 0,
            apply_bindings_repeat_calls: 0,
            apply_bindings_time: Duration::default(),
            apply_bindings_canonical_reuse_hits: 0,
            apply_bindings_signatures: HashMap::new(),
            apply_bindings_labels: HashMap::new(),
            requalify_calls: 0,
            requalify_repeat_calls: 0,
            requalify_time: Duration::default(),
            requalify_canonical_reuse_hits: 0,
            requalify_signatures: HashMap::new(),
            requalify_labels: HashMap::new(),
        }
    }
}

impl MonomorphizationStats {
    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn apply_bindings_unique_signatures(&self) -> usize {
        self.apply_bindings_signatures.len()
    }

    pub(crate) fn requalify_unique_signatures(&self) -> usize {
        self.requalify_signatures.len()
    }

    pub(crate) fn top_apply_bindings_labels(&self, limit: usize) -> Vec<(String, u64)> {
        top_signature_labels(
            &self.apply_bindings_signatures,
            &self.apply_bindings_labels,
            limit,
        )
    }

    pub(crate) fn top_requalify_labels(&self, limit: usize) -> Vec<(String, u64)> {
        top_signature_labels(&self.requalify_signatures, &self.requalify_labels, limit)
    }

    fn record_apply_bindings_request(&mut self, signature: u64, label: String) {
        if !self.enabled {
            return;
        }
        self.apply_bindings_calls += 1;
        let count = self.apply_bindings_signatures.entry(signature).or_default();
        if *count > 0 {
            self.apply_bindings_repeat_calls += 1;
        }
        *count += 1;
        self.apply_bindings_labels.entry(signature).or_insert(label);
    }

    fn record_requalify_request(&mut self, signature: u64, label: String) {
        if !self.enabled {
            return;
        }
        self.requalify_calls += 1;
        let count = self.requalify_signatures.entry(signature).or_default();
        if *count > 0 {
            self.requalify_repeat_calls += 1;
        }
        *count += 1;
        self.requalify_labels.entry(signature).or_insert(label);
    }

    fn record_apply_bindings_time(&mut self, elapsed: Duration) {
        if self.enabled {
            self.apply_bindings_time += elapsed;
        }
    }

    fn record_requalify_time(&mut self, elapsed: Duration) {
        if self.enabled {
            self.requalify_time += elapsed;
        }
    }
}

#[derive(Clone, Copy)]
enum CanonicalizationContext {
    ApplyBindings,
    Requalify,
}

fn top_signature_labels(
    counts: &HashMap<u64, u64>,
    labels: &HashMap<u64, String>,
    limit: usize,
) -> Vec<(String, u64)> {
    let mut entries = counts
        .iter()
        .map(|(signature, count)| {
            (
                labels
                    .get(signature)
                    .cloned()
                    .unwrap_or_else(|| format!("signature={signature:016x}")),
                *count,
            )
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    entries.truncate(limit);
    entries
}

fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(value, &mut hasher);
    std::hash::Hasher::finish(&hasher)
}

fn apply_bindings_signature<S: ArenaFamily>(
    source: PyTypeParametricKey<S>,
    bindings: &Bindings<S>,
) -> u64 {
    let mut type_vars = bindings
        .type_vars
        .iter()
        .map(|(id, key)| (id.clone(), *key))
        .collect::<Vec<_>>();
    type_vars.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    let mut param_specs = bindings
        .param_specs
        .iter()
        .map(|(id, key)| (id.clone(), *key))
        .collect::<Vec<_>>();
    param_specs.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    hash_value(&(source, type_vars, param_specs))
}

fn requalify_signature<S: ArenaFamily>(
    target: PyTypeConcreteKey<S>,
    additional: &Qualifier,
) -> u64 {
    hash_value(&(target, additional))
}

fn parametric_root_label<S: ArenaFamily>(
    source: PyTypeParametricKey<S>,
    bindings: &Bindings<S>,
    arenas: &TypeArenas<S>,
) -> String {
    let root = match source {
        PyType::Sentinel(_) => "Sentinel".to_string(),
        PyType::TypeVar(key) => arenas
            .parametric
            .type_vars
            .get(&key)
            .map(|value| format!("TypeVar {}", value.inner.descriptor.display_name.as_ref()))
            .unwrap_or_else(|| "TypeVar <dangling>".to_string()),
        PyType::ParamSpec(key) => arenas
            .parametric
            .param_specs
            .get(&key)
            .map(|value| format!("ParamSpec {}", value.inner.descriptor.display_name.as_ref()))
            .unwrap_or_else(|| "ParamSpec <dangling>".to_string()),
        PyType::Plain(key) => arenas
            .parametric
            .plains
            .get(&key)
            .map(|value| format!("Plain {}", value.inner.descriptor.display_name.as_ref()))
            .unwrap_or_else(|| "Plain <dangling>".to_string()),
        PyType::Protocol(key) => arenas
            .parametric
            .protocols
            .get(&key)
            .map(|value| format!("Protocol {}", value.inner.descriptor.display_name.as_ref()))
            .unwrap_or_else(|| "Protocol <dangling>".to_string()),
        PyType::TypedDict(key) => arenas
            .parametric
            .typed_dicts
            .get(&key)
            .map(|value| format!("TypedDict {}", value.inner.descriptor.display_name.as_ref()))
            .unwrap_or_else(|| "TypedDict <dangling>".to_string()),
        PyType::Union(_) => "Union".to_string(),
        PyType::Callable(_) => "Callable".to_string(),
        PyType::LazyRef(_) => "LazyRef".to_string(),
    };
    format!(
        "{} tv={} ps={}",
        root,
        bindings.type_vars.len(),
        bindings.param_specs.len()
    )
}

fn concrete_root_label<S: ArenaFamily>(
    target: PyTypeConcreteKey<S>,
    additional: &Qualifier,
    arenas: &TypeArenas<S>,
) -> String {
    let root = match target {
        PyType::Sentinel(_) => "Sentinel".to_string(),
        PyType::TypeVar(key) => arenas
            .concrete
            .type_vars
            .get(&key)
            .map(|value| format!("TypeVar {}", value.inner.descriptor.display_name.as_ref()))
            .unwrap_or_else(|| "TypeVar <dangling>".to_string()),
        PyType::ParamSpec(key) => arenas
            .concrete
            .param_specs
            .get(&key)
            .map(|value| format!("ParamSpec {}", value.inner.descriptor.display_name.as_ref()))
            .unwrap_or_else(|| "ParamSpec <dangling>".to_string()),
        PyType::Plain(key) => arenas
            .concrete
            .plains
            .get(&key)
            .map(|value| format!("Plain {}", value.inner.descriptor.display_name.as_ref()))
            .unwrap_or_else(|| "Plain <dangling>".to_string()),
        PyType::Protocol(key) => arenas
            .concrete
            .protocols
            .get(&key)
            .map(|value| format!("Protocol {}", value.inner.descriptor.display_name.as_ref()))
            .unwrap_or_else(|| "Protocol <dangling>".to_string()),
        PyType::TypedDict(key) => arenas
            .concrete
            .typed_dicts
            .get(&key)
            .map(|value| format!("TypedDict {}", value.inner.descriptor.display_name.as_ref()))
            .unwrap_or_else(|| "TypedDict <dangling>".to_string()),
        PyType::Union(_) => "Union".to_string(),
        PyType::Callable(_) => "Callable".to_string(),
        PyType::LazyRef(_) => "LazyRef".to_string(),
    };
    format!("{} -> {}", root, additional.display_compact())
}

// --- TypeArenas method ---

impl<S: ArenaFamily> TypeArenas<S> {
    pub(crate) fn apply_bindings(
        &mut self,
        source: PyTypeParametricKey<S>,
        bindings: &Bindings<S>,
    ) -> PyTypeConcreteKey<S> {
        let started = Instant::now();
        if self.monomorphization_stats.enabled() {
            self.monomorphization_stats.record_apply_bindings_request(
                apply_bindings_signature(source, bindings),
                parametric_root_label(source, bindings, self),
            );
        }
        let result = apply_bindings_inner(source, bindings, self, &mut HashMap::new());
        self.monomorphization_stats
            .record_apply_bindings_time(started.elapsed());
        result
    }

    fn canonicalize_concrete(&mut self, key: PyTypeConcreteKey<S>) -> PyTypeConcreteKey<S> {
        let mut canonical_concrete = std::mem::take(&mut self.canonical_concrete_qualified);
        let canonical = canonical_concrete
            .get(key, self)
            .copied()
            .unwrap_or_else(|| {
                canonical_concrete.insert(key, key, self);
                key
            });
        self.canonical_concrete_qualified = canonical_concrete;
        canonical
    }
}

fn value_has_unresolved_children<S: ArenaFamily, T>(
    value: &Qualified<T>,
    arenas: &TypeArenas<S>,
    visited: &mut HashSet<PyTypeConcreteKey<S>>,
) -> bool
where
    Qualified<T>: TypeChildren<PyTypeConcreteKey<S>>,
{
    value
        .children()
        .copied()
        .any(|child| key_has_unresolved_placeholder(child, arenas, visited))
}

fn key_has_unresolved_placeholder<S: ArenaFamily>(
    key: PyTypeConcreteKey<S>,
    arenas: &TypeArenas<S>,
    visited: &mut HashSet<PyTypeConcreteKey<S>>,
) -> bool {
    if !visited.insert(key) {
        return false;
    }

    let unresolved = match key {
        PyType::Sentinel(key) => arenas.sentinels.get(&key).is_none(),
        PyType::ParamSpec(key) => arenas
            .concrete
            .param_specs
            .get(&key)
            .is_none_or(|value| value_has_unresolved_children(value, arenas, visited)),
        PyType::Plain(key) => arenas
            .concrete
            .plains
            .get(&key)
            .is_none_or(|value| value_has_unresolved_children(value, arenas, visited)),
        PyType::Protocol(key) => arenas
            .concrete
            .protocols
            .get(&key)
            .is_none_or(|value| value_has_unresolved_children(value, arenas, visited)),
        PyType::TypedDict(key) => arenas
            .concrete
            .typed_dicts
            .get(&key)
            .is_none_or(|value| value_has_unresolved_children(value, arenas, visited)),
        PyType::Union(key) => arenas
            .concrete
            .unions
            .get(&key)
            .is_none_or(|value| value_has_unresolved_children(value, arenas, visited)),
        PyType::Callable(key) => arenas
            .concrete
            .callables
            .get(&key)
            .is_none_or(|value| value_has_unresolved_children(value, arenas, visited)),
        PyType::LazyRef(key) => arenas
            .concrete
            .lazy_refs
            .get(&key)
            .is_none_or(|value| value_has_unresolved_children(value, arenas, visited)),
        PyType::TypeVar(key) => arenas
            .concrete
            .type_vars
            .get(&key)
            .is_none_or(|value| value_has_unresolved_children(value, arenas, visited)),
    };

    visited.remove(&key);
    unresolved
}

fn canonicalize_if_resolved<S: ArenaFamily>(
    key: PyTypeConcreteKey<S>,
    arenas: &mut TypeArenas<S>,
    context: CanonicalizationContext,
) -> PyTypeConcreteKey<S> {
    if key_has_unresolved_placeholder(key, arenas, &mut HashSet::new()) {
        return key;
    }
    let canonical = arenas.canonicalize_concrete(key);
    if canonical != key && arenas.monomorphization_stats.enabled() {
        match context {
            CanonicalizationContext::ApplyBindings => {
                arenas
                    .monomorphization_stats
                    .apply_bindings_canonical_reuse_hits += 1;
            }
            CanonicalizationContext::Requalify => {
                arenas.monomorphization_stats.requalify_canonical_reuse_hits += 1;
            }
        }
    }
    canonical
}

fn apply_bindings_inner<S: ArenaFamily>(
    source: PyTypeParametricKey<S>,
    bindings: &Bindings<S>,
    arenas: &mut TypeArenas<S>,
    memo: &mut HashMap<PyTypeParametricKey<S>, PyTypeConcreteKey<S>>,
) -> PyTypeConcreteKey<S> {
    if let Some(&cached) = memo.get(&source) {
        return cached;
    }

    let result = match source {
        // Binding lookup uses Python TypeVar identity (PyTypeId), not arena
        // slot key. The same logical TypeVar may occupy different slots due to
        // different qualifier contexts (return type vs params after include).
        PyType::TypeVar(key) => {
            let tv = arenas.parametric.type_vars.get(&key).expect("dangling key");
            let qualifier = tv.qualifier.clone();
            match bindings.type_vars.get(&tv.inner.descriptor.id) {
                Some(&bound) => {
                    if qualifier.is_unqualified() {
                        bound
                    } else {
                        requalify_concrete(bound, &qualifier, arenas)
                    }
                }
                None => {
                    let opaque = Qualified {
                        inner: OpaqueTypeVar {
                            descriptor: tv.inner.descriptor.clone(),
                        },
                        qualifier: tv.qualifier.clone(),
                    };
                    PyType::TypeVar(arenas.concrete.type_vars.insert(opaque))
                }
            }
        }
        PyType::ParamSpec(key) => {
            let ps = arenas
                .parametric
                .param_specs
                .get(&key)
                .expect("dangling key");
            let qualifier = ps.qualifier.clone();
            match bindings.param_specs.get(&ps.inner.descriptor.id) {
                Some(&bound) => {
                    if qualifier.is_unqualified() {
                        bound
                    } else {
                        requalify_concrete(bound, &qualifier, arenas)
                    }
                }
                None => {
                    let opaque = Qualified {
                        inner: OpaqueParamSpec {
                            descriptor: ps.inner.descriptor.clone(),
                        },
                        qualifier: ps.qualifier.clone(),
                    };
                    PyType::ParamSpec(arenas.concrete.param_specs.insert(opaque))
                }
            }
        }
        PyType::Sentinel(key) => PyType::Sentinel(key),
        PyType::Plain(key) => {
            let placeholder = arenas.concrete.plains.insert_placeholder();
            let result = PyType::Plain(placeholder);
            memo.insert(source, result);
            let val = arenas
                .parametric
                .plains
                .get(&key)
                .expect("dangling key")
                .clone();
            let output = val.map_children(&mut |k| apply_bindings_inner(k, bindings, arenas, memo));
            assert!(
                arenas
                    .concrete
                    .plains
                    .replace(placeholder, output)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::Protocol(key) => {
            let placeholder = arenas.concrete.protocols.insert_placeholder();
            let result = PyType::Protocol(placeholder);
            memo.insert(source, result);
            let val = arenas
                .parametric
                .protocols
                .get(&key)
                .expect("dangling key")
                .clone();
            let output = val.map_children(&mut |k| apply_bindings_inner(k, bindings, arenas, memo));
            assert!(
                arenas
                    .concrete
                    .protocols
                    .replace(placeholder, output)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::TypedDict(key) => {
            let placeholder = arenas.concrete.typed_dicts.insert_placeholder();
            let result = PyType::TypedDict(placeholder);
            memo.insert(source, result);
            let val = arenas
                .parametric
                .typed_dicts
                .get(&key)
                .expect("dangling key")
                .clone();
            let output = val.map_children(&mut |k| apply_bindings_inner(k, bindings, arenas, memo));
            assert!(
                arenas
                    .concrete
                    .typed_dicts
                    .replace(placeholder, output)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::Union(key) => {
            let placeholder = arenas.concrete.unions.insert_placeholder();
            let result = PyType::Union(placeholder);
            memo.insert(source, result);
            let val = arenas
                .parametric
                .unions
                .get(&key)
                .expect("dangling key")
                .clone();
            let output = val.map_children(&mut |k| apply_bindings_inner(k, bindings, arenas, memo));
            assert!(
                arenas
                    .concrete
                    .unions
                    .replace(placeholder, output)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::Callable(key) => {
            let placeholder = arenas.concrete.callables.insert_placeholder();
            let result = PyType::Callable(placeholder);
            memo.insert(source, result);
            let val = arenas
                .parametric
                .callables
                .get(&key)
                .expect("dangling key")
                .clone();
            let output = val.map_children(&mut |k| apply_bindings_inner(k, bindings, arenas, memo));
            assert!(
                arenas
                    .concrete
                    .callables
                    .replace(placeholder, output)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::LazyRef(key) => {
            let placeholder = arenas.concrete.lazy_refs.insert_placeholder();
            let result = PyType::LazyRef(placeholder);
            memo.insert(source, result);
            let val = arenas
                .parametric
                .lazy_refs
                .get(&key)
                .expect("dangling key")
                .clone();
            let output = val.map_children(&mut |k| apply_bindings_inner(k, bindings, arenas, memo));
            assert!(
                arenas
                    .concrete
                    .lazy_refs
                    .replace(placeholder, output)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
    };

    let result = canonicalize_if_resolved(result, arenas, CanonicalizationContext::ApplyBindings);
    memo.insert(source, result);
    result
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use super::*;
    use crate::types::{PlainType, PyTypeDescriptor, PyTypeId, SlotBackend};

    fn duplicate_plain_key(
        arenas: &mut TypeArenas<SlotBackend>,
        descriptor: &PyTypeDescriptor,
        args: Vec<PyTypeConcreteKey<SlotBackend>>,
    ) -> PyTypeConcreteKey<SlotBackend> {
        let placeholder = arenas.concrete.plains.insert_placeholder();
        let replaced = arenas.concrete.plains.replace(
            placeholder,
            Qualified {
                inner: PlainType {
                    descriptor: descriptor.clone(),
                    args,
                },
                qualifier: Qualifier::any(),
            },
        );
        assert!(
            replaced.expect("placeholder key should exist").is_none(),
            "placeholder key already filled"
        );
        PyType::Plain(placeholder)
    }

    fn qualifier(tag: &str) -> Qualifier {
        let mut alternative = BTreeSet::new();
        alternative.insert(tag.to_string());
        let mut alternatives = BTreeSet::new();
        alternatives.insert(alternative);
        Qualifier::from(alternatives)
    }

    #[test]
    fn apply_bindings_reuses_structurally_equal_concrete_keys() {
        let mut arenas = TypeArenas::<SlotBackend>::default();
        let source = PyType::Plain(arenas.parametric.plains.insert(Qualified {
            inner: PlainType {
                descriptor: PyTypeDescriptor {
                    id: PyTypeId::new("builtins.int".to_string()),
                    display_name: Arc::from("int"),
                },
                args: vec![],
            },
            qualifier: Qualifier::any(),
        }));

        let first = arenas.apply_bindings(source, &Bindings::default());
        let second = arenas.apply_bindings(source, &Bindings::default());

        assert!(first == second);
    }

    #[test]
    fn requalify_concrete_reuses_structurally_equal_nested_keys() {
        let mut arenas = TypeArenas::<SlotBackend>::default();
        let child_descriptor = PyTypeDescriptor {
            id: PyTypeId::new("bench.ChildState".to_string()),
            display_name: Arc::from("ChildState"),
        };
        let parent_descriptor = PyTypeDescriptor {
            id: PyTypeId::new("bench.WriteState".to_string()),
            display_name: Arc::from("WriteState"),
        };

        let child_a = duplicate_plain_key(&mut arenas, &child_descriptor, vec![]);
        let child_b = duplicate_plain_key(&mut arenas, &child_descriptor, vec![]);
        let parent_a = duplicate_plain_key(&mut arenas, &parent_descriptor, vec![child_a]);
        let parent_b = duplicate_plain_key(&mut arenas, &parent_descriptor, vec![child_b]);

        let first = requalify_concrete(parent_a, &qualifier("write"), &mut arenas);
        let second = requalify_concrete(parent_b, &qualifier("write"), &mut arenas);

        assert!(first == second);
    }

    #[test]
    fn requalify_concrete_requalifies_nested_children() {
        let mut arenas = TypeArenas::<SlotBackend>::default();
        let child_descriptor = PyTypeDescriptor {
            id: PyTypeId::new("bench.VectorClock".to_string()),
            display_name: Arc::from("VectorClock"),
        };
        let parent_descriptor = PyTypeDescriptor {
            id: PyTypeId::new("bench.Constants".to_string()),
            display_name: Arc::from("Constants"),
        };

        let child = duplicate_plain_key(&mut arenas, &child_descriptor, vec![]);
        let parent = duplicate_plain_key(&mut arenas, &parent_descriptor, vec![child]);

        let requalified = requalify_concrete(parent, &qualifier("write"), &mut arenas);
        let PyType::Plain(parent_key) = requalified else {
            panic!("expected plain requalified parent");
        };
        let parent_value = arenas
            .concrete
            .plains
            .get(&parent_key)
            .expect("dangling parent key");
        let child_key = parent_value.inner.args[0];

        assert_eq!(
            arenas
                .qualifier_of_concrete(child_key)
                .expect("child qualifier must exist")
                .display_compact(),
            "EMPTY | write"
        );
    }
}

fn reinsert_requalified<T, A>(store: &mut A, key: A::Key, additional: &Qualifier) -> A::Key
where
    T: Hash + Eq + Clone + 'static,
    A: Arena<Qualified<T>>,
{
    let (inner, new_qual) = {
        let val = store.get(&key).expect("dangling key");
        let new_qual = requalified_qualifier(&val.qualifier, additional);
        if new_qual == val.qualifier {
            return key;
        }
        (val.inner.clone(), new_qual)
    };
    store.insert(Qualified {
        inner,
        qualifier: new_qual,
    })
}

fn requalified_qualifier(current: &Qualifier, additional: &Qualifier) -> Qualifier {
    if current.is_any() {
        return additional.with_base_scope_if_write();
    }
    current.intersect(additional)
}

fn requalify_concrete_inner<S: ArenaFamily>(
    target: PyTypeConcreteKey<S>,
    additional: &Qualifier,
    arenas: &mut TypeArenas<S>,
    memo: &mut HashMap<PyTypeConcreteKey<S>, PyTypeConcreteKey<S>>,
) -> PyTypeConcreteKey<S> {
    if let Some(&cached) = memo.get(&target) {
        return cached;
    }

    let result = match target {
        PyType::Sentinel(_) => target,
        PyType::TypeVar(key) => PyType::TypeVar(reinsert_requalified(
            &mut arenas.concrete.type_vars,
            key,
            additional,
        )),
        PyType::ParamSpec(key) => PyType::ParamSpec(reinsert_requalified(
            &mut arenas.concrete.param_specs,
            key,
            additional,
        )),
        PyType::Plain(key) => {
            let placeholder = arenas.concrete.plains.insert_placeholder();
            let result = PyType::Plain(placeholder);
            memo.insert(target, result);
            let value = arenas
                .concrete
                .plains
                .get(&key)
                .expect("dangling key")
                .clone();
            let output = Qualified {
                inner: super::PlainType {
                    descriptor: value.inner.descriptor,
                    args: value
                        .inner
                        .args
                        .into_iter()
                        .map(|child| requalify_concrete_inner(child, additional, arenas, memo))
                        .collect(),
                },
                qualifier: requalified_qualifier(&value.qualifier, additional),
            };
            assert!(
                arenas
                    .concrete
                    .plains
                    .replace(placeholder, output)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::Protocol(key) => {
            let placeholder = arenas.concrete.protocols.insert_placeholder();
            let result = PyType::Protocol(placeholder);
            memo.insert(target, result);
            let value = arenas
                .concrete
                .protocols
                .get(&key)
                .expect("dangling key")
                .clone();
            let output = Qualified {
                inner: super::ProtocolType {
                    descriptor: value.inner.descriptor,
                    methods: value
                        .inner
                        .methods
                        .into_iter()
                        .map(|(name, child)| {
                            (
                                name,
                                requalify_concrete_inner(child, additional, arenas, memo),
                            )
                        })
                        .collect(),
                    attributes: value
                        .inner
                        .attributes
                        .into_iter()
                        .map(|(name, child)| {
                            (
                                name,
                                requalify_concrete_inner(child, additional, arenas, memo),
                            )
                        })
                        .collect(),
                    properties: value
                        .inner
                        .properties
                        .into_iter()
                        .map(|(name, child)| {
                            (
                                name,
                                requalify_concrete_inner(child, additional, arenas, memo),
                            )
                        })
                        .collect(),
                    type_params: value
                        .inner
                        .type_params
                        .into_iter()
                        .map(|child| requalify_concrete_inner(child, additional, arenas, memo))
                        .collect(),
                },
                qualifier: requalified_qualifier(&value.qualifier, additional),
            };
            assert!(
                arenas
                    .concrete
                    .protocols
                    .replace(placeholder, output)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::TypedDict(key) => {
            let placeholder = arenas.concrete.typed_dicts.insert_placeholder();
            let result = PyType::TypedDict(placeholder);
            memo.insert(target, result);
            let value = arenas
                .concrete
                .typed_dicts
                .get(&key)
                .expect("dangling key")
                .clone();
            let output = Qualified {
                inner: super::TypedDictType {
                    descriptor: value.inner.descriptor,
                    attributes: value
                        .inner
                        .attributes
                        .into_iter()
                        .map(|(name, child)| {
                            (
                                name,
                                requalify_concrete_inner(child, additional, arenas, memo),
                            )
                        })
                        .collect(),
                    type_params: value
                        .inner
                        .type_params
                        .into_iter()
                        .map(|child| requalify_concrete_inner(child, additional, arenas, memo))
                        .collect(),
                },
                qualifier: requalified_qualifier(&value.qualifier, additional),
            };
            assert!(
                arenas
                    .concrete
                    .typed_dicts
                    .replace(placeholder, output)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::Union(key) => {
            let placeholder = arenas.concrete.unions.insert_placeholder();
            let result = PyType::Union(placeholder);
            memo.insert(target, result);
            let value = arenas
                .concrete
                .unions
                .get(&key)
                .expect("dangling key")
                .clone();
            let output = Qualified {
                inner: super::UnionType {
                    variants: value
                        .inner
                        .variants
                        .into_iter()
                        .map(|child| requalify_concrete_inner(child, additional, arenas, memo))
                        .collect(),
                },
                qualifier: requalified_qualifier(&value.qualifier, additional),
            };
            assert!(
                arenas
                    .concrete
                    .unions
                    .replace(placeholder, output)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::Callable(key) => {
            let placeholder = arenas.concrete.callables.insert_placeholder();
            let result = PyType::Callable(placeholder);
            memo.insert(target, result);
            let value = arenas
                .concrete
                .callables
                .get(&key)
                .expect("dangling key")
                .clone();
            let output = Qualified {
                inner: super::CallableType {
                    params: value
                        .inner
                        .params
                        .into_iter()
                        .map(|(name, child)| {
                            (
                                name,
                                requalify_concrete_inner(child, additional, arenas, memo),
                            )
                        })
                        .collect(),
                    param_kinds: value.inner.param_kinds,
                    param_has_default: value.inner.param_has_default,
                    return_type: requalify_concrete_inner(
                        value.inner.return_type,
                        additional,
                        arenas,
                        memo,
                    ),
                    return_wrapper: value.inner.return_wrapper,
                    type_params: value
                        .inner
                        .type_params
                        .into_iter()
                        .map(|child| requalify_concrete_inner(child, additional, arenas, memo))
                        .collect(),
                    function_name: value.inner.function_name,
                },
                qualifier: requalified_qualifier(&value.qualifier, additional),
            };
            assert!(
                arenas
                    .concrete
                    .callables
                    .replace(placeholder, output)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
        PyType::LazyRef(key) => {
            let placeholder = arenas.concrete.lazy_refs.insert_placeholder();
            let result = PyType::LazyRef(placeholder);
            memo.insert(target, result);
            let value = arenas
                .concrete
                .lazy_refs
                .get(&key)
                .expect("dangling key")
                .clone();
            let output = Qualified {
                inner: super::LazyRefType {
                    target: requalify_concrete_inner(value.inner.target, additional, arenas, memo),
                },
                qualifier: requalified_qualifier(&value.qualifier, additional),
            };
            assert!(
                arenas
                    .concrete
                    .lazy_refs
                    .replace(placeholder, output)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            result
        }
    };

    let result = canonicalize_if_resolved(result, arenas, CanonicalizationContext::Requalify);
    memo.insert(target, result);
    result
}

pub(crate) fn requalify_concrete<S: ArenaFamily>(
    target: PyTypeConcreteKey<S>,
    additional: &Qualifier,
    arenas: &mut TypeArenas<S>,
) -> PyTypeConcreteKey<S> {
    let started = Instant::now();
    if arenas.monomorphization_stats.enabled() {
        arenas.monomorphization_stats.record_requalify_request(
            requalify_signature(target, additional),
            concrete_root_label(target, additional, arenas),
        );
    }
    let result = requalify_concrete_inner(target, additional, arenas, &mut HashMap::new());
    arenas
        .monomorphization_stats
        .record_requalify_time(started.elapsed());
    result
}
