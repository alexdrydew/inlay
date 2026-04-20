use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

use context_solver::rule::ResolutionEnv;
use derive_where::derive_where;

use crate::qualifier::{qualifier_matches, Qualifier};
use crate::registry::{to_constant_type, Hook, MethodImplementation, TransitionBindingKey};
use crate::rules::display_concrete_ref;
use crate::types::TypeArenas;
use crate::{
    registry::{ConstantType, Constructor, Source, SourceKind, SourceType},
    types::{
        Arena, ArenaFamily, Bindings, CallableKey, Concrete, Parametric, ProtocolKey, PyType,
        PyTypeConcreteKey, PyTypeKey, QualifiedMode, ShallowTypeKeyMap, TypeKeyMap, TypeVarSupport,
        TypedDictKey, UnqualifiedMode,
    },
};

#[derive_where(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ConstructorLookup<S: ArenaFamily> {
    pub(crate) constructor: Arc<Constructor<S>>,
    pub(crate) concrete_callable_key: CallableKey<S, Concrete>,
}

#[derive_where(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct MethodLookup<S: ArenaFamily> {
    pub(crate) implementation: Arc<MethodImplementation<S>>,
    pub(crate) concrete_callable_key: CallableKey<S, Concrete>,
    pub(crate) concrete_bound_to: Option<PyTypeConcreteKey<S>>,
}

#[derive_where(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct HookLookup<S: ArenaFamily> {
    pub(crate) hook: Arc<Hook<S>>,
    pub(crate) concrete_callable_key: CallableKey<S, Concrete>,
}

#[derive_where(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Property<S: ArenaFamily, G: TypeVarSupport> {
    pub(crate) name: Arc<str>,
    pub(crate) source_type: ProtocolKey<S, G>,
    pub(crate) member_type: PyTypeKey<S, G>,
    pub(crate) source: Source<S>,
}

struct ParametricProperty<S: ArenaFamily> {
    name: Arc<str>,
    source_type: ProtocolKey<S, Parametric>,
    member_type: PyTypeKey<S, Parametric>,
    source: Source<S>,
}

#[derive_where(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Attribute<S: ArenaFamily, G: TypeVarSupport> {
    pub(crate) name: Arc<str>,
    pub(crate) source_type: SourceType<S, G>,
    pub(crate) member_type: PyTypeKey<S, G>,
    pub(crate) source: Source<S>,
}

struct ParametricAttribute<S: ArenaFamily> {
    name: Arc<str>,
    source_type: SourceType<S, Parametric>,
    member_type: PyTypeKey<S, Parametric>,
    source: Source<S>,
}

type ExactLookupCache<S, T> = TypeKeyMap<S, UnqualifiedMode, Vec<(PyTypeConcreteKey<S>, Vec<T>)>>;
type ParametricPropertyEntry<S> = (
    Arc<str>,
    ProtocolKey<S, Parametric>,
    PyTypeKey<S, Parametric>,
    Source<S>,
);
type ParametricAttributeEntry<S> = (
    Arc<str>,
    SourceType<S, Parametric>,
    PyTypeKey<S, Parametric>,
    Source<S>,
);
type ConstructorsByHeadTypeReturn<S> =
    ShallowTypeKeyMap<S, UnqualifiedMode, Vec<Arc<Constructor<S>>>>;
type ParametricPropertyMap<S> = ShallowTypeKeyMap<S, UnqualifiedMode, Vec<ParametricProperty<S>>>;
type ParametricAttributeMap<S> = ShallowTypeKeyMap<S, UnqualifiedMode, Vec<ParametricAttribute<S>>>;

#[derive(Default)]
struct KindStats {
    calls: u64,
    results: u64,
    errors: u64,
    time: Duration,
}

#[derive(Default)]
struct RegistryStats {
    lookup_by_kind: BTreeMap<&'static str, KindStats>,
    rule_by_kind: BTreeMap<&'static str, KindStats>,
    query_by_key: BTreeMap<String, QueryStats>,
}

#[derive(Default)]
struct QueryStats {
    calls: u64,
    errors: u64,
    env_sizes: BTreeMap<usize, u64>,
    sample_envs: BTreeSet<String>,
}

fn summarize_source_kind<S: ArenaFamily>(
    source_kind: &SourceKind<S>,
    types: &TypeArenas<S>,
) -> String {
    match source_kind {
        SourceKind::ProviderResult(_) => "provider".to_string(),
        SourceKind::TransitionBinding(TransitionBindingKey {
            name,
            constant_type,
        }) => format!(
            "bind:{}:{}",
            name,
            display_concrete_ref(types, (*constant_type).into())
        ),
        SourceKind::TransitionResult(constant_type) => {
            format!(
                "result:{}",
                display_concrete_ref(types, (*constant_type).into())
            )
        }
    }
}

fn summarize_env<S: ArenaFamily>(env: &RegistryEnv<S>, types: &TypeArenas<S>) -> String {
    if env.root_constants.is_empty() {
        return "n=0 []".to_string();
    }

    let mut bindings = env
        .root_constants
        .keys()
        .take(6)
        .map(|source| summarize_source_kind(&source.kind, types))
        .collect::<Vec<_>>();
    if env.root_constants.len() > bindings.len() {
        bindings.push(format!(
            "+{} more",
            env.root_constants.len() - bindings.len()
        ));
    }
    format!("n={} [{}]", env.root_constants.len(), bindings.join(", "))
}

fn get_exact_cached<S: ArenaFamily, T: Clone>(
    cache: &ExactLookupCache<S, T>,
    request: PyTypeConcreteKey<S>,
    types: &mut TypeArenas<S>,
) -> Option<Vec<T>> {
    cache.get(request, types).and_then(|variants| {
        variants
            .iter()
            .find(|(key, _)| types.deep_eq_concrete::<QualifiedMode>(request, *key))
            .map(|(_, cached)| cached.clone())
    })
}

fn cache_exact_lookup<S: ArenaFamily, T: Clone>(
    cache: &mut ExactLookupCache<S, T>,
    request: PyTypeConcreteKey<S>,
    results: &[T],
    types: &mut TypeArenas<S>,
) {
    cache
        .get_or_insert_default(request, types)
        .push((request, results.to_vec()));
}

fn materialize_parametric_matches<S: ArenaFamily, Entry, T>(
    request: PyTypeConcreteKey<S>,
    entries: impl IntoIterator<Item = Entry>,
    types: &mut TypeArenas<S>,
    member_type: impl Fn(&Entry) -> PyTypeKey<S, Parametric>,
    mut materialize: impl FnMut(Entry, Bindings<S>, &mut TypeArenas<S>) -> T,
) -> Vec<T> {
    let mut results = Vec::new();

    for entry in entries {
        if let Ok(bindings) = types.cross_unify(request, member_type(&entry)) {
            results.push(materialize(entry, bindings, types));
        }
    }

    results
}

fn filter_with_matching_qualifiers<S: ArenaFamily, T: Clone>(
    entries: &[T],
    request: PyTypeConcreteKey<S>,
    types: &TypeArenas<S>,
    registered_type: impl Fn(&T, &TypeArenas<S>) -> Option<PyTypeConcreteKey<S>>,
) -> Vec<T> {
    let request_qual = types.qualifier_of_concrete(request).expect("dangling key");

    entries
        .iter()
        .filter(|entry| {
            registered_type(entry, types)
                .and_then(|registered_type| types.qualifier_of_concrete(registered_type))
                .is_some_and(|registration_qual| qualifier_matches(request_qual, registration_qual))
        })
        .cloned()
        .collect()
}

#[derive_where(Default)]
struct RegistryEnvSharedState<S: ArenaFamily> {
    methods_by_name: HashMap<Arc<str>, Vec<Arc<MethodImplementation<S>>>>,
    hooks_by_name: HashMap<Arc<str>, Vec<Arc<Hook<S>>>>,

    constructors_by_head_type_return: ConstructorsByHeadTypeReturn<S>,
    constructors_by_concrete_return: ExactLookupCache<S, ConstructorLookup<S>>,
    methods_by_concrete_request: ExactLookupCache<S, MethodLookup<S>>,
    hooks_by_query: HashMap<(Arc<str>, Option<Qualifier>), Vec<HookLookup<S>>>,

    concrete_properties: ExactLookupCache<S, Property<S, Concrete>>,
    parametric_properties: ParametricPropertyMap<S>,

    concrete_attributes: ExactLookupCache<S, Attribute<S, Concrete>>,
    parametric_attributes: ParametricAttributeMap<S>,
}

#[derive_where(Default)]
struct RegistryEnvLocalState<S: ArenaFamily> {
    unqualified_constants: TypeKeyMap<S, UnqualifiedMode, Vec<(ConstantType<S>, Source<S>)>>,
    unqualified_properties: TypeKeyMap<S, UnqualifiedMode, Vec<Property<S, Concrete>>>,
    unqualified_attributes: TypeKeyMap<S, UnqualifiedMode, Vec<Attribute<S, Concrete>>>,
}

pub(crate) struct RegistrySharedState<S: ArenaFamily> {
    shared: RegistryEnvSharedState<S>,
    env_local_caches: HashMap<Arc<RegistryEnv<S>>, RegistryEnvLocalState<S>>,
    stats: Option<RegistryStats>,
    types: TypeArenas<S>,
}

impl<S: ArenaFamily> RegistrySharedState<S> {
    pub(crate) fn new(
        constructors: &[Arc<Constructor<S>>],
        methods: &[Arc<MethodImplementation<S>>],
        hooks: &[Arc<Hook<S>>],
        mut types: TypeArenas<S>,
    ) -> Self {
        let shared = RegistryEnvSharedState::new(constructors, methods, hooks, &mut types);
        Self {
            shared,
            env_local_caches: HashMap::new(),
            stats: (std::env::var_os("INLAY_COMPILE_STATS").is_some()
                || std::env::var_os("INLAY_QUERY_STATS").is_some())
            .then(RegistryStats::default),
            types,
        }
    }

    pub(crate) fn types(&mut self) -> &mut TypeArenas<S> {
        &mut self.types
    }

    pub(crate) fn into_types(self) -> TypeArenas<S> {
        self.types
    }

    pub(crate) fn record_lookup(
        &mut self,
        kind: &'static str,
        result_count: usize,
        elapsed: Duration,
    ) {
        let Some(stats) = &mut self.stats else {
            return;
        };
        let entry = stats.lookup_by_kind.entry(kind).or_default();
        entry.calls += 1;
        entry.results += result_count as u64;
        entry.time += elapsed;
    }

    pub(crate) fn record_rule_run(&mut self, kind: &'static str, ok: bool, elapsed: Duration) {
        let Some(stats) = &mut self.stats else {
            return;
        };
        let entry = stats.rule_by_kind.entry(kind).or_default();
        entry.calls += 1;
        if !ok {
            entry.errors += 1;
        }
        entry.time += elapsed;
    }

    pub(crate) fn record_query_run(
        &mut self,
        rule_label: &'static str,
        query: PyTypeConcreteKey<S>,
        env: &RegistryEnv<S>,
        ok: bool,
    ) {
        if std::env::var_os("INLAY_QUERY_STATS").is_none() {
            return;
        }

        let query_display = display_concrete_ref(&self.types, query);
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        let query_hash = hasher.finish();
        let env_summary = summarize_env(env, &self.types);
        let entry = self
            .stats
            .as_mut()
            .expect("query stats require registry stats")
            .query_by_key
            .entry(format!(
                "query={query_hash:x} {} {}",
                rule_label, query_display
            ))
            .or_default();
        entry.calls += 1;
        if !ok {
            entry.errors += 1;
        }
        *entry.env_sizes.entry(env.root_constants.len()).or_default() += 1;
        if entry.sample_envs.len() < 6 {
            entry.sample_envs.insert(env_summary);
        }
    }

    pub(crate) fn emit_stats(&self) {
        let Some(stats) = &self.stats else {
            return;
        };

        for (kind, entry) in &stats.lookup_by_kind {
            eprintln!(
                concat!(
                    "[inlay-lookup-stats] ",
                    "kind={} ",
                    "calls={} ",
                    "results={} ",
                    "ms={:.3}"
                ),
                kind,
                entry.calls,
                entry.results,
                entry.time.as_secs_f64() * 1000.0,
            );
        }

        for (kind, entry) in &stats.rule_by_kind {
            eprintln!(
                concat!(
                    "[inlay-rule-stats] ",
                    "kind={} ",
                    "runs={} ",
                    "errors={} ",
                    "ms={:.3}"
                ),
                kind,
                entry.calls,
                entry.errors,
                entry.time.as_secs_f64() * 1000.0,
            );
        }

        if std::env::var_os("INLAY_QUERY_STATS").is_some() {
            let mut query_entries = stats.query_by_key.iter().collect::<Vec<_>>();
            query_entries.sort_by(|left, right| right.1.calls.cmp(&left.1.calls));
            for (key, entry) in query_entries.into_iter().take(25) {
                let env_sizes = entry
                    .env_sizes
                    .iter()
                    .map(|(size, count)| format!("{}:{}", size, count))
                    .collect::<Vec<_>>()
                    .join(",");
                let sample_envs = entry
                    .sample_envs
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" | ");
                eprintln!(
                    concat!(
                        "[inlay-query-stats] ",
                        "key={} ",
                        "calls={} ",
                        "errors={} ",
                        "env_sizes={} ",
                        "samples={}"
                    ),
                    key, entry.calls, entry.errors, env_sizes, sample_envs,
                );
            }
        }
    }
}

fn lookup_result_len<S: ArenaFamily>(result: &ResolutionLookupResult<S>) -> usize {
    match result {
        ResolutionLookupResult::Constants(entries) => entries.len(),
        ResolutionLookupResult::Constructors(entries) => entries.len(),
        ResolutionLookupResult::Methods(entries) => entries.len(),
        ResolutionLookupResult::Hooks(entries) => entries.len(),
        ResolutionLookupResult::Properties(entries) => entries.len(),
        ResolutionLookupResult::Attributes(entries) => entries.len(),
    }
}

impl<S: ArenaFamily> RegistryEnvSharedState<S> {
    fn new(
        constructors: &[Arc<Constructor<S>>],
        methods: &[Arc<MethodImplementation<S>>],
        hooks: &[Arc<Hook<S>>],
        types: &mut TypeArenas<S>,
    ) -> Self {
        let mut state = Self::default();
        state.index_constructors(constructors, types);
        state.collect_parametric_members(constructors, types);
        state.index_methods(methods);
        state.index_hooks(hooks);
        state
    }

    fn constructor_source(constructor: &Arc<Constructor<S>>) -> Source<S> {
        Source {
            kind: SourceKind::ProviderResult(Arc::clone(&constructor.implementation)),
        }
    }

    fn index_constructors(
        &mut self,
        constructors: &[Arc<Constructor<S>>],
        types: &mut TypeArenas<S>,
    ) {
        for constructor in constructors {
            let callable = types
                .parametric
                .callables
                .get(&constructor.fn_type)
                .expect("dangling key");
            self.constructors_by_head_type_return
                .get_or_insert_default(callable.inner.return_type, types)
                .push(Arc::clone(constructor));
        }
    }

    fn collect_parametric_members(
        &mut self,
        constructors: &[Arc<Constructor<S>>],
        types: &mut TypeArenas<S>,
    ) {
        for constructor in constructors {
            let callable = types
                .parametric
                .callables
                .get(&constructor.fn_type)
                .expect("dangling key");
            let source = Self::constructor_source(constructor);
            let mut visited_protocols = HashSet::new();
            let mut visited_typed_dicts = HashSet::new();

            match callable.inner.return_type {
                PyType::Protocol(key) => {
                    visited_protocols.insert(key);
                    self.register_parametric_protocol_members(
                        key,
                        &source,
                        types,
                        &mut visited_protocols,
                        &mut visited_typed_dicts,
                    );
                }
                PyType::TypedDict(key) => {
                    visited_typed_dicts.insert(key);
                    self.register_parametric_typed_dict_members(
                        key,
                        &source,
                        types,
                        &mut visited_protocols,
                        &mut visited_typed_dicts,
                    );
                }
                _ => {}
            }
        }
    }

    fn register_parametric_protocol_members(
        &mut self,
        key: ProtocolKey<S, Parametric>,
        source: &Source<S>,
        types: &mut TypeArenas<S>,
        visited_protocols: &mut HashSet<ProtocolKey<S, Parametric>>,
        visited_typed_dicts: &mut HashSet<TypedDictKey<S, Parametric>>,
    ) {
        let protocol = types.parametric.protocols.get(&key).expect("dangling key");
        let properties: Vec<_> = protocol
            .inner
            .properties
            .iter()
            .map(|(name, &member_type)| (Arc::clone(name), member_type))
            .collect();
        let attributes: Vec<_> = protocol
            .inner
            .attributes
            .iter()
            .map(|(name, &member_type)| (Arc::clone(name), member_type))
            .collect();

        for (name, member_type) in &properties {
            if matches!(member_type, PyType::TypeVar(_) | PyType::ParamSpec(_)) {
                continue;
            }
            self.parametric_properties
                .get_or_insert_default(*member_type, types)
                .push(ParametricProperty {
                    name: Arc::clone(name),
                    source_type: key,
                    member_type: *member_type,
                    source: source.clone(),
                });
        }
        for (name, member_type) in &attributes {
            if matches!(member_type, PyType::TypeVar(_) | PyType::ParamSpec(_)) {
                continue;
            }
            self.parametric_attributes
                .get_or_insert_default(*member_type, types)
                .push(ParametricAttribute {
                    name: Arc::clone(name),
                    source_type: SourceType::Protocol(key),
                    member_type: *member_type,
                    source: source.clone(),
                });
        }

        let all_member_types: Vec<_> = properties
            .iter()
            .map(|(_, member_type)| *member_type)
            .chain(attributes.iter().map(|(_, member_type)| *member_type))
            .collect();
        for member_type in all_member_types {
            match member_type {
                PyType::Protocol(nested) if visited_protocols.insert(nested) => {
                    self.register_parametric_protocol_members(
                        nested,
                        source,
                        types,
                        visited_protocols,
                        visited_typed_dicts,
                    );
                }
                PyType::TypedDict(nested) if visited_typed_dicts.insert(nested) => {
                    self.register_parametric_typed_dict_members(
                        nested,
                        source,
                        types,
                        visited_protocols,
                        visited_typed_dicts,
                    );
                }
                _ => {}
            }
        }
    }

    fn register_parametric_typed_dict_members(
        &mut self,
        key: TypedDictKey<S, Parametric>,
        source: &Source<S>,
        types: &mut TypeArenas<S>,
        visited_protocols: &mut HashSet<ProtocolKey<S, Parametric>>,
        visited_typed_dicts: &mut HashSet<TypedDictKey<S, Parametric>>,
    ) {
        let typed_dict = types
            .parametric
            .typed_dicts
            .get(&key)
            .expect("dangling key");
        let attributes: Vec<_> = typed_dict
            .inner
            .attributes
            .iter()
            .map(|(name, &member_type)| (Arc::clone(name), member_type))
            .collect();

        for (name, member_type) in &attributes {
            self.parametric_attributes
                .get_or_insert_default(*member_type, types)
                .push(ParametricAttribute {
                    name: Arc::clone(name),
                    source_type: SourceType::TypedDict(key),
                    member_type: *member_type,
                    source: source.clone(),
                });
        }

        for (_, member_type) in &attributes {
            match *member_type {
                PyType::Protocol(nested) if visited_protocols.insert(nested) => {
                    self.register_parametric_protocol_members(
                        nested,
                        source,
                        types,
                        visited_protocols,
                        visited_typed_dicts,
                    );
                }
                PyType::TypedDict(nested) if visited_typed_dicts.insert(nested) => {
                    self.register_parametric_typed_dict_members(
                        nested,
                        source,
                        types,
                        visited_protocols,
                        visited_typed_dicts,
                    );
                }
                _ => {}
            }
        }
    }

    fn index_methods(&mut self, methods: &[Arc<MethodImplementation<S>>]) {
        for method in methods {
            self.methods_by_name
                .entry(Arc::clone(&method.name))
                .or_default()
                .push(Arc::clone(method));
        }
    }

    fn index_hooks(&mut self, hooks: &[Arc<Hook<S>>]) {
        for hook in hooks {
            self.hooks_by_name
                .entry(Arc::clone(&hook.name))
                .or_default()
                .push(Arc::clone(hook));
        }
    }

    fn lookup_constructors(
        &self,
        request: PyTypeConcreteKey<S>,
        types: &mut TypeArenas<S>,
    ) -> Vec<ConstructorLookup<S>> {
        let request_qual = types
            .qualifier_of_concrete(request)
            .expect("dangling key")
            .clone();
        let constructors: Vec<_> = self
            .constructors_by_head_type_return
            .get(request, types)
            .flat_map(|bucket| bucket.iter().map(Arc::clone))
            .collect();

        constructors
            .into_iter()
            .filter_map(|constructor| {
                let callable = types
                    .parametric
                    .callables
                    .get(&constructor.fn_type)
                    .expect("dangling key");
                if !qualifier_matches(&request_qual, &callable.qualifier) {
                    return None;
                }
                let bindings = types
                    .cross_unify(request, callable.inner.return_type)
                    .ok()?;
                let concrete_callable =
                    types.apply_bindings(PyType::Callable(constructor.fn_type), &bindings);
                let PyType::Callable(concrete_callable_key) = concrete_callable else {
                    unreachable!("apply_bindings on Callable must return Callable")
                };

                Some(ConstructorLookup {
                    constructor,
                    concrete_callable_key,
                })
            })
            .collect()
    }

    fn lookup_methods(
        &self,
        request: PyTypeConcreteKey<S>,
        types: &mut TypeArenas<S>,
    ) -> Vec<MethodLookup<S>> {
        let PyType::Callable(request_key) = request else {
            return Vec::new();
        };

        let request_qual = types
            .qualifier_of_concrete(request)
            .expect("dangling key")
            .clone();
        let function_name = types
            .concrete
            .callables
            .get(&request_key)
            .expect("dangling key")
            .inner
            .function_name
            .clone();
        let methods: Vec<_> = match function_name {
            Some(name) => self
                .methods_by_name
                .get(&name)
                .map(|bucket| bucket.iter().map(Arc::clone).collect())
                .unwrap_or_default(),
            None => self
                .methods_by_name
                .values()
                .flat_map(|bucket| bucket.iter().map(Arc::clone))
                .collect(),
        };

        methods
            .into_iter()
            .filter_map(|implementation| {
                let bindings = types
                    .cross_unify_callable_params(request_key, implementation.fn_type)
                    .ok()?;
                let parametric_callable = types
                    .parametric
                    .callables
                    .get(&implementation.fn_type)
                    .expect("dangling key");
                if !qualifier_matches(&request_qual, &parametric_callable.qualifier) {
                    return None;
                }

                let concrete_callable =
                    types.apply_bindings(PyType::Callable(implementation.fn_type), &bindings);
                let PyType::Callable(concrete_callable_key) = concrete_callable else {
                    unreachable!("apply_bindings on Callable must return Callable")
                };
                let concrete_bound_to = implementation
                    .bound_to
                    .map(|bound_to| types.apply_bindings(bound_to, &bindings));

                Some(MethodLookup {
                    implementation,
                    concrete_callable_key,
                    concrete_bound_to,
                })
            })
            .collect()
    }

    fn lookup_hooks(
        &self,
        name: &str,
        method_qual: Option<&Qualifier>,
        types: &mut TypeArenas<S>,
    ) -> Vec<HookLookup<S>> {
        let hooks: Vec<_> = self
            .hooks_by_name
            .get(name)
            .map(|bucket| bucket.iter().map(Arc::clone).collect())
            .unwrap_or_default();

        hooks
            .into_iter()
            .filter_map(|hook| {
                if let Some(scope_qual) = method_qual {
                    let hook_callable = types
                        .parametric
                        .callables
                        .get(&hook.fn_type)
                        .expect("dangling key");
                    if !qualifier_matches(scope_qual, &hook_callable.qualifier) {
                        return None;
                    }
                }

                let concrete_callable_ref =
                    types.apply_bindings(PyType::Callable(hook.fn_type), &Bindings::default());
                let PyType::Callable(concrete_callable_key) = concrete_callable_ref else {
                    unreachable!("apply_bindings on Callable must return Callable")
                };

                Some(HookLookup {
                    hook,
                    concrete_callable_key,
                })
            })
            .collect()
    }

    fn lookup_properties(
        &mut self,
        request: PyTypeConcreteKey<S>,
        types: &mut TypeArenas<S>,
    ) -> Vec<Property<S, Concrete>> {
        if let Some(cached) = get_exact_cached(&self.concrete_properties, request, types) {
            return cached;
        }

        let parametric_properties: Vec<ParametricPropertyEntry<S>> = self
            .parametric_properties
            .get(request, types)
            .flat_map(|bucket| {
                bucket.iter().map(|property| {
                    (
                        Arc::clone(&property.name),
                        property.source_type,
                        property.member_type,
                        property.source.clone(),
                    )
                })
            })
            .collect();
        let properties = materialize_parametric_matches(
            request,
            parametric_properties,
            types,
            |(_, _, member_type, _)| *member_type,
            |(name, source_type, _, source), bindings, types| {
                let concrete_ref = types.apply_bindings(PyType::Protocol(source_type), &bindings);
                let PyType::Protocol(concrete_key) = concrete_ref else {
                    unreachable!("apply_bindings on Protocol must return Protocol")
                };
                let member_type = *types
                    .concrete
                    .protocols
                    .get(&concrete_key)
                    .expect("just inserted")
                    .inner
                    .properties
                    .get(&name)
                    .expect("property must exist after monomorphization");
                Property {
                    name,
                    source_type: concrete_key,
                    member_type,
                    source,
                }
            },
        );

        cache_exact_lookup(&mut self.concrete_properties, request, &properties, types);

        properties
    }

    fn lookup_attributes(
        &mut self,
        request: PyTypeConcreteKey<S>,
        types: &mut TypeArenas<S>,
    ) -> Vec<Attribute<S, Concrete>> {
        if let Some(cached) = get_exact_cached(&self.concrete_attributes, request, types) {
            return cached;
        }

        let parametric_attributes: Vec<ParametricAttributeEntry<S>> = self
            .parametric_attributes
            .get(request, types)
            .flat_map(|bucket| {
                bucket.iter().map(|attribute| {
                    (
                        Arc::clone(&attribute.name),
                        attribute.source_type,
                        attribute.member_type,
                        attribute.source.clone(),
                    )
                })
            })
            .collect();
        let attributes = materialize_parametric_matches(
            request,
            parametric_attributes,
            types,
            |(_, _, member_type, _)| *member_type,
            |(name, source_type, _, source), bindings, types| match source_type {
                SourceType::Protocol(key) => {
                    let concrete_ref = types.apply_bindings(PyType::Protocol(key), &bindings);
                    let PyType::Protocol(concrete_key) = concrete_ref else {
                        unreachable!("apply_bindings on Protocol must return Protocol")
                    };
                    let member_type = *types
                        .concrete
                        .protocols
                        .get(&concrete_key)
                        .expect("just inserted")
                        .inner
                        .attributes
                        .get(&name)
                        .expect("attribute must exist after monomorphization");
                    Attribute {
                        name,
                        source_type: SourceType::Protocol(concrete_key),
                        member_type,
                        source: source.clone(),
                    }
                }
                SourceType::TypedDict(key) => {
                    let concrete_ref = types.apply_bindings(PyType::TypedDict(key), &bindings);
                    let PyType::TypedDict(concrete_key) = concrete_ref else {
                        unreachable!("apply_bindings on TypedDict must return TypedDict")
                    };
                    let member_type = *types
                        .concrete
                        .typed_dicts
                        .get(&concrete_key)
                        .expect("just inserted")
                        .inner
                        .attributes
                        .get(&name)
                        .expect("attribute must exist after monomorphization");
                    Attribute {
                        name,
                        source_type: SourceType::TypedDict(concrete_key),
                        member_type,
                        source,
                    }
                }
            },
        );

        cache_exact_lookup(&mut self.concrete_attributes, request, &attributes, types);

        attributes
    }
}

impl<S: ArenaFamily> RegistrySharedState<S> {
    fn build_local_state(
        env: &Arc<RegistryEnv<S>>,
        types: &mut TypeArenas<S>,
    ) -> RegistryEnvLocalState<S> {
        let mut state = RegistryEnvLocalState::default();

        for (source, constant) in &env.root_constants {
            state
                .unqualified_constants
                .get_or_insert_default((*constant).into(), types)
                .push((*constant, source.clone()));

            let mut visited = HashSet::new();
            match constant {
                ConstantType::Protocol(key) => Self::register_concrete_protocol_members(
                    &mut state,
                    *key,
                    source,
                    types,
                    &mut visited,
                ),
                ConstantType::TypedDict(key) => Self::register_concrete_typed_dict_members(
                    &mut state,
                    *key,
                    source,
                    types,
                    &mut visited,
                ),
                ConstantType::Plain(_) => {}
            }
        }

        state
    }

    fn register_concrete_protocol_members(
        state: &mut RegistryEnvLocalState<S>,
        key: ProtocolKey<S, Concrete>,
        source: &Source<S>,
        types: &mut TypeArenas<S>,
        visited: &mut HashSet<PyTypeConcreteKey<S>>,
    ) {
        if !visited.insert(PyType::Protocol(key)) {
            return;
        }

        let protocol = types.concrete.protocols.get(&key).expect("dangling key");
        let properties: Vec<_> = protocol
            .inner
            .properties
            .iter()
            .map(|(name, &member_type)| (Arc::clone(name), member_type))
            .collect();
        let attributes: Vec<_> = protocol
            .inner
            .attributes
            .iter()
            .map(|(name, &member_type)| (Arc::clone(name), member_type))
            .collect();

        for (name, member_type) in &properties {
            state
                .unqualified_properties
                .get_or_insert_default(*member_type, types)
                .push(Property {
                    name: Arc::clone(name),
                    source_type: key,
                    member_type: *member_type,
                    source: source.clone(),
                });
        }
        for (name, member_type) in &attributes {
            state
                .unqualified_attributes
                .get_or_insert_default(*member_type, types)
                .push(Attribute {
                    name: Arc::clone(name),
                    source_type: SourceType::Protocol(key),
                    member_type: *member_type,
                    source: source.clone(),
                });
        }

        let all_member_types: Vec<_> = properties
            .iter()
            .map(|(_, member_type)| *member_type)
            .chain(attributes.iter().map(|(_, member_type)| *member_type))
            .collect();
        for member_type in all_member_types {
            match member_type {
                PyType::Protocol(nested) => {
                    Self::register_concrete_protocol_members(state, nested, source, types, visited);
                }
                PyType::TypedDict(nested) => {
                    Self::register_concrete_typed_dict_members(
                        state, nested, source, types, visited,
                    );
                }
                _ => {}
            }
        }
    }

    fn register_concrete_typed_dict_members(
        state: &mut RegistryEnvLocalState<S>,
        key: TypedDictKey<S, Concrete>,
        source: &Source<S>,
        types: &mut TypeArenas<S>,
        visited: &mut HashSet<PyTypeConcreteKey<S>>,
    ) {
        if !visited.insert(PyType::TypedDict(key)) {
            return;
        }

        let typed_dict = types.concrete.typed_dicts.get(&key).expect("dangling key");
        let attributes: Vec<_> = typed_dict
            .inner
            .attributes
            .iter()
            .map(|(name, &member_type)| (Arc::clone(name), member_type))
            .collect();

        for (name, member_type) in &attributes {
            state
                .unqualified_attributes
                .get_or_insert_default(*member_type, types)
                .push(Attribute {
                    name: Arc::clone(name),
                    source_type: SourceType::TypedDict(key),
                    member_type: *member_type,
                    source: source.clone(),
                });
        }

        for (_, member_type) in &attributes {
            match *member_type {
                PyType::Protocol(nested) => {
                    Self::register_concrete_protocol_members(state, nested, source, types, visited);
                }
                PyType::TypedDict(nested) => {
                    Self::register_concrete_typed_dict_members(
                        state, nested, source, types, visited,
                    );
                }
                _ => {}
            }
        }
    }

    fn lookup_constants(
        &mut self,
        env: &Arc<RegistryEnv<S>>,
        type_ref: PyTypeConcreteKey<S>,
    ) -> Vec<(ConstantType<S>, Source<S>)> {
        let entries = self
            .env_local_caches
            .entry(Arc::clone(env))
            .or_insert_with(|| Self::build_local_state(env, &mut self.types))
            .unqualified_constants
            .get(type_ref, &mut self.types);
        let Some(entries) = entries else {
            return Vec::new();
        };

        filter_with_matching_qualifiers(entries, type_ref, &self.types, |(constant, _), _| {
            Some((*constant).into())
        })
    }

    fn lookup_constructors(
        &mut self,
        _env: &Arc<RegistryEnv<S>>,
        type_ref: PyTypeConcreteKey<S>,
    ) -> Vec<ConstructorLookup<S>> {
        if let Some(cached) = get_exact_cached(
            &self.shared.constructors_by_concrete_return,
            type_ref,
            &mut self.types,
        ) {
            return cached;
        }

        let results = filter_with_matching_qualifiers(
            self.shared
                .lookup_constructors(type_ref, &mut self.types)
                .as_slice(),
            type_ref,
            &self.types,
            |entry, types| {
                types
                    .concrete
                    .callables
                    .get(&entry.concrete_callable_key)
                    .map(|callable| callable.inner.return_type)
            },
        );

        cache_exact_lookup(
            &mut self.shared.constructors_by_concrete_return,
            type_ref,
            &results,
            &mut self.types,
        );

        results
    }

    fn lookup_methods(
        &mut self,
        _env: &Arc<RegistryEnv<S>>,
        type_ref: PyTypeConcreteKey<S>,
    ) -> Vec<MethodLookup<S>> {
        if let Some(cached) = get_exact_cached(
            &self.shared.methods_by_concrete_request,
            type_ref,
            &mut self.types,
        ) {
            return cached;
        }

        let results = self.shared.lookup_methods(type_ref, &mut self.types);

        cache_exact_lookup(
            &mut self.shared.methods_by_concrete_request,
            type_ref,
            &results,
            &mut self.types,
        );

        results
    }

    fn lookup_hooks(
        &mut self,
        _env: &Arc<RegistryEnv<S>>,
        name: &Arc<str>,
        method_qual: Option<&Qualifier>,
    ) -> Vec<HookLookup<S>> {
        let query = (Arc::clone(name), method_qual.cloned());

        if let Some(cached) = self.shared.hooks_by_query.get(&query) {
            return cached.clone();
        }

        let results = self.shared.lookup_hooks(name, method_qual, &mut self.types);
        self.shared.hooks_by_query.insert(query, results.clone());

        results
    }

    fn lookup_properties(
        &mut self,
        env: &Arc<RegistryEnv<S>>,
        type_ref: PyTypeConcreteKey<S>,
    ) -> Vec<Property<S, Concrete>> {
        let mut entries = self
            .env_local_caches
            .entry(Arc::clone(env))
            .or_insert_with(|| Self::build_local_state(env, &mut self.types))
            .unqualified_properties
            .get(type_ref, &mut self.types)
            .cloned()
            .unwrap_or_default();
        entries.extend(self.shared.lookup_properties(type_ref, &mut self.types));

        filter_with_matching_qualifiers(entries.as_slice(), type_ref, &self.types, |property, _| {
            Some(property.member_type)
        })
    }

    fn lookup_attributes(
        &mut self,
        env: &Arc<RegistryEnv<S>>,
        type_ref: PyTypeConcreteKey<S>,
    ) -> Vec<Attribute<S, Concrete>> {
        let mut entries = self
            .env_local_caches
            .entry(Arc::clone(env))
            .or_insert_with(|| Self::build_local_state(env, &mut self.types))
            .unqualified_attributes
            .get(type_ref, &mut self.types)
            .cloned()
            .unwrap_or_default();
        entries.extend(self.shared.lookup_attributes(type_ref, &mut self.types));

        filter_with_matching_qualifiers(
            entries.as_slice(),
            type_ref,
            &self.types,
            |attribute, _| Some(attribute.member_type),
        )
    }
}

#[derive_where(Clone, PartialEq, Eq, Hash)]
pub(crate) struct RegistryEnv<S: ArenaFamily> {
    root_constants: BTreeMap<Source<S>, ConstantType<S>>,
}

impl<S: ArenaFamily> RegistryEnv<S> {
    pub(crate) fn root() -> Self {
        Self {
            root_constants: BTreeMap::new(),
        }
    }

    pub(crate) fn transition_param_source(
        &self,
        name: Arc<str>,
        param_type: PyTypeConcreteKey<S>,
    ) -> Option<Source<S>> {
        to_constant_type(param_type).map(|constant_type| Source {
            kind: SourceKind::TransitionBinding(TransitionBindingKey::<S>::from_constant_type(
                name,
                constant_type,
            )),
        })
    }

    pub(crate) fn transition_result_source(
        &self,
        return_type: PyTypeConcreteKey<S>,
    ) -> Option<Source<S>> {
        to_constant_type(return_type).map(|constant_type| Source {
            kind: SourceKind::TransitionResult(constant_type),
        })
    }

    pub(crate) fn with_transition(
        &self,
        params: Vec<(Arc<str>, PyTypeConcreteKey<S>)>,
        return_type: Option<PyTypeConcreteKey<S>>,
        result_bindings: Vec<(Arc<str>, PyTypeConcreteKey<S>)>,
    ) -> Self {
        let mut root_constants = self.root_constants.clone();

        for (name, param_type) in params {
            if let Some(source) = self.transition_param_source(name, param_type) {
                let constant = to_constant_type(param_type)
                    .expect("transition param source implies constant type");
                root_constants.insert(source, constant);
            }
        }

        if let Some(return_type) = return_type {
            if let Some(source) = self.transition_result_source(return_type) {
                let constant = to_constant_type(return_type)
                    .expect("transition result source implies constant type");
                root_constants.insert(source, constant);
            }
        }

        for (name, binding_type) in result_bindings {
            if let Some(source) = self.transition_param_source(name, binding_type) {
                let constant = to_constant_type(binding_type)
                    .expect("transition binding implies constant type");
                root_constants.insert(source, constant);
            }
        }

        Self { root_constants }
    }
}

impl<S: ArenaFamily> std::fmt::Debug for RegistryEnv<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryEnv")
            .field("root_constants", &self.root_constants.len())
            .finish()
    }
}

#[derive_where(PartialEq, Eq, Clone, Hash)]
pub(crate) enum ResolutionLookup<S: ArenaFamily> {
    Constant(PyTypeConcreteKey<S>),
    Constructor(PyTypeConcreteKey<S>),
    Method(PyTypeConcreteKey<S>),
    Hook {
        name: Arc<str>,
        method_qual: Option<Qualifier>,
    },
    Property(PyTypeConcreteKey<S>),
    Attribute(PyTypeConcreteKey<S>),
}

#[derive_where(Clone, PartialEq, Eq, Hash)]
pub(crate) enum ResolutionLookupResult<S: ArenaFamily> {
    Constants(BTreeSet<(ConstantType<S>, Source<S>)>),
    Constructors(BTreeSet<ConstructorLookup<S>>),
    Methods(BTreeSet<MethodLookup<S>>),
    Hooks(BTreeSet<HookLookup<S>>),
    Properties(BTreeSet<Property<S, Concrete>>),
    Attributes(BTreeSet<Attribute<S, Concrete>>),
}

impl<S: ArenaFamily> ResolutionEnv for RegistryEnv<S> {
    type SharedState = RegistrySharedState<S>;
    type Query = ResolutionLookup<S>;
    type QueryResult = ResolutionLookupResult<S>;

    fn lookup(
        self: &Arc<Self>,
        shared_state: &mut Self::SharedState,
        query: &Self::Query,
    ) -> Self::QueryResult {
        let started = Instant::now();
        let (kind, result) = match query {
            ResolutionLookup::Constant(type_ref) => (
                "constant",
                ResolutionLookupResult::Constants(
                    shared_state
                        .lookup_constants(self, *type_ref)
                        .into_iter()
                        .collect(),
                ),
            ),
            ResolutionLookup::Constructor(type_ref) => (
                "constructor",
                ResolutionLookupResult::Constructors(
                    shared_state
                        .lookup_constructors(self, *type_ref)
                        .into_iter()
                        .collect(),
                ),
            ),
            ResolutionLookup::Method(type_ref) => (
                "method",
                ResolutionLookupResult::Methods(
                    shared_state
                        .lookup_methods(self, *type_ref)
                        .into_iter()
                        .collect(),
                ),
            ),
            ResolutionLookup::Hook { name, method_qual } => (
                "hook",
                ResolutionLookupResult::Hooks(
                    shared_state
                        .lookup_hooks(self, name, method_qual.as_ref())
                        .into_iter()
                        .collect(),
                ),
            ),
            ResolutionLookup::Property(type_ref) => (
                "property",
                ResolutionLookupResult::Properties(
                    shared_state
                        .lookup_properties(self, *type_ref)
                        .into_iter()
                        .collect(),
                ),
            ),
            ResolutionLookup::Attribute(type_ref) => (
                "attribute",
                ResolutionLookupResult::Attributes(
                    shared_state
                        .lookup_attributes(self, *type_ref)
                        .into_iter()
                        .collect(),
                ),
            ),
        };
        shared_state.record_lookup(kind, lookup_result_len(&result), started.elapsed());
        result
    }
}
