use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use context_solver::rule::ResolutionEnv;
use derive_where::derive_where;
use inlay_instrument_macros::instrumented;

use crate::instrument::inlay_span_record;
use crate::qualifier::{Qualifier, qualifier_matches};
use crate::registry::{Hook, MethodImplementation, TransitionBindingKey, to_constant_type};
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

fn hash_trace_value<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

pub(crate) fn summarize_lookup_for_trace<S: ArenaFamily>(query: &ResolutionLookup<S>) -> String {
    match query {
        ResolutionLookup::Constant {
            type_ref,
            requested_name,
        } => match requested_name {
            Some(name) => format!("constant:type={:x}@{name}", hash_trace_value(type_ref)),
            None => format!("constant:type={:x}", hash_trace_value(type_ref)),
        },
        ResolutionLookup::Constructor(type_ref) => {
            format!("constructor:type={:x}", hash_trace_value(type_ref))
        }
        ResolutionLookup::Method(type_ref) => {
            format!("method:type={:x}", hash_trace_value(type_ref))
        }
        ResolutionLookup::Hook { name, method_qual } => match method_qual {
            Some(qualifier) => format!("hook:{name}@{qualifier:?}"),
            None => format!("hook:{name}"),
        },
        ResolutionLookup::Property(type_ref) => {
            format!("property:type={:x}", hash_trace_value(type_ref))
        }
        ResolutionLookup::Attribute(type_ref) => {
            format!("attribute:type={:x}", hash_trace_value(type_ref))
        }
    }
}

pub(crate) fn summarize_lookup_result_for_trace<S: ArenaFamily>(
    result: &ResolutionLookupResult<S>,
) -> String {
    match result {
        ResolutionLookupResult::Constants(entries) => {
            format!(
                "constants[len={} hash={:x}]",
                entries.len(),
                hash_trace_value(entries)
            )
        }
        ResolutionLookupResult::Constructors(entries) => format!(
            "constructors[len={} hash={:x}]",
            entries.len(),
            hash_trace_value(entries)
        ),
        ResolutionLookupResult::Methods(entries) => {
            format!(
                "methods[len={} hash={:x}]",
                entries.len(),
                hash_trace_value(entries)
            )
        }
        ResolutionLookupResult::Hooks(entries) => {
            format!(
                "hooks[len={} hash={:x}]",
                entries.len(),
                hash_trace_value(entries)
            )
        }
        ResolutionLookupResult::Properties(entries) => format!(
            "properties[len={} hash={:x}]",
            entries.len(),
            hash_trace_value(entries)
        ),
        ResolutionLookupResult::Attributes(entries) => format!(
            "attributes[len={} hash={:x}]",
            entries.len(),
            hash_trace_value(entries)
        ),
    }
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
    let matching = entries
        .iter()
        .filter_map(|entry| {
            let registration_qual = registered_type(entry, types)
                .and_then(|registered_type| types.qualifier_of_concrete(registered_type))?;
            qualifier_matches(request_qual, registration_qual)
                .then_some((entry.clone(), registration_qual == request_qual))
        })
        .collect::<Vec<_>>();

    if matching.iter().any(|(_, exact)| *exact) {
        return matching
            .into_iter()
            .filter_map(|(entry, exact)| exact.then_some(entry))
            .collect();
    }

    matching.into_iter().map(|(entry, _)| entry).collect()
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
    named_constants:
        HashMap<Arc<str>, TypeKeyMap<S, UnqualifiedMode, Vec<(ConstantType<S>, Source<S>)>>>,
    unqualified_properties: TypeKeyMap<S, UnqualifiedMode, Vec<Property<S, Concrete>>>,
    unqualified_attributes: TypeKeyMap<S, UnqualifiedMode, Vec<Attribute<S, Concrete>>>,
}

pub(crate) struct RegistrySharedState<S: ArenaFamily> {
    shared: RegistryEnvSharedState<S>,
    env_local_caches: HashMap<Arc<RegistryEnv<S>>, RegistryEnvLocalState<S>>,
    types: TypeArenas<S>,
}

impl<S: ArenaFamily> std::fmt::Debug for RegistrySharedState<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistrySharedState")
            .field("methods_by_name", &self.shared.methods_by_name.len())
            .field("hooks_by_name", &self.shared.hooks_by_name.len())
            .field("env_local_caches", &self.env_local_caches.len())
            .finish()
    }
}

impl<S: ArenaFamily> RegistrySharedState<S> {
    #[instrumented(
        name = "inlay.registry_shared_state.new",
        target = "inlay",
        level = "trace",
        skip(types),
        fields(
            constructors = constructors.len() as u64,
            methods = methods.len() as u64,
            hooks = hooks.len() as u64
        )
    )]
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
            types,
        }
    }

    pub(crate) fn types(&mut self) -> &mut TypeArenas<S> {
        &mut self.types
    }

    pub(crate) fn into_types(self) -> TypeArenas<S> {
        self.types
    }

    fn should_cache_local_state(env: &RegistryEnv<S>) -> bool {
        env.cache_local_state && std::env::var_os("INLAY_DISABLE_ENV_LOCAL_CACHE").is_none()
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
                let return_qual = types
                    .get(callable.inner.return_type)
                    .map(|value| value.qualifier())
                    .expect("dangling key");
                if !qualifier_matches(&request_qual, return_qual) {
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
    #[instrumented(
        name = "inlay.registry_shared_state.build_local_state",
        target = "inlay",
        level = "trace",
        skip(types),
        fields(
            root_constants = env.root_constants.len() as u64,
            named_constants
        )
    )]
    fn build_local_state(
        env: &Arc<RegistryEnv<S>>,
        types: &mut TypeArenas<S>,
    ) -> RegistryEnvLocalState<S> {
        let mut state = RegistryEnvLocalState::default();

        let constants = env
            .root_constants
            .iter()
            .map(|(source, constant)| (source.clone(), *constant))
            .collect::<Vec<_>>();

        for (source, constant) in &constants {
            state
                .unqualified_constants
                .get_or_insert_default((*constant).into(), types)
                .push((*constant, source.clone()));
            if let SourceKind::TransitionBinding(binding) = &source.kind {
                state
                    .named_constants
                    .entry(Arc::clone(&binding.name))
                    .or_default()
                    .get_or_insert_default((*constant).into(), types)
                    .push((*constant, source.clone()));
            }

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

        inlay_span_record!(named_constants = state.named_constants.len() as u64);
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
        requested_name: Option<&Arc<str>>,
    ) -> Vec<(ConstantType<S>, Source<S>)> {
        let constants = env
            .root_constants
            .iter()
            .map(|(source, constant)| (*constant, source.clone()))
            .filter(|(constant, _)| {
                self.types
                    .deep_eq_concrete::<UnqualifiedMode>(type_ref, (*constant).into())
            })
            .collect::<Vec<_>>();

        if let Some(requested_name) = requested_name {
            let named_entries = constants
                .iter()
                .filter(|(_, source)| {
                    matches!(
                        &source.kind,
                        SourceKind::TransitionBinding(binding)
                            if binding.name.as_ref() == requested_name.as_ref()
                    )
                })
                .cloned()
                .collect::<Vec<_>>();
            let named_matches = filter_with_matching_qualifiers(
                named_entries.as_slice(),
                type_ref,
                &self.types,
                |(constant, _), _| Some((*constant).into()),
            );
            if !named_matches.is_empty() {
                return named_matches;
            }
        }

        filter_with_matching_qualifiers(
            constants.as_slice(),
            type_ref,
            &self.types,
            |(constant, _), _| Some((*constant).into()),
        )
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
        let mut entries = if Self::should_cache_local_state(env) {
            self.env_local_caches
                .entry(Arc::clone(env))
                .or_insert_with(|| Self::build_local_state(env, &mut self.types))
                .unqualified_properties
                .get(type_ref, &mut self.types)
                .cloned()
                .unwrap_or_default()
        } else {
            Self::build_local_state(env, &mut self.types)
                .unqualified_properties
                .get(type_ref, &mut self.types)
                .cloned()
                .unwrap_or_default()
        };
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
        let mut entries = if Self::should_cache_local_state(env) {
            self.env_local_caches
                .entry(Arc::clone(env))
                .or_insert_with(|| Self::build_local_state(env, &mut self.types))
                .unqualified_attributes
                .get(type_ref, &mut self.types)
                .cloned()
                .unwrap_or_default()
        } else {
            Self::build_local_state(env, &mut self.types)
                .unqualified_attributes
                .get(type_ref, &mut self.types)
                .cloned()
                .unwrap_or_default()
        };
        entries.extend(self.shared.lookup_attributes(type_ref, &mut self.types));

        filter_with_matching_qualifiers(
            entries.as_slice(),
            type_ref,
            &self.types,
            |attribute, _| Some(attribute.member_type),
        )
    }
}

pub(crate) struct RegistryEnv<S: ArenaFamily> {
    root_constants: BTreeMap<Source<S>, ConstantType<S>>,
    cache_local_state: bool,
}

#[derive_where(Clone, PartialEq, Eq, Hash)]
pub(crate) struct RegistryEnvDelta<S: ArenaFamily> {
    inserted_constants: Vec<(Source<S>, ConstantType<S>)>,
}

impl<S: ArenaFamily> std::fmt::Debug for RegistryEnvDelta<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryEnvDelta")
            .field("inserted_constants", &self.inserted_constants.len())
            .finish()
    }
}

pub(crate) fn summarize_env_for_trace<S: ArenaFamily>(env: &RegistryEnv<S>) -> String {
    if env.root_constants.is_empty() {
        return "n=0 []".to_string();
    }

    let include_keys = std::env::var_os("INLAY_TRACE_SOURCE_KEYS").is_some();
    let bindings = env
        .root_constants
        .keys()
        .take(8)
        .map(|source| match &source.kind {
            SourceKind::ProviderResult(_) => "provider".to_string(),
            SourceKind::TransitionBinding(binding) => {
                if include_keys {
                    match binding.constant_type {
                        ConstantType::Plain(key) => {
                            format!("bind:{}#plain:{key:?}", binding.name)
                        }
                        ConstantType::Protocol(key) => {
                            format!("bind:{}#protocol:{key:?}", binding.name)
                        }
                        ConstantType::TypedDict(key) => {
                            format!("bind:{}#typed_dict:{key:?}", binding.name)
                        }
                    }
                } else {
                    format!("bind:{}", binding.name)
                }
            }
            SourceKind::TransitionResult(constant_type) => {
                if include_keys {
                    match constant_type {
                        ConstantType::Plain(key) => format!("result#plain:{key:?}"),
                        ConstantType::Protocol(key) => format!("result#protocol:{key:?}"),
                        ConstantType::TypedDict(key) => {
                            format!("result#typed_dict:{key:?}")
                        }
                    }
                } else {
                    "result".to_string()
                }
            }
        })
        .collect::<Vec<_>>();
    let more = env.root_constants.len().saturating_sub(bindings.len());
    if more == 0 {
        format!("n={} [{}]", env.root_constants.len(), bindings.join(", "))
    } else {
        format!(
            "n={} [{} ,+{} more]",
            env.root_constants.len(),
            bindings.join(", "),
            more
        )
    }
}

impl<S: ArenaFamily> RegistryEnv<S> {
    pub(crate) fn root() -> Self {
        Self {
            root_constants: BTreeMap::new(),
            cache_local_state: true,
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

    #[instrumented(
        name = "inlay.registry_env.with_transition",
        target = "inlay",
        level = "trace",
        ret,
        skip(params, return_type, result_bindings),
        fields(
            parent_items = self.root_constants.len() as u64,
            params = params.len() as u64,
            has_return = return_type.is_some(),
            result_bindings = result_bindings.len() as u64,
            child_items
        )
    )]
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

        let env = Self {
            root_constants,
            cache_local_state: true,
        };
        inlay_span_record!(child_items = env.root_constants.len() as u64);
        env
    }
}

impl<S: ArenaFamily> Clone for RegistryEnv<S> {
    fn clone(&self) -> Self {
        Self {
            root_constants: self.root_constants.clone(),
            cache_local_state: self.cache_local_state,
        }
    }
}

impl<S: ArenaFamily> PartialEq for RegistryEnv<S> {
    fn eq(&self, other: &Self) -> bool {
        self.root_constants == other.root_constants
    }
}

impl<S: ArenaFamily> Eq for RegistryEnv<S> {}

impl<S: ArenaFamily> std::hash::Hash for RegistryEnv<S> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.root_constants.hash(state);
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
    Constant {
        type_ref: PyTypeConcreteKey<S>,
        requested_name: Option<Arc<str>>,
    },
    Constructor(PyTypeConcreteKey<S>),
    Method(PyTypeConcreteKey<S>),
    Hook {
        name: Arc<str>,
        method_qual: Option<Qualifier>,
    },
    Property(PyTypeConcreteKey<S>),
    Attribute(PyTypeConcreteKey<S>),
}

impl<S: ArenaFamily> std::fmt::Debug for ResolutionLookup<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&summarize_lookup_for_trace(self))
    }
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

impl<S: ArenaFamily> std::fmt::Debug for ResolutionLookupResult<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&summarize_lookup_result_for_trace(self))
    }
}

impl<S: ArenaFamily> ResolutionEnv for RegistryEnv<S> {
    type SharedState = RegistrySharedState<S>;
    type Query = ResolutionLookup<S>;
    type QueryResult = ResolutionLookupResult<S>;
    type DependencyEnvDelta = RegistryEnvDelta<S>;

    #[instrumented(
        name = "inlay.registry_env.lookup",
        target = "inlay",
        level = "trace",
        ret,
        fields(
            query_hash = hash_trace_value(query),
            env_items = self.root_constants.len() as u64,
            query_label = %summarize_lookup_for_trace(query)
        )
    )]
    fn lookup(
        self: &Arc<Self>,
        shared_state: &mut Self::SharedState,
        query: &Self::Query,
    ) -> Self::QueryResult {
        match query {
            ResolutionLookup::Constant {
                type_ref,
                requested_name,
            } => ResolutionLookupResult::Constants(
                shared_state
                    .lookup_constants(self, *type_ref, requested_name.as_ref())
                    .into_iter()
                    .collect(),
            ),
            ResolutionLookup::Constructor(type_ref) => ResolutionLookupResult::Constructors(
                shared_state
                    .lookup_constructors(self, *type_ref)
                    .into_iter()
                    .collect(),
            ),
            ResolutionLookup::Method(type_ref) => ResolutionLookupResult::Methods(
                shared_state
                    .lookup_methods(self, *type_ref)
                    .into_iter()
                    .collect(),
            ),
            ResolutionLookup::Hook { name, method_qual } => ResolutionLookupResult::Hooks(
                shared_state
                    .lookup_hooks(self, name, method_qual.as_ref())
                    .into_iter()
                    .collect(),
            ),
            ResolutionLookup::Property(type_ref) => ResolutionLookupResult::Properties(
                shared_state
                    .lookup_properties(self, *type_ref)
                    .into_iter()
                    .collect(),
            ),
            ResolutionLookup::Attribute(type_ref) => ResolutionLookupResult::Attributes(
                shared_state
                    .lookup_attributes(self, *type_ref)
                    .into_iter()
                    .collect(),
            ),
        }
    }

    fn dependency_env_delta(parent: &Arc<Self>, child: &Arc<Self>) -> Self::DependencyEnvDelta {
        let inserted_constants = child
            .root_constants
            .iter()
            .filter_map(|(source, constant)| {
                (parent.root_constants.get(source) != Some(constant))
                    .then(|| (source.clone(), *constant))
            })
            .collect();

        Self::DependencyEnvDelta { inserted_constants }
    }

    fn apply_dependency_env_delta(
        parent: &Arc<Self>,
        delta: &Self::DependencyEnvDelta,
    ) -> Arc<Self> {
        if delta.inserted_constants.is_empty() {
            return Arc::clone(parent);
        }
        let mut root_constants = parent.root_constants.clone();
        for (source, constant) in &delta.inserted_constants {
            root_constants.insert(source.clone(), *constant);
        }
        Arc::new(Self {
            root_constants,
            cache_local_state: false,
        })
    }

    fn env_item_count(env: &Self) -> usize {
        env.root_constants.len()
    }

    fn dependency_env_delta_item_count(delta: &Self::DependencyEnvDelta) -> usize {
        delta.inserted_constants.len()
    }
}
