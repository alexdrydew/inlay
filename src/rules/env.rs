use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use context_solver::{ResolutionEnv, RuleLookupSupport};
use derive_where::derive_where;
use inlay_instrument::{inlay_span_record, instrumented};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};

use crate::qualifier::qualifier_matches;
use crate::types::TypeArenas;
use crate::{
    registry::{
        Constructor, MethodImplementation, Source, SourceKind, SourceType, TransitionBindingKey,
        TransitionResultKey,
    },
    types::{
        Bindings, CallableKey, Concrete, Parametric, ProtocolKey, PyType, PyTypeConcreteKey,
        PyTypeKey, QualifiedMode, ShallowTypeKeyMap, TypeKeyMap, TypeVarSupport, TypedDictKey,
        UnqualifiedMode,
    },
};

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ConstructorLookup<'types> {
    pub(crate) constructor: Arc<Constructor<'types>>,
    pub(crate) concrete_callable_key: CallableKey<'types, Concrete>,
}

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct MethodLookup<'types> {
    pub(crate) implementation: Arc<MethodImplementation<'types>>,
    pub(crate) concrete_public_callable_key: CallableKey<'types, Concrete>,
    pub(crate) concrete_implementation_callable_key: CallableKey<'types, Concrete>,
    pub(crate) concrete_bound_to: Option<PyTypeConcreteKey<'types>>,
}

#[derive_where(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Property<'types, G: TypeVarSupport> {
    pub(crate) name: Arc<str>,
    pub(crate) source_type: ProtocolKey<'types, G>,
    pub(crate) member_type: PyTypeKey<'types, G>,
    pub(crate) source: Source<'types>,
}

struct ParametricProperty<'types> {
    name: Arc<str>,
    source_type: ProtocolKey<'types, Parametric>,
    member_type: PyTypeKey<'types, Parametric>,
    source: Source<'types>,
}

#[derive_where(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Attribute<'types, G: TypeVarSupport> {
    pub(crate) name: Arc<str>,
    pub(crate) source_type: SourceType<'types, G>,
    pub(crate) member_type: PyTypeKey<'types, G>,
    pub(crate) source: Source<'types>,
}

struct ParametricAttribute<'types> {
    name: Arc<str>,
    source_type: SourceType<'types, Parametric>,
    member_type: PyTypeKey<'types, Parametric>,
    source: Source<'types>,
}

type ExactLookupCache<'types, T> =
    TypeKeyMap<'types, UnqualifiedMode, Vec<(PyTypeConcreteKey<'types>, Vec<T>)>>;
type ParametricPropertyEntry<'types> = (
    Arc<str>,
    ProtocolKey<'types, Parametric>,
    PyTypeKey<'types, Parametric>,
    Source<'types>,
);
type ParametricAttributeEntry<'types> = (
    Arc<str>,
    SourceType<'types, Parametric>,
    PyTypeKey<'types, Parametric>,
    Source<'types>,
);
type ConstantSources<'types> = Vec<(PyTypeConcreteKey<'types>, Source<'types>)>;
type ConstantMap<'types> = TypeKeyMap<'types, UnqualifiedMode, ConstantSources<'types>>;
type NamedConstantMap<'types> = HashMap<Arc<str>, ConstantMap<'types>>;
type ConstructorsByHeadTypeReturn<'types> =
    ShallowTypeKeyMap<'types, UnqualifiedMode, Vec<Arc<Constructor<'types>>>>;
type ParametricPropertyMap<'types> =
    ShallowTypeKeyMap<'types, UnqualifiedMode, Vec<ParametricProperty<'types>>>;
type ParametricAttributeMap<'types> =
    ShallowTypeKeyMap<'types, UnqualifiedMode, Vec<ParametricAttribute<'types>>>;

fn hash_trace_value<T: Hash>(value: &T) -> u64 {
    let mut hasher = FxHasher::default();
    value.hash(&mut hasher);
    hasher.finish()
}

pub(crate) fn summarize_lookup_for_trace(query: &ResolutionLookup<'_>) -> String {
    match query {
        ResolutionLookup::Constant {
            type_ref,
            requested_name,
        } => match requested_name {
            Some(name) => format!("constant:type={:x}@{name}", hash_trace_value(type_ref)),
            None => format!("constant:type={:x}", hash_trace_value(type_ref)),
        },
        ResolutionLookup::Property(type_ref) => {
            format!("property:type={:x}", hash_trace_value(type_ref))
        }
        ResolutionLookup::Attribute(type_ref) => {
            format!("attribute:type={:x}", hash_trace_value(type_ref))
        }
    }
}

pub(crate) fn summarize_lookup_result_for_trace(result: &ResolutionLookupResult<'_>) -> String {
    match result {
        ResolutionLookupResult::Constants(entries) => {
            format!(
                "constants[len={} hash={:x}]",
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

fn get_exact_cached<'types, T: Clone>(
    cache: &ExactLookupCache<'types, T>,
    request: PyTypeConcreteKey<'types>,
    types: &mut TypeArenas<'types>,
) -> Option<Vec<T>> {
    cache.get(request, types).and_then(|variants| {
        variants
            .iter()
            .find(|(key, _)| types.deep_eq_concrete::<QualifiedMode>(request, *key))
            .map(|(_, cached)| cached.clone())
    })
}

fn cache_exact_lookup<'types, T: Clone>(
    cache: &mut ExactLookupCache<'types, T>,
    request: PyTypeConcreteKey<'types>,
    results: &[T],
    types: &mut TypeArenas<'types>,
) {
    cache
        .get_or_insert_default(request, types)
        .push((request, results.to_vec()));
}

fn materialize_parametric_matches<'types, Entry, T>(
    request: PyTypeConcreteKey<'types>,
    entries: impl IntoIterator<Item = Entry>,
    types: &mut TypeArenas<'types>,
    member_type: impl Fn(&Entry) -> PyTypeKey<'types, Parametric>,
    mut materialize: impl FnMut(Entry, Bindings<'types>, &mut TypeArenas<'types>) -> T,
) -> Vec<T> {
    let mut results = Vec::new();

    for entry in entries {
        if let Ok(bindings) = types.cross_unify(request, member_type(&entry)) {
            results.push(materialize(entry, bindings, types));
        }
    }

    results
}

fn filter_with_matching_qualifiers<'types, T: Clone>(
    entries: &[T],
    request: PyTypeConcreteKey<'types>,
    types: &TypeArenas<'types>,
    registered_type: impl Fn(&T) -> Option<PyTypeConcreteKey<'types>>,
) -> Vec<T> {
    let request_qual = types.qualifier_of_concrete(request);
    let matching = entries
        .iter()
        .filter_map(|entry| {
            let registration_qual = types.qualifier_of_concrete(registered_type(entry)?);
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

#[derive(Default)]
struct RegistryEnvSharedState<'types> {
    methods_by_name: HashMap<Arc<str>, Vec<Arc<MethodImplementation<'types>>>>,

    constructors_by_head_type_return: ConstructorsByHeadTypeReturn<'types>,
    constructors_by_concrete_return: ExactLookupCache<'types, ConstructorLookup<'types>>,
    methods_by_concrete_request: ExactLookupCache<'types, MethodLookup<'types>>,

    concrete_properties: ExactLookupCache<'types, Property<'types, Concrete>>,
    parametric_properties: ParametricPropertyMap<'types>,

    concrete_attributes: ExactLookupCache<'types, Attribute<'types, Concrete>>,
    parametric_attributes: ParametricAttributeMap<'types>,
}

#[derive(Default)]
struct RegistryEnvLocalState<'types> {
    unqualified_constants: ConstantMap<'types>,
    named_constants: NamedConstantMap<'types>,
    unqualified_properties: TypeKeyMap<'types, UnqualifiedMode, Vec<Property<'types, Concrete>>>,
    unqualified_attributes: TypeKeyMap<'types, UnqualifiedMode, Vec<Attribute<'types, Concrete>>>,
}

pub(crate) struct RegistrySharedState<'types> {
    shared: RegistryEnvSharedState<'types>,
    env_local_caches: HashMap<Arc<RegistryEnv<'types>>, RegistryEnvLocalState<'types>>,
    projection_snapshots:
        HashMap<RegistryProjectionCacheKey<'types>, Arc<RegistryProjectionSnapshot<'types>>>,
    canonical_concrete_unqualified: TypeKeyMap<'types, UnqualifiedMode, PyTypeConcreteKey<'types>>,
    pub(crate) types: TypeArenas<'types>,
}

impl std::fmt::Debug for RegistrySharedState<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistrySharedState")
            .field("methods_by_name", &self.shared.methods_by_name.len())
            .field("env_local_caches", &self.env_local_caches.len())
            .field("projection_snapshots", &self.projection_snapshots.len())
            .finish()
    }
}

impl<'types> RegistrySharedState<'types> {
    #[instrumented(
        name = "inlay.registry_shared_state.new",
        target = "inlay",
        level = "trace",
        skip(types),
        fields(
            constructors = constructors.len() as u64,
            methods = methods.len() as u64
        )
    )]
    pub(crate) fn new(
        constructors: &[Constructor<'types>],
        methods: &[MethodImplementation<'types>],
        mut types: TypeArenas<'types>,
    ) -> Self {
        let constructors = constructors
            .iter()
            .cloned()
            .map(Arc::new)
            .collect::<Vec<_>>();
        let methods = methods.iter().cloned().map(Arc::new).collect::<Vec<_>>();
        let shared = RegistryEnvSharedState::new(&constructors, &methods, &mut types);
        Self {
            shared,
            env_local_caches: HashMap::default(),
            projection_snapshots: HashMap::default(),
            canonical_concrete_unqualified: TypeKeyMap::new(),
            types,
        }
    }

    pub(crate) fn types(&mut self) -> &mut TypeArenas<'types> {
        &mut self.types
    }

    fn canonical_unqualified_concrete(
        &mut self,
        type_ref: PyTypeConcreteKey<'types>,
    ) -> PyTypeConcreteKey<'types> {
        let mut canonical = std::mem::take(&mut self.canonical_concrete_unqualified);
        let result = canonical
            .get(type_ref, &mut self.types)
            .copied()
            .unwrap_or_else(|| {
                canonical.insert(type_ref, type_ref, &mut self.types);
                type_ref
            });
        self.canonical_concrete_unqualified = canonical;
        result
    }
}

impl<'types> RegistryEnvSharedState<'types> {
    fn new(
        constructors: &[Arc<Constructor<'types>>],
        methods: &[Arc<MethodImplementation<'types>>],
        types: &mut TypeArenas<'types>,
    ) -> Self {
        let mut state = Self::default();
        state.index_constructors(constructors, types);
        state.collect_parametric_members(constructors, types);
        state.index_methods(methods);
        state
    }

    fn constructor_source(constructor: &Arc<Constructor<'types>>) -> Source<'types> {
        Source {
            kind: SourceKind::ProviderResult(Arc::clone(&constructor.implementation)),
        }
    }

    fn index_constructors(
        &mut self,
        constructors: &[Arc<Constructor<'types>>],
        types: &mut TypeArenas<'types>,
    ) {
        for constructor in constructors {
            let callable = types.parametric.callables.get(constructor.fn_type);
            self.constructors_by_head_type_return
                .get_or_insert_default(callable.inner.return_type, types)
                .push(Arc::clone(constructor));
        }
    }

    fn collect_parametric_members(
        &mut self,
        constructors: &[Arc<Constructor<'types>>],
        types: &mut TypeArenas<'types>,
    ) {
        for constructor in constructors {
            let callable = types.parametric.callables.get(constructor.fn_type);
            let source = Self::constructor_source(constructor);
            let mut visited_protocols = HashSet::default();
            let mut visited_typed_dicts = HashSet::default();

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
        key: ProtocolKey<'types, Parametric>,
        source: &Source<'types>,
        types: &mut TypeArenas<'types>,
        visited_protocols: &mut HashSet<ProtocolKey<'types, Parametric>>,
        visited_typed_dicts: &mut HashSet<TypedDictKey<'types, Parametric>>,
    ) {
        let protocol = types.parametric.protocols.get(key);
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
        key: TypedDictKey<'types, Parametric>,
        source: &Source<'types>,
        types: &mut TypeArenas<'types>,
        visited_protocols: &mut HashSet<ProtocolKey<'types, Parametric>>,
        visited_typed_dicts: &mut HashSet<TypedDictKey<'types, Parametric>>,
    ) {
        let typed_dict = types.parametric.typed_dicts.get(key);
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

    fn index_methods(&mut self, methods: &[Arc<MethodImplementation<'types>>]) {
        for method in methods {
            self.methods_by_name
                .entry(Arc::clone(&method.name))
                .or_default()
                .push(Arc::clone(method));
        }
    }

    fn lookup_constructors(
        &self,
        request: PyTypeConcreteKey<'types>,
        types: &mut TypeArenas<'types>,
    ) -> Vec<ConstructorLookup<'types>> {
        let request_qual = types.qualifier_of_concrete(request).clone();
        let constructors: Vec<_> = self
            .constructors_by_head_type_return
            .get(request, types)
            .flat_map(|bucket| bucket.iter().map(Arc::clone))
            .collect();

        constructors
            .into_iter()
            .filter_map(|constructor| {
                let callable = types.parametric.callables.get(constructor.fn_type);
                let return_type = types.get(callable.inner.return_type);
                let return_qual = return_type.qualifier();
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
        request: PyTypeConcreteKey<'types>,
        types: &mut TypeArenas<'types>,
    ) -> Vec<MethodLookup<'types>> {
        let PyType::Callable(request_key) = request else {
            return Vec::new();
        };

        let request_qual = types.qualifier_of_concrete(request).clone();
        let function_name = types
            .concrete
            .callables
            .get(request_key)
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
                    .cross_unify_callable_signature(request_key, implementation.public_fn_type)
                    .ok()?;
                let parametric_callable = types
                    .parametric
                    .callables
                    .get(implementation.public_fn_type);
                if !qualifier_matches(&request_qual, &parametric_callable.qualifier) {
                    return None;
                }

                let concrete_public_callable = types
                    .apply_bindings(PyType::Callable(implementation.public_fn_type), &bindings);
                let PyType::Callable(concrete_public_callable_key) = concrete_public_callable
                else {
                    unreachable!("apply_bindings on Callable must return Callable")
                };
                let concrete_implementation_callable = types.apply_bindings(
                    PyType::Callable(implementation.implementation_fn_type),
                    &bindings,
                );
                let PyType::Callable(concrete_implementation_callable_key) =
                    concrete_implementation_callable
                else {
                    unreachable!("apply_bindings on Callable must return Callable")
                };
                let concrete_bound_to = implementation
                    .bound_to
                    .map(|bound_to| types.apply_bindings(bound_to, &bindings));

                Some(MethodLookup {
                    implementation,
                    concrete_public_callable_key,
                    concrete_implementation_callable_key,
                    concrete_bound_to,
                })
            })
            .collect()
    }

    fn lookup_properties(
        &mut self,
        request: PyTypeConcreteKey<'types>,
        types: &mut TypeArenas<'types>,
    ) -> Vec<Property<'types, Concrete>> {
        if let Some(cached) = get_exact_cached(&self.concrete_properties, request, types) {
            return cached;
        }

        let parametric_properties: Vec<ParametricPropertyEntry<'types>> = self
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
                    .get(concrete_key)
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
        request: PyTypeConcreteKey<'types>,
        types: &mut TypeArenas<'types>,
    ) -> Vec<Attribute<'types, Concrete>> {
        if let Some(cached) = get_exact_cached(&self.concrete_attributes, request, types) {
            return cached;
        }

        let parametric_attributes: Vec<ParametricAttributeEntry<'types>> = self
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
                        .get(concrete_key)
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
                        .get(concrete_key)
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

impl<'types> RegistrySharedState<'types> {
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
        env: &Arc<RegistryEnv<'types>>,
        types: &mut TypeArenas<'types>,
    ) -> RegistryEnvLocalState<'types> {
        let mut state = RegistryEnvLocalState::default();

        let constants = env
            .root_constants
            .iter()
            .map(|(source, constant)| (source.clone(), *constant))
            .collect::<Vec<_>>();

        for (source, constant) in &constants {
            state
                .unqualified_constants
                .get_or_insert_default(*constant, types)
                .push((*constant, source.clone()));
            if let SourceKind::TransitionBinding(binding) = &source.kind {
                state
                    .named_constants
                    .entry(Arc::clone(&binding.name))
                    .or_default()
                    .get_or_insert_default(*constant, types)
                    .push((*constant, source.clone()));
            }

            let mut visited = HashSet::default();
            match constant {
                PyType::Protocol(key) => Self::register_concrete_protocol_members(
                    &mut state,
                    *key,
                    source,
                    types,
                    &mut visited,
                ),
                PyType::TypedDict(key) => Self::register_concrete_typed_dict_members(
                    &mut state,
                    *key,
                    source,
                    types,
                    &mut visited,
                ),
                _ => {}
            }
        }

        inlay_span_record!(named_constants = state.named_constants.len() as u64);
        state
    }

    fn register_concrete_protocol_members(
        state: &mut RegistryEnvLocalState<'types>,
        key: ProtocolKey<'types, Concrete>,
        source: &Source<'types>,
        types: &mut TypeArenas<'types>,
        visited: &mut HashSet<PyTypeConcreteKey<'types>>,
    ) {
        if !visited.insert(PyType::Protocol(key)) {
            return;
        }

        let protocol = types.concrete.protocols.get(key);
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
        state: &mut RegistryEnvLocalState<'types>,
        key: TypedDictKey<'types, Concrete>,
        source: &Source<'types>,
        types: &mut TypeArenas<'types>,
        visited: &mut HashSet<PyTypeConcreteKey<'types>>,
    ) {
        if !visited.insert(PyType::TypedDict(key)) {
            return;
        }

        let typed_dict = types.concrete.typed_dicts.get(key);
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
        env: &Arc<RegistryEnv<'types>>,
        type_ref: PyTypeConcreteKey<'types>,
        requested_name: Option<&Arc<str>>,
    ) -> Vec<(PyTypeConcreteKey<'types>, Source<'types>)> {
        if let Some(requested_name) = requested_name {
            let named_entries = self
                .env_local_caches
                .entry(Arc::clone(env))
                .or_insert_with(|| Self::build_local_state(env, &mut self.types))
                .named_constants
                .get(requested_name)
                .and_then(|constants| constants.get(type_ref, &mut self.types))
                .cloned()
                .unwrap_or_default();
            let named_matches = filter_with_matching_qualifiers(
                named_entries.as_slice(),
                type_ref,
                &self.types,
                |(constant, _)| Some(*constant),
            );
            if !named_matches.is_empty() {
                return named_matches;
            }
        }

        let constants = self
            .env_local_caches
            .entry(Arc::clone(env))
            .or_insert_with(|| Self::build_local_state(env, &mut self.types))
            .unqualified_constants
            .get(type_ref, &mut self.types)
            .cloned()
            .unwrap_or_default();

        filter_with_matching_qualifiers(
            constants.as_slice(),
            type_ref,
            &self.types,
            |(constant, _)| Some(*constant),
        )
    }

    pub(crate) fn lookup_constructors(
        &mut self,
        type_ref: PyTypeConcreteKey<'types>,
    ) -> Vec<ConstructorLookup<'types>> {
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
            |entry| {
                Some(
                    self.types
                        .concrete
                        .callables
                        .get(entry.concrete_callable_key)
                        .inner
                        .return_type,
                )
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

    pub(crate) fn lookup_methods(
        &mut self,
        type_ref: PyTypeConcreteKey<'types>,
    ) -> Vec<MethodLookup<'types>> {
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

    fn lookup_properties(
        &mut self,
        env: &Arc<RegistryEnv<'types>>,
        type_ref: PyTypeConcreteKey<'types>,
    ) -> Vec<Property<'types, Concrete>> {
        let mut entries = self
            .env_local_caches
            .entry(Arc::clone(env))
            .or_insert_with(|| Self::build_local_state(env, &mut self.types))
            .unqualified_properties
            .get(type_ref, &mut self.types)
            .cloned()
            .unwrap_or_default();
        entries.extend(self.shared.lookup_properties(type_ref, &mut self.types));

        filter_with_matching_qualifiers(entries.as_slice(), type_ref, &self.types, |property| {
            Some(property.member_type)
        })
    }

    fn lookup_attributes(
        &mut self,
        env: &Arc<RegistryEnv<'types>>,
        type_ref: PyTypeConcreteKey<'types>,
    ) -> Vec<Attribute<'types, Concrete>> {
        let mut entries = self
            .env_local_caches
            .entry(Arc::clone(env))
            .or_insert_with(|| Self::build_local_state(env, &mut self.types))
            .unqualified_attributes
            .get(type_ref, &mut self.types)
            .cloned()
            .unwrap_or_default();
        entries.extend(self.shared.lookup_attributes(type_ref, &mut self.types));

        filter_with_matching_qualifiers(entries.as_slice(), type_ref, &self.types, |attribute| {
            Some(attribute.member_type)
        })
    }

    fn projection_support(
        &mut self,
        env: &Arc<RegistryEnv<'types>>,
        kind: RegistryProjectionKind,
        type_ref: PyTypeConcreteKey<'types>,
    ) -> RegistryProjectionSupport<'types> {
        let domain = RegistryProjectionDomain {
            kind,
            type_family: self.canonical_unqualified_concrete(type_ref),
            ignored_sources: BTreeSet::new(),
        };
        let expected = self.projection_snapshot(env, &domain);
        RegistryProjectionSupport { domain, expected }
    }

    fn projection_snapshot(
        &mut self,
        env: &Arc<RegistryEnv<'types>>,
        domain: &RegistryProjectionDomain<'types>,
    ) -> RegistryProjectionSnapshot<'types> {
        let base = self.base_projection_snapshot(env, domain.kind, domain.type_family);
        if domain.ignored_sources.is_empty() {
            return base.as_ref().clone();
        }

        base.filter_ignored_sources(&domain.ignored_sources)
    }

    fn projection_snapshot_matches(
        &mut self,
        env: &Arc<RegistryEnv<'types>>,
        support: &RegistryProjectionSupport<'types>,
    ) -> bool {
        let base =
            self.base_projection_snapshot(env, support.domain.kind, support.domain.type_family);
        if support.domain.ignored_sources.is_empty() {
            return support.expected == *base.as_ref();
        }

        base.matches_filtered(&support.expected, &support.domain.ignored_sources)
    }

    fn base_projection_snapshot(
        &mut self,
        env: &Arc<RegistryEnv<'types>>,
        kind: RegistryProjectionKind,
        type_family: PyTypeConcreteKey<'types>,
    ) -> Arc<RegistryProjectionSnapshot<'types>> {
        let key = RegistryProjectionCacheKey {
            env: Arc::clone(env),
            kind,
            type_family,
        };
        if let Some(snapshot) = self.projection_snapshots.get(&key) {
            return Arc::clone(snapshot);
        }

        let snapshot = Arc::new(self.build_projection_snapshot(env, kind, type_family));
        self.projection_snapshots.insert(key, Arc::clone(&snapshot));
        snapshot
    }

    fn build_projection_snapshot(
        &mut self,
        env: &Arc<RegistryEnv<'types>>,
        kind: RegistryProjectionKind,
        type_family: PyTypeConcreteKey<'types>,
    ) -> RegistryProjectionSnapshot<'types> {
        match kind {
            RegistryProjectionKind::Constants => {
                RegistryProjectionSnapshot::Constants(self.projection_constants(env, type_family))
            }
            RegistryProjectionKind::Properties => {
                RegistryProjectionSnapshot::Properties(self.projection_properties(env, type_family))
            }
            RegistryProjectionKind::Attributes => {
                RegistryProjectionSnapshot::Attributes(self.projection_attributes(env, type_family))
            }
        }
    }

    fn projection_constants(
        &mut self,
        env: &Arc<RegistryEnv<'types>>,
        type_ref: PyTypeConcreteKey<'types>,
    ) -> BTreeSet<(PyTypeConcreteKey<'types>, Source<'types>)> {
        let entries = self
            .env_local_caches
            .entry(Arc::clone(env))
            .or_insert_with(|| Self::build_local_state(env, &mut self.types))
            .unqualified_constants
            .get(type_ref, &mut self.types)
            .cloned()
            .unwrap_or_default();

        entries.into_iter().collect()
    }

    fn projection_properties(
        &mut self,
        env: &Arc<RegistryEnv<'types>>,
        type_ref: PyTypeConcreteKey<'types>,
    ) -> BTreeSet<Property<'types, Concrete>> {
        let entries = self
            .env_local_caches
            .entry(Arc::clone(env))
            .or_insert_with(|| Self::build_local_state(env, &mut self.types))
            .unqualified_properties
            .get(type_ref, &mut self.types)
            .cloned()
            .unwrap_or_default();

        entries.into_iter().collect()
    }

    fn projection_attributes(
        &mut self,
        env: &Arc<RegistryEnv<'types>>,
        type_ref: PyTypeConcreteKey<'types>,
    ) -> BTreeSet<Attribute<'types, Concrete>> {
        let entries = self
            .env_local_caches
            .entry(Arc::clone(env))
            .or_insert_with(|| Self::build_local_state(env, &mut self.types))
            .unqualified_attributes
            .get(type_ref, &mut self.types)
            .cloned()
            .unwrap_or_default();

        entries.into_iter().collect()
    }
}

pub(crate) struct RegistryEnv<'types> {
    root_constants: BTreeMap<Source<'types>, PyTypeConcreteKey<'types>>,
    hash: u64,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct RegistryEnvDelta<'types> {
    inserted_constants: Vec<(Source<'types>, PyTypeConcreteKey<'types>)>,
}

impl std::fmt::Debug for RegistryEnvDelta<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryEnvDelta")
            .field("inserted_constants", &self.inserted_constants.len())
            .finish()
    }
}

impl RegistryEnvDelta<'_> {
    fn is_empty(&self) -> bool {
        self.inserted_constants.is_empty()
    }
}

impl<'types> RegistryEnv<'types> {
    fn new(root_constants: BTreeMap<Source<'types>, PyTypeConcreteKey<'types>>) -> Self {
        let hash = hash_trace_value(&root_constants);
        Self {
            root_constants,
            hash,
        }
    }

    pub(crate) fn transition_param_source(
        &self,
        name: Arc<str>,
        param_type: PyTypeConcreteKey<'types>,
        scope: usize,
    ) -> Source<'types> {
        Source {
            kind: SourceKind::TransitionBinding(TransitionBindingKey::from_type_ref(
                name, param_type, scope,
            )),
        }
    }

    pub(crate) fn transition_result_source(
        &self,
        return_type: PyTypeConcreteKey<'types>,
        scope: usize,
    ) -> Source<'types> {
        Source {
            kind: SourceKind::TransitionResult(TransitionResultKey {
                type_ref: return_type,
                scope,
            }),
        }
    }

    #[instrumented(
        name = "inlay.registry_env.with_transition",
        target = "inlay",
        level = "trace",
        ret,
        skip(params, result_types),
        fields(
            parent_items = self.root_constants.len() as u64,
            params = params.len() as u64,
            result_types = result_types.len() as u64,
            child_items
        )
    )]
    pub(crate) fn with_transition(
        &self,
        params: Vec<(Arc<str>, PyTypeConcreteKey<'types>)>,
        result_types: Vec<(PyTypeConcreteKey<'types>, usize)>,
    ) -> Self {
        let mut root_constants = self.root_constants.clone();

        for (name, param_type) in params {
            root_constants.insert(
                self.transition_param_source(name, param_type, 0),
                param_type,
            );
        }

        for (return_type, scope) in result_types {
            root_constants.insert(
                self.transition_result_source(return_type, scope),
                return_type,
            );
        }

        let env = Self::new(root_constants);
        inlay_span_record!(child_items = env.root_constants.len() as u64);
        env
    }
}

impl Default for RegistryEnv<'_> {
    fn default() -> Self {
        Self::new(BTreeMap::new())
    }
}

impl Clone for RegistryEnv<'_> {
    fn clone(&self) -> Self {
        Self {
            root_constants: self.root_constants.clone(),
            hash: self.hash,
        }
    }
}

impl PartialEq for RegistryEnv<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.root_constants == other.root_constants
    }
}

impl Eq for RegistryEnv<'_> {}

impl std::hash::Hash for RegistryEnv<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

impl std::fmt::Debug for RegistryEnv<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryEnv")
            .field("root_constants", &self.root_constants.len())
            .finish()
    }
}

#[derive(PartialEq, Eq, Clone, Hash)]
pub(crate) enum ResolutionLookup<'types> {
    Constant {
        type_ref: PyTypeConcreteKey<'types>,
        requested_name: Option<Arc<str>>,
    },
    Property(PyTypeConcreteKey<'types>),
    Attribute(PyTypeConcreteKey<'types>),
}

impl std::fmt::Debug for ResolutionLookup<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&summarize_lookup_for_trace(self))
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum ResolutionLookupResult<'types> {
    Constants(BTreeSet<(PyTypeConcreteKey<'types>, Source<'types>)>),
    Properties(BTreeSet<Property<'types, Concrete>>),
    Attributes(BTreeSet<Attribute<'types, Concrete>>),
}

impl std::fmt::Debug for ResolutionLookupResult<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&summarize_lookup_result_for_trace(self))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RegistryProjectionKind {
    Constants,
    Properties,
    Attributes,
}

#[derive(Clone)]
struct RegistryProjectionCacheKey<'types> {
    env: Arc<RegistryEnv<'types>>,
    kind: RegistryProjectionKind,
    type_family: PyTypeConcreteKey<'types>,
}

impl PartialEq for RegistryProjectionCacheKey<'_> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.env, &other.env)
            && self.kind == other.kind
            && self.type_family == other.type_family
    }
}

impl Eq for RegistryProjectionCacheKey<'_> {}

impl Hash for RegistryProjectionCacheKey<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.env).hash(state);
        self.kind.hash(state);
        self.type_family.hash(state);
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct RegistryProjectionDomain<'types> {
    kind: RegistryProjectionKind,
    type_family: PyTypeConcreteKey<'types>,
    ignored_sources: BTreeSet<Source<'types>>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct RegistryProjectionSupport<'types> {
    domain: RegistryProjectionDomain<'types>,
    expected: RegistryProjectionSnapshot<'types>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum RegistryProjectionSnapshot<'types> {
    Constants(BTreeSet<(PyTypeConcreteKey<'types>, Source<'types>)>),
    Properties(BTreeSet<Property<'types, Concrete>>),
    Attributes(BTreeSet<Attribute<'types, Concrete>>),
}

impl std::fmt::Debug for RegistryProjectionDomain<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryProjectionDomain")
            .field("kind", &self.kind)
            .field("type_family", &hash_trace_value(&self.type_family))
            .field("ignored_sources", &self.ignored_sources.len())
            .finish()
    }
}

impl std::fmt::Debug for RegistryProjectionSupport<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryProjectionSupport")
            .field("domain", &self.domain)
            .field("expected", &self.expected.len())
            .finish()
    }
}

impl<'types> RegistryProjectionSnapshot<'types> {
    fn matches_filtered(
        &self,
        expected: &Self,
        ignored_sources: &BTreeSet<Source<'types>>,
    ) -> bool {
        match (self, expected) {
            (Self::Constants(current), Self::Constants(expected)) => {
                let mut visible = 0;
                for entry in current {
                    if ignored_sources.contains(&entry.1) {
                        continue;
                    }
                    visible += 1;
                    if !expected.contains(entry) {
                        return false;
                    }
                }
                visible == expected.len()
            }
            (Self::Properties(current), Self::Properties(expected)) => {
                let mut visible = 0;
                for entry in current {
                    if ignored_sources.contains(&entry.source) {
                        continue;
                    }
                    visible += 1;
                    if !expected.contains(entry) {
                        return false;
                    }
                }
                visible == expected.len()
            }
            (Self::Attributes(current), Self::Attributes(expected)) => {
                let mut visible = 0;
                for entry in current {
                    if ignored_sources.contains(&entry.source) {
                        continue;
                    }
                    visible += 1;
                    if !expected.contains(entry) {
                        return false;
                    }
                }
                visible == expected.len()
            }
            _ => false,
        }
    }

    fn filter_ignored_sources(&self, ignored_sources: &BTreeSet<Source<'types>>) -> Self {
        match self {
            Self::Constants(entries) => Self::Constants(
                entries
                    .iter()
                    .filter(|(_, source)| !ignored_sources.contains(source))
                    .cloned()
                    .collect(),
            ),
            Self::Properties(entries) => Self::Properties(
                entries
                    .iter()
                    .filter(|property| !ignored_sources.contains(&property.source))
                    .cloned()
                    .collect(),
            ),
            Self::Attributes(entries) => Self::Attributes(
                entries
                    .iter()
                    .filter(|attribute| !ignored_sources.contains(&attribute.source))
                    .cloned()
                    .collect(),
            ),
        }
    }

    fn union(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Self::Constants(left), Self::Constants(right)) => {
                Some(Self::Constants(left.union(right).cloned().collect()))
            }
            (Self::Properties(left), Self::Properties(right)) => {
                Some(Self::Properties(left.union(right).cloned().collect()))
            }
            (Self::Attributes(left), Self::Attributes(right)) => {
                Some(Self::Attributes(left.union(right).cloned().collect()))
            }
            _ => None,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Constants(entries) => entries.len(),
            Self::Properties(entries) => entries.len(),
            Self::Attributes(entries) => entries.len(),
        }
    }
}

impl RuleLookupSupport for RegistryProjectionSupport<'_> {
    fn merge_lookup_support(&self, other: &Self) -> Option<Self> {
        if self.domain.kind != other.domain.kind
            || self.domain.type_family != other.domain.type_family
        {
            return None;
        }

        let domain = RegistryProjectionDomain {
            kind: self.domain.kind,
            type_family: self.domain.type_family,
            ignored_sources: self
                .domain
                .ignored_sources
                .intersection(&other.domain.ignored_sources)
                .cloned()
                .collect(),
        };
        Some(Self {
            expected: self
                .expected
                .union(&other.expected)?
                .filter_ignored_sources(&domain.ignored_sources),
            domain,
        })
    }
}

impl<'types> ResolutionEnv for RegistryEnv<'types> {
    type SharedState = RegistrySharedState<'types>;
    type Query = ResolutionLookup<'types>;
    type QueryResult = ResolutionLookupResult<'types>;
    type DependencyEnvDelta = RegistryEnvDelta<'types>;
    type LookupSupport = RegistryProjectionSupport<'types>;

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

    fn lookup_support(
        self: &Arc<Self>,
        shared_state: &mut Self::SharedState,
        query: &Self::Query,
        _result: &Self::QueryResult,
    ) -> Self::LookupSupport {
        match query {
            ResolutionLookup::Constant { type_ref, .. } => {
                shared_state.projection_support(self, RegistryProjectionKind::Constants, *type_ref)
            }
            ResolutionLookup::Property(type_ref) => {
                shared_state.projection_support(self, RegistryProjectionKind::Properties, *type_ref)
            }
            ResolutionLookup::Attribute(type_ref) => {
                shared_state.projection_support(self, RegistryProjectionKind::Attributes, *type_ref)
            }
        }
    }

    fn lookup_support_matches(
        self: &Arc<Self>,
        shared_state: &mut Self::SharedState,
        support: &Self::LookupSupport,
    ) -> bool {
        shared_state.projection_snapshot_matches(self, support)
    }

    fn pullback_lookup_support(
        support: &Self::LookupSupport,
        delta: &Self::DependencyEnvDelta,
    ) -> Self::LookupSupport {
        if delta.is_empty() {
            return support.clone();
        }

        let mut ignored_sources = support.domain.ignored_sources.clone();
        ignored_sources.extend(
            delta
                .inserted_constants
                .iter()
                .map(|(source, _)| source.clone()),
        );
        let expected = support.expected.filter_ignored_sources(&ignored_sources);
        RegistryProjectionSupport {
            domain: RegistryProjectionDomain {
                kind: support.domain.kind,
                type_family: support.domain.type_family,
                ignored_sources,
            },
            expected,
        }
    }

    fn dependency_env_delta(parent: &Arc<Self>, child: &Arc<Self>) -> Self::DependencyEnvDelta {
        if Arc::ptr_eq(parent, child) {
            return Self::DependencyEnvDelta {
                inserted_constants: Vec::new(),
            };
        }

        let inserted_constants = child
            .root_constants
            .iter()
            .filter(|&(source, constant)| parent.root_constants.get(source) != Some(constant))
            .map(|(source, constant)| (source.clone(), *constant))
            .collect();

        Self::DependencyEnvDelta { inserted_constants }
    }

    fn compose_dependency_env_delta(
        first: &Self::DependencyEnvDelta,
        second: &Self::DependencyEnvDelta,
    ) -> Self::DependencyEnvDelta {
        if first.is_empty() {
            return second.clone();
        }
        if second.is_empty() {
            return first.clone();
        }

        let mut inserted_constants = first
            .inserted_constants
            .iter()
            .map(|(source, constant)| (source.clone(), *constant))
            .collect::<BTreeMap<_, _>>();
        for (source, constant) in &second.inserted_constants {
            inserted_constants.insert(source.clone(), *constant);
        }

        Self::DependencyEnvDelta {
            inserted_constants: inserted_constants.into_iter().collect(),
        }
    }
}
