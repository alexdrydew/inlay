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
    registry::{Constructor, MethodImplementation, Source, SourceType},
    types::{
        Bindings, CallableKey, Concrete, Parametric, ProtocolKey, PyType, PyTypeConcreteKey,
        PyTypeKey, QualifiedMode, ShallowTypeKeyMap, TypeKeyMap, TypeVarSupport, TypedDictKey,
        UnqualifiedMode,
    },
};

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ConstructorLookup<'ty> {
    pub(crate) constructor: Arc<Constructor<'ty>>,
    pub(crate) concrete_callable_key: CallableKey<'ty, Concrete>,
}

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct MethodLookup<'ty> {
    pub(crate) implementation: Arc<MethodImplementation<'ty>>,
    pub(crate) concrete_public_callable_key: CallableKey<'ty, Concrete>,
    pub(crate) concrete_implementation_callable_key: CallableKey<'ty, Concrete>,
    pub(crate) concrete_bound_to: Option<PyTypeConcreteKey<'ty>>,
}

#[derive_where(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Property<'ty, G: TypeVarSupport> {
    pub(crate) name: Arc<str>,
    pub(crate) source_type: ProtocolKey<'ty, G>,
    pub(crate) member_type: PyTypeKey<'ty, G>,
    pub(crate) source: Source<'ty>,
}

struct ParametricProperty<'ty> {
    name: Arc<str>,
    source_type: ProtocolKey<'ty, Parametric>,
    member_type: PyTypeKey<'ty, Parametric>,
    source: Source<'ty>,
}

#[derive_where(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Attribute<'ty, G: TypeVarSupport> {
    pub(crate) name: Arc<str>,
    pub(crate) source_type: SourceType<'ty, G>,
    pub(crate) member_type: PyTypeKey<'ty, G>,
    pub(crate) source: Source<'ty>,
}

struct ParametricAttribute<'ty> {
    name: Arc<str>,
    source_type: SourceType<'ty, Parametric>,
    member_type: PyTypeKey<'ty, Parametric>,
    source: Source<'ty>,
}

type ExactLookupCache<'ty, T> =
    TypeKeyMap<'ty, UnqualifiedMode, Vec<(PyTypeConcreteKey<'ty>, Vec<T>)>>;
type ParametricPropertyEntry<'ty> = (
    Arc<str>,
    ProtocolKey<'ty, Parametric>,
    PyTypeKey<'ty, Parametric>,
    Source<'ty>,
);
type ParametricAttributeEntry<'ty> = (
    Arc<str>,
    SourceType<'ty, Parametric>,
    PyTypeKey<'ty, Parametric>,
    Source<'ty>,
);
type ConstantSources<'ty> = Vec<Source<'ty>>;
type ConstantMap<'ty> = TypeKeyMap<'ty, UnqualifiedMode, ConstantSources<'ty>>;
type NamedConstantMap<'ty> = HashMap<Arc<str>, ConstantMap<'ty>>;
type SourceSet<'ty> = BTreeSet<Source<'ty>>;
type NamedSourceSets<'ty> = BTreeMap<Arc<str>, SourceSet<'ty>>;
type ConstructorsByHeadTypeReturn<'ty> =
    ShallowTypeKeyMap<'ty, UnqualifiedMode, Vec<Arc<Constructor<'ty>>>>;
type ParametricPropertyMap<'ty> =
    ShallowTypeKeyMap<'ty, UnqualifiedMode, Vec<ParametricProperty<'ty>>>;
type ParametricAttributeMap<'ty> =
    ShallowTypeKeyMap<'ty, UnqualifiedMode, Vec<ParametricAttribute<'ty>>>;

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
        ResolutionLookupResult::Constants { entries, .. } => {
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

fn get_exact_cached<'ty, T: Clone>(
    cache: &ExactLookupCache<'ty, T>,
    request: PyTypeConcreteKey<'ty>,
    types: &mut TypeArenas<'ty>,
) -> Option<Vec<T>> {
    cache.get(request, types).and_then(|variants| {
        variants
            .iter()
            .find(|(key, _)| types.deep_eq_concrete::<QualifiedMode>(request, *key))
            .map(|(_, cached)| cached.clone())
    })
}

fn cache_exact_lookup<'ty, T: Clone>(
    cache: &mut ExactLookupCache<'ty, T>,
    request: PyTypeConcreteKey<'ty>,
    results: &[T],
    types: &mut TypeArenas<'ty>,
) {
    cache
        .get_or_insert_default(request, types)
        .push((request, results.to_vec()));
}

fn materialize_parametric_matches<'ty, Entry, T>(
    request: PyTypeConcreteKey<'ty>,
    entries: impl IntoIterator<Item = Entry>,
    types: &mut TypeArenas<'ty>,
    member_type: impl Fn(&Entry) -> PyTypeKey<'ty, Parametric>,
    mut materialize: impl FnMut(Entry, Bindings<'ty>, &mut TypeArenas<'ty>) -> T,
) -> Vec<T> {
    let mut results = Vec::new();

    for entry in entries {
        if let Ok(bindings) = types.cross_unify(request, member_type(&entry)) {
            results.push(materialize(entry, bindings, types));
        }
    }

    results
}

fn filter_with_matching_qualifiers<'ty, T: Clone>(
    entries: &[T],
    request: PyTypeConcreteKey<'ty>,
    types: &TypeArenas<'ty>,
    registered_type: impl Fn(&T) -> Option<PyTypeConcreteKey<'ty>>,
) -> Vec<T> {
    let request_qual = types.qualifier_of_concrete(request);
    entries
        .iter()
        .filter_map(|entry| {
            let registration_qual = types.qualifier_of_concrete(registered_type(entry)?);
            qualifier_matches(request_qual, registration_qual).then_some(entry.clone())
        })
        .collect()
}

#[derive(Default)]
struct RegistryEnvSharedState<'ty> {
    methods_by_name: HashMap<Arc<str>, Vec<Arc<MethodImplementation<'ty>>>>,

    constructors_by_head_type_return: ConstructorsByHeadTypeReturn<'ty>,
    constructors_by_concrete_return: ExactLookupCache<'ty, ConstructorLookup<'ty>>,
    methods_by_concrete_request: ExactLookupCache<'ty, MethodLookup<'ty>>,

    concrete_properties: ExactLookupCache<'ty, Property<'ty, Concrete>>,
    parametric_properties: ParametricPropertyMap<'ty>,

    concrete_attributes: ExactLookupCache<'ty, Attribute<'ty, Concrete>>,
    parametric_attributes: ParametricAttributeMap<'ty>,
}

#[derive(Default)]
struct RegistryEnvLocalState<'ty> {
    unqualified_constants: ConstantMap<'ty>,
    named_constants: NamedConstantMap<'ty>,
    unqualified_properties: TypeKeyMap<'ty, UnqualifiedMode, Vec<Property<'ty, Concrete>>>,
    unqualified_attributes: TypeKeyMap<'ty, UnqualifiedMode, Vec<Attribute<'ty, Concrete>>>,
}

pub(crate) struct RegistrySharedState<'ty> {
    shared: RegistryEnvSharedState<'ty>,
    env_local_caches: HashMap<Arc<RegistryEnv<'ty>>, RegistryEnvLocalState<'ty>>,
    projection_snapshots:
        HashMap<RegistryProjectionCacheKey<'ty>, Arc<RegistryProjectionSnapshot<'ty>>>,
    canonical_concrete_unqualified: TypeKeyMap<'ty, UnqualifiedMode, PyTypeConcreteKey<'ty>>,
    pub(crate) types: TypeArenas<'ty>,
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

impl<'ty> RegistrySharedState<'ty> {
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
        constructors: &[Constructor<'ty>],
        methods: &[MethodImplementation<'ty>],
        mut types: TypeArenas<'ty>,
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

    pub(crate) fn types(&mut self) -> &mut TypeArenas<'ty> {
        &mut self.types
    }

    fn canonical_unqualified_concrete(
        &mut self,
        type_ref: PyTypeConcreteKey<'ty>,
    ) -> PyTypeConcreteKey<'ty> {
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

impl<'ty> RegistryEnvSharedState<'ty> {
    fn new(
        constructors: &[Arc<Constructor<'ty>>],
        methods: &[Arc<MethodImplementation<'ty>>],
        types: &mut TypeArenas<'ty>,
    ) -> Self {
        let mut state = Self::default();
        state.index_constructors(constructors, types);
        state.collect_parametric_members(constructors, types);
        state.index_methods(methods);
        state
    }

    fn constructor_source(constructor: &Arc<Constructor<'ty>>) -> Source<'ty> {
        Source::provider_result(Arc::clone(&constructor.implementation))
    }

    fn index_constructors(
        &mut self,
        constructors: &[Arc<Constructor<'ty>>],
        types: &mut TypeArenas<'ty>,
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
        constructors: &[Arc<Constructor<'ty>>],
        types: &mut TypeArenas<'ty>,
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
        key: ProtocolKey<'ty, Parametric>,
        source: &Source<'ty>,
        types: &mut TypeArenas<'ty>,
        visited_protocols: &mut HashSet<ProtocolKey<'ty, Parametric>>,
        visited_typed_dicts: &mut HashSet<TypedDictKey<'ty, Parametric>>,
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
        key: TypedDictKey<'ty, Parametric>,
        source: &Source<'ty>,
        types: &mut TypeArenas<'ty>,
        visited_protocols: &mut HashSet<ProtocolKey<'ty, Parametric>>,
        visited_typed_dicts: &mut HashSet<TypedDictKey<'ty, Parametric>>,
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

    fn index_methods(&mut self, methods: &[Arc<MethodImplementation<'ty>>]) {
        for method in methods {
            self.methods_by_name
                .entry(Arc::clone(&method.name))
                .or_default()
                .push(Arc::clone(method));
        }
    }

    fn lookup_constructors(
        &self,
        request: PyTypeConcreteKey<'ty>,
        types: &mut TypeArenas<'ty>,
    ) -> Vec<ConstructorLookup<'ty>> {
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
        request: PyTypeConcreteKey<'ty>,
        types: &mut TypeArenas<'ty>,
    ) -> Vec<MethodLookup<'ty>> {
        let PyType::Callable(request_key) = request else {
            return Vec::new();
        };

        let request_qual = types.qualifier_of_concrete(request).clone();
        let request_callable = types.concrete.callables.get(request_key);
        let function_name = request_callable.inner.function_name.clone();
        let request_return_qual = types
            .qualifier_of_concrete(request_callable.inner.return_type)
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
                let public_return_type = types
                    .concrete
                    .callables
                    .get(concrete_public_callable_key)
                    .inner
                    .return_type;
                if !qualifier_matches(
                    &request_return_qual,
                    types.qualifier_of_concrete(public_return_type),
                ) {
                    return None;
                }
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
        request: PyTypeConcreteKey<'ty>,
        types: &mut TypeArenas<'ty>,
    ) -> Vec<Property<'ty, Concrete>> {
        if let Some(cached) = get_exact_cached(&self.concrete_properties, request, types) {
            return cached;
        }

        let parametric_properties: Vec<ParametricPropertyEntry<'ty>> = self
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
        request: PyTypeConcreteKey<'ty>,
        types: &mut TypeArenas<'ty>,
    ) -> Vec<Attribute<'ty, Concrete>> {
        if let Some(cached) = get_exact_cached(&self.concrete_attributes, request, types) {
            return cached;
        }

        let parametric_attributes: Vec<ParametricAttributeEntry<'ty>> = self
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

impl<'ty> RegistrySharedState<'ty> {
    #[instrumented(
        name = "inlay.registry_shared_state.build_local_state",
        target = "inlay",
        level = "trace",
        skip(types),
        fields(
            unnamed_constants = env.unnamed_constants.len() as u64,
            named_constants
        )
    )]
    fn build_local_state(
        env: &Arc<RegistryEnv<'ty>>,
        types: &mut TypeArenas<'ty>,
    ) -> RegistryEnvLocalState<'ty> {
        let mut state = RegistryEnvLocalState::default();

        let constants = env.unnamed_constants.iter().cloned().collect::<Vec<_>>();

        for source in &constants {
            let constant = source
                .transition_type_ref()
                .expect("env constants must be transition sources");
            state
                .unqualified_constants
                .get_or_insert_default(constant, types)
                .push(source.clone());

            let mut visited = HashSet::default();
            match constant {
                PyType::Protocol(key) => Self::register_concrete_protocol_members(
                    &mut state,
                    key,
                    source,
                    types,
                    &mut visited,
                ),
                PyType::TypedDict(key) => Self::register_concrete_typed_dict_members(
                    &mut state,
                    key,
                    source,
                    types,
                    &mut visited,
                ),
                _ => {}
            }
        }

        for (name, sources) in &env.named_constants {
            for source in sources {
                let constant = source
                    .transition_type_ref()
                    .expect("env constants must be transition sources");
                state
                    .named_constants
                    .entry(Arc::clone(name))
                    .or_default()
                    .get_or_insert_default(constant, types)
                    .push(source.clone());
            }
        }

        inlay_span_record!(named_constants = state.named_constants.len() as u64);
        state
    }

    fn register_concrete_protocol_members(
        state: &mut RegistryEnvLocalState<'ty>,
        key: ProtocolKey<'ty, Concrete>,
        source: &Source<'ty>,
        types: &mut TypeArenas<'ty>,
        visited: &mut HashSet<PyTypeConcreteKey<'ty>>,
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
        state: &mut RegistryEnvLocalState<'ty>,
        key: TypedDictKey<'ty, Concrete>,
        source: &Source<'ty>,
        types: &mut TypeArenas<'ty>,
        visited: &mut HashSet<PyTypeConcreteKey<'ty>>,
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
        env: &Arc<RegistryEnv<'ty>>,
        type_ref: PyTypeConcreteKey<'ty>,
        requested_name: Option<&Arc<str>>,
    ) -> (Vec<Source<'ty>>, bool) {
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
                Source::transition_type_ref,
            );
            if !named_matches.is_empty() {
                return (named_matches, false);
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

        let matches = filter_with_matching_qualifiers(
            constants.as_slice(),
            type_ref,
            &self.types,
            Source::transition_type_ref,
        );
        (matches, requested_name.is_some())
    }

    pub(crate) fn lookup_constructors(
        &mut self,
        type_ref: PyTypeConcreteKey<'ty>,
    ) -> Vec<ConstructorLookup<'ty>> {
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
        type_ref: PyTypeConcreteKey<'ty>,
    ) -> Vec<MethodLookup<'ty>> {
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
        env: &Arc<RegistryEnv<'ty>>,
        type_ref: PyTypeConcreteKey<'ty>,
    ) -> Vec<Property<'ty, Concrete>> {
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
        env: &Arc<RegistryEnv<'ty>>,
        type_ref: PyTypeConcreteKey<'ty>,
    ) -> Vec<Attribute<'ty, Concrete>> {
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
        env: &Arc<RegistryEnv<'ty>>,
        kind: RegistryProjectionKind,
        type_ref: PyTypeConcreteKey<'ty>,
    ) -> RegistrySingleProjectionSupport<'ty> {
        let domain = RegistryProjectionDomain {
            kind,
            type_family: self.canonical_unqualified_concrete(type_ref),
            ignored_sources: BTreeSet::new(),
        };
        let expected = self.projection_snapshot(env, &domain);
        RegistrySingleProjectionSupport { domain, expected }
    }

    fn projection_snapshot(
        &mut self,
        env: &Arc<RegistryEnv<'ty>>,
        domain: &RegistryProjectionDomain<'ty>,
    ) -> RegistryProjectionSnapshot<'ty> {
        let base = self.base_projection_snapshot(env, domain.kind.clone(), domain.type_family);
        if domain.ignored_sources.is_empty() {
            return base.as_ref().clone();
        }

        base.filter_ignored_sources(&domain.ignored_sources)
    }

    fn projection_snapshot_matches(
        &mut self,
        env: &Arc<RegistryEnv<'ty>>,
        support: &[RegistrySingleProjectionSupport<'ty>],
    ) -> bool {
        let matches_single = |shared: &mut Self, support: &RegistrySingleProjectionSupport<'ty>| {
            let base = shared.base_projection_snapshot(
                env,
                support.domain.kind.clone(),
                support.domain.type_family,
            );
            if support.domain.ignored_sources.is_empty() {
                return support.expected == *base.as_ref();
            }

            base.matches_filtered(&support.expected, &support.domain.ignored_sources)
        };

        support.iter().all(|support| matches_single(self, support))
    }

    fn base_projection_snapshot(
        &mut self,
        env: &Arc<RegistryEnv<'ty>>,
        kind: RegistryProjectionKind,
        type_family: PyTypeConcreteKey<'ty>,
    ) -> Arc<RegistryProjectionSnapshot<'ty>> {
        let key = RegistryProjectionCacheKey {
            env: Arc::clone(env),
            kind: kind.clone(),
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
        env: &Arc<RegistryEnv<'ty>>,
        kind: RegistryProjectionKind,
        type_family: PyTypeConcreteKey<'ty>,
    ) -> RegistryProjectionSnapshot<'ty> {
        match kind {
            RegistryProjectionKind::Constants(mode) => RegistryProjectionSnapshot::Constants(
                self.projection_constants(env, &mode, type_family),
            ),
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
        env: &Arc<RegistryEnv<'ty>>,
        mode: &ConstantProjectionMode,
        type_ref: PyTypeConcreteKey<'ty>,
    ) -> BTreeSet<Source<'ty>> {
        let local_state = self
            .env_local_caches
            .entry(Arc::clone(env))
            .or_insert_with(|| Self::build_local_state(env, &mut self.types));
        let entries = match mode {
            ConstantProjectionMode::Unnamed => local_state
                .unqualified_constants
                .get(type_ref, &mut self.types),
            ConstantProjectionMode::Named(name) => local_state
                .named_constants
                .get(name)
                .and_then(|constants| constants.get(type_ref, &mut self.types)),
        }
        .cloned()
        .unwrap_or_default();

        entries.into_iter().collect()
    }

    fn projection_properties(
        &mut self,
        env: &Arc<RegistryEnv<'ty>>,
        type_ref: PyTypeConcreteKey<'ty>,
    ) -> BTreeSet<Property<'ty, Concrete>> {
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
        env: &Arc<RegistryEnv<'ty>>,
        type_ref: PyTypeConcreteKey<'ty>,
    ) -> BTreeSet<Attribute<'ty, Concrete>> {
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

pub(crate) struct RegistryEnv<'ty> {
    unnamed_constants: SourceSet<'ty>,
    named_constants: NamedSourceSets<'ty>,
    hash: u64,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct RegistryEnvDelta<'ty> {
    unnamed_added: SourceSet<'ty>,
    unnamed_removed: SourceSet<'ty>,
    named_added: NamedSourceSets<'ty>,
    named_removed: NamedSourceSets<'ty>,
}

impl std::fmt::Debug for RegistryEnvDelta<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryEnvDelta")
            .field("unnamed_added", &self.unnamed_added.len())
            .field("unnamed_removed", &self.unnamed_removed.len())
            .field("named_added", &self.named_added.len())
            .field("named_removed", &self.named_removed.len())
            .finish()
    }
}

impl RegistryEnvDelta<'_> {
    fn is_empty(&self) -> bool {
        self.unnamed_added.is_empty()
            && self.unnamed_removed.is_empty()
            && self.named_added.values().all(BTreeSet::is_empty)
            && self.named_removed.values().all(BTreeSet::is_empty)
    }
}

fn empty_env_delta<'ty>() -> RegistryEnvDelta<'ty> {
    RegistryEnvDelta {
        unnamed_added: BTreeSet::new(),
        unnamed_removed: BTreeSet::new(),
        named_added: BTreeMap::new(),
        named_removed: BTreeMap::new(),
    }
}

fn set_added<'ty>(parent: &SourceSet<'ty>, child: &SourceSet<'ty>) -> SourceSet<'ty> {
    child.difference(parent).cloned().collect()
}

fn set_removed<'ty>(parent: &SourceSet<'ty>, child: &SourceSet<'ty>) -> SourceSet<'ty> {
    parent.difference(child).cloned().collect()
}

fn compose_set_delta<'ty>(
    first_added: &SourceSet<'ty>,
    first_removed: &SourceSet<'ty>,
    second_added: &SourceSet<'ty>,
    second_removed: &SourceSet<'ty>,
) -> (SourceSet<'ty>, SourceSet<'ty>) {
    let added = first_added
        .difference(second_removed)
        .chain(second_added.difference(first_removed))
        .cloned()
        .collect();
    let removed = first_removed
        .difference(second_added)
        .chain(second_removed.difference(first_added))
        .cloned()
        .collect();
    (added, removed)
}

fn compose_named_delta<'ty>(
    first_added: &NamedSourceSets<'ty>,
    first_removed: &NamedSourceSets<'ty>,
    second_added: &NamedSourceSets<'ty>,
    second_removed: &NamedSourceSets<'ty>,
) -> (NamedSourceSets<'ty>, NamedSourceSets<'ty>) {
    let names: BTreeSet<_> = first_added
        .keys()
        .chain(first_removed.keys())
        .chain(second_added.keys())
        .chain(second_removed.keys())
        .cloned()
        .collect();
    let empty = BTreeSet::new();
    let mut added_by_name = BTreeMap::new();
    let mut removed_by_name = BTreeMap::new();

    for name in names {
        let (added, removed) = compose_set_delta(
            first_added.get(&name).unwrap_or(&empty),
            first_removed.get(&name).unwrap_or(&empty),
            second_added.get(&name).unwrap_or(&empty),
            second_removed.get(&name).unwrap_or(&empty),
        );
        if !added.is_empty() {
            added_by_name.insert(Arc::clone(&name), added);
        }
        if !removed.is_empty() {
            removed_by_name.insert(name, removed);
        }
    }

    (added_by_name, removed_by_name)
}

fn source_shadows<'ty>(new: &Source<'ty>, old: &Source<'ty>, types: &TypeArenas<'ty>) -> bool {
    let Some(new_type) = new.transition_type_ref() else {
        return false;
    };
    let Some(old_type) = old.transition_type_ref() else {
        return false;
    };

    types.deep_eq_concrete::<UnqualifiedMode>(new_type, old_type)
        && qualifier_matches(
            types.qualifier_of_concrete(old_type),
            types.qualifier_of_concrete(new_type),
        )
}

fn insert_unnamed_constant<'ty>(
    constants: &mut BTreeSet<Source<'ty>>,
    source: Source<'ty>,
    types: &TypeArenas<'ty>,
) {
    constants.retain(|existing| !source_shadows(&source, existing, types));
    constants.insert(source);
}

fn insert_named_constant<'ty>(
    constants: &mut BTreeMap<Arc<str>, BTreeSet<Source<'ty>>>,
    name: Arc<str>,
    source: Source<'ty>,
    types: &TypeArenas<'ty>,
) {
    let bucket = constants.entry(name).or_default();
    bucket.retain(|existing| !source_shadows(&source, existing, types));
    bucket.insert(source);
}

impl<'ty> RegistryEnv<'ty> {
    fn new(
        unnamed_constants: BTreeSet<Source<'ty>>,
        named_constants: BTreeMap<Arc<str>, BTreeSet<Source<'ty>>>,
    ) -> Self {
        let hash = hash_trace_value(&(unnamed_constants.clone(), named_constants.clone()));
        Self {
            unnamed_constants,
            named_constants,
            hash,
        }
    }

    pub(crate) fn transition_param_source(
        &self,
        name: Arc<str>,
        param_type: PyTypeConcreteKey<'ty>,
    ) -> Source<'ty> {
        Source::transition(Some(name), param_type)
    }

    pub(crate) fn transition_result_source(
        &self,
        return_type: PyTypeConcreteKey<'ty>,
    ) -> Source<'ty> {
        Source::transition(None, return_type)
    }

    #[instrumented(
        name = "inlay.registry_env.with_transition_sources",
        target = "inlay",
        level = "trace",
        ret,
        skip(sources, types),
        fields(
            parent_unnamed = self.unnamed_constants.len() as u64,
            sources = sources.len() as u64,
            child_items
        )
    )]
    pub(crate) fn with_transition_sources(
        &self,
        sources: Vec<Source<'ty>>,
        types: &TypeArenas<'ty>,
    ) -> Self {
        let mut unnamed_constants = self.unnamed_constants.clone();
        let mut named_constants = self.named_constants.clone();

        for source in sources {
            insert_unnamed_constant(&mut unnamed_constants, source.clone(), types);
            if let Some(name) = source.transition_name() {
                insert_named_constant(&mut named_constants, Arc::clone(name), source, types);
            }
        }

        let env = Self::new(unnamed_constants, named_constants);
        inlay_span_record!(child_items = env.unnamed_constants.len() as u64);
        env
    }
}

impl Default for RegistryEnv<'_> {
    fn default() -> Self {
        Self::new(BTreeSet::new(), BTreeMap::new())
    }
}

impl Clone for RegistryEnv<'_> {
    fn clone(&self) -> Self {
        Self {
            unnamed_constants: self.unnamed_constants.clone(),
            named_constants: self.named_constants.clone(),
            hash: self.hash,
        }
    }
}

impl PartialEq for RegistryEnv<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
            && self.unnamed_constants == other.unnamed_constants
            && self.named_constants == other.named_constants
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
            .field("unnamed_constants", &self.unnamed_constants.len())
            .field("named_constants", &self.named_constants.len())
            .finish()
    }
}

#[derive(PartialEq, Eq, Clone, Hash)]
pub(crate) enum ResolutionLookup<'ty> {
    Constant {
        type_ref: PyTypeConcreteKey<'ty>,
        requested_name: Option<Arc<str>>,
    },
    Property(PyTypeConcreteKey<'ty>),
    Attribute(PyTypeConcreteKey<'ty>),
}

impl std::fmt::Debug for ResolutionLookup<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&summarize_lookup_for_trace(self))
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum ResolutionLookupResult<'ty> {
    Constants {
        entries: BTreeSet<Source<'ty>>,
        used_fallback: bool,
    },
    Properties(BTreeSet<Property<'ty, Concrete>>),
    Attributes(BTreeSet<Attribute<'ty, Concrete>>),
}

impl std::fmt::Debug for ResolutionLookupResult<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&summarize_lookup_result_for_trace(self))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RegistryProjectionKind {
    Constants(ConstantProjectionMode),
    Properties,
    Attributes,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ConstantProjectionMode {
    Unnamed,
    Named(Arc<str>),
}

#[derive(Clone)]
struct RegistryProjectionCacheKey<'ty> {
    env: Arc<RegistryEnv<'ty>>,
    kind: RegistryProjectionKind,
    type_family: PyTypeConcreteKey<'ty>,
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
pub(crate) struct RegistryProjectionDomain<'ty> {
    kind: RegistryProjectionKind,
    type_family: PyTypeConcreteKey<'ty>,
    ignored_sources: BTreeSet<Source<'ty>>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct RegistrySingleProjectionSupport<'ty> {
    domain: RegistryProjectionDomain<'ty>,
    expected: RegistryProjectionSnapshot<'ty>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum RegistryProjectionSnapshot<'ty> {
    Constants(BTreeSet<Source<'ty>>),
    Properties(BTreeSet<Property<'ty, Concrete>>),
    Attributes(BTreeSet<Attribute<'ty, Concrete>>),
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

impl std::fmt::Debug for RegistrySingleProjectionSupport<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistrySingleProjectionSupport")
            .field("domain", &self.domain)
            .field("expected", &self.expected.len())
            .finish()
    }
}

impl<'ty> RegistryProjectionSnapshot<'ty> {
    fn matches_filtered(&self, expected: &Self, ignored_sources: &BTreeSet<Source<'ty>>) -> bool {
        match (self, expected) {
            (Self::Constants(current), Self::Constants(expected)) => {
                let mut visible = 0;
                for entry in current {
                    if ignored_sources.contains(entry) {
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

    fn filter_ignored_sources(&self, ignored_sources: &BTreeSet<Source<'ty>>) -> Self {
        match self {
            Self::Constants(entries) => Self::Constants(
                entries
                    .iter()
                    .filter(|source| !ignored_sources.contains(source))
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

fn merge_single_projection_support<'ty>(
    left: &RegistrySingleProjectionSupport<'ty>,
    right: &RegistrySingleProjectionSupport<'ty>,
) -> Option<RegistrySingleProjectionSupport<'ty>> {
    if left.domain.kind != right.domain.kind || left.domain.type_family != right.domain.type_family
    {
        return None;
    }

    let domain = RegistryProjectionDomain {
        kind: left.domain.kind.clone(),
        type_family: left.domain.type_family,
        ignored_sources: left
            .domain
            .ignored_sources
            .intersection(&right.domain.ignored_sources)
            .cloned()
            .collect(),
    };
    Some(RegistrySingleProjectionSupport {
        expected: left
            .expected
            .union(&right.expected)?
            .filter_ignored_sources(&domain.ignored_sources),
        domain,
    })
}

impl RuleLookupSupport for RegistrySingleProjectionSupport<'_> {
    fn merge_lookup_support(&self, other: &Self) -> Option<Self> {
        merge_single_projection_support(self, other)
    }
}

impl<'ty> ResolutionEnv for RegistryEnv<'ty> {
    type SharedState = RegistrySharedState<'ty>;
    type Query = ResolutionLookup<'ty>;
    type QueryResult = ResolutionLookupResult<'ty>;
    type DependencyEnvDelta = RegistryEnvDelta<'ty>;
    type LookupSupport = Vec<RegistrySingleProjectionSupport<'ty>>;

    #[instrumented(
        name = "inlay.registry_env.lookup",
        target = "inlay",
        level = "trace",
        ret,
        fields(
            query_hash = hash_trace_value(query),
            env_items = self.unnamed_constants.len() as u64,
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
            } => {
                let (entries, used_fallback) =
                    shared_state.lookup_constants(self, *type_ref, requested_name.as_ref());
                ResolutionLookupResult::Constants {
                    entries: entries.into_iter().collect(),
                    used_fallback,
                }
            }
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
        result: &Self::QueryResult,
    ) -> Self::LookupSupport {
        match query {
            ResolutionLookup::Constant {
                type_ref,
                requested_name,
            } => {
                let named_fallback = matches!(
                    result,
                    ResolutionLookupResult::Constants {
                        used_fallback: true,
                        ..
                    }
                );
                match requested_name {
                    Some(name) if named_fallback => vec![
                        shared_state.projection_support(
                            self,
                            RegistryProjectionKind::Constants(ConstantProjectionMode::Named(
                                Arc::clone(name),
                            )),
                            *type_ref,
                        ),
                        shared_state.projection_support(
                            self,
                            RegistryProjectionKind::Constants(ConstantProjectionMode::Unnamed),
                            *type_ref,
                        ),
                    ],
                    Some(name) => vec![shared_state.projection_support(
                        self,
                        RegistryProjectionKind::Constants(ConstantProjectionMode::Named(
                            Arc::clone(name),
                        )),
                        *type_ref,
                    )],
                    None => vec![shared_state.projection_support(
                        self,
                        RegistryProjectionKind::Constants(ConstantProjectionMode::Unnamed),
                        *type_ref,
                    )],
                }
            }
            ResolutionLookup::Property(type_ref) => {
                vec![shared_state.projection_support(
                    self,
                    RegistryProjectionKind::Properties,
                    *type_ref,
                )]
            }
            ResolutionLookup::Attribute(type_ref) => {
                vec![shared_state.projection_support(
                    self,
                    RegistryProjectionKind::Attributes,
                    *type_ref,
                )]
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

        fn pullback_single<'ty>(
            support: &RegistrySingleProjectionSupport<'ty>,
            delta: &RegistryEnvDelta<'ty>,
        ) -> RegistrySingleProjectionSupport<'ty> {
            let mut ignored_sources = support.domain.ignored_sources.clone();
            match &support.domain.kind {
                RegistryProjectionKind::Constants(ConstantProjectionMode::Unnamed)
                | RegistryProjectionKind::Properties
                | RegistryProjectionKind::Attributes => {
                    ignored_sources.extend(delta.unnamed_added.iter().cloned());
                    ignored_sources.extend(delta.unnamed_removed.iter().cloned());
                }
                RegistryProjectionKind::Constants(ConstantProjectionMode::Named(name)) => {
                    if let Some(added) = delta.named_added.get(name) {
                        ignored_sources.extend(added.iter().cloned());
                    }
                    if let Some(removed) = delta.named_removed.get(name) {
                        ignored_sources.extend(removed.iter().cloned());
                    }
                }
            }
            let expected = support.expected.filter_ignored_sources(&ignored_sources);
            RegistrySingleProjectionSupport {
                domain: RegistryProjectionDomain {
                    kind: support.domain.kind.clone(),
                    type_family: support.domain.type_family,
                    ignored_sources,
                },
                expected,
            }
        }

        support
            .iter()
            .map(|support| pullback_single(support, delta))
            .collect()
    }

    fn dependency_env_delta(parent: &Arc<Self>, child: &Arc<Self>) -> Self::DependencyEnvDelta {
        if Arc::ptr_eq(parent, child) {
            return empty_env_delta();
        }

        let unnamed_added = set_added(&parent.unnamed_constants, &child.unnamed_constants);
        let unnamed_removed = set_removed(&parent.unnamed_constants, &child.unnamed_constants);

        let names: BTreeSet<_> = parent
            .named_constants
            .keys()
            .chain(child.named_constants.keys())
            .cloned()
            .collect();
        let empty = BTreeSet::new();
        let mut named_added = BTreeMap::new();
        let mut named_removed = BTreeMap::new();
        for name in names {
            let parent_set = parent.named_constants.get(&name).unwrap_or(&empty);
            let child_set = child.named_constants.get(&name).unwrap_or(&empty);
            let added = set_added(parent_set, child_set);
            let removed = set_removed(parent_set, child_set);
            if !added.is_empty() {
                named_added.insert(Arc::clone(&name), added);
            }
            if !removed.is_empty() {
                named_removed.insert(name, removed);
            }
        }

        Self::DependencyEnvDelta {
            unnamed_added,
            unnamed_removed,
            named_added,
            named_removed,
        }
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

        let (unnamed_added, unnamed_removed) = compose_set_delta(
            &first.unnamed_added,
            &first.unnamed_removed,
            &second.unnamed_added,
            &second.unnamed_removed,
        );
        let (named_added, named_removed) = compose_named_delta(
            &first.named_added,
            &first.named_removed,
            &second.named_added,
            &second.named_removed,
        );

        Self::DependencyEnvDelta {
            unnamed_added,
            unnamed_removed,
            named_added,
            named_removed,
        }
    }
}
