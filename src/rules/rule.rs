use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::mem;
use std::sync::Arc;

use context_solver::{
    Arena as ResultsArena, LazyDepthMode, ReplaceError, Rule as SolverRule, RuleContext, RunError,
    solve::{SolveError, SolveResult},
};
use inlay_instrument::{inlay_event, inlay_span_record, instrumented};
use rustc_hash::{FxHashSet as HashSet, FxHasher};

use crate::{
    python_identity::PythonIdentity,
    qualifier::{Qualifier, qualifier_matches},
    registry::{Constructor, Source, SourceType},
    types::{
        Concrete, Keyed, MemberAccessKind, ParamKind, ProtocolBase, PyType, PyTypeConcreteKey,
        Qual, SentinelTypeKind, TypeArenas, UnqualifiedMode, WrapperKind, requalify_concrete,
    },
};

use super::{
    MethodOverrideResolution, ResolutionError, RuleArena, RuleId, RuleMode, StaticPolicy,
    TransitionParam, TypeFamilyRules,
    env::{
        Attribute, BoundImplementation, ConstructorLookup, MethodLookup, Property, RegistryEnv,
        RegistryEnvDeltaRequest, RegistryEnvTag, ResolutionLookup, ResolutionLookupResult,
    },
};

type MethodLookupBases<'ty> = Vec<ProtocolBase<Qual<Keyed<'ty>>, Concrete>>;
type MethodLookupContext<'ty> = (Qualifier, MethodLookupBases<'ty>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct SolverResolutionRef(u32);

impl SolverResolutionRef {
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

type SolverResolutionResult<'ty> = Result<SolverResolvedNode<'ty>, ResolutionError<'ty>>;
type RegistryRuleContext<'a, 'solver, 'ty> = RuleContext<'a, 'solver, RegistryResolutionRule<'ty>>;
type RegistryRunError<'ty> = RunError<RegistryResolutionRule<'ty>>;
type RegistryRunResult<'ty, T> = Result<T, RegistryRunError<'ty>>;
type MemberResolutionMap = BTreeMap<Arc<str>, SolverResolutionRef>;
type MemberResolutionErrors<'ty> = Vec<Arc<ResolutionError<'ty>>>;
type MemberResolutionResult<'ty> = Result<MemberResolutionMap, MemberResolutionErrors<'ty>>;
type CandidateResolution<'ty> =
    Result<ResolvedRule<SolverResolutionNode<'ty>>, Vec<Arc<ResolutionError<'ty>>>>;
type MethodMember<'ty> = (Arc<str>, PyTypeConcreteKey<'ty>);
type ResolvedParams = Vec<(SolverResolutionRef, Arc<str>, ParamKind)>;

#[derive(Debug)]
struct ResolvedRule<T> {
    resolution: T,
    dynamic: bool,
}

impl<T> ResolvedRule<T> {
    fn static_(resolution: T) -> Self {
        Self {
            resolution,
            dynamic: false,
        }
    }

    fn dynamic(resolution: T) -> Self {
        Self {
            resolution,
            dynamic: true,
        }
    }

    fn map<U>(self, f: impl FnOnce(T) -> U) -> ResolvedRule<U> {
        ResolvedRule {
            resolution: f(self.resolution),
            dynamic: self.dynamic,
        }
    }
}

#[derive(Debug)]
struct ResolvedChild {
    result_ref: SolverResolutionRef,
    dynamic: bool,
}

struct CallableImplementationCandidate<'ty> {
    public_callable_key: crate::types::CallableKey<'ty, Concrete>,
    implementation_callable_key: crate::types::CallableKey<'ty, Concrete>,
    implementation: SolverTransitionImplementationCallable<'ty>,
    bound_to: Option<PyTypeConcreteKey<'ty>>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct SolverRuntimeUnionBranch<'ty> {
    pub(crate) implementation_variant: PyTypeConcreteKey<'ty>,
    pub(crate) target: SolverResolutionRef,
    pub(crate) arm_source: Source<'ty>,
}

#[derive(Clone, Copy)]
enum TypeFamily {
    Sentinel,
    ParamSpec,
    Plain,
    Class,
    Protocol,
    TypedDict,
    Union,
    Callable,
    CallableImplementation,
    ReadCell,
    Cell,
    TypeVar,
}

impl TypeFamily {
    fn of(type_ref: PyTypeConcreteKey<'_>) -> Self {
        match type_ref {
            PyType::Sentinel(_) => Self::Sentinel,
            PyType::ParamSpec(_) => Self::ParamSpec,
            PyType::Plain(_) => Self::Plain,
            PyType::Class(_) => Self::Class,
            PyType::Protocol(_) => Self::Protocol,
            PyType::TypedDict(_) => Self::TypedDict,
            PyType::Union(_) => Self::Union,
            PyType::Callable(_) => Self::Callable,
            PyType::CallableImplementation(_) => Self::CallableImplementation,
            PyType::ReadCell(_) => Self::ReadCell,
            PyType::Cell(_) => Self::Cell,
            PyType::TypeVar(_) => Self::TypeVar,
        }
    }

    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    fn label(self) -> &'static str {
        match self {
            Self::Sentinel => "sentinel",
            Self::ParamSpec => "param_spec",
            Self::Plain => "plain",
            Self::Class => "class",
            Self::Protocol => "protocol",
            Self::TypedDict => "typed_dict",
            Self::Union => "union",
            Self::Callable => "callable",
            Self::CallableImplementation => "callable_implementation",
            Self::ReadCell => "read_cell",
            Self::Cell => "cell",
            Self::TypeVar => "type_var",
        }
    }

    fn rules(self, rules: &TypeFamilyRules) -> &[RuleId] {
        let selected = match self {
            Self::Sentinel => rules.sentinel.as_slice(),
            Self::ParamSpec => rules.param_spec.as_slice(),
            Self::Plain => rules.plain.as_slice(),
            Self::Class => rules.class_.as_slice(),
            Self::Protocol => rules.protocol.as_slice(),
            Self::TypedDict => rules.typed_dict.as_slice(),
            Self::Union => rules.union.as_slice(),
            Self::Callable => rules.callable.as_slice(),
            Self::CallableImplementation => rules.fallback.as_slice(),
            Self::ReadCell => rules.read_cell.as_slice(),
            Self::Cell => rules.cell.as_slice(),
            Self::TypeVar => rules.type_var.as_slice(),
        };
        if selected.is_empty() {
            rules.fallback.as_slice()
        } else {
            selected
        }
    }
}

fn debug_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = FxHasher::default();
    value.hash(&mut hasher);
    hasher.finish()
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct ResolutionQuery<'ty> {
    pub(crate) type_ref: PyTypeConcreteKey<'ty>,
    pub(crate) requested_name: Option<Arc<str>>,
    pub(crate) method_protocol: Option<PyTypeConcreteKey<'ty>>,
}

impl<'ty> ResolutionQuery<'ty> {
    pub(crate) fn unnamed(type_ref: PyTypeConcreteKey<'ty>) -> Self {
        Self {
            type_ref,
            requested_name: None,
            method_protocol: None,
        }
    }

    pub(crate) fn named(type_ref: PyTypeConcreteKey<'ty>, requested_name: Arc<str>) -> Self {
        Self {
            type_ref,
            requested_name: Some(requested_name),
            method_protocol: None,
        }
    }

    pub(crate) fn method(
        type_ref: PyTypeConcreteKey<'ty>,
        requested_name: Arc<str>,
        method_protocol: PyTypeConcreteKey<'ty>,
    ) -> Self {
        Self {
            type_ref,
            requested_name: Some(requested_name),
            method_protocol: Some(method_protocol),
        }
    }
}

impl std::fmt::Debug for ResolutionQuery<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolutionQuery")
            .field("type_hash", &debug_hash(&self.type_ref))
            .field("requested_name", &self.requested_name)
            .field(
                "method_protocol_hash",
                &self.method_protocol.as_ref().map(debug_hash),
            )
            .finish()
    }
}

#[derive(Default)]
pub(crate) struct SolverResolutionArena<'ty> {
    results: Vec<Option<SolverResolutionResult<'ty>>>,
}

impl std::fmt::Debug for SolverResolutionArena<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolverResolutionArena")
            .field("results", &self.results.len())
            .finish()
    }
}

impl<'ty> ResultsArena<SolverResolutionResult<'ty>> for SolverResolutionArena<'ty> {
    type Key = SolverResolutionRef;

    fn insert(&mut self, val: SolverResolutionResult<'ty>) -> Self::Key
    where
        SolverResolutionResult<'ty>: std::hash::Hash + Eq,
    {
        let key = SolverResolutionRef(
            self.results
                .len()
                .try_into()
                .expect("solver result arena cannot exceed u32::MAX entries"),
        );
        self.results.push(Some(val));
        key
    }

    fn insert_placeholder(&mut self) -> Self::Key {
        let key = SolverResolutionRef(
            self.results
                .len()
                .try_into()
                .expect("solver result arena cannot exceed u32::MAX entries"),
        );
        self.results.push(None);
        key
    }

    fn replace(
        &mut self,
        key: Self::Key,
        val: SolverResolutionResult<'ty>,
    ) -> Result<Option<SolverResolutionResult<'ty>>, ReplaceError>
    where
        SolverResolutionResult<'ty>: std::hash::Hash + Eq,
    {
        Ok(self
            .results
            .get_mut(key.index())
            .ok_or(ReplaceError::InvalidKey)?
            .replace(val))
    }

    fn get(&self, key: &Self::Key) -> Option<&SolverResolutionResult<'ty>> {
        self.results.get(key.index())?.as_ref()
    }

    fn len(&self) -> usize {
        self.results.len()
    }
}

pub(crate) enum SolverTransitionImplementationCallable<'ty> {
    Static(Arc<pyo3::Py<pyo3::PyAny>>),
    Source(Source<'ty>),
}

impl Clone for SolverTransitionImplementationCallable<'_> {
    fn clone(&self) -> Self {
        match self {
            Self::Static(implementation) => Self::Static(Arc::clone(implementation)),
            Self::Source(source) => Self::Source(source.clone()),
        }
    }
}

impl PartialEq for SolverTransitionImplementationCallable<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Static(left), Self::Static(right)) => {
                PythonIdentity::from_arc_py_any(left) == PythonIdentity::from_arc_py_any(right)
            }
            (Self::Source(left), Self::Source(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for SolverTransitionImplementationCallable<'_> {}

impl Hash for SolverTransitionImplementationCallable<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Static(implementation) => {
                PythonIdentity::from_arc_py_any(implementation).hash(state);
            }
            Self::Source(source) => source.hash(state),
        }
    }
}

impl std::fmt::Debug for SolverTransitionImplementationCallable<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static(_) => f.write_str("Static"),
            Self::Source(_) => f.write_str("Source"),
        }
    }
}

pub(crate) struct SolverResolvedTransitionImplementation<'ty> {
    pub(crate) implementation: SolverTransitionImplementationCallable<'ty>,
    pub(crate) bound_to: Option<SolverResolutionRef>,
    pub(crate) params: Vec<(SolverResolutionRef, Arc<str>, ParamKind)>,
    pub(crate) return_wrapper: WrapperKind,
    pub(crate) result_source: Option<Source<'ty>>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct SolverResolvedTransition<'ty> {
    pub(crate) return_wrapper: WrapperKind,
    pub(crate) accepts_varargs: bool,
    pub(crate) accepts_varkw: bool,
    pub(crate) params: Vec<TransitionParam<'ty>>,
    pub(crate) implementations: Vec<SolverResolvedTransitionImplementation<'ty>>,
    pub(crate) target: SolverResolutionRef,
}

impl Clone for SolverResolvedTransitionImplementation<'_> {
    fn clone(&self) -> Self {
        Self {
            implementation: self.implementation.clone(),
            bound_to: self.bound_to,
            params: self.params.clone(),
            return_wrapper: self.return_wrapper,
            result_source: self.result_source.clone(),
        }
    }
}

impl PartialEq for SolverResolvedTransitionImplementation<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.implementation == other.implementation
            && self.bound_to == other.bound_to
            && self.params == other.params
            && self.return_wrapper == other.return_wrapper
            && self.result_source == other.result_source
    }
}

impl Eq for SolverResolvedTransitionImplementation<'_> {}

impl Hash for SolverResolvedTransitionImplementation<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.implementation.hash(state);
        self.bound_to.hash(state);
        self.params.hash(state);
        self.return_wrapper.hash(state);
        self.result_source.hash(state);
    }
}

#[derive(Clone)]
pub(crate) struct SolverInitImplementation {
    pub(crate) implementation: Arc<pyo3::Py<pyo3::PyAny>>,
}

impl PartialEq for SolverInitImplementation {
    fn eq(&self, other: &Self) -> bool {
        PythonIdentity::from_arc_py_any(&self.implementation)
            == PythonIdentity::from_arc_py_any(&other.implementation)
    }
}

impl Eq for SolverInitImplementation {}

impl Hash for SolverInitImplementation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        PythonIdentity::from_arc_py_any(&self.implementation).hash(state);
    }
}

impl std::fmt::Debug for SolverResolvedTransitionImplementation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolverResolvedTransitionImplementation")
            .field("bound_to", &self.bound_to)
            .field("params", &self.params.len())
            .field("return_wrapper", &self.return_wrapper)
            .field("implementation", &self.implementation)
            .field("has_result_source", &self.result_source.is_some())
            .finish()
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum SolverResolutionNode<'ty> {
    Constant {
        source: Source<'ty>,
    },
    Property {
        source: SolverResolutionRef,
        property_name: Arc<str>,
    },
    ReadCell {
        target: SolverResolutionRef,
    },
    Cell {
        target: SolverResolutionRef,
    },
    None,
    UnionVariant {
        target: SolverResolutionRef,
    },
    RuntimeUnionDispatch {
        source: Source<'ty>,
        branches: Vec<SolverRuntimeUnionBranch<'ty>>,
    },
    Protocol {
        members: BTreeMap<Arc<str>, SolverResolutionRef>,
    },
    TypedDict {
        members: BTreeMap<Arc<str>, SolverResolutionRef>,
    },
    Transition(SolverResolvedTransition<'ty>),
    Attribute {
        source: SolverResolutionRef,
        attribute_name: Arc<str>,
        access_kind: MemberAccessKind,
    },
    Constructor {
        implementation: Arc<Constructor<'ty>>,
        params: Vec<(SolverResolutionRef, Arc<str>, ParamKind)>,
    },
    Init {
        implementation: SolverInitImplementation,
        params: Vec<(SolverResolutionRef, Arc<str>, ParamKind)>,
    },
    Delegate(SolverResolutionRef),
}

impl std::fmt::Debug for SolverResolutionNode<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant { .. } => f.debug_struct("Constant").finish(),
            Self::Property {
                source,
                property_name,
            } => f
                .debug_struct("Property")
                .field("source", source)
                .field("property_name", property_name)
                .finish(),
            Self::ReadCell { target } => {
                f.debug_struct("ReadCell").field("target", target).finish()
            }
            Self::Cell { target } => f.debug_struct("Cell").field("target", target).finish(),
            Self::None => f.debug_struct("None").finish(),
            Self::UnionVariant { target } => f
                .debug_struct("UnionVariant")
                .field("target", target)
                .finish(),
            Self::RuntimeUnionDispatch { branches, .. } => f
                .debug_struct("RuntimeUnionDispatch")
                .field("branches", &branches.len())
                .finish(),
            Self::Protocol { members } => f
                .debug_struct("Protocol")
                .field("members", &members.len())
                .finish(),
            Self::TypedDict { members } => f
                .debug_struct("TypedDict")
                .field("members", &members.len())
                .finish(),
            Self::Transition(transition) => f
                .debug_struct("Transition")
                .field("params", &transition.params.len())
                .field("implementations", &transition.implementations.len())
                .field("target", &transition.target)
                .finish(),
            Self::Attribute {
                source,
                attribute_name,
                access_kind,
            } => f
                .debug_struct("Attribute")
                .field("source", source)
                .field("attribute_name", attribute_name)
                .field("access_kind", access_kind)
                .finish(),
            Self::Constructor { params, .. } => f
                .debug_struct("Constructor")
                .field("params", &params.len())
                .finish(),
            Self::Init { params, .. } => f
                .debug_struct("Init")
                .field("params", &params.len())
                .finish(),
            Self::Delegate(result_ref) => f.debug_tuple("Delegate").field(result_ref).finish(),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum SolverWritableDependency<'ty> {
    Result(SolverResolutionRef),
    Source(Source<'ty>),
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct SolverResolvedNode<'ty> {
    pub(crate) target_type: PyTypeConcreteKey<'ty>,
    pub(crate) dynamic: bool,
    pub(crate) writable_dependencies: Vec<SolverWritableDependency<'ty>>,
    pub(crate) resolution: SolverResolutionNode<'ty>,
}

impl std::fmt::Debug for SolverResolvedNode<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolverResolvedNode")
            .field("target_hash", &debug_hash(&self.target_type))
            .field("dynamic", &self.dynamic)
            .field("writable_dependencies", &self.writable_dependencies.len())
            .field("resolution", &self.resolution)
            .finish()
    }
}

#[derive(Clone)]
pub(crate) struct RegistryResolutionRule<'ty> {
    rules: Arc<RuleArena>,
    _marker: std::marker::PhantomData<&'ty ()>,
}

impl std::fmt::Debug for RegistryResolutionRule<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryResolutionRule").finish()
    }
}

impl<'ty> RegistryResolutionRule<'ty> {
    pub(crate) fn new(rules: Arc<RuleArena>) -> Self {
        Self {
            rules,
            _marker: std::marker::PhantomData,
        }
    }

    fn rule_label(&self, rule_id: RuleId) -> &'static str {
        self.rules
            .get(rule_id)
            .map(RuleMode::label)
            .unwrap_or("unknown")
    }

    fn is_none_type(
        &self,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> bool {
        let PyType::Sentinel(key) = type_ref else {
            return false;
        };
        matches!(
            ctx.shared().types().sentinels.get(key).inner.value,
            SentinelTypeKind::None
        )
    }

    fn erased_static_delta() -> RegistryEnvDeltaRequest<'ty> {
        RegistryEnvDeltaRequest::identity()
            .remove_tag(RegistryEnvTag::Static)
            .erase_tag_support(RegistryEnvTag::Static)
    }

    fn required_static_delta() -> RegistryEnvDeltaRequest<'ty> {
        RegistryEnvDeltaRequest::identity()
            .add_tag(RegistryEnvTag::Static)
            .erase_tag_support(RegistryEnvTag::Static)
    }

    fn result_dynamic(
        result_ref: SolverResolutionRef,
        ctx: &RegistryRuleContext<'_, '_, 'ty>,
    ) -> bool {
        matches!(ctx.result(result_ref), Some(Ok(node)) if node.dynamic)
    }

    fn static_required(ctx: &mut RegistryRuleContext<'_, '_, 'ty>) -> bool {
        matches!(
            ctx.lookup(&ResolutionLookup::Tag(RegistryEnvTag::Static)),
            ResolutionLookupResult::TagPresent(true)
        )
    }

    fn solve_child_query(
        &self,
        query: ResolutionQuery<'ty>,
        state_id: RuleId,
        lazy_depth_mode: LazyDepthMode,
        delta: RegistryEnvDeltaRequest<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionRef> {
        let type_ref = query.type_ref;
        match ctx.solve_with_env_delta(query, state_id, lazy_depth_mode, delta) {
            Ok(SolveResult::Resolved { result, result_ref }) => match result {
                Ok(_) => Ok(result_ref),
                Err(err) => Err(RunError::Rule(err.clone())),
            },
            Ok(SolveResult::Lazy { result_ref }) => Ok(result_ref),
            Err(SolveError::SameDepthCycle) => {
                Err(RunError::Rule(ResolutionError::Cycle(type_ref)))
            }
            Err(error) => Err(RunError::Solve(error)),
        }
    }

    fn solve_child(
        &self,
        query: PyTypeConcreteKey<'ty>,
        state_id: RuleId,
        lazy_depth_mode: LazyDepthMode,
        delta: RegistryEnvDeltaRequest<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionRef> {
        self.solve_child_query(
            ResolutionQuery::unnamed(query),
            state_id,
            lazy_depth_mode,
            delta,
            ctx,
        )
    }

    fn solve_eager_child(
        &self,
        query: PyTypeConcreteKey<'ty>,
        state_id: RuleId,
        delta: RegistryEnvDeltaRequest<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, ResolvedChild> {
        match ctx.solve_with_env_delta(
            ResolutionQuery::unnamed(query),
            state_id,
            LazyDepthMode::Keep,
            delta,
        ) {
            Ok(SolveResult::Resolved { result, result_ref }) => match result {
                Ok(node) => Ok(ResolvedChild {
                    result_ref,
                    dynamic: node.dynamic,
                }),
                Err(err) => Err(RunError::Rule(err.clone())),
            },
            Ok(SolveResult::Lazy { .. }) | Err(SolveError::SameDepthCycle) => {
                Err(RunError::Rule(ResolutionError::Cycle(query)))
            }
            Err(error) => Err(RunError::Solve(error)),
        }
    }

    fn solve_child_named(
        &self,
        query: PyTypeConcreteKey<'ty>,
        requested_name: Arc<str>,
        state_id: RuleId,
        lazy_depth_mode: LazyDepthMode,
        delta: RegistryEnvDeltaRequest<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionRef> {
        self.solve_child_query(
            ResolutionQuery::named(query, requested_name),
            state_id,
            lazy_depth_mode,
            delta,
            ctx,
        )
    }

    fn lookup_constants(
        &self,
        query: &ResolutionQuery<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> Vec<Source<'ty>> {
        let ResolutionLookupResult::Constants { entries, .. } =
            ctx.lookup(&ResolutionLookup::Constant {
                type_ref: query.type_ref,
                requested_name: query.requested_name.clone(),
            })
        else {
            unreachable!();
        };
        entries.into_iter().collect()
    }

    fn lookup_bound_implementations(
        &self,
        public_type: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> Vec<BoundImplementation<'ty>> {
        let ResolutionLookupResult::BoundImplementations(entries) =
            ctx.lookup(&ResolutionLookup::BoundImplementation(public_type))
        else {
            unreachable!();
        };
        entries.into_iter().collect()
    }

    fn lookup_bound_union_implementations(
        &self,
        public_type: PyTypeConcreteKey<'ty>,
        arity: usize,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> Vec<BoundImplementation<'ty>> {
        let ResolutionLookupResult::BoundImplementations(entries) =
            ctx.lookup(&ResolutionLookup::BoundUnionImplementation { public_type, arity })
        else {
            unreachable!();
        };
        entries.into_iter().collect()
    }

    fn single_bound_implementation(
        &self,
        type_ref: PyTypeConcreteKey<'ty>,
        bindings: Vec<BoundImplementation<'ty>>,
    ) -> RegistryRunResult<'ty, BoundImplementation<'ty>> {
        match bindings.as_slice() {
            [binding] => Ok(binding.clone()),
            [] => Err(RunError::Rule(ResolutionError::NoBoundImplementationFound(
                type_ref,
            ))),
            _ => Err(RunError::Rule(
                ResolutionError::AmbiguousBoundImplementation(type_ref),
            )),
        }
    }

    fn lookup_constructors(
        &self,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> Vec<ConstructorLookup<'ty>> {
        let entries: BTreeSet<_> = ctx
            .shared()
            .lookup_constructors(type_ref)
            .into_iter()
            .collect();
        entries.into_iter().collect()
    }

    fn lookup_methods(
        &self,
        type_ref: PyTypeConcreteKey<'ty>,
        effective_protocol_qualifier: &Qualifier,
        lookup_bases: &[ProtocolBase<Qual<Keyed<'ty>>, Concrete>],
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> Vec<MethodLookup<'ty>> {
        let entries: BTreeSet<_> = ctx
            .shared()
            .lookup_methods(type_ref, effective_protocol_qualifier, lookup_bases)
            .into_iter()
            .collect();
        entries.into_iter().collect()
    }

    fn method_lookup_bases(
        &self,
        method_protocol: PyTypeConcreteKey<'ty>,
        method_name: &Arc<str>,
        override_resolution: MethodOverrideResolution,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> Result<MethodLookupContext<'ty>, ResolutionError<'ty>> {
        let PyType::Protocol(protocol_key) = method_protocol else {
            return Err(ResolutionError::IncompatibleType(method_protocol));
        };
        let types = ctx.shared().types();
        let protocol_type = types.concrete.protocols.get(protocol_key);
        let effective_protocol_qualifier = protocol_type.qualifier.clone();
        let mut protocol_mro = protocol_type.inner.protocol_mro.clone();
        if protocol_mro.is_empty() {
            protocol_mro.push(ProtocolBase {
                descriptor: protocol_type.inner.descriptor.clone(),
                type_params: protocol_type.inner.type_params.clone(),
                direct_methods: protocol_type.inner.direct_methods.clone(),
            });
        }

        let mut direct_declaration = None;
        for (index, protocol) in protocol_mro.iter().enumerate() {
            if protocol
                .direct_methods
                .iter()
                .any(|direct| direct.as_ref() == method_name.as_ref())
            {
                match override_resolution {
                    MethodOverrideResolution::Restrict => {
                        if direct_declaration.is_some() {
                            return Err(ResolutionError::MethodOverrideInLineage(method_protocol));
                        }
                        direct_declaration = Some(index);
                    }
                    MethodOverrideResolution::Closest => {
                        direct_declaration = Some(index);
                        break;
                    }
                }
            }
        }

        if let Some(index) = direct_declaration {
            protocol_mro.truncate(index + 1);
        }
        Ok((effective_protocol_qualifier, protocol_mro))
    }

    fn lookup_properties(
        &self,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> Vec<Property<'ty, crate::types::Concrete>> {
        let ResolutionLookupResult::Properties(entries) =
            ctx.lookup(&ResolutionLookup::Property(type_ref))
        else {
            unreachable!();
        };
        entries.into_iter().collect()
    }

    fn lookup_attributes(
        &self,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> Vec<Attribute<'ty, crate::types::Concrete>> {
        let ResolutionLookupResult::Attributes(entries) =
            ctx.lookup(&ResolutionLookup::Attribute(type_ref))
        else {
            unreachable!();
        };
        entries.into_iter().collect()
    }

    #[instrumented(
        name = "inlay.rule.resolve",
        target = "inlay",
        level = "trace",
        ret,
        err,
        fields(
            rule = rule.label(),
            type_hash = debug_hash(&query.type_ref),
            requested_name = query.requested_name.as_deref().unwrap_or("")
        )
    )]
    fn resolve_rule(
        &self,
        rule: RuleMode,
        query: &ResolutionQuery<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, ResolvedRule<SolverResolutionNode<'ty>>> {
        let type_ref = query.type_ref;
        inlay_event!(
            name: "inlay.rule.resolve.type",
            rule = rule.label(),
            type_hash = debug_hash(&type_ref),
            requested_name = query.requested_name.as_deref().unwrap_or(""),
        );
        match rule {
            RuleMode::Constant => self
                .resolve_constant(query, ctx)
                .map(ResolvedRule::static_)
                .map_err(RunError::Rule),
            RuleMode::Property { inner } => self.resolve_property(inner, query, ctx),
            RuleMode::ReadCell { inner } => self
                .resolve_read_cell(inner, type_ref, ctx)
                .map(ResolvedRule::static_),
            RuleMode::Cell { inner } => self
                .resolve_cell(inner, type_ref, ctx)
                .map(ResolvedRule::static_),
            RuleMode::Union { variant_rules } => self.resolve_union(variant_rules, type_ref, ctx),
            RuleMode::Protocol {
                property_rule,
                attribute_rule,
                method_rule,
            } => self
                .resolve_protocol(property_rule, attribute_rule, method_rule, type_ref, ctx)
                .map(ResolvedRule::dynamic),
            RuleMode::TypedDict { attribute_rule } => self
                .resolve_typed_dict(attribute_rule, type_ref, ctx)
                .map(ResolvedRule::dynamic),
            RuleMode::SentinelNone => self
                .resolve_sentinel_none(type_ref, ctx)
                .map(ResolvedRule::static_)
                .map_err(RunError::Rule),
            RuleMode::MethodImpl {
                target_rules,
                override_resolution,
            } => self
                .resolve_method_impl(
                    target_rules,
                    override_resolution,
                    type_ref,
                    query.requested_name.clone(),
                    query.method_protocol,
                    ctx,
                )
                .map(ResolvedRule::static_),
            RuleMode::BoundedCallable { target_rules } => {
                self.resolve_bounded_callable(target_rules, type_ref, ctx)
            }
            RuleMode::BoundedUnion { pointwise_rules } => {
                self.resolve_bounded_union(pointwise_rules, type_ref, ctx)
            }
            RuleMode::AttributeSource { inner } => self.resolve_attribute_source(inner, query, ctx),
            RuleMode::Constructor {
                param_rules,
                static_policy,
            } => self.resolve_constructor(param_rules, static_policy, type_ref, ctx),
            RuleMode::Init {
                param_rules,
                whitelist,
                blacklist,
                static_policy,
            } => self.resolve_init(
                param_rules,
                &whitelist,
                &blacklist,
                static_policy,
                query,
                ctx,
            ),
            RuleMode::MatchFirst { rules } => self.resolve_match_first(&rules, query, ctx),
            RuleMode::MatchByType { rules } => {
                self.resolve_match_by_type(rules.as_ref(), query, ctx)
            }
        }
    }

    #[instrumented(
        name = "inlay.rule.resolve_members",
        target = "inlay",
        level = "trace",
        ret,
        err,
        skip(members),
        fields(members = members.len() as u64, rule_id = rule_id.index() as u64)
    )]
    fn resolve_members(
        &self,
        members: &[(Arc<str>, PyTypeConcreteKey<'ty>)],
        rule_id: RuleId,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, MemberResolutionResult<'ty>> {
        let mut resolved = BTreeMap::new();
        let mut errors = Vec::new();

        for (name, member_type) in members {
            match self.solve_child_named(
                *member_type,
                Arc::clone(name),
                rule_id,
                LazyDepthMode::Keep,
                Self::erased_static_delta(),
                ctx,
            ) {
                Ok(result_ref) => {
                    resolved.insert(Arc::clone(name), result_ref);
                }
                Err(RunError::Rule(error)) => errors.push(Arc::new(ResolutionError::MemberError {
                    member_name: Arc::clone(name),
                    cause: Arc::new(error),
                })),
                Err(RunError::Solve(error)) => return Err(RunError::Solve(error)),
            }
        }

        if errors.is_empty() {
            Ok(Ok(resolved))
        } else {
            Ok(Err(errors))
        }
    }

    fn resolve_method_members(
        &self,
        members: &[MethodMember<'ty>],
        protocol: PyTypeConcreteKey<'ty>,
        rule_id: RuleId,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, MemberResolutionResult<'ty>> {
        let mut resolved = BTreeMap::new();
        let mut errors = Vec::new();

        for (name, member_type) in members {
            match self.solve_child_query(
                ResolutionQuery::method(*member_type, Arc::clone(name), protocol),
                rule_id,
                LazyDepthMode::Keep,
                Self::erased_static_delta(),
                ctx,
            ) {
                Ok(result_ref) => {
                    resolved.insert(Arc::clone(name), result_ref);
                }
                Err(RunError::Rule(error)) => errors.push(Arc::new(ResolutionError::MemberError {
                    member_name: Arc::clone(name),
                    cause: Arc::new(error),
                })),
                Err(RunError::Solve(error)) => return Err(RunError::Solve(error)),
            }
        }

        if errors.is_empty() {
            Ok(Ok(resolved))
        } else {
            Ok(Err(errors))
        }
    }

    #[instrumented(
        name = "inlay.rule.resolve_constant",
        target = "inlay",
        level = "trace",
        ret,
        err,
        fields(
            type_hash = debug_hash(&query.type_ref),
            requested_name = query.requested_name.as_deref().unwrap_or("")
        )
    )]
    fn resolve_constant(
        &self,
        query: &ResolutionQuery<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> Result<SolverResolutionNode<'ty>, ResolutionError<'ty>> {
        let entries = self.lookup_constants(query, ctx);
        let type_ref = query.type_ref;
        if entries.is_empty() {
            return Err(ResolutionError::NoConstantFound(type_ref));
        }

        match entries.as_slice() {
            [source] => Ok(SolverResolutionNode::Constant {
                source: source.clone(),
            }),
            [] => Err(ResolutionError::NoConstantFound(type_ref)),
            _ => Err(ResolutionError::AmbiguousConstant(type_ref)),
        }
    }

    fn resolve_property_candidates(
        &self,
        candidates: &[&Property<'ty, Concrete>],
        inner: RuleId,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, CandidateResolution<'ty>> {
        let mut resolved = None;
        let mut resolved_keys = HashSet::default();
        let mut errors = Vec::new();

        for &property in candidates {
            let candidate_key = (property.source.clone(), Arc::clone(&property.name));
            if resolved_keys.contains(&candidate_key) {
                continue;
            }

            match self.solve_eager_child(
                PyType::Protocol(property.source_type),
                inner,
                Self::erased_static_delta(),
                ctx,
            ) {
                Ok(source) => {
                    if !resolved_keys.insert(candidate_key) {
                        continue;
                    }
                    if resolved.is_some() {
                        return Err(RunError::Rule(ResolutionError::AmbiguousProperty(type_ref)));
                    }
                    resolved = Some(SolverResolutionNode::Property {
                        source: source.result_ref,
                        property_name: Arc::clone(&property.name),
                    });
                }
                Err(RunError::Rule(error)) => errors.push(Arc::new(error)),
                Err(RunError::Solve(error)) => return Err(RunError::Solve(error)),
            }
        }

        Ok(match resolved {
            Some(node) => Ok(ResolvedRule::dynamic(node)),
            None => Err(errors),
        })
    }

    #[instrumented(
        name = "inlay.rule.resolve_property",
        target = "inlay",
        level = "trace",
        ret,
        err,
        skip(query),
        fields(
            type_hash = debug_hash(&query.type_ref),
            requested_name = query.requested_name.as_deref().unwrap_or(""),
            matched_properties
        )
    )]
    fn resolve_property(
        &self,
        inner: RuleId,
        query: &ResolutionQuery<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, ResolvedRule<SolverResolutionNode<'ty>>> {
        let type_ref = query.type_ref;
        let matched = self.lookup_properties(type_ref, ctx);
        inlay_span_record!(matched_properties = matched.len() as u64);

        if matched.is_empty() {
            return Err(RunError::Rule(ResolutionError::NoPropertyFound(type_ref)));
        }

        match &query.requested_name {
            Some(requester_name) => {
                let (named, rest): (Vec<_>, Vec<_>) = matched
                    .iter()
                    .partition(|property| property.name.as_ref() == requester_name.as_ref());

                let named_errors = match self.resolve_property_candidates(
                    named.as_slice(),
                    inner,
                    type_ref,
                    ctx,
                )? {
                    Ok(node) => return Ok(node),
                    Err(errors) => errors,
                };

                let unnamed_errors = match self.resolve_property_candidates(
                    rest.as_slice(),
                    inner,
                    type_ref,
                    ctx,
                )? {
                    Ok(node) => return Ok(node),
                    Err(errors) => errors,
                };

                Err(RunError::Rule(ResolutionError::MissingDependency(
                    type_ref,
                    [named_errors, unnamed_errors].concat(),
                )))
            }
            None => {
                let all = matched.iter().collect::<Vec<_>>();
                let errors =
                    match self.resolve_property_candidates(all.as_slice(), inner, type_ref, ctx)? {
                        Ok(node) => return Ok(node),
                        Err(errors) => errors,
                    };
                Err(RunError::Rule(ResolutionError::MissingDependency(
                    type_ref, errors,
                )))
            }
        }
    }

    #[instrumented(
        name = "inlay.rule.resolve_read_cell",
        target = "inlay",
        level = "trace",
        ret,
        err,
        skip(type_ref),
        fields(type_hash = debug_hash(&type_ref), inner_rule = inner.index() as u64)
    )]
    fn resolve_read_cell(
        &self,
        inner: RuleId,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionNode<'ty>> {
        let PyType::ReadCell(key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };
        let target = ctx
            .shared()
            .types()
            .concrete
            .read_cells
            .get(key)
            .inner
            .target;
        let target = self.solve_child(
            target,
            inner,
            LazyDepthMode::Increment,
            Self::erased_static_delta(),
            ctx,
        )?;
        Ok(SolverResolutionNode::ReadCell { target })
    }

    #[instrumented(
        name = "inlay.rule.resolve_cell",
        target = "inlay",
        level = "trace",
        ret,
        err,
        skip(type_ref),
        fields(type_hash = debug_hash(&type_ref), inner_rule = inner.index() as u64)
    )]
    fn resolve_cell(
        &self,
        inner: RuleId,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionNode<'ty>> {
        let PyType::Cell(key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };
        let target = ctx.shared().types().concrete.cells.get(key).inner.target;
        let target = self.solve_child(
            target,
            inner,
            LazyDepthMode::Increment,
            Self::required_static_delta(),
            ctx,
        )?;
        Ok(SolverResolutionNode::Cell { target })
    }

    #[instrumented(
        name = "inlay.rule.resolve_union",
        target = "inlay",
        level = "trace",
        ret,
        err,
        skip(type_ref),
        fields(
            type_hash = debug_hash(&type_ref),
            variant_rule = variant_rules.index() as u64,
            variants,
            resolved_variants,
            errors
        )
    )]
    fn resolve_union(
        &self,
        variant_rules: RuleId,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, ResolvedRule<SolverResolutionNode<'ty>>> {
        let PyType::Union(key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };
        let variants = ctx
            .shared()
            .types()
            .concrete
            .unions
            .get(key)
            .inner
            .variants
            .clone();

        let mut resolved = Vec::new();
        let mut errors = Vec::new();
        inlay_span_record!(variants = variants.len() as u64);
        for &variant in &variants {
            match ctx.solve(
                ResolutionQuery::unnamed(variant),
                variant_rules,
                LazyDepthMode::Keep,
            ) {
                Ok(SolveResult::Resolved { result, result_ref }) => match result {
                    Ok(_) => resolved.push((variant, result_ref)),
                    Err(error) => errors.push(Arc::new(error.clone())),
                },
                Ok(SolveResult::Lazy { result_ref }) => resolved.push((variant, result_ref)),
                Err(SolveError::SameDepthCycle) => {
                    errors.push(Arc::new(ResolutionError::Cycle(variant)));
                }
                Err(error) => return Err(RunError::Solve(error)),
            }
        }
        inlay_span_record!(
            resolved_variants = resolved.len() as u64,
            errors = errors.len() as u64
        );

        if !resolved.is_empty() {
            let types = ctx.shared().types();
            resolved.sort_by(|left, right| {
                union_subtype_sort_key(left.0, &variants, &*types)
                    .cmp(&union_subtype_sort_key(right.0, &variants, &*types))
            });
            let target = resolved[0].1;
            Ok(ResolvedRule {
                resolution: SolverResolutionNode::UnionVariant { target },
                dynamic: Self::result_dynamic(target, ctx),
            })
        } else {
            Err(RunError::Rule(ResolutionError::MissingDependency(
                type_ref, errors,
            )))
        }
    }

    fn same_unqualified_type(
        &self,
        left: PyTypeConcreteKey<'ty>,
        right: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> bool {
        ctx.shared()
            .types()
            .deep_eq_concrete::<UnqualifiedMode>(left, right)
    }

    fn qualifier_compatible(
        &self,
        public_type: PyTypeConcreteKey<'ty>,
        implementation_type: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> bool {
        let types = ctx.shared().types();
        qualifier_matches(
            types.qualifier_of_concrete(public_type),
            types.qualifier_of_concrete(implementation_type),
        )
    }

    fn runtime_union_variant_matchable(
        &self,
        implementation_variant: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> bool {
        match implementation_variant {
            PyType::Sentinel(key) => matches!(
                ctx.shared().types().sentinels.get(key).inner.value,
                SentinelTypeKind::None
            ),
            PyType::Plain(_) | PyType::Class(_) | PyType::Callable(_) => true,
            _ => false,
        }
    }

    fn narrowed_arm_source(
        &self,
        source: &Source<'ty>,
        implementation_variant: PyTypeConcreteKey<'ty>,
    ) -> Source<'ty> {
        Source::transition(source.transition_name().cloned(), implementation_variant)
    }

    fn branch_delta_with_narrowed_source(
        &self,
        public_type: PyTypeConcreteKey<'ty>,
        implementation_type: PyTypeConcreteKey<'ty>,
        source: &Source<'ty>,
        arm_source: &Source<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryEnvDeltaRequest<'ty> {
        let mut delta = RegistryEnvDeltaRequest::identity()
            .replace_transition_source(source.clone(), arm_source.clone());
        if !self.same_unqualified_type(public_type, implementation_type, ctx)
            || !self.qualifier_compatible(public_type, implementation_type, ctx)
        {
            delta = delta.add_bound_implementation(BoundImplementation {
                public_type,
                implementation_type,
                source: arm_source.clone(),
            });
        }
        delta
    }

    fn resolve_bounded_callable(
        &self,
        target_rules: RuleId,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, ResolvedRule<SolverResolutionNode<'ty>>> {
        let PyType::Callable(public_key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };

        let bindings = self.lookup_bound_implementations(type_ref, ctx);
        let callable_bound_implementations: Vec<_> = bindings
            .iter()
            .filter(|binding| matches!(binding.implementation_type, PyType::Callable(_)))
            .cloned()
            .collect();
        if callable_bound_implementations.is_empty() {
            return self.resolve_bounded_callable_union(target_rules, type_ref, bindings, ctx);
        }
        let binding = self.single_bound_implementation(type_ref, callable_bound_implementations)?;
        let PyType::Callable(implementation_key) = binding.implementation_type else {
            unreachable!("binding filter ensures callable implementation type")
        };

        let candidates = vec![CallableImplementationCandidate {
            public_callable_key: public_key,
            implementation_callable_key: implementation_key,
            implementation: SolverTransitionImplementationCallable::Source(binding.source),
            bound_to: None,
        }];

        self.resolve_callable_transition(target_rules, public_key, candidates, ctx)
            .map(|transition| ResolvedRule::static_(SolverResolutionNode::Transition(transition)))
    }

    fn resolve_bounded_callable_union(
        &self,
        target_rules: RuleId,
        type_ref: PyTypeConcreteKey<'ty>,
        bindings: Vec<BoundImplementation<'ty>>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, ResolvedRule<SolverResolutionNode<'ty>>> {
        let mut applicable = Vec::new();
        for binding in bindings {
            let PyType::Union(union_key) = binding.implementation_type else {
                continue;
            };
            let implementation_variants = ctx
                .shared()
                .types()
                .concrete
                .unions
                .get(union_key)
                .inner
                .variants
                .clone();
            let mut branches = Vec::new();
            for implementation_variant in implementation_variants {
                if !matches!(implementation_variant, PyType::Callable(_)) {
                    continue;
                }
                let arm_source = self.narrowed_arm_source(&binding.source, implementation_variant);
                let branch_delta = self.branch_delta_with_narrowed_source(
                    type_ref,
                    implementation_variant,
                    &binding.source,
                    &arm_source,
                    ctx,
                );
                match self.solve_child(
                    type_ref,
                    target_rules,
                    LazyDepthMode::Keep,
                    branch_delta,
                    ctx,
                ) {
                    Ok(target) => branches.push(SolverRuntimeUnionBranch {
                        implementation_variant,
                        target,
                        arm_source,
                    }),
                    Err(RunError::Rule(_)) => {}
                    Err(error) => return Err(error),
                }
            }
            if !branches.is_empty() {
                applicable.push((binding.source, branches));
            }
        }

        match applicable.as_slice() {
            [(source, branches)] => Ok(Self::resolved_runtime_union_dispatch(
                source.clone(),
                branches.clone(),
                ctx,
            )),
            [] => Err(RunError::Rule(ResolutionError::NoBoundImplementationFound(
                type_ref,
            ))),
            _ => Err(RunError::Rule(
                ResolutionError::AmbiguousBoundImplementation(type_ref),
            )),
        }
    }

    fn resolved_runtime_union_dispatch(
        source: Source<'ty>,
        branches: Vec<SolverRuntimeUnionBranch<'ty>>,
        ctx: &RegistryRuleContext<'_, '_, 'ty>,
    ) -> ResolvedRule<SolverResolutionNode<'ty>> {
        let dynamic = branches
            .iter()
            .any(|branch| Self::result_dynamic(branch.target, ctx));
        ResolvedRule {
            resolution: SolverResolutionNode::RuntimeUnionDispatch { source, branches },
            dynamic,
        }
    }

    fn resolve_bounded_union_candidate(
        &self,
        pointwise_rules: RuleId,
        public_variants: &[PyTypeConcreteKey<'ty>],
        binding: BoundImplementation<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, Option<Vec<SolverRuntimeUnionBranch<'ty>>>> {
        let PyType::Union(implementation_union_key) = binding.implementation_type else {
            return Ok(None);
        };
        let implementation_variants = ctx
            .shared()
            .types()
            .concrete
            .unions
            .get(implementation_union_key)
            .inner
            .variants
            .clone();
        if implementation_variants.len() != public_variants.len() {
            return Ok(None);
        }

        let mut branches = Vec::with_capacity(public_variants.len());
        for (&public_variant, &implementation_variant) in
            public_variants.iter().zip(implementation_variants.iter())
        {
            if !self.runtime_union_variant_matchable(implementation_variant, ctx) {
                return Err(RunError::Rule(
                    ResolutionError::UnsupportedRuntimeUnionMatcher(implementation_variant),
                ));
            }
            let arm_source = self.narrowed_arm_source(&binding.source, implementation_variant);
            let branch_delta = self.branch_delta_with_narrowed_source(
                public_variant,
                implementation_variant,
                &binding.source,
                &arm_source,
                ctx,
            );
            match self.solve_child(
                public_variant,
                pointwise_rules,
                LazyDepthMode::Keep,
                branch_delta,
                ctx,
            ) {
                Ok(target) => branches.push(SolverRuntimeUnionBranch {
                    implementation_variant,
                    target,
                    arm_source,
                }),
                Err(RunError::Rule(_)) => return Ok(None),
                Err(error) => return Err(error),
            }
        }
        Ok(Some(branches))
    }

    fn resolve_bounded_union(
        &self,
        pointwise_rules: RuleId,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, ResolvedRule<SolverResolutionNode<'ty>>> {
        let PyType::Union(public_union_key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };
        let public_variants = ctx
            .shared()
            .types()
            .concrete
            .unions
            .get(public_union_key)
            .inner
            .variants
            .clone();
        let bindings =
            self.lookup_bound_union_implementations(type_ref, public_variants.len(), ctx);

        let mut applicable = Vec::new();
        for binding in bindings {
            let source = binding.source.clone();
            if let Some(branches) = self.resolve_bounded_union_candidate(
                pointwise_rules,
                &public_variants,
                binding,
                ctx,
            )? {
                applicable.push((source, branches));
            }
        }

        match applicable.as_slice() {
            [(source, branches)] => Ok(Self::resolved_runtime_union_dispatch(
                source.clone(),
                branches.clone(),
                ctx,
            )),
            [] => Err(RunError::Rule(ResolutionError::NoBoundImplementationFound(
                type_ref,
            ))),
            _ => Err(RunError::Rule(
                ResolutionError::AmbiguousBoundImplementation(type_ref),
            )),
        }
    }

    #[instrumented(
        name = "inlay.rule.resolve_protocol",
        target = "inlay",
        level = "trace",
        ret,
        err,
        skip(type_ref),
        fields(
            type_hash = debug_hash(&type_ref),
            property_rule = property_rule.index() as u64,
            attribute_rule = attribute_rule.index() as u64,
            method_rule = method_rule.index() as u64,
            property_members,
            attribute_members,
            method_members,
            errors
        )
    )]
    fn resolve_protocol(
        &self,
        property_rule: RuleId,
        attribute_rule: RuleId,
        method_rule: RuleId,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionNode<'ty>> {
        let PyType::Protocol(key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };
        let (property_members, attribute_members, method_members) = {
            let protocol = ctx.shared().types().concrete.protocols.get(key).clone();
            let property_members: Vec<_> = protocol
                .inner
                .properties
                .iter()
                .map(|(name, member_type)| (Arc::clone(name), *member_type))
                .collect();
            let attribute_members: Vec<_> = protocol
                .inner
                .attributes
                .iter()
                .map(|(name, member_type)| (Arc::clone(name), *member_type))
                .collect();
            let method_members: Vec<_> = protocol
                .inner
                .methods
                .iter()
                .map(|(name, method)| (Arc::clone(name), method.callable))
                .collect();
            (property_members, attribute_members, method_members)
        };
        inlay_event!(
            name: "inlay.rule.resolve_protocol.members",
            type_hash = debug_hash(&type_ref),
            property_members = property_members.len() as u64,
            attribute_members = attribute_members.len() as u64,
            method_members = method_members.len() as u64,
        );
        inlay_span_record!(
            property_members = property_members.len() as u64,
            attribute_members = attribute_members.len() as u64,
            method_members = method_members.len() as u64
        );

        let mut members = BTreeMap::new();
        let mut errors = Vec::new();

        for (rule_id, member_list) in [
            (property_rule, property_members.as_slice()),
            (attribute_rule, attribute_members.as_slice()),
        ] {
            match self.resolve_members(member_list, rule_id, ctx)? {
                Ok(resolved) => members.extend(resolved),
                Err(member_errors) => errors.extend(member_errors),
            }
        }
        match self.resolve_method_members(&method_members, type_ref, method_rule, ctx)? {
            Ok(resolved) => members.extend(resolved),
            Err(member_errors) => errors.extend(member_errors),
        }
        inlay_span_record!(errors = errors.len() as u64);

        if errors.is_empty() {
            Ok(SolverResolutionNode::Protocol { members })
        } else {
            Err(RunError::Rule(ResolutionError::MissingDependency(
                type_ref, errors,
            )))
        }
    }

    #[instrumented(
        name = "inlay.rule.resolve_typed_dict",
        target = "inlay",
        level = "trace",
        ret,
        err,
        skip(type_ref),
        fields(
            type_hash = debug_hash(&type_ref),
            attribute_rule = attribute_rule.index() as u64,
            attribute_members
        )
    )]
    fn resolve_typed_dict(
        &self,
        attribute_rule: RuleId,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionNode<'ty>> {
        let PyType::TypedDict(key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };
        let typed_dict = ctx.shared().types().concrete.typed_dicts.get(key);
        let mut required_members = Vec::new();
        let mut optional_members = Vec::new();
        for (name, member_type) in typed_dict.inner.attributes.iter() {
            let member = (Arc::clone(name), *member_type);
            if typed_dict.inner.required_keys.contains(name) {
                required_members.push(member);
            } else {
                optional_members.push(member);
            }
        }
        inlay_event!(
            name: "inlay.rule.resolve_typed_dict.members",
            type_hash = debug_hash(&type_ref),
            attribute_members = (required_members.len() + optional_members.len()) as u64,
        );
        inlay_span_record!(
            attribute_members = (required_members.len() + optional_members.len()) as u64
        );

        let mut members = match self.resolve_members(&required_members, attribute_rule, ctx)? {
            Ok(members) => members,
            Err(errors) => {
                return Err(RunError::Rule(ResolutionError::MissingDependency(
                    type_ref, errors,
                )));
            }
        };

        for (name, member_type) in optional_members {
            match self.solve_child_named(
                member_type,
                Arc::clone(&name),
                attribute_rule,
                LazyDepthMode::Keep,
                Self::erased_static_delta(),
                ctx,
            ) {
                Ok(result_ref) => {
                    members.insert(name, result_ref);
                }
                Err(RunError::Rule(_)) => {}
                Err(RunError::Solve(error)) => return Err(RunError::Solve(error)),
            }
        }

        Ok(SolverResolutionNode::TypedDict { members })
    }

    #[instrumented(
        name = "inlay.rule.resolve_sentinel_none",
        target = "inlay",
        level = "trace",
        ret,
        err,
        skip(type_ref),
        fields(type_hash = debug_hash(&type_ref))
    )]
    fn resolve_sentinel_none(
        &self,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> Result<SolverResolutionNode<'ty>, ResolutionError<'ty>> {
        let PyType::Sentinel(key) = type_ref else {
            return Err(ResolutionError::IncompatibleType(type_ref));
        };
        let sentinel = ctx.shared().types().sentinels.get(key);
        if matches!(sentinel.inner.value, SentinelTypeKind::None) {
            Ok(SolverResolutionNode::None)
        } else {
            Err(ResolutionError::IncompatibleType(type_ref))
        }
    }

    fn resolve_callable_transition(
        &self,
        target_rules: RuleId,
        request_key: crate::types::CallableKey<'ty, Concrete>,
        candidates: Vec<CallableImplementationCandidate<'ty>>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolvedTransition<'ty>> {
        let (request_result_type, return_wrapper, accepts_varargs, accepts_varkw, param_info) = {
            let types = ctx.shared().types();
            let callable = types.concrete.callables.get(request_key);
            let param_info: Vec<(Arc<str>, PyTypeConcreteKey<'ty>, ParamKind)> = callable
                .inner
                .params
                .iter()
                .zip(callable.inner.param_kinds.iter())
                .map(|((name, &param_type), &kind)| (Arc::clone(name), param_type, kind))
                .collect();
            (
                callable.inner.return_type,
                callable.inner.return_wrapper,
                callable.inner.accepts_varargs,
                callable.inner.accepts_varkw,
                param_info,
            )
        };

        let child_param_info: Vec<(Arc<str>, PyTypeConcreteKey<'ty>, ParamKind)> = {
            let types = ctx.shared().types();
            let child_qual = types.qualifier_of_concrete(request_result_type).clone();
            param_info
                .into_iter()
                .map(|(name, param_type, kind)| {
                    let param_type = requalify_concrete(param_type, &child_qual, types);
                    (name, param_type, kind)
                })
                .collect()
        };
        let mut params: Vec<TransitionParam<'ty>> = child_param_info
            .into_iter()
            .map(|(name, param_type, kind)| TransitionParam {
                logical_sources: BTreeSet::from([ctx
                    .env()
                    .transition_param_source(Arc::clone(&name), param_type)]),
                name,
                kind,
            })
            .collect();
        inlay_event!(
            name: "inlay.rule.resolve_callable_transition.params",
            type_hash = debug_hash(
                &PyType::<Qual<Keyed<'ty>>, Qual<Keyed<'ty>>, Concrete>::Callable(request_key)
            ),
            params = params.len() as u64,
        );
        inlay_span_record!(params = params.len() as u64);

        let child_param_sources: Vec<_> = params
            .iter()
            .flat_map(|param| param.logical_sources.iter().cloned())
            .collect();
        let mut result_delta = Self::erased_static_delta();
        let mut child_delta =
            Self::erased_static_delta().add_transition_sources(child_param_sources);
        let mut implementations = Vec::with_capacity(candidates.len());

        for candidate in candidates {
            let public_callable = ctx
                .shared()
                .types()
                .concrete
                .callables
                .get(candidate.public_callable_key)
                .clone();
            let requires_sources: Vec<_> = public_callable
                .inner
                .params
                .iter()
                .map(|(name, &param_type)| {
                    ctx.env()
                        .transition_param_source(Arc::clone(name), param_type)
                })
                .collect();
            for (param, source) in params.iter_mut().zip(requires_sources.iter()) {
                param.logical_sources.insert(source.clone());
            }
            let impl_delta = result_delta
                .clone()
                .add_transition_sources(requires_sources);

            let callable = ctx
                .shared()
                .types()
                .concrete
                .callables
                .get(candidate.implementation_callable_key)
                .clone();
            let param_info: Vec<(Arc<str>, PyTypeConcreteKey<'ty>, ParamKind, bool)> = callable
                .inner
                .params
                .iter()
                .zip(callable.inner.param_kinds.iter())
                .zip(callable.inner.param_has_default.iter())
                .map(|(((name, &param_type), &kind), &has_default)| {
                    (Arc::clone(name), param_type, kind, has_default)
                })
                .collect();
            let result_type = callable.inner.return_type;

            let bound_to = candidate.bound_to.map(|bound_type| {
                self.solve_child(
                    bound_type,
                    target_rules,
                    LazyDepthMode::Keep,
                    impl_delta.clone(),
                    ctx,
                )
            });
            let bound_to = match bound_to {
                Some(Ok(bound_to)) => Some(bound_to),
                Some(Err(error)) => return Err(error),
                None => None,
            };

            let mut implementation_params = Vec::with_capacity(param_info.len());
            for (name, param_type, kind, has_default) in param_info {
                match self.solve_child_named(
                    param_type,
                    Arc::clone(&name),
                    target_rules,
                    LazyDepthMode::Keep,
                    impl_delta.clone(),
                    ctx,
                ) {
                    Ok(result_ref) => implementation_params.push((result_ref, name, kind)),
                    Err(RunError::Rule(_)) if has_default => {}
                    Err(error) => return Err(error),
                }
            }

            let result_source = if self.is_none_type(result_type, ctx) {
                None
            } else {
                let result_source = ctx.env().transition_result_source(result_type);
                result_delta = result_delta.add_transition_sources(vec![result_source.clone()]);
                child_delta = child_delta.add_bound_implementation(BoundImplementation {
                    public_type: request_result_type,
                    implementation_type: result_type,
                    source: result_source.clone(),
                });
                Some(result_source)
            };

            implementations.push(SolverResolvedTransitionImplementation {
                implementation: candidate.implementation,
                bound_to,
                params: implementation_params,
                return_wrapper: callable.inner.return_wrapper,
                result_source,
            });
        }
        inlay_event!(
            name: "inlay.rule.resolve_callable_transition.implementations",
            type_hash = debug_hash(
                &PyType::<Qual<Keyed<'ty>>, Qual<Keyed<'ty>>, Concrete>::Callable(request_key)
            ),
            implementations = implementations.len() as u64,
        );
        inlay_span_record!(implementations = implementations.len() as u64);

        let target = self.resolve_callable_transition_target(
            target_rules,
            request_result_type,
            child_delta,
            ctx,
        )?;

        Ok(SolverResolvedTransition {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            implementations,
            target,
        })
    }

    fn resolve_callable_transition_target(
        &self,
        target_rules: RuleId,
        request_result_type: PyTypeConcreteKey<'ty>,
        child_delta: RegistryEnvDeltaRequest<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionRef> {
        self.solve_child(
            request_result_type,
            target_rules,
            LazyDepthMode::Increment,
            child_delta,
            ctx,
        )
    }

    #[instrumented(
        name = "inlay.rule.resolve_method_impl",
        target = "inlay",
        level = "trace",
        ret,
        err,
        skip(type_ref, method_name, method_protocol),
        fields(
            type_hash = debug_hash(&type_ref),
            method_name = method_name.as_deref().unwrap_or(""),
            method_protocol_hash = method_protocol.as_ref().map(debug_hash),
            method_lookup_bases,
            target_rule = target_rules.index() as u64,
            matched_methods,
            params,
            implementations
        )
    )]
    fn resolve_method_impl(
        &self,
        target_rules: RuleId,
        override_resolution: MethodOverrideResolution,
        type_ref: PyTypeConcreteKey<'ty>,
        method_name: Option<Arc<str>>,
        method_protocol: Option<PyTypeConcreteKey<'ty>>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionNode<'ty>> {
        let PyType::Callable(request_key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };
        if !self.lookup_bound_implementations(type_ref, ctx).is_empty() {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        }

        let (effective_protocol_qualifier, lookup_bases) =
            match (method_protocol, method_name.as_ref()) {
                (Some(protocol), Some(method_name)) => self
                    .method_lookup_bases(protocol, method_name, override_resolution, ctx)
                    .map_err(RunError::Rule)?,
                _ => (Qualifier::unqualified(), Vec::new()),
            };
        inlay_span_record!(method_lookup_bases = lookup_bases.len() as u64);
        let matched =
            self.lookup_methods(type_ref, &effective_protocol_qualifier, &lookup_bases, ctx);
        inlay_event!(
            name: "inlay.rule.resolve_method_impl.matched",
            type_hash = debug_hash(&type_ref),
            matched_methods = matched.len() as u64,
        );
        inlay_span_record!(matched_methods = matched.len() as u64);
        let candidates = matched
            .into_iter()
            .map(|matched| CallableImplementationCandidate {
                public_callable_key: matched.concrete_public_callable_key,
                implementation_callable_key: matched.concrete_implementation_callable_key,
                implementation: SolverTransitionImplementationCallable::Static(Arc::clone(
                    &matched.implementation.implementation,
                )),
                bound_to: matched.concrete_bound_to,
            })
            .collect();

        self.resolve_callable_transition(target_rules, request_key, candidates, ctx)
            .map(SolverResolutionNode::Transition)
    }

    fn attribute_source_type_and_access(
        attribute: &Attribute<'ty, Concrete>,
    ) -> (PyTypeConcreteKey<'ty>, MemberAccessKind) {
        match attribute.source_type {
            SourceType::Protocol(source_type) => {
                (PyType::Protocol(source_type), MemberAccessKind::Attribute)
            }
            SourceType::TypedDict(source_type) => {
                (PyType::TypedDict(source_type), MemberAccessKind::DictItem)
            }
        }
    }

    fn resolve_attribute_candidates(
        &self,
        candidates: &[&Attribute<'ty, Concrete>],
        inner: RuleId,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, CandidateResolution<'ty>> {
        let mut resolved = None;
        let mut resolved_keys = HashSet::default();
        let mut errors = Vec::new();

        for &attribute in candidates {
            let (source_type, access_kind) = Self::attribute_source_type_and_access(attribute);
            let candidate_key = (
                attribute.source.clone(),
                Arc::clone(&attribute.name),
                access_kind,
            );
            if resolved_keys.contains(&candidate_key) {
                continue;
            }

            match self.solve_eager_child(
                source_type,
                inner,
                RegistryEnvDeltaRequest::identity(),
                ctx,
            ) {
                Ok(source) => {
                    if !resolved_keys.insert(candidate_key) {
                        continue;
                    }
                    if resolved.is_some() {
                        return Err(RunError::Rule(ResolutionError::AmbiguousAttribute(
                            type_ref,
                        )));
                    }
                    resolved = Some(ResolvedRule {
                        resolution: SolverResolutionNode::Attribute {
                            source: source.result_ref,
                            attribute_name: Arc::clone(&attribute.name),
                            access_kind,
                        },
                        dynamic: source.dynamic,
                    });
                }
                Err(RunError::Rule(error)) => errors.push(Arc::new(error)),
                Err(RunError::Solve(error)) => return Err(RunError::Solve(error)),
            }
        }

        Ok(match resolved {
            Some(node) => Ok(node),
            None => Err(errors),
        })
    }

    #[instrumented(
        name = "inlay.rule.resolve_attribute_source",
        target = "inlay",
        level = "trace",
        ret,
        err,
        skip(query),
        fields(
            type_hash = debug_hash(&query.type_ref),
            requested_name = query.requested_name.as_deref().unwrap_or(""),
            matched_attributes
        )
    )]
    fn resolve_attribute_source(
        &self,
        inner: RuleId,
        query: &ResolutionQuery<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, ResolvedRule<SolverResolutionNode<'ty>>> {
        let type_ref = query.type_ref;
        let matched = self.lookup_attributes(type_ref, ctx);
        inlay_event!(
            name: "inlay.rule.resolve_attribute_source.matched",
            type_hash = debug_hash(&type_ref),
            matched_attributes = matched.len() as u64,
            requested_name = query.requested_name.as_deref().unwrap_or(""),
        );
        inlay_span_record!(matched_attributes = matched.len() as u64);

        if matched.is_empty() {
            return Err(RunError::Rule(ResolutionError::NoAttributeFound(type_ref)));
        }

        match &query.requested_name {
            Some(requester_name) => {
                let (named, rest): (Vec<_>, Vec<_>) = matched
                    .iter()
                    .partition(|attribute| attribute.name.as_ref() == requester_name.as_ref());

                let named_errors = match self.resolve_attribute_candidates(
                    named.as_slice(),
                    inner,
                    type_ref,
                    ctx,
                )? {
                    Ok(node) => return Ok(node),
                    Err(errors) => errors,
                };

                let unnamed_errors = match self.resolve_attribute_candidates(
                    rest.as_slice(),
                    inner,
                    type_ref,
                    ctx,
                )? {
                    Ok(node) => return Ok(node),
                    Err(errors) => errors,
                };

                Err(RunError::Rule(ResolutionError::MissingDependency(
                    type_ref,
                    [named_errors, unnamed_errors].concat(),
                )))
            }
            None => {
                let all = matched.iter().collect::<Vec<_>>();
                let errors = match self.resolve_attribute_candidates(
                    all.as_slice(),
                    inner,
                    type_ref,
                    ctx,
                )? {
                    Ok(node) => return Ok(node),
                    Err(errors) => errors,
                };
                Err(RunError::Rule(ResolutionError::MissingDependency(
                    type_ref, errors,
                )))
            }
        }
    }

    #[instrumented(
        name = "inlay.rule.resolve_constructor",
        target = "inlay",
        level = "trace",
        ret,
        err,
        skip(type_ref),
        fields(
            type_hash = debug_hash(&type_ref),
            param_rule = param_rules.index() as u64,
            matched_constructors,
            params
        )
    )]
    fn resolve_constructor(
        &self,
        param_rules: RuleId,
        static_policy: StaticPolicy,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, ResolvedRule<SolverResolutionNode<'ty>>> {
        let matched = self.lookup_constructors(type_ref, ctx);
        inlay_event!(
            name: "inlay.rule.resolve_constructor.matched",
            type_hash = debug_hash(&type_ref),
            matched_constructors = matched.len() as u64,
        );
        inlay_span_record!(matched_constructors = matched.len() as u64);
        let matched = match matched.as_slice() {
            [] => {
                return Err(RunError::Rule(ResolutionError::NoConstructorFound(
                    type_ref,
                )));
            }
            [matched] => matched.clone(),
            [_, _, ..] => {
                return Err(RunError::Rule(ResolutionError::AmbiguousConstructor(
                    type_ref,
                )));
            }
        };

        let callable = ctx
            .shared()
            .types()
            .concrete
            .callables
            .get(matched.concrete_callable_key)
            .clone();
        let param_info: Vec<(Arc<str>, PyTypeConcreteKey<'ty>, ParamKind, bool)> = callable
            .inner
            .params
            .iter()
            .zip(callable.inner.param_kinds.iter())
            .zip(callable.inner.param_has_default.iter())
            .map(|(((name, &param_type), &kind), &has_default)| {
                (Arc::clone(name), param_type, kind, has_default)
            })
            .collect();

        let params = self.solve_policy_params(param_info, param_rules, static_policy, ctx)?;
        inlay_event!(
            name: "inlay.rule.resolve_constructor.params",
            type_hash = debug_hash(&type_ref),
            params = params.resolution.len() as u64,
        );
        inlay_span_record!(params = params.resolution.len() as u64);

        Ok(params.map(|params| SolverResolutionNode::Constructor {
            implementation: matched.constructor,
            params,
        }))
    }

    fn solve_policy_params(
        &self,
        param_info: Vec<(Arc<str>, PyTypeConcreteKey<'ty>, ParamKind, bool)>,
        param_rules: RuleId,
        static_policy: StaticPolicy,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, ResolvedRule<ResolvedParams>> {
        let param_delta = match static_policy {
            StaticPolicy::IfStaticDependencies => RegistryEnvDeltaRequest::identity(),
            StaticPolicy::Always | StaticPolicy::Never => Self::erased_static_delta(),
        };
        let mut params = Vec::with_capacity(param_info.len());
        let mut any_dynamic_param = false;
        for (name, param_type, kind, has_default) in param_info {
            match self.solve_child_named(
                param_type,
                Arc::clone(&name),
                param_rules,
                LazyDepthMode::Keep,
                param_delta.clone(),
                ctx,
            ) {
                Ok(result_ref) => {
                    any_dynamic_param |= Self::result_dynamic(result_ref, ctx);
                    params.push((result_ref, name, kind));
                }
                Err(RunError::Rule(_)) if has_default => {}
                Err(error) => return Err(error),
            }
        }
        Ok(ResolvedRule {
            resolution: params,
            dynamic: policy_dynamic(static_policy, any_dynamic_param),
        })
    }

    #[instrumented(
        name = "inlay.rule.resolve_init",
        target = "inlay",
        level = "trace",
        ret,
        err,
        skip(query),
        fields(
            type_hash = debug_hash(&query.type_ref),
            param_rule = param_rules.index() as u64,
            params
        )
    )]
    fn resolve_init(
        &self,
        param_rules: RuleId,
        whitelist: &BTreeSet<PythonIdentity>,
        blacklist: &BTreeSet<PythonIdentity>,
        static_policy: StaticPolicy,
        query: &ResolutionQuery<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, ResolvedRule<SolverResolutionNode<'ty>>> {
        let type_ref = query.type_ref;
        let PyType::Class(key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };

        if !self.lookup_constants(query, ctx).is_empty()
            || !self.lookup_properties(type_ref, ctx).is_empty()
            || !self.lookup_attributes(type_ref, ctx).is_empty()
            || !self.lookup_constructors(type_ref, ctx).is_empty()
        {
            return Err(RunError::Rule(ResolutionError::NoConstructorFound(
                type_ref,
            )));
        }

        let (implementation, param_info) = {
            let class = ctx.shared().types().concrete.classes.get(key).clone();
            let class_identity = PythonIdentity::from_arc_py_any(&class.inner.constructor);
            if blacklist.contains(&class_identity)
                || (!whitelist.is_empty() && !whitelist.contains(&class_identity))
            {
                return Err(RunError::Rule(ResolutionError::NoConstructorFound(
                    type_ref,
                )));
            }
            let Some(init) = class.inner.init else {
                return Err(RunError::Rule(ResolutionError::NoConstructorFound(
                    type_ref,
                )));
            };
            let param_info = init
                .params
                .into_iter()
                .zip(init.param_kinds)
                .zip(init.param_has_default)
                .map(|(((name, param_type), kind), has_default)| {
                    (name, param_type, kind, has_default)
                })
                .collect::<Vec<_>>();
            (class.inner.constructor, param_info)
        };

        let params = self.solve_policy_params(param_info, param_rules, static_policy, ctx)?;
        inlay_span_record!(params = params.resolution.len() as u64);

        Ok(params.map(|params| SolverResolutionNode::Init {
            implementation: SolverInitImplementation { implementation },
            params,
        }))
    }

    #[instrumented(
        name = "inlay.rule.resolve_match_first",
        target = "inlay",
        level = "trace",
        ret,
        err,
        fields(
            type_hash = debug_hash(&query.type_ref),
            rules = rules.len() as u64,
            causes
        )
    )]
    fn resolve_match_first(
        &self,
        rules: &[RuleId],
        query: &ResolutionQuery<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, ResolvedRule<SolverResolutionNode<'ty>>> {
        let mut causes = Vec::new();
        let mut cause_count = 0;
        let result =
            self.resolve_first_matching_rule(rules, query, ctx, &mut causes, &mut cause_count);
        inlay_span_record!(causes = cause_count as u64);
        result
    }

    #[instrumented(
        name = "inlay.rule.resolve_match_by_type",
        target = "inlay",
        level = "trace",
        ret,
        err,
        fields(
            type_hash = debug_hash(&query.type_ref),
            family,
            rules,
            causes
        )
    )]
    fn resolve_match_by_type(
        &self,
        family_rules: &TypeFamilyRules,
        query: &ResolutionQuery<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
    ) -> RegistryRunResult<'ty, ResolvedRule<SolverResolutionNode<'ty>>> {
        let family = TypeFamily::of(query.type_ref);
        let rules = family.rules(family_rules);
        inlay_span_record!(family = family.label(), rules = rules.len() as u64);

        let mut causes = Vec::new();
        let mut cause_count = 0;
        let result =
            self.resolve_first_matching_rule(rules, query, ctx, &mut causes, &mut cause_count);
        inlay_span_record!(causes = cause_count as u64);
        result
    }

    fn resolve_first_matching_rule(
        &self,
        rules: &[RuleId],
        query: &ResolutionQuery<'ty>,
        ctx: &mut RegistryRuleContext<'_, '_, 'ty>,
        causes: &mut Vec<Arc<ResolutionError<'ty>>>,
        cause_count: &mut usize,
    ) -> RegistryRunResult<'ty, ResolvedRule<SolverResolutionNode<'ty>>> {
        let type_ref = query.type_ref;

        for &rule_id in rules {
            let rule_label = self.rule_label(rule_id);
            match ctx.solve(query.clone(), rule_id, LazyDepthMode::Keep) {
                Ok(SolveResult::Resolved { result, result_ref }) => match result {
                    Ok(node) => {
                        return Ok(ResolvedRule {
                            resolution: SolverResolutionNode::Delegate(result_ref),
                            dynamic: node.dynamic,
                        });
                    }
                    Err(error) => causes.push(Arc::new(ResolutionError::RuleError {
                        rule_label,
                        cause: Arc::new(error.clone()),
                    })),
                },
                Ok(SolveResult::Lazy { result_ref }) => {
                    return Ok(ResolvedRule {
                        resolution: SolverResolutionNode::Delegate(result_ref),
                        dynamic: Self::result_dynamic(result_ref, ctx),
                    });
                }
                Err(SolveError::SameDepthCycle) => {
                    causes.push(Arc::new(ResolutionError::RuleError {
                        rule_label,
                        cause: Arc::new(ResolutionError::Cycle(type_ref)),
                    }));
                }
                Err(error) => return Err(RunError::Solve(error)),
            }
        }

        *cause_count = causes.len();
        Err(RunError::Rule(ResolutionError::MissingDependency(
            type_ref,
            mem::take(causes),
        )))
    }
}

impl<'ty> SolverRule for RegistryResolutionRule<'ty> {
    type Query = ResolutionQuery<'ty>;
    type Output = SolverResolvedNode<'ty>;
    type Err = ResolutionError<'ty>;
    type Env = RegistryEnv<'ty>;
    type ResultsArena = SolverResolutionArena<'ty>;
    type RuleStateId = RuleId;

    #[instrumented(
        name = "inlay.rule.run",
        target = "inlay",
        level = "trace",
        ret,
        err,
        fields(
            rule_id = ctx.state_id().index() as u64,
            rule_label = self.rule_label(ctx.state_id()),
            query_hash = debug_hash(&query),
            type_hash = debug_hash(&query.type_ref),
            requested_name = query.requested_name.as_deref().unwrap_or("")
        )
    )]
    fn run(
        &self,
        query: Self::Query,
        ctx: &mut RuleContext<Self>,
    ) -> Result<Self::Output, RunError<Self>> {
        let rule = self
            .rules
            .get(ctx.state_id())
            .ok_or_else(|| RunError::Rule(ResolutionError::InvalidRuleId(ctx.state_id())))?
            .clone();
        let resolved = self.resolve_rule(rule, &query, ctx)?;

        if resolved.dynamic && Self::static_required(ctx) {
            return Err(RunError::Rule(ResolutionError::DynamicResultRejected(
                query.type_ref,
            )));
        }

        let writable_dependencies = solver_writable_dependencies(&resolved.resolution);
        Ok(SolverResolvedNode {
            target_type: query.type_ref,
            dynamic: resolved.dynamic,
            writable_dependencies,
            resolution: resolved.resolution,
        })
    }
}

fn solver_writable_dependencies<'ty>(
    resolution: &SolverResolutionNode<'ty>,
) -> Vec<SolverWritableDependency<'ty>> {
    match resolution {
        SolverResolutionNode::Property { source, .. }
        | SolverResolutionNode::Attribute { source, .. } => {
            vec![SolverWritableDependency::Result(*source)]
        }
        SolverResolutionNode::RuntimeUnionDispatch { source, .. } => {
            vec![SolverWritableDependency::Source(source.clone())]
        }
        SolverResolutionNode::Constructor { params, .. }
        | SolverResolutionNode::Init { params, .. } => params
            .iter()
            .map(|(result_ref, _, _)| SolverWritableDependency::Result(*result_ref))
            .collect(),
        _ => Vec::new(),
    }
}

fn policy_dynamic(static_policy: StaticPolicy, any_dynamic_param: bool) -> bool {
    match static_policy {
        StaticPolicy::Always => false,
        StaticPolicy::Never => true,
        StaticPolicy::IfStaticDependencies => any_dynamic_param,
    }
}

fn union_subtype_sort_key<'ty>(
    sub: PyTypeConcreteKey<'ty>,
    target_variants: &[PyTypeConcreteKey<'ty>],
    arenas: &TypeArenas<'ty>,
) -> (bool, i32, Vec<usize>) {
    let target_positions = target_variants
        .iter()
        .enumerate()
        .map(|(index, &variant)| (variant, index))
        .collect::<BTreeMap<_, _>>();
    let fallback = target_variants.len();

    if let PyType::Union(key) = sub {
        let sub_variants = &arenas.concrete.unions.get(key).inner.variants;
        let mut positions = sub_variants
            .iter()
            .map(|variant| *target_positions.get(variant).unwrap_or(&fallback))
            .collect::<Vec<_>>();
        positions.sort();
        (
            is_none_type(sub, arenas),
            -(sub_variants.len() as i32),
            positions,
        )
    } else {
        (
            is_none_type(sub, arenas),
            -1,
            vec![*target_positions.get(&sub).unwrap_or(&fallback)],
        )
    }
}

fn is_none_type<'ty>(type_ref: PyTypeConcreteKey<'ty>, arenas: &TypeArenas<'ty>) -> bool {
    if let PyType::Sentinel(key) = type_ref {
        matches!(
            arenas.sentinels.get(key).inner.value,
            SentinelTypeKind::None
        )
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use context_solver::solve::Solver;
    use indexmap::IndexMap;
    use pyo3::{Py, PyAny, Python};

    use super::*;
    use crate::rules::env::RegistrySharedState;
    use crate::types::{
        CallableType, ClassInit, ClassType, Parametric, PlainType, ProtocolKey, ProtocolType,
        PyTypeDescriptor, PyTypeId, Qualified, UnionType, WrapperKind,
    };

    fn descriptor(name: &str) -> PyTypeDescriptor {
        PyTypeDescriptor {
            id: PyTypeId::new(name.to_string()),
            display_name: Arc::from(name),
            origin: None,
        }
    }

    fn insert_plain<'ty>(types: &mut TypeArenas<'ty>, name: &str) -> PyTypeConcreteKey<'ty> {
        let key = types.concrete.plains.insert(Qualified {
            inner: PlainType::<Qual<Keyed>, Concrete> {
                descriptor: descriptor(name),
                args: Vec::new(),
            },
            qualifier: Qualifier::unqualified(),
        });
        PyType::Plain(key)
    }

    fn insert_protocol<'ty>(
        types: &mut TypeArenas<'ty>,
        name: &str,
        attributes: Vec<(Arc<str>, PyTypeConcreteKey<'ty>)>,
    ) -> ProtocolKey<'ty, Concrete> {
        types.concrete.protocols.insert(Qualified {
            inner: ProtocolType {
                descriptor: descriptor(name),
                protocol_mro: Vec::new(),
                direct_methods: Vec::new(),
                methods: Vec::new().into(),
                attributes: attributes.into(),
                properties: Vec::new().into(),
                type_params: Vec::new(),
            },
            qualifier: Qualifier::unqualified(),
        })
    }

    fn insert_union<'ty>(
        types: &mut TypeArenas<'ty>,
        variants: Vec<PyTypeConcreteKey<'ty>>,
    ) -> PyTypeConcreteKey<'ty> {
        PyType::Union(types.concrete.unions.insert(Qualified {
            inner: UnionType { variants },
            qualifier: Qualifier::unqualified(),
        }))
    }

    fn python_value() -> Arc<Py<PyAny>> {
        Python::initialize();
        Python::attach(|py| Arc::new(py.None()))
    }

    fn insert_init_fixture<'ty>(
        types: &mut TypeArenas<'ty>,
    ) -> (PyTypeConcreteKey<'ty>, PyTypeConcreteKey<'ty>) {
        let param = PyType::Protocol(insert_protocol(types, "Param", Vec::new()));
        let mut params = IndexMap::new();
        params.insert(Arc::from("value"), param);
        let class = types.concrete.classes.insert(Qualified {
            inner: ClassType {
                descriptor: descriptor("Target"),
                constructor: python_value(),
                args: Vec::new(),
                init: Some(ClassInit {
                    params,
                    param_kinds: vec![ParamKind::PositionalOrKeyword],
                    param_has_default: vec![false],
                }),
            },
            qualifier: Qualifier::unqualified(),
        });
        (PyType::Class(class), param)
    }

    fn insert_constructor_fixture<'ty>(
        types: &mut TypeArenas<'ty>,
    ) -> (
        PyTypeConcreteKey<'ty>,
        PyTypeConcreteKey<'ty>,
        Constructor<'ty>,
    ) {
        let target_descriptor = descriptor("Target");
        let param_descriptor = descriptor("Param");
        let target = PyType::Plain(types.concrete.plains.insert(Qualified {
            inner: PlainType::<Qual<Keyed>, Concrete> {
                descriptor: target_descriptor.clone(),
                args: Vec::new(),
            },
            qualifier: Qualifier::unqualified(),
        }));
        let param = PyType::Protocol(types.concrete.protocols.insert(Qualified {
            inner: ProtocolType::<Qual<Keyed>, Concrete> {
                descriptor: param_descriptor.clone(),
                protocol_mro: Vec::new(),
                direct_methods: Vec::new(),
                methods: Vec::new().into(),
                attributes: Vec::new().into(),
                properties: Vec::new().into(),
                type_params: Vec::new(),
            },
            qualifier: Qualifier::unqualified(),
        }));
        let parametric_target = PyType::Plain(types.parametric.plains.insert(Qualified {
            inner: PlainType::<Qual<Keyed>, Parametric> {
                descriptor: target_descriptor,
                args: Vec::new(),
            },
            qualifier: Qualifier::unqualified(),
        }));
        let parametric_param = PyType::Protocol(types.parametric.protocols.insert(Qualified {
            inner: ProtocolType::<Qual<Keyed>, Parametric> {
                descriptor: param_descriptor,
                protocol_mro: Vec::new(),
                direct_methods: Vec::new(),
                methods: Vec::new().into(),
                attributes: Vec::new().into(),
                properties: Vec::new().into(),
                type_params: Vec::new(),
            },
            qualifier: Qualifier::unqualified(),
        }));
        let mut params = IndexMap::new();
        params.insert(Arc::from("value"), parametric_param);
        let fn_type = types.parametric.callables.insert(Qualified {
            inner: CallableType::<Qual<Keyed>, Parametric> {
                params,
                param_kinds: vec![ParamKind::PositionalOrKeyword],
                param_has_default: vec![false],
                accepts_varargs: false,
                accepts_varkw: false,
                return_type: parametric_target,
                return_wrapper: WrapperKind::None,
                type_params: Vec::new(),
                function_name: None,
            },
            qualifier: Qualifier::unqualified(),
        });
        (
            target,
            param,
            Constructor {
                fn_type,
                implementation: python_value(),
            },
        )
    }

    fn solver<'ty>(
        rules: Vec<RuleMode>,
        types: TypeArenas<'ty>,
    ) -> Solver<RegistryResolutionRule<'ty>> {
        solver_with_constructors(rules, types, Vec::new())
    }

    fn solver_with_constructors<'ty>(
        rules: Vec<RuleMode>,
        types: TypeArenas<'ty>,
        constructors: Vec<Constructor<'ty>>,
    ) -> Solver<RegistryResolutionRule<'ty>> {
        Solver::new(
            RegistryResolutionRule::new(Arc::new(RuleArena::from(rules))),
            RegistrySharedState::new(&constructors, &[], types),
            64,
            64,
        )
    }

    fn solve<'ty>(
        solver: &mut Solver<RegistryResolutionRule<'ty>>,
        type_ref: PyTypeConcreteKey<'ty>,
        env: RegistryEnv<'ty>,
    ) -> SolverResolutionRef {
        solver
            .solve_with_env(
                ResolutionQuery::unnamed(type_ref),
                RuleId::new(0),
                Arc::new(env),
            )
            .expect("solve must not hit solver limits")
    }

    fn resolved<'a, 'ty>(
        solver: &'a Solver<RegistryResolutionRule<'ty>>,
        result_ref: SolverResolutionRef,
    ) -> &'a SolverResolutionResult<'ty> {
        solver.result(result_ref).expect("result must be stored")
    }

    #[test]
    fn static_constant_answer_is_reused_under_static_env() {
        let mut types = TypeArenas::default();
        let target = insert_plain(&mut types, "X");
        let source = Source::transition(None, target);
        let env = RegistryEnv::default().with_transition_sources(vec![source], &types);
        let tagged = env.tagged(RegistryEnvTag::Static);
        let mut solver = solver(vec![RuleMode::Constant], types);

        let untagged_ref = solve(&mut solver, target, env);
        let tagged_ref = solve(&mut solver, target, tagged);

        let node = resolved(&solver, untagged_ref)
            .as_ref()
            .expect("constant must resolve");
        assert!(!node.dynamic);
        assert_eq!(untagged_ref, tagged_ref);
    }

    #[test]
    fn dynamic_protocol_is_rejected_under_static_env() {
        let mut types = TypeArenas::default();
        let protocol = PyType::Protocol(insert_protocol(&mut types, "P", Vec::new()));
        let env = RegistryEnv::default();
        let tagged = env.tagged(RegistryEnvTag::Static);
        let rules = vec![RuleMode::Protocol {
            property_rule: RuleId::new(0),
            attribute_rule: RuleId::new(0),
            method_rule: RuleId::new(0),
        }];
        let mut solver = solver(rules, types);

        let untagged_ref = solve(&mut solver, protocol, env);
        let tagged_ref = solve(&mut solver, protocol, tagged);

        let node = resolved(&solver, untagged_ref)
            .as_ref()
            .expect("protocol must resolve in untagged env");
        assert!(node.dynamic);
        assert_ne!(untagged_ref, tagged_ref);
        assert!(matches!(
            resolved(&solver, tagged_ref),
            Err(ResolutionError::DynamicResultRejected(_))
        ));
    }

    #[test]
    fn static_rejection_is_not_reused_in_untagged_env() {
        let mut types = TypeArenas::default();
        let protocol = PyType::Protocol(insert_protocol(&mut types, "P", Vec::new()));
        let env = RegistryEnv::default();
        let tagged = env.tagged(RegistryEnvTag::Static);
        let rules = vec![RuleMode::Protocol {
            property_rule: RuleId::new(0),
            attribute_rule: RuleId::new(0),
            method_rule: RuleId::new(0),
        }];
        let mut solver = solver(rules, types);

        let tagged_ref = solve(&mut solver, protocol, tagged);
        let untagged_ref = solve(&mut solver, protocol, env);

        assert!(matches!(
            resolved(&solver, tagged_ref),
            Err(ResolutionError::DynamicResultRejected(_))
        ));
        assert_ne!(untagged_ref, tagged_ref);
        let node = resolved(&solver, untagged_ref)
            .as_ref()
            .expect("protocol must resolve in untagged env");
        assert!(node.dynamic);
    }

    #[test]
    fn attribute_inherits_host_dynamicness() {
        let mut types = TypeArenas::default();
        let member = insert_plain(&mut types, "X");
        let name: Arc<str> = Arc::from("x");
        let protocol = insert_protocol(&mut types, "P", vec![(Arc::clone(&name), member)]);
        let host_source = Source::transition(None, PyType::Protocol(protocol));
        let member_source = Source::transition(None, member);
        let env = RegistryEnv::default()
            .with_transition_sources(vec![host_source, member_source], &types);
        let tagged = env.tagged(RegistryEnvTag::Static);
        let rules = vec![
            RuleMode::AttributeSource {
                inner: RuleId::new(1),
            },
            RuleMode::MatchByType {
                rules: Box::new(TypeFamilyRules {
                    protocol: vec![RuleId::new(2)],
                    fallback: vec![RuleId::new(3)],
                    ..TypeFamilyRules::default()
                }),
            },
            RuleMode::Protocol {
                property_rule: RuleId::new(3),
                attribute_rule: RuleId::new(3),
                method_rule: RuleId::new(3),
            },
            RuleMode::Constant,
        ];
        let mut solver = solver(rules, types);

        let untagged_ref = solve(&mut solver, member, env);
        let tagged_ref = solve(&mut solver, member, tagged);

        let node = resolved(&solver, untagged_ref)
            .as_ref()
            .expect("attribute must resolve in untagged env");
        assert!(node.dynamic);
        assert!(matches!(
            node.resolution,
            SolverResolutionNode::Attribute { .. }
        ));
        assert!(resolved(&solver, tagged_ref).is_err());
    }

    #[test]
    fn union_variant_inherits_dynamicness() {
        let mut types = TypeArenas::default();
        let protocol = PyType::Protocol(insert_protocol(&mut types, "P", Vec::new()));
        let union = insert_union(&mut types, vec![protocol]);
        let env = RegistryEnv::default();
        let tagged = env.tagged(RegistryEnvTag::Static);
        let rules = vec![
            RuleMode::MatchByType {
                rules: Box::new(TypeFamilyRules {
                    union: vec![RuleId::new(1)],
                    protocol: vec![RuleId::new(2)],
                    ..TypeFamilyRules::default()
                }),
            },
            RuleMode::Union {
                variant_rules: RuleId::new(0),
            },
            RuleMode::Protocol {
                property_rule: RuleId::new(0),
                attribute_rule: RuleId::new(0),
                method_rule: RuleId::new(0),
            },
        ];
        let mut solver = solver(rules, types);

        let untagged_ref = solve(&mut solver, union, env);
        let tagged_ref = solve(&mut solver, union, tagged);

        let node = resolved(&solver, untagged_ref)
            .as_ref()
            .expect("union must resolve in untagged env");
        assert!(node.dynamic);
        assert!(resolved(&solver, tagged_ref).is_err());
    }

    #[test]
    fn protocol_member_edges_erase_static_requirement() {
        let mut types = TypeArenas::default();
        let inner_protocol = PyType::Protocol(insert_protocol(&mut types, "Inner", Vec::new()));
        let name: Arc<str> = Arc::from("x");
        let outer_protocol = PyType::Protocol(insert_protocol(
            &mut types,
            "Outer",
            vec![(Arc::clone(&name), inner_protocol)],
        ));
        let env = RegistryEnv::default();
        let rules = vec![RuleMode::MatchByType {
            rules: Box::new(TypeFamilyRules {
                protocol: vec![RuleId::new(1)],
                ..TypeFamilyRules::default()
            }),
        }]
        .into_iter()
        .chain([RuleMode::Protocol {
            property_rule: RuleId::new(0),
            attribute_rule: RuleId::new(0),
            method_rule: RuleId::new(0),
        }])
        .collect();
        let tagged = env.tagged(RegistryEnvTag::Static);
        let mut solver = solver(rules, types);

        let untagged_ref = solve(&mut solver, outer_protocol, env);
        let tagged_ref = solve(&mut solver, outer_protocol, tagged);

        let node = resolved(&solver, untagged_ref)
            .as_ref()
            .expect("nested protocol must resolve in untagged env");
        assert!(node.dynamic);
        let SolverResolutionNode::Delegate(inner_ref) = node.resolution else {
            panic!("match_by_type root must delegate");
        };
        let SolverResolutionNode::Protocol { ref members } = resolved(&solver, inner_ref)
            .as_ref()
            .expect("delegated protocol must resolve")
            .resolution
        else {
            panic!("expected protocol aggregate");
        };
        let member_node = resolved(&solver, members[&name])
            .as_ref()
            .expect("dynamic member must resolve behind erased static edge");
        assert!(member_node.dynamic);

        assert!(matches!(
            resolved(&solver, tagged_ref),
            Err(ResolutionError::MissingDependency(_, causes))
                if causes.iter().all(|cause| matches!(
                    cause.as_ref(),
                    ResolutionError::RuleError { cause, .. }
                        if matches!(cause.as_ref(), ResolutionError::DynamicResultRejected(_))
                ))
        ));
    }

    fn insert_recursive_protocol<'ty>(
        types: &mut TypeArenas<'ty>,
        name: &str,
        member: Arc<str>,
    ) -> PyTypeConcreteKey<'ty> {
        let protocol_key = types.concrete.protocols.future_key(0);
        let lazy_key = types.concrete.read_cells.future_key(0);
        types.concrete.protocols.insert(Qualified {
            inner: ProtocolType {
                descriptor: descriptor(name),
                protocol_mro: Vec::new(),
                direct_methods: Vec::new(),
                methods: Vec::new().into(),
                attributes: vec![(member, PyType::ReadCell(lazy_key))].into(),
                properties: Vec::new().into(),
                type_params: Vec::new(),
            },
            qualifier: Qualifier::unqualified(),
        });
        types.concrete.read_cells.insert(Qualified {
            inner: crate::types::ReadCellType {
                target: PyType::Protocol(protocol_key),
            },
            qualifier: Qualifier::unqualified(),
        });
        PyType::Protocol(protocol_key)
    }

    fn recursive_protocol_rules() -> Vec<RuleMode> {
        vec![
            RuleMode::MatchByType {
                rules: Box::new(TypeFamilyRules {
                    protocol: vec![RuleId::new(1)],
                    read_cell: vec![RuleId::new(2)],
                    ..TypeFamilyRules::default()
                }),
            },
            RuleMode::Protocol {
                property_rule: RuleId::new(0),
                attribute_rule: RuleId::new(0),
                method_rule: RuleId::new(0),
            },
            RuleMode::ReadCell {
                inner: RuleId::new(3),
            },
            RuleMode::MatchFirst {
                rules: vec![RuleId::new(0)],
            },
        ]
    }

    #[test]
    fn lazy_cycle_dynamicness_converges_from_provisional_static() {
        let mut types = TypeArenas::default();
        let name: Arc<str> = Arc::from("x");
        let protocol = insert_recursive_protocol(&mut types, "P", Arc::clone(&name));
        let mut solver = solver(recursive_protocol_rules(), types);

        let root_ref = solve(&mut solver, protocol, RegistryEnv::default());

        let root = resolved(&solver, root_ref)
            .as_ref()
            .expect("recursive protocol must resolve");
        assert!(root.dynamic);
        let SolverResolutionNode::Delegate(protocol_ref) = root.resolution else {
            panic!("match_by_type root must delegate");
        };
        let SolverResolutionNode::Protocol { ref members } = resolved(&solver, protocol_ref)
            .as_ref()
            .expect("protocol aggregate must resolve")
            .resolution
        else {
            panic!("expected protocol aggregate");
        };
        let SolverResolutionNode::Delegate(lazy_outer) = resolved(&solver, members[&name])
            .as_ref()
            .expect("member must resolve")
            .resolution
        else {
            panic!("member match_by_type must delegate");
        };
        let lazy_node = resolved(&solver, lazy_outer)
            .as_ref()
            .expect("lazy ref must resolve");
        assert!(!lazy_node.dynamic);
        let SolverResolutionNode::ReadCell { target } = lazy_node.resolution else {
            panic!("expected lazy ref handle");
        };
        let cyclic = resolved(&solver, target)
            .as_ref()
            .expect("cyclic delegate must resolve");
        assert!(matches!(
            cyclic.resolution,
            SolverResolutionNode::Delegate(inner) if inner == root_ref
        ));
        assert!(cyclic.dynamic);
    }

    #[test]
    fn lazy_cycle_is_rejected_under_static_env_without_diverging() {
        let mut types = TypeArenas::default();
        let name: Arc<str> = Arc::from("x");
        let protocol = insert_recursive_protocol(&mut types, "P", Arc::clone(&name));
        let tagged = RegistryEnv::default().tagged(RegistryEnvTag::Static);
        let mut solver = solver(recursive_protocol_rules(), types);

        let tagged_ref = solve(&mut solver, protocol, tagged);
        let untagged_ref = solve(&mut solver, protocol, RegistryEnv::default());

        assert!(matches!(
            resolved(&solver, tagged_ref),
            Err(ResolutionError::MissingDependency(_, _))
        ));
        let untagged = resolved(&solver, untagged_ref)
            .as_ref()
            .expect("recursive protocol must resolve in untagged env");
        assert!(untagged.dynamic);
        assert_ne!(untagged_ref, tagged_ref);
    }

    #[test]
    fn match_first_falls_back_to_static_candidate_under_static_env() {
        let mut types = TypeArenas::default();
        let protocol = PyType::Protocol(insert_protocol(&mut types, "P", Vec::new()));
        let source = Source::transition(None, protocol);
        let env = RegistryEnv::default().with_transition_sources(vec![source], &types);
        let tagged = env.tagged(RegistryEnvTag::Static);
        let rules = vec![
            RuleMode::MatchFirst {
                rules: vec![RuleId::new(1), RuleId::new(2)],
            },
            RuleMode::Protocol {
                property_rule: RuleId::new(0),
                attribute_rule: RuleId::new(0),
                method_rule: RuleId::new(0),
            },
            RuleMode::Constant,
        ];
        let mut solver = solver(rules, types);

        let untagged_ref = solve(&mut solver, protocol, env);
        let tagged_ref = solve(&mut solver, protocol, tagged);

        let untagged = resolved(&solver, untagged_ref)
            .as_ref()
            .expect("dynamic candidate must win untagged");
        assert!(untagged.dynamic);
        let SolverResolutionNode::Delegate(inner) = untagged.resolution else {
            panic!("match_first must delegate");
        };
        assert!(matches!(
            resolved(&solver, inner)
                .as_ref()
                .expect("protocol candidate must resolve")
                .resolution,
            SolverResolutionNode::Protocol { .. }
        ));

        let tagged_node = resolved(&solver, tagged_ref)
            .as_ref()
            .expect("static fallback candidate must win under Static");
        assert!(!tagged_node.dynamic);
        let SolverResolutionNode::Delegate(inner) = tagged_node.resolution else {
            panic!("match_first must delegate");
        };
        assert!(matches!(
            resolved(&solver, inner)
                .as_ref()
                .expect("constant candidate must resolve")
                .resolution,
            SolverResolutionNode::Constant { .. }
        ));
    }

    #[test]
    fn static_answer_solved_under_static_env_is_reused_untagged() {
        let mut types = TypeArenas::default();
        let target = insert_plain(&mut types, "X");
        let source = Source::transition(None, target);
        let env = RegistryEnv::default().with_transition_sources(vec![source], &types);
        let tagged = env.tagged(RegistryEnvTag::Static);
        let mut solver = solver(vec![RuleMode::Constant], types);

        let tagged_ref = solve(&mut solver, target, tagged);
        let untagged_ref = solve(&mut solver, target, env);

        assert_eq!(tagged_ref, untagged_ref);
        assert!(
            !resolved(&solver, tagged_ref)
                .as_ref()
                .expect("constant must resolve")
                .dynamic
        );
    }

    #[test]
    fn constructor_static_policy_controls_dynamic_params() {
        for policy in [
            StaticPolicy::Always,
            StaticPolicy::Never,
            StaticPolicy::IfStaticDependencies,
        ] {
            let mut types = TypeArenas::default();
            let (target, _param, constructor) = insert_constructor_fixture(&mut types);
            let rules = vec![
                RuleMode::Constructor {
                    param_rules: RuleId::new(1),
                    static_policy: policy,
                },
                RuleMode::Protocol {
                    property_rule: RuleId::new(1),
                    attribute_rule: RuleId::new(1),
                    method_rule: RuleId::new(1),
                },
            ];
            let mut solver = solver_with_constructors(rules, types, vec![constructor]);
            let tagged = RegistryEnv::default().tagged(RegistryEnvTag::Static);

            let tagged_ref = solve(&mut solver, target, tagged);
            match policy {
                StaticPolicy::Always => {
                    let node = resolved(&solver, tagged_ref)
                        .as_ref()
                        .expect("always constructor must resolve under Static");
                    assert!(!node.dynamic);
                    let SolverResolutionNode::Constructor { ref params, .. } = node.resolution
                    else {
                        panic!("expected constructor");
                    };
                    assert!(
                        resolved(&solver, params[0].0)
                            .as_ref()
                            .expect("dynamic parameter must resolve behind erased edge")
                            .dynamic
                    );
                }
                StaticPolicy::Never => assert!(matches!(
                    resolved(&solver, tagged_ref),
                    Err(ResolutionError::DynamicResultRejected(rejected)) if *rejected == target
                )),
                StaticPolicy::IfStaticDependencies => match resolved(&solver, tagged_ref) {
                    Err(ResolutionError::DynamicResultRejected(rejected)) => {
                        assert!(*rejected != target);
                    }
                    Err(error) => panic!("unexpected constructor error: {error}"),
                    Ok(_) => panic!("if_static_dependencies constructor must be rejected"),
                },
            }

            let untagged_ref = solve(&mut solver, target, RegistryEnv::default());
            let node = resolved(&solver, untagged_ref)
                .as_ref()
                .expect("constructor must resolve untagged");
            assert_eq!(node.dynamic, policy != StaticPolicy::Always);
        }
    }

    #[test]
    fn init_static_policy_controls_dynamic_params() {
        for policy in [
            StaticPolicy::Always,
            StaticPolicy::Never,
            StaticPolicy::IfStaticDependencies,
        ] {
            let mut types = TypeArenas::default();
            let (target, param) = insert_init_fixture(&mut types);
            let rules = vec![
                RuleMode::Init {
                    param_rules: RuleId::new(1),
                    whitelist: BTreeSet::new(),
                    blacklist: BTreeSet::new(),
                    static_policy: policy,
                },
                RuleMode::Protocol {
                    property_rule: RuleId::new(1),
                    attribute_rule: RuleId::new(1),
                    method_rule: RuleId::new(1),
                },
            ];
            let mut solver = solver(rules, types);
            let tagged = RegistryEnv::default().tagged(RegistryEnvTag::Static);

            let tagged_ref = solve(&mut solver, target, tagged);
            match policy {
                StaticPolicy::Always => {
                    let node = resolved(&solver, tagged_ref)
                        .as_ref()
                        .expect("always init must resolve under Static");
                    assert!(!node.dynamic);
                    let SolverResolutionNode::Init { ref params, .. } = node.resolution else {
                        panic!("expected init");
                    };
                    assert!(
                        resolved(&solver, params[0].0)
                            .as_ref()
                            .expect("dynamic parameter must resolve behind erased edge")
                            .dynamic
                    );
                }
                StaticPolicy::Never => assert!(matches!(
                    resolved(&solver, tagged_ref),
                    Err(ResolutionError::DynamicResultRejected(rejected)) if *rejected == target
                )),
                StaticPolicy::IfStaticDependencies => assert!(matches!(
                    resolved(&solver, tagged_ref),
                    Err(ResolutionError::DynamicResultRejected(rejected)) if *rejected == param
                )),
            }

            let untagged_ref = solve(&mut solver, target, RegistryEnv::default());
            let node = resolved(&solver, untagged_ref)
                .as_ref()
                .expect("init must resolve untagged");
            assert_eq!(node.dynamic, policy != StaticPolicy::Always);
        }
    }

    #[test]
    fn policy_dynamic_maps_policies() {
        assert!(!policy_dynamic(StaticPolicy::Always, true));
        assert!(!policy_dynamic(StaticPolicy::Always, false));
        assert!(policy_dynamic(StaticPolicy::Never, false));
        assert!(policy_dynamic(StaticPolicy::Never, true));
        assert!(!policy_dynamic(StaticPolicy::IfStaticDependencies, false));
        assert!(policy_dynamic(StaticPolicy::IfStaticDependencies, true));
    }
}
