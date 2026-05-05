use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::mem;
use std::sync::Arc;

use context_solver::{
    Arena as ResultsArena, LazyDepthMode, ReplaceError, Rule as SolverRule, RuleContext, RunError,
    solve::{SolveError, SolveResult},
};
use inlay_instrument::{inlay_span_record, instrumented};
use rustc_hash::{FxHashSet as HashSet, FxHasher};

use crate::{
    registry::{Constructor, MethodImplementation, Source, SourceType},
    types::{
        MemberAccessKind, ParamKind, PyType, PyTypeConcreteKey, SentinelTypeKind, TypeArenas,
        WrapperKind, requalify_concrete,
    },
};

use super::{
    MethodParam, ResolutionError, RuleArena, RuleId, RuleMode, TypeFamilyRules,
    env::{
        Attribute, ConstructorLookup, MethodLookup, Property, RegistryEnv, ResolutionLookup,
        ResolutionLookupResult,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct SolverResolutionRef(u32);

impl SolverResolutionRef {
    fn index(self) -> usize {
        self.0 as usize
    }
}

type SolverResolutionResult<'ty> = Result<SolverResolvedNode<'ty>, ResolutionError<'ty>>;
type RegistryRuleContext<'a, 'ty> = RuleContext<'a, RegistryResolutionRule<'ty>>;
type RegistryRunError<'ty> = RunError<RegistryResolutionRule<'ty>>;
type RegistryRunResult<'ty, T> = Result<T, RegistryRunError<'ty>>;
type MemberResolutionMap = BTreeMap<Arc<str>, SolverResolutionRef>;
type MemberResolutionErrors<'ty> = Vec<Arc<ResolutionError<'ty>>>;
type MemberResolutionResult<'ty> = Result<MemberResolutionMap, MemberResolutionErrors<'ty>>;

#[derive(Clone, Copy)]
enum TypeFamily {
    Sentinel,
    ParamSpec,
    Plain,
    Protocol,
    TypedDict,
    Union,
    Callable,
    LazyRef,
    TypeVar,
}

impl TypeFamily {
    fn of(type_ref: PyTypeConcreteKey<'_>) -> Self {
        match type_ref {
            PyType::Sentinel(_) => Self::Sentinel,
            PyType::ParamSpec(_) => Self::ParamSpec,
            PyType::Plain(_) => Self::Plain,
            PyType::Protocol(_) => Self::Protocol,
            PyType::TypedDict(_) => Self::TypedDict,
            PyType::Union(_) => Self::Union,
            PyType::Callable(_) => Self::Callable,
            PyType::LazyRef(_) => Self::LazyRef,
            PyType::TypeVar(_) => Self::TypeVar,
        }
    }

    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    fn label(self) -> &'static str {
        match self {
            Self::Sentinel => "sentinel",
            Self::ParamSpec => "param_spec",
            Self::Plain => "plain",
            Self::Protocol => "protocol",
            Self::TypedDict => "typed_dict",
            Self::Union => "union",
            Self::Callable => "callable",
            Self::LazyRef => "lazy_ref",
            Self::TypeVar => "type_var",
        }
    }

    fn rules(self, rules: &TypeFamilyRules) -> &[RuleId] {
        let selected = match self {
            Self::Sentinel => rules.sentinel.as_slice(),
            Self::ParamSpec => rules.param_spec.as_slice(),
            Self::Plain => rules.plain.as_slice(),
            Self::Protocol => rules.protocol.as_slice(),
            Self::TypedDict => rules.typed_dict.as_slice(),
            Self::Union => rules.union.as_slice(),
            Self::Callable => rules.callable.as_slice(),
            Self::LazyRef => rules.lazy_ref.as_slice(),
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
}

impl<'ty> ResolutionQuery<'ty> {
    pub(crate) fn unnamed(type_ref: PyTypeConcreteKey<'ty>) -> Self {
        Self {
            type_ref,
            requested_name: None,
        }
    }

    pub(crate) fn named(type_ref: PyTypeConcreteKey<'ty>, requested_name: Arc<str>) -> Self {
        Self {
            type_ref,
            requested_name: Some(requested_name),
        }
    }
}

impl std::fmt::Debug for ResolutionQuery<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolutionQuery")
            .field("type_hash", &debug_hash(&self.type_ref))
            .field("requested_name", &self.requested_name)
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

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct SolverResolvedMethodImplementation<'ty> {
    pub(crate) implementation: Arc<MethodImplementation<'ty>>,
    pub(crate) bound_to: Option<SolverResolutionRef>,
    pub(crate) params: Vec<(SolverResolutionRef, Arc<str>, ParamKind)>,
    pub(crate) return_wrapper: WrapperKind,
    pub(crate) result_source: Option<Source<'ty>>,
}

impl std::fmt::Debug for SolverResolvedMethodImplementation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolverResolvedMethodImplementation")
            .field("bound_to", &self.bound_to)
            .field("params", &self.params.len())
            .field("return_wrapper", &self.return_wrapper)
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
    LazyRef {
        target: SolverResolutionRef,
    },
    None,
    UnionVariant {
        target: SolverResolutionRef,
    },
    Protocol {
        members: BTreeMap<Arc<str>, SolverResolutionRef>,
    },
    TypedDict {
        members: BTreeMap<Arc<str>, SolverResolutionRef>,
    },
    Method {
        return_wrapper: WrapperKind,
        accepts_varargs: bool,
        accepts_varkw: bool,
        params: Vec<MethodParam<'ty>>,
        implementations: Vec<SolverResolvedMethodImplementation<'ty>>,
        target: SolverResolutionRef,
    },
    AutoMethod {
        return_wrapper: WrapperKind,
        accepts_varargs: bool,
        accepts_varkw: bool,
        params: Vec<MethodParam<'ty>>,
        target: SolverResolutionRef,
    },
    Attribute {
        source: SolverResolutionRef,
        attribute_name: Arc<str>,
        access_kind: MemberAccessKind,
    },
    Constructor {
        implementation: Arc<Constructor<'ty>>,
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
            Self::LazyRef { target } => f.debug_struct("LazyRef").field("target", target).finish(),
            Self::None => f.debug_struct("None").finish(),
            Self::UnionVariant { target } => f
                .debug_struct("UnionVariant")
                .field("target", target)
                .finish(),
            Self::Protocol { members } => f
                .debug_struct("Protocol")
                .field("members", &members.len())
                .finish(),
            Self::TypedDict { members } => f
                .debug_struct("TypedDict")
                .field("members", &members.len())
                .finish(),
            Self::Method {
                params,
                implementations,
                target,
                ..
            } => f
                .debug_struct("Method")
                .field("params", &params.len())
                .field("implementations", &implementations.len())
                .field("target", target)
                .finish(),
            Self::AutoMethod { params, target, .. } => f
                .debug_struct("AutoMethod")
                .field("params", &params.len())
                .field("target", target)
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
            Self::Delegate(result_ref) => f.debug_tuple("Delegate").field(result_ref).finish(),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct SolverResolvedNode<'ty> {
    pub(crate) target_type: PyTypeConcreteKey<'ty>,
    pub(crate) resolution: SolverResolutionNode<'ty>,
}

impl std::fmt::Debug for SolverResolvedNode<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolverResolvedNode")
            .field("target_hash", &debug_hash(&self.target_type))
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

    fn current_env(&self, ctx: &RegistryRuleContext<'_, 'ty>) -> Arc<RegistryEnv<'ty>> {
        ctx.env_arc()
    }

    fn transition_env(
        &self,
        ctx: &RegistryRuleContext<'_, 'ty>,
        params: Vec<(Arc<str>, PyTypeConcreteKey<'ty>)>,
        result_types: Vec<(PyTypeConcreteKey<'ty>, usize)>,
    ) -> Arc<RegistryEnv<'ty>> {
        Arc::new(ctx.env().with_transition(params, result_types))
    }

    fn is_none_type(
        &self,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> bool {
        let PyType::Sentinel(key) = type_ref else {
            return false;
        };
        matches!(
            ctx.shared().types().sentinels.get(key).inner.value,
            SentinelTypeKind::None
        )
    }

    fn solve_child_query(
        &self,
        query: ResolutionQuery<'ty>,
        state_id: RuleId,
        lazy_depth_mode: LazyDepthMode,
        env: Arc<RegistryEnv<'ty>>,
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionRef> {
        let type_ref = query.type_ref;
        match ctx.solve(query, state_id, lazy_depth_mode, env) {
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
        env: Arc<RegistryEnv<'ty>>,
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionRef> {
        self.solve_child_query(
            ResolutionQuery::unnamed(query),
            state_id,
            lazy_depth_mode,
            env,
            ctx,
        )
    }

    fn solve_child_named(
        &self,
        query: PyTypeConcreteKey<'ty>,
        requested_name: Arc<str>,
        state_id: RuleId,
        lazy_depth_mode: LazyDepthMode,
        env: Arc<RegistryEnv<'ty>>,
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionRef> {
        self.solve_child_query(
            ResolutionQuery::named(query, requested_name),
            state_id,
            lazy_depth_mode,
            env,
            ctx,
        )
    }

    fn lookup_constants(
        &self,
        query: &ResolutionQuery<'ty>,
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> Vec<(PyTypeConcreteKey<'ty>, Source<'ty>)> {
        let ResolutionLookupResult::Constants(entries) = ctx.lookup(&ResolutionLookup::Constant {
            type_ref: query.type_ref,
            requested_name: query.requested_name.clone(),
        }) else {
            unreachable!();
        };
        entries.into_iter().collect()
    }

    fn lookup_constructors(
        &self,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, 'ty>,
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
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> Vec<MethodLookup<'ty>> {
        let entries: BTreeSet<_> = ctx.shared().lookup_methods(type_ref).into_iter().collect();
        entries.into_iter().collect()
    }

    fn lookup_properties(
        &self,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, 'ty>,
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
        ctx: &mut RegistryRuleContext<'_, 'ty>,
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
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionNode<'ty>> {
        let type_ref = query.type_ref;
        match rule {
            RuleMode::Constant => self.resolve_constant(query, ctx).map_err(RunError::Rule),
            RuleMode::Property { inner } => self.resolve_property(inner, type_ref, ctx),
            RuleMode::LazyRef { inner } => self.resolve_lazy_ref(inner, type_ref, ctx),
            RuleMode::Union {
                variant_rules,
                allow_none_fallback,
            } => self.resolve_union(variant_rules, allow_none_fallback, type_ref, ctx),
            RuleMode::Protocol {
                property_rule,
                attribute_rule,
                method_rule,
            } => self.resolve_protocol(property_rule, attribute_rule, method_rule, type_ref, ctx),
            RuleMode::TypedDict { attribute_rule } => {
                self.resolve_typed_dict(attribute_rule, type_ref, ctx)
            }
            RuleMode::SentinelNone => self
                .resolve_sentinel_none(type_ref, ctx)
                .map_err(RunError::Rule),
            RuleMode::MethodImpl { target_rules } => {
                self.resolve_method_impl(target_rules, type_ref, ctx)
            }
            RuleMode::AutoMethod { target_rules } => {
                self.resolve_auto_method(target_rules, type_ref, ctx)
            }
            RuleMode::AttributeSource { inner } => {
                self.resolve_attribute_source(inner, type_ref, ctx)
            }
            RuleMode::Constructor { param_rules } => {
                self.resolve_constructor(param_rules, type_ref, ctx)
            }
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
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> RegistryRunResult<'ty, MemberResolutionResult<'ty>> {
        let env = self.current_env(ctx);
        let mut resolved = BTreeMap::new();
        let mut errors = Vec::new();

        for (name, member_type) in members {
            match self.solve_child_named(
                *member_type,
                Arc::clone(name),
                rule_id,
                LazyDepthMode::Keep,
                Arc::clone(&env),
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
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> Result<SolverResolutionNode<'ty>, ResolutionError<'ty>> {
        let entries = self.lookup_constants(query, ctx);
        let type_ref = query.type_ref;
        if entries.is_empty() {
            return Err(ResolutionError::NoConstantFound(type_ref));
        }

        match entries.as_slice() {
            [(_, source)] => Ok(SolverResolutionNode::Constant {
                source: source.clone(),
            }),
            [] => Err(ResolutionError::NoConstantFound(type_ref)),
            _ => Err(ResolutionError::AmbiguousConstant(type_ref)),
        }
    }

    #[instrumented(
        name = "inlay.rule.resolve_property",
        target = "inlay",
        level = "trace",
        ret,
        err,
        skip(type_ref),
        fields(type_hash = debug_hash(&type_ref), matched_properties)
    )]
    fn resolve_property(
        &self,
        inner: RuleId,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionNode<'ty>> {
        let mut matched = self.lookup_properties(type_ref, ctx);
        let mut seen = HashSet::default();
        matched.retain(|property| {
            seen.insert((property.source.kind.clone(), Arc::clone(&property.name)))
        });
        inlay_span_record!(matched_properties = matched.len() as u64);

        match matched.as_slice() {
            [] => Err(RunError::Rule(ResolutionError::NoPropertyFound(type_ref))),
            [property] => {
                let source = self.solve_child(
                    PyType::Protocol(property.source_type),
                    inner,
                    LazyDepthMode::Keep,
                    self.current_env(ctx),
                    ctx,
                )?;
                Ok(SolverResolutionNode::Property {
                    source,
                    property_name: Arc::clone(&property.name),
                })
            }
            _ => {
                let mut errors = Vec::new();
                for property in &matched {
                    match self.solve_child(
                        PyType::Protocol(property.source_type),
                        inner,
                        LazyDepthMode::Keep,
                        self.current_env(ctx),
                        ctx,
                    ) {
                        Ok(source) => {
                            return Ok(SolverResolutionNode::Property {
                                source,
                                property_name: Arc::clone(&property.name),
                            });
                        }
                        Err(RunError::Rule(error)) => errors.push(Arc::new(error)),
                        Err(RunError::Solve(error)) => return Err(RunError::Solve(error)),
                    }
                }
                Err(RunError::Rule(ResolutionError::MissingDependency(
                    type_ref, errors,
                )))
            }
        }
    }

    #[instrumented(
        name = "inlay.rule.resolve_lazy_ref",
        target = "inlay",
        level = "trace",
        ret,
        err,
        skip(type_ref),
        fields(type_hash = debug_hash(&type_ref), inner_rule = inner.index() as u64)
    )]
    fn resolve_lazy_ref(
        &self,
        inner: RuleId,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionNode<'ty>> {
        let PyType::LazyRef(key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };
        let target = ctx
            .shared()
            .types()
            .concrete
            .lazy_refs
            .get(key)
            .inner
            .target;
        let target = self.solve_child(
            target,
            inner,
            LazyDepthMode::Increment,
            self.current_env(ctx),
            ctx,
        )?;
        Ok(SolverResolutionNode::LazyRef { target })
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
            allow_none_fallback,
            variants,
            resolved_variants,
            errors
        )
    )]
    fn resolve_union(
        &self,
        variant_rules: RuleId,
        allow_none_fallback: bool,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionNode<'ty>> {
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
                self.current_env(ctx),
            ) {
                Ok(SolveResult::Resolved { result, result_ref }) => match result {
                    Ok(output)
                        if !allow_none_fallback
                            && matches!(output.resolution, SolverResolutionNode::None) => {}
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
            Ok(SolverResolutionNode::UnionVariant {
                target: resolved[0].1,
            })
        } else if allow_none_fallback && union_contains_none(&*ctx.shared().types(), &variants) {
            Ok(SolverResolutionNode::None)
        } else {
            Err(RunError::Rule(ResolutionError::MissingDependency(
                type_ref, errors,
            )))
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
        ctx: &mut RegistryRuleContext<'_, 'ty>,
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
                .map(|(name, &member_type)| (Arc::clone(name), member_type))
                .collect();
            let attribute_members: Vec<_> = protocol
                .inner
                .attributes
                .iter()
                .map(|(name, &member_type)| (Arc::clone(name), member_type))
                .collect();
            let method_members: Vec<_> = protocol
                .inner
                .methods
                .iter()
                .map(|(name, &member_type)| (Arc::clone(name), member_type))
                .collect();
            (property_members, attribute_members, method_members)
        };
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
            (method_rule, method_members.as_slice()),
        ] {
            match self.resolve_members(member_list, rule_id, ctx)? {
                Ok(resolved) => members.extend(resolved),
                Err(member_errors) => errors.extend(member_errors),
            }
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
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionNode<'ty>> {
        let PyType::TypedDict(key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };
        let attribute_members = ctx
            .shared()
            .types()
            .concrete
            .typed_dicts
            .get(key)
            .inner
            .attributes
            .iter()
            .map(|(name, &member_type)| (Arc::clone(name), member_type))
            .collect::<Vec<_>>();
        inlay_span_record!(attribute_members = attribute_members.len() as u64);

        match self.resolve_members(&attribute_members, attribute_rule, ctx)? {
            Ok(members) => Ok(SolverResolutionNode::TypedDict { members }),
            Err(errors) => Err(RunError::Rule(ResolutionError::MissingDependency(
                type_ref, errors,
            ))),
        }
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
        ctx: &mut RegistryRuleContext<'_, 'ty>,
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

    #[instrumented(
        name = "inlay.rule.resolve_auto_method",
        target = "inlay",
        level = "trace",
        ret,
        err,
        skip(type_ref),
        fields(
            type_hash = debug_hash(&type_ref),
            target_rule = target_rules.index() as u64,
            params
        )
    )]
    fn resolve_auto_method(
        &self,
        target_rules: RuleId,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionNode<'ty>> {
        let PyType::Callable(request_key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };
        let (result_type, return_wrapper, accepts_varargs, accepts_varkw, param_info, method_qual) = {
            let types = ctx.shared().types();
            let callable = types.concrete.callables.get(request_key).clone();
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
                types.qualifier_of_concrete(type_ref).clone(),
            )
        };
        let param_info: Vec<(Arc<str>, PyTypeConcreteKey<'ty>, ParamKind)> = {
            let types = ctx.shared().types();
            param_info
                .into_iter()
                .map(|(name, param_type, kind)| {
                    let param_type = requalify_concrete(param_type, &method_qual, types);
                    (name, param_type, kind)
                })
                .collect()
        };
        let params: Vec<MethodParam<'ty>> = param_info
            .into_iter()
            .map(|(name, param_type, kind)| MethodParam {
                source: ctx
                    .env()
                    .transition_param_source(Arc::clone(&name), param_type, 0),
                name,
                kind,
                param_type,
            })
            .collect();
        inlay_span_record!(params = params.len() as u64);

        let transition_params = params
            .iter()
            .map(|param| (Arc::clone(&param.name), param.param_type))
            .collect();
        let env = self.transition_env(ctx, transition_params, Vec::new());
        let target = self.solve_child(
            result_type,
            target_rules,
            LazyDepthMode::Increment,
            Arc::clone(&env),
            ctx,
        )?;

        Ok(SolverResolutionNode::AutoMethod {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            target,
        })
    }

    #[instrumented(
        name = "inlay.rule.resolve_method_impl",
        target = "inlay",
        level = "trace",
        ret,
        err,
        skip(type_ref),
        fields(
            type_hash = debug_hash(&type_ref),
            target_rule = target_rules.index() as u64,
            matched_methods,
            params,
            implementations
        )
    )]
    fn resolve_method_impl(
        &self,
        target_rules: RuleId,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionNode<'ty>> {
        let PyType::Callable(request_key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };
        let (
            request_result_type,
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            param_info,
            request_method_qual,
        ) = {
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
                types.qualifier_of_concrete(type_ref).clone(),
            )
        };

        let matched = self.lookup_methods(type_ref, ctx);
        inlay_span_record!(matched_methods = matched.len() as u64);
        if matched.is_empty() {
            return Err(RunError::Rule(ResolutionError::NoMethodFound(type_ref)));
        }

        let param_info: Vec<(Arc<str>, PyTypeConcreteKey<'ty>, ParamKind)> = {
            let types = ctx.shared().types();
            param_info
                .into_iter()
                .map(|(name, param_type, kind)| {
                    let param_type = requalify_concrete(param_type, &request_method_qual, types);
                    (name, param_type, kind)
                })
                .collect()
        };
        let params: Vec<MethodParam<'ty>> = param_info
            .into_iter()
            .map(|(name, param_type, kind)| MethodParam {
                source: ctx
                    .env()
                    .transition_param_source(Arc::clone(&name), param_type, 0),
                name,
                kind,
                param_type,
            })
            .collect();
        inlay_span_record!(params = params.len() as u64);

        let transition_params = params
            .iter()
            .map(|param| (Arc::clone(&param.name), param.param_type))
            .collect();
        let mut env = self.transition_env(ctx, transition_params, Vec::new());
        let mut implementations = Vec::with_capacity(matched.len());

        for matched in matched {
            let callable = ctx
                .shared()
                .types()
                .concrete
                .callables
                .get(matched.concrete_implementation_callable_key)
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

            let bound_to = matched.concrete_bound_to.map(|bound_type| {
                self.solve_child(
                    bound_type,
                    target_rules,
                    LazyDepthMode::Keep,
                    Arc::clone(&env),
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
                    Arc::clone(&env),
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
                let scope = matched.implementation.order + 1;
                let result_source = ctx.env().transition_result_source(result_type, scope);
                env = Arc::new(env.with_transition(Vec::new(), vec![(result_type, scope)]));
                Some(result_source)
            };

            implementations.push(SolverResolvedMethodImplementation {
                implementation: matched.implementation,
                bound_to,
                params: implementation_params,
                return_wrapper: callable.inner.return_wrapper,
                result_source,
            });
        }
        inlay_span_record!(implementations = implementations.len() as u64);

        let target = self.solve_child(
            request_result_type,
            target_rules,
            LazyDepthMode::Increment,
            Arc::clone(&env),
            ctx,
        )?;

        Ok(SolverResolutionNode::Method {
            return_wrapper,
            accepts_varargs,
            accepts_varkw,
            params,
            implementations,
            target,
        })
    }

    #[instrumented(
        name = "inlay.rule.resolve_attribute_source",
        target = "inlay",
        level = "trace",
        ret,
        err,
        skip(type_ref),
        fields(type_hash = debug_hash(&type_ref), matched_attributes)
    )]
    fn resolve_attribute_source(
        &self,
        inner: RuleId,
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionNode<'ty>> {
        let mut matched = self.lookup_attributes(type_ref, ctx);
        let mut seen = HashSet::default();
        matched.retain(|attribute| {
            seen.insert((attribute.source.kind.clone(), Arc::clone(&attribute.name)))
        });
        inlay_span_record!(matched_attributes = matched.len() as u64);

        match matched.as_slice() {
            [] => Err(RunError::Rule(ResolutionError::NoAttributeFound(type_ref))),
            [attribute] => {
                let (source_type, access_kind) = match attribute.source_type {
                    SourceType::Protocol(source_type) => {
                        (PyType::Protocol(source_type), MemberAccessKind::Attribute)
                    }
                    SourceType::TypedDict(source_type) => {
                        (PyType::TypedDict(source_type), MemberAccessKind::DictItem)
                    }
                };
                let source = self.solve_child(
                    source_type,
                    inner,
                    LazyDepthMode::Keep,
                    self.current_env(ctx),
                    ctx,
                )?;
                Ok(SolverResolutionNode::Attribute {
                    source,
                    attribute_name: Arc::clone(&attribute.name),
                    access_kind,
                })
            }
            _ => {
                let mut errors = Vec::new();
                for attribute in &matched {
                    let (source_type, access_kind) = match attribute.source_type {
                        SourceType::Protocol(source_type) => {
                            (PyType::Protocol(source_type), MemberAccessKind::Attribute)
                        }
                        SourceType::TypedDict(source_type) => {
                            (PyType::TypedDict(source_type), MemberAccessKind::DictItem)
                        }
                    };
                    match self.solve_child(
                        source_type,
                        inner,
                        LazyDepthMode::Keep,
                        self.current_env(ctx),
                        ctx,
                    ) {
                        Ok(source) => {
                            return Ok(SolverResolutionNode::Attribute {
                                source,
                                attribute_name: Arc::clone(&attribute.name),
                                access_kind,
                            });
                        }
                        Err(RunError::Rule(error)) => errors.push(Arc::new(error)),
                        Err(RunError::Solve(error)) => return Err(RunError::Solve(error)),
                    }
                }
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
        type_ref: PyTypeConcreteKey<'ty>,
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionNode<'ty>> {
        let matched = self.lookup_constructors(type_ref, ctx);
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

        let mut params = Vec::with_capacity(param_info.len());
        for (name, param_type, kind, has_default) in param_info {
            match self.solve_child_named(
                param_type,
                Arc::clone(&name),
                param_rules,
                LazyDepthMode::Keep,
                self.current_env(ctx),
                ctx,
            ) {
                Ok(result_ref) => params.push((result_ref, name, kind)),
                Err(RunError::Rule(_)) if has_default => {}
                Err(error) => return Err(error),
            }
        }
        inlay_span_record!(params = params.len() as u64);

        Ok(SolverResolutionNode::Constructor {
            implementation: matched.constructor,
            params,
        })
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
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionNode<'ty>> {
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
        ctx: &mut RegistryRuleContext<'_, 'ty>,
    ) -> RegistryRunResult<'ty, SolverResolutionNode<'ty>> {
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
        ctx: &mut RegistryRuleContext<'_, 'ty>,
        causes: &mut Vec<Arc<ResolutionError<'ty>>>,
        cause_count: &mut usize,
    ) -> RegistryRunResult<'ty, SolverResolutionNode<'ty>> {
        let type_ref = query.type_ref;

        for &rule_id in rules {
            let rule_label = self.rule_label(rule_id);
            match ctx.solve(
                query.clone(),
                rule_id,
                LazyDepthMode::Keep,
                self.current_env(ctx),
            ) {
                Ok(SolveResult::Resolved { result, result_ref }) => match result {
                    Ok(_) => return Ok(SolverResolutionNode::Delegate(result_ref)),
                    Err(error) => causes.push(Arc::new(ResolutionError::RuleError {
                        rule_label,
                        cause: Arc::new(error.clone()),
                    })),
                },
                Ok(SolveResult::Lazy { result_ref }) => {
                    return Ok(SolverResolutionNode::Delegate(result_ref));
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
        let resolution = self.resolve_rule(rule, &query, ctx);
        let resolution = resolution?;

        Ok(SolverResolvedNode {
            target_type: query.type_ref,
            resolution,
        })
    }
}

fn union_contains_none<'ty>(arenas: &TypeArenas<'ty>, variants: &[PyTypeConcreteKey<'ty>]) -> bool {
    variants.iter().any(|variant| {
        if let PyType::Sentinel(key) = variant {
            matches!(
                arenas.sentinels.get(*key).inner.value,
                SentinelTypeKind::None
            )
        } else {
            false
        }
    })
}

fn union_subtype_sort_key<'ty>(
    sub: PyTypeConcreteKey<'ty>,
    target_variants: &[PyTypeConcreteKey<'ty>],
    arenas: &TypeArenas<'ty>,
) -> (i32, Vec<usize>) {
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
        (-(sub_variants.len() as i32), positions)
    } else {
        (-1, vec![*target_positions.get(&sub).unwrap_or(&fallback)])
    }
}
