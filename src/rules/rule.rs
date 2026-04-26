use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Arc;

use context_solver::{
    Arena as ResultsArena, LazyDepthMode, ReplaceError, Rule as SolverRule, RuleContext, RunError,
    solve::{SolveError, SolveResult},
};
use derive_where::derive_where;
use inlay_instrument_macros::instrumented;
use rustc_hash::{FxHashSet as HashSet, FxHasher};
use slotmap::{SlotMap, new_key_type};

use crate::{
    instrument::inlay_span_record,
    qualifier::Qualifier,
    registry::{ConstantType, Constructor, Hook, MethodImplementation, Source, SourceType},
    types::{
        Arena, ArenaFamily, ParamKind, PyType, PyTypeConcreteKey, SentinelTypeKind, TypeArenas,
        WrapperKind, requalify_concrete,
    },
};

use super::{
    MethodParam, ResolutionError, RuleArena, RuleId, RuleMode, TransitionResultBinding,
    env::{
        Attribute, ConstructorLookup, HookLookup, MethodLookup, Property, RegistryEnv,
        ResolutionLookup, ResolutionLookupResult, summarize_env_for_trace,
        summarize_lookup_for_trace, summarize_lookup_result_for_trace,
    },
};

new_key_type! {
    pub(crate) struct SolverResolutionRef;
}

type SolverResolutionResult<S> = Result<SolverResolvedNode<S>, ResolutionError<S>>;
type RegistryRuleContext<'a, S> = RuleContext<'a, RegistryResolutionRule<S>>;
type RegistryRunError<S> = RunError<RegistryResolutionRule<S>>;
type RegistryRunResult<T, S> = Result<T, RegistryRunError<S>>;

fn debug_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = FxHasher::default();
    value.hash(&mut hasher);
    hasher.finish()
}

#[derive_where(Clone, PartialEq, Eq, Hash)]
pub(crate) struct ResolutionQuery<S: ArenaFamily> {
    pub(crate) type_ref: PyTypeConcreteKey<S>,
    pub(crate) requested_name: Option<Arc<str>>,
}

impl<S: ArenaFamily> ResolutionQuery<S> {
    pub(crate) fn unnamed(type_ref: PyTypeConcreteKey<S>) -> Self {
        Self {
            type_ref,
            requested_name: None,
        }
    }

    pub(crate) fn named(type_ref: PyTypeConcreteKey<S>, requested_name: Arc<str>) -> Self {
        Self {
            type_ref,
            requested_name: Some(requested_name),
        }
    }
}

impl<S: ArenaFamily> std::fmt::Debug for ResolutionQuery<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolutionQuery")
            .field("type_hash", &debug_hash(&self.type_ref))
            .field("requested_name", &self.requested_name)
            .finish()
    }
}

pub(crate) struct SolverResolutionArena<S: ArenaFamily> {
    results: SlotMap<SolverResolutionRef, Option<SolverResolutionResult<S>>>,
}

impl<S: ArenaFamily> std::fmt::Debug for SolverResolutionArena<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolverResolutionArena")
            .field("results", &self.results.len())
            .finish()
    }
}

impl<S: ArenaFamily> Default for SolverResolutionArena<S> {
    fn default() -> Self {
        Self {
            results: SlotMap::with_key(),
        }
    }
}

impl<S: ArenaFamily> ResultsArena<SolverResolutionResult<S>> for SolverResolutionArena<S> {
    type Key = SolverResolutionRef;

    fn insert(&mut self, val: SolverResolutionResult<S>) -> Self::Key
    where
        SolverResolutionResult<S>: std::hash::Hash + Eq,
    {
        self.results.insert(Some(val))
    }

    fn insert_placeholder(&mut self) -> Self::Key {
        self.results.insert(None)
    }

    fn replace(
        &mut self,
        key: Self::Key,
        val: SolverResolutionResult<S>,
    ) -> Result<Option<SolverResolutionResult<S>>, ReplaceError>
    where
        SolverResolutionResult<S>: std::hash::Hash + Eq,
    {
        Ok(self
            .results
            .get_mut(key)
            .ok_or(ReplaceError::InvalidKey)?
            .replace(val))
    }

    fn get(&self, key: &Self::Key) -> Option<&SolverResolutionResult<S>> {
        self.results.get(*key)?.as_ref()
    }

    fn len(&self) -> usize {
        self.results.len()
    }
}

impl<S: ArenaFamily> SolverResolutionArena<S> {
    pub(crate) fn len(&self) -> usize {
        self.results.len()
    }
}

#[derive_where(Clone, PartialEq, Eq, Hash)]
pub(crate) struct SolverResolvedHook<S: ArenaFamily> {
    pub(crate) hook: Arc<Hook<S>>,
    pub(crate) params: Vec<(SolverResolutionRef, Arc<str>, ParamKind)>,
}

impl<S: ArenaFamily> std::fmt::Debug for SolverResolvedHook<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolverResolvedHook")
            .field("params", &self.params.len())
            .finish()
    }
}

#[derive_where(Clone, PartialEq, Eq, Hash)]
pub(crate) enum SolverResolutionNode<S: ArenaFamily> {
    Constant {
        source: Source<S>,
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
        implementation: Arc<MethodImplementation<S>>,
        return_wrapper: WrapperKind,
        bound_to: Option<SolverResolutionRef>,
        params: Vec<MethodParam<S>>,
        result_source: Option<Source<S>>,
        result_bindings: Vec<super::TransitionResultBinding<S>>,
        target: SolverResolutionRef,
        hooks: Vec<SolverResolvedHook<S>>,
    },
    AutoMethod {
        return_wrapper: WrapperKind,
        params: Vec<MethodParam<S>>,
        target: SolverResolutionRef,
        hooks: Vec<SolverResolvedHook<S>>,
    },
    Attribute {
        source: SolverResolutionRef,
        attribute_name: Arc<str>,
    },
    Constructor {
        implementation: Arc<Constructor<S>>,
        params: Vec<(SolverResolutionRef, Arc<str>, ParamKind)>,
    },
    Delegate(SolverResolutionRef),
}

impl<S: ArenaFamily> std::fmt::Debug for SolverResolutionNode<S> {
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
                bound_to,
                params,
                result_bindings,
                target,
                hooks,
                ..
            } => f
                .debug_struct("Method")
                .field("bound_to", bound_to)
                .field("params", &params.len())
                .field("result_bindings", &result_bindings.len())
                .field("target", target)
                .field("hooks", &hooks.len())
                .finish(),
            Self::AutoMethod {
                params,
                target,
                hooks,
                ..
            } => f
                .debug_struct("AutoMethod")
                .field("params", &params.len())
                .field("target", target)
                .field("hooks", &hooks.len())
                .finish(),
            Self::Attribute {
                source,
                attribute_name,
            } => f
                .debug_struct("Attribute")
                .field("source", source)
                .field("attribute_name", attribute_name)
                .finish(),
            Self::Constructor { params, .. } => f
                .debug_struct("Constructor")
                .field("params", &params.len())
                .finish(),
            Self::Delegate(result_ref) => f.debug_tuple("Delegate").field(result_ref).finish(),
        }
    }
}

#[derive_where(Clone, PartialEq, Eq, Hash)]
pub(crate) struct SolverResolvedNode<S: ArenaFamily> {
    pub(crate) target_type: PyTypeConcreteKey<S>,
    pub(crate) resolution: SolverResolutionNode<S>,
}

impl<S: ArenaFamily> std::fmt::Debug for SolverResolvedNode<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolverResolvedNode")
            .field("target_hash", &debug_hash(&self.target_type))
            .field("resolution", &self.resolution)
            .finish()
    }
}

#[derive(Clone)]
pub(crate) struct RegistryResolutionRule<S: ArenaFamily> {
    rules: Arc<RuleArena>,
    _phantom: PhantomData<S>,
}

impl<S: ArenaFamily> std::fmt::Debug for RegistryResolutionRule<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryResolutionRule").finish()
    }
}

impl<S: ArenaFamily> RegistryResolutionRule<S> {
    pub(crate) fn new(rules: Arc<RuleArena>) -> Self {
        Self {
            rules,
            _phantom: PhantomData,
        }
    }

    fn rule_label(&self, rule_id: RuleId) -> &'static str {
        self.rules
            .get(rule_id)
            .map(RuleMode::label)
            .unwrap_or("unknown")
    }

    fn current_env(&self, ctx: &RegistryRuleContext<'_, S>) -> Arc<RegistryEnv<S>> {
        Arc::new(ctx.env().clone())
    }

    fn transition_env(
        &self,
        ctx: &RegistryRuleContext<'_, S>,
        params: Vec<(Arc<str>, PyTypeConcreteKey<S>)>,
        return_type: Option<PyTypeConcreteKey<S>>,
        result_bindings: Vec<(Arc<str>, PyTypeConcreteKey<S>)>,
    ) -> Arc<RegistryEnv<S>> {
        Arc::new(
            ctx.env()
                .with_transition(params, return_type, result_bindings),
        )
    }

    fn transition_result_bindings(
        &self,
        result_type: PyTypeConcreteKey<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> (
        Vec<(Arc<str>, PyTypeConcreteKey<S>)>,
        Vec<TransitionResultBinding<S>>,
    ) {
        let PyType::TypedDict(key) = result_type else {
            return (Vec::new(), Vec::new());
        };

        let attributes: Vec<_> = ctx
            .shared()
            .types()
            .concrete
            .typed_dicts
            .get(&key)
            .expect("dangling key")
            .inner
            .attributes
            .iter()
            .map(|(name, &member_type)| (Arc::clone(name), member_type))
            .collect();

        let mut env_bindings = Vec::new();
        let mut runtime_bindings = Vec::new();
        for (name, member_type) in attributes {
            let Some(source) = ctx
                .env()
                .transition_param_source(Arc::clone(&name), member_type)
            else {
                continue;
            };
            env_bindings.push((Arc::clone(&name), member_type));
            runtime_bindings.push(TransitionResultBinding { name, source });
        }

        (env_bindings, runtime_bindings)
    }

    fn solve_child_query(
        &self,
        query: ResolutionQuery<S>,
        state_id: RuleId,
        lazy_depth_mode: LazyDepthMode,
        env: Arc<RegistryEnv<S>>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> RegistryRunResult<SolverResolutionRef, S> {
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
        query: PyTypeConcreteKey<S>,
        state_id: RuleId,
        lazy_depth_mode: LazyDepthMode,
        env: Arc<RegistryEnv<S>>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> RegistryRunResult<SolverResolutionRef, S> {
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
        query: PyTypeConcreteKey<S>,
        requested_name: Arc<str>,
        state_id: RuleId,
        lazy_depth_mode: LazyDepthMode,
        env: Arc<RegistryEnv<S>>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> RegistryRunResult<SolverResolutionRef, S> {
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
        query: &ResolutionQuery<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> Vec<(ConstantType<S>, Source<S>)> {
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
        type_ref: PyTypeConcreteKey<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> Vec<ConstructorLookup<S>> {
        let entries: BTreeSet<_> = ctx
            .shared()
            .lookup_constructors(type_ref)
            .into_iter()
            .collect();
        entries.into_iter().collect()
    }

    fn lookup_methods(
        &self,
        type_ref: PyTypeConcreteKey<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> Vec<MethodLookup<S>> {
        let entries: BTreeSet<_> = ctx.shared().lookup_methods(type_ref).into_iter().collect();
        entries.into_iter().collect()
    }

    fn lookup_hooks(
        &self,
        name: &str,
        method_qual: Option<&Qualifier>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> Vec<HookLookup<S>> {
        let entries: BTreeSet<_> = ctx
            .shared()
            .lookup_hooks(&Arc::from(name), method_qual)
            .into_iter()
            .collect();
        entries.into_iter().collect()
    }

    fn lookup_properties(
        &self,
        type_ref: PyTypeConcreteKey<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> Vec<Property<S, crate::types::Concrete>> {
        let ResolutionLookupResult::Properties(entries) =
            ctx.lookup(&ResolutionLookup::Property(type_ref))
        else {
            unreachable!();
        };
        entries.into_iter().collect()
    }

    fn lookup_attributes(
        &self,
        type_ref: PyTypeConcreteKey<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> Vec<Attribute<S, crate::types::Concrete>> {
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
        query: &ResolutionQuery<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> RegistryRunResult<SolverResolutionNode<S>, S> {
        let type_ref = query.type_ref;
        match rule {
            RuleMode::ConstantRule => self.resolve_constant(query, ctx).map_err(RunError::Rule),
            RuleMode::PropertyRule { inner } => self.resolve_property(inner, type_ref, ctx),
            RuleMode::LazyRefRule { inner } => self.resolve_lazy_ref(inner, type_ref, ctx),
            RuleMode::UnionRule {
                variant_rules,
                allow_none_fallback,
            } => self.resolve_union(variant_rules, allow_none_fallback, type_ref, ctx),
            RuleMode::ProtocolRule {
                property_rule,
                attribute_rule,
                method_rule,
            } => self.resolve_protocol(property_rule, attribute_rule, method_rule, type_ref, ctx),
            RuleMode::TypedDictRule { attribute_rule } => {
                self.resolve_typed_dict(attribute_rule, type_ref, ctx)
            }
            RuleMode::SentinelNoneRule => self
                .resolve_sentinel_none(type_ref, ctx)
                .map_err(RunError::Rule),
            RuleMode::MethodImplRule {
                target_rules,
                hook_param_rule,
            } => self.resolve_method_impl(target_rules, hook_param_rule, type_ref, ctx),
            RuleMode::AutoMethodRule {
                target_rules,
                hook_param_rule,
            } => self.resolve_auto_method(target_rules, hook_param_rule, type_ref, ctx),
            RuleMode::AttributeSourceRule { inner } => {
                self.resolve_attribute_source(inner, type_ref, ctx)
            }
            RuleMode::ConstructorRule { param_rules } => {
                self.resolve_constructor(param_rules, type_ref, ctx)
            }
            RuleMode::MatchFirstRule { rules } => self.resolve_match_first(&rules, query, ctx),
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
        members: &[(Arc<str>, PyTypeConcreteKey<S>)],
        rule_id: RuleId,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> RegistryRunResult<
        Result<BTreeMap<Arc<str>, SolverResolutionRef>, Vec<Arc<ResolutionError<S>>>>,
        S,
    > {
        let env = self.current_env(ctx);
        let mut resolved = BTreeMap::new();
        let mut errors = Vec::new();

        for (name, member_type) in members {
            match self.solve_child(
                *member_type,
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
        query: &ResolutionQuery<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> Result<SolverResolutionNode<S>, ResolutionError<S>> {
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
        type_ref: PyTypeConcreteKey<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> RegistryRunResult<SolverResolutionNode<S>, S> {
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
        type_ref: PyTypeConcreteKey<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> RegistryRunResult<SolverResolutionNode<S>, S> {
        let PyType::LazyRef(key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };
        let target = ctx
            .shared()
            .types()
            .concrete
            .lazy_refs
            .get(&key)
            .expect("dangling key")
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
        type_ref: PyTypeConcreteKey<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> RegistryRunResult<SolverResolutionNode<S>, S> {
        let PyType::Union(key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };
        let variants = ctx
            .shared()
            .types()
            .concrete
            .unions
            .get(&key)
            .expect("dangling key")
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
        type_ref: PyTypeConcreteKey<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> RegistryRunResult<SolverResolutionNode<S>, S> {
        let PyType::Protocol(key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };
        let (property_members, attribute_members, method_members) = {
            let protocol = ctx
                .shared()
                .types()
                .concrete
                .protocols
                .get(&key)
                .expect("dangling key")
                .clone();
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

        if method_members.iter().any(|(_, member_type)| {
            !is_structural_protocol_method_return(*member_type, ctx.shared().types())
        }) {
            return Err(RunError::Rule(ResolutionError::NoConstructorFound(
                type_ref,
            )));
        }

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
        name = "inlay.rule.resolve_hooks",
        target = "inlay",
        level = "trace",
        ret,
        err,
        fields(
            method_name,
            hook_param_rule = hook_param_rule.index() as u64,
            has_method_qual = method_qual.is_some(),
            hooks,
            resolved_hooks
        )
    )]
    fn resolve_hooks(
        &self,
        method_name: &str,
        hook_param_rule: RuleId,
        method_qual: Option<&Qualifier>,
        env: Arc<RegistryEnv<S>>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> RegistryRunResult<Vec<SolverResolvedHook<S>>, S> {
        let hooks = self.lookup_hooks(method_name, method_qual, ctx);
        inlay_span_record!(hooks = hooks.len() as u64);
        let mut resolved_hooks = Vec::new();

        for hook_lookup in hooks {
            let callable = ctx
                .shared()
                .types()
                .concrete
                .callables
                .get(&hook_lookup.concrete_callable_key)
                .expect("dangling key")
                .clone();
            let param_info: Vec<(Arc<str>, PyTypeConcreteKey<S>, ParamKind, bool)> = callable
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
                    hook_param_rule,
                    LazyDepthMode::Keep,
                    Arc::clone(&env),
                    ctx,
                ) {
                    Ok(result_ref) => params.push((result_ref, name, kind)),
                    Err(RunError::Rule(_)) if has_default => {}
                    Err(error) => return Err(error),
                }
            }

            resolved_hooks.push(SolverResolvedHook {
                hook: hook_lookup.hook,
                params,
            });
        }

        inlay_span_record!(resolved_hooks = resolved_hooks.len() as u64);
        Ok(resolved_hooks)
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
        type_ref: PyTypeConcreteKey<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> RegistryRunResult<SolverResolutionNode<S>, S> {
        let PyType::TypedDict(key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };
        let attribute_members = ctx
            .shared()
            .types()
            .concrete
            .typed_dicts
            .get(&key)
            .expect("dangling key")
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
        type_ref: PyTypeConcreteKey<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> Result<SolverResolutionNode<S>, ResolutionError<S>> {
        let PyType::Sentinel(key) = type_ref else {
            return Err(ResolutionError::IncompatibleType(type_ref));
        };
        let sentinel = ctx
            .shared()
            .types()
            .sentinels
            .get(&key)
            .expect("dangling key");
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
            has_hook_param_rule = hook_param_rule.is_some(),
            params,
            hooks
        )
    )]
    fn resolve_auto_method(
        &self,
        target_rules: RuleId,
        hook_param_rule: Option<RuleId>,
        type_ref: PyTypeConcreteKey<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> RegistryRunResult<SolverResolutionNode<S>, S> {
        let PyType::Callable(request_key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };
        let (result_type, return_wrapper, method_name, param_info, method_qual) = {
            let types = ctx.shared().types();
            let callable = types
                .concrete
                .callables
                .get(&request_key)
                .expect("dangling key")
                .clone();
            let param_info: Vec<(Arc<str>, PyTypeConcreteKey<S>, ParamKind)> = callable
                .inner
                .params
                .iter()
                .zip(callable.inner.param_kinds.iter())
                .map(|((name, &param_type), &kind)| (Arc::clone(name), param_type, kind))
                .collect();
            (
                callable.inner.return_type,
                callable.inner.return_wrapper,
                callable.inner.function_name,
                param_info,
                types.qualifier_of_concrete(type_ref).cloned(),
            )
        };
        let param_info: Vec<(Arc<str>, PyTypeConcreteKey<S>, ParamKind)> = {
            let types = ctx.shared().types();
            param_info
                .into_iter()
                .map(|(name, param_type, kind)| {
                    let param_type = method_qual.as_ref().map_or(param_type, |qualifier| {
                        requalify_concrete(param_type, qualifier, types)
                    });
                    (name, param_type, kind)
                })
                .collect()
        };
        let params: Vec<MethodParam<S>> = param_info
            .into_iter()
            .map(|(name, param_type, kind)| MethodParam {
                source: ctx
                    .env()
                    .transition_param_source(Arc::clone(&name), param_type),
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
        let env = self.transition_env(ctx, transition_params, None, Vec::new());
        let target = self.solve_child(
            result_type,
            target_rules,
            LazyDepthMode::Increment,
            Arc::clone(&env),
            ctx,
        )?;
        let hooks = match (hook_param_rule, method_name.as_deref()) {
            (Some(hook_param_rule), Some(name)) => {
                self.resolve_hooks(name, hook_param_rule, method_qual.as_ref(), env, ctx)?
            }
            _ => Vec::new(),
        };
        inlay_span_record!(hooks = hooks.len() as u64);

        Ok(SolverResolutionNode::AutoMethod {
            return_wrapper,
            params,
            target,
            hooks,
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
            has_hook_param_rule = hook_param_rule.is_some(),
            matched_methods,
            params,
            result_bindings,
            hooks,
            has_bound_to
        )
    )]
    fn resolve_method_impl(
        &self,
        target_rules: RuleId,
        hook_param_rule: Option<RuleId>,
        type_ref: PyTypeConcreteKey<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> RegistryRunResult<SolverResolutionNode<S>, S> {
        let PyType::Callable(request_key) = type_ref else {
            return Err(RunError::Rule(ResolutionError::IncompatibleType(type_ref)));
        };
        let (request_result_type, request_method_qual, request_result_qual) = {
            let types = ctx.shared().types();
            let callable = types
                .concrete
                .callables
                .get(&request_key)
                .expect("dangling key");
            (
                callable.inner.return_type,
                types
                    .qualifier_of_concrete(type_ref)
                    .expect("dangling key")
                    .clone(),
                types
                    .qualifier_of_concrete(callable.inner.return_type)
                    .expect("dangling key")
                    .clone(),
            )
        };

        let matched = self.lookup_methods(type_ref, ctx);
        inlay_span_record!(matched_methods = matched.len() as u64);
        let matched = match matched.as_slice() {
            [] => return Err(RunError::Rule(ResolutionError::NoMethodFound(type_ref))),
            [matched] => matched.clone(),
            [_, _, ..] => {
                return Err(RunError::Rule(ResolutionError::AmbiguousMethod(type_ref)));
            }
        };

        let (result_type, return_wrapper, param_info) = {
            let callable = ctx
                .shared()
                .types()
                .concrete
                .callables
                .get(&matched.concrete_callable_key)
                .expect("dangling key")
                .clone();
            let param_info: Vec<(Arc<str>, PyTypeConcreteKey<S>, ParamKind)> = callable
                .inner
                .params
                .iter()
                .zip(callable.inner.param_kinds.iter())
                .map(|((name, &param_type), &kind)| (Arc::clone(name), param_type, kind))
                .collect();
            (
                callable.inner.return_type,
                callable.inner.return_wrapper,
                param_info,
            )
        };
        let transition_result_type = {
            let types = ctx.shared().types();
            requalify_concrete(result_type, &request_result_qual, types)
        };
        let param_info: Vec<(Arc<str>, PyTypeConcreteKey<S>, ParamKind)> = {
            let types = ctx.shared().types();
            param_info
                .into_iter()
                .map(|(name, param_type, kind)| {
                    let param_type = requalify_concrete(param_type, &request_result_qual, types);
                    (name, param_type, kind)
                })
                .collect()
        };
        let params: Vec<MethodParam<S>> = param_info
            .into_iter()
            .map(|(name, param_type, kind)| MethodParam {
                source: ctx
                    .env()
                    .transition_param_source(Arc::clone(&name), param_type),
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
        let method_name = Arc::clone(&matched.implementation.name);
        let (result_binding_types, result_bindings) =
            self.transition_result_bindings(transition_result_type, ctx);
        inlay_span_record!(result_bindings = result_bindings.len() as u64);
        let env = self.transition_env(
            ctx,
            transition_params,
            Some(transition_result_type),
            result_binding_types,
        );
        let target = self.solve_child(
            request_result_type,
            target_rules,
            LazyDepthMode::Increment,
            Arc::clone(&env),
            ctx,
        )?;
        let hooks = match hook_param_rule {
            Some(hook_param_rule) => self.resolve_hooks(
                method_name.as_ref(),
                hook_param_rule,
                Some(&request_method_qual),
                env,
                ctx,
            )?,
            None => Vec::new(),
        };
        inlay_span_record!(hooks = hooks.len() as u64);
        let bound_to = matched.concrete_bound_to.map(|bound_type| {
            self.solve_child(
                bound_type,
                target_rules,
                LazyDepthMode::Keep,
                self.current_env(ctx),
                ctx,
            )
        });
        let bound_to = match bound_to {
            Some(Ok(bound_to)) => Some(bound_to),
            Some(Err(error)) => return Err(error),
            None => None,
        };
        inlay_span_record!(has_bound_to = bound_to.is_some());

        Ok(SolverResolutionNode::Method {
            implementation: matched.implementation,
            return_wrapper,
            bound_to,
            params,
            result_source: ctx.env().transition_result_source(result_type),
            result_bindings,
            target,
            hooks,
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
        type_ref: PyTypeConcreteKey<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> RegistryRunResult<SolverResolutionNode<S>, S> {
        let mut matched = self.lookup_attributes(type_ref, ctx);
        let mut seen = HashSet::default();
        matched.retain(|attribute| {
            seen.insert((attribute.source.kind.clone(), Arc::clone(&attribute.name)))
        });
        inlay_span_record!(matched_attributes = matched.len() as u64);

        match matched.as_slice() {
            [] => Err(RunError::Rule(ResolutionError::NoAttributeFound(type_ref))),
            [attribute] => {
                let source = self.solve_child(
                    match attribute.source_type {
                        SourceType::Protocol(source_type) => PyType::Protocol(source_type),
                        SourceType::TypedDict(source_type) => PyType::TypedDict(source_type),
                    },
                    inner,
                    LazyDepthMode::Keep,
                    self.current_env(ctx),
                    ctx,
                )?;
                Ok(SolverResolutionNode::Attribute {
                    source,
                    attribute_name: Arc::clone(&attribute.name),
                })
            }
            _ => {
                let mut errors = Vec::new();
                for attribute in &matched {
                    match self.solve_child(
                        match attribute.source_type {
                            SourceType::Protocol(source_type) => PyType::Protocol(source_type),
                            SourceType::TypedDict(source_type) => PyType::TypedDict(source_type),
                        },
                        inner,
                        LazyDepthMode::Keep,
                        self.current_env(ctx),
                        ctx,
                    ) {
                        Ok(source) => {
                            return Ok(SolverResolutionNode::Attribute {
                                source,
                                attribute_name: Arc::clone(&attribute.name),
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
        type_ref: PyTypeConcreteKey<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> RegistryRunResult<SolverResolutionNode<S>, S> {
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
            .get(&matched.concrete_callable_key)
            .expect("dangling key")
            .clone();
        let param_info: Vec<(Arc<str>, PyTypeConcreteKey<S>, ParamKind, bool)> = callable
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
        query: &ResolutionQuery<S>,
        ctx: &mut RegistryRuleContext<'_, S>,
    ) -> RegistryRunResult<SolverResolutionNode<S>, S> {
        let mut causes = Vec::new();
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

        inlay_span_record!(causes = causes.len() as u64);
        Err(RunError::Rule(ResolutionError::MissingDependency(
            type_ref, causes,
        )))
    }
}

impl<S: ArenaFamily> SolverRule for RegistryResolutionRule<S> {
    type Query = ResolutionQuery<S>;
    type Output = SolverResolvedNode<S>;
    type Err = ResolutionError<S>;
    type Env = RegistryEnv<S>;
    type ResultsArena = SolverResolutionArena<S>;
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

    fn debug_query_label(
        &self,
        query: &Self::Query,
        state_id: Self::RuleStateId,
    ) -> Option<String> {
        let mut hasher = FxHasher::default();
        query.hash(&mut hasher);
        let label = format!(
            "query={:x}#rule={}",
            hasher.finish(),
            self.rule_label(state_id)
        );
        match query.requested_name.as_deref() {
            Some(requested_name) => Some(format!("{label}#name={requested_name}")),
            None => Some(label),
        }
    }

    fn debug_env_label(&self, env: &Self::Env) -> Option<String> {
        Some(summarize_env_for_trace(env))
    }

    fn debug_lookup_query_label(&self, query: &ResolutionLookup<S>) -> Option<String> {
        Some(summarize_lookup_for_trace(query))
    }

    fn debug_lookup_result_label(&self, result: &ResolutionLookupResult<S>) -> Option<String> {
        Some(summarize_lookup_result_for_trace::<S>(result))
    }
}

fn union_contains_none<S: ArenaFamily>(
    arenas: &TypeArenas<S>,
    variants: &[PyTypeConcreteKey<S>],
) -> bool {
    variants.iter().any(|variant| {
        if let PyType::Sentinel(key) = variant {
            arenas
                .sentinels
                .get(key)
                .is_some_and(|sentinel| matches!(sentinel.inner.value, SentinelTypeKind::None))
        } else {
            false
        }
    })
}

fn is_structural_protocol_method_return<S: ArenaFamily>(
    callable_type: PyTypeConcreteKey<S>,
    arenas: &TypeArenas<S>,
) -> bool {
    let PyType::Callable(key) = callable_type else {
        return false;
    };
    let callable = arenas.concrete.callables.get(&key).expect("dangling key");
    is_structural_protocol_target(callable.inner.return_type, arenas)
}

fn is_structural_protocol_target<S: ArenaFamily>(
    type_ref: PyTypeConcreteKey<S>,
    arenas: &TypeArenas<S>,
) -> bool {
    match type_ref {
        PyType::Protocol(_) | PyType::TypedDict(_) | PyType::LazyRef(_) => true,
        PyType::Union(key) => arenas
            .concrete
            .unions
            .get(&key)
            .expect("dangling key")
            .inner
            .variants
            .iter()
            .all(|variant| match variant {
                PyType::Sentinel(sentinel_key) => arenas
                    .sentinels
                    .get(sentinel_key)
                    .is_some_and(|sentinel| matches!(sentinel.inner.value, SentinelTypeKind::None)),
                _ => is_structural_protocol_target(*variant, arenas),
            }),
        _ => false,
    }
}

fn union_subtype_sort_key<S: ArenaFamily>(
    sub: PyTypeConcreteKey<S>,
    target_variants: &[PyTypeConcreteKey<S>],
    arenas: &TypeArenas<S>,
) -> (i32, Vec<usize>) {
    let target_positions = target_variants
        .iter()
        .enumerate()
        .map(|(index, &variant)| (variant, index))
        .collect::<BTreeMap<_, _>>();
    let fallback = target_variants.len();

    if let PyType::Union(key) = sub {
        let sub_variants = &arenas
            .concrete
            .unions
            .get(&key)
            .expect("dangling key")
            .inner
            .variants;
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
