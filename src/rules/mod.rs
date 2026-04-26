pub(crate) mod builder;
mod env;
mod rule;

use std::sync::Arc;

use derive_where::derive_where;
use thiserror::Error;

use crate::{
    qualifier::Qualifier,
    registry::Source,
    types::{
        Arena, ArenaFamily, ParamKind, PyType, PyTypeConcreteKey, SentinelTypeKind, SlotBackend,
        TypeArenas,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct RuleId(usize);

impl RuleId {
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }

    pub(crate) fn index(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for RuleId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RuleArena(Vec<RuleMode>);

impl RuleArena {
    pub(crate) fn get(&self, rule_id: RuleId) -> Option<&RuleMode> {
        self.0.get(rule_id.index())
    }
}

impl From<Vec<RuleMode>> for RuleArena {
    fn from(value: Vec<RuleMode>) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone)]
pub(crate) enum RuleMode {
    ConstantRule,
    PropertyRule {
        inner: RuleId,
    },
    LazyRefRule {
        inner: RuleId,
    },
    UnionRule {
        variant_rules: RuleId,
        allow_none_fallback: bool,
    },
    ProtocolRule {
        property_rule: RuleId,
        attribute_rule: RuleId,
        method_rule: RuleId,
    },
    TypedDictRule {
        attribute_rule: RuleId,
    },
    SentinelNoneRule,
    MethodImplRule {
        target_rules: RuleId,
        hook_param_rule: Option<RuleId>,
    },
    AutoMethodRule {
        target_rules: RuleId,
        hook_param_rule: Option<RuleId>,
    },
    AttributeSourceRule {
        inner: RuleId,
    },
    ConstructorRule {
        param_rules: RuleId,
    },
    MatchFirstRule {
        rules: Vec<RuleId>,
    },
}

impl RuleMode {
    fn label(&self) -> &'static str {
        match self {
            RuleMode::ConstantRule => "constant",
            RuleMode::PropertyRule { .. } => "property",
            RuleMode::LazyRefRule { .. } => "lazy_ref",
            RuleMode::UnionRule { .. } => "union",
            RuleMode::ProtocolRule { .. } => "protocol",
            RuleMode::TypedDictRule { .. } => "typed_dict",
            RuleMode::SentinelNoneRule => "sentinel_none",
            RuleMode::MethodImplRule { .. } => "method_impl",
            RuleMode::AutoMethodRule { .. } => "auto_method",
            RuleMode::AttributeSourceRule { .. } => "attribute",
            RuleMode::ConstructorRule { .. } => "constructor",
            RuleMode::MatchFirstRule { .. } => "match_first",
        }
    }
}

#[derive_where(Clone, PartialEq, Eq, Hash)]
pub(crate) struct MethodParam<S: ArenaFamily> {
    pub(crate) name: Arc<str>,
    pub(crate) kind: ParamKind,
    pub(crate) param_type: PyTypeConcreteKey<S>,
    pub(crate) source: Option<Source<S>>,
}

#[derive_where(Clone, PartialEq, Eq, Hash)]
pub(crate) struct TransitionResultBinding<S: ArenaFamily> {
    pub(crate) name: Arc<str>,
    pub(crate) source: Source<S>,
}

#[derive(Error)]
#[derive_where(Clone, PartialEq, Eq, Hash)]
pub(crate) enum ResolutionError<S: ArenaFamily> {
    #[error("invalid rule id")]
    InvalidRuleId(RuleId),
    #[error("no constant found")]
    NoConstantFound(PyTypeConcreteKey<S>),
    #[error("ambiguous constant")]
    AmbiguousConstant(PyTypeConcreteKey<S>),
    #[error("no property found")]
    NoPropertyFound(PyTypeConcreteKey<S>),
    #[error("cycle detected")]
    Cycle(PyTypeConcreteKey<S>),
    #[error("incompatible type")]
    IncompatibleType(PyTypeConcreteKey<S>),
    #[error("missing dependency")]
    MissingDependency(PyTypeConcreteKey<S>, Vec<Arc<ResolutionError<S>>>),
    #[error("no method found")]
    NoMethodFound(PyTypeConcreteKey<S>),
    #[error("ambiguous method")]
    AmbiguousMethod(PyTypeConcreteKey<S>),
    #[error("no attribute found")]
    NoAttributeFound(PyTypeConcreteKey<S>),
    #[error("no constructor found")]
    NoConstructorFound(PyTypeConcreteKey<S>),
    #[error("ambiguous constructor")]
    AmbiguousConstructor(PyTypeConcreteKey<S>),
    #[error("solver fixpoint limit reached")]
    FixpointLimitReached(PyTypeConcreteKey<S>),
    #[error("solver stack overflow depth reached")]
    StackOverflowDepthReached(PyTypeConcreteKey<S>),
    #[error("unexpected same depth cycle escaped to root solve")]
    UnexpectedSameDepthCycle(PyTypeConcreteKey<S>),
    #[error("answer support closure is incomplete")]
    AnswerSupportClosureIncomplete(PyTypeConcreteKey<S>),
    #[error("member error for '{member_name}'")]
    MemberError {
        member_name: Arc<str>,
        cause: Arc<ResolutionError<S>>,
    },
    #[error("rule error in {rule_label}")]
    RuleError {
        rule_label: &'static str,
        cause: Arc<ResolutionError<S>>,
    },
}

impl<S: ArenaFamily> std::fmt::Debug for MethodParam<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MethodParam")
    }
}

impl<S: ArenaFamily> std::fmt::Debug for ResolutionError<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

fn format_qualifier(qualifier: &Qualifier) -> String {
    if qualifier.is_unqualified() {
        String::new()
    } else {
        format!("<{}>", qualifier.display_compact())
    }
}

pub(crate) fn display_concrete_ref<S: ArenaFamily>(
    arenas: &TypeArenas<S>,
    r: PyTypeConcreteKey<S>,
) -> String {
    let qual = arenas
        .qualifier_of_concrete(r)
        .map(format_qualifier)
        .unwrap_or_default();
    match r {
        PyType::Plain(k) => {
            let name = arenas
                .concrete
                .plains
                .get(&k)
                .map(|t| t.inner.descriptor.display_name.clone())
                .unwrap_or_else(|| "<unknown plain>".into());
            format!("{name}{qual}")
        }
        PyType::Protocol(k) => {
            let name = arenas
                .concrete
                .protocols
                .get(&k)
                .map(|t| t.inner.descriptor.display_name.clone())
                .unwrap_or_else(|| "<unknown protocol>".into());
            format!("{name}{qual}")
        }
        PyType::TypedDict(k) => {
            let name = arenas
                .concrete
                .typed_dicts
                .get(&k)
                .map(|t| t.inner.descriptor.display_name.clone())
                .unwrap_or_else(|| "<unknown typed_dict>".into());
            format!("{name}{qual}")
        }
        PyType::Sentinel(k) => arenas
            .sentinels
            .get(&k)
            .map(|s| match s.inner.value {
                SentinelTypeKind::None => "None".into(),
                SentinelTypeKind::Ellipsis => "...".into(),
            })
            .unwrap_or_else(|| "Sentinel".into()),
        PyType::Union(k) => {
            let body = arenas
                .concrete
                .unions
                .get(&k)
                .map(|u| {
                    u.inner
                        .variants
                        .iter()
                        .map(|v| display_concrete_ref(arenas, *v))
                        .collect::<Vec<_>>()
                        .join(" | ")
                })
                .unwrap_or_else(|| "<unknown union>".into());
            if qual.is_empty() {
                body
            } else {
                format!("({body}){qual}")
            }
        }
        PyType::Callable(k) => {
            let body = arenas
                .concrete
                .callables
                .get(&k)
                .map(|c| {
                    let params: Vec<_> = c
                        .inner
                        .params
                        .iter()
                        .map(|(_, &t)| display_concrete_ref(arenas, t))
                        .collect();
                    let ret = display_concrete_ref(arenas, c.inner.return_type);
                    let name = c.inner.function_name.as_deref().unwrap_or("");
                    format!("{name}({}) -> {}", params.join(", "), ret)
                })
                .unwrap_or_else(|| "<unknown callable>".into());
            if qual.is_empty() {
                body
            } else {
                format!("({body}){qual}")
            }
        }
        PyType::LazyRef(k) => {
            let body = arenas
                .concrete
                .lazy_refs
                .get(&k)
                .map(|l| {
                    let target = display_concrete_ref(arenas, l.inner.target);
                    format!("Lazy[{target}]")
                })
                .unwrap_or_else(|| "<unknown lazy_ref>".into());
            if qual.is_empty() {
                body
            } else {
                format!("({body}){qual}")
            }
        }
        PyType::TypeVar(k) => arenas
            .concrete
            .type_vars
            .get(&k)
            .map(|t| format!("~{}", t.inner.descriptor.display_name))
            .unwrap_or_else(|| "<unknown typevar>".into()),
        PyType::ParamSpec(k) => arenas
            .concrete
            .param_specs
            .get(&k)
            .map(|t| format!("**{}", t.inner.descriptor.display_name))
            .unwrap_or_else(|| "<unknown paramspec>".into()),
    }
}

fn format_error_leaf(
    err: &ResolutionError<SlotBackend>,
    arenas: &TypeArenas<SlotBackend>,
) -> String {
    match err {
        ResolutionError::InvalidRuleId(id) => format!("invalid rule id: {id:?}"),
        ResolutionError::NoConstantFound(r) => {
            format!(
                "no constant found for type '{}'",
                display_concrete_ref(arenas, *r)
            )
        }
        ResolutionError::AmbiguousConstant(r) => {
            format!(
                "ambiguous constant for type '{}'",
                display_concrete_ref(arenas, *r)
            )
        }
        ResolutionError::NoPropertyFound(r) => {
            format!(
                "no property source found for type '{}'",
                display_concrete_ref(arenas, *r)
            )
        }
        ResolutionError::Cycle(r) => {
            format!(
                "cycle detected resolving type '{}'",
                display_concrete_ref(arenas, *r)
            )
        }
        ResolutionError::IncompatibleType(r) => {
            format!("incompatible type '{}'", display_concrete_ref(arenas, *r))
        }
        ResolutionError::MissingDependency(r, _) => {
            format!("Missing dependency: {}", display_concrete_ref(arenas, *r))
        }
        ResolutionError::NoMethodFound(r) => {
            format!(
                "no method found for type '{}'",
                display_concrete_ref(arenas, *r)
            )
        }
        ResolutionError::AmbiguousMethod(r) => {
            format!(
                "ambiguous method for type '{}'",
                display_concrete_ref(arenas, *r)
            )
        }
        ResolutionError::NoAttributeFound(r) => {
            format!(
                "no attribute source found for type '{}'",
                display_concrete_ref(arenas, *r)
            )
        }
        ResolutionError::NoConstructorFound(r) => {
            format!(
                "no constructor found for type '{}'",
                display_concrete_ref(arenas, *r)
            )
        }
        ResolutionError::AmbiguousConstructor(r) => {
            format!(
                "ambiguous constructor for type '{}'",
                display_concrete_ref(arenas, *r)
            )
        }
        ResolutionError::FixpointLimitReached(r) => {
            format!(
                "solver fixpoint limit reached resolving type '{}'",
                display_concrete_ref(arenas, *r)
            )
        }
        ResolutionError::StackOverflowDepthReached(r) => {
            format!(
                "solver stack overflow depth reached resolving type '{}'",
                display_concrete_ref(arenas, *r)
            )
        }
        ResolutionError::UnexpectedSameDepthCycle(r) => {
            format!(
                "unexpected same depth cycle escaped to root solve resolving type '{}'",
                display_concrete_ref(arenas, *r)
            )
        }
        ResolutionError::AnswerSupportClosureIncomplete(r) => {
            format!(
                "answer support closure is incomplete resolving type '{}'",
                display_concrete_ref(arenas, *r)
            )
        }
        ResolutionError::MemberError { member_name, cause } => {
            format!(
                "member '{}': {}",
                member_name,
                format_error_leaf(cause.as_ref(), arenas)
            )
        }
        ResolutionError::RuleError { rule_label, cause } => {
            format!(
                "[{rule_label}] {}",
                format_error_leaf(cause.as_ref(), arenas)
            )
        }
    }
}

struct FormatLimits {
    max_depth: usize,
    max_children_per_node: usize,
    max_total_lines: usize,
}

impl FormatLimits {
    fn standard() -> Self {
        Self {
            max_depth: 10,
            max_children_per_node: 10,
            max_total_lines: 200,
        }
    }

    fn unlimited() -> Self {
        Self {
            max_depth: usize::MAX,
            max_children_per_node: usize::MAX,
            max_total_lines: usize::MAX,
        }
    }
}

fn is_leaf_error<S: ArenaFamily>(err: &ResolutionError<S>) -> bool {
    match err {
        ResolutionError::RuleError { cause, .. } => is_leaf_error(cause.as_ref()),
        ResolutionError::NoConstantFound(_)
        | ResolutionError::NoConstructorFound(_)
        | ResolutionError::NoMethodFound(_)
        | ResolutionError::NoPropertyFound(_)
        | ResolutionError::NoAttributeFound(_)
        | ResolutionError::IncompatibleType(_)
        | ResolutionError::InvalidRuleId(_)
        | ResolutionError::Cycle(_)
        | ResolutionError::FixpointLimitReached(_)
        | ResolutionError::StackOverflowDepthReached(_)
        | ResolutionError::UnexpectedSameDepthCycle(_)
        | ResolutionError::AnswerSupportClosureIncomplete(_) => true,
        _ => false,
    }
}

fn format_error_tree(
    err: &ResolutionError<SlotBackend>,
    arenas: &TypeArenas<SlotBackend>,
    depth: usize,
    line_budget: &mut usize,
    limits: &FormatLimits,
) -> String {
    if *line_budget == 0 {
        return String::new();
    }

    match err {
        ResolutionError::MissingDependency(r, causes) if !causes.is_empty() => {
            let header = format!("Missing dependency: {}", display_concrete_ref(arenas, *r));
            *line_budget = line_budget.saturating_sub(1);

            if depth >= limits.max_depth {
                let count = causes.len();
                let noun = if count == 1 { "error" } else { "errors" };
                *line_budget = line_budget.saturating_sub(1);
                return join_tree(&header, &[format!("... ({count} deeper {noun} omitted)")]);
            }

            let mut leaf_count = 0usize;
            let mut substantive: Vec<String> = Vec::new();

            for cause in causes {
                if *line_budget == 0 {
                    break;
                }
                if is_leaf_error(cause.as_ref()) {
                    leaf_count += 1;
                } else if substantive.len() < limits.max_children_per_node {
                    substantive.push(format_error_tree(
                        cause.as_ref(),
                        arenas,
                        depth + 1,
                        line_budget,
                        limits,
                    ));
                }
            }

            let total_substantive = causes.iter().filter(|c| !is_leaf_error(c.as_ref())).count();
            let omitted = total_substantive.saturating_sub(substantive.len());
            if omitted > 0 {
                let noun = if omitted == 1 { "error" } else { "errors" };
                substantive.push(format!("... and {omitted} more {noun}"));
                *line_budget = line_budget.saturating_sub(1);
            }

            if leaf_count > 0 {
                let noun = if leaf_count == 1 { "rule" } else { "rules" };
                substantive.push(format!("({leaf_count} {noun} returned no match)"));
                *line_budget = line_budget.saturating_sub(1);
            }

            substantive.retain(|s| !s.is_empty());

            if substantive.is_empty() {
                header
            } else {
                join_tree(&header, &substantive)
            }
        }
        ResolutionError::MemberError { member_name, cause } => {
            let header = format!("member '{member_name}'");
            *line_budget = line_budget.saturating_sub(1);
            if depth >= limits.max_depth {
                return format!("{header}: (deeper errors omitted)");
            }
            let child = format_error_tree(cause.as_ref(), arenas, depth + 1, line_budget, limits);
            if child.is_empty() {
                header
            } else {
                join_tree(&header, &[child])
            }
        }
        ResolutionError::RuleError { rule_label, cause } => {
            let inner = format_error_tree(cause.as_ref(), arenas, depth, line_budget, limits);
            if inner.is_empty() {
                return String::new();
            }
            let mut lines: Vec<&str> = inner.split('\n').collect();
            let first = format!("[{rule_label}] {}", lines[0]);
            lines[0] = &first;
            lines.join("\n")
        }
        _ => {
            *line_budget = line_budget.saturating_sub(1);
            format_error_leaf(err, arenas)
        }
    }
}

fn join_tree(header: &str, children: &[String]) -> String {
    let mut lines = vec![header.to_string()];
    for (i, child) in children.iter().enumerate() {
        let is_last = i == children.len() - 1;
        let connector = if is_last { "└── " } else { "├── " };
        let continuation = if is_last { "    " } else { "│   " };
        let child_lines: Vec<&str> = child.split('\n').collect();
        if let Some((first, rest)) = child_lines.split_first() {
            lines.push(format!("{connector}{first}"));
            for line in rest {
                lines.push(format!("{continuation}{line}"));
            }
        }
    }
    lines.join("\n")
}

impl ResolutionError<SlotBackend> {
    pub(crate) fn into_py_err(self, arenas: &TypeArenas<SlotBackend>) -> pyo3::PyErr {
        let limits = match std::env::var("DISABLE_ERROR_TRUNCATION").as_deref() {
            Ok("1") => FormatLimits::unlimited(),
            _ => FormatLimits::standard(),
        };
        let mut budget = limits.max_total_lines;
        let msg = format_error_tree(&self, arenas, 0, &mut budget, &limits);
        crate::ResolutionError::new_err(msg)
    }
}

pub(crate) use env::{RegistryEnv, RegistrySharedState};
pub(crate) use rule::{
    RegistryResolutionRule, ResolutionQuery, SolverResolutionArena, SolverResolutionNode,
    SolverResolutionRef, SolverResolvedHook, SolverResolvedNode,
};
