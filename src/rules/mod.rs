pub(crate) mod builder;
mod diagnostic;
mod env;
mod rule;

use std::collections::BTreeSet;
use std::sync::Arc;

use thiserror::Error;

use crate::{
    python_identity::PythonIdentity,
    qualifier::Qualifier,
    registry::Source,
    types::{ParamKind, PyType, PyTypeConcreteKey, SentinelTypeKind, TypeArenas},
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

    pub(crate) fn rules(&self) -> &[RuleMode] {
        &self.0
    }
}

impl From<Vec<RuleMode>> for RuleArena {
    fn from(value: Vec<RuleMode>) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone)]
pub(crate) enum RuleMode {
    Constant,
    Property {
        inner: RuleId,
    },
    LazyRef {
        inner: RuleId,
    },
    Union {
        variant_rules: RuleId,
    },
    Protocol {
        property_rule: RuleId,
        attribute_rule: RuleId,
        method_rule: RuleId,
    },
    TypedDict {
        attribute_rule: RuleId,
    },
    SentinelNone,
    MethodImpl {
        target_rules: RuleId,
    },
    BoundedCallable {
        target_rules: RuleId,
    },
    BoundedUnion {
        pointwise_rules: RuleId,
    },
    AttributeSource {
        inner: RuleId,
    },
    Constructor {
        param_rules: RuleId,
    },
    Init {
        param_rules: RuleId,
        whitelist: BTreeSet<PythonIdentity>,
        blacklist: BTreeSet<PythonIdentity>,
    },
    MatchFirst {
        rules: Vec<RuleId>,
    },
    MatchByType {
        rules: Box<TypeFamilyRules>,
    },
}

#[derive(Debug, Clone, Default)]
pub(crate) struct TypeFamilyRules {
    pub(crate) sentinel: Vec<RuleId>,
    pub(crate) param_spec: Vec<RuleId>,
    pub(crate) plain: Vec<RuleId>,
    pub(crate) class_: Vec<RuleId>,
    pub(crate) protocol: Vec<RuleId>,
    pub(crate) typed_dict: Vec<RuleId>,
    pub(crate) union: Vec<RuleId>,
    pub(crate) callable: Vec<RuleId>,
    pub(crate) lazy_ref: Vec<RuleId>,
    pub(crate) type_var: Vec<RuleId>,
    pub(crate) fallback: Vec<RuleId>,
}

impl RuleMode {
    fn label(&self) -> &'static str {
        match self {
            RuleMode::Constant => "constant",
            RuleMode::Property { .. } => "property",
            RuleMode::LazyRef { .. } => "lazy_ref",
            RuleMode::Union { .. } => "union",
            RuleMode::Protocol { .. } => "protocol",
            RuleMode::TypedDict { .. } => "typed_dict",
            RuleMode::SentinelNone => "sentinel_none",
            RuleMode::MethodImpl { .. } => "method_impl",
            RuleMode::BoundedCallable { .. } => "bounded_callable",
            RuleMode::BoundedUnion { .. } => "bounded_union",
            RuleMode::AttributeSource { .. } => "attribute",
            RuleMode::Constructor { .. } => "constructor",
            RuleMode::Init { .. } => "init",
            RuleMode::MatchFirst { .. } => "match_first",
            RuleMode::MatchByType { .. } => "match_by_type",
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct TransitionParam<'ty> {
    pub(crate) name: Arc<str>,
    pub(crate) kind: ParamKind,
    pub(crate) logical_sources: BTreeSet<Source<'ty>>,
}

#[derive(Error, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ResolutionError<'ty> {
    #[error("invalid rule id")]
    InvalidRuleId(RuleId),
    #[error("no constant found")]
    NoConstantFound(PyTypeConcreteKey<'ty>),
    #[error("ambiguous constant")]
    AmbiguousConstant(PyTypeConcreteKey<'ty>),
    #[error("no bound implementation found")]
    NoBoundImplementationFound(PyTypeConcreteKey<'ty>),
    #[error("ambiguous bound implementation")]
    AmbiguousBoundImplementation(PyTypeConcreteKey<'ty>),
    #[error("unsupported runtime union matcher")]
    UnsupportedRuntimeUnionMatcher(PyTypeConcreteKey<'ty>),
    #[error("no property found")]
    NoPropertyFound(PyTypeConcreteKey<'ty>),
    #[error("ambiguous property")]
    AmbiguousProperty(PyTypeConcreteKey<'ty>),
    #[error("cycle detected")]
    Cycle(PyTypeConcreteKey<'ty>),
    #[error("incompatible type")]
    IncompatibleType(PyTypeConcreteKey<'ty>),
    #[error("missing dependency")]
    MissingDependency(PyTypeConcreteKey<'ty>, Vec<Arc<ResolutionError<'ty>>>),
    #[error("method override in lookup lineage")]
    MethodOverrideInLineage(PyTypeConcreteKey<'ty>),
    #[error("no attribute found")]
    NoAttributeFound(PyTypeConcreteKey<'ty>),
    #[error("ambiguous attribute")]
    AmbiguousAttribute(PyTypeConcreteKey<'ty>),
    #[error("no constructor found")]
    NoConstructorFound(PyTypeConcreteKey<'ty>),
    #[error("ambiguous constructor")]
    AmbiguousConstructor(PyTypeConcreteKey<'ty>),
    #[error("solver fixpoint limit reached")]
    FixpointLimitReached(PyTypeConcreteKey<'ty>),
    #[error("solver stack overflow depth reached")]
    StackOverflowDepthReached(PyTypeConcreteKey<'ty>),
    #[error("unexpected same depth cycle escaped to root solve")]
    UnexpectedSameDepthCycle(PyTypeConcreteKey<'ty>),
    #[error("answer support closure is incomplete")]
    AnswerSupportClosureIncomplete(PyTypeConcreteKey<'ty>),
    #[error("member error for '{member_name}'")]
    MemberError {
        member_name: Arc<str>,
        cause: Arc<ResolutionError<'ty>>,
    },
    #[error("rule error in {rule_label}")]
    RuleError {
        rule_label: &'static str,
        cause: Arc<ResolutionError<'ty>>,
    },
}

impl std::fmt::Debug for TransitionParam<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TransitionParam")
    }
}

impl std::fmt::Debug for ResolutionError<'_> {
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

pub(crate) fn display_concrete_ref<'ty>(
    arenas: &TypeArenas<'ty>,
    r: PyTypeConcreteKey<'ty>,
) -> String {
    let qual = format_qualifier(arenas.qualifier_of_concrete(r));
    match r {
        PyType::Plain(k) => {
            let name = arenas
                .concrete
                .plains
                .get(k)
                .inner
                .descriptor
                .display_name
                .clone();
            format!("{name}{qual}")
        }
        PyType::Class(k) => {
            let name = arenas
                .concrete
                .classes
                .get(k)
                .inner
                .descriptor
                .display_name
                .clone();
            format!("{name}{qual}")
        }
        PyType::Protocol(k) => {
            let name = arenas
                .concrete
                .protocols
                .get(k)
                .inner
                .descriptor
                .display_name
                .clone();
            format!("{name}{qual}")
        }
        PyType::TypedDict(k) => {
            let name = arenas
                .concrete
                .typed_dicts
                .get(k)
                .inner
                .descriptor
                .display_name
                .clone();
            format!("{name}{qual}")
        }
        PyType::Sentinel(k) => match arenas.sentinels.get(k).inner.value {
            SentinelTypeKind::None => "None".into(),
            SentinelTypeKind::Ellipsis => "...".into(),
        },
        PyType::Union(k) => {
            let body = arenas
                .concrete
                .unions
                .get(k)
                .inner
                .variants
                .iter()
                .map(|v| display_concrete_ref(arenas, *v))
                .collect::<Vec<_>>()
                .join(" | ");
            if qual.is_empty() {
                body
            } else {
                format!("({body}){qual}")
            }
        }
        PyType::Callable(k) => {
            let c = arenas.concrete.callables.get(k);
            let params: Vec<_> = c
                .inner
                .params
                .iter()
                .map(|(_, &t)| display_concrete_ref(arenas, t))
                .collect();
            let ret = display_concrete_ref(arenas, c.inner.return_type);
            let name = c.inner.function_name.as_deref().unwrap_or("");
            let body = format!("{name}({}) -> {}", params.join(", "), ret);
            if qual.is_empty() {
                body
            } else {
                format!("({body}){qual}")
            }
        }
        PyType::CallableImplementation(k) => {
            let c = arenas.concrete.callable_implementations.get(k);
            let signature = display_concrete_ref(arenas, c.inner.signature);
            let body = format!("CallableType({signature})");
            if qual.is_empty() {
                body
            } else {
                format!("({body}){qual}")
            }
        }
        PyType::LazyRef(k) => {
            let target =
                display_concrete_ref(arenas, arenas.concrete.lazy_refs.get(k).inner.target);
            let body = format!("Lazy[{target}]");
            if qual.is_empty() {
                body
            } else {
                format!("({body}){qual}")
            }
        }
        PyType::TypeVar(k) => arenas
            .concrete
            .type_vars
            .get(k)
            .inner
            .descriptor
            .display_name
            .to_string(),
        PyType::ParamSpec(k) => arenas
            .concrete
            .param_specs
            .get(k)
            .inner
            .descriptor
            .display_name
            .to_string(),
    }
}

impl<'ty> ResolutionError<'ty> {
    pub(crate) fn into_py_err(self, arenas: &TypeArenas<'ty>) -> pyo3::PyErr {
        let mut msg = String::new();
        diagnostic::write_resolution_error(&mut msg, &self, arenas)
            .expect("writing resolution error to string cannot fail");
        crate::ResolutionError::new_err(msg)
    }
}

pub(crate) use diagnostic::{
    PyStdoutWriter, ResolutionGraphJsonError, write_resolution_graph_json,
};
pub(crate) use env::{BoundImplementation, RegistryEnv, RegistrySharedState};
pub(crate) use rule::{
    RegistryResolutionRule, ResolutionQuery, SolverResolutionArena, SolverResolutionNode,
    SolverResolutionRef, SolverResolvedNode, SolverResolvedTransition,
    SolverResolvedTransitionImplementation, SolverRuntimeUnionBranch,
    SolverTransitionImplementationCallable,
};
