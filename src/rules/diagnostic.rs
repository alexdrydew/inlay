use std::collections::BTreeSet;
use std::fmt;
use std::io::{self, Write as IoWrite};
use std::sync::Arc;

use context_solver::Arena as ResultsArena;
use pyo3::types::PyAnyMethods;
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Serialize, Serializer};

use crate::registry::{Source, SourceKind};
use crate::rules::{
    ResolutionError, SolverResolutionArena, SolverResolutionNode, SolverResolutionRef,
    SolverResolvedNode, SolverTransitionImplementationCallable, display_concrete_ref,
};
use crate::types::{MemberAccessKind, ParamKind, TypeArenas, WrapperKind};

struct FormatLimits {
    max_depth: usize,
    max_children_per_node: usize,
    max_total_lines: usize,
}

impl FormatLimits {
    fn standard() -> Self {
        Self {
            max_depth: 30,
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

    fn from_env() -> Self {
        match std::env::var("DISABLE_ERROR_TRUNCATION").as_deref() {
            Ok("1") => Self::unlimited(),
            _ => Self::standard(),
        }
    }
}

pub(crate) fn write_resolution_error<'ty, W: fmt::Write>(
    out: &mut W,
    err: &ResolutionError<'ty>,
    arenas: &TypeArenas<'ty>,
) -> fmt::Result {
    let limits = FormatLimits::from_env();
    let mut budget = limits.max_total_lines;
    let mut writer = ErrorTextWriter::new(out);
    write_error_tree(
        &mut writer,
        err,
        arenas,
        0,
        &mut budget,
        &limits,
        LineContext::root(),
    )
}

struct ErrorTextWriter<'a, W: fmt::Write> {
    out: &'a mut W,
    wrote_line: bool,
}

impl<'a, W: fmt::Write> ErrorTextWriter<'a, W> {
    fn new(out: &'a mut W) -> Self {
        Self {
            out,
            wrote_line: false,
        }
    }

    fn write_line(
        &mut self,
        context: &LineContext,
        write_body: impl FnOnce(&mut W) -> fmt::Result,
    ) -> fmt::Result {
        if self.wrote_line {
            self.out.write_char('\n')?;
        }
        self.wrote_line = true;
        self.out.write_str(&context.prefix)?;
        self.out.write_str(context.connector)?;
        self.out.write_str(&context.first_line_prefix)?;
        write_body(self.out)
    }
}

struct LineContext {
    prefix: String,
    connector: &'static str,
    first_line_prefix: String,
}

impl LineContext {
    fn root() -> Self {
        Self {
            prefix: String::new(),
            connector: "",
            first_line_prefix: String::new(),
        }
    }

    fn child(&self, is_last: bool) -> Self {
        let mut prefix = self.prefix.clone();
        match self.connector {
            "├── " => prefix.push_str("│   "),
            "└── " => prefix.push_str("    "),
            _ => {}
        }
        Self {
            prefix,
            connector: if is_last { "└── " } else { "├── " },
            first_line_prefix: String::new(),
        }
    }

    fn with_rule_prefix(&self, rule_label: &str) -> Self {
        let mut first_line_prefix = self.first_line_prefix.clone();
        first_line_prefix.push('[');
        first_line_prefix.push_str(rule_label);
        first_line_prefix.push_str("] ");
        Self {
            prefix: self.prefix.clone(),
            connector: self.connector,
            first_line_prefix,
        }
    }
}

fn write_error_leaf<'ty, W: fmt::Write>(
    out: &mut W,
    err: &ResolutionError<'ty>,
    arenas: &TypeArenas<'ty>,
) -> fmt::Result {
    match err {
        ResolutionError::InvalidRuleId(id) => write!(out, "invalid rule id: {id:?}"),
        ResolutionError::NoConstantFound(r) => write!(
            out,
            "no constant found for type '{}'",
            display_concrete_ref(arenas, *r)
        ),
        ResolutionError::AmbiguousConstant(r) => write!(
            out,
            "ambiguous constant for type '{}'",
            display_concrete_ref(arenas, *r)
        ),
        ResolutionError::NoBoundImplementationFound(r) => write!(
            out,
            "no bound implementation found for type '{}'",
            display_concrete_ref(arenas, *r)
        ),
        ResolutionError::AmbiguousBoundImplementation(r) => write!(
            out,
            "ambiguous bound implementation for type '{}'",
            display_concrete_ref(arenas, *r)
        ),
        ResolutionError::UnsupportedRuntimeUnionMatcher(r) => write!(
            out,
            "unsupported runtime union matcher for type '{}'",
            display_concrete_ref(arenas, *r)
        ),
        ResolutionError::NoPropertyFound(r) => write!(
            out,
            "no property source found for type '{}'",
            display_concrete_ref(arenas, *r)
        ),
        ResolutionError::AmbiguousProperty(r) => write!(
            out,
            "ambiguous property source for type '{}'",
            display_concrete_ref(arenas, *r)
        ),
        ResolutionError::Cycle(r) => write!(
            out,
            "cycle detected resolving type '{}'",
            display_concrete_ref(arenas, *r)
        ),
        ResolutionError::IncompatibleType(r) => {
            write!(
                out,
                "incompatible type '{}'",
                display_concrete_ref(arenas, *r)
            )
        }
        ResolutionError::MissingDependency(r, _) => {
            write!(
                out,
                "Missing dependency: {}",
                display_concrete_ref(arenas, *r)
            )
        }
        ResolutionError::MethodOverrideInLineage(r) => write!(
            out,
            "method override in lookup lineage for type '{}'",
            display_concrete_ref(arenas, *r)
        ),
        ResolutionError::NoAttributeFound(r) => write!(
            out,
            "no attribute source found for type '{}'",
            display_concrete_ref(arenas, *r)
        ),
        ResolutionError::AmbiguousAttribute(r) => write!(
            out,
            "ambiguous attribute source for type '{}'",
            display_concrete_ref(arenas, *r)
        ),
        ResolutionError::NoConstructorFound(r) => write!(
            out,
            "no constructor found for type '{}'",
            display_concrete_ref(arenas, *r)
        ),
        ResolutionError::AmbiguousConstructor(r) => write!(
            out,
            "ambiguous constructor for type '{}'",
            display_concrete_ref(arenas, *r)
        ),
        ResolutionError::FixpointLimitReached(r) => write!(
            out,
            "solver fixpoint limit reached resolving type '{}'",
            display_concrete_ref(arenas, *r)
        ),
        ResolutionError::StackOverflowDepthReached(r) => write!(
            out,
            "solver stack overflow depth reached resolving type '{}'",
            display_concrete_ref(arenas, *r)
        ),
        ResolutionError::UnexpectedSameDepthCycle(r) => write!(
            out,
            "unexpected same depth cycle escaped to root solve resolving type '{}'",
            display_concrete_ref(arenas, *r)
        ),
        ResolutionError::AnswerSupportClosureIncomplete(r) => write!(
            out,
            "answer support closure is incomplete resolving type '{}'",
            display_concrete_ref(arenas, *r)
        ),
        ResolutionError::MemberError { member_name, cause } => {
            write!(out, "member '{member_name}': ")?;
            write_error_leaf(out, cause.as_ref(), arenas)
        }
        ResolutionError::RuleError { rule_label, cause } => {
            write!(out, "[{rule_label}] ")?;
            write_error_leaf(out, cause.as_ref(), arenas)
        }
    }
}

fn is_leaf_error(err: &ResolutionError<'_>) -> bool {
    match err {
        ResolutionError::RuleError { cause, .. } => is_leaf_error(cause.as_ref()),
        ResolutionError::NoConstantFound(_)
        | ResolutionError::NoConstructorFound(_)
        | ResolutionError::NoPropertyFound(_)
        | ResolutionError::NoAttributeFound(_)
        | ResolutionError::IncompatibleType(_)
        | ResolutionError::MethodOverrideInLineage(_)
        | ResolutionError::InvalidRuleId(_)
        | ResolutionError::Cycle(_)
        | ResolutionError::FixpointLimitReached(_)
        | ResolutionError::StackOverflowDepthReached(_)
        | ResolutionError::UnexpectedSameDepthCycle(_)
        | ResolutionError::AnswerSupportClosureIncomplete(_) => true,
        _ => false,
    }
}

enum ErrorChild<'a, 'ty> {
    Error(&'a ResolutionError<'ty>),
    Summary(String),
}

fn write_error_tree<'ty, W: fmt::Write>(
    writer: &mut ErrorTextWriter<'_, W>,
    err: &ResolutionError<'ty>,
    arenas: &TypeArenas<'ty>,
    depth: usize,
    line_budget: &mut usize,
    limits: &FormatLimits,
    context: LineContext,
) -> fmt::Result {
    if *line_budget == 0 {
        return Ok(());
    }

    match err {
        ResolutionError::MissingDependency(r, causes) if !causes.is_empty() => {
            writer.write_line(&context, |out| {
                write!(
                    out,
                    "Missing dependency: {}",
                    display_concrete_ref(arenas, *r)
                )
            })?;
            *line_budget = line_budget.saturating_sub(1);

            if depth >= limits.max_depth {
                let count = causes.len();
                let noun = if count == 1 { "error" } else { "errors" };
                let child_context = context.child(true);
                writer.write_line(&child_context, |out| {
                    write!(out, "... ({count} deeper {noun} omitted)")
                })?;
                *line_budget = line_budget.saturating_sub(1);
                return Ok(());
            }

            let mut leaf_count = 0usize;
            let mut substantive = Vec::new();

            for cause in causes {
                if is_leaf_error(cause.as_ref()) {
                    leaf_count += 1;
                } else if substantive.len() < limits.max_children_per_node {
                    substantive.push(ErrorChild::Error(cause.as_ref()));
                }
            }

            let total_substantive = causes.iter().filter(|c| !is_leaf_error(c.as_ref())).count();
            let omitted = total_substantive.saturating_sub(substantive.len());
            if omitted > 0 {
                let noun = if omitted == 1 { "error" } else { "errors" };
                substantive.push(ErrorChild::Summary(format!(
                    "... and {omitted} more {noun}"
                )));
            }

            if leaf_count > 0 {
                let noun = if leaf_count == 1 { "rule" } else { "rules" };
                substantive.push(ErrorChild::Summary(format!(
                    "({leaf_count} {noun} returned no match)"
                )));
            }

            let last_index = substantive.len().saturating_sub(1);
            for (index, child) in substantive.into_iter().enumerate() {
                if *line_budget == 0 {
                    break;
                }
                let child_context = context.child(index == last_index);
                match child {
                    ErrorChild::Error(child) => write_error_tree(
                        writer,
                        child,
                        arenas,
                        depth + 1,
                        line_budget,
                        limits,
                        child_context,
                    )?,
                    ErrorChild::Summary(summary) => {
                        writer.write_line(&child_context, |out| out.write_str(&summary))?;
                        *line_budget = line_budget.saturating_sub(1);
                    }
                }
            }
            Ok(())
        }
        ResolutionError::MemberError { member_name, cause } => {
            writer.write_line(&context, |out| write!(out, "member '{member_name}'"))?;
            *line_budget = line_budget.saturating_sub(1);
            if depth >= limits.max_depth {
                writer.out.write_str(": (deeper errors omitted)")?;
                return Ok(());
            }
            if *line_budget > 0 {
                let child_context = context.child(true);
                write_error_tree(
                    writer,
                    cause.as_ref(),
                    arenas,
                    depth + 1,
                    line_budget,
                    limits,
                    child_context,
                )?;
            }
            Ok(())
        }
        ResolutionError::RuleError { rule_label, cause } => write_error_tree(
            writer,
            cause.as_ref(),
            arenas,
            depth,
            line_budget,
            limits,
            context.with_rule_prefix(rule_label),
        ),
        _ => {
            writer.write_line(&context, |out| write_error_leaf(out, err, arenas))?;
            *line_budget = line_budget.saturating_sub(1);
            Ok(())
        }
    }
}

pub(crate) fn write_resolution_graph_json<'ty, W: IoWrite>(
    mut writer: W,
    results: &SolverResolutionArena<'ty>,
    root: SolverResolutionRef,
    arenas: &TypeArenas<'ty>,
) -> Result<(), ResolutionGraphJsonError<'ty>> {
    let root = resolved_root_ref(results, root)?;
    let mut stack = vec![root];
    let mut visited = BTreeSet::new();
    let mut first_node = true;

    writer.write_all(b"{\"root\":")?;
    serde_json::to_writer(&mut writer, &root.index())?;
    writer.write_all(b",\"nodes\":[")?;

    while let Some(node_ref) = stack.pop() {
        if !visited.insert(node_ref) {
            continue;
        }

        let node = get_resolved_node(results, node_ref)?;
        let mut references = Vec::new();
        collect_resolution_refs(&node.resolution, &mut references);
        stack.extend(references.into_iter().rev());

        if first_node {
            first_node = false;
        } else {
            writer.write_all(b",")?;
        }
        serde_json::to_writer(
            &mut writer,
            &NodeJson {
                node_ref,
                node,
                arenas,
            },
        )?;
    }

    writer.write_all(b"]}")?;
    Ok(())
}

fn resolved_root_ref<'ty>(
    results: &SolverResolutionArena<'ty>,
    mut root: SolverResolutionRef,
) -> Result<SolverResolutionRef, ResolutionError<'ty>> {
    let mut seen = BTreeSet::new();
    while seen.insert(root) {
        let node = get_resolved_node(results, root)?;
        match node.resolution {
            SolverResolutionNode::Delegate(target)
            | SolverResolutionNode::UnionVariant { target } => {
                root = target;
            }
            _ => return Ok(root),
        }
    }
    Ok(root)
}

#[derive(Debug)]
pub(crate) enum ResolutionGraphJsonError<'ty> {
    Resolution(ResolutionError<'ty>),
    Json(serde_json::Error),
    Io(io::Error),
}

impl<'ty> From<ResolutionError<'ty>> for ResolutionGraphJsonError<'ty> {
    fn from(value: ResolutionError<'ty>) -> Self {
        Self::Resolution(value)
    }
}

impl From<serde_json::Error> for ResolutionGraphJsonError<'_> {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<io::Error> for ResolutionGraphJsonError<'_> {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

struct NodeJson<'a, 'ty> {
    node_ref: SolverResolutionRef,
    node: &'a SolverResolvedNode<'ty>,
    arenas: &'a TypeArenas<'ty>,
}

impl Serialize for NodeJson<'_, '_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(3))?;
        map.serialize_entry("id", &self.node_ref.index())?;
        map.serialize_entry(
            "target",
            &display_concrete_ref(self.arenas, self.node.target_type),
        )?;
        map.serialize_entry(
            "resolution",
            &ResolutionJson {
                resolution: &self.node.resolution,
                arenas: self.arenas,
            },
        )?;
        map.end()
    }
}

struct ResolutionJson<'a, 'ty> {
    resolution: &'a SolverResolutionNode<'ty>,
    arenas: &'a TypeArenas<'ty>,
}

impl Serialize for ResolutionJson<'_, '_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self.resolution {
            SolverResolutionNode::Constant { source } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("kind", "constant")?;
                map.serialize_entry(
                    "source",
                    &SourceJson {
                        source,
                        arenas: self.arenas,
                    },
                )?;
                map.end()
            }
            SolverResolutionNode::Property {
                source,
                property_name,
            } => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("kind", "property")?;
                map.serialize_entry("source", &source.index())?;
                map.serialize_entry("property_name", property_name.as_ref())?;
                map.end()
            }
            SolverResolutionNode::LazyRef { target } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("kind", "lazy_ref")?;
                map.serialize_entry("target", &target.index())?;
                map.end()
            }
            SolverResolutionNode::None => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("kind", "none")?;
                map.end()
            }
            SolverResolutionNode::UnionVariant { target } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("kind", "union_variant")?;
                map.serialize_entry("target", &target.index())?;
                map.end()
            }
            SolverResolutionNode::RuntimeUnionDispatch { source, branches } => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("kind", "runtime_union_dispatch")?;
                map.serialize_entry(
                    "source",
                    &SourceJson {
                        source,
                        arenas: self.arenas,
                    },
                )?;
                map.serialize_entry(
                    "branches",
                    &RuntimeUnionBranchesJson {
                        branches,
                        arenas: self.arenas,
                    },
                )?;
                map.end()
            }
            SolverResolutionNode::Protocol { members } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("kind", "protocol")?;
                map.serialize_entry("members", &MembersJson { members })?;
                map.end()
            }
            SolverResolutionNode::TypedDict { members } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("kind", "typed_dict")?;
                map.serialize_entry("members", &MembersJson { members })?;
                map.end()
            }
            SolverResolutionNode::Transition(transition) => {
                let mut map = serializer.serialize_map(Some(7))?;
                map.serialize_entry("kind", "transition")?;
                map.serialize_entry(
                    "return_wrapper",
                    wrapper_kind_label(transition.return_wrapper),
                )?;
                map.serialize_entry("accepts_varargs", &transition.accepts_varargs)?;
                map.serialize_entry("accepts_varkw", &transition.accepts_varkw)?;
                map.serialize_entry(
                    "params",
                    &TransitionParamsJson {
                        params: &transition.params,
                        arenas: self.arenas,
                    },
                )?;
                map.serialize_entry(
                    "implementations",
                    &TransitionImplementationsJson {
                        implementations: &transition.implementations,
                        arenas: self.arenas,
                    },
                )?;
                map.serialize_entry("target", &transition.target.index())?;
                map.end()
            }
            SolverResolutionNode::Attribute {
                source,
                attribute_name,
                access_kind,
            } => {
                let mut map = serializer.serialize_map(Some(4))?;
                map.serialize_entry("kind", "attribute")?;
                map.serialize_entry("source", &source.index())?;
                map.serialize_entry("attribute_name", attribute_name.as_ref())?;
                map.serialize_entry("access_kind", member_access_kind_label(*access_kind))?;
                map.end()
            }
            SolverResolutionNode::Constructor { params, .. } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("kind", "constructor")?;
                map.serialize_entry("params", &RefParamsJson { params })?;
                map.end()
            }
            SolverResolutionNode::Init { params, .. } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("kind", "init")?;
                map.serialize_entry("params", &RefParamsJson { params })?;
                map.end()
            }
            SolverResolutionNode::Delegate(target) => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("kind", "delegate")?;
                map.serialize_entry("target", &target.index())?;
                map.end()
            }
        }
    }
}

struct SourceJson<'a, 'ty> {
    source: &'a Source<'ty>,
    arenas: &'a TypeArenas<'ty>,
}

impl Serialize for SourceJson<'_, '_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match &self.source.kind {
            SourceKind::ProviderResult(_) => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("kind", "provider_result")?;
                map.end()
            }
            SourceKind::Transition { name, type_ref } => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("kind", "transition")?;
                map.serialize_entry("name", &name.as_ref().map(Arc::as_ref))?;
                map.serialize_entry("type", &display_concrete_ref(self.arenas, *type_ref))?;
                map.end()
            }
        }
    }
}

struct RuntimeUnionBranchesJson<'a, 'ty> {
    branches: &'a [crate::rules::SolverRuntimeUnionBranch<'ty>],
    arenas: &'a TypeArenas<'ty>,
}

impl Serialize for RuntimeUnionBranchesJson<'_, '_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.branches.len()))?;
        for branch in self.branches {
            seq.serialize_element(&RuntimeUnionBranchJson {
                branch,
                arenas: self.arenas,
            })?;
        }
        seq.end()
    }
}

struct RuntimeUnionBranchJson<'a, 'ty> {
    branch: &'a crate::rules::SolverRuntimeUnionBranch<'ty>,
    arenas: &'a TypeArenas<'ty>,
}

impl Serialize for RuntimeUnionBranchJson<'_, '_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(3))?;
        map.serialize_entry(
            "implementation_variant",
            &display_concrete_ref(self.arenas, self.branch.implementation_variant),
        )?;
        map.serialize_entry("target", &self.branch.target.index())?;
        map.serialize_entry(
            "arm_source",
            &SourceJson {
                source: &self.branch.arm_source,
                arenas: self.arenas,
            },
        )?;
        map.end()
    }
}

struct MembersJson<'a> {
    members: &'a std::collections::BTreeMap<Arc<str>, SolverResolutionRef>,
}

impl Serialize for MembersJson<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.members.len()))?;
        for (name, node_ref) in self.members {
            seq.serialize_element(&NamedRefJson {
                name,
                node_ref: *node_ref,
            })?;
        }
        seq.end()
    }
}

struct NamedRefJson<'a> {
    name: &'a str,
    node_ref: SolverResolutionRef,
}

impl Serialize for NamedRefJson<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("name", self.name)?;
        map.serialize_entry("node", &self.node_ref.index())?;
        map.end()
    }
}

struct TransitionParamsJson<'a, 'ty> {
    params: &'a [crate::rules::TransitionParam<'ty>],
    arenas: &'a TypeArenas<'ty>,
}

impl Serialize for TransitionParamsJson<'_, '_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.params.len()))?;
        for param in self.params {
            seq.serialize_element(&TransitionParamJson {
                param,
                arenas: self.arenas,
            })?;
        }
        seq.end()
    }
}

struct TransitionParamJson<'a, 'ty> {
    param: &'a crate::rules::TransitionParam<'ty>,
    arenas: &'a TypeArenas<'ty>,
}

impl Serialize for TransitionParamJson<'_, '_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(3))?;
        map.serialize_entry("name", self.param.name.as_ref())?;
        map.serialize_entry("kind", param_kind_label(self.param.kind))?;
        map.serialize_entry(
            "logical_sources",
            &SourcesJson {
                sources: &self.param.logical_sources,
                arenas: self.arenas,
            },
        )?;
        map.end()
    }
}

struct SourcesJson<'a, 'ty> {
    sources: &'a BTreeSet<Source<'ty>>,
    arenas: &'a TypeArenas<'ty>,
}

impl Serialize for SourcesJson<'_, '_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.sources.len()))?;
        for source in self.sources {
            seq.serialize_element(&SourceJson {
                source,
                arenas: self.arenas,
            })?;
        }
        seq.end()
    }
}

struct TransitionImplementationsJson<'a, 'ty> {
    implementations: &'a [crate::rules::SolverResolvedTransitionImplementation<'ty>],
    arenas: &'a TypeArenas<'ty>,
}

impl Serialize for TransitionImplementationsJson<'_, '_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.implementations.len()))?;
        for implementation in self.implementations {
            seq.serialize_element(&TransitionImplementationJson {
                implementation,
                arenas: self.arenas,
            })?;
        }
        seq.end()
    }
}

struct TransitionImplementationJson<'a, 'ty> {
    implementation: &'a crate::rules::SolverResolvedTransitionImplementation<'ty>,
    arenas: &'a TypeArenas<'ty>,
}

impl Serialize for TransitionImplementationJson<'_, '_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(5))?;
        map.serialize_entry(
            "implementation",
            &ImplementationCallableJson {
                callable: &self.implementation.implementation,
                arenas: self.arenas,
            },
        )?;
        map.serialize_entry(
            "bound_to",
            &self.implementation.bound_to.map(SolverResolutionRef::index),
        )?;
        map.serialize_entry(
            "params",
            &RefParamsJson {
                params: &self.implementation.params,
            },
        )?;
        map.serialize_entry(
            "return_wrapper",
            wrapper_kind_label(self.implementation.return_wrapper),
        )?;
        map.serialize_entry(
            "result_source",
            &OptionalSourceJson {
                source: self.implementation.result_source.as_ref(),
                arenas: self.arenas,
            },
        )?;
        map.end()
    }
}

struct ImplementationCallableJson<'a, 'ty> {
    callable: &'a SolverTransitionImplementationCallable<'ty>,
    arenas: &'a TypeArenas<'ty>,
}

impl Serialize for ImplementationCallableJson<'_, '_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self.callable {
            SolverTransitionImplementationCallable::Static(_) => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("kind", "static")?;
                map.end()
            }
            SolverTransitionImplementationCallable::Source(source) => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("kind", "source")?;
                map.serialize_entry(
                    "source",
                    &SourceJson {
                        source,
                        arenas: self.arenas,
                    },
                )?;
                map.end()
            }
        }
    }
}

struct OptionalSourceJson<'a, 'ty> {
    source: Option<&'a Source<'ty>>,
    arenas: &'a TypeArenas<'ty>,
}

impl Serialize for OptionalSourceJson<'_, '_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self.source {
            Some(source) => SourceJson {
                source,
                arenas: self.arenas,
            }
            .serialize(serializer),
            None => serializer.serialize_none(),
        }
    }
}

struct RefParamsJson<'a> {
    params: &'a [(SolverResolutionRef, Arc<str>, ParamKind)],
}

impl Serialize for RefParamsJson<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.params.len()))?;
        for (node_ref, name, kind) in self.params {
            seq.serialize_element(&RefParamJson {
                node_ref: *node_ref,
                name,
                kind: *kind,
            })?;
        }
        seq.end()
    }
}

struct RefParamJson<'a> {
    node_ref: SolverResolutionRef,
    name: &'a str,
    kind: ParamKind,
}

impl Serialize for RefParamJson<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(3))?;
        map.serialize_entry("name", self.name)?;
        map.serialize_entry("kind", param_kind_label(self.kind))?;
        map.serialize_entry("node", &self.node_ref.index())?;
        map.end()
    }
}

fn get_resolved_node<'a, 'ty>(
    results: &'a SolverResolutionArena<'ty>,
    node_ref: SolverResolutionRef,
) -> Result<&'a SolverResolvedNode<'ty>, ResolutionError<'ty>> {
    match results
        .get(&node_ref)
        .expect("solver result ref must point to a stored result")
    {
        Ok(node) => Ok(node),
        Err(err) => Err(err.clone()),
    }
}

fn collect_resolution_refs<'ty>(
    resolution: &SolverResolutionNode<'ty>,
    refs: &mut Vec<SolverResolutionRef>,
) {
    match resolution {
        SolverResolutionNode::Constant { .. } | SolverResolutionNode::None => {}
        SolverResolutionNode::Property { source, .. }
        | SolverResolutionNode::LazyRef { target: source }
        | SolverResolutionNode::UnionVariant { target: source }
        | SolverResolutionNode::Attribute { source, .. }
        | SolverResolutionNode::Delegate(source) => refs.push(*source),
        SolverResolutionNode::RuntimeUnionDispatch { branches, .. } => {
            refs.extend(branches.iter().map(|branch| branch.target));
        }
        SolverResolutionNode::Protocol { members }
        | SolverResolutionNode::TypedDict { members } => {
            refs.extend(members.values().copied());
        }
        SolverResolutionNode::Transition(transition) => {
            for implementation in &transition.implementations {
                if let Some(bound_to) = implementation.bound_to {
                    refs.push(bound_to);
                }
                refs.extend(
                    implementation
                        .params
                        .iter()
                        .map(|(node_ref, _, _)| *node_ref),
                );
            }
            refs.push(transition.target);
        }
        SolverResolutionNode::Constructor { params, .. }
        | SolverResolutionNode::Init { params, .. } => {
            refs.extend(params.iter().map(|(node_ref, _, _)| *node_ref));
        }
    }
}

fn param_kind_label(kind: ParamKind) -> &'static str {
    match kind {
        ParamKind::PositionalOnly => "positional_only",
        ParamKind::PositionalOrKeyword => "positional_or_keyword",
        ParamKind::KeywordOnly => "keyword_only",
    }
}

fn member_access_kind_label(kind: MemberAccessKind) -> &'static str {
    match kind {
        MemberAccessKind::Attribute => "attribute",
        MemberAccessKind::DictItem => "dict_item",
    }
}

fn wrapper_kind_label(kind: WrapperKind) -> &'static str {
    match kind {
        WrapperKind::None => "none",
        WrapperKind::Awaitable => "awaitable",
        WrapperKind::ContextManager => "context_manager",
        WrapperKind::AsyncContextManager => "async_context_manager",
    }
}

pub(crate) struct PyStdoutWriter<'py> {
    stdout: pyo3::Bound<'py, pyo3::PyAny>,
    buffer: Vec<u8>,
}

impl<'py> PyStdoutWriter<'py> {
    pub(crate) fn new(py: pyo3::Python<'py>) -> pyo3::PyResult<Self> {
        let stdout = py.import("sys")?.getattr("stdout")?;
        Ok(Self {
            stdout,
            buffer: Vec::with_capacity(8192),
        })
    }
}

impl IoWrite for PyStdoutWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        if self.buffer.len() >= 8192 {
            self.flush()?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let text = std::str::from_utf8(&self.buffer)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        self.stdout
            .call_method1("write", (text,))
            .map_err(|error| io::Error::other(error.to_string()))?;
        self.buffer.clear();
        Ok(())
    }
}
