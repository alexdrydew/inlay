from collections.abc import Callable
from dataclasses import dataclass, field

from inlay._native import RuleGraph

# --- Rule descriptors ---


@dataclass(frozen=True)
class SentinelNoneRule: ...


@dataclass(frozen=True)
class ConstantRule: ...


@dataclass(frozen=True)
class LazyRefRule:
    resolve: Rule


@dataclass(frozen=True)
class PropertyRule:
    inner: Rule


@dataclass(frozen=True)
class AttributeSourceRule:
    inner: Rule


@dataclass(frozen=True)
class ConstructorRule:
    param_rules: Rule


@dataclass(frozen=True)
class InitRule:
    param_rules: Rule
    whitelist: tuple[type, ...] = field(default_factory=tuple)
    blacklist: tuple[type, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class UnionRule:
    variant_rules: Rule


@dataclass(frozen=True)
class ProtocolRule:
    property_rule: Rule
    attribute_rule: Rule
    method_rule: Rule


@dataclass(frozen=True)
class TypedDictRule:
    attribute_rule: Rule


@dataclass(frozen=True)
class MethodImplRule:
    target_rules: Rule


@dataclass(frozen=True)
class BoundedCallableRule:
    target_rules: Rule


@dataclass(frozen=True)
class BoundedUnionRule:
    pointwise_rules: Rule


@dataclass(frozen=True)
class CallableBindingRule:
    target_rules: Rule


@dataclass(frozen=True)
class MatchFirstRule:
    rules: tuple[Rule, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TypeMatchFirstRule:
    sentinel: tuple[Rule, ...] = field(default_factory=tuple)
    param_spec: tuple[Rule, ...] = field(default_factory=tuple)
    plain: tuple[Rule, ...] = field(default_factory=tuple)
    class_: tuple[Rule, ...] = field(default_factory=tuple)
    protocol: tuple[Rule, ...] = field(default_factory=tuple)
    typed_dict: tuple[Rule, ...] = field(default_factory=tuple)
    union: tuple[Rule, ...] = field(default_factory=tuple)
    callable: tuple[Rule, ...] = field(default_factory=tuple)
    callable_binding: tuple[Rule, ...] = field(default_factory=tuple)
    lazy_ref: tuple[Rule, ...] = field(default_factory=tuple)
    type_var: tuple[Rule, ...] = field(default_factory=tuple)
    fallback: tuple[Rule, ...] = field(default_factory=tuple)


class Placeholder:
    rule: Rule | None

    def __init__(self) -> None:
        self.rule = None


type Rule = (
    SentinelNoneRule
    | ConstantRule
    | LazyRefRule
    | PropertyRule
    | AttributeSourceRule
    | ConstructorRule
    | InitRule
    | UnionRule
    | ProtocolRule
    | TypedDictRule
    | MethodImplRule
    | BoundedCallableRule
    | BoundedUnionRule
    | CallableBindingRule
    | MatchFirstRule
    | TypeMatchFirstRule
    | Placeholder
)

# --- Builder ---


class RuleGraphBuilder:
    def __init__(self) -> None:
        self._deferred: list[tuple[Placeholder, Callable[[], Rule]]] = []

    def lazy(self, func: Callable[[], Rule]) -> Placeholder:
        p = Placeholder()
        self._deferred.append((p, func))
        return p

    def build(self) -> RuleGraph:
        for placeholder, func in self._deferred:
            placeholder.rule = func()
        return RuleGraph(root=self._deferred[0][0])
