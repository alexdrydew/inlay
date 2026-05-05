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
class UnionRule:
    variant_rules: Rule
    allow_none_fallback: bool = True


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
class AutoMethodRule:
    target_rules: Rule


@dataclass(frozen=True)
class MatchFirstRule:
    rules: tuple[Rule, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TypeMatchFirstRule:
    sentinel: tuple[Rule, ...] = field(default_factory=tuple)
    param_spec: tuple[Rule, ...] = field(default_factory=tuple)
    plain: tuple[Rule, ...] = field(default_factory=tuple)
    protocol: tuple[Rule, ...] = field(default_factory=tuple)
    typed_dict: tuple[Rule, ...] = field(default_factory=tuple)
    union: tuple[Rule, ...] = field(default_factory=tuple)
    callable: tuple[Rule, ...] = field(default_factory=tuple)
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
    | UnionRule
    | ProtocolRule
    | TypedDictRule
    | MethodImplRule
    | AutoMethodRule
    | MatchFirstRule
    | TypeMatchFirstRule
    | Placeholder
)


# --- Factory functions ---


def sentinel_none_rule() -> SentinelNoneRule:
    return SentinelNoneRule()


def constant_rule() -> ConstantRule:
    return ConstantRule()


def lazy_ref_rule(*, resolve: Rule) -> LazyRefRule:
    return LazyRefRule(resolve=resolve)


def property_source_rule(*, resolve: Rule) -> PropertyRule:
    return PropertyRule(inner=resolve)


def attribute_source_rule(*, resolve: Rule) -> AttributeSourceRule:
    return AttributeSourceRule(inner=resolve)


def constructor_rule(*, param_rules: Rule) -> ConstructorRule:
    return ConstructorRule(param_rules=param_rules)


def union_rule(*, variant_rules: Rule, allow_none_fallback: bool = True) -> UnionRule:
    return UnionRule(
        variant_rules=variant_rules, allow_none_fallback=allow_none_fallback
    )


def protocol_rule(*, resolve: Rule, method_rules: Rule) -> ProtocolRule:
    return ProtocolRule(
        property_rule=resolve,
        attribute_rule=resolve,
        method_rule=method_rules,
    )


def typeddict_rule(*, resolve: Rule) -> TypedDictRule:
    return TypedDictRule(attribute_rule=resolve)


def method_impl_rule(
    *,
    target_rules: Rule,
) -> MethodImplRule:
    return MethodImplRule(target_rules=target_rules)


def auto_method_rule(
    *,
    target_rules: Rule,
) -> AutoMethodRule:
    return AutoMethodRule(target_rules=target_rules)


def match_first(*rules: Rule) -> MatchFirstRule:
    return MatchFirstRule(rules=rules)


def match_by_type(
    *,
    sentinel: tuple[Rule, ...] = (),
    param_spec: tuple[Rule, ...] = (),
    plain: tuple[Rule, ...] = (),
    protocol: tuple[Rule, ...] = (),
    typed_dict: tuple[Rule, ...] = (),
    union: tuple[Rule, ...] = (),
    callable: tuple[Rule, ...] = (),
    lazy_ref: tuple[Rule, ...] = (),
    type_var: tuple[Rule, ...] = (),
    fallback: tuple[Rule, ...] = (),
) -> TypeMatchFirstRule:
    return TypeMatchFirstRule(
        sentinel=tuple(sentinel),
        param_spec=tuple(param_spec),
        plain=tuple(plain),
        protocol=tuple(protocol),
        typed_dict=tuple(typed_dict),
        union=tuple(union),
        callable=tuple(callable),
        lazy_ref=tuple(lazy_ref),
        type_var=tuple(type_var),
        fallback=tuple(fallback),
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
