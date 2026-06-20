"""Default rule graph shipped with inlay."""

from typing import TypedDict, Unpack

from inlay._native import RuleGraph
from inlay.rules import (
    AttributeSourceRule,
    BoundedCallableRule,
    BoundedUnionRule,
    ConstantRule,
    ConstructorRule,
    InitRule,
    LazyRefRule,
    MethodImplRule,
    MethodOverrideResolution,
    PropertyRule,
    ProtocolRule,
    Rule,
    RuleGraphBuilder,
    SentinelNoneRule,
    TypedDictRule,
    TypeMatchFirstRule,
    UnionRule,
)


class DefaultRulesArgs(TypedDict, total=False):
    init_whitelist: tuple[type, ...]
    init_blacklist: tuple[type, ...]
    method_override_resolution: MethodOverrideResolution


def default_rules(**kwargs: Unpack[DefaultRulesArgs]) -> RuleGraph:
    init_whitelist = kwargs.get('init_whitelist', ())
    init_blacklist = kwargs.get('init_blacklist', ())
    method_override_resolution: MethodOverrideResolution = kwargs.get(
        'method_override_resolution', 'restrict'
    )

    builder = RuleGraphBuilder()

    pipeline: Rule
    self_ref = builder.lazy(lambda: pipeline)

    method_rules = MethodImplRule(
        target_rules=self_ref,
        override_resolution=method_override_resolution,
    )
    bounded_callable = BoundedCallableRule(target_rules=self_ref)
    bounded_union = BoundedUnionRule(pointwise_rules=self_ref)
    sentinel = SentinelNoneRule()
    constant = ConstantRule()
    lazy_ref = LazyRefRule(resolve=self_ref)
    attribute = AttributeSourceRule(inner=self_ref)
    property_ = PropertyRule(inner=self_ref)
    constructor = ConstructorRule(param_rules=self_ref)
    init = InitRule(
        param_rules=self_ref,
        whitelist=init_whitelist,
        blacklist=init_blacklist,
    )
    union = UnionRule(variant_rules=self_ref)
    protocol = ProtocolRule(
        property_rule=self_ref,
        attribute_rule=self_ref,
        method_rule=method_rules,
    )
    typed_dict = TypedDictRule(attribute_rule=self_ref)

    registry_rules = (constant, attribute, property_, constructor)
    pipeline = TypeMatchFirstRule(
        sentinel=(sentinel, *registry_rules),
        param_spec=registry_rules,
        plain=registry_rules,
        class_=(*registry_rules, init),
        protocol=(*registry_rules, protocol),
        typed_dict=(*registry_rules, typed_dict),
        union=(*registry_rules, bounded_union, union),
        callable=(*registry_rules, bounded_callable, method_rules),
        lazy_ref=(constant, lazy_ref, attribute, property_, constructor),
        type_var=registry_rules,
    )

    return builder.build()


__all__ = ['DefaultRulesArgs', 'default_rules']
