"""Default rule graph shipped with inlay."""

from typing import TypedDict, Unpack

from inlay._native import RuleGraph
from inlay.rules import (
    RuleGraphBuilder,
    attribute_source_rule,
    constant_rule,
    constructor_rule,
    init_rule,
    lazy_ref_rule,
    match_by_type,
    method_impl_rule,
    property_source_rule,
    protocol_rule,
    sentinel_none_rule,
    typeddict_rule,
    union_rule,
)


class DefaultRulesArgs(TypedDict, total=False):
    init_whitelist: tuple[type, ...]
    init_blacklist: tuple[type, ...]


def default_rules(**kwargs: Unpack[DefaultRulesArgs]) -> RuleGraph:
    init_whitelist = kwargs.get('init_whitelist', ())
    init_blacklist = kwargs.get('init_blacklist', ())

    builder = RuleGraphBuilder()

    self_ref = builder.lazy(lambda: pipeline)

    method_rules = method_impl_rule(target_rules=self_ref)
    sentinel = sentinel_none_rule()
    constant = constant_rule()
    lazy_ref = lazy_ref_rule(resolve=self_ref)
    attribute = attribute_source_rule(resolve=self_ref)
    property_ = property_source_rule(resolve=self_ref)
    constructor = constructor_rule(param_rules=self_ref)
    init = init_rule(
        param_rules=self_ref,
        whitelist=init_whitelist,
        blacklist=init_blacklist,
    )
    union = union_rule(variant_rules=self_ref)
    protocol = protocol_rule(resolve=self_ref, method_rules=method_rules)
    typed_dict = typeddict_rule(resolve=self_ref)

    registry_rules = (constant, attribute, property_, constructor)
    pipeline = match_by_type(
        sentinel=(sentinel, *registry_rules),
        param_spec=registry_rules,
        plain=registry_rules,
        class_=(*registry_rules, init),
        protocol=(*registry_rules, protocol),
        typed_dict=(*registry_rules, typed_dict),
        union=(*registry_rules, union),
        callable=(*registry_rules, method_rules),
        lazy_ref=(constant, lazy_ref, attribute, property_, constructor),
        type_var=registry_rules,
    )

    return builder.build()


__all__ = ['DefaultRulesArgs', 'default_rules']
