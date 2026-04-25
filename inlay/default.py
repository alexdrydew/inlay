"""Default rule graph shipped with inlay."""

from inlay._native import RuleGraph
from inlay.rules import (
    RuleGraphBuilder,
    attribute_source_rule,
    auto_method_rule,
    constant_rule,
    constructor_rule,
    lazy_ref_rule,
    match_first,
    method_impl_rule,
    property_source_rule,
    protocol_rule,
    sentinel_none_rule,
    typeddict_rule,
    union_rule,
)


def default_rules() -> RuleGraph:
    builder = RuleGraphBuilder()

    self_ref = builder.lazy(lambda: pipeline)
    strict_ref = builder.lazy(lambda: strict_pipeline)

    method_rules = match_first(
        method_impl_rule(target_rules=self_ref, hook_param_rule=self_ref),
        auto_method_rule(target_rules=strict_ref, hook_param_rule=self_ref),
    )

    pipeline = match_first(
        sentinel_none_rule(),
        constant_rule(),
        lazy_ref_rule(resolve=self_ref),
        attribute_source_rule(resolve=self_ref),
        property_source_rule(resolve=self_ref),
        constructor_rule(param_rules=self_ref),
        union_rule(variant_rules=self_ref),
        protocol_rule(resolve=self_ref, method_rules=method_rules),
        typeddict_rule(resolve=self_ref),
        auto_method_rule(target_rules=self_ref),
    )

    strict_pipeline = match_first(
        sentinel_none_rule(),
        constant_rule(),
        lazy_ref_rule(resolve=self_ref),
        attribute_source_rule(resolve=self_ref),
        property_source_rule(resolve=self_ref),
        constructor_rule(param_rules=self_ref),
        union_rule(variant_rules=self_ref, allow_none_fallback=False),
        protocol_rule(resolve=self_ref, method_rules=method_rules),
        typeddict_rule(resolve=self_ref),
        auto_method_rule(target_rules=strict_ref),
    )

    return builder.build()


__all__ = ['default_rules']
