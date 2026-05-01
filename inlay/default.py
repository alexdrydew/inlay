"""Default rule graph shipped with inlay."""

from inlay._native import RuleGraph
from inlay.rules import (
    Rule,
    RuleGraphBuilder,
    attribute_source_rule,
    auto_method_rule,
    constant_rule,
    constructor_rule,
    lazy_ref_rule,
    match_by_type,
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

    def pipeline_for(*, auto_target: Rule, allow_none_fallback: bool = True) -> Rule:
        sentinel = sentinel_none_rule()
        constant = constant_rule()
        lazy_ref = lazy_ref_rule(resolve=self_ref)
        attribute = attribute_source_rule(resolve=self_ref)
        property_ = property_source_rule(resolve=self_ref)
        constructor = constructor_rule(param_rules=self_ref)
        union = union_rule(
            variant_rules=self_ref, allow_none_fallback=allow_none_fallback
        )
        protocol = protocol_rule(resolve=self_ref, method_rules=method_rules)
        typed_dict = typeddict_rule(resolve=self_ref)
        auto_method = auto_method_rule(target_rules=auto_target)

        registry_rules = (constant, attribute, property_, constructor)
        return match_by_type(
            sentinel=(sentinel, *registry_rules),
            param_spec=registry_rules,
            plain=registry_rules,
            protocol=(*registry_rules, protocol),
            typed_dict=(*registry_rules, typed_dict),
            union=(*registry_rules, union),
            callable=(*registry_rules, auto_method),
            lazy_ref=(constant, lazy_ref, attribute, property_, constructor),
            type_var=registry_rules,
        )

    pipeline = pipeline_for(auto_target=self_ref)
    strict_pipeline = pipeline_for(
        auto_target=strict_ref,
        allow_none_fallback=False,
    )

    return builder.build()


__all__ = ['default_rules']
