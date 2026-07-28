from typing import cast

import pytest

from inlay import Registry, RuleGraph, normalize
from inlay.rules import (
    ConstantRule,
    ConstructorRule,
    InitRule,
    MatchFirstRule,
    RuleGraphBuilder,
    StaticPolicy,
)


def _build_rules(policy: StaticPolicy) -> RuleGraph:
    builder = RuleGraphBuilder()

    self_ref = builder.lazy(lambda: pipeline)
    pipeline = MatchFirstRule((
        ConstantRule(),
        ConstructorRule(param_rules=self_ref, static_policy=policy),
        InitRule(param_rules=self_ref, static_policy=policy),
    ))

    return builder.build()


class TestStaticPolicy:
    def test_default_policy_is_always(self) -> None:
        assert ConstructorRule(param_rules=ConstantRule()).static_policy == 'always'
        assert InitRule(param_rules=ConstantRule()).static_policy == 'always'

    @pytest.mark.parametrize('policy', ['always', 'never', 'if_static_dependencies'])
    def test_policies_compile_constructor_graphs(self, policy: StaticPolicy) -> None:
        class Config:
            pass

        class MyService:
            def __init__(self, config: Config) -> None:
                self.config: Config = config

        registry = Registry().register(Config)(Config).register(MyService)(MyService)
        native = registry.build(_build_rules(policy))

        result = native.compile(normalize(MyService))

        assert isinstance(result, MyService)
        assert isinstance(result.config, Config)

    def test_unknown_policy_is_rejected(self) -> None:
        builder = RuleGraphBuilder()
        self_ref = builder.lazy(lambda: pipeline)
        pipeline = MatchFirstRule((
            ConstantRule(),
            ConstructorRule(
                param_rules=self_ref,
                static_policy=cast(StaticPolicy, cast(object, 'bogus')),
            ),
        ))

        with pytest.raises(ValueError, match='unknown static_policy'):
            _ = builder.build()

    def test_distinct_policies_produce_distinct_rules(self) -> None:
        class Config:
            pass

        registry = Registry().register(Config)(Config)
        policies: tuple[StaticPolicy, ...] = (
            'always',
            'never',
            'if_static_dependencies',
        )
        for policy in policies:
            native = registry.build(_build_rules(policy))
            assert isinstance(native.compile(normalize(Config)), Config)
