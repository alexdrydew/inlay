"""Default parameter handling tests."""

import typing

from inlay import RegistryBuilder, RuleGraph, compile


class TestDefaultParameterSkipping:
    """Constructor params with defaults are skipped when unresolvable."""

    def test_constructor_skips_unresolvable_default_param(
        self, rules: RuleGraph
    ) -> None:
        """A constructor param with a default value is skipped when the
        DI container cannot resolve its type.
        """

        DEFAULT_BATCH = 100

        class Dep:
            def __init__(self) -> None:
                self.value: int = 42

        class Service:
            def __init__(self, dep: Dep, batch_size: int = DEFAULT_BATCH) -> None:
                self.dep: Dep = dep
                self.batch_size: int = batch_size

        class Root(typing.Protocol):
            @property
            def service(self) -> Service: ...

        registry = RegistryBuilder().register(Dep)(Dep).register(Service)(Service)

        result = compile(Root, registry.build(), rules)

        assert result.service.dep.value == 42
        assert result.service.batch_size == DEFAULT_BATCH

    def test_constructor_uses_resolved_default_param_when_available(
        self, rules: RuleGraph
    ) -> None:
        """When a defaulted param CAN be resolved, use the resolved value
        (best-effort behavior).
        """

        class Dep:
            pass

        class Config:
            def __init__(self) -> None:
                self.x: int = 99

        DEFAULT_CONFIG = Config()

        class Service:
            def __init__(self, dep: Dep, config: Config = DEFAULT_CONFIG) -> None:
                self.dep: Dep = dep
                self.config: Config = config

        class Root(typing.Protocol):
            @property
            def service(self) -> Service: ...

        registry = (
            RegistryBuilder()
            .register(Dep)(Dep)
            .register(Config)(Config)
            .register(Service)(Service)
        )

        result = compile(Root, registry.build(), rules)

        # Config was resolved by the DI container (fresh instance), not the default
        assert result.service.config is not DEFAULT_CONFIG
        assert result.service.config.x == 99
