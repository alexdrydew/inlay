"""Constructor override registration tests."""

import typing

import pytest

from inlay import RegistryBuilder, RuleGraph, compile, normalize, qual


class TestConstructorOverrides:
    def test_override_replaces_regular_registration(self, rules: RuleGraph) -> None:
        class Service: ...

        class DefaultService(Service): ...

        class OverrideService(Service): ...

        registry = (
            RegistryBuilder()
            .register(Service)(DefaultService)
            .override(Service)(OverrideService)
        )

        result = compile(Service, registry.build(), rules)

        assert isinstance(result, OverrideService)

    def test_regular_registration_after_override_does_not_reenable_base(
        self, rules: RuleGraph
    ) -> None:
        class Service: ...

        class DefaultService(Service): ...

        class OverrideService(Service): ...

        registry = (
            RegistryBuilder()
            .override(Service)(OverrideService)
            .register(Service)(DefaultService)
        )

        result = compile(Service, registry.build(), rules)

        assert isinstance(result, OverrideService)

    def test_override_include_order_does_not_matter(self, rules: RuleGraph) -> None:
        class Service: ...

        class DefaultService(Service): ...

        class OverrideService(Service): ...

        base = RegistryBuilder().register(Service)(DefaultService)
        overrides = RegistryBuilder().override(Service)(OverrideService)

        base_then_override = RegistryBuilder().include(base).include(overrides)
        override_then_base = RegistryBuilder().include(overrides).include(base)

        assert isinstance(
            compile(Service, base_then_override.build(), rules), OverrideService
        )
        assert isinstance(
            compile(Service, override_then_base.build(), rules), OverrideService
        )

    def test_standalone_override_is_a_constructor(self, rules: RuleGraph) -> None:
        class Service: ...

        class OverrideService(Service): ...

        registry = RegistryBuilder().override(Service)(OverrideService)

        result = compile(Service, registry.build(), rules)

        assert isinstance(result, OverrideService)

    def test_override_replaces_factory_registration(self, rules: RuleGraph) -> None:
        class Service: ...

        class DefaultService(Service): ...

        class OverrideService(Service): ...

        def provide_default() -> Service:
            return DefaultService()

        registry = (
            RegistryBuilder()
            .register_factory(provide_default)
            .override(Service)(OverrideService)
        )

        result = compile(Service, registry.build(), rules)

        assert isinstance(result, OverrideService)

    def test_override_replaces_value_registration(self, rules: RuleGraph) -> None:
        class Service: ...

        class DefaultService(Service): ...

        class OverrideService(Service): ...

        default = DefaultService()
        override = OverrideService()

        def provide_override() -> Service:
            return override

        registry = (
            RegistryBuilder()
            .register_value(Service)(default)
            .override(Service)(provide_override)
        )

        result = compile(Service, registry.build(), rules)

        assert result is override

    def test_qualified_override_matches_exact_target_key(
        self, rules: RuleGraph
    ) -> None:
        class Service: ...

        class DefaultService(Service): ...

        class GameDefaultService(Service): ...

        class GameOverrideService(Service): ...

        registry = (
            RegistryBuilder()
            .register(Service)(DefaultService)
            .register(Service, qualifiers=qual('game'))(GameDefaultService)
            .override(Service, qualifiers=qual('game'))(GameOverrideService)
            .build()
        )

        unqualified = compile(Service, registry, rules)
        qualified = registry.compile(
            rules, normalize(typing.Annotated[Service, qual('game')])
        )

        assert isinstance(unqualified, DefaultService)
        assert isinstance(qualified, GameOverrideService)

    def test_included_override_uses_include_qualifier(self, rules: RuleGraph) -> None:
        class Service: ...

        class DefaultService(Service): ...

        class OverrideService(Service): ...

        base = RegistryBuilder().register(Service)(DefaultService)
        overrides = RegistryBuilder().override(Service)(OverrideService)
        registry = (
            RegistryBuilder()
            .include(base, qualifiers=qual('game'))
            .include(overrides, qualifiers=qual('game'))
            .build()
        )

        result = registry.compile(
            rules, normalize(typing.Annotated[Service, qual('game')])
        )

        assert isinstance(result, OverrideService)

    def test_multiple_overrides_stay_ambiguous(self, rules: RuleGraph) -> None:
        class Service: ...

        class DefaultService(Service): ...

        class FirstOverride(Service): ...

        class SecondOverride(Service): ...

        registry = (
            RegistryBuilder()
            .register(Service)(DefaultService)
            .override(Service)(FirstOverride)
            .override(Service)(SecondOverride)
        )

        with pytest.raises(Exception) as exc_info:
            _ = compile(Service, registry.build(), rules)

        assert type(exc_info.value).__name__ == 'ResolutionError'
        assert 'ambiguous constructor' in str(exc_info.value).lower()
