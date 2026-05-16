"""Sequence provider registration tests."""

from collections.abc import Sequence
from typing import Annotated

from inlay import Registry, RuleGraph, compile, qual


class TestSequenceProviders:
    def test_register_sequence_resolves_registered_items_in_order(
        self, rules: RuleGraph
    ) -> None:
        class Plugin: ...

        class AuthPlugin(Plugin): ...

        class MetricsPlugin(Plugin): ...

        registry = (
            Registry()
            .register(AuthPlugin)(AuthPlugin)
            .register(MetricsPlugin)(MetricsPlugin)
            .register_sequence(Sequence[Plugin], (AuthPlugin, MetricsPlugin))
        )

        result = compile(Sequence[Plugin], registry.build(), rules)

        assert [type(item) for item in result] == [AuthPlugin, MetricsPlugin]

    def test_register_sequence_resolves_duplicate_runtime_params_by_name(
        self, rules: RuleGraph
    ) -> None:
        class Plugin: ...

        registry = Registry().register_sequence(
            Sequence[Plugin],
            (Plugin, Plugin),
        )

        def make_plugins(item_0: Plugin, item_1: Plugin) -> Sequence[Plugin]:
            raise AssertionError((item_0, item_1))

        factory = compile(make_plugins, registry.build(), rules)
        first = Plugin()
        second = Plugin()

        assert factory(first, second) == [first, second]

    def test_register_sequence_resolves_qualified_items(self, rules: RuleGraph) -> None:
        class Service: ...

        class PrimaryService(Service): ...

        class SecondaryService(Service): ...

        registry = (
            Registry()
            .register(Service, qualifiers=qual('primary'))(PrimaryService)
            .register(Service, qualifiers=qual('secondary'))(SecondaryService)
            .register_sequence(
                Sequence[Service],
                (
                    Annotated[Service, qual('primary')],
                    Annotated[Service, qual('secondary')],
                ),
            )
        )

        result = compile(Sequence[Service], registry.build(), rules)

        assert [type(item) for item in result] == [PrimaryService, SecondaryService]
