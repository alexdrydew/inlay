"""Recursive type compile tests."""

import typing

from inlay import Registry, RuleGraph, compile, normalize

type _RecursiveList = int | list[_RecursiveList]
type _RecursiveTree = dict[str, _RecursiveTree]
type _RecursiveJson = (
    str | int | None | list[_RecursiveJson] | dict[str, _RecursiveJson]
)


class TestRecursiveTypeAlias:
    def test_recursive_union_normalizes(self) -> None:
        """Recursive union type alias produces a valid normalized type."""
        _ = normalize(_RecursiveList)

    def test_recursive_plain_normalizes(self) -> None:
        """Recursive plain type alias produces a valid normalized type."""
        _ = normalize(_RecursiveTree)

    def test_build_registry_with_recursive_type(self) -> None:
        """Registry containing a recursive type alias should build without crashing."""
        registry = Registry().register(_RecursiveList)(lambda: 42)
        _ = registry.build()

    def test_build_registry_with_recursive_mapping_type(self) -> None:
        """Registry containing a recursive mapping type should build without
        crashing.
        """
        registry = Registry().register(_RecursiveTree)(lambda: {})
        _ = registry.build()


class TestRecursiveTypeInProtocolMethod:
    def test_compile_protocol_with_recursive_type_in_method_param(
        self, rules: RuleGraph
    ) -> None:
        """Protocol whose method takes a recursive type alias should not
        segfault during compilation.
        """

        class Sender(typing.Protocol):
            def send(self, payload: _RecursiveJson) -> None: ...

        class SenderImpl:
            def send(self, payload: _RecursiveJson) -> None:
                _ = payload
                pass

        class HasSender(typing.Protocol):
            @property
            def sender(self) -> Sender: ...

        registry = Registry().register(Sender)(SenderImpl)

        ctx = compile(HasSender, registry.build(), rules)
        assert ctx.sender is not None
