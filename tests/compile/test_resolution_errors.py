"""Resolution error reporting tests."""

import typing

import pytest

from inlay import RegistryBuilder, RuleGraph, compile


class TestResolutionErrors:
    """Verify that resolution errors carry full context."""

    def test_unresolvable_plain_type_raises_resolution_error(
        self, rules: RuleGraph
    ) -> None:
        """An unregistered plain type produces a ResolutionError (not RuntimeError)."""

        class Missing:
            pass

        registry = RegistryBuilder()

        with pytest.raises(
            Exception, match=r'Missing dependency: .*Missing'
        ) as exc_info:
            compile(Missing, registry.build(), rules)

        assert type(exc_info.value).__name__ == 'ResolutionError'

    def test_error_is_not_runtime_error(self, rules: RuleGraph) -> None:
        """ResolutionError is a distinct exception type, not a generic RuntimeError."""

        class Missing:
            pass

        registry = RegistryBuilder()

        with pytest.raises(Exception) as exc_info:
            compile(Missing, registry.build(), rules)

        assert type(exc_info.value).__name__ == 'ResolutionError'
        assert not isinstance(exc_info.value, RuntimeError)

    def test_no_match_rules_collapsed(self, rules: RuleGraph) -> None:
        """Leaf 'no match' errors are collapsed into a count summary."""

        class Missing:
            pass

        registry = RegistryBuilder()

        with pytest.raises(Exception) as exc_info:
            compile(Missing, registry.build(), rules)

        msg = str(exc_info.value)
        assert 'rules returned no match' in msg

    def test_protocol_member_failure_shows_member_type(self, rules: RuleGraph) -> None:
        """Protocol member failure shows which type failed."""

        class Database:
            pass

        class MyProto(typing.Protocol):
            @property
            def db(self) -> Database: ...

        registry = RegistryBuilder()

        with pytest.raises(Exception) as exc_info:
            compile(MyProto, registry.build(), rules)

        msg = str(exc_info.value)
        # The tree should mention both the protocol and the failing member type
        assert 'MyProto' in msg
        assert 'Database' in msg

    def test_nested_protocol_failure_shows_tree(self, rules: RuleGraph) -> None:
        """A protocol depending on another unresolvable protocol shows a nested tree."""

        class Inner(typing.Protocol):
            @property
            def value(self) -> int: ...

        class Outer(typing.Protocol):
            @property
            def inner(self) -> Inner: ...

        registry = RegistryBuilder()

        with pytest.raises(Exception) as exc_info:
            compile(Outer, registry.build(), rules)

        msg = str(exc_info.value)
        assert 'Outer' in msg
        assert 'Inner' in msg
        # Tree structure uses box-drawing characters
        assert '├── ' in msg or '└── ' in msg

    def test_constructor_missing_param_shows_context(self, rules: RuleGraph) -> None:
        """Constructor param failure shows the param type."""

        class Dep:
            pass

        class Service:
            def __init__(self, dep: Dep) -> None:
                self.dep = dep

        registry = RegistryBuilder().register(Service)(Service)

        with pytest.raises(Exception) as exc_info:
            compile(Service, registry.build(), rules)

        msg = str(exc_info.value)
        assert 'Dep' in msg

    def test_callable_type_displayed_with_signature(self, rules: RuleGraph) -> None:
        """Callable types in errors show their parameter and return types."""

        class MyProto(typing.Protocol):
            def process(self, x: int) -> str: ...

        registry = RegistryBuilder()

        with pytest.raises(Exception) as exc_info:
            compile(MyProto, registry.build(), rules)

        msg = str(exc_info.value)
        # The callable type should show its signature, not just "Callable[...]"
        assert 'int' in msg
        assert 'str' in msg
