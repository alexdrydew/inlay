"""Resolution error reporting tests."""

import os
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
            _ = compile(Missing, registry.build(), rules)

        assert type(exc_info.value).__name__ == 'ResolutionError'

    def test_error_is_not_runtime_error(self, rules: RuleGraph) -> None:
        """ResolutionError is a distinct exception type, not a generic RuntimeError."""

        class Missing:
            pass

        registry = RegistryBuilder()

        with pytest.raises(Exception) as exc_info:
            _ = compile(Missing, registry.build(), rules)

        assert type(exc_info.value).__name__ == 'ResolutionError'
        assert not isinstance(exc_info.value, RuntimeError)

    def test_no_match_rules_collapsed(self, rules: RuleGraph) -> None:
        """Leaf 'no match' errors are collapsed into a count summary."""

        class Missing:
            pass

        registry = RegistryBuilder()

        with pytest.raises(Exception) as exc_info:
            _ = compile(Missing, registry.build(), rules)

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
            _ = compile(MyProto, registry.build(), rules)

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
            _ = compile(Outer, registry.build(), rules)

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
            dep: Dep

            def __init__(self, dep: Dep) -> None:
                self.dep = dep

        registry = RegistryBuilder().register(Service)(Service)

        with pytest.raises(Exception) as exc_info:
            _ = compile(Service, registry.build(), rules)

        msg = str(exc_info.value)
        assert 'Dep' in msg

    def test_callable_type_displayed_with_signature(self, rules: RuleGraph) -> None:
        """Callable types in errors show their parameter and return types."""

        class MyProto(typing.Protocol):
            def process(self, x: int) -> str: ...

        registry = RegistryBuilder()

        with pytest.raises(Exception) as exc_info:
            _ = compile(MyProto, registry.build(), rules)

        msg = str(exc_info.value)
        # The callable type should show its signature, not just "Callable[...]"
        assert 'int' in msg
        assert 'str' in msg

    def test_variadic_provider_is_not_treated_as_zero_arg_constructor(
        self, rules: RuleGraph
    ) -> None:
        """Providers with *args/**kwargs must stay unresolvable."""

        class Service:
            pass

        def provide_service(*_args: object, **_kwargs: object) -> Service:
            return Service()

        registry = RegistryBuilder().register(Service)(provide_service)

        with pytest.raises(Exception) as exc_info:
            _ = compile(Service, registry.build(), rules)

        assert type(exc_info.value).__name__ == 'ResolutionError'

    def test_registered_typeddict_does_not_leak_scalar_field_providers(
        self, rules: RuleGraph
    ) -> None:
        """Registering a TypedDict must not create fake scalar providers."""

        class State(typing.TypedDict):
            branch_id: int

        class NeedsInt:
            value: int

            def __init__(self, value: int) -> None:
                self.value = value

        registry = RegistryBuilder().register(State)(State).register(NeedsInt)(NeedsInt)

        with pytest.raises(Exception) as exc_info:
            _ = compile(NeedsInt, registry.build(), rules)

        assert type(exc_info.value).__name__ == 'ResolutionError'

    def test_transition_scope_duplicate_scalar_bindings_surface_ambiguity(
        self, rules: RuleGraph
    ) -> None:
        """Multiple same-typed transition bindings should stay ambiguous."""

        class Service:
            def __init__(self, value: int) -> None:
                self.value: int = value

        class Child(typing.Protocol):
            @property
            def service(self) -> Service: ...

        @typing.final
        class PairTransition:
            def with_pair(self, branch_id: int, _session_id: int) -> int:
                return branch_id

        class HasPair(typing.Protocol):
            def with_pair(self, branch_id: int, session_id: int) -> Child: ...

        class Root(HasPair, typing.Protocol):
            pass

        # given
        registry = (
            RegistryBuilder()
            .register(Service)(Service)
            .register_method(HasPair, HasPair.with_pair)(PairTransition)
        )

        # when
        previous = os.environ.get('DISABLE_ERROR_TRUNCATION')
        os.environ['DISABLE_ERROR_TRUNCATION'] = '1'
        try:
            with pytest.raises(Exception) as exc_info:
                _ = compile(Root, registry.build(), rules)
        finally:
            if previous is None:
                _ = os.environ.pop('DISABLE_ERROR_TRUNCATION', None)
            else:
                os.environ['DISABLE_ERROR_TRUNCATION'] = previous

        # then
        assert type(exc_info.value).__name__ == 'ResolutionError'
        assert 'ambiguous constant' in str(exc_info.value).lower()
