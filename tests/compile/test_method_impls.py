"""Method implementation resolution tests."""

import typing

import pytest

from inlay import RegistryBuilder, RuleGraph, compile
from inlay.rules import (
    RuleGraphBuilder,
    attribute_source_rule,
    constant_rule,
    match_first,
    method_impl_rule,
    protocol_rule,
    sentinel_none_rule,
)


def _build_method_impl_only_rules() -> RuleGraph:
    builder = RuleGraphBuilder()

    self_ref = builder.lazy(lambda: pipeline)
    method_rules = method_impl_rule(target_rules=self_ref)
    pipeline = match_first(
        sentinel_none_rule(),
        constant_rule(),
        attribute_source_rule(resolve=self_ref),
        protocol_rule(resolve=self_ref, method_rules=method_rules),
    )

    return builder.build()


class TestMethodImplNameFiltering:
    """method_impl filters registered implementations by method name
    via callable.inner.function_name.
    """

    def test_zero_param_impl_does_not_leak_to_other_methods(
        self, rules: RuleGraph
    ) -> None:
        """A registered zero-param method impl (with_read) must NOT be used
        for an unrelated zero-param auto-method (with_child). The auto_method
        rule should handle with_child instead.
        """
        from typing import Annotated

        from inlay import qual

        class ChildCtx(typing.Protocol):
            pass

        class HasChild(typing.Protocol):
            def with_child(self) -> Annotated[ChildCtx, qual('x')]: ...

        class ReadTransition[T](typing.Protocol):
            def with_read(self) -> T: ...

        class RootCtx(HasChild, typing.Protocol): ...

        called = False

        def provide_read() -> dict[str, str]:
            nonlocal called
            called = True
            return {'marker': 'from_provide_read'}

        read_method = typing.cast(
            typing.Callable[..., object], ReadTransition.with_read
        )
        module_registry = RegistryBuilder().register_method(
            ReadTransition, read_method
        )(provide_read)

        registry = RegistryBuilder().include(module_registry, qualifiers=qual('a'))
        root = compile(RootCtx, registry.build(), rules)
        _ = root.with_child()
        assert not called, (
            'provide_read should NOT have been called for with_child - '
            'method_impl must filter by method name'
        )

    def test_multiple_module_includes_no_ambiguity_for_unrelated_method(
        self, rules: RuleGraph
    ) -> None:
        """Multiple copies of a zero-param method impl (from multi-module
        includes) must not cause ambiguity for an unrelated method, even
        when the child protocol has unresolvable dependencies.
        """
        from typing import Annotated, TypedDict

        from inlay import qual

        class ReadConstants(TypedDict):
            pass

        class ReadTransition[T](typing.Protocol):
            def with_read(self) -> T: ...

        class Dependency:
            pass

        class ChildCtx(typing.Protocol):
            @property
            def dep(self) -> Dependency: ...

        class HasChild(typing.Protocol):
            def with_child(self) -> Annotated[ChildCtx, qual('x')]: ...

        class RootCtx(HasChild, typing.Protocol): ...

        def provide_read() -> ReadConstants:
            return {}

        read_method = typing.cast(
            typing.Callable[..., object], ReadTransition.with_read
        )
        module_registry = RegistryBuilder().register_method(
            ReadTransition, read_method
        )(provide_read)

        registry = (
            RegistryBuilder()
            .include(module_registry, qualifiers=qual('a'))
            .include(module_registry, qualifiers=qual('b'))
        )

        # with_child has an unresolvable dep, so auto_method will fail.
        # But method_impl should NOT match with_read impls for with_child.
        # The error should be about Dependency, not about ambiguous method.
        with pytest.raises(Exception, match='Dependency'):
            _ = compile(RootCtx, registry.build(), rules)


class TestMethodImplQualifierSplit:
    def test_method_provides_does_not_qualify_implementation_params(
        self, rules: RuleGraph
    ) -> None:
        from typing import Annotated

        from inlay import qual

        class Dep:
            pass

        class ChildCtx(typing.Protocol): ...

        class RootCtx(typing.Protocol):
            def enter(self) -> Annotated[ChildCtx, qual('child')]: ...

        seen: list[Dep] = []

        def enter(dep: Dep) -> None:
            seen.append(dep)

        registry = (
            RegistryBuilder()
            .register(Dep)(Dep)
            .register_method(RootCtx, RootCtx.enter, provides=qual('child'))(enter)
        )

        root = compile(RootCtx, registry.build(), rules)
        _ = root.enter()

        assert len(seen) == 1
        assert isinstance(seen[0], Dep)

    def test_method_requires_qualifies_source_and_implementation_params(
        self, rules: RuleGraph
    ) -> None:
        from typing import Annotated

        from inlay import normalize, qual

        class Dep:
            pass

        class ModDep(Dep):
            pass

        class ChildCtx(typing.Protocol): ...

        class RootCtx(typing.Protocol):
            def enter(self) -> ChildCtx: ...

        seen: list[Dep] = []

        def enter(dep: Dep) -> None:
            seen.append(dep)

        registry = (
            RegistryBuilder()
            .register(Dep, provides=qual('mod'))(ModDep)
            .register_method(RootCtx, RootCtx.enter, requires=qual('mod'))(enter)
            .build()
        )

        root = typing.cast(
            RootCtx,
            registry.compile(rules, normalize(Annotated[RootCtx, qual('mod')])),
        )
        _ = root.enter()

        assert len(seen) == 1
        assert isinstance(seen[0], ModDep)

    def test_method_requires_qualifies_auto_bound_constructor_through_include(
        self, rules: RuleGraph
    ) -> None:
        from typing import Annotated, final

        from inlay import normalize, qual

        class Config:
            pass

        class ModConfig(Config):
            pass

        class ChildCtx(typing.Protocol): ...

        class RootCtx(typing.Protocol):
            def enter(self) -> ChildCtx: ...

        seen: list[Config] = []

        @final
        class Transition:
            def __init__(self, config: Config) -> None:
                self.config = config

            def enter(self) -> None:
                seen.append(self.config)

        module_registry = RegistryBuilder().register_method(RootCtx, RootCtx.enter)(
            Transition
        )
        registry = (
            RegistryBuilder()
            .register(Config, provides=qual('mod'))(ModConfig)
            .include(module_registry, requires=qual('mod'))
            .build()
        )

        root = typing.cast(
            RootCtx,
            registry.compile(rules, normalize(Annotated[RootCtx, qual('mod')])),
        )
        _ = root.enter()

        assert len(seen) == 1
        assert isinstance(seen[0], ModConfig)


class TestClassBasedMethodImpl:
    """Class-based method implementations (register_method with a class)
    should match against the actual method signature, not the class
    constructor. Constructor dependencies are resolved via bound_to.
    """

    def test_class_method_impl_matches_protocol_method(self, rules: RuleGraph) -> None:
        """A class whose constructor has dependencies should still match
        a zero-param protocol method when registered via register_method.

        Before the fix, _build_method built the callable type from the
        class __init__ (2 params) instead of the method (0 params), so
        cross_unify_callable_params failed with a param count mismatch.
        """
        from collections.abc import AsyncGenerator
        from contextlib import AbstractAsyncContextManager, asynccontextmanager
        from typing import Protocol, TypedDict, final

        class Transaction:
            pass

        class WriteConstants(TypedDict):
            transaction: Transaction

        class Config:
            def __init__(self) -> None:
                self.value: int = 42

        @final
        class UowTransition:
            def __init__(self, config: Config) -> None:
                self._config = config

            @asynccontextmanager
            async def with_write(self) -> AsyncGenerator[WriteConstants]:
                yield {'transaction': Transaction()}

        class WriteContext(Protocol):
            @property
            def transaction(self) -> Transaction: ...

        class HasUnitOfWork[T](Protocol):
            def with_write(self) -> AbstractAsyncContextManager[T]: ...

        class RootContext(HasUnitOfWork[WriteContext], Protocol):
            pass

        registry = (
            RegistryBuilder()
            .register(Config)(Config)
            .register_method(
                HasUnitOfWork,
                typing.cast(typing.Callable[..., object], HasUnitOfWork.with_write),
            )(UowTransition)
        )

        def factory() -> RootContext: ...

        ctx = compile(factory, registry.build(), rules)
        assert ctx is not None

    def test_class_method_impl_with_params(self, rules: RuleGraph) -> None:
        """A class-based method impl where the method has call-time params."""
        from typing import Protocol, final

        class SessionId:
            def __init__(self, value: str) -> None:
                self.value: str = value

        class SessionContext(Protocol):
            @property
            def session_id(self) -> SessionId: ...

        class Config:
            pass

        @final
        class SessionProvider:
            def __init__(self, config: Config) -> None:
                self._config = config

            def with_session(self, session_id: SessionId) -> SessionId:
                return session_id

        class HasSession(Protocol):
            def with_session(self, session_id: SessionId) -> SessionContext: ...

        class RootContext(HasSession, Protocol):
            pass

        registry = (
            RegistryBuilder()
            .register(Config)(Config)
            .register_method(HasSession, HasSession.with_session)(SessionProvider)
        )

        def factory() -> RootContext: ...

        compiled_factory = compile(factory, registry.build(), rules)
        assert compiled_factory is not None


class TestMethodImplWrapperCompatibility:
    def test_sync_protocol_accepts_async_impl_but_call_rejects_it(
        self,
    ) -> None:
        class State(typing.TypedDict):
            value: int

        class Child(typing.Protocol):
            @property
            def value(self) -> int: ...

        class Root(typing.Protocol):
            def load(self) -> Child: ...

        async def load() -> State:
            return {'value': 1}

        registry = RegistryBuilder().register_method(Root, Root.load)(load)

        root = compile(Root, registry.build(), _build_method_impl_only_rules())
        with pytest.raises(RuntimeError, match='awaitable method implementation'):
            _ = root.load()

    def test_positional_only_protocol_can_feed_keyword_only_implementation(
        self,
    ) -> None:
        class State(typing.TypedDict):
            value: int

        class Child(typing.Protocol):
            @property
            def value(self) -> int: ...

        class Root(typing.Protocol):
            def load(self, value: int, /) -> Child: ...

        def load(*, value: int) -> State:
            return {'value': value}

        registry = RegistryBuilder().register_method(Root, Root.load)(load)

        root = compile(Root, registry.build(), _build_method_impl_only_rules())
        assert root.load(1).value == 1

    def test_variadic_method_impl_call_uses_fixed_prefix_for_transition_scope(
        self,
        rules: RuleGraph,
    ) -> None:
        class State(typing.TypedDict):
            value: int

        class Child(typing.Protocol):
            @property
            def value(self) -> int: ...

        class Root(typing.Protocol):
            def run(self, first: int, *rest: int) -> Child: ...

        def run(first: int, *rest: int) -> State:
            return {'value': first + sum(rest)}

        root = compile(
            Root,
            RegistryBuilder().register_method(Root, Root.run)(run).build(),
            rules,
        )

        assert root.run(1, 2, 3).value == 1


class TestTransitionTypedDictQualifierPropagation:
    """TypedDict returned by a transition method_impl should have its
    fields available at the child scope's qualifier.

    Bug: the TypedDict is normalized at definition time with empty qualifier.
    When the transition is inside an include(qual('write')) chain, the
    callable gets {mod, write} but the return TypedDict fields keep {}.
    The resolver at {mod, write} can't find them.
    """

    def test_transition_typeddict_fields_available_in_qualified_child_scope(
        self, rules: RuleGraph
    ) -> None:
        from typing import Annotated, TypedDict

        from inlay import qual

        class Transaction:
            pass

        class UowConstants(TypedDict):
            transaction: Transaction

        class WriteTransition[T](typing.Protocol):
            def with_write(self) -> Annotated[T, qual('write')]: ...

        def provide_uow() -> UowConstants:
            return {'transaction': Transaction()}

        class Service:
            def __init__(self, transaction: Transaction) -> None:
                self.transaction: Transaction = transaction

        class WriteCtx(typing.Protocol):
            @property
            def service(self) -> Service: ...

        class ModuleCtx(WriteTransition[WriteCtx], typing.Protocol): ...

        class RootCtx(typing.Protocol):
            def with_module(self) -> Annotated[ModuleCtx, qual('mod')]: ...

        # Service is in write scope (inside qual('write') include).
        # provide_uow has explicit provides=qual('write') at module level.
        # This means:
        #   requires = {mod} (matching - found at {mod} scope)
        #   provides = {mod, write} (return type normalized with this)
        # So TypedDict field Transaction gets {mod, write} qualifier.
        write_registry = RegistryBuilder().register(Service)(Service)

        module_registry = (
            RegistryBuilder()
            .include(write_registry, qualifiers=qual('write'))
            .register_method(
                WriteTransition,
                typing.cast(typing.Callable[..., object], WriteTransition.with_write),
                provides=qual('write'),
            )(provide_uow)
        )

        registry = RegistryBuilder().include(module_registry, qualifiers=qual('mod'))

        ctx = compile(RootCtx, registry.build(), rules)
        write_ctx = ctx.with_module().with_write()
        assert isinstance(write_ctx.service.transaction, Transaction)


class TestTransitionResultBindings:
    def test_repeated_zero_arg_explicit_method_uses_current_result_binding(
        self, rules: RuleGraph
    ) -> None:
        from typing import TypedDict

        class State(TypedDict):
            value: int

        class Child(typing.Protocol):
            @property
            def value(self) -> int: ...

        class Root(typing.Protocol):
            def next(self) -> Child: ...

        counter = 0

        def next_() -> State:
            nonlocal counter
            counter += 1
            return {'value': counter}

        # given
        registry = RegistryBuilder().register_method(Root, Root.next)(next_)

        # when
        root = compile(Root, registry.build(), rules)

        # then
        assert root.next().value == 1
        assert root.next().value == 2
