"""Advanced signature and normalization compile tests."""

import typing

from inlay import Registry, RuleGraph, compile, normalize


class TestSelfType:
    def test_protocol_with_self_return_type(self, rules: RuleGraph) -> None:
        """Protocol method returning Self should be normalizable."""
        from typing import Self

        class Builder(typing.Protocol):
            def set_name(self, name: str) -> Self: ...

        class HasBuilder(typing.Protocol):
            @property
            def builder(self) -> Builder: ...

        class ConcreteBuilder:
            def set_name(self, name: str) -> ConcreteBuilder:
                _ = name
                return self

        registry = Registry().register(Builder)(ConcreteBuilder)

        ctx = compile(HasBuilder, registry.build(), rules)

        assert ctx.builder is not None

    def test_protocol_with_self_parameter(self) -> None:
        """Protocol method with Self as parameter type should be normalizable."""
        from typing import Self

        class Comparable(typing.Protocol):
            def __lt__(self, other: Self) -> bool: ...

        class ComparableImpl:
            def __lt__(self, other: Self) -> bool:
                return False

        registry = Registry()

        _ = registry.register(Comparable)(ComparableImpl).build()


class TestProtocolWithDunderMethods:
    def test_protocol_with_dunder_returning_complex_type(self) -> None:
        """Protocol with dunder methods returning deeply nested types.

        Protocols may define dunder methods whose return types are complex
        nested unions of TypedDicts. The normalizer should handle these.
        """
        from collections.abc import Mapping
        from typing import Literal, TypedDict

        class SchemaA(TypedDict):
            type: Literal['a']
            value: str

        class SchemaB(TypedDict):
            type: Literal['b']
            items: list[SchemaA]

        class HasSchema(typing.Protocol):
            def model_dump(
                self,
                *,
                mode: Literal['json', 'python'] = 'python',
                by_alias: bool = False,
                exclude_none: bool = False,
            ) -> Mapping[str, object]: ...

            def __get_schema__(
                self,
            ) -> SchemaA | SchemaB | Mapping[str, SchemaA | SchemaB]: ...

        _ = normalize(HasSchema)


class TestGenericMethods:
    def test_protocol_with_generic_method(self) -> None:
        """Protocol with a generic method should be buildable."""

        class Loader(typing.Protocol):
            def load[T](self, item: T) -> T: ...

        class LoaderImpl:
            def load[T](self, item: T) -> T:
                return item

        registry = Registry().register(Loader)(LoaderImpl)
        _ = registry.build()

    def test_protocol_with_bounded_generic_method(self) -> None:
        """Protocol with a bounded generic method should be buildable."""

        class State(typing.Protocol):
            pass

        class Loader(typing.Protocol):
            async def load[S: State](self, query: S) -> S: ...

        class LoaderImpl:
            async def load[S: State](self, query: S) -> S:
                return query

        registry = Registry().register(Loader)(LoaderImpl)
        _ = registry.build()

    def test_compile_protocol_with_generic_method_member(
        self, rules: RuleGraph
    ) -> None:
        """Compiling a protocol whose member has generic methods should not panic.

        Method-scoped TypeVars become OpaqueTypeVar during monomorphization.
        Resolution fails gracefully because no rule can resolve them yet.
        """

        class Loader(typing.Protocol):
            def load[T](self, item: T) -> T: ...

        class LoaderImpl:
            def load[T](self, item: T) -> T:
                return item

        class HasLoader(typing.Protocol):
            @property
            def loader(self) -> Loader: ...

        registry = Registry().register(Loader)(LoaderImpl)

        ctx = compile(HasLoader, registry.build(), rules)

        assert ctx.loader is not None

    def test_compile_protocol_with_bounded_generic_method_member(
        self, rules: RuleGraph
    ) -> None:
        """Compiling a protocol whose member has bounded generic methods.

        Reproduces the real-world pattern: StoryReadContext contains
        HasDeferredDecisionExecutor, whose execute method has method-level
        TypeVars like [**P, ErrT, R].
        """

        class State(typing.Protocol):
            pass

        class Executor(typing.Protocol):
            async def execute[S: State, R](self, query: tuple[S, R]) -> tuple[S, R]: ...

        class ExecutorImpl:
            async def execute[S: State, R](self, _query: tuple[S, R]) -> tuple[S, R]:
                raise NotImplementedError

        def provide_executor() -> Executor:
            return typing.cast(Executor, typing.cast(object, ExecutorImpl()))

        class HasExecutor(typing.Protocol):
            @property
            def executor(self) -> Executor: ...

        registry = Registry().register(Executor)(provide_executor)

        ctx = compile(HasExecutor, registry.build(), rules)

        assert ctx.executor is not None

    def test_compile_protocol_with_unbound_class_generic_in_signature(
        self, rules: RuleGraph
    ) -> None:
        """Compiling a protocol that uses a generic class without binding its TypeVar.

        Reproduces the real-world pattern: EventStore methods use
        DCBEvent[E] and PersistedEvent[E] without binding E.
        """

        class Event[E](typing.Protocol):
            @property
            def payload(self) -> E: ...

        class Store(typing.Protocol):
            def get_events(self) -> list[Event[object]]: ...

        class StoreImpl:
            def get_events(self) -> list[Event[object]]:
                return []

        class HasStore(typing.Protocol):
            @property
            def store(self) -> Store: ...

        registry = Registry().register(Store)(StoreImpl)

        ctx = compile(HasStore, registry.build(), rules)

        assert ctx.store is not None

    def test_compile_protocol_with_paramspec_method_does_not_panic(
        self, rules: RuleGraph
    ) -> None:
        """Protocol with a method using ParamSpec should not abort/panic."""
        from collections.abc import Callable
        from typing import Concatenate

        class Ctx:
            pass

        class Executor(typing.Protocol):
            def execute[**P, R](
                self,
                method: Callable[Concatenate[Ctx, P], R],
                *args: P.args,
                **kwargs: P.kwargs,
            ) -> R: ...

        class ExecutorImpl:
            def execute[**P, R](
                self,
                method: Callable[Concatenate[Ctx, P], R],
                *args: P.args,
                **kwargs: P.kwargs,
            ) -> R:
                return method(Ctx(), *args, **kwargs)

        class HasExecutor(typing.Protocol):
            @property
            def executor(self) -> Executor: ...

        registry = Registry().register(Executor)(ExecutorImpl)

        ctx = compile(HasExecutor, registry.build(), rules)
        assert isinstance(ctx.executor, ExecutorImpl)
