"""Protocol constructor registration tests."""

import typing

from inlay import RegistryBuilder, RuleGraph, compile


class TestConstructorForProtocolType:
    def test_register_constructor_for_protocol_type(self, rules: RuleGraph) -> None:
        """A Protocol type registered via .register(Proto)(Impl) should
        be resolvable via the constructor rule.
        """

        class MyProto(typing.Protocol):
            def do_thing(self) -> str: ...

        class MyImpl:
            def do_thing(self) -> str:
                return 'done'

        registry = RegistryBuilder().register(MyProto)(MyImpl)

        result = compile(MyProto, registry.build(), rules)

        assert result.do_thing() == 'done'

    def test_protocol_property_depends_on_constructor_registered_protocol(
        self, rules: RuleGraph
    ) -> None:
        """A protocol property whose type is itself a Protocol registered
        via constructor should resolve.
        """

        class Inner(typing.Protocol):
            def value(self) -> int: ...

        class InnerImpl:
            def value(self) -> int:
                return 42

        class Outer(typing.Protocol):
            @property
            def inner(self) -> Inner: ...

        registry = RegistryBuilder().register(Inner)(InnerImpl)

        result = compile(Outer, registry.build(), rules)

        assert result.inner.value() == 42

    def test_protocol_with_generic_methods_resolved_via_constructor(
        self, rules: RuleGraph
    ) -> None:
        """A Protocol with generic methods registered via constructor should
        be resolvable when the protocol also has non-generic members.
        """

        class Executor(typing.Protocol):
            def execute[T](self, item: T) -> T: ...
            def run(self) -> None: ...

        class ExecutorImpl:
            def execute[T](self, item: T) -> T:
                return item

            def run(self) -> None:
                pass

        class Context(typing.Protocol):
            @property
            def executor(self) -> Executor: ...

        registry = RegistryBuilder().register(Executor)(ExecutorImpl)

        result = compile(Context, registry.build(), rules)

        assert result.executor.execute('hello') == 'hello'

    def test_qualified_protocol_with_generic_methods_via_constructor(
        self, rules: RuleGraph
    ) -> None:
        """A qualified Protocol with generic methods registered via constructor
        should resolve when the context is compiled with a qualifier.

        Reproduces: DeferredDecisionExecutor registered with qual('story')
        is not found when StoryReadContext is compiled in qual('story') scope.
        """
        from typing import Annotated

        from inlay import qual

        class Executor(typing.Protocol):
            def execute[T](self, item: T) -> T: ...

        class ExecutorImpl:
            def execute[T](self, item: T) -> T:
                return item

        class ChildCtx(typing.Protocol):
            @property
            def executor(self) -> Executor: ...

        class RootCtx(typing.Protocol):
            def enter(self) -> Annotated[ChildCtx, qual('scoped')]: ...

        registry = RegistryBuilder().register(Executor, qualifiers=qual('scoped'))(
            ExecutorImpl
        )

        root = compile(RootCtx, registry.build(), rules)
        child = root.enter()

        assert child.executor.execute('hello') == 'hello'


class TestParamSpecProtocolConstructor:
    def test_protocol_with_paramspec_resolved_via_constructor(
        self, rules: RuleGraph
    ) -> None:
        """A Protocol with a ParamSpec method registered via constructor
        should be resolvable directly.
        """
        from collections.abc import Callable
        from typing import Concatenate

        class Ctx:
            pass

        class Exec(typing.Protocol):
            def execute[**P, R](
                self,
                method: Callable[Concatenate[Ctx, P], R],
                *args: P.args,
                **kwargs: P.kwargs,
            ) -> R: ...

        class ExecImpl:
            def execute[**P, R](
                self,
                method: Callable[Concatenate[Ctx, P], R],
                *args: P.args,
                **kwargs: P.kwargs,
            ) -> R:
                return method(Ctx(), *args, **kwargs)

        registry = RegistryBuilder().register(Exec)(ExecImpl)

        result = compile(Exec, registry.build(), rules)

        assert isinstance(result, ExecImpl)

    def test_protocol_with_paramspec_as_property_dep(self, rules: RuleGraph) -> None:
        """A Protocol with ParamSpec methods used as a property type
        in another protocol should resolve via constructor.
        """
        from collections.abc import Callable
        from typing import Concatenate

        class Ctx:
            pass

        class Exec(typing.Protocol):
            def execute[**P, R](
                self,
                method: Callable[Concatenate[Ctx, P], R],
                *args: P.args,
                **kwargs: P.kwargs,
            ) -> R: ...

        class ExecImpl:
            def execute[**P, R](
                self,
                method: Callable[Concatenate[Ctx, P], R],
                *args: P.args,
                **kwargs: P.kwargs,
            ) -> R:
                return method(Ctx(), *args, **kwargs)

        class HasExec(typing.Protocol):
            @property
            def executor(self) -> Exec: ...

        registry = RegistryBuilder().register(Exec)(ExecImpl)

        result = compile(HasExec, registry.build(), rules)

        assert isinstance(result.executor, ExecImpl)
