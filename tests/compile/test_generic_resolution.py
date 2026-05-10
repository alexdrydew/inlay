"""Generic resolution compile tests."""

import typing

from inlay import Registry, RuleGraph, compile


class TestParametricConstructorRegistration:
    def test_parametric_registration_matches_bare_request(
        self, rules: RuleGraph
    ) -> None:
        """register(Foo[T])(impl) should be found when resolving bare Foo."""

        class Repo[T](typing.Protocol):
            def get(self, _id: str) -> T: ...

        class RepoImpl:
            def get(self, _id: str) -> object:
                return None

        T = typing.TypeVar('T')
        registry = Registry().register(Repo[T])(RepoImpl)  # pyright: ignore[reportGeneralTypeIssues]

        target = typing.cast(type[Repo[object]], Repo)
        result: Repo[object] = compile(target, registry.build(), rules)

        assert result.get('x') is None

    def test_parametric_registration_as_property_dep(self, rules: RuleGraph) -> None:
        """Property requiring bare Foo should match Foo[T] registration."""

        class Repo[T](typing.Protocol):
            def get(self, _id: str) -> T: ...

        class RepoImpl:
            def get(self, _id: str) -> object:
                return None

        T = typing.TypeVar('T')

        class HasRepo(typing.Protocol):
            @property
            def repo(self) -> Repo[object]: ...

        registry = Registry().register(Repo[T])(RepoImpl)  # pyright: ignore[reportGeneralTypeIssues]

        result = compile(HasRepo, registry.build(), rules)

        assert result.repo.get('x') is None


class TestTypeVarDefaultSubstitution:
    def test_bare_generic_with_default_resolves_to_concrete_registration(
        self, rules: RuleGraph
    ) -> None:
        """Bare HasThing (with T: Base = Base) should normalize to
        HasThing[Base] and resolve a constructor registered for that type.
        """

        class Base(typing.Protocol):
            def value(self) -> int: ...

        class HasThing[T: Base = Base](typing.Protocol):
            @property
            def thing(self) -> T: ...

        class BaseImpl:
            def value(self) -> int:
                return 42

        class HasThingImpl:
            @property
            def thing(self) -> BaseImpl:
                return BaseImpl()

        registry = Registry().register(HasThing[Base])(HasThingImpl)

        result = compile(HasThing, registry.build(), rules)

        assert result.thing.value() == 42

    def test_bare_generic_default_with_bridge_factory(self, rules: RuleGraph) -> None:
        """Real-world pattern: register HasThing[Concrete], provide a bridge
        factory HasThing[Concrete] -> HasThing[Base], resolve bare HasThing.
        """

        class Base(typing.Protocol):
            def value(self) -> int: ...

        class Concrete:
            def value(self) -> int:
                return 99

        class HasThing[T: Base = Base](typing.Protocol):
            @property
            def thing(self) -> T: ...

        class ConcreteHasThingImpl:
            @property
            def thing(self) -> Concrete:
                return Concrete()

        def bridge(ctx: HasThing[Concrete]) -> HasThing[Base]:
            return ctx  # type: ignore[return-value]

        registry = (
            Registry()
            .register(HasThing[Concrete])(ConcreteHasThingImpl)
            .register_factory(bridge)
        )

        result = compile(HasThing, registry.build(), rules)

        assert result.thing.value() == 99

    def test_bare_generic_default_as_property_dep(self, rules: RuleGraph) -> None:
        """Bare HasThing with default TypeVar used as a property dependency."""

        class Base(typing.Protocol):
            def value(self) -> int: ...

        class HasThing[T: Base = Base](typing.Protocol):
            @property
            def thing(self) -> T: ...

        class BaseImpl:
            def value(self) -> int:
                return 7

        class HasThingImpl:
            @property
            def thing(self) -> BaseImpl:
                return BaseImpl()

        class Root(typing.Protocol):
            @property
            def has_thing(self) -> HasThing: ...

        registry = Registry().register(HasThing[Base])(HasThingImpl)

        result = compile(Root, registry.build(), rules)

        assert result.has_thing.thing.value() == 7

    def test_bare_generic_base_typevar_default_substituted_in_child(
        self, rules: RuleGraph
    ) -> None:
        """When a protocol inherits a bare generic base (HasTx[T = Tx]),
        the TypeVar default must propagate into the child's members.

        This mirrors the real-world AiWriteContext(HasTransaction, ...)
        pattern where HasTransaction.transaction returns TxT which
        should default to Transaction.
        """

        class Tx:
            def __init__(self) -> None:
                self.active: bool = True

        class HasTx[T: Tx = Tx](typing.Protocol):
            @property
            def tx(self) -> T: ...

        class WriteCtx(HasTx, typing.Protocol):
            @property
            def name(self) -> str: ...

        class WriteCtxImpl:
            @property
            def tx(self) -> Tx:
                return Tx()

            @property
            def name(self) -> str:
                return 'write'

        registry = Registry().register(WriteCtx)(WriteCtxImpl)

        result = compile(WriteCtx, registry.build(), rules)

        assert result.tx.active is True
        assert result.name == 'write'

    def test_bare_generic_base_transitive_typevar_default(
        self, rules: RuleGraph
    ) -> None:
        """TypeVar default resolves through intermediate protocol chain.

        HasOptBase[T=Tx] -> HasBase[T=Tx](HasOptBase[T]) -> Child(HasBase)
        Python flattens __orig_bases__ so HasBase disappears, leaving
        HasOptBase[TxT_from_HasBase]. The TxT must still resolve to Tx.
        """

        class Tx:
            def __init__(self) -> None:
                self.active: bool = True

        class HasOptBase[T: Tx = Tx](typing.Protocol):
            @property
            def tx(self) -> T | None: ...

        class HasBase[T: Tx = Tx](HasOptBase[T], typing.Protocol):
            @property
            @typing.override
            def tx(self) -> T: ...

        class Child(HasBase, typing.Protocol):
            @property
            def name(self) -> str: ...

        class ChildImpl:
            @property
            def tx(self) -> Tx:
                return Tx()

            @property
            def name(self) -> str:
                return 'child'

        registry = Registry().register(Child)(ChildImpl)

        result = compile(Child, registry.build(), rules)

        assert result.tx.active is True
        assert result.name == 'child'


class TestConstructorProtocolThroughGenericBaseChain:
    def test_constructor_protocol_through_generic_base_and_qualified_transition(
        self, rules: RuleGraph
    ) -> None:
        """Protocol registered as constructor should resolve when accessed
        through: qualified transition -> generic base substitution -> property.
        """
        from typing import Annotated

        from inlay import qual

        class Executor(typing.Protocol):
            def execute[T](self, item: T) -> T: ...

        class ExecutorImpl:
            def execute[T](self, item: T) -> T:
                return item

        class ReadCtx(typing.Protocol):
            @property
            def executor(self) -> Executor: ...

        class HasRead[T](typing.Protocol):
            def with_read(self) -> T: ...

        class ModuleCtx(HasRead[ReadCtx], typing.Protocol): ...

        class HasModule(typing.Protocol):
            def with_module(self) -> Annotated[ModuleCtx, qual('mod')]: ...

        registry = Registry().register(Executor, qualifiers=qual('mod'))(ExecutorImpl)

        root = compile(HasModule, registry.build(), rules)
        module = root.with_module()
        read = module.with_read()

        assert read.executor.execute('hello') == 'hello'


class TestGenericBaseProtocol:
    def test_generic_base_protocol_typevar_substituted(self, rules: RuleGraph) -> None:
        """When a protocol inherits from a generic base, the TypeVar should be
        substituted with the concrete type argument.

        Reproduces: StoryContext extends WriteTransition[StoryWriteContext],
        so with_write() should return StoryWriteContext, not ~TxCtxT.
        """

        class ChildCtx(typing.Protocol):
            pass

        class HasTransition[T](typing.Protocol):
            def enter(self) -> T: ...

        class MyContext(HasTransition[ChildCtx], typing.Protocol): ...

        registry = Registry()

        ctx = compile(MyContext, registry.build(), rules)
        child = ctx.enter()

        assert child is not None

    def test_generic_base_with_property_child(self, rules: RuleGraph) -> None:
        """Generic base protocol with TypeVar in transition return type,
        where the child context has resolvable properties.
        """

        class Service:
            pass

        class ChildCtx(typing.Protocol):
            @property
            def svc(self) -> Service: ...

        class HasTransition[T](typing.Protocol):
            def enter(self) -> T: ...

        class MyContext(HasTransition[ChildCtx], typing.Protocol): ...

        registry = Registry().register(Service)(Service)

        ctx = compile(MyContext, registry.build(), rules)
        child = ctx.enter()

        assert isinstance(child.svc, Service)

    def test_multiple_generic_bases(self, rules: RuleGraph) -> None:
        """Protocol inheriting from multiple generic bases with different TypeVars.

        Reproduces: StoryContext extends both WriteTransition[StoryWriteContext]
        and ReadContextTransition[StoryReadContext].
        """

        class WriteCtx(typing.Protocol):
            pass

        class ReadCtx(typing.Protocol):
            pass

        class HasWrite[T](typing.Protocol):
            def with_write(self) -> T: ...

        class HasRead[T](typing.Protocol):
            def with_read(self) -> T: ...

        class MyContext(HasWrite[WriteCtx], HasRead[ReadCtx], typing.Protocol): ...

        registry = Registry()

        ctx = compile(MyContext, registry.build(), rules)

        write = ctx.with_write()
        assert write is not None

        read = ctx.with_read()
        assert read is not None

    def test_nested_specialized_base_property_with_qualifier_resolves(
        self,
        rules: RuleGraph,
    ) -> None:
        from typing import Annotated, TypedDict

        from inlay import qual

        class Interface(typing.Protocol):
            def use(self) -> None: ...

        class Implementation:
            def use(self) -> None:
                pass

        class Constants(TypedDict):
            value: Annotated[Interface, qual('slot')]

        def provide_constants() -> Constants:
            return {'value': Implementation()}

        class HasValue[ValueT: Interface = Interface](typing.Protocol):
            @property
            def value(self) -> Annotated[ValueT, qual('slot')]: ...

        class Alias[AliasT: Interface](HasValue[AliasT], typing.Protocol): ...

        class SpecializedContext(Alias[Interface], typing.Protocol): ...

        registry = Registry().register_factory(provide_constants)

        result = compile(SpecializedContext, registry.build(), rules)

        result.value.use()
