"""Runtime tests for the compiled context factory.

These tests exercise the full path: compile -> call factory -> call transitions,
verifying that the runtime scope chain correctly propagates constants.
"""

from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    asynccontextmanager,
    contextmanager,
)
from types import TracebackType
from typing import Annotated, Protocol, TypedDict, cast, final

import anyio
import pytest

from inlay import Registry, compile, compiled, qual
from inlay.default import default_rules
from inlay.rules import (
    RuleGraphBuilder,
    constant_rule,
    constructor_rule,
    match_first,
    method_impl_rule,
    protocol_rule,
    sentinel_none_rule,
)


def _build_annotated_transition_rules():
    builder = RuleGraphBuilder()

    self_ref = builder.lazy(lambda: pipeline)

    method_rules = method_impl_rule(target_rules=self_ref)

    pipeline = match_first(
        sentinel_none_rule(),
        constant_rule(),
        constructor_rule(param_rules=self_ref),
        protocol_rule(resolve=self_ref, method_rules=method_rules),
    )

    return builder.build()


class _YieldOnce:
    def __await__(self) -> Generator[None]:
        yield None


class TestAutoTransitionCallSignature:
    def test_transition_param_is_visible_to_child(self) -> None:
        @final
        class Token:
            pass

        @final
        class UsesToken:
            def __init__(self, token: Token) -> None:
                self.token = token

        class Child(Protocol):
            @property
            def value(self) -> UsesToken: ...

        class Root(Protocol):
            def with_token(self, token: Token) -> Child: ...

        registry = Registry().register(UsesToken)(UsesToken)

        token = Token()
        root = compile(Root, registry.build(), _build_annotated_transition_rules())

        assert root.with_token(token).value.token is token

    def test_plain_param_still_reaches_method_implementation(self) -> None:
        @final
        class Token:
            pass

        class Child(Protocol):
            pass

        class Root(Protocol):
            def with_token(self, token: Token) -> Child: ...

        calls: list[Token] = []

        def with_token(token: Token) -> Child:
            calls.append(token)
            return cast(Child, object())

        token = Token()
        registry = Registry().register_method(Root, Root.with_token)(with_token)
        root = compile(Root, registry.build(), _build_annotated_transition_rules())

        _ = root.with_token(token)

        assert calls == [token]

    def test_auto_transition_rejects_invalid_call_shapes(self) -> None:
        class Child(Protocol):
            @property
            def value(self) -> int: ...

        class Root(Protocol):
            def with_child(self, value: int, /, label: str, *, flag: bool) -> Child: ...

        root = compile(Root, Registry().build(), default_rules())
        with_child = cast(Callable[..., object], root.with_child)

        assert root.with_child(1, 'ok', flag=True).value == 1
        with pytest.raises(TypeError, match='positional-only'):
            _ = with_child(value=1, label='ok', flag=True)
        with pytest.raises(TypeError, match='positional arguments'):
            _ = with_child(1, 'ok', True, flag=False)
        with pytest.raises(TypeError, match='unexpected keyword'):
            _ = with_child(1, 'ok', flag=True, extra=False)
        with pytest.raises(TypeError, match='multiple values'):
            _ = with_child(1, 'ok', label='duplicate', flag=True)

    def test_auto_transition_allows_variadic_positional_tail(self) -> None:
        class Child(Protocol):
            @property
            def value(self) -> int: ...

        class Root(Protocol):
            def run(self, first: int, *rest: int) -> Child: ...

        root = compile(Root, Registry().build(), default_rules())

        assert root.run(1, 2, 3).value == 1


class TestCompiledDecorator:
    def test_compiled_decorator_returns_compiled_factory(self) -> None:
        @final
        class Service:
            pass

        class Root(Protocol):
            @property
            def service(self) -> Service: ...

        @compiled(Registry().register(Service)(Service))
        def factory() -> Root: ...

        root = factory()

        assert isinstance(root.service, Service)

    def test_compiled_factory_wires_method_registered_on_base_protocol(self) -> None:
        events: list[str] = []

        class Hook(Protocol):
            async def on_new_message(self) -> None: ...

        class DurableContext(Hook, Protocol):
            pass

        async def record() -> None:
            events.append('called')

        registry = Registry().register_method(Hook, Hook.on_new_message)(record)

        @compiled(registry)
        def make_context() -> DurableContext: ...

        ctx = make_context()
        anyio.run(ctx.on_new_message)

        assert events == ['called']


class TestClassBasedMethodImplRuntime:
    """Runtime behavior of class-based method implementations.

    Method implementation resolution uses the actual method callable, not
    __init__, so class-based implementations can provide child-scope constants.
    """

    def test_transition_provides_constants_to_child(self) -> None:
        """Class-based method impl returns a TypedDict whose fields become
        constants in the child scope.  The child protocol's members should
        resolve via those constants at runtime.

        Pattern:  factory(seed) -> RootCtx.with_write() -> WriteCtx
        """

        class Config:
            pass

        class Transaction:
            pass

        class WriteConstants(TypedDict):
            transaction: Transaction

        @final
        class UowTransition:
            def __init__(self, config: Config) -> None:
                self._config = config

            def with_write(self) -> WriteConstants:
                return {'transaction': Transaction()}

        class WriteContext(Protocol):
            @property
            def transaction(self) -> Transaction: ...

        class HasUnitOfWork(Protocol):
            def with_write(self) -> WriteContext: ...

        class RootContext(HasUnitOfWork, Protocol):
            pass

        registry = (
            Registry()
            .register(Config)(Config)
            .register_method(HasUnitOfWork, HasUnitOfWork.with_write)(UowTransition)
        )
        rules = default_rules()

        def factory(_config: Config) -> RootContext: ...

        compiled_factory = compile(factory, registry.build(), rules)

        # when
        root = compiled_factory(Config())
        write_ctx = root.with_write()

        # then
        assert isinstance(write_ctx.transaction, Transaction)

    def test_nested_zero_impl_method_then_class_method_impl(self) -> None:
        """Two-level nesting: zero-implementation method transition followed by
        a class-based method implementation transition.

        Pattern:  factory(seed) -> Root.with_module() -> Module.with_write() -> WriteCtx

        This mirrors the real-world flow:
          app_factory(storages) -> ctx.with_chat_context() -> chat.with_write()
        """

        class Storages(TypedDict):
            db_name: str

        class Transaction:
            pass

        class WriteConstants(TypedDict):
            transaction: Transaction

        @final
        class UowTransition:
            def __init__(self, db_name: str) -> None:
                self._db_name = db_name

            def with_write(self) -> WriteConstants:
                return {'transaction': Transaction()}

        class WriteContext(Protocol):
            @property
            def transaction(self) -> Transaction: ...

        class HasUnitOfWork(Protocol):
            def with_write(self) -> WriteContext: ...

        class ModuleContext(HasUnitOfWork, Protocol):
            @property
            def db_name(self) -> str: ...

        class HasModule(Protocol):
            def with_module(self) -> ModuleContext: ...

        class RootContext(HasModule, Protocol):
            pass

        registry = Registry().register_method(HasUnitOfWork, HasUnitOfWork.with_write)(
            UowTransition
        )
        rules = default_rules()

        def factory(_storages: Storages) -> RootContext: ...

        compiled_factory = compile(factory, registry.build(), rules)

        # when
        root = compiled_factory({'db_name': 'test'})
        module = root.with_module()
        write_ctx = module.with_write()

        # then
        assert module.db_name == 'test'
        assert isinstance(write_ctx.transaction, Transaction)


class TestExplicitMemberAccess:
    def test_protocol_source_uses_attribute_access(self) -> None:
        class MappingWithAttribute(dict[str, int]):
            value: int

            def __init__(self) -> None:
                super().__init__(value=1)
                self.value = 2

        class Source(Protocol):
            value: int

        class Root(Protocol):
            value: int

        source = MappingWithAttribute()

        def provide_source() -> Source:
            return source

        registry = Registry().register_factory(provide_source)
        root = compile(Root, registry.build(), default_rules())

        assert root.value == 2

        root.value = 3

        assert source.value == 3
        assert source['value'] == 1

    def test_typed_dict_source_uses_item_access(self) -> None:
        class MappingWithAttribute(dict[str, int]):
            value: int

            def __init__(self) -> None:
                super().__init__(value=1)
                self.value = 2

        class State(TypedDict):
            value: int

        class Root(Protocol):
            value: int

        source = MappingWithAttribute()

        def provide_state() -> State:
            return cast(State, cast(object, source))

        registry = Registry().register_factory(provide_state)
        root = compile(Root, registry.build(), default_rules())

        assert root.value == 1

        root.value = 3

        assert source['value'] == 3
        assert source.value == 2


class TestTypeVarSubstitutionInGenericProtocol:
    """When a factory references a generic protocol like WriteTransition[TxCtxT],
    the protocol's members use the CLASS's TypeVar while the factory binds its
    OWN TypeVar.  apply_bindings must substitute both correctly.
    """

    def test_factory_typevar_propagates_into_protocol_members(self) -> None:
        """Factory provide_executor[T](src: Source[T]) -> Executor[T]
        where Source[T] is a protocol with a method returning T.

        When resolving Executor[Concrete], the factory binds T=Concrete.
        Source[T]'s method `get() -> T` must become `get() -> Concrete`,
        not `get() -> ~T`.
        """
        from inlay import LazyRef

        class Concrete:
            pass

        class Source[T](Protocol):
            @property
            def value(self) -> T: ...

        class Executor:
            def __init__(self, source: LazyRef[Source[Concrete]]) -> None:
                self._source: LazyRef[Source[Concrete]] = source

        class RootContext(Protocol):
            @property
            def executor(self) -> Executor: ...

        class SourceImpl(TypedDict):
            value: Concrete

        def provide_source() -> SourceImpl:
            return {'value': Concrete()}

        registry = (
            Registry()
            .register(Executor)(Executor)
            .register(Concrete)(Concrete)
            .register_factory(provide_source)
        )
        rules = default_rules()

        def factory() -> RootContext: ...

        # This should compile — Source[Concrete].value -> Concrete (not ~T)
        compiled_factory = compile(factory, registry.build(), rules)
        root = compiled_factory()
        assert root.executor is not None

    def test_factory_typevar_propagates_through_method_transition(self) -> None:
        """A generic protocol WriteTransition[TxCtxT] with a method
        `with_write() -> TxCtxT` is referenced by a factory that binds
        TxCtxT.  The method's return type must be the bound type.

        This reproduces the real-world pattern:
          provide_transaction_executor[TxCtxT](
              write_source: LazyRef[WriteTransition[TxCtxT]],
          ) -> TransactionExecutor[TxCtxT]
        """
        from inlay import LazyRef

        class WriteCtx:
            @property
            def value(self) -> WriteCtx:
                raise NotImplementedError

        class WriteConstants(TypedDict):
            value: WriteCtx

        class WriteTransition[T](Protocol):
            def with_write(self) -> T: ...

        @final
        class WriteTransitionImpl:
            def with_write(self) -> WriteConstants:
                return {'value': WriteCtx()}

        class Executor:
            def __init__(self, source: LazyRef[WriteTransition[WriteCtx]]) -> None:
                self._source: LazyRef[WriteTransition[WriteCtx]] = source

        class RootContext(WriteTransition[WriteCtx], Protocol):
            @property
            def executor(self) -> Executor: ...

        registry = (
            Registry()
            .register(Executor)(Executor)
            .register(WriteCtx)(WriteCtx)
            .register_method(
                WriteTransition,
                cast(Callable[..., object], WriteTransition.with_write),
            )(WriteTransitionImpl)
        )
        rules = default_rules()

        def factory() -> RootContext: ...

        compiled_factory = compile(factory, registry.build(), rules)
        root = compiled_factory()
        assert root.executor is not None
        write_ctx = root.with_write()
        assert isinstance(write_ctx.value, WriteCtx)


class TestConstructorIdentityAcrossQualifiers:
    """Constructed values should be shared across qualifier contexts when
    the same constructor (same registration) resolves the dependency.

    When a dependency T is registered with qual('a') | qual('b') | qual(),
    all three qualifier contexts use the same constructor. The runtime
    should reuse the constructed instance rather than calling the
    constructor separately for each qualified request.
    """

    def test_same_provider_registered_for_multiple_targets_shares_result(self) -> None:
        class A:
            pass

        class B:
            pass

        class Both(A, B):
            pass

        calls: list[Both] = []

        def make() -> Both:
            value = Both()
            calls.append(value)
            return value

        class Root(Protocol):
            @property
            def a(self) -> A: ...

            @property
            def b(self) -> B: ...

        registry = Registry().register(A)(make).register(B)(make)
        rules = default_rules()

        root = compile(Root, registry.build(), rules)

        assert root.a is root.b
        assert len(calls) == 1

    def test_zero_impl_method_transition_shares_constructed_value(self) -> None:
        """Parent.prop and parent.with_a().prop should be the same object
        when both resolve T from the same constructor registration."""

        @final
        class T:
            pass

        def make_t() -> T:
            return T()

        class AChild(Protocol):
            @property
            def prop(self) -> T: ...

        class Parent(AChild, Protocol):
            def with_a(self) -> Annotated[AChild, qual('a')]: ...

        def parent_factory() -> Parent: ...

        registry = Registry().register(T, qualifiers=qual('a') | qual())(make_t)
        rules = default_rules()

        factory = compile(parent_factory, registry.build(), rules)
        parent = factory()

        assert parent.prop is parent.with_a().prop

    def test_zero_impl_method_transition_shares_constructed_value_when_child_first(
        self,
    ) -> None:
        """Cache sharing must not depend on parent-vs-child access order."""

        @final
        class T:
            pass

        calls: list[T] = []

        def make_t() -> T:
            value = T()
            calls.append(value)
            return value

        class AChild(Protocol):
            @property
            def prop(self) -> T: ...

        class Parent(AChild, Protocol):
            def with_a(self) -> Annotated[AChild, qual('a')]: ...

        def parent_factory() -> Parent: ...

        registry = Registry().register(T, qualifiers=qual('a') | qual())(make_t)
        rules = default_rules()

        factory = compile(parent_factory, registry.build(), rules)
        parent = factory()
        child = parent.with_a()
        child_prop = child.prop

        assert parent.prop is child_prop
        assert calls == [child_prop]

    def test_multiple_transitions_share_constructed_value(self) -> None:
        """Parent.prop, parent.with_a().prop, and parent.with_b().prop
        should all be the same object when all resolve T from the same
        constructor registration."""

        @final
        class T:
            pass

        def make_t() -> T:
            return T()

        class AChild(Protocol):
            @property
            def prop(self) -> T: ...

        class BChild(Protocol):
            @property
            def prop(self) -> T: ...

        class Parent(AChild, Protocol):
            def with_a(self) -> Annotated[AChild, qual('a')]: ...
            def with_b(self) -> Annotated[BChild, qual('b')]: ...

        def parent_factory() -> Parent: ...

        registry = Registry().register(T, qualifiers=qual('a') | qual('b') | qual())(
            make_t
        )
        rules = default_rules()

        factory = compile(parent_factory, registry.build(), rules)
        parent = factory()
        a_child = parent.with_a()
        b_child = parent.with_b()

        assert parent.prop is a_child.prop
        assert parent.prop is b_child.prop
        assert a_child.prop is b_child.prop

    def test_constant_already_shared_across_qualifiers(self) -> None:
        """Constants (factory params) should already be shared across
        qualifier contexts — this is a baseline sanity check."""

        @final
        class T:
            pass

        class AChild(Protocol):
            @property
            def prop(self) -> T: ...

        class Parent(AChild, Protocol):
            def with_a(self) -> Annotated[AChild, qual('a')]: ...

        def parent_factory(
            _prop: Annotated[T, qual('a') | qual()],
        ) -> Parent: ...

        registry = Registry()
        rules = default_rules()

        factory = compile(parent_factory, registry.build(), rules)
        t = T()
        parent = factory(t)

        assert parent.prop is parent.with_a().prop

    def test_constructed_value_with_dependencies_shared(self) -> None:
        """A constructor with dependencies should also share its result
        across qualifier contexts when all dependencies resolve to the
        same values."""

        @final
        class Dep:
            pass

        @final
        class T:
            def __init__(self, dep: Dep) -> None:
                self.dep = dep

        class AChild(Protocol):
            @property
            def prop(self) -> T: ...

        class Parent(AChild, Protocol):
            def with_a(self) -> Annotated[AChild, qual('a')]: ...

        def parent_factory() -> Parent: ...

        registry = (
            Registry()
            .register(Dep, qualifiers=qual('a') | qual())(Dep)
            .register(T, qualifiers=qual('a') | qual())(T)
        )
        rules = default_rules()

        factory = compile(parent_factory, registry.build(), rules)
        parent = factory()

        assert parent.prop is parent.with_a().prop
        assert parent.prop.dep is parent.with_a().prop.dep

    def test_transition_target_is_part_of_constructor_cache_identity(self) -> None:
        """Constructors that capture transitions must not share cache entries
        when the captured transitions execute different child targets.
        """

        class Good:
            pass

        class Bad:
            pass

        class Source[T](Protocol):
            def get(self) -> T: ...

        class Holder[T]:
            def __init__(self, source: Source[T]) -> None:
                self.source: Source[T] = source

            def get(self) -> T:
                return self.source.get()

        def make_holder[T](source: Source[T]) -> Holder[T]:
            return Holder(source)

        class Root(Protocol):
            @property
            def good_holder(self) -> Holder[Good]: ...

            @property
            def bad_holder(self) -> Holder[Bad]: ...

        registry = (
            Registry()
            .register(Good)(Good)
            .register(Bad)(Bad)
            .register_factory(make_holder)
        )
        rules = default_rules()

        root = compile(Root, registry.build(), rules)

        assert isinstance(root.good_holder.get(), Good)
        assert isinstance(root.bad_holder.get(), Bad)
        assert root.good_holder is not root.bad_holder

    def test_transition_implementation_params_are_part_of_cache_identity(self) -> None:
        """Implementation params are observable when a cached constructor
        captures a transition.
        """

        class Child:
            pass

        @final
        class Audit:
            def __init__(self, label: str) -> None:
                self.label: str = label

        class Source(Protocol):
            def get(self) -> Child: ...

        class Holder[T: Source]:
            def __init__(self, source: T) -> None:
                self.source: T = source

            def get(self) -> object:
                return self.source.get()

        def make_holder[T: Source](source: T) -> Holder[T]:
            return Holder(source)

        def make_audit_a() -> Annotated[Audit, qual('a')]:
            return Audit('a')

        def make_audit_b() -> Annotated[Audit, qual('b')]:
            return Audit('b')

        calls: list[str] = []

        def record_audit(audit: Audit) -> None:
            calls.append(audit.label)

        class Root(Protocol):
            @property
            def a_holder(self) -> Holder[Annotated[Source, qual('a')]]: ...

            @property
            def b_holder(self) -> Holder[Annotated[Source, qual('b')]]: ...

        method_registry = Registry().register_method(Source, Source.get)(record_audit)
        registry = (
            Registry()
            .register(Child, qualifiers=qual('a') | qual('b'))(Child)
            .register_factory(make_holder)
            .register_factory(make_audit_a)
            .register_factory(make_audit_b)
            .include(method_registry, qualifiers=qual('a'))
            .include(method_registry, qualifiers=qual('b'))
        )
        rules = default_rules()

        root = compile(Root, registry.build(), rules)

        assert root.a_holder is not root.b_holder
        assert isinstance(root.a_holder.get(), Child)
        assert isinstance(root.b_holder.get(), Child)
        assert calls == ['a', 'b']

    def test_transition_implementation_can_access_lazy_ref_param(self) -> None:
        from inlay import LazyRef

        @final
        class Dep:
            pass

        class Child:
            pass

        class Source(Protocol):
            def get(self) -> Child: ...

        class Root(Protocol):
            @property
            def source(self) -> Source: ...

        seen: list[Dep] = []

        def record_dep(dep: LazyRef[Dep]) -> None:
            seen.append(dep.get())

        registry = (
            Registry()
            .register(Dep)(Dep)
            .register(Child)(Child)
            .register_method(Source, Source.get)(record_dep)
        )
        rules = default_rules()

        root = compile(Root, registry.build(), rules)

        assert isinstance(root.source.get(), Child)
        assert len(seen) == 1
        assert isinstance(seen[0], Dep)


class TestRuntimeResourceOwnership:
    def test_protocol_members_are_materialized_lazily(self) -> None:
        @final
        class A:
            pass

        @final
        class B:
            pass

        calls: list[str] = []

        def make_a() -> A:
            calls.append('a')
            return A()

        def make_b() -> B:
            calls.append('b')
            return B()

        class Root(Protocol):
            @property
            def a(self) -> A: ...

            @property
            def b(self) -> B: ...

        registry = Registry().register(A)(make_a).register(B)(make_b)
        rules = default_rules()

        root = compile(Root, registry.build(), rules)

        assert calls == []
        a = root.a
        assert isinstance(a, A)
        assert root.a is a
        assert calls == ['a']
        assert isinstance(root.b, B)
        assert calls == ['a', 'b']

    def test_method_member_access_preserves_identity(self) -> None:
        class Child:
            pass

        class Root(Protocol):
            def child(self) -> Child: ...

        registry = Registry().register(Child)(Child)
        rules = default_rules()

        root = compile(Root, registry.build(), rules)
        child = root.child

        assert root.child is child

    def test_extracted_sibling_transitions_share_root_bound_cache(self) -> None:
        import gc

        @final
        class Shared:
            pass

        calls: list[Shared] = []

        def make_shared() -> Shared:
            value = Shared()
            calls.append(value)
            return value

        class Child(Protocol):
            @property
            def value(self) -> Shared: ...

        class Root(Protocol):
            def with_a(self) -> Child: ...

            def with_b(self) -> Child: ...

        registry = Registry().register(Shared)(make_shared)
        rules = default_rules()

        root = compile(Root, registry.build(), rules)
        with_a = root.with_a
        with_b = root.with_b
        del root
        _ = gc.collect()

        a_value = with_a().value
        b_value = with_b().value

        assert a_value is b_value
        assert calls == [a_value]

    def test_factory_calls_do_not_share_source_free_cache_slots(self) -> None:
        @final
        class Shared:
            pass

        class Root(Protocol):
            @property
            def value(self) -> Shared: ...

        def factory() -> Root: ...

        registry = Registry().register(Shared)(Shared)
        rules = default_rules()

        compiled = compile(factory, registry.build(), rules)
        first = compiled()
        second = compiled()

        assert first.value is not second.value


class TestSourceCentricCaching:
    def test_transition_source_dependency_rebuilds_optional_constructor(self) -> None:
        calls: list[int | None] = []

        @final
        class A:
            def __init__(self, value: None | int) -> None:
                self.value = value

        def make_a(a: None | int) -> A:
            calls.append(a)
            return A(a)

        class Child(Protocol):
            @property
            def value(self) -> A: ...

        class Root(Child, Protocol):
            def with_a(self, a: int) -> Child: ...

        registry = Registry().register(A)(make_a)
        rules = default_rules()

        root = compile(Root, registry.build(), rules)
        root_value = root.value
        child = root.with_a(7)
        child_value = child.value

        assert root_value.value is None
        assert child_value.value == 7
        assert root_value is not child_value
        assert calls == [None, 7]

    def test_transition_callable_param_is_visible_to_child_constructor(self) -> None:
        class UsesCallback:
            def __init__(self, callback: Callable[[], int]) -> None:
                self.callback: Callable[[], int] = callback

        class Child(Protocol):
            @property
            def value(self) -> UsesCallback: ...

        class Root(Protocol):
            def with_callback(self, callback: Callable[[], int]) -> Child: ...

        registry = Registry().register(UsesCallback)(UsesCallback)
        rules = default_rules()
        root = compile(Root, registry.build(), rules)

        def callback() -> int:
            return 42

        child = root.with_callback(callback)

        assert child.value.callback is callback
        assert child.value.callback() == 42

    def test_lazy_ref_target_source_dependency_rebuilds_constructor(self) -> None:
        from inlay import LazyRef

        @final
        class Tenant:
            pass

        @final
        class B:
            def __init__(self, tenant: Tenant) -> None:
                self.tenant: Tenant = tenant

        @final
        class A:
            def __init__(self, b: LazyRef[B]) -> None:
                self.b: LazyRef[B] = b

        class Child(Protocol):
            @property
            def a(self) -> A: ...

        class Root(Child, Protocol):
            def with_tenant(self, tenant: Tenant) -> Child: ...

        def factory(tenant: Tenant) -> Root:
            raise NotImplementedError(tenant)

        registry = Registry().register(A)(A).register(B)(B)
        rules = default_rules()
        compiled_factory = compile(factory, registry.build(), rules)
        root_tenant = Tenant()
        child_tenant = Tenant()
        root = compiled_factory(root_tenant)

        root_a = root.a
        child_a = root.with_tenant(child_tenant).a

        assert root_a is not child_a
        assert root_a.b.get().tenant is root_tenant
        assert child_a.b.get().tenant is child_tenant

    def test_lazy_ref_cells_are_fresh_but_constructor_target_is_cached(self) -> None:
        from inlay import LazyRef

        calls: list[object] = []

        @final
        class A:
            pass

        @final
        class X:
            def __init__(self, value: LazyRef[A]) -> None:
                self.value: LazyRef[A] = value

        @final
        class Y:
            def __init__(self, value: LazyRef[A]) -> None:
                self.value: LazyRef[A] = value

        def make_a() -> A:
            value = A()
            calls.append(value)
            return value

        class Root(Protocol):
            @property
            def x(self) -> X: ...

            @property
            def y(self) -> Y: ...

        registry = Registry().register(A)(make_a).register(X)(X).register(Y)(Y)
        rules = default_rules()
        root = compile(Root, registry.build(), rules)

        assert root.x.value is not root.y.value
        assert root.x.value.get() is root.y.value.get()
        assert calls == [root.x.value.get()]

    def test_lazy_refs_created_while_binding_are_bound(self) -> None:
        from inlay import LazyRef

        @final
        class C:
            pass

        @final
        class B:
            def __init__(self, c: LazyRef[C]) -> None:
                self.c: LazyRef[C] = c

        @final
        class A:
            def __init__(self, b: LazyRef[B]) -> None:
                self.b: LazyRef[B] = b

        class Root(Protocol):
            @property
            def a(self) -> A: ...

        registry = Registry().register(A)(A).register(B)(B).register(C)(C)
        rules = default_rules()
        root = compile(Root, registry.build(), rules)

        assert isinstance(root.a.b.get().c.get(), C)

    def test_root_context_manager_keeps_parent_scope_until_enter(self) -> None:
        import gc

        @final
        class Service:
            pass

        class Child(Protocol):
            @property
            def service(self) -> Service: ...

        def factory() -> AbstractContextManager[Child]: ...

        registry = Registry().register(Service)(Service)
        rules = default_rules()
        manager = compile(factory, registry.build(), rules)()

        _ = gc.collect()

        with manager as child:
            assert isinstance(child.service, Service)

    def test_root_awaitable_keeps_parent_scope_until_await(self) -> None:
        import gc

        import anyio

        @final
        class Service:
            pass

        class Child(Protocol):
            @property
            def service(self) -> Service: ...

        def factory() -> Awaitable[Child]: ...

        registry = Registry().register(Service)(Service)
        rules = default_rules()
        awaitable = compile(factory, registry.build(), rules)()

        _ = gc.collect()

        async def run() -> None:
            child = await awaitable
            assert isinstance(child.service, Service)

        anyio.run(run)

    def test_awaitable_transition_awaits_method_implementation(self) -> None:
        import anyio

        class State(TypedDict):
            value: int

        class Child(Protocol):
            @property
            def value(self) -> int: ...

        class Root(Protocol):
            def load(self) -> Awaitable[Child]: ...

        async def load_impl() -> State:
            return {'value': 7}

        registry = Registry().register_method(Root, Root.load)(load_impl)
        rules = default_rules()
        root = compile(Root, registry.build(), rules)

        async def run() -> None:
            child = await root.load()
            assert child.value == 7

        anyio.run(run)

    def test_root_async_context_manager_keeps_parent_scope_until_enter(self) -> None:
        import gc

        import anyio

        @final
        class Service:
            pass

        class Child(Protocol):
            @property
            def service(self) -> Service: ...

        def factory() -> AbstractAsyncContextManager[Child]: ...

        registry = Registry().register(Service)(Service)
        rules = default_rules()
        manager = compile(factory, registry.build(), rules)()

        _ = gc.collect()

        async def run() -> None:
            async with manager as child:
                assert isinstance(child.service, Service)

        anyio.run(run)

    def test_context_manager_transition_exits_if_child_setup_fails(self) -> None:
        @final
        class Service:
            pass

        class ChildState(TypedDict):
            service: Service

        class Child(Protocol):
            @property
            def service(self) -> Service: ...

        class Root(Protocol):
            def open(self) -> AbstractContextManager[Child]: ...

        events: list[object] = []

        @final
        class Manager:
            def __enter__(self) -> ChildState:
                events.append('enter')
                return {'service': Service()}

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: TracebackType | None,
            ) -> bool:
                events.append((
                    'exit',
                    exc_type,
                    str(exc_val),
                    exc_tb is not None,
                ))
                return False

        def open_impl() -> AbstractContextManager[ChildState]:
            return Manager()

        def fail_impl() -> None:
            events.append('impl')
            raise RuntimeError('impl failed')

        registry = (
            Registry()
            .register_method(Root, Root.open)(open_impl)
            .register_method(Root, Root.open)(fail_impl)
        )
        rules = default_rules()
        root = compile(Root, registry.build(), rules)

        with pytest.raises(RuntimeError, match='impl failed'):
            with root.open():
                raise AssertionError('body should not run')

        assert events == [
            'enter',
            'impl',
            ('exit', RuntimeError, 'impl failed', True),
        ]

    def test_context_manager_enter_suppression_does_not_run_body(self) -> None:
        class Root(Protocol):
            def open(self) -> AbstractContextManager[None]: ...

        events: list[object] = []

        @final
        class OuterManager:
            def __enter__(self) -> None:
                events.append('outer_enter')

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: TracebackType | None,
            ) -> bool:
                events.append((
                    'outer_exit',
                    exc_type,
                    str(exc_val),
                    exc_tb is not None,
                ))
                return True

        @final
        class InnerManager:
            def __enter__(self) -> None:
                events.append('inner_enter')
                raise RuntimeError('enter failed')

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: TracebackType | None,
            ) -> bool:
                events.append('inner_exit')
                return False

        def outer_impl() -> AbstractContextManager[None]:
            return OuterManager()

        def inner_impl() -> AbstractContextManager[None]:
            return InnerManager()

        registry = (
            Registry()
            .register_method(Root, Root.open)(outer_impl)
            .register_method(Root, Root.open)(inner_impl)
        )
        root = compile(Root, registry.build(), default_rules())

        with pytest.raises(
            RuntimeError,
            match='context manager enter did not produce a value',
        ):
            with root.open():
                events.append('body')

        assert events == [
            'outer_enter',
            'inner_enter',
            ('outer_exit', RuntimeError, 'enter failed', True),
        ]

    def test_async_context_manager_transition_exits_if_child_setup_fails(
        self,
    ) -> None:
        @final
        class Service:
            pass

        class ChildState(TypedDict):
            service: Service

        class Child(Protocol):
            @property
            def service(self) -> Service: ...

        class Root(Protocol):
            def open(self) -> AbstractAsyncContextManager[Child]: ...

        events: list[object] = []

        @final
        class Manager:
            async def __aenter__(self) -> ChildState:
                events.append('aenter')
                return {'service': Service()}

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: TracebackType | None,
            ) -> bool:
                events.append((
                    'aexit',
                    exc_type,
                    str(exc_val),
                    exc_tb is not None,
                ))
                return False

        def open_impl() -> AbstractAsyncContextManager[ChildState]:
            return Manager()

        def fail_impl() -> None:
            events.append('impl')
            raise RuntimeError('impl failed')

        registry = (
            Registry()
            .register_method(Root, Root.open)(open_impl)
            .register_method(Root, Root.open)(fail_impl)
        )
        rules = default_rules()
        root = compile(Root, registry.build(), rules)

        async def run() -> None:
            with pytest.raises(RuntimeError, match='impl failed'):
                async with root.open():
                    raise AssertionError('body should not run')

        anyio.run(run)

        assert events == [
            'aenter',
            'impl',
            ('aexit', RuntimeError, 'impl failed', True),
        ]

    def test_async_context_manager_enter_suppression_does_not_run_body(
        self,
    ) -> None:
        class Root(Protocol):
            def open(self) -> AbstractAsyncContextManager[None]: ...

        events: list[object] = []

        @final
        class OuterManager:
            async def __aenter__(self) -> None:
                events.append('outer_enter')

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: TracebackType | None,
            ) -> bool:
                events.append((
                    'outer_exit',
                    exc_type,
                    str(exc_val),
                    exc_tb is not None,
                ))
                return True

        @final
        class InnerManager:
            async def __aenter__(self) -> None:
                events.append('inner_enter')
                raise RuntimeError('enter failed')

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: TracebackType | None,
            ) -> bool:
                events.append('inner_exit')
                return False

        def outer_impl() -> AbstractAsyncContextManager[None]:
            return OuterManager()

        def inner_impl() -> AbstractAsyncContextManager[None]:
            return InnerManager()

        registry = (
            Registry()
            .register_method(Root, Root.open)(outer_impl)
            .register_method(Root, Root.open)(inner_impl)
        )
        root = compile(Root, registry.build(), default_rules())

        async def run() -> None:
            with pytest.raises(
                RuntimeError,
                match='context manager enter did not produce a value',
            ):
                async with root.open():
                    events.append('body')

        anyio.run(run)

        assert events == [
            'outer_enter',
            'inner_enter',
            ('outer_exit', RuntimeError, 'enter failed', True),
        ]

    def test_async_enter_close_continues_existing_cleanup_drainer(self) -> None:
        @final
        class Service:
            pass

        class Child(Protocol):
            @property
            def service(self) -> Service: ...

        class Root(Protocol):
            def open(self) -> AbstractAsyncContextManager[Child]: ...

        events: list[object] = []

        @final
        class OuterManager:
            async def __aenter__(self) -> None:
                events.append('outer_enter')

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: TracebackType | None,
            ) -> bool:
                events.append((
                    'outer_exit',
                    exc_type,
                    exc_val is not None,
                    exc_tb is not None,
                ))
                return False

        @final
        class InnerManager:
            async def __aenter__(self) -> None:
                events.append('inner_enter')

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: TracebackType | None,
            ) -> bool:
                events.append(('inner_exit_start', exc_type))
                try:
                    await _YieldOnce()
                finally:
                    events.append('inner_exit_closed')
                return False

        def outer_impl() -> AbstractAsyncContextManager[None]:
            return OuterManager()

        def inner_impl() -> AbstractAsyncContextManager[None]:
            return InnerManager()

        def fail_impl() -> None:
            events.append('fail')
            raise RuntimeError('setup failed')

        registry = (
            Registry()
            .register(Service)(Service)
            .register_method(Root, Root.open)(outer_impl)
            .register_method(Root, Root.open)(inner_impl)
            .register_method(Root, Root.open)(fail_impl)
        )
        root = compile(Root, registry.build(), default_rules())

        enter_awaitable = root.open().__aenter__()
        iterator = enter_awaitable.__await__()

        assert next(iterator) is None
        enter_awaitable.close()

        assert events == [
            'outer_enter',
            'inner_enter',
            'fail',
            ('inner_exit_start', RuntimeError),
            'inner_exit_closed',
            ('outer_exit', GeneratorExit, True, False),
        ]

    def test_async_exit_close_continues_remaining_drainer(self) -> None:
        @final
        class Service:
            pass

        class Child(Protocol):
            @property
            def service(self) -> Service: ...

        class Root(Protocol):
            def open(self) -> AbstractAsyncContextManager[Child]: ...

        events: list[object] = []

        @final
        class OuterManager:
            async def __aenter__(self) -> None:
                events.append('outer_enter')

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: TracebackType | None,
            ) -> bool:
                events.append((
                    'outer_exit',
                    exc_type,
                    exc_val is not None,
                    exc_tb is not None,
                ))
                return False

        @final
        class InnerManager:
            async def __aenter__(self) -> None:
                events.append('inner_enter')

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: TracebackType | None,
            ) -> bool:
                events.append(('inner_exit_start', exc_type))
                try:
                    await _YieldOnce()
                finally:
                    events.append('inner_exit_closed')
                return False

        def outer_impl() -> AbstractAsyncContextManager[None]:
            return OuterManager()

        def inner_impl() -> AbstractAsyncContextManager[None]:
            return InnerManager()

        registry = (
            Registry()
            .register(Service)(Service)
            .register_method(Root, Root.open)(outer_impl)
            .register_method(Root, Root.open)(inner_impl)
        )
        root = compile(Root, registry.build(), default_rules())

        context = root.open()
        enter_iterator = context.__aenter__().__await__()
        with pytest.raises(StopIteration):
            next(enter_iterator)

        exit_awaitable = context.__aexit__(
            ValueError,
            ValueError('body failed'),
            None,
        )
        exit_iterator = exit_awaitable.__await__()

        assert next(exit_iterator) is None
        exit_awaitable.close()

        assert events == [
            'outer_enter',
            'inner_enter',
            ('inner_exit_start', ValueError),
            'inner_exit_closed',
            ('outer_exit', GeneratorExit, True, False),
        ]

    def test_factory_arg_attribute_stays_live(self) -> None:
        # given
        class State(TypedDict):
            value: int

        class Root(Protocol):
            value: int

        registry = Registry()
        rules = default_rules()

        def factory(_state: State) -> Root: ...

        compiled_factory = compile(factory, registry.build(), rules)
        state: State = {'value': 1}
        root = compiled_factory(state)

        # when
        state['value'] = 2

        # then
        assert root.value == 2

    def test_explicit_transition_result_attribute_stays_live(self) -> None:
        # given
        class State(TypedDict):
            value: int

        @final
        class WithStateImpl:
            def __init__(self, state: State) -> None:
                self._state = state

            def with_state(self) -> State:
                return self._state

        class Child(Protocol):
            value: int

        class Root(Protocol):
            def with_state(self) -> Child: ...

        registry = Registry().register_method(Root, Root.with_state)(WithStateImpl)
        rules = default_rules()

        def factory(_state: State) -> Root: ...

        compiled_factory = compile(factory, registry.build(), rules)
        state: State = {'value': 1}
        root = compiled_factory(state)
        child = root.with_state()

        # when
        state['value'] = 2

        # then
        assert child.value == 2


@final
class _StackingService:
    pass


class _StackingState(TypedDict):
    service: _StackingService


class _StackingChild(Protocol):
    @property
    def service(self) -> _StackingService: ...


def _build_stacking_impl(
    kind: str,
    events: list[str],
    label: str,
    *,
    returns_state: bool,
) -> Callable[..., object]:
    if kind == 'plain':
        if returns_state:

            def plain_state_impl() -> _StackingState:
                events.append(label)
                return {'service': _StackingService()}

            return plain_state_impl

        def plain_none_impl() -> None:
            events.append(label)

        return plain_none_impl

    if kind == 'cm':
        if returns_state:

            @contextmanager
            def cm_state_impl() -> Generator[_StackingState]:
                events.append(f'{label}:enter')
                yield {'service': _StackingService()}
                events.append(f'{label}:exit')

            return cm_state_impl

        @contextmanager
        def cm_none_impl() -> Generator[None]:
            events.append(f'{label}:enter')
            yield None
            events.append(f'{label}:exit')

        return cm_none_impl

    if kind == 'awaitable':
        if returns_state:

            async def awaitable_state_impl() -> _StackingState:
                events.append(label)
                return {'service': _StackingService()}

            return awaitable_state_impl

        async def awaitable_none_impl() -> None:
            events.append(label)

        return awaitable_none_impl

    if kind == 'acm':
        if returns_state:

            @asynccontextmanager
            async def acm_state_impl() -> AsyncGenerator[_StackingState]:
                events.append(f'{label}:enter')
                yield {'service': _StackingService()}
                events.append(f'{label}:exit')

            return acm_state_impl

        @asynccontextmanager
        async def acm_none_impl() -> AsyncGenerator[None]:
            events.append(f'{label}:enter')
            yield None
            events.append(f'{label}:exit')

        return acm_none_impl

    raise ValueError(kind)


def _register_stack(
    builder: Registry,
    protocol: type,
    method: Callable[..., object],
    impls: list[Callable[..., object]],
) -> Registry:
    for impl in impls:
        builder = builder.register_method(protocol, method)(impl)
    return builder


def _expected_enter_events(stack: list[str]) -> list[str]:
    return [
        f'{kind}{i}:enter' if kind in {'cm', 'acm'} else f'{kind}{i}'
        for i, kind in enumerate(stack)
    ]


class TestMethodImplWrapperRuntimeStacking:
    """Stacking compatible method implementations of various wrapper kinds in
    different orders should execute correctly at runtime.

    Each impl appends to a shared event log when it runs. The LAST impl in
    the stack returns a _StackingState TypedDict that the child uses; earlier
    impls return None and only contribute side effects.
    """

    @pytest.mark.parametrize(
        'stack',
        [
            ['plain'],
            ['plain', 'plain'],
            ['plain', 'plain', 'plain'],
        ],
    )
    def test_plain_method_stacking(self, stack: list[str]) -> None:
        events: list[str] = []
        impls = [
            _build_stacking_impl(
                kind, events, f'{kind}{i}', returns_state=(i == len(stack) - 1)
            )
            for i, kind in enumerate(stack)
        ]

        class Root(Protocol):
            def load(self) -> _StackingChild: ...

        builder = _register_stack(Registry(), Root, Root.load, impls)
        root = compile(Root, builder.build(), default_rules())

        child = root.load()
        assert isinstance(child.service, _StackingService)
        assert events == [f'{kind}{i}' for i, kind in enumerate(stack)]

    @pytest.mark.parametrize(
        'stack',
        [
            ['plain'],
            ['cm'],
            ['plain', 'cm'],
            ['cm', 'plain'],
            ['cm', 'cm'],
            ['plain', 'cm', 'plain'],
        ],
    )
    def test_context_manager_method_stacking(self, stack: list[str]) -> None:
        events: list[str] = []
        impls = [
            _build_stacking_impl(
                kind, events, f'{kind}{i}', returns_state=(i == len(stack) - 1)
            )
            for i, kind in enumerate(stack)
        ]

        class Root(Protocol):
            def load(self) -> AbstractContextManager[_StackingChild]: ...

        builder = _register_stack(Registry(), Root, Root.load, impls)
        root = compile(Root, builder.build(), default_rules())

        with root.load() as child:
            assert isinstance(child.service, _StackingService)

        enter_events = [event for event in events if not event.endswith(':exit')]
        assert enter_events == _expected_enter_events(stack)

    @pytest.mark.parametrize(
        'stack',
        [
            ['plain'],
            ['awaitable'],
            ['plain', 'awaitable'],
            ['awaitable', 'plain'],
            ['awaitable', 'awaitable'],
        ],
    )
    def test_awaitable_method_stacking(self, stack: list[str]) -> None:
        events: list[str] = []
        impls = [
            _build_stacking_impl(
                kind, events, f'{kind}{i}', returns_state=(i == len(stack) - 1)
            )
            for i, kind in enumerate(stack)
        ]

        class Root(Protocol):
            def load(self) -> Awaitable[_StackingChild]: ...

        builder = _register_stack(Registry(), Root, Root.load, impls)
        root = compile(Root, builder.build(), default_rules())

        async def run() -> None:
            child = await root.load()
            assert isinstance(child.service, _StackingService)

        anyio.run(run)

        assert events == [f'{kind}{i}' for i, kind in enumerate(stack)]

    @pytest.mark.parametrize(
        'stack',
        [
            ['plain'],
            ['cm'],
            ['awaitable'],
            ['acm'],
            ['plain', 'acm'],
            ['acm', 'plain'],
            ['cm', 'acm'],
            ['acm', 'cm'],
            ['awaitable', 'acm'],
            ['acm', 'awaitable'],
            ['plain', 'cm', 'awaitable', 'acm'],
        ],
    )
    def test_async_context_manager_method_stacking(self, stack: list[str]) -> None:
        events: list[str] = []
        impls = [
            _build_stacking_impl(
                kind, events, f'{kind}{i}', returns_state=(i == len(stack) - 1)
            )
            for i, kind in enumerate(stack)
        ]

        class Root(Protocol):
            def load(self) -> AbstractAsyncContextManager[_StackingChild]: ...

        builder = _register_stack(Registry(), Root, Root.load, impls)
        root = compile(Root, builder.build(), default_rules())

        async def run() -> None:
            async with root.load() as child:
                assert isinstance(child.service, _StackingService)

        anyio.run(run)

        enter_events = [event for event in events if not event.endswith(':exit')]
        assert enter_events == _expected_enter_events(stack)
