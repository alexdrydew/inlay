"""Runtime tests for the compiled context factory.

These tests exercise the full path: compile -> call factory -> call transitions,
verifying that the runtime scope chain correctly propagates constants.
"""

from collections.abc import Awaitable, Callable
from typing import Annotated, Protocol, TypedDict, cast, final

import pytest

from inlay import RegistryBuilder, compile, qual
from inlay.rules import (
    RuleGraphBuilder,
    attribute_source_rule,
    auto_method_rule,
    constant_rule,
    constructor_rule,
    lazy_ref_rule,
    match_first,
    method_impl_rule,
    property_source_rule,
    protocol_rule,
    sentinel_none_rule,
    typeddict_rule,
    union_rule,
)


def _build_default_rules():
    builder = RuleGraphBuilder()

    self_ref = builder.lazy(lambda: pipeline)
    strict_ref = builder.lazy(lambda: strict_pipeline)

    method_rules = match_first(
        method_impl_rule(target_rules=self_ref, hook_param_rule=self_ref),
        auto_method_rule(target_rules=strict_ref, hook_param_rule=self_ref),
    )

    pipeline = match_first(
        sentinel_none_rule(),
        constant_rule(),
        lazy_ref_rule(resolve=self_ref),
        attribute_source_rule(resolve=self_ref),
        property_source_rule(resolve=self_ref),
        constructor_rule(param_rules=self_ref),
        union_rule(variant_rules=self_ref),
        protocol_rule(resolve=self_ref, method_rules=method_rules),
        typeddict_rule(resolve=self_ref),
        auto_method_rule(target_rules=self_ref),
    )

    strict_pipeline = match_first(
        sentinel_none_rule(),
        constant_rule(),
        lazy_ref_rule(resolve=self_ref),
        attribute_source_rule(resolve=self_ref),
        property_source_rule(resolve=self_ref),
        constructor_rule(param_rules=self_ref),
        union_rule(variant_rules=self_ref, allow_none_fallback=False),
        protocol_rule(resolve=self_ref, method_rules=method_rules),
        typeddict_rule(resolve=self_ref),
        auto_method_rule(target_rules=strict_ref),
    )

    return builder.build()


class TestAutoTransitionCallSignature:
    def test_auto_transition_rejects_invalid_call_shapes(self) -> None:
        class Child(Protocol):
            @property
            def value(self) -> int: ...

        class Root(Protocol):
            def with_child(self, value: int, /, label: str, *, flag: bool) -> Child: ...

        root = compile(Root, RegistryBuilder().build(), _build_default_rules())
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

        root = compile(Root, RegistryBuilder().build(), _build_default_rules())

        assert root.run(1, 2, 3).value == 1


class TestClassBasedMethodImplRuntime:
    """Runtime behavior of class-based method implementations.

    After the _build_method fix, method_impl matches for class-based impls
    (callable type is built from the actual method, not __init__).  This
    changes resolution from AutoMethod to Method nodes, which affects how
    the runtime creates child scopes and looks up constants.
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

        class HasUnitOfWork[T](Protocol):
            def with_write(self) -> T: ...

        class RootContext(HasUnitOfWork[WriteContext], Protocol):
            pass

        registry = (
            RegistryBuilder()
            .register(Config)(Config)
            .register_method(HasUnitOfWork, method_name='with_write')(UowTransition)
        )
        rules = _build_default_rules()

        def factory(_config: Config) -> RootContext: ...

        compiled_factory = compile(factory, registry.build(), rules)

        # when
        root = compiled_factory(Config())
        write_ctx = root.with_write()

        # then
        assert isinstance(write_ctx.transaction, Transaction)

    def test_nested_auto_method_then_class_method_impl(self) -> None:
        """Two-level nesting: auto_method transition followed by a
        class-based method_impl transition.

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

        class HasUnitOfWork[T](Protocol):
            def with_write(self) -> T: ...

        class ModuleContext(HasUnitOfWork[WriteContext], Protocol):
            @property
            def db_name(self) -> str: ...

        class HasModule(Protocol):
            def with_module(self) -> ModuleContext: ...

        class RootContext(HasModule, Protocol):
            pass

        registry = RegistryBuilder().register_method(
            HasUnitOfWork, method_name='with_write'
        )(UowTransition)
        rules = _build_default_rules()

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

        registry = RegistryBuilder().register_factory(provide_source)
        root = compile(Root, registry.build(), _build_default_rules())

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

        registry = RegistryBuilder().register_factory(provide_state)
        root = compile(Root, registry.build(), _build_default_rules())

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
            RegistryBuilder()
            .register(Executor)(Executor)
            .register(Concrete)(Concrete)
            .register_factory(provide_source)
        )
        rules = _build_default_rules()

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
            RegistryBuilder()
            .register(Executor)(Executor)
            .register(WriteCtx)(WriteCtx)
            .register_method(WriteTransition, method_name='with_write')(
                WriteTransitionImpl
            )
        )
        rules = _build_default_rules()

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

        registry = RegistryBuilder().register(A)(make).register(B)(make)
        rules = _build_default_rules()

        root = compile(Root, registry.build(), rules)

        assert root.a is root.b
        assert len(calls) == 1

    def test_auto_method_transition_shares_constructed_value(self) -> None:
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

        registry = RegistryBuilder().register(T, qualifiers=qual('a') | qual())(make_t)
        rules = _build_default_rules()

        factory = compile(parent_factory, registry.build(), rules)
        parent = factory()

        assert parent.prop is parent.with_a().prop

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

        registry = RegistryBuilder().register(
            T, qualifiers=qual('a') | qual('b') | qual()
        )(make_t)
        rules = _build_default_rules()

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

        registry = RegistryBuilder()
        rules = _build_default_rules()

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
            RegistryBuilder()
            .register(Dep, qualifiers=qual('a') | qual())(Dep)
            .register(T, qualifiers=qual('a') | qual())(T)
        )
        rules = _build_default_rules()

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
            RegistryBuilder()
            .register(Good)(Good)
            .register(Bad)(Bad)
            .register_factory(make_holder)
        )
        rules = _build_default_rules()

        root = compile(Root, registry.build(), rules)

        assert isinstance(root.good_holder.get(), Good)
        assert isinstance(root.bad_holder.get(), Bad)
        assert root.good_holder is not root.bad_holder

    def test_transition_hook_params_are_part_of_cache_identity(self) -> None:
        """Hook params are observable when a cached constructor captures a
        transition.
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

        def record_hook(audit: Audit) -> None:
            calls.append(audit.label)

        class Root(Protocol):
            @property
            def a_holder(self) -> Holder[Annotated[Source, qual('a')]]: ...

            @property
            def b_holder(self) -> Holder[Annotated[Source, qual('b')]]: ...

        registry = (
            RegistryBuilder()
            .register(Child, qualifiers=qual('a') | qual('b'))(Child)
            .register_factory(make_holder)
            .register_factory(make_audit_a)
            .register_factory(make_audit_b)
            .register_method_hook(Source, method_name='get', qualifiers=qual('a'))(
                record_hook
            )
            .register_method_hook(Source, method_name='get', qualifiers=qual('b'))(
                record_hook
            )
        )
        rules = _build_default_rules()

        root = compile(Root, registry.build(), rules)

        assert root.a_holder is not root.b_holder
        assert isinstance(root.a_holder.get(), Child)
        assert isinstance(root.b_holder.get(), Child)
        assert calls == ['a', 'b']


class TestSourceCentricCaching:
    def test_transition_source_dependency_rebuilds_optional_constructor(self) -> None:
        calls: list[int | None] = []

        @final
        class A:
            def __init__(self, value: int | None) -> None:
                self.value = value

        def make_a(a: int | None) -> A:
            calls.append(a)
            return A(a)

        class Child(Protocol):
            @property
            def value(self) -> A: ...

        class Root(Child, Protocol):
            def with_a(self, a: int) -> Child: ...

        registry = RegistryBuilder().register(A)(make_a)
        rules = _build_default_rules()

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

        registry = RegistryBuilder().register(UsesCallback)(UsesCallback)
        rules = _build_default_rules()
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

        registry = RegistryBuilder().register(A)(A).register(B)(B)
        rules = _build_default_rules()
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

        registry = RegistryBuilder().register(A)(make_a).register(X)(X).register(Y)(Y)
        rules = _build_default_rules()
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

        registry = RegistryBuilder().register(A)(A).register(B)(B).register(C)(C)
        rules = _build_default_rules()
        root = compile(Root, registry.build(), rules)

        assert isinstance(root.a.b.get().c.get(), C)

    def test_root_context_manager_keeps_parent_scope_until_enter(self) -> None:
        import gc
        from contextlib import AbstractContextManager

        @final
        class Service:
            pass

        class Child(Protocol):
            @property
            def service(self) -> Service: ...

        def factory() -> AbstractContextManager[Child]: ...

        registry = RegistryBuilder().register(Service)(Service)
        rules = _build_default_rules()
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

        registry = RegistryBuilder().register(Service)(Service)
        rules = _build_default_rules()
        awaitable = compile(factory, registry.build(), rules)()

        _ = gc.collect()

        async def run() -> None:
            child = await awaitable
            assert isinstance(child.service, Service)

        anyio.run(run)

    def test_root_async_context_manager_keeps_parent_scope_until_enter(self) -> None:
        import gc
        from contextlib import AbstractAsyncContextManager

        import anyio

        @final
        class Service:
            pass

        class Child(Protocol):
            @property
            def service(self) -> Service: ...

        def factory() -> AbstractAsyncContextManager[Child]: ...

        registry = RegistryBuilder().register(Service)(Service)
        rules = _build_default_rules()
        manager = compile(factory, registry.build(), rules)()

        _ = gc.collect()

        async def run() -> None:
            async with manager as child:
                assert isinstance(child.service, Service)

        anyio.run(run)

    def test_factory_arg_attribute_stays_live(self) -> None:
        # given
        class State(TypedDict):
            value: int

        class Root(Protocol):
            value: int

        registry = RegistryBuilder()
        rules = _build_default_rules()

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

        registry = RegistryBuilder().register_method(Root, method_name='with_state')(
            WithStateImpl
        )
        rules = _build_default_rules()

        def factory(_state: State) -> Root: ...

        compiled_factory = compile(factory, registry.build(), rules)
        state: State = {'value': 1}
        root = compiled_factory(state)
        child = root.with_state()

        # when
        state['value'] = 2

        # then
        assert child.value == 2
