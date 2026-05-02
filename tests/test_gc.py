"""Tests that compiled contexts are garbage-collected when all Python
references are dropped.

Before the Weak scope-handle fix, every compile() and transition call
created an unreclaimable Arc cycle:

    Scope.computed → Py<ContextProxy> → Py<Transition> → Arc<ScopeHandle> → Scope

These tests verify the cycle is broken and objects are collected.
"""

import gc
import weakref
from typing import Protocol, TypedDict, final

from inlay import RegistryBuilder, compile
from inlay.default import default_rules


class Service:
    pass


class ChildConstants(TypedDict):
    value: Service


class Root(Protocol):
    @property
    def service(self) -> Service: ...

    def with_service(self, value: Service) -> Child: ...


class Child(Protocol):
    @property
    def service(self) -> Service: ...


@final
class WithServiceImpl:
    def with_service(self, value: Service) -> ChildConstants:
        return {'value': value}


class TestChildContextGarbageCollection:
    """Child contexts created by transition calls must be collected when the
    user drops all references, even while the parent context stays alive."""

    def test_child_context_is_collected(self) -> None:
        # given
        registry = (
            RegistryBuilder()
            .register(Service)(Service)
            .register_method(Root, Root.with_service)(WithServiceImpl)
        )
        rules = default_rules()

        def factory() -> Root: ...

        compiled = compile(factory, registry.build(), rules)
        root = compiled()

        # when - create a child context and track it
        child = root.with_service(Service())
        ref = weakref.ref(child)
        del child
        _ = gc.collect()

        # then
        assert ref() is None

    def test_many_child_contexts_do_not_leak(self) -> None:
        # given
        registry = (
            RegistryBuilder()
            .register(Service)(Service)
            .register_method(Root, Root.with_service)(WithServiceImpl)
        )
        rules = default_rules()

        def factory() -> Root: ...

        compiled = compile(factory, registry.build(), rules)
        root = compiled()

        # when - create many children and drop them
        refs: list[weakref.ReferenceType[object]] = []
        for _ in range(100):
            child = root.with_service(Service())
            refs.append(weakref.ref(child))
            del child

        _ = gc.collect()

        # then - all should be collected
        alive = sum(1 for r in refs if r() is not None)
        assert alive == 0


class TestRootContextGarbageCollection:
    """Root contexts returned by compile() must be collected when the user
    drops all references."""

    def test_root_context_is_collected(self) -> None:
        # given
        registry = (
            RegistryBuilder()
            .register(Service)(Service)
            .register_method(Root, Root.with_service)(WithServiceImpl)
        )
        rules = default_rules()

        def factory() -> Root: ...

        compiled = compile(factory, registry.build(), rules)
        root = compiled()
        ref = weakref.ref(root)

        # when
        del root
        _ = gc.collect()

        # then
        assert ref() is None

    def test_compiled_factory_is_collected(self) -> None:
        # given
        registry = (
            RegistryBuilder()
            .register(Service)(Service)
            .register_method(Root, Root.with_service)(WithServiceImpl)
        )
        rules = default_rules()

        def factory() -> Root: ...

        compiled = compile(factory, registry.build(), rules)

        # when
        ref = weakref.ref(compiled)
        del compiled
        _ = gc.collect()

        # then
        assert ref() is None
