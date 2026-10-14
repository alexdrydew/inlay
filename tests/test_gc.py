"""Tests that compiled contexts are garbage-collected when all Python
references are dropped.

Before the Weak scope-handle fix, every compile() and transition call
created an unreclaimable Arc cycle:

    Scope.computed → Py<ContextProxy> → Py<Transition> → Arc<ScopeHandle> → Scope

These tests verify the cycle is broken and objects are collected.
"""

import gc
import subprocess
import sys
import weakref
from typing import Protocol, TypedDict, final

import pytest

from inlay import Registry, compile
from inlay.default import default_rules


@pytest.mark.parametrize('resource_kind', ['cached', 'source'])
@pytest.mark.parametrize('retained_owner', ['root', 'transition', 'child'])
def test_shared_resources_survive_young_collection(
    resource_kind: str, retained_owner: str
) -> None:
    # Run in a fresh process: incorrect traversal can also abort a debug Python
    # build, and must not corrupt the pytest process's GC state.
    script = """
import gc
import sys
import weakref
from typing import Protocol

from inlay import Registry, compiled

class Reader:
    def __init__(self, value: str) -> None:
        self.value = value
        self.owner: object | None = None

class RequestContext(Protocol):
    @property
    def reader(self) -> Reader: ...

class RootContext(Protocol):
    def with_request(self, request_id: int) -> RequestContext: ...

    @property
    def reader(self) -> Reader: ...

    @reader.setter
    def reader(self, value: Reader) -> None: ...

resource_kind, retained_owner = sys.argv[1:]
gc.disable()
if resource_kind == 'cached':
    @compiled(Registry())
    def make_root(value: str) -> RootContext: ...
    root = make_root('initialized')
else:
    @compiled(Registry())
    def make_root(reader: Reader) -> RootContext: ...
    root = make_root(Reader('initialized'))

transition = root.with_request
gc.collect()  # Promote owners before creating the young cycle.
if resource_kind == 'source':
    # Replace the shared source with a young object after its owners are old.
    root.reader = Reader('initialized')
context = transition(1)
reader = context.reader
reader.owner = context
reader_ref = weakref.ref(reader)
context_ref = weakref.ref(context)
del reader

if retained_owner != 'child':
    del context
if retained_owner != 'root':
    del root
if retained_owner != 'transition':
    del transition

gc.collect(0)
if retained_owner == 'root':
    reader = root.reader
elif retained_owner == 'transition':
    reader = transition(2).reader
else:
    reader = context.reader
assert reader is reader_ref()
assert reader.value == 'initialized'
assert reader.owner is context_ref()
del reader

# Removing all owners must collect the cycle too, rather than hide it from GC.
if retained_owner == 'root':
    del root
elif retained_owner == 'transition':
    del transition
else:
    del context
gc.collect()
assert reader_ref() is None
assert context_ref() is None
"""
    result = subprocess.run(
        [sys.executable, '-c', script, resource_kind, retained_owner],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
            Registry()
            .register(Service)(Service)
            .register_method(Root, Root.with_service)(WithServiceImpl)
        )
        rules = default_rules()

        def factory() -> Root: ...

        compiled = compile(factory, registry.build(rules))
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
            Registry()
            .register(Service)(Service)
            .register_method(Root, Root.with_service)(WithServiceImpl)
        )
        rules = default_rules()

        def factory() -> Root: ...

        compiled = compile(factory, registry.build(rules))
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
            Registry()
            .register(Service)(Service)
            .register_method(Root, Root.with_service)(WithServiceImpl)
        )
        rules = default_rules()

        def factory() -> Root: ...

        compiled = compile(factory, registry.build(rules))
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
            Registry()
            .register(Service)(Service)
            .register_method(Root, Root.with_service)(WithServiceImpl)
        )
        rules = default_rules()

        def factory() -> Root: ...

        compiled = compile(factory, registry.build(rules))

        # when
        ref = weakref.ref(compiled)
        del compiled
        _ = gc.collect()

        # then
        assert ref() is None
