import pickle
from typing import Annotated, Protocol, TypedDict

import pytest

from inlay import Registry, compile, qual


class _FactoryRoot(Protocol):
    @property
    def value(self) -> int: ...


def _make_factory_root(value: int) -> _FactoryRoot: ...


class _Service:
    pass


class _HasService(Protocol):
    @property
    def service(self) -> _Service: ...


_cached_service_calls = 0


class _CachedService:
    pass


def _make_cached_service() -> _CachedService:
    global _cached_service_calls
    _cached_service_calls += 1
    return _CachedService()


class _HasCachedService(Protocol):
    @property
    def service(self) -> _CachedService: ...


_unpicklable_service_calls = 0


class _UnpicklableService:
    def __reduce__(self) -> object:
        raise TypeError('sentinel unpicklable service')


def _make_unpicklable_service() -> _UnpicklableService:
    global _unpicklable_service_calls
    _unpicklable_service_calls += 1
    return _UnpicklableService()


class _HasUnpicklableService(Protocol):
    @property
    def service(self) -> _UnpicklableService: ...


_pickle_payload_getstate_calls = 0


class _PicklePayload:
    def __init__(self, value: str) -> None:
        self.value = value

    def __getstate__(self) -> dict[str, str]:
        global _pickle_payload_getstate_calls
        _pickle_payload_getstate_calls += 1
        return {'value': self.value}

    def __setstate__(self, state: dict[str, str]) -> None:
        self.value = state['value']


class _HasPayload(Protocol):
    @property
    def payload(self) -> _PicklePayload: ...


def _make_payload_root(payload: _PicklePayload) -> _HasPayload: ...


class _PayloadPairRoot(Protocol):
    @property
    def left(self) -> _PicklePayload: ...

    @property
    def right(self) -> _PicklePayload: ...


def _make_payload_pair_root(payload: _PicklePayload) -> _PayloadPairRoot: ...


class _TransitionChild(Protocol):
    @property
    def seed(self) -> str: ...


class _TransitionRoot(Protocol):
    def child(self) -> _TransitionChild: ...


def _make_transition_root(seed: str) -> _TransitionRoot: ...


class _QualifiedService:
    pass


class _HasQualifiedService(Protocol):
    @property
    def service(self) -> Annotated[_QualifiedService, qual('q')]: ...


class _PickleDict(TypedDict):
    value: int


def _make_pickle_dict(value: int) -> _PickleDict: ...


def _roundtrip[T](obj: T) -> T:
    return pickle.loads(pickle.dumps(obj))


def test_qualifier_round_trips() -> None:
    assert _roundtrip(qual('a') | qual('b')) == qual('a') | qual('b')
    assert _roundtrip(qual('a') & qual('b')) == qual('a') & qual('b')
    assert _roundtrip(qual.ANY) == qual.ANY


def test_compiled_factory_round_trips() -> None:
    factory = compile(_make_factory_root, Registry().build())

    restored = _roundtrip(factory)

    assert restored(3).value == 3


def test_unmaterialized_compiled_context_round_trips() -> None:
    registry = Registry().register(_Service)(_Service)
    root = compile(_HasService, registry.build())

    restored = _roundtrip(root)

    assert isinstance(restored.service, _Service)


def test_materialized_context_does_not_rerun_constructor_after_roundtrip() -> None:
    global _cached_service_calls
    _cached_service_calls = 0
    registry = Registry().register(_CachedService)(_make_cached_service)
    root = compile(_HasCachedService, registry.build())

    assert isinstance(root.service, _CachedService)
    assert _cached_service_calls == 1

    restored = _roundtrip(root)

    assert isinstance(restored.service, _CachedService)
    assert _cached_service_calls == 1


def test_unmaterialized_proxy_member_is_not_pickled() -> None:
    global _unpicklable_service_calls
    _unpicklable_service_calls = 0
    registry = Registry().register(_UnpicklableService)(_make_unpicklable_service)
    root = compile(_HasUnpicklableService, registry.build())

    restored = _roundtrip(root)

    assert _unpicklable_service_calls == 0
    assert isinstance(restored.service, _UnpicklableService)
    assert _unpicklable_service_calls == 1


def test_materialized_proxy_member_is_pickled_by_python() -> None:
    global _unpicklable_service_calls
    _unpicklable_service_calls = 0
    registry = Registry().register(_UnpicklableService)(_make_unpicklable_service)
    root = compile(_HasUnpicklableService, registry.build())

    assert isinstance(root.service, _UnpicklableService)
    assert _unpicklable_service_calls == 1

    with pytest.raises(TypeError, match='sentinel unpicklable service'):
        pickle.dumps(root)


def test_source_python_ref_is_pickled_by_python() -> None:
    global _pickle_payload_getstate_calls
    _pickle_payload_getstate_calls = 0
    payload = _PicklePayload('kept')
    factory = compile(_make_payload_root, Registry().build())
    root = factory(payload)

    restored = _roundtrip(root)

    assert _pickle_payload_getstate_calls == 1
    assert restored.payload.value == 'kept'
    assert restored.payload is not payload


def test_duplicate_python_refs_preserve_pickle_aliasing() -> None:
    global _pickle_payload_getstate_calls
    _pickle_payload_getstate_calls = 0
    payload = _PicklePayload('shared')
    factory = compile(_make_payload_pair_root, Registry().build())
    root = factory(payload)

    assert root.left is payload
    restored = _roundtrip(root)

    assert _pickle_payload_getstate_calls == 1
    assert restored.left is restored.right


def test_transition_captures_parent_resources_after_roundtrip() -> None:
    factory = compile(_make_transition_root, Registry().build())
    root = factory('kept')

    restored_child = _roundtrip(root.child)()

    assert restored_child.seed == 'kept'


def test_qualified_compiled_context_round_trips() -> None:
    registry = Registry().register(_QualifiedService, qualifiers=qual('q'))(
        _QualifiedService
    )
    root = compile(_HasQualifiedService, registry.build())

    restored = _roundtrip(root)

    assert isinstance(restored.service, _QualifiedService)


def test_compiled_typed_dict_round_trips() -> None:
    factory = compile(_make_pickle_dict, Registry().build())

    restored = _roundtrip(factory(7))

    assert restored['value'] == 7
