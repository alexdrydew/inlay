import asyncio
import warnings
from collections.abc import AsyncIterator, Generator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Protocol

import pytest

from inlay import Registry, compiled

pytestmark = pytest.mark.filterwarnings('error::DeprecationWarning')


class AsyncContext(Protocol):
    async def value(self) -> str: ...


def value_iterator(caught: list[ValueError]) -> Generator[object, None, str]:
    async def provide_value() -> str:
        try:
            await asyncio.sleep(0)
        except ValueError as error:
            caught.append(error)
            return 'handled'
        return 'normal'

    @compiled(
        Registry().register_method(AsyncContext, AsyncContext.value)(provide_value)
    )
    def context() -> AsyncContext: ...

    return context().value().__await__()


@pytest.mark.parametrize('exception_arg', [ValueError('probe'), ValueError])
def test_single_argument_throw_is_warning_free(
    exception_arg: ValueError | type[ValueError],
) -> None:
    caught: list[ValueError] = []
    iterator = value_iterator(caught)
    assert iterator.send(None) is None

    with pytest.raises(StopIteration) as completed:
        iterator.throw(exception_arg)

    assert completed.value.value == 'handled'
    assert len(caught) == 1
    if isinstance(exception_arg, ValueError):
        assert caught[0] is exception_arg


@pytest.mark.parametrize('trailing_args', [(None,), (None, None)])
def test_explicit_none_arguments_still_emit_deprecation(
    trailing_args: tuple[None, ...],
) -> None:
    caught: list[ValueError] = []
    iterator = value_iterator(caught)
    assert iterator.send(None) is None
    error = ValueError('probe')

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always', DeprecationWarning)
        with pytest.raises(StopIteration) as completed:
            iterator.throw(error, *trailing_args)

    assert completed.value.value == 'handled'
    assert caught == [error]
    assert len(captured) == 1
    assert captured[0].category is DeprecationWarning
    assert 'signature of throw() is deprecated' in str(captured[0].message)


def test_three_argument_throw_preserves_exception_and_traceback() -> None:
    caught: list[ValueError] = []
    iterator = value_iterator(caught)
    assert iterator.send(None) is None
    error = ValueError('probe')
    try:
        raise error
    except ValueError:
        traceback = error.__traceback__
    assert traceback is not None

    with pytest.warns(DeprecationWarning, match='signature of throw'):
        with pytest.raises(StopIteration) as completed:
            iterator.throw(ValueError, error, traceback)

    assert completed.value.value == 'handled'
    assert caught[0] is error
    forwarded_traceback = caught[0].__traceback__
    while forwarded_traceback is not None and forwarded_traceback is not traceback:
        forwarded_traceback = forwarded_traceback.tb_next
    assert forwarded_traceback is traceback


@pytest.mark.parametrize('started', [False, True])
@pytest.mark.parametrize('args', [(), (ValueError('probe'), None, None, None)])
def test_invalid_throw_arity_does_not_consume_iterator(
    started: bool, args: tuple[object, ...]
) -> None:
    iterator = value_iterator([])
    if started:
        assert iterator.send(None) is None

    with pytest.raises(TypeError, match='throw expected 1 to 3 arguments'):
        iterator.throw(*args)

    if not started:
        assert iterator.send(None) is None
    with pytest.raises(StopIteration) as completed:
        iterator.send(None)
    assert completed.value.value == 'normal'


class ResourceContext(Protocol):
    def open(self) -> AbstractAsyncContextManager[str]: ...


@pytest.mark.parametrize('phase', ['enter', 'exit'])
def test_cancellation_during_async_resource_cleanup_is_warning_free(phase: str) -> None:
    caught: list[asyncio.CancelledError] = []

    @asynccontextmanager
    async def provide_resource() -> AsyncIterator[str]:
        try:
            if phase == 'enter':
                await asyncio.sleep(0)
            try:
                yield 'ready'
            finally:
                if phase == 'exit':
                    await asyncio.sleep(0)
        except asyncio.CancelledError as error:
            caught.append(error)
            raise

    @compiled(
        Registry().register_method(ResourceContext, ResourceContext.open)(
            provide_resource
        )
    )
    def context() -> ResourceContext: ...

    resource = context().open()
    iterator = resource.__aenter__().__await__()
    if phase == 'exit':
        with pytest.raises(StopIteration) as entered:
            iterator.send(None)
        assert entered.value.value == 'ready'
        iterator = resource.__aexit__(None, None, None).__await__()
    assert iterator.send(None) is None

    cancellation = asyncio.CancelledError('cancel resource')
    with pytest.raises(asyncio.CancelledError) as raised:
        iterator.throw(cancellation)
    assert raised.value is cancellation
    assert caught == [cancellation]


def test_two_argument_throw_preserves_exception_instance() -> None:
    caught: list[ValueError] = []
    iterator = value_iterator(caught)
    assert iterator.send(None) is None
    error = ValueError('probe')

    with pytest.warns(DeprecationWarning, match='signature of throw'):
        with pytest.raises(StopIteration) as completed:
            iterator.throw(ValueError, error)

    assert completed.value.value == 'handled'
    assert caught[0] is error
