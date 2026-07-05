"""Return-type wrapper normalization helpers."""

from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Coroutine,
    Generator,
    Iterator,
)
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import Literal

from inlay._native import PlainType
from inlay.type_utils.normalized_type import NormalizedType

type WrapperKind = Literal[
    'none', 'awaitable', 'context_manager', 'async_context_manager'
]

_CONTEXT_MANAGER_ORIGINS: frozenset[type] = frozenset({
    AbstractContextManager,
    Generator,
    Iterator,
})
_ASYNC_CONTEXT_MANAGER_ORIGINS: frozenset[type] = frozenset({
    AbstractAsyncContextManager,
    AsyncGenerator,
    AsyncIterator,
})
_AWAITABLE_ORIGINS: frozenset[type] = frozenset({Awaitable, Coroutine})
WRAPPER_ORIGINS = (
    _CONTEXT_MANAGER_ORIGINS | _ASYNC_CONTEXT_MANAGER_ORIGINS | _AWAITABLE_ORIGINS
)


def unwrap_return_type(
    return_type: NormalizedType,
) -> tuple[NormalizedType, WrapperKind]:
    """Unwrap well-known wrapper types from a return type.

    Returns the inner type and the wrapper kind.
    """
    if not isinstance(return_type, PlainType):
        return return_type, 'none'
    origin = return_type.origin
    args = return_type.args
    if not args:
        return return_type, 'none'
    inner = args[0]
    if origin in _CONTEXT_MANAGER_ORIGINS:
        return inner, 'context_manager'
    if origin in _ASYNC_CONTEXT_MANAGER_ORIGINS:
        return inner, 'async_context_manager'
    if origin in _AWAITABLE_ORIGINS:
        unwrapped, wrapper = unwrap_return_type(inner)
        if wrapper == 'none':
            return unwrapped, 'awaitable'
        return unwrapped, wrapper
    return return_type, 'none'
