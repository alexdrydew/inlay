"""Shared type-introspection helpers."""

import annotationlib
import inspect
import typing
from collections.abc import Callable
from typing import Literal, Protocol, TypeVar, cast, get_args

from typing_extensions import Sentinel

from inlay.type_utils.errors import UnresolvedTypeAnnotationError


class Subscriptable(Protocol):
    def __getitem__(self, item: object) -> object: ...


class Unionable(Protocol):
    def __or__(self, other: object) -> object: ...


def get_type_args(tp: object) -> tuple[object, ...]:
    return cast(tuple[object, ...], get_args(tp))


def get_type_params(obj: object) -> tuple[object, ...]:
    params = cast(tuple[object, ...], getattr(obj, '__type_params__', ()))
    if params:
        return params
    return cast(tuple[object, ...], getattr(obj, '__parameters__', ()))


def orig_bases(cls: type) -> tuple[object, ...]:
    return cast(tuple[object, ...], getattr(cls, '__orig_bases__', ()))


TYPEVAR_DEFAULT_MISSING = Sentinel('TYPEVAR_DEFAULT_MISSING')
TYPEVAR_SUBSTITUTION_MISSING = Sentinel('TYPEVAR_SUBSTITUTION_MISSING')


def typevar_default(tv: TypeVar) -> object:
    default = cast(object, getattr(tv, '__default__', typing.NoDefault))
    if default is typing.NoDefault:
        return TYPEVAR_DEFAULT_MISSING
    return default


def callable_name(fn: object) -> str:
    name = getattr(fn, '__name__', None)
    if isinstance(name, str):
        return name
    return type(fn).__name__


def class_init(cls: type) -> Callable[..., object]:
    return cast(Callable[..., object], cls.__init__)  # type: ignore[misc]


type ParamKind = Literal['positional_only', 'positional_or_keyword', 'keyword_only']


def param_kind(p: inspect.Parameter) -> ParamKind:
    match p.kind:
        case inspect.Parameter.POSITIONAL_ONLY:
            return 'positional_only'
        case inspect.Parameter.POSITIONAL_OR_KEYWORD:
            return 'positional_or_keyword'
        case inspect.Parameter.KEYWORD_ONLY:
            return 'keyword_only'
        case _:
            raise ValueError(f'unexpected parameter kind: {p.kind}')


_NO_INIT_OR_REPLACE_INIT: object = getattr(typing, '_no_init_or_replace_init', None)


def is_default_class_init(init: object) -> bool:
    return init is object.__init__ or init is _NO_INIT_OR_REPLACE_INIT


def is_hashable(value: object) -> bool:
    try:
        _ = hash(value)
    except TypeError:
        return False
    return True


def is_newtype(t: object) -> bool:
    return callable(t) and hasattr(t, '__supertype__')


def get_annotations(obj: object) -> dict[str, object]:
    try:
        return annotationlib.get_annotations(obj, eval_str=True)
    except NameError as exc:
        raise UnresolvedTypeAnnotationError.from_name_error(exc) from exc


def signature(obj: object) -> inspect.Signature:
    try:
        return inspect.signature(cast(Callable[..., object], obj))
    except NameError as exc:
        raise UnresolvedTypeAnnotationError.from_name_error(exc) from exc
