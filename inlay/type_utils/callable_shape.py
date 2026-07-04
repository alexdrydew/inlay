# pyright: reportPrivateUsage=false
"""Un-normalized callable signature introspection."""

import inspect
import typing
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar

from inlay.type_utils.errors import (
    MissingTypeAnnotationError,
    UnsupportedVariadicParameterError,
)
from inlay.type_utils.introspection import (
    ParamKind,
    _class_init,
    _get_annotations,
    _is_default_class_init,
    _param_kind,
    _signature,
    _type_args,
    _type_params,
)
from inlay.type_utils.substitution import _substitute_typevars
from inlay.type_utils.wrappers import WrapperKind


@dataclass(slots=True)
class _RawParamInfo:
    name: str
    type: object
    has_default: bool
    kind: ParamKind


@dataclass(slots=True)
class _CallableShape:
    params: list[_RawParamInfo]
    return_type: object
    type_params: tuple[object, ...]
    return_wrapper: WrapperKind
    accepts_varargs: bool = False
    accepts_varkw: bool = False


@lru_cache(maxsize=1024)
def _get_callable_shape(
    fn: Callable[..., object], *, skip_self: bool = True, allow_variadics: bool = True
) -> _CallableShape:
    """Inspect callable metadata without normalizing type hints."""
    if isinstance(fn, type):
        return _get_class_callable_shape(fn, allow_variadics=allow_variadics)

    origin = typing.get_origin(fn)
    if origin is not None and isinstance(origin, type):
        return _get_generic_alias_callable_shape(
            fn,
            origin,
            allow_variadics=allow_variadics,
        )

    sig = _signature(fn)
    hints = _get_annotations(fn)

    params, accepts_varargs, accepts_varkw = _collect_callable_shape_params(
        sig,
        hints,
        skip_self=skip_self,
        allow_variadics=allow_variadics,
    )
    return _CallableShape(
        params,
        hints.get('return', type(None)),
        _type_params(fn),
        'awaitable' if inspect.iscoroutinefunction(fn) else 'none',
        accepts_varargs,
        accepts_varkw,
    )


def get_callable_shape(
    fn: Callable[..., object], *, skip_self: bool = True, allow_variadics: bool = True
) -> _CallableShape:
    return _get_callable_shape(
        fn,
        skip_self=skip_self,
        allow_variadics=allow_variadics,
    )


def _collect_callable_shape_params(
    sig: inspect.Signature,
    hints: dict[str, object],
    *,
    skip_self: bool,
    allow_variadics: bool,
    transform_hint: Callable[[object], object] | None = None,
) -> tuple[list[_RawParamInfo], bool, bool]:
    params: list[_RawParamInfo] = []
    accepts_varargs = False
    accepts_varkw = False
    for name, param in sig.parameters.items():
        if skip_self and name == 'self':
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            if not allow_variadics:
                raise UnsupportedVariadicParameterError(
                    f'Variadic parameter {name!r} is not supported here'
                )
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                accepts_varargs = True
            else:
                accepts_varkw = True
            continue

        if name not in hints:
            raise MissingTypeAnnotationError(
                f'Parameter {name!r} has no type annotation'
            )

        params.append(
            _RawParamInfo(
                name=name,
                type=transform_hint(hints[name]) if transform_hint else hints[name],
                has_default=param.default is not inspect.Parameter.empty,  # pyright: ignore[reportAny]
                kind=_param_kind(param),
            )
        )

    return params, accepts_varargs, accepts_varkw


# ---------------------------------------------------------------------------
# Core normalization


def _get_class_callable_shape(
    cls: type, *, allow_variadics: bool = True
) -> _CallableShape:
    init = _class_init(cls)
    if _is_default_class_init(init):
        # Inspecting object.__init__ directly reports `(self, /, *args, **kwargs)`,
        # but classes inheriting object.__init__ or Protocol's placeholder init
        # have the real call signature `()`.
        return _CallableShape(
            params=[],
            return_type=cls,
            type_params=(),
            return_wrapper='none',
        )

    sig = _signature(init)
    hints = _get_annotations(init)
    params, accepts_varargs, accepts_varkw = _collect_callable_shape_params(
        sig,
        hints,
        skip_self=True,
        allow_variadics=allow_variadics,
    )
    return _CallableShape(
        params=params,
        return_type=cls,
        type_params=(),
        return_wrapper='none',
        accepts_varargs=accepts_varargs,
        accepts_varkw=accepts_varkw,
    )


def _get_generic_alias_callable_shape(
    alias: object, origin: type, *, allow_variadics: bool = True
) -> _CallableShape:
    init = _class_init(origin)
    if _is_default_class_init(init):
        # Inspecting object.__init__ directly reports `(self, /, *args, **kwargs)`,
        # but classes inheriting object.__init__ or Protocol's placeholder init
        # have the real call signature `()`.
        return _CallableShape(
            params=[],
            return_type=alias,
            type_params=(),
            return_wrapper='none',
        )

    sig = _signature(init)
    type_args = _type_args(alias)
    type_params = _type_params(origin)
    substitutions: dict[TypeVar, object] = {
        tv: arg
        for tv, arg in zip(type_params, type_args, strict=False)
        if isinstance(tv, TypeVar)
    }

    def substitute_hint(param_type: object) -> object:
        if isinstance(param_type, TypeVar) and param_type in substitutions:
            return substitutions[param_type]
        return _substitute_typevars(param_type, substitutions)

    hints = _get_annotations(init)
    params, accepts_varargs, accepts_varkw = _collect_callable_shape_params(
        sig,
        hints,
        skip_self=True,
        allow_variadics=allow_variadics,
        transform_hint=substitute_hint,
    )
    return _CallableShape(
        params=params,
        return_type=alias,
        type_params=(),
        return_wrapper='none',
        accepts_varargs=accepts_varargs,
        accepts_varkw=accepts_varkw,
    )
