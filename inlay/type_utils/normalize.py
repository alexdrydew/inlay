"""Type normalization and introspection."""

import annotationlib
import inspect
import typing
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Iterator,
)
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass
from functools import lru_cache
from types import UnionType as PyUnionType
from typing import (
    Annotated,
    Literal,
    ParamSpec,
    Protocol,
    TypeAliasType,
    TypeVar,
    Union,  # pyright: ignore[reportDeprecated]
    cast,
    get_args,
    get_origin,
)

from inlay._native import (
    CallableType,
    ClassType,
    CyclePlaceholder,
    LazyRefType,
    ParamSpecType,
    PlainType,
    ProtocolType,
    Qualifier,
    SentinelType,
    TypedDictType,
    TypeVarType,
    UnionType,
)
from inlay.type_utils.errors import (
    MissingTypeAnnotationError,
    NormalizationError,
    UnresolvedTypeAnnotationError,
    UnsupportedVariadicParameterError,
)
from inlay.type_utils.markers import (
    UNQUALIFIED,
    LazyRef,
    extract_type_qualifier,
)

type NormalizedType = (
    SentinelType
    | TypeVarType
    | ParamSpecType
    | PlainType
    | ProtocolType
    | TypedDictType
    | UnionType
    | CallableType
    | ClassType
    | LazyRefType
)


class _Subscriptable(Protocol):
    def __getitem__(self, item: object) -> object: ...


def _type_args(tp: object) -> tuple[object, ...]:
    return cast(tuple[object, ...], get_args(tp))


def _type_params(obj: object) -> tuple[object, ...]:
    params = cast(tuple[object, ...], getattr(obj, '__type_params__', ()))
    if params:
        return params
    return cast(tuple[object, ...], getattr(obj, '__parameters__', ()))


def _orig_bases(cls: type) -> tuple[object, ...]:
    return cast(tuple[object, ...], getattr(cls, '__orig_bases__', ()))


def _typevar_default(tv: TypeVar) -> object | None:
    default = cast(object, getattr(tv, '__default__', typing.NoDefault))
    if default is typing.NoDefault:
        return None
    return default


def _callable_name(fn: object) -> str:
    name = getattr(fn, '__name__', None)
    if isinstance(name, str):
        return name
    return type(fn).__name__


def _deep_replace(
    root: NormalizedType,
    old: CyclePlaceholder,
    new: NormalizedType,
) -> None:
    """Recursively walk `root`, replacing `old` with `new` in all children."""
    visited: set[int] = set()
    _deep_replace_walk(root, old, new, visited)


def _deep_replace_walk(
    node: object,
    old: CyclePlaceholder,
    new: NormalizedType,
    visited: set[int],
) -> None:
    node_id = id(node)
    if node_id in visited:
        return
    visited.add(node_id)

    match node:
        case PlainType():
            node._replace_child(old, new)
            for arg in node.args:
                _deep_replace_walk(arg, old, new, visited)
        case ProtocolType():
            node._replace_child(old, new)
            for tp in node.type_params:
                _deep_replace_walk(tp, old, new, visited)
            for v in node.methods.values():
                _deep_replace_walk(v, old, new, visited)
            for v in node.attributes.values():
                _deep_replace_walk(v, old, new, visited)
            for v in node.properties.values():
                _deep_replace_walk(v, old, new, visited)
        case TypedDictType():
            node._replace_child(old, new)
            for tp in node.type_params:
                _deep_replace_walk(tp, old, new, visited)
            for v in node.attributes.values():
                _deep_replace_walk(v, old, new, visited)
        case UnionType():
            node._replace_child(old, new)
            for v in node.variants:
                _deep_replace_walk(v, old, new, visited)
        case CallableType():
            node._replace_child(old, new)
            for p in node.params:
                _deep_replace_walk(p, old, new, visited)
            _deep_replace_walk(node.return_type, old, new, visited)
        case ClassType():
            node._replace_child(old, new)
            for arg in node.args:
                _deep_replace_walk(arg, old, new, visited)
            init_params = node.init_params
            if init_params is not None:
                for p in init_params:
                    _deep_replace_walk(p, old, new, visited)
        case LazyRefType():
            node._replace_child(old, new)
            _deep_replace_walk(node.target, old, new, visited)
        case _:
            pass


type WrapperKind = Literal[
    'none', 'awaitable', 'context_manager', 'async_context_manager'
]
type ParamKind = Literal['positional_only', 'positional_or_keyword', 'keyword_only']


def _param_kind(p: inspect.Parameter) -> ParamKind:
    match p.kind:
        case inspect.Parameter.POSITIONAL_ONLY:
            return 'positional_only'
        case inspect.Parameter.POSITIONAL_OR_KEYWORD:
            return 'positional_or_keyword'
        case inspect.Parameter.KEYWORD_ONLY:
            return 'keyword_only'
        case _:
            raise ValueError(f'unexpected parameter kind: {p.kind}')


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
_WRAPPER_ORIGINS = (
    _CONTEXT_MANAGER_ORIGINS | _ASYNC_CONTEXT_MANAGER_ORIGINS | _AWAITABLE_ORIGINS
)
_NO_INIT_OR_REPLACE_INIT: object = getattr(typing, '_no_init_or_replace_init', None)


def _is_default_class_init(init: object) -> bool:
    return init is object.__init__ or init is _NO_INIT_OR_REPLACE_INIT


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


@dataclass(slots=True)
class ParamInfo:
    name: str
    type: NormalizedType
    has_default: bool
    kind: ParamKind


@dataclass(slots=True)
class CallableInfo:
    params: list[ParamInfo]
    return_type: NormalizedType
    return_wrapper: WrapperKind
    type_params: tuple[NormalizedType, ...]
    accepts_varargs: bool = False
    accepts_varkw: bool = False


@dataclass(slots=True)
class ClassInitInfo:
    params: list[ParamInfo]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize(t: object) -> NormalizedType:
    """Convert a Python type hint into a NormalizedType."""
    return _normalize(t, UNQUALIFIED, {})


def normalize_callable(fn: Callable[..., object]) -> CallableType:
    """Normalize a callable value (function/method) into a CallableType."""
    return _normalize_method_member(
        fn,
        UNQUALIFIED,
        {},
        function_name=_callable_name(fn),
    )


def normalize_with_qualifier(t: object, qualifiers: Qualifier) -> NormalizedType:
    """Convert a Python type hint into a NormalizedType with a specific qualifier."""
    return _normalize(t, qualifiers, {})


@lru_cache(maxsize=1024)
def get_callable_info(
    fn: Callable[..., object], *, skip_self: bool = True, allow_variadics: bool = True
) -> CallableInfo:
    """Inspect a callable and return its normalized parameter/return types."""
    if isinstance(fn, type):
        return _get_class_callable_info(fn, allow_variadics=allow_variadics)

    origin = typing.get_origin(fn)
    if origin is not None and isinstance(origin, type):
        return _get_generic_alias_callable_info(
            fn,
            origin,
            allow_variadics=allow_variadics,
        )

    sig = _signature(fn)
    hints = _get_annotations(fn)

    params: list[ParamInfo] = []
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
            ParamInfo(
                name=name,
                type=normalize(hints[name]),
                has_default=param.default is not inspect.Parameter.empty,  # pyright: ignore[reportAny]
                kind=_param_kind(param),
            )
        )

    return_type = normalize(hints.get('return', type(None)))
    unwrapped_return, return_wrapper = unwrap_return_type(return_type)
    if return_wrapper == 'none' and inspect.iscoroutinefunction(fn):
        return_wrapper = 'awaitable'
    fn_type_params = tuple(normalize(tp) for tp in _type_params(fn))
    return CallableInfo(
        params,
        unwrapped_return,
        return_wrapper,
        fn_type_params,
        accepts_varargs,
        accepts_varkw,
    )


# ---------------------------------------------------------------------------
# Core normalization
# ---------------------------------------------------------------------------


type _CacheEntry = tuple[Qualifier, list[CyclePlaceholder]]
type _NormCache = dict[int, _CacheEntry]


def _normalize(
    t: object,
    qualifiers: Qualifier,
    cache: _NormCache,
) -> NormalizedType:
    key = id(t)
    if key in cache:
        entry_qualifiers, placeholders = cache[key]
        if qualifiers != entry_qualifiers:
            raise NormalizationError(
                'Recursive type alias with differing qualifiers at back-reference '
                + 'is not supported'
            )
        placeholder = CyclePlaceholder()
        placeholders.append(placeholder)
        return placeholder  # pyright: ignore[reportReturnType]

    cache[key] = (qualifiers, [])
    result = _do_normalize(t, qualifiers, cache)

    _, placeholders = cache[key]
    for placeholder in placeholders:
        _deep_replace(result, placeholder, result)
    del cache[key]
    return result


def _do_normalize(
    t: object,
    qualifiers: Qualifier,
    cache: _NormCache,
) -> NormalizedType:
    origin = get_origin(t)
    args = _type_args(t)

    if origin is Annotated:
        base_type = args[0]
        metadata = args[1:]
        return _normalize(
            base_type,
            _extract_qualifiers(metadata, qualifiers),
            cache,
        )

    type_qual = extract_type_qualifier(t)
    if type_qual.is_qualified:
        qualifiers = qualifiers & type_qual

    if t is None:
        return SentinelType(value=None, qualifiers=qualifiers)

    if t is ...:
        return SentinelType(value=..., qualifiers=qualifiers)

    if isinstance(t, TypeVar):
        return TypeVarType(typevar=t, qualifiers=qualifiers)

    if isinstance(t, ParamSpec):
        return ParamSpecType(paramspec=t, qualifiers=qualifiers)

    if _is_newtype(t):
        return PlainType(origin=t, args=(), qualifiers=qualifiers)  # pyright: ignore[reportArgumentType]

    if isinstance(t, TypeAliasType):
        return _normalize(t.__value__, qualifiers, cache)  # pyright: ignore[reportAny]

    if origin is LazyRef:
        if not args:
            raise NormalizationError(f'LazyRef must have a type argument: {t!r}')
        target = _normalize(args[0], qualifiers, cache)
        return LazyRefType(target=target, qualifiers=qualifiers)

    if origin is Union or isinstance(t, PyUnionType):  # pyright: ignore[reportDeprecated]
        if not args:
            args = cast(tuple[object, ...], getattr(t, '__args__', ()))
        variants = tuple(
            _normalize_union_variant(arg, qualifiers, cache) for arg in args
        )
        return UnionType(variants=variants, qualifiers=qualifiers)

    if origin is Literal:
        return PlainType(origin=t, args=(), qualifiers=qualifiers)  # pyright: ignore[reportArgumentType]

    if origin is Callable:
        return _normalize_callable(t, args, qualifiers, cache)

    if origin is not None:
        normalized_args = tuple(_normalize(arg, qualifiers, cache) for arg in args)
        return _make_origin_type(
            cast(type, origin), normalized_args, qualifiers, cache, raw_type_args=args
        )

    if isinstance(t, type):
        type_params = _type_params(t)
        if type_params:
            raw_type_args: list[object] = []
            normalized_args_list: list[NormalizedType] = []
            for tp in type_params:
                if (
                    isinstance(tp, TypeVar)
                    and (default := _typevar_default(tp)) is not None
                ):
                    raw_type_args.append(default)
                    normalized_args_list.append(_normalize(default, qualifiers, cache))
                else:
                    raw_type_args.append(tp)
                    normalized_args_list.append(
                        TypeVarType(typevar=tp, qualifiers=qualifiers)
                        if isinstance(tp, TypeVar)
                        else _normalize(tp, qualifiers, cache)
                    )
            return _make_origin_type(
                t,
                tuple(normalized_args_list),
                qualifiers,
                cache,
                raw_type_args=tuple(raw_type_args),
            )
        return _make_origin_type(t, args=(), qualifiers=qualifiers, cache=cache)

    return PlainType(origin=t, args=(), qualifiers=qualifiers)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _make_origin_type(
    origin: type,
    args: tuple[NormalizedType, ...],
    qualifiers: Qualifier,
    cache: _NormCache,
    raw_type_args: tuple[object, ...] = (),
) -> PlainType | ProtocolType | TypedDictType | ClassType:
    if typing.is_protocol(origin):
        methods, attributes, properties = _extract_protocol_members(
            origin, qualifiers, cache, raw_type_args=raw_type_args
        )
        return ProtocolType(
            origin=origin,
            type_params=args,
            methods=methods,
            attributes=attributes,
            properties=properties,
            qualifiers=qualifiers,
        )
    if typing.is_typeddict(origin):
        class_type_params: tuple[object, ...] = getattr(origin, '__type_params__', ())
        subs: dict[TypeVar, object] = {}
        for tv, arg in zip(class_type_params, raw_type_args, strict=False):
            if isinstance(tv, TypeVar):
                subs[tv] = arg
        hints = _apply_substitutions(_get_annotations(origin), subs)
        attrs = {
            name: _normalize(hint, qualifiers, cache) for name, hint in hints.items()
        }
        return TypedDictType(
            origin=origin,
            type_params=args,
            attributes=attrs,
            qualifiers=qualifiers,
        )
    if (
        inspect.isclass(origin)
        and getattr(origin, '__module__', None) != 'builtins'
        and origin not in _WRAPPER_ORIGINS
    ):
        init = _get_class_init_info(
            origin,
            qualifiers,
            cache,
            raw_type_args=raw_type_args,
        )
        return ClassType(
            origin=origin,
            args=args,
            init_params=None if init is None else tuple(p.type for p in init.params),
            init_param_names=() if init is None else tuple(p.name for p in init.params),
            init_param_kinds=() if init is None else tuple(p.kind for p in init.params),
            init_param_has_default=()
            if init is None
            else tuple(p.has_default for p in init.params),
            qualifiers=qualifiers,
        )
    return PlainType(origin=origin, args=args, qualifiers=qualifiers)


def _normalize_callable(
    t: object,
    args: tuple[object, ...],
    qualifiers: Qualifier,
    cache: _NormCache,
) -> CallableType:
    if not args:
        raise NormalizationError(f'Callable must have type arguments: {t!r}')

    raw_params = args[0]
    param_types: list[object] = (
        list(raw_params) if isinstance(raw_params, (list, tuple)) else []  # pyright: ignore[reportUnknownArgumentType]
    )
    is_open_callable = raw_params is Ellipsis
    return_type = args[1] if len(args) > 1 else type(None)

    normalized_params = tuple(_normalize(p, qualifiers, cache) for p in param_types)
    normalized_return = _normalize(return_type, qualifiers, cache)
    unwrapped_return, return_wrapper = unwrap_return_type(normalized_return)
    return CallableType(
        params=normalized_params,
        param_names=tuple(f'_{i}' for i in range(len(normalized_params))),
        param_kinds=tuple(
            'positional_or_keyword' for _ in range(len(normalized_params))
        ),
        return_type=unwrapped_return,
        return_wrapper=return_wrapper,
        type_params=(),
        qualifiers=qualifiers,
        function_name=None,
        accepts_varargs=is_open_callable,
        accepts_varkw=is_open_callable,
    )


def _extract_qualifiers(
    metadata: tuple[object, ...],
    existing: Qualifier,
) -> Qualifier:
    result = existing
    for item in metadata:
        if isinstance(item, Qualifier):
            if item == Qualifier.ANY and not result.is_qualified:
                result = item
            else:
                result = result & item
    return result


def _normalize_union_variant(
    t: object,
    qualifiers: Qualifier,
    cache: _NormCache,
) -> NormalizedType:
    if t is type(None):
        return SentinelType(value=None, qualifiers=qualifiers)
    return _normalize(t, qualifiers, cache)


def _is_newtype(t: object) -> bool:
    return callable(t) and hasattr(t, '__supertype__')


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def _unresolved_annotation_error(exc: NameError) -> UnresolvedTypeAnnotationError:
    name = exc.name or '<unknown>'
    return UnresolvedTypeAnnotationError(f'Could not resolve type annotation {name!r}')


def _get_annotations(obj: object) -> dict[str, object]:
    try:
        return annotationlib.get_annotations(obj, eval_str=True)
    except NameError as exc:
        raise _unresolved_annotation_error(exc) from exc


def _signature(obj: object) -> inspect.Signature:
    try:
        return inspect.signature(obj)  # pyright: ignore[reportArgumentType]
    except NameError as exc:
        raise _unresolved_annotation_error(exc) from exc


def _build_typevar_substitutions(cls: type) -> dict[TypeVar, object]:
    subs = _collect_typevar_substitutions(cls, set())
    _resolve_typevar_substitutions(subs)
    return subs


def _collect_typevar_substitutions(
    cls: type,
    visited: set[int],
) -> dict[TypeVar, object]:
    cls_id = id(cls)
    if cls_id in visited:
        return {}
    visited.add(cls_id)

    subs: dict[TypeVar, object] = {}
    for base in _orig_bases(cls):
        origin = cast(object, get_origin(base))
        if origin is None:
            # Bare generic base (not subscripted): substitute TypeVar defaults.
            # e.g. HasValue[ValueT: Interface = Interface] used as plain
            # base -> ValueT should map to Interface.
            for tv in _type_params(base):
                if (
                    isinstance(tv, TypeVar)
                    and (default := _typevar_default(tv)) is not None
                ):
                    subs[tv] = default
            if isinstance(base, type):
                inherited = _collect_typevar_substitutions(base, visited)
                combined = {**inherited, **subs}
                for tv, arg in inherited.items():
                    subs[tv] = _substitute_typevars(arg, combined)
            continue

        local: dict[TypeVar, object] = {}
        for tv, arg in zip(_type_params(origin), _type_args(base), strict=False):
            if isinstance(tv, TypeVar):
                local[tv] = arg

        inherited: dict[TypeVar, object] = {}
        if isinstance(origin, type):
            inherited = _collect_typevar_substitutions(origin, visited)

        combined = {**inherited, **subs, **local}
        for tv, arg in local.items():
            subs[tv] = _substitute_typevars(arg, combined)
        combined = {**inherited, **subs}
        for tv, arg in inherited.items():
            subs[tv] = _substitute_typevars(arg, combined)

    return subs


def _resolve_typevar_substitutions(subs: dict[TypeVar, object]) -> None:
    while True:
        changed = False

        extra: dict[TypeVar, object] = {}
        for val in subs.values():
            if (
                isinstance(val, TypeVar)
                and val not in subs
                and (default := _typevar_default(val)) is not None
            ):
                extra[val] = default
        for tv, default in extra.items():
            if tv not in subs:
                subs[tv] = default
                changed = True

        for tv, val in list(subs.items()):
            new_val = _substitute_typevars(val, subs)
            if new_val != val:
                subs[tv] = new_val
                changed = True

        if not changed:
            return


def _apply_substitutions(
    hints: dict[str, object],
    subs: dict[TypeVar, object],
) -> dict[str, object]:
    if not subs:
        return hints
    return {k: _substitute_typevars(v, subs) for k, v in hints.items()}


def _extract_protocol_members(
    cls: type,
    qualifiers: Qualifier,
    cache: _NormCache,
    raw_type_args: tuple[object, ...] = (),
) -> tuple[
    dict[str, NormalizedType],
    dict[str, NormalizedType],
    dict[str, NormalizedType],
]:
    """Extract protocol members, returning (methods, attributes, properties)."""
    methods: dict[str, NormalizedType] = {}
    attributes: dict[str, NormalizedType] = {}
    properties: dict[str, NormalizedType] = {}

    hints = _get_annotations(cls)
    protocol_attrs = typing.get_protocol_members(cls)
    subs = _build_typevar_substitutions(cls)

    # When the protocol is subscripted (e.g. WriteTransition[TxCtxT_A]),
    # map the class's own TypeVars to the subscript args so member types
    # reference the caller's TypeVars instead of the class's.
    class_type_params = _type_params(cls)
    for tv, arg in zip(class_type_params, raw_type_args, strict=False):
        if isinstance(tv, TypeVar):
            subs[tv] = arg

    hints = _apply_substitutions(hints, subs)

    for name in protocol_attrs:
        attr: object = getattr(cls, name, None)

        if attr is None:
            if name in hints:
                attributes[name] = _normalize(hints[name], qualifiers, cache)
            continue

        if isinstance(attr, property):
            if name in hints:
                member_type = _normalize(hints[name], qualifiers, cache)
            else:
                fget = attr.fget
                if fget is not None:
                    fget_hints = _apply_substitutions(_get_annotations(fget), subs)
                    member_type = _normalize(
                        fget_hints.get('return', object), qualifiers, cache
                    )
                else:
                    member_type = _normalize(object, qualifiers, cache)
            properties[name] = member_type
        elif callable(attr):
            methods[name] = _normalize_method_member(
                attr, qualifiers, cache, subs, function_name=name
            )
        elif name in hints:
            attributes[name] = _normalize(hints[name], qualifiers, cache)

    return methods, attributes, properties


def _get_class_init_info(
    cls: type,
    qualifiers: Qualifier,
    cache: _NormCache,
    raw_type_args: tuple[object, ...] = (),
) -> ClassInitInfo | None:
    if inspect.isabstract(cls):
        return None
    if _is_default_class_init(cls.__init__):
        return ClassInitInfo(params=[])

    try:
        sig = _signature(cls.__init__)
        hints = _get_annotations(cls.__init__)
    except TypeError, ValueError, UnresolvedTypeAnnotationError:
        return None

    substitutions = _build_typevar_substitutions(cls)
    for tv, arg in zip(_type_params(cls), raw_type_args, strict=False):
        if isinstance(tv, TypeVar):
            substitutions[tv] = arg

    params: list[ParamInfo] = []
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return None

        if name not in hints:
            return None

        param_type = _substitute_typevars(hints[name], substitutions)
        params.append(
            ParamInfo(
                name=name,
                type=_normalize(param_type, qualifiers, cache),
                has_default=param.default is not inspect.Parameter.empty,  # pyright: ignore[reportAny]
                kind=_param_kind(param),
            )
        )

    return ClassInitInfo(params=params)


def _normalize_method_member(
    attr: object,
    qualifiers: Qualifier,
    cache: _NormCache,
    typevar_subs: dict[TypeVar, object] | None = None,
    *,
    function_name: str,
) -> CallableType:
    """Normalize a protocol method, propagating qualifiers."""
    sig = _signature(attr)
    method_hints = _get_annotations(attr)
    if typevar_subs:
        method_hints = _apply_substitutions(method_hints, typevar_subs)

    method_params: list[NormalizedType] = []
    param_names: list[str] = []
    param_kinds: list[ParamKind] = []
    accepts_varargs = False
    accepts_varkw = False
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                accepts_varargs = True
            else:
                accepts_varkw = True
            continue
        if param_name not in method_hints:
            raise MissingTypeAnnotationError(
                f'Parameter {param_name!r} has no type annotation'
            )
        method_params.append(_normalize(method_hints[param_name], qualifiers, cache))
        param_names.append(param_name)
        param_kinds.append(_param_kind(param))

    return_hint = method_hints.get('return', type(None))
    return_type = _normalize(return_hint, qualifiers, cache)
    unwrapped_return, return_wrapper = unwrap_return_type(return_type)
    if return_wrapper == 'none' and inspect.iscoroutinefunction(attr):
        return_wrapper = 'awaitable'

    type_params = tuple(_normalize(tp, qualifiers, cache) for tp in _type_params(attr))
    return CallableType(
        params=tuple(method_params),
        param_names=tuple(param_names),
        param_kinds=tuple(param_kinds),
        return_type=unwrapped_return,
        return_wrapper=return_wrapper,
        type_params=type_params,
        qualifiers=qualifiers,
        function_name=function_name,
        accepts_varargs=accepts_varargs,
        accepts_varkw=accepts_varkw,
    )


def _get_class_callable_info(
    cls: type, *, allow_variadics: bool = True
) -> CallableInfo:
    if _is_default_class_init(cls.__init__):
        # Inspecting object.__init__ directly reports `(self, /, *args, **kwargs)`,
        # but classes inheriting object.__init__ or Protocol's placeholder init
        # have the real call signature `()`.
        return CallableInfo(
            params=[], return_type=normalize(cls), return_wrapper='none', type_params=()
        )

    sig = _signature(cls.__init__)
    hints = _get_annotations(cls.__init__)

    params: list[ParamInfo] = []
    accepts_varargs = False
    accepts_varkw = False
    for name, param in sig.parameters.items():
        if name == 'self':
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
            ParamInfo(
                name=name,
                type=normalize(hints[name]),
                has_default=param.default is not inspect.Parameter.empty,  # pyright: ignore[reportAny]
                kind=_param_kind(param),
            )
        )

    return CallableInfo(
        params=params,
        return_type=normalize(cls),
        return_wrapper='none',
        type_params=(),
        accepts_varargs=accepts_varargs,
        accepts_varkw=accepts_varkw,
    )


def _get_generic_alias_callable_info(
    alias: object, origin: type, *, allow_variadics: bool = True
) -> CallableInfo:
    if _is_default_class_init(origin.__init__):
        # Inspecting object.__init__ directly reports `(self, /, *args, **kwargs)`,
        # but classes inheriting object.__init__ or Protocol's placeholder init
        # have the real call signature `()`.
        return CallableInfo(
            params=[],
            return_type=normalize(alias),
            return_wrapper='none',
            type_params=(),
        )

    sig = _signature(origin.__init__)

    type_args = _type_args(alias)
    type_params = _type_params(origin)
    substitutions: dict[TypeVar, object] = {
        tv: arg
        for tv, arg in zip(type_params, type_args, strict=False)
        if isinstance(tv, TypeVar)
    }

    hints = _get_annotations(origin.__init__)

    params: list[ParamInfo] = []
    accepts_varargs = False
    accepts_varkw = False
    for name, param in sig.parameters.items():
        if name == 'self':
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

        param_type = hints[name]
        if isinstance(param_type, TypeVar) and param_type in substitutions:
            param_type = substitutions[param_type]
        else:
            param_type = _substitute_typevars(param_type, substitutions)

        params.append(
            ParamInfo(
                name=name,
                type=normalize(param_type),
                has_default=param.default is not inspect.Parameter.empty,  # pyright: ignore[reportAny]
                kind=_param_kind(param),
            )
        )

    return CallableInfo(
        params=params,
        return_type=normalize(alias),
        return_wrapper='none',
        type_params=(),
        accepts_varargs=accepts_varargs,
        accepts_varkw=accepts_varkw,
    )


def _substitute_typevars(t: object, subs: dict[TypeVar, object]) -> object:
    return _substitute_typevars_inner(t, subs, set())


def _lookup_typevar_substitution(
    t: TypeVar,
    subs: dict[TypeVar, object],
) -> object | None:
    if t in subs:
        return subs[t]
    for candidate, replacement in subs.items():
        if candidate.__name__ == t.__name__:
            return replacement
    return None


def _make_union_type(args: tuple[object, ...]) -> object:
    result = args[0]
    for arg in args[1:]:
        result = cast(object, result | arg)  # pyright: ignore[reportOperatorIssue]
    return result


def _substitute_typevars_inner(
    t: object,
    subs: dict[TypeVar, object],
    seen: set[int],
) -> object:
    if isinstance(t, TypeVar):
        if id(t) in seen:
            return t
        replacement = _lookup_typevar_substitution(t, subs)
        if replacement is None or replacement is t:
            return t
        seen.add(id(t))
        try:
            return _substitute_typevars_inner(replacement, subs, seen)
        finally:
            seen.remove(id(t))

    origin = get_origin(t)
    if origin is None:
        return t

    if origin is Annotated:
        args = _type_args(t)
        if not args:
            return t
        inner, *metadata = args
        new_inner = _substitute_typevars_inner(inner, subs, seen)
        return Annotated[new_inner, *metadata]

    args = _type_args(t)
    new_args = tuple(_substitute_typevars_inner(arg, subs, seen) for arg in args)
    if not new_args:
        return t

    if origin is Union or origin is PyUnionType:  # pyright: ignore[reportDeprecated]
        return _make_union_type(new_args)

    subscriptable = cast(_Subscriptable, origin)
    if len(new_args) == 1:
        return subscriptable[new_args[0]]
    return subscriptable[new_args]
