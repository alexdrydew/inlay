"""Type normalization."""

import inspect
import typing
from collections.abc import Callable
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
    get_origin,
)

from typing_extensions import Sentinel

from inlay._native import (
    CallableSignatureType,
    CellType,
    ClassType,
    CyclePlaceholder,
    ParamSpecType,
    PlainType,
    ProtocolBase,
    ProtocolMethod,
    ProtocolType,
    Qualifier,
    ReadCellType,
    SentinelType,
    TypedDictType,
    TypeVarType,
    UnionType,
)
from inlay.constants import RECURSIVE_QUALIFIER_LIMITATION_URL
from inlay.type_utils.callable_shape import get_callable_shape
from inlay.type_utils.cycles import (
    IdInterner,
    NormalizationStack,
    NormMemo,
    active_normalization_entry,
    deep_replace,
)
from inlay.type_utils.errors import (
    MissingTypeAnnotationError,
    NormalizationError,
    UnresolvedTypeAnnotationError,
)
from inlay.type_utils.introspection import (
    TYPEVAR_DEFAULT_MISSING,
    ParamKind,
    callable_name,
    class_init,
    get_annotations,
    get_type_args,
    get_type_params,
    is_default_class_init,
    is_hashable,
    is_newtype,
    orig_bases,
    param_kind,
    signature,
    typevar_default,
)
from inlay.type_utils.markers import (
    UNQUALIFIED,
    Cell,
    ReadCell,
    extract_type_qualifier,
)
from inlay.type_utils.normalize.helpers import (
    extract_qualifiers,
    strip_typeddict_requiredness,
    typed_dict_required_optional_keys,
)
from inlay.type_utils.normalized_type import NormalizedType
from inlay.type_utils.substitution import (
    apply_substitutions,
    build_typevar_substitutions,
    is_self_type,
    owner_self_type,
    replace_self_type,
    substitute_typevars,
)
from inlay.type_utils.wrappers import (
    WRAPPER_ORIGINS,
    WrapperKind,
    unwrap_return_type,
)


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


def normalize(t: object) -> NormalizedType:
    """Convert a Python type hint into a NormalizedType."""
    # types are usually hashable, but since annotations can contain arbitrary objects
    # we use uncached fallback
    if is_hashable((t, UNQUALIFIED)):
        return _normalize_cached(t, UNQUALIFIED)
    return _normalize_uncached(t, UNQUALIFIED)


def normalize_callable(fn: Callable[..., object]) -> CallableSignatureType:
    """Normalize a callable value (function/method) into a signature type."""
    if is_hashable(fn):
        return _normalize_callable_value_cached(fn)
    return _normalize_callable_value_uncached(fn)


def normalize_with_qualifier(t: object, qualifiers: Qualifier) -> NormalizedType:
    """Convert a Python type hint into a NormalizedType with a specific qualifier."""
    if is_hashable((t, qualifiers)):
        return _normalize_cached(t, qualifiers)
    return _normalize_uncached(t, qualifiers)


def _normalize_uncached(t: object, qualifiers: Qualifier) -> NormalizedType:
    return _normalize(t, qualifiers, {}, {}, IdInterner())


@lru_cache(maxsize=4096)
def _normalize_cached(t: object, qualifiers: Qualifier) -> NormalizedType:
    return _normalize_uncached(t, qualifiers)


def _normalize_callable_value_uncached(
    fn: Callable[..., object],
) -> CallableSignatureType:
    return _normalize_method_member(
        fn,
        UNQUALIFIED,
        {},
        {},
        IdInterner(),
        function_name=callable_name(fn),
    )


@lru_cache(maxsize=1024)
def _normalize_callable_value_cached(
    fn: Callable[..., object],
) -> CallableSignatureType:
    return _normalize_callable_value_uncached(fn)


@lru_cache(maxsize=1024)
def get_callable_info(
    fn: Callable[..., object], *, skip_self: bool = True, allow_variadics: bool = True
) -> CallableInfo:
    """Inspect a callable and return its normalized parameter/return types."""
    shape = get_callable_shape(
        fn,
        skip_self=skip_self,
        allow_variadics=allow_variadics,
    )
    params = [
        ParamInfo(
            name=p.name,
            type=normalize(p.type),
            has_default=p.has_default,
            kind=p.kind,
        )
        for p in shape.params
    ]
    return_type = normalize(shape.return_type)
    unwrapped_return, return_wrapper = unwrap_return_type(return_type)
    if return_wrapper == 'none':
        return_wrapper = shape.return_wrapper
    fn_type_params = tuple(normalize(tp) for tp in shape.type_params)
    return CallableInfo(
        params,
        unwrapped_return,
        return_wrapper,
        fn_type_params,
        shape.accepts_varargs,
        shape.accepts_varkw,
    )


def _normalize(
    t: object,
    qualifiers: Qualifier,
    stack: NormalizationStack,
    cache: NormMemo,
    interner: IdInterner,
) -> NormalizedType:
    key = interner.get_id(t)
    if key in stack:
        entry_qualifiers, placeholders = stack[key]
        if qualifiers != entry_qualifiers:
            raise NormalizationError(
                'Recursive type alias with differing qualifiers at back-reference '
                + 'is not supported; see '
                + RECURSIVE_QUALIFIER_LIMITATION_URL
            )
        placeholder = CyclePlaceholder()
        placeholders.append(placeholder)
        return cast(NormalizedType, cast(object, placeholder))

    cache_key = (key, qualifiers)
    if cache_key in cache:
        return cache[cache_key]

    with active_normalization_entry(stack, key, qualifiers) as placeholders:
        result = _do_normalize(t, qualifiers, stack, cache, interner)
        for placeholder in placeholders:
            deep_replace(result, placeholder, result)

    cache[cache_key] = result
    return result


def _normalize_with_self_type(
    t: object,
    qualifiers: Qualifier,
    stack: NormalizationStack,
    cache: NormMemo,
    interner: IdInterner,
    self_type: object | None,
) -> NormalizedType:
    if self_type is not None:
        t = replace_self_type(t, self_type)
    return _normalize(t, qualifiers, stack, cache, interner)


def normalize_with_self_type(
    t: object,
    qualifiers: Qualifier,
    self_type: object | None,
) -> NormalizedType:
    if self_type is None:
        return normalize_with_qualifier(t, qualifiers)
    return normalize_with_qualifier(replace_self_type(t, self_type), qualifiers)


def _do_normalize(
    t: object,
    qualifiers: Qualifier,
    stack: NormalizationStack,
    cache: NormMemo,
    interner: IdInterner,
) -> NormalizedType:
    origin = get_origin(t)
    args = get_type_args(t)

    if origin is Annotated:
        base_type = args[0]
        metadata = args[1:]
        return _normalize(
            base_type,
            extract_qualifiers(metadata, qualifiers),
            stack,
            cache,
            interner,
        )

    type_qual = extract_type_qualifier(t)
    if type_qual.is_qualified:
        qualifiers = qualifiers & type_qual

    if is_self_type(t):
        raise NormalizationError(
            'typing.Self can only be normalized in a class or protocol context'
        )

    if t is None or t is type(None):
        return SentinelType(value=None, qualifiers=qualifiers)

    if t is ...:
        return SentinelType(value=..., qualifiers=qualifiers)

    if isinstance(t, TypeVar):
        return TypeVarType(typevar=t, qualifiers=qualifiers)

    if isinstance(t, ParamSpec):
        return ParamSpecType(paramspec=t, qualifiers=qualifiers)

    if is_newtype(t):
        return PlainType(origin=cast(type, t), args=(), qualifiers=qualifiers)

    if isinstance(t, TypeAliasType):
        return _normalize(t.__value__, qualifiers, stack, cache, interner)  # pyright: ignore[reportAny]

    if isinstance(origin, TypeAliasType):
        subs: dict[TypeVar, object] = {}
        for tv, arg in zip(get_type_params(origin), args, strict=False):
            if isinstance(tv, TypeVar):
                subs[tv] = arg
        value = substitute_typevars(origin.__value__, subs)  # pyright: ignore[reportAny]
        return _normalize(value, qualifiers, stack, cache, interner)

    if origin is ReadCell:
        if not args:
            raise NormalizationError(f'ReadCell must have a type argument: {t!r}')
        target = _normalize(args[0], qualifiers, stack, cache, interner)
        return ReadCellType(target=target, qualifiers=qualifiers)

    if origin is Cell:
        if not args:
            raise NormalizationError(f'Cell must have a type argument: {t!r}')
        target = _normalize(args[0], qualifiers, stack, cache, interner)
        return CellType(target=target, qualifiers=qualifiers)

    if origin is Union or isinstance(t, PyUnionType):  # pyright: ignore[reportDeprecated]
        if not args:
            args = cast(tuple[object, ...], getattr(t, '__args__', ()))
        variants = tuple(
            _normalize_union_variant(arg, qualifiers, stack, cache, interner)
            for arg in args
        )
        return UnionType(variants=variants, qualifiers=qualifiers)

    if origin is Literal:
        return PlainType(origin=cast(type, t), args=(), qualifiers=qualifiers)

    if origin is Callable:
        return _normalize_callable(t, args, qualifiers, stack, cache, interner)

    if origin is not None:
        normalized_args = tuple(
            _normalize(arg, qualifiers, stack, cache, interner) for arg in args
        )
        return _make_origin_type(
            cast(type, origin),
            normalized_args,
            qualifiers,
            stack,
            cache,
            interner,
            raw_type_args=args,
            self_type=t,
        )

    if isinstance(t, type):
        type_params = get_type_params(t)
        if type_params:
            raw_type_args: list[object] = []
            normalized_args_list: list[NormalizedType] = []
            for tp in type_params:
                default = (
                    typevar_default(tp)
                    if isinstance(tp, TypeVar)
                    else TYPEVAR_DEFAULT_MISSING
                )
                if default is not TYPEVAR_DEFAULT_MISSING:
                    raw_type_args.append(default)
                    normalized_args_list.append(
                        _normalize(default, qualifiers, stack, cache, interner)
                    )
                else:
                    raw_type_args.append(tp)
                    normalized_args_list.append(
                        TypeVarType(typevar=tp, qualifiers=qualifiers)
                        if isinstance(tp, TypeVar)
                        else _normalize(tp, qualifiers, stack, cache, interner)
                    )
            raw_type_args_tuple = tuple(raw_type_args)
            return _make_origin_type(
                t,
                tuple(normalized_args_list),
                qualifiers,
                stack,
                cache,
                interner,
                raw_type_args=raw_type_args_tuple,
                self_type=owner_self_type(t, raw_type_args_tuple),
            )
        return _make_origin_type(
            t,
            args=(),
            qualifiers=qualifiers,
            stack=stack,
            cache=cache,
            interner=interner,
            self_type=t,
        )

    return PlainType(origin=cast(type, t), args=(), qualifiers=qualifiers)


def _make_origin_type(
    origin: type,
    args: tuple[NormalizedType, ...],
    qualifiers: Qualifier,
    stack: NormalizationStack,
    cache: NormMemo,
    interner: IdInterner,
    raw_type_args: tuple[object, ...] = (),
    self_type: object | None = None,
) -> PlainType | ProtocolType | TypedDictType | ClassType:
    if typing.is_protocol(origin):
        _reject_qualified_protocol_bases(origin)
        methods, attributes, properties, direct_methods = _extract_protocol_members(
            origin,
            qualifiers,
            stack,
            cache,
            interner,
            raw_type_args=raw_type_args,
            self_type=self_type,
        )
        current_base = ProtocolBase(
            origin,
            args,
            direct_methods,
        )
        protocol_mro = _collect_protocol_mro(
            origin,
            qualifiers,
            stack,
            cache,
            interner,
            _build_protocol_substitutions(origin, raw_type_args),
            current_base,
            self_type,
        )
        return ProtocolType(
            origin=origin,
            type_params=args,
            methods=methods,
            attributes=attributes,
            properties=properties,
            qualifiers=qualifiers,
            protocol_mro=protocol_mro,
            direct_methods=direct_methods,
        )
    if typing.is_typeddict(origin):
        class_type_params: tuple[object, ...] = getattr(origin, '__type_params__', ())
        subs: dict[TypeVar, object] = {}
        for tv, arg in zip(class_type_params, raw_type_args, strict=False):
            if isinstance(tv, TypeVar):
                subs[tv] = arg
        hints = apply_substitutions(get_annotations(origin), subs)
        required_keys, optional_keys = typed_dict_required_optional_keys(origin, hints)
        attrs = {
            name: _normalize(
                strip_typeddict_requiredness(hint),
                qualifiers,
                stack,
                cache,
                interner,
            )
            for name, hint in hints.items()
        }
        return TypedDictType(
            origin=origin,
            type_params=args,
            attributes=attrs,
            qualifiers=qualifiers,
            required_keys=required_keys,
            optional_keys=optional_keys,
        )
    if (
        inspect.isclass(origin)
        and getattr(origin, '__module__', None) != 'builtins'
        and origin not in WRAPPER_ORIGINS
    ):
        init = _get_class_init_info(
            origin,
            qualifiers,
            stack,
            cache,
            interner,
            raw_type_args=raw_type_args,
            self_type=self_type,
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
    stack: NormalizationStack,
    cache: NormMemo,
    interner: IdInterner,
) -> CallableSignatureType:
    if not args:
        raise NormalizationError(f'Callable must have type arguments: {t!r}')

    raw_params = args[0]
    param_types: list[object] = (
        list(raw_params) if isinstance(raw_params, (list, tuple)) else []  # pyright: ignore[reportUnknownArgumentType]
    )
    is_open_callable = raw_params is Ellipsis
    return_type = args[1] if len(args) > 1 else type(None)

    normalized_params = tuple(
        _normalize(p, qualifiers, stack, cache, interner) for p in param_types
    )
    normalized_return = _normalize(return_type, qualifiers, stack, cache, interner)
    unwrapped_return, return_wrapper = unwrap_return_type(normalized_return)
    return CallableSignatureType(
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


def _normalize_union_variant(
    t: object,
    qualifiers: Qualifier,
    stack: NormalizationStack,
    cache: NormMemo,
    interner: IdInterner,
) -> NormalizedType:
    if t is type(None):
        return SentinelType(value=None, qualifiers=qualifiers)
    return _normalize(t, qualifiers, stack, cache, interner)


def _build_protocol_substitutions(
    cls: type,
    raw_type_args: tuple[object, ...],
) -> dict[TypeVar, object]:
    subs = build_typevar_substitutions(cls)
    class_type_params = get_type_params(cls)
    for tv, arg in zip(class_type_params, raw_type_args, strict=False):
        if isinstance(tv, TypeVar):
            subs[tv] = arg
    return subs


MISSING = Sentinel('MISSING')


def _is_protocol_base(base: object) -> bool:
    origin = _protocol_base_origin(base)
    return (
        isinstance(origin, type)
        and origin is not Protocol
        and typing.is_protocol(origin)
    )


def _protocol_base_origin(base: object) -> object:
    while get_origin(base) is Annotated:
        args = get_type_args(base)
        if not args:
            break
        base = args[0]
    return get_origin(base) or base


def _reject_qualified_protocol_bases(cls: type) -> None:
    for base in orig_bases(cls):
        if _is_protocol_base(base) and extract_type_qualifier(base).is_qualified:
            raise NormalizationError('Qualified protocol bases are not supported')


def _protocol_bases(cls: type) -> tuple[object, ...]:
    bases: list[object] = []
    seen_origins: set[type] = set()
    for base in orig_bases(cls):
        if _is_protocol_base(base) and extract_type_qualifier(base).is_qualified:
            raise NormalizationError('Qualified protocol bases are not supported')
        bases.append(base)
        origin = _protocol_base_origin(base)
        if isinstance(origin, type):
            seen_origins.add(origin)
    for base in cls.__bases__:
        if base is Protocol or base is object or base in seen_origins:
            continue
        bases.append(base)
    return tuple(bases)


def _collect_protocol_annotations(
    cls: type,
    subs: dict[TypeVar, object],
) -> dict[str, object]:
    hints: dict[str, object] = {}
    for base in reversed(cls.__mro__):
        if base is Protocol or base is object or not typing.is_protocol(base):
            continue
        hints.update(get_annotations(base))
    return apply_substitutions(hints, subs)


def _collect_protocol_mro(
    cls: type,
    qualifiers: Qualifier,
    stack: NormalizationStack,
    cache: NormMemo,
    interner: IdInterner,
    subs: dict[TypeVar, object],
    current_base: ProtocolBase,
    self_type: object | None,
) -> tuple[ProtocolBase, ...]:
    protocols_by_origin: dict[type, ProtocolBase] = {}
    for base in _protocol_bases(cls):
        if not _is_protocol_base(base):
            continue
        substituted_base = substitute_typevars(base, subs)
        if not _is_protocol_base(substituted_base):
            continue

        normalized_base = _normalize_with_self_type(
            substituted_base,
            qualifiers,
            stack,
            cache,
            interner,
            self_type,
        )
        if not isinstance(normalized_base, ProtocolType):
            continue

        for normalized_protocol in normalized_base.protocol_mro:
            _ = protocols_by_origin.setdefault(
                normalized_protocol.origin, normalized_protocol
            )

    result: list[ProtocolBase] = [current_base]
    seen: set[type] = {cls}
    for base in cls.__mro__[1:]:
        if base is Protocol or base is object or not typing.is_protocol(base):
            continue
        protocol = protocols_by_origin.get(base)
        if protocol is None or protocol.origin in seen:
            continue
        result.append(protocol)
        seen.add(protocol.origin)
    return tuple(result)


def _extract_protocol_members(
    cls: type,
    qualifiers: Qualifier,
    stack: NormalizationStack,
    cache: NormMemo,
    interner: IdInterner,
    raw_type_args: tuple[object, ...] = (),
    self_type: object | None = None,
) -> tuple[
    dict[str, ProtocolMethod],
    dict[str, NormalizedType],
    dict[str, NormalizedType],
    tuple[str, ...],
]:
    """Extract protocol members."""
    methods: dict[str, ProtocolMethod] = {}
    attributes: dict[str, NormalizedType] = {}
    properties: dict[str, NormalizedType] = {}

    protocol_attrs = typing.get_protocol_members(cls)
    subs = _build_protocol_substitutions(cls, raw_type_args)
    hints = _collect_protocol_annotations(cls, subs)
    class_dict: dict[str, object] = dict(vars(cls))
    direct_methods = tuple(
        sorted(
            name
            for name, value in class_dict.items()
            if name in protocol_attrs and callable(value)
        )
    )
    for name in protocol_attrs:
        direct_attr = class_dict.get(name, MISSING)

        if direct_attr is not MISSING and callable(direct_attr):
            methods[name] = ProtocolMethod(
                _normalize_method_member(
                    direct_attr,
                    qualifiers,
                    stack,
                    cache,
                    interner,
                    subs,
                    function_name=name,
                    self_type=self_type,
                )
            )
            continue

        attr: object = getattr(cls, name, None)

        if callable(attr):
            methods[name] = ProtocolMethod(
                _normalize_method_member(
                    attr,
                    qualifiers,
                    stack,
                    cache,
                    interner,
                    subs,
                    function_name=name,
                    self_type=self_type,
                )
            )
            continue

        if attr is None:
            if name in hints:
                attributes[name] = _normalize_with_self_type(
                    hints[name], qualifiers, stack, cache, interner, self_type
                )
            continue

        if isinstance(attr, property):
            if name in hints:
                member_type = _normalize_with_self_type(
                    hints[name], qualifiers, stack, cache, interner, self_type
                )
            else:
                fget = attr.fget
                if fget is not None:
                    fget_hints = apply_substitutions(get_annotations(fget), subs)
                    member_type = _normalize_with_self_type(
                        fget_hints.get('return', object),
                        qualifiers,
                        stack,
                        cache,
                        interner,
                        self_type,
                    )
                else:
                    member_type = _normalize(object, qualifiers, stack, cache, interner)
            properties[name] = member_type
        elif name in hints:
            attributes[name] = _normalize_with_self_type(
                hints[name], qualifiers, stack, cache, interner, self_type
            )

    return methods, attributes, properties, direct_methods


def _get_class_init_info(
    cls: type,
    qualifiers: Qualifier,
    stack: NormalizationStack,
    cache: NormMemo,
    interner: IdInterner,
    raw_type_args: tuple[object, ...] = (),
    self_type: object | None = None,
) -> ClassInitInfo | None:
    if inspect.isabstract(cls):
        return None
    init = class_init(cls)
    if is_default_class_init(init):
        return ClassInitInfo(params=[])

    try:
        sig = signature(init)
        hints = get_annotations(init)
    except TypeError, ValueError, UnresolvedTypeAnnotationError:
        return None

    substitutions = build_typevar_substitutions(cls)
    for tv, arg in zip(get_type_params(cls), raw_type_args, strict=False):
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

        param_type = substitute_typevars(hints[name], substitutions)
        params.append(
            ParamInfo(
                name=name,
                type=_normalize_with_self_type(
                    param_type,
                    qualifiers,
                    stack,
                    cache,
                    interner,
                    self_type,
                ),
                has_default=param.default is not inspect.Parameter.empty,  # pyright: ignore[reportAny]
                kind=param_kind(param),
            )
        )

    return ClassInitInfo(params=params)


def _normalize_method_member(
    attr: object,
    qualifiers: Qualifier,
    stack: NormalizationStack,
    cache: NormMemo,
    interner: IdInterner,
    typevar_subs: dict[TypeVar, object] | None = None,
    *,
    function_name: str,
    self_type: object | None = None,
) -> CallableSignatureType:
    """Normalize a protocol method, propagating qualifiers."""
    sig = signature(attr)
    method_hints = get_annotations(attr)
    if typevar_subs:
        method_hints = apply_substitutions(method_hints, typevar_subs)

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
        method_params.append(
            _normalize_with_self_type(
                method_hints[param_name],
                qualifiers,
                stack,
                cache,
                interner,
                self_type,
            )
        )
        param_names.append(param_name)
        param_kinds.append(param_kind(param))

    return_hint = method_hints.get('return', type(None))
    return_type = _normalize_with_self_type(
        return_hint,
        qualifiers,
        stack,
        cache,
        interner,
        self_type,
    )
    unwrapped_return, return_wrapper = unwrap_return_type(return_type)
    if return_wrapper == 'none' and inspect.iscoroutinefunction(attr):
        return_wrapper = 'awaitable'

    type_params = tuple(
        _normalize(tp, qualifiers, stack, cache, interner)
        for tp in get_type_params(attr)
    )
    return CallableSignatureType(
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
