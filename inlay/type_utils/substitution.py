"""TypeVar substitution and typing.Self replacement helpers."""

from types import UnionType as PyUnionType
from typing import (
    Annotated,
    TypeAliasType,
    TypeVar,
    Union,  # pyright: ignore[reportDeprecated]
    cast,
    get_origin,
)
from typing import (
    Self as TypingSelf,
)

from inlay.type_utils.introspection import (
    TYPEVAR_DEFAULT_MISSING,
    TYPEVAR_SUBSTITUTION_MISSING,
    Subscriptable,
    Unionable,
    get_type_args,
    get_type_params,
    orig_bases,
    typevar_default,
)


def is_self_type(t: object) -> bool:
    return t is TypingSelf


def replace_self_type(t: object, self_type: object) -> object:
    replaced, _ = replace_self_type_inner(t, self_type, set())
    return replaced


def replace_self_type_inner(
    t: object,
    self_type: object,
    seen: set[int],
) -> tuple[object, bool]:
    if is_self_type(t):
        return self_type, True

    if isinstance(t, list):
        changed = False
        list_result: list[object] = []
        for item in cast(list[object], t):
            new_item, item_changed = replace_self_type_inner(item, self_type, seen)
            list_result.append(new_item)
            changed = changed or item_changed
        return (list_result, True) if changed else (cast(object, t), False)

    if isinstance(t, tuple):
        changed = False
        tuple_result: list[object] = []
        tuple_items = cast(tuple[object, ...], t)  # ty: ignore[redundant-cast]
        for item in tuple_items:
            new_item, item_changed = replace_self_type_inner(item, self_type, seen)
            tuple_result.append(new_item)
            changed = changed or item_changed
        return (tuple(tuple_result), True) if changed else (cast(object, t), False)

    t_id = id(t)
    if t_id in seen:
        return t, False

    if isinstance(t, TypeAliasType):
        seen.add(t_id)
        try:
            value, changed = replace_self_type_inner(t.__value__, self_type, seen)  # pyright: ignore[reportAny]
        finally:
            seen.remove(t_id)
        return (value, True) if changed else (t, False)

    origin = get_origin(t)
    if origin is None:
        return t, False

    if isinstance(origin, TypeAliasType):
        subs: dict[TypeVar, object] = {}
        for tv, arg in zip(get_type_params(origin), get_type_args(t), strict=False):
            if isinstance(tv, TypeVar):
                subs[tv] = arg
        value = substitute_typevars(origin.__value__, subs)  # pyright: ignore[reportAny]
        replaced, changed = replace_self_type_inner(value, self_type, seen)
        return (replaced, True) if changed else (t, False)

    if origin is Annotated:
        args = get_type_args(t)
        if not args:
            return t, False
        inner, *metadata = args
        new_inner, changed = replace_self_type_inner(inner, self_type, seen)
        if not changed:
            return t, False
        return Annotated[new_inner, *metadata], True  # pyrefly: ignore[not-a-type]

    args = get_type_args(t)
    changed = False
    new_args: list[object] = []
    for arg in args:
        new_arg, arg_changed = replace_self_type_inner(arg, self_type, seen)
        new_args.append(new_arg)
        changed = changed or arg_changed
    if not changed:
        return t, False

    return _rebuild_subscripted_type(cast(object, origin), tuple(new_args)), True


def _rebuild_subscripted_type(origin: object, args: tuple[object, ...]) -> object:
    if origin is Union or origin is PyUnionType:  # pyright: ignore[reportDeprecated]
        return _make_union_type(args)

    subscriptable = cast(Subscriptable, origin)
    if len(args) == 1:
        return subscriptable[args[0]]
    return subscriptable[args]


def owner_self_type(origin: type, raw_type_args: tuple[object, ...]) -> object:
    if not raw_type_args:
        return origin
    return _rebuild_subscripted_type(origin, raw_type_args)


def build_typevar_substitutions(cls: type) -> dict[TypeVar, object]:
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
    for base in orig_bases(cls):
        origin = cast(object, get_origin(base))
        if origin is None:
            # Bare generic base (not subscripted): substitute TypeVar defaults.
            # e.g. HasValue[ValueT: Interface = Interface] used as plain
            # base -> ValueT should map to Interface.
            for tv in get_type_params(base):
                if not isinstance(tv, TypeVar):
                    continue
                default = typevar_default(tv)
                if default is not TYPEVAR_DEFAULT_MISSING:
                    subs[tv] = default
            if isinstance(base, type):
                inherited_from_base = _collect_typevar_substitutions(base, visited)
                combined = {**inherited_from_base, **subs}
                for tv, arg in inherited_from_base.items():
                    subs[tv] = substitute_typevars(arg, combined)
            continue

        local: dict[TypeVar, object] = {}
        for tv, arg in zip(get_type_params(origin), get_type_args(base), strict=False):
            if isinstance(tv, TypeVar):
                local[tv] = arg

        inherited: dict[TypeVar, object] = {}
        if isinstance(origin, type):
            inherited = _collect_typevar_substitutions(origin, visited)

        combined = {**inherited, **subs, **local}
        for tv, arg in local.items():
            subs[tv] = substitute_typevars(arg, combined)
        combined = {**inherited, **subs}
        for tv, arg in inherited.items():
            subs[tv] = substitute_typevars(arg, combined)

    return subs


def _resolve_typevar_substitutions(subs: dict[TypeVar, object]) -> None:
    while True:
        changed = False

        extra: dict[TypeVar, object] = {}
        for val in subs.values():
            default = typevar_default(val) if isinstance(val, TypeVar) else None
            if (
                isinstance(val, TypeVar)
                and val not in subs
                and default is not TYPEVAR_DEFAULT_MISSING
            ):
                extra[val] = default
        for tv, default in extra.items():
            if tv not in subs:
                subs[tv] = default
                changed = True

        for tv, val in list(subs.items()):
            new_val = substitute_typevars(val, subs)
            if new_val != val:
                subs[tv] = new_val
                changed = True

        if not changed:
            return


def apply_substitutions(
    hints: dict[str, object],
    subs: dict[TypeVar, object],
) -> dict[str, object]:
    if not subs:
        return hints
    return {k: substitute_typevars(v, subs) for k, v in hints.items()}


def substitute_typevars(t: object, subs: dict[TypeVar, object]) -> object:
    return substitute_typevars_inner(t, subs, set())


def _lookup_typevar_substitution(
    t: TypeVar,
    subs: dict[TypeVar, object],
) -> object:
    if t in subs:
        return subs[t]
    for candidate, replacement in subs.items():
        if candidate.__name__ == t.__name__:
            return replacement
    return TYPEVAR_SUBSTITUTION_MISSING


def _make_union_type(args: tuple[object, ...]) -> object:
    result = args[0]
    for arg in args[1:]:
        result = cast(Unionable, result) | arg
    return result


def substitute_typevars_inner(
    t: object,
    subs: dict[TypeVar, object],
    seen: set[int],
) -> object:
    if isinstance(t, TypeVar):
        if id(t) in seen:
            return t
        replacement = _lookup_typevar_substitution(t, subs)
        if replacement is TYPEVAR_SUBSTITUTION_MISSING or replacement is t:
            return t
        seen.add(id(t))
        try:
            return substitute_typevars_inner(replacement, subs, seen)
        finally:
            seen.remove(id(t))

    if isinstance(t, list):
        list_items = cast(list[object], t)
        return [substitute_typevars_inner(item, subs, seen) for item in list_items]

    if isinstance(t, tuple):
        tuple_items = cast(tuple[object, ...], t)  # ty: ignore[redundant-cast]
        return tuple(
            substitute_typevars_inner(item, subs, seen) for item in tuple_items
        )

    origin = get_origin(t)
    if origin is None:
        return t

    if origin is Annotated:
        args = get_type_args(t)
        if not args:
            return t
        inner, *metadata = args
        new_inner = substitute_typevars_inner(inner, subs, seen)
        return Annotated[new_inner, *metadata]  # pyrefly: ignore[not-a-type]

    args = get_type_args(t)
    new_args = tuple(substitute_typevars_inner(arg, subs, seen) for arg in args)
    if not new_args:
        return t

    if origin is Union or origin is PyUnionType:  # pyright: ignore[reportDeprecated]
        return _make_union_type(new_args)

    subscriptable = cast(Subscriptable, origin)
    if len(new_args) == 1:
        return subscriptable[new_args[0]]
    return subscriptable[new_args]
