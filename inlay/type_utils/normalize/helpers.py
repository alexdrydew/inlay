"""Helper functions for type normalization."""

from collections.abc import Iterable
from typing import Annotated, NotRequired, Required, cast, get_origin

from inlay._native import Qualifier
from inlay.type_utils.errors import NormalizationError
from inlay.type_utils.introspection import get_type_args


def _is_typeddict_requiredness_origin(origin: object) -> bool:
    return origin is Required or origin is NotRequired


def strip_typeddict_requiredness(t: object) -> object:
    origin = get_origin(t)
    if origin is Annotated:
        args = get_type_args(t)
        if not args:
            return t
        inner, *metadata = args
        stripped = strip_typeddict_requiredness(inner)
        if stripped is inner:
            return t
        return Annotated[stripped, *metadata]  # pyrefly: ignore[not-a-type]

    if _is_typeddict_requiredness_origin(origin):
        args = get_type_args(t)
        if len(args) != 1:
            raise NormalizationError(
                f'TypedDict field marker must wrap one type: {t!r}'
            )
        return args[0]

    return t


def typed_dict_required_optional_keys(
    origin: type,
    hints: dict[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    annotation_keys = set(hints)
    raw_required = getattr(origin, '__required_keys__', None)
    raw_optional = getattr(origin, '__optional_keys__', None)
    total = bool(getattr(origin, '__total__', True))

    if raw_required is None or raw_optional is None:
        if total:
            return tuple(sorted(annotation_keys)), ()
        return (), tuple(sorted(annotation_keys))

    required = set(cast(Iterable[str], raw_required)) & annotation_keys
    optional = set(cast(Iterable[str], raw_optional)) & annotation_keys
    missing = annotation_keys - required - optional
    if total:
        required |= missing
    else:
        optional |= missing

    return tuple(sorted(required)), tuple(sorted(optional))


def extract_qualifiers(
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
