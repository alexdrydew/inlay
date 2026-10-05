from typing import Annotated, Protocol, Self

import pytest

from inlay import ProtocolType, normalize, qual
from inlay.type_utils.normalize import (
    normalize_with_qualifier,
    normalize_with_self_type,
)


class _Owner[T](Protocol):
    @property
    def value(self) -> T: ...

    @property
    def peer(self) -> Self: ...


@pytest.mark.parametrize('owner', [_Owner[int], _Owner[str]])
def test_self_substitution_reuses_normalization_without_losing_owner(
    owner: object,
) -> None:
    result = normalize_with_self_type(Self, qual(), owner)

    assert result is normalize(owner)
    assert isinstance(result, ProtocolType)
    assert result.properties['peer'] is result


def test_self_substitution_preserves_qualifiers_and_owner() -> None:
    annotation = Annotated[Self, qual('scoped')]  # pyright: ignore[reportGeneralTypeIssues]
    integer = normalize_with_self_type(annotation, qual(), _Owner[int])
    string = normalize_with_self_type(annotation, qual(), _Owner[str])

    assert integer is normalize_with_self_type(annotation, qual(), _Owner[int])
    assert isinstance(integer, ProtocolType)
    assert isinstance(string, ProtocolType)
    assert integer.qualifiers == string.qualifiers == qual('scoped')
    assert integer.properties['value'] == normalize_with_qualifier(int, qual('scoped'))
    assert string.properties['value'] == normalize_with_qualifier(str, qual('scoped'))
    assert integer.properties['peer'] is integer
    assert string.properties['peer'] is string


def test_self_substitution_keeps_unhashable_metadata_supported() -> None:
    annotation = Annotated[Self, []]  # pyright: ignore[reportGeneralTypeIssues]
    result = normalize_with_self_type(annotation, qual(), _Owner[int])

    assert isinstance(result, ProtocolType)
    assert result.origin is _Owner
    assert result.properties['value'] == normalize(int)
    assert result.properties['peer'] is result
