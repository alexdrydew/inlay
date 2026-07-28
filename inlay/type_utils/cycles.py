"""Recursive normalization cycle handling."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import NewType

from inlay._native import (
    CallableSignatureType,
    CallableType,
    CellType,
    ClassType,
    CyclePlaceholder,
    PlainType,
    ProtocolBase,
    ProtocolMethod,
    ProtocolType,
    Qualifier,
    ReadCellType,
    TypedDictType,
    UnionType,
)
from inlay.type_utils.normalized_type import NormalizedType


def deep_replace(
    root: NormalizedType,
    old: CyclePlaceholder,
    new: NormalizedType,
) -> None:
    """Recursively walk `root`, replacing `old` with `new` in all children."""
    visited: set[int] = set()
    deep_replace_walk(root, old, new, visited)


def deep_replace_walk(
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
                deep_replace_walk(arg, old, new, visited)
        case ProtocolType():
            node._replace_child(old, new)
            for tp in node.type_params:
                deep_replace_walk(tp, old, new, visited)
            for protocol in node.protocol_mro:
                deep_replace_walk(protocol, old, new, visited)
            for method in node.methods.values():
                deep_replace_walk(method, old, new, visited)
            for attribute in node.attributes.values():
                deep_replace_walk(attribute, old, new, visited)
            for property_type in node.properties.values():
                deep_replace_walk(property_type, old, new, visited)
        case ProtocolMethod():
            node._replace_child(old, new)
            deep_replace_walk(node.callable, old, new, visited)
        case ProtocolBase():
            node._replace_child(old, new)
            for tp in node.type_params:
                deep_replace_walk(tp, old, new, visited)
        case TypedDictType():
            node._replace_child(old, new)
            for tp in node.type_params:
                deep_replace_walk(tp, old, new, visited)
            for attribute in node.attributes.values():
                deep_replace_walk(attribute, old, new, visited)
        case UnionType():
            node._replace_child(old, new)
            for variant in node.variants:
                deep_replace_walk(variant, old, new, visited)
        case CallableSignatureType():
            node._replace_child(old, new)
            for p in node.params:
                deep_replace_walk(p, old, new, visited)
            deep_replace_walk(node.return_type, old, new, visited)
        case CallableType():
            node._replace_child(old, new)
            deep_replace_walk(node.signature, old, new, visited)
        case ClassType():
            node._replace_child(old, new)
            for arg in node.args:
                deep_replace_walk(arg, old, new, visited)
            init_params = node.init_params
            if init_params is not None:
                for p in init_params:
                    deep_replace_walk(p, old, new, visited)
        case ReadCellType() | CellType():
            node._replace_child(old, new)
            deep_replace_walk(node.target, old, new, visited)
        case _:
            pass


_TypeId = NewType('_TypeId', int)

type _ActiveCacheEntry = tuple[Qualifier, list[CyclePlaceholder]]
type NormalizationStack = dict[_TypeId, _ActiveCacheEntry]
type NormMemo = dict[tuple[_TypeId, Qualifier], NormalizedType]


@contextmanager
def active_normalization_entry(
    stack: NormalizationStack,
    key: _TypeId,
    qualifiers: Qualifier,
) -> Generator[list[CyclePlaceholder]]:
    placeholders: list[CyclePlaceholder] = []
    stack[key] = (qualifiers, placeholders)
    try:
        yield placeholders
    finally:
        del stack[key]


class IdInterner:
    """Keep annotation objects alive while their ``id`` is used as a cache key."""

    def __init__(self) -> None:
        self._roots: dict[_TypeId, object] = {}

    def get_id(self, obj: object) -> _TypeId:
        # CPython only guarantees id uniqueness among live objects. Normalization
        # creates temporary typing aliases/unions during substitution; if one is
        # freed, a later unrelated type object can reuse its id and hit the wrong
        # cache entry. Keeping keyed objects alive makes id-based keys safe for the
        # duration of this normalization pass, then the interner is dropped.
        key = _TypeId(id(obj))
        _ = self._roots.setdefault(key, obj)
        return key
