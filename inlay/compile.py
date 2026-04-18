"""Compile function — resolves a type against a registry and rule graph."""

from collections.abc import Callable
from typing import overload

from typing_extensions import TypeForm

from inlay._native import Registry, RuleGraph
from inlay.type_utils.normalize import normalize, normalize_callable


@overload
def compile[T](
    target: TypeForm[T],
    registry: Registry,
    rules: RuleGraph,
) -> T: ...


@overload
def compile[C: Callable[..., object]](
    target: C,
    registry: Registry,
    rules: RuleGraph,
) -> C: ...


def compile(
    target: object,
    registry: Registry,
    rules: RuleGraph,
) -> object:
    if callable(target) and not isinstance(target, type):
        return registry.compile(rules, normalize_callable(target))
    return registry.compile(rules, normalize(target))
