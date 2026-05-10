"""Compile function — resolves a type against a registry and rule graph."""

import typing
from collections.abc import Callable
from typing import overload

from typing_extensions import TypeForm

from inlay._native import RegistryInstance, RuleGraph
from inlay.default import default_rules
from inlay.registry import Registry
from inlay.type_utils.normalize import normalize, normalize_callable


@overload
def compile[T](
    target: TypeForm[T],
    registry: RegistryInstance,
    rules: RuleGraph | None = None,
    *,
    solver_fixpoint_iteration_limit: int = 1024,
    solver_stack_depth_limit: int = 1024,
) -> T: ...


@overload
def compile[C: Callable[..., object]](
    target: C,
    registry: RegistryInstance,
    rules: RuleGraph | None = None,
    *,
    solver_fixpoint_iteration_limit: int = 1024,
    solver_stack_depth_limit: int = 1024,
) -> C: ...


def compile(
    target: object,
    registry: RegistryInstance,
    rules: RuleGraph | None = None,
    *,
    solver_fixpoint_iteration_limit: int = 1024,
    solver_stack_depth_limit: int = 1024,
) -> object:
    if rules is None:
        rules = default_rules()

    origin = typing.get_origin(target)
    if (
        callable(target)
        and not isinstance(target, type)
        and not isinstance(origin, type)
    ):
        return registry.compile(
            rules,
            normalize_callable(target),
            solver_fixpoint_iteration_limit=solver_fixpoint_iteration_limit,
            solver_stack_depth_limit=solver_stack_depth_limit,
        )
    return registry.compile(
        rules,
        normalize(target),
        solver_fixpoint_iteration_limit=solver_fixpoint_iteration_limit,
        solver_stack_depth_limit=solver_stack_depth_limit,
    )


@overload
def compiled[C: Callable[..., object]](fn: C, /) -> C: ...


@overload
def compiled[C: Callable[..., object]]() -> Callable[[C], C]: ...


@overload
def compiled[C: Callable[..., object]](
    registry: Registry,
) -> Callable[[C], C]: ...


def compiled[C: Callable[..., object]](
    registry: Registry | C | None = None,
) -> C | Callable[[C], C]:
    if registry is not None and not isinstance(registry, Registry):
        return compile(registry, Registry().build(), default_rules())

    registry_config = Registry() if registry is None else registry

    def decorator(fn: C) -> C:
        return compile(fn, registry_config.build(), default_rules())

    return decorator
