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
    *,
    solver_fixpoint_iteration_limit: int = 1024,
    solver_stack_depth_limit: int = 1024,
) -> T: ...


@overload
def compile[C: Callable[..., object]](
    target: C,
    registry: Registry,
    rules: RuleGraph,
    *,
    solver_fixpoint_iteration_limit: int = 1024,
    solver_stack_depth_limit: int = 1024,
) -> C: ...


def compile(
    target: object,
    registry: Registry,
    rules: RuleGraph,
    *,
    solver_fixpoint_iteration_limit: int = 1024,
    solver_stack_depth_limit: int = 1024,
) -> object:
    if callable(target) and not isinstance(target, type):
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
