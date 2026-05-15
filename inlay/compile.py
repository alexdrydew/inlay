"""Compile function — resolves a type against a registry and rule graph."""

import typing
from collections.abc import Callable
from typing import overload

from typing_extensions import TypeForm

from inlay._native import RegistryInstance, RuleGraph
from inlay.default import DefaultRulesArgs, default_rules
from inlay.registry import Registry
from inlay.type_utils.normalize import normalize, normalize_callable


@overload
def compile[T](
    target: TypeForm[T],
    registry: RegistryInstance,
    rules: RuleGraph,
    *,
    solver_fixpoint_iteration_limit: int = 1024,
    solver_stack_depth_limit: int = 1024,
) -> T: ...


@overload
def compile[T](
    target: TypeForm[T],
    registry: RegistryInstance,
    rules: None = None,
    *,
    solver_fixpoint_iteration_limit: int = 1024,
    solver_stack_depth_limit: int = 1024,
    **default_rules_args: typing.Unpack[DefaultRulesArgs],
) -> T: ...


@overload
def compile[C: Callable[..., object]](
    target: C,
    registry: RegistryInstance,
    rules: RuleGraph,
    *,
    solver_fixpoint_iteration_limit: int = 1024,
    solver_stack_depth_limit: int = 1024,
) -> C: ...


@overload
def compile[C: Callable[..., object]](
    target: C,
    registry: RegistryInstance,
    rules: None = None,
    *,
    solver_fixpoint_iteration_limit: int = 1024,
    solver_stack_depth_limit: int = 1024,
    **default_rules_args: typing.Unpack[DefaultRulesArgs],
) -> C: ...


def compile(
    target: object,
    registry: RegistryInstance,
    rules: RuleGraph | None = None,
    *,
    solver_fixpoint_iteration_limit: int = 1024,
    solver_stack_depth_limit: int = 1024,
    **default_rules_args: typing.Unpack[DefaultRulesArgs],
) -> object:
    rules = _select_rules(rules, default_rules_args)

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


def _select_rules(
    rules: RuleGraph | None,
    default_rules_args: DefaultRulesArgs,
) -> RuleGraph:
    if rules is not None:
        if default_rules_args:
            raise TypeError(
                'default rule arguments cannot be combined with explicit rules'
            )
        return rules
    return default_rules(**default_rules_args)


@overload
def compiled[C: Callable[..., object]](fn: C, /, rules: RuleGraph) -> C: ...


@overload
def compiled[C: Callable[..., object]](
    fn: C, /, rules: None = None, **default_rules_args: typing.Unpack[DefaultRulesArgs]
) -> C: ...


@overload
def compiled[C: Callable[..., object]](rules: RuleGraph, /) -> Callable[[C], C]: ...


@overload
def compiled[C: Callable[..., object]](*, rules: RuleGraph) -> Callable[[C], C]: ...


@overload
def compiled[C: Callable[..., object]](
    **default_rules_args: typing.Unpack[DefaultRulesArgs],
) -> Callable[[C], C]: ...


@overload
def compiled[C: Callable[..., object]](
    registry: Registry,
    rules: RuleGraph,
) -> Callable[[C], C]: ...


@overload
def compiled[C: Callable[..., object]](
    registry: Registry,
    rules: None = None,
    **default_rules_args: typing.Unpack[DefaultRulesArgs],
) -> Callable[[C], C]: ...


def compiled[C: Callable[..., object]](
    registry: Registry | RuleGraph | C | None = None,
    rules: RuleGraph | None = None,
    **default_rules_args: typing.Unpack[DefaultRulesArgs],
) -> C | Callable[[C], C]:
    if isinstance(registry, RuleGraph):
        if rules is not None:
            raise TypeError('rules provided twice')
        rules = registry
        registry = None

    if registry is not None and not isinstance(registry, Registry):
        selected_rules = _select_rules(rules, default_rules_args)
        return compile(
            registry,
            Registry().build(),
            selected_rules,
        )

    registry_config = Registry() if registry is None else registry
    selected_rules = _select_rules(rules, default_rules_args)

    def decorator(fn: C) -> C:
        return compile(fn, registry_config.build(), selected_rules)

    return decorator
