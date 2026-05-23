"""Compile function — resolves a type against a registry and rule graph."""

import typing
from collections.abc import Callable
from typing import overload

from typing_extensions import TypeForm

from inlay._native import Compiler, RuleGraph
from inlay.default import DefaultRulesArgs
from inlay.registry import Registry
from inlay.type_utils.normalize import normalize, normalize_callable


@overload
def compile[T](
    target: TypeForm[T],
    registry: Compiler,
) -> T: ...


@overload
def compile[C: Callable[..., object]](
    target: C,
    registry: Compiler,
) -> C: ...


def compile(
    target: object,
    registry: Compiler,
) -> object:
    origin = typing.get_origin(target)
    if (
        callable(target)
        and not isinstance(target, type)
        and not isinstance(origin, type)
    ):
        return registry.compile(
            normalize_callable(target),
        )
    return registry.compile(
        normalize(target),
    )


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

    if rules is not None and default_rules_args:
        raise TypeError('default rule arguments cannot be combined with explicit rules')

    if registry is not None and not isinstance(registry, Registry):
        return compile(
            registry,
            Registry().build(rules, **default_rules_args),
        )

    registry_config = Registry() if registry is None else registry

    def decorator(fn: C) -> C:
        return compile(fn, registry_config.build(rules, **default_rules_args))

    return decorator
