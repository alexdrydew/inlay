"""Compile function — resolves a type against a registry and rule graph."""

import inspect
import typing
from collections.abc import Callable
from typing import overload

from typing_extensions import TypeForm

from inlay._native import (
    CallableSignatureType,
    CallableType,
    Compiler,
    RuleGraph,
)
from inlay.default import DefaultRulesArgs
from inlay.registry import (
    _ALLOWED_IMPL_WRAPPERS,  # pyright: ignore[reportPrivateUsage]
    Registry,
    _build_callable_implementation_type,  # pyright: ignore[reportPrivateUsage]
    _build_callable_type,  # pyright: ignore[reportPrivateUsage]
    _build_callable_type_from_params,  # pyright: ignore[reportPrivateUsage]
    _CallableParam,  # pyright: ignore[reportPrivateUsage]
)
from inlay.type_utils.markers import UNQUALIFIED
from inlay.type_utils.normalize import (
    NormalizedType,
    WrapperKind,
    normalize,
    normalize_callable,
    normalize_with_qualifier,
)


@overload
def compile[T](  # type: ignore[overload-overlap]
    target: TypeForm[T],
    registry: Compiler,
    *,
    debug: bool = False,
) -> T: ...


@overload
def compile[C: Callable[..., object]](
    target: C,
    registry: Compiler,
    *,
    debug: bool = False,
) -> C: ...


def compile(
    target: object,
    registry: Compiler,
    *,
    debug: bool = False,
) -> object:
    return _compile(target, registry, debug=debug)


def _compile(
    target: object,
    registry: Compiler,
    *,
    debug: bool = False,
) -> object:
    if callable(target) and not _is_type_expression(target):
        return registry.compile(
            normalize_callable(target),
            debug=debug,
        )
    return registry.compile(
        normalize(target),
        debug=debug,
    )


@overload
def compiled[C: Callable[..., object]](
    fn: C, /, rules: RuleGraph, *, debug: bool = False
) -> C: ...


@overload
def compiled[C: Callable[..., object]](
    fn: C,
    /,
    rules: None = None,
    *,
    debug: bool = False,
    **default_rules_args: typing.Unpack[DefaultRulesArgs],
) -> C: ...


@overload
def compiled[C: Callable[..., object]](
    rules: RuleGraph, /, *, debug: bool = False
) -> Callable[[C], C]: ...


@overload
def compiled[C: Callable[..., object]](
    *, rules: RuleGraph, debug: bool = False
) -> Callable[[C], C]: ...


@overload
def compiled[C: Callable[..., object]](
    *,
    debug: bool = False,
    **default_rules_args: typing.Unpack[DefaultRulesArgs],
) -> Callable[[C], C]: ...


@overload
def compiled[C: Callable[..., object]](
    registry: Registry,
    rules: RuleGraph,
    *,
    debug: bool = False,
) -> Callable[[C], C]: ...


@overload
def compiled[C: Callable[..., object]](
    registry: Registry,
    rules: None = None,
    *,
    debug: bool = False,
    **default_rules_args: typing.Unpack[DefaultRulesArgs],
) -> Callable[[C], C]: ...


def compiled[C: Callable[..., object]](
    registry: Registry | RuleGraph | C | None = None,
    rules: RuleGraph | None = None,
    *,
    debug: bool = False,
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
        return typing.cast(
            C,
            _compile(
                registry,
                Registry().build(rules, **default_rules_args),
                debug=debug,
            ),
        )

    registry_config = Registry() if registry is None else registry

    def decorator(fn: C) -> C:
        return typing.cast(
            C,
            _compile(
                fn,
                registry_config.build(rules, **default_rules_args),
                debug=debug,
            ),
        )

    return decorator


def _return_annotation(fn: Callable[..., object], label: str) -> object:
    return_annotation = typing.cast(object, inspect.signature(fn).return_annotation)
    if return_annotation is inspect.Signature.empty:
        raise TypeError(f'{label} must have a return annotation')
    return typing.cast(
        object,
        typing.get_type_hints(fn, include_extras=True).get('return', type(None)),
    )


def _check_partial_wrapper(
    public_wrapper: WrapperKind,
    impl_wrapper: WrapperKind,
) -> None:
    if impl_wrapper not in _ALLOWED_IMPL_WRAPPERS[public_wrapper]:
        raise TypeError(
            'incompatible partial implementation wrapper: '
            + f'partial declares {public_wrapper!r} but implementation returns '
            + f'{impl_wrapper!r}'
        )


def _is_callable_stub(partial: object) -> bool:
    return callable(partial) and not _is_type_expression(partial)


def _is_type_expression(partial: object) -> bool:
    return (
        isinstance(partial, type)
        or isinstance(partial, typing.TypeAliasType)
        or typing.get_origin(partial) is not None
    )


def _build_partial_public_signature(
    partial: object,
) -> tuple[CallableSignatureType, str]:
    if _is_type_expression(partial):
        normalized = normalize_with_qualifier(partial, UNQUALIFIED)
        if not isinstance(normalized, CallableSignatureType):
            raise TypeError(
                'partial must be a callable or Callable[...] type expression'
            )
        return normalized, normalized.function_name or 'partial'

    if _is_callable_stub(partial):
        partial_callable = typing.cast(Callable[..., object], partial)
        public_return = normalize_with_qualifier(
            _return_annotation(partial_callable, 'partial'),
            UNQUALIFIED,
        )
        return (
            _build_callable_type(
                partial_callable,
                public_return,
                UNQUALIFIED,
                allow_variadics=True,
                qualifiers=UNQUALIFIED,
            ),
            getattr(partial_callable, '__name__', None) or 'partial',
        )

    normalized = normalize_with_qualifier(partial, UNQUALIFIED)
    if not isinstance(normalized, CallableSignatureType):
        raise TypeError('partial must be a callable or Callable[...] type expression')
    return normalized, normalized.function_name or 'partial'


def _build_source_binding_signature(
    source: object,
    return_type: NormalizedType,
    function_name: str,
) -> CallableSignatureType:
    return _build_callable_type_from_params(
        function_name=function_name,
        return_type=return_type,
        return_wrapper='none',
        type_params=(),
        params=(_CallableParam('source', source),),
        accepts_varargs=False,
        accepts_varkw=False,
        param_qualifiers=UNQUALIFIED,
        qualifiers=UNQUALIFIED,
    )


def _build_partial_implementation(impl: Callable[..., typing.Any]) -> CallableType:
    implementation_return = normalize_with_qualifier(
        _return_annotation(impl, 'implementation'),
        UNQUALIFIED,
    )
    return _build_callable_implementation_type(
        impl,
        implementation_return,
        UNQUALIFIED,
        allow_variadics=True,
        qualifiers=UNQUALIFIED,
    )


def _compile_source_partial[S](
    source: TypeForm[S],
    inner_signature: CallableSignatureType,
    implementation: CallableType,
    function_name: str,
    registry: Compiler,
    debug: bool,
) -> Callable[[S], Callable[..., typing.Any]]:
    outer_signature = _build_source_binding_signature(
        source, inner_signature, function_name
    )
    compiled = registry.compile_with_bound(
        outer_signature, inner_signature, implementation, debug=debug
    )
    return typing.cast(
        Callable[[S], Callable[..., typing.Any]],
        compiled,
    )


def _make_partial_explicit[S](
    source: TypeForm[S],
    partial: object,
    *,
    registry: Compiler,
    debug: bool,
) -> Callable[[Callable[..., typing.Any]], Callable[[S], Callable[..., typing.Any]]]:
    """Build a partial whose public signature is declared by ``partial``."""
    public_signature, function_name = _build_partial_public_signature(partial)
    public_wrapper = public_signature.return_wrapper

    def decorator(
        impl: Callable[..., typing.Any],
    ) -> Callable[[S], Callable[..., typing.Any]]:
        implementation = _build_partial_implementation(impl)
        impl_wrapper = implementation.signature.return_wrapper
        _check_partial_wrapper(public_wrapper, impl_wrapper)

        return _compile_source_partial(
            source, public_signature, implementation, function_name, registry, debug
        )

    return decorator


def _make_partial_implicit[S](
    source: TypeForm[S],
    *,
    registry: Compiler,
    debug: bool,
) -> Callable[[Callable[..., typing.Any]], Callable[[S], Callable[..., typing.Any]]]:
    """Build a partial whose public signature is inferred from the implementation."""

    def decorator(
        impl: Callable[..., typing.Any],
    ) -> Callable[[S], Callable[..., typing.Any]]:
        implementation = _build_partial_implementation(impl)
        function_name = getattr(impl, '__name__', None) or 'partial'
        inner_signature = _build_callable_type_from_params(
            function_name=function_name,
            return_type=implementation.signature.return_type,
            return_wrapper=implementation.signature.return_wrapper,
            type_params=(),
            params=(),
            accepts_varargs=False,
            accepts_varkw=False,
            param_qualifiers=UNQUALIFIED,
            qualifiers=UNQUALIFIED,
        )
        return _compile_source_partial(
            source, inner_signature, implementation, function_name, registry, debug
        )

    return decorator


@overload
def make_partial[S, **P, RT](
    source: TypeForm[S],
    partial: Callable[P, RT],
    *,
    registry: Compiler,
    debug: bool = False,
) -> Callable[[Callable[..., object]], Callable[[S], Callable[P, RT]]]: ...


@overload
def make_partial[S, **P, RT](
    source: TypeForm[S],
    partial: TypeForm[Callable[P, RT]],
    *,
    registry: Compiler,
    debug: bool = False,
) -> Callable[[Callable[..., object]], Callable[[S], Callable[P, RT]]]: ...


@overload
def make_partial[S, RT](
    source: TypeForm[S],
    *,
    registry: Compiler,
    debug: bool = False,
) -> Callable[[Callable[..., RT]], Callable[[S], Callable[[], RT]]]: ...


def make_partial[S](
    source: TypeForm[S],
    partial: object | None = None,
    *,
    registry: Compiler,
    debug: bool = False,
) -> Callable[[Callable[..., typing.Any]], Callable[[S], Callable[..., typing.Any]]]:
    if partial is None:
        return _make_partial_implicit(source, registry=registry, debug=debug)
    return _make_partial_explicit(source, partial, registry=registry, debug=debug)
