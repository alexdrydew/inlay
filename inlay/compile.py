"""Compile function — resolves a type against a registry and rule graph."""

import inspect
import typing
from collections.abc import Callable
from typing import overload

from typing_extensions import TypeForm

from inlay._native import (
    CallableBindingType,
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
        return compile(  # pyrefly: ignore[no-matching-overload]
            registry,
            Registry().build(rules, **default_rules_args),
        )

    registry_config = Registry() if registry is None else registry

    def decorator(fn: C) -> C:
        return compile(  # pyrefly: ignore[no-matching-overload]
            fn, registry_config.build(rules, **default_rules_args)
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


def _replace_callable_return(
    signature: CallableSignatureType,
    return_type: NormalizedType,
) -> CallableSignatureType:
    return CallableSignatureType(
        params=signature.params,
        param_names=signature.param_names,
        param_kinds=signature.param_kinds,
        return_type=return_type,
        return_wrapper=signature.return_wrapper,
        type_params=signature.type_params,
        qualifiers=signature.qualifiers,
        function_name=signature.function_name,
        accepts_varargs=signature.accepts_varargs,
        accepts_varkw=signature.accepts_varkw,
    )


def _build_callable_binding(
    public_signature: CallableSignatureType,
    implementation_signature: CallableSignatureType,
) -> CallableBindingType:
    _check_partial_wrapper(
        public_signature.return_wrapper,
        implementation_signature.return_wrapper,
    )
    public_return = public_signature.return_type
    implementation_return = implementation_signature.return_type
    if isinstance(public_return, CallableSignatureType) or isinstance(
        implementation_return, CallableSignatureType
    ):
        if not isinstance(public_return, CallableSignatureType) or not isinstance(
            implementation_return, CallableSignatureType
        ):
            raise TypeError('callable return types must match in partial binding')
        nested_binding = _build_callable_binding(public_return, implementation_return)
        public_signature = _replace_callable_return(public_signature, nested_binding)

    return CallableBindingType(
        public_signature=public_signature,
        implementation=implementation_signature,
        qualifiers=UNQUALIFIED,
    )


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
) -> Callable[[S], Callable[..., typing.Any]]:
    binding = CallableBindingType(
        public_signature=inner_signature,
        implementation=implementation,
        qualifiers=UNQUALIFIED,
    )
    outer_signature = _build_source_binding_signature(source, binding, function_name)
    return typing.cast(
        Callable[[S], Callable[..., typing.Any]], registry.compile(outer_signature)
    )


def _make_partial_explicit[S](
    source: TypeForm[S],
    partial: Callable[..., object],
    *,
    registry: Compiler,
) -> Callable[[Callable[..., typing.Any]], Callable[[S], Callable[..., typing.Any]]]:
    """Build a partial whose public signature is declared by ``partial``."""
    public_return = normalize_with_qualifier(
        _return_annotation(partial, 'partial'),
        UNQUALIFIED,
    )
    public_signature = _build_callable_type(
        partial,
        public_return,
        UNQUALIFIED,
        allow_variadics=True,
        qualifiers=UNQUALIFIED,
    )
    public_wrapper = public_signature.return_wrapper
    function_name = getattr(partial, '__name__', None) or 'partial'

    def decorator(
        impl: Callable[..., typing.Any],
    ) -> Callable[[S], Callable[..., typing.Any]]:
        implementation = _build_partial_implementation(impl)
        impl_wrapper = implementation.signature.return_wrapper
        _check_partial_wrapper(public_wrapper, impl_wrapper)

        impl_return = implementation.signature.return_type
        if isinstance(impl_return, CallableSignatureType):
            nested_binding = _build_callable_binding(public_signature, impl_return)
            outer_signature = _build_source_binding_signature(
                source, nested_binding, function_name
            )
            _check_partial_wrapper(outer_signature.return_wrapper, impl_wrapper)
            binding = CallableBindingType(
                public_signature=outer_signature,
                implementation=implementation,
                qualifiers=UNQUALIFIED,
            )
            return typing.cast(
                Callable[[S], Callable[..., typing.Any]], registry.compile(binding)
            )

        return _compile_source_partial(
            source, public_signature, implementation, function_name, registry
        )

    return decorator


def _make_partial_implicit[S](
    source: TypeForm[S],
    *,
    registry: Compiler,
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
            source, inner_signature, implementation, function_name, registry
        )

    return decorator


@overload
def make_partial[S, **P, RT](
    source: TypeForm[S],
    partial: Callable[P, RT],
    *,
    registry: Compiler,
) -> Callable[[Callable[..., object]], Callable[[S], Callable[P, RT]]]: ...


@overload
def make_partial[S, RT](
    source: TypeForm[S],
    *,
    registry: Compiler,
) -> Callable[[Callable[..., RT]], Callable[[S], Callable[[], RT]]]: ...


def make_partial[S](
    source: TypeForm[S],
    partial: Callable[..., object] | None = None,
    *,
    registry: Compiler,
) -> Callable[[Callable[..., typing.Any]], Callable[[S], Callable[..., typing.Any]]]:
    if partial is None:
        return _make_partial_implicit(source, registry=registry)
    return _make_partial_explicit(source, partial, registry=registry)
