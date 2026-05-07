"""Immutable registry of constructors and methods.

Registration is lazy: entries store raw Python type hints and qualifiers.
Normalization and validation happen during build().
"""

import annotationlib
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

from typing_extensions import TypeForm

from inlay._native import (
    CallableType,
    Qualifier,
    Registry,
)
from inlay.type_utils.errors import UnsupportedVariadicParameterError
from inlay.type_utils.markers import UNQUALIFIED
from inlay.type_utils.normalize import (
    NormalizedType,
    get_callable_info,
    normalize_with_qualifier,
    unwrap_return_type,
)

# --- Entry types (lazy — raw types, no normalization) ---


type ConstructorEntryOperation = typing.Literal['register', 'override']


@dataclass(frozen=True)
class ConstructorEntry:
    constructor: Callable[..., object]
    target_type: object
    provides: Qualifier
    requires: Qualifier = UNQUALIFIED
    operation: ConstructorEntryOperation = 'register'
    provides_from_requires: bool = False


@dataclass(frozen=True)
class MethodEntry:
    method: Callable[..., object]
    implementation: Callable[..., object]
    provides: Qualifier
    requires: Qualifier
    bound_to: type | None


# --- Registrar protocols ---


class _ConstructorRegistrar[T](typing.Protocol):
    def __call__(self, constructor: Callable[..., T]) -> RegistryBuilder: ...


class _ValueRegistrar[T](typing.Protocol):
    def __call__(self, value: T) -> RegistryBuilder: ...


class _AliasRegistrar[T](typing.Protocol):
    @typing.overload
    def __call__(
        self,
        source_type: TypeForm[object],
        qualifiers: Qualifier,
        *,
        requires: None = None,
    ) -> RegistryBuilder: ...

    @typing.overload
    def __call__(
        self,
        source_type: TypeForm[object],
        qualifiers: None = None,
        *,
        requires: Qualifier | None = None,
    ) -> RegistryBuilder: ...

    def __call__(
        self,
        source_type: TypeForm[object],
        qualifiers: Qualifier | None = None,
        *,
        requires: Qualifier | None = None,
    ) -> RegistryBuilder: ...


# --- Helpers ---


def _intersect_qualifiers(a: Qualifier, b: Qualifier) -> Qualifier:
    if not a.is_qualified:
        return b
    if not b.is_qualified:
        return a
    return a & b


@dataclass(frozen=True)
class _QualifierSplit:
    provides: Qualifier
    requires: Qualifier


def _split_qualifiers(
    *,
    qualifiers: Qualifier | None,
    provides: Qualifier | None,
    requires: Qualifier | None,
) -> _QualifierSplit:
    if qualifiers is not None and (provides is not None or requires is not None):
        raise ValueError('qualifiers cannot be combined with provides or requires')

    if qualifiers is not None:
        return _QualifierSplit(provides=qualifiers, requires=qualifiers)

    return _QualifierSplit(
        provides=provides if provides is not None else UNQUALIFIED,
        requires=requires if requires is not None else UNQUALIFIED,
    )


def _split_requires_only(
    *,
    qualifiers: Qualifier | None,
    requires: Qualifier | None,
) -> Qualifier:
    if qualifiers is not None and requires is not None:
        raise ValueError('qualifiers cannot be combined with requires')

    if qualifiers is not None:
        return qualifiers
    if requires is not None:
        return requires
    return UNQUALIFIED


# --- Built entry types (produced by build(), consumed by Rust converter) ---


@dataclass(frozen=True)
class BuiltConstructorEntry:
    callable_type: CallableType
    constructor: Callable[..., object]


@dataclass(frozen=True)
class BuiltMethodEntry:
    public_callable_type: CallableType
    implementation_callable_type: CallableType
    implementation: Callable[..., object]
    bound_to: NormalizedType | None
    order: int


# --- Registry ---


@dataclass(frozen=True)
class RegistryBuilder:
    constructors: tuple[ConstructorEntry, ...] = ()
    methods: dict[str, tuple[MethodEntry, ...]] = field(default_factory=dict)

    @typing.overload
    def register[T](
        self,
        target_type: TypeForm[T],
        qualifiers: Qualifier,
        *,
        provides: None = None,
        requires: None = None,
    ) -> _ConstructorRegistrar[T]: ...

    @typing.overload
    def register[T](
        self,
        target_type: TypeForm[T],
        qualifiers: None = None,
        *,
        provides: Qualifier | None = None,
        requires: Qualifier | None = None,
    ) -> _ConstructorRegistrar[T]: ...

    def register[T](
        self,
        target_type: TypeForm[T],
        qualifiers: Qualifier | None = None,
        *,
        provides: Qualifier | None = None,
        requires: Qualifier | None = None,
    ) -> _ConstructorRegistrar[T]:
        split = _split_qualifiers(
            qualifiers=qualifiers,
            provides=provides,
            requires=requires,
        )

        def decorator(constructor: Callable[..., object]) -> RegistryBuilder:
            entry = ConstructorEntry(
                constructor=constructor,
                target_type=target_type,
                provides=split.provides,
                requires=split.requires,
            )
            return RegistryBuilder(
                (*self.constructors, entry),
                dict(self.methods),
            )

        return decorator

    @typing.overload
    def override[T](
        self,
        target_type: TypeForm[T],
        qualifiers: Qualifier,
        *,
        provides: None = None,
        requires: None = None,
    ) -> _ConstructorRegistrar[T]: ...

    @typing.overload
    def override[T](
        self,
        target_type: TypeForm[T],
        qualifiers: None = None,
        *,
        provides: Qualifier | None = None,
        requires: Qualifier | None = None,
    ) -> _ConstructorRegistrar[T]: ...

    def override[T](
        self,
        target_type: TypeForm[T],
        qualifiers: Qualifier | None = None,
        *,
        provides: Qualifier | None = None,
        requires: Qualifier | None = None,
    ) -> _ConstructorRegistrar[T]:
        split = _split_qualifiers(
            qualifiers=qualifiers,
            provides=provides,
            requires=requires,
        )

        def decorator(constructor: Callable[..., object]) -> RegistryBuilder:
            entry = ConstructorEntry(
                constructor=constructor,
                target_type=target_type,
                provides=split.provides,
                requires=split.requires,
                operation='override',
            )
            return RegistryBuilder(
                (*self.constructors, entry),
                dict(self.methods),
            )

        return decorator

    @typing.overload
    def register_factory(
        self,
        factory: Callable[..., object],
        *,
        qualifiers: Qualifier,
        provides: None = None,
        requires: None = None,
    ) -> RegistryBuilder: ...

    @typing.overload
    def register_factory(
        self,
        factory: Callable[..., object],
        *,
        qualifiers: None = None,
        provides: Qualifier | None = None,
        requires: Qualifier | None = None,
    ) -> RegistryBuilder: ...

    def register_factory(
        self,
        factory: Callable[..., object],
        *,
        qualifiers: Qualifier | None = None,
        provides: Qualifier | None = None,
        requires: Qualifier | None = None,
    ) -> RegistryBuilder:
        split = _split_qualifiers(
            qualifiers=qualifiers,
            provides=provides,
            requires=requires,
        )
        hints = typing.get_type_hints(factory, include_extras=True)
        target_type: object = hints.get('return', type(None))  # pyright: ignore[reportAny]
        entry = ConstructorEntry(
            constructor=factory,
            target_type=target_type,
            provides=split.provides,
            requires=split.requires,
        )
        return RegistryBuilder(
            (*self.constructors, entry),
            dict(self.methods),
        )

    def register_value[T](
        self,
        target_type: TypeForm[T],
        qualifiers: Qualifier | None = None,
    ) -> _ValueRegistrar[T]:
        def decorator(value: T) -> RegistryBuilder:
            def _constructor() -> T:
                return value

            _constructor.__annotations__ = {'return': target_type}
            _constructor.__name__ = f'value_{getattr(target_type, "__name__", "type")}'
            if qualifiers is None:
                return self.register(target_type)(_constructor)
            return self.register(target_type, qualifiers)(_constructor)

        return decorator

    @typing.overload
    def register_alias[T](
        self,
        target_type: TypeForm[T],
        qualifiers: Qualifier,
        *,
        provides: None = None,
        requires: None = None,
    ) -> _AliasRegistrar[T]: ...

    @typing.overload
    def register_alias[T](
        self,
        target_type: TypeForm[T],
        qualifiers: None = None,
        *,
        provides: Qualifier | None = None,
        requires: Qualifier | None = None,
    ) -> _AliasRegistrar[T]: ...

    def register_alias[T](
        self,
        target_type: TypeForm[T],
        qualifiers: Qualifier | None = None,
        *,
        provides: Qualifier | None = None,
        requires: Qualifier | None = None,
    ) -> _AliasRegistrar[T]:
        split = _split_qualifiers(
            qualifiers=qualifiers,
            provides=provides,
            requires=requires,
        )

        def _alias(
            source_type: TypeForm[object],
            qualifiers: Qualifier | None = None,
            *,
            requires: Qualifier | None = None,
        ) -> RegistryBuilder:
            source_requires = _split_requires_only(
                qualifiers=qualifiers,
                requires=requires,
            )
            param_annotation: object = (
                typing.Annotated[source_type, source_requires]
                if source_requires.is_qualified
                else source_type
            )

            def _constructor(value: object) -> object:
                return value

            _constructor.__annotations__ = {
                'value': param_annotation,
                'return': target_type,
            }
            _constructor.__name__ = f'alias_{getattr(target_type, "__name__", "type")}'

            return self.register(
                target_type,
                provides=split.provides,
                requires=split.requires,
            )(typing.cast(Callable[..., T], _constructor))

        return _alias

    @typing.overload
    def register_method(
        self,
        protocol: type,
        method: Callable[..., object],
        /,
        *,
        qualifiers: Qualifier,
        provides: None = None,
        requires: None = None,
    ) -> Callable[[type | Callable[..., object]], RegistryBuilder]: ...

    @typing.overload
    def register_method(
        self,
        protocol: type,
        method: Callable[..., object],
        /,
        *,
        qualifiers: None = None,
        provides: Qualifier | None = None,
        requires: Qualifier | None = None,
    ) -> Callable[[type | Callable[..., object]], RegistryBuilder]: ...

    def register_method(
        self,
        protocol: type,
        method: Callable[..., object],
        /,
        *,
        qualifiers: Qualifier | None = None,
        provides: Qualifier | None = None,
        requires: Qualifier | None = None,
    ) -> Callable[[type | Callable[..., object]], RegistryBuilder]:
        split = _split_qualifiers(
            qualifiers=qualifiers,
            provides=provides,
            requires=requires,
        )
        public_method = method
        method_name = public_method.__name__

        origin = typing.get_origin(protocol) or protocol
        if not isinstance(origin, type) or not typing.is_protocol(origin):
            raise TypeError(f'{origin} is not a Protocol')

        if method_name not in typing.get_protocol_members(origin):
            raise ValueError(f"'{method_name}' is not a member of {origin}")

        def decorator(impl: type | Callable[..., object]) -> RegistryBuilder:
            if callable(impl) and not isinstance(impl, type):
                bound_to = None
                base = self
            else:
                impl_method = getattr(impl, method_name)  # pyright: ignore[reportAny]
                if not callable(impl_method):  # pyright: ignore[reportAny]
                    raise ValueError(f'{impl}.{method_name} is not callable')
                bound_to = impl
                if any(e.constructor is impl for e in self.constructors):
                    base = self
                else:
                    constructor_entry = ConstructorEntry(
                        constructor=impl,
                        target_type=impl,
                        provides=split.requires,
                        requires=split.requires,
                        provides_from_requires=True,
                    )
                    base = RegistryBuilder(
                        (*self.constructors, constructor_entry),
                        dict(self.methods),
                    )

            entry = MethodEntry(
                method=public_method,
                implementation=impl,
                provides=split.provides,
                requires=split.requires,
                bound_to=bound_to,
            )

            new_methods = dict(base.methods)
            existing = new_methods.get(method_name, ())
            new_methods[method_name] = (*existing, entry)
            return RegistryBuilder(base.constructors, new_methods)

        return decorator

    @typing.overload
    def include(
        self,
        other: RegistryBuilder,
        /,
        *others: RegistryBuilder,
        qualifiers: Qualifier,
        provides: None = None,
        requires: None = None,
    ) -> RegistryBuilder: ...

    @typing.overload
    def include(
        self,
        other: RegistryBuilder,
        /,
        *others: RegistryBuilder,
        qualifiers: None = None,
        provides: Qualifier | None = None,
        requires: Qualifier | None = None,
    ) -> RegistryBuilder: ...

    def include(
        self,
        other: RegistryBuilder,
        /,
        *others: RegistryBuilder,
        qualifiers: Qualifier | None = None,
        provides: Qualifier | None = None,
        requires: Qualifier | None = None,
    ) -> RegistryBuilder:
        split = _split_qualifiers(
            qualifiers=qualifiers,
            provides=provides,
            requires=requires,
        )
        result = self

        for reg in (other, *others):
            qualified_constructors = tuple(
                ConstructorEntry(
                    constructor=e.constructor,
                    target_type=e.target_type,
                    provides=_intersect_qualifiers(
                        e.provides,
                        split.requires if e.provides_from_requires else split.provides,
                    ),
                    requires=_intersect_qualifiers(e.requires, split.requires),
                    operation=e.operation,
                    provides_from_requires=e.provides_from_requires,
                )
                for e in reg.constructors
            )
            new_constructors = (*result.constructors, *qualified_constructors)

            new_methods = dict(result.methods)
            for method_name, entries in reg.methods.items():
                qualified = tuple(
                    MethodEntry(
                        method=e.method,
                        implementation=e.implementation,
                        provides=_intersect_qualifiers(e.provides, split.provides),
                        bound_to=e.bound_to,
                        requires=_intersect_qualifiers(e.requires, split.requires),
                    )
                    for e in entries
                )
                existing = new_methods.get(method_name, ())
                new_methods[method_name] = (*existing, *qualified)

            result = RegistryBuilder(new_constructors, new_methods)

        return result

    def build(self) -> Registry:
        built_constructors = _build_constructors(self.constructors)
        built_methods = _build_methods(self.methods)
        return Registry(
            _BuiltRegistry(
                constructors=built_constructors,
                methods=built_methods,
            )
        )


# --- Build helpers ---


@dataclass(frozen=True)
class _BuiltRegistry:
    constructors: tuple[BuiltConstructorEntry, ...]
    methods: dict[str, tuple[BuiltMethodEntry, ...]]


def _build_callable_type(
    fn: Callable[..., object],
    return_type: NormalizedType,
    param_qualifiers: Qualifier,
    *,
    skip_self: bool = False,
    allow_variadics: bool = True,
    qualifiers: Qualifier = UNQUALIFIED,
) -> CallableType:
    info = get_callable_info(
        fn,
        skip_self=skip_self,
        allow_variadics=allow_variadics,
    )
    if isinstance(fn, type):
        hints = annotationlib.get_annotations(
            fn.__init__, format=annotationlib.Format.FORWARDREF
        )
    else:
        hints = annotationlib.get_annotations(
            fn, format=annotationlib.Format.FORWARDREF
        )

    params: list[NormalizedType] = []
    for p in info.params:
        raw_hint = hints.get(p.name)
        if raw_hint is not None:
            params.append(normalize_with_qualifier(raw_hint, param_qualifiers))
        else:
            params.append(p.type)

    unwrapped_return, return_wrapper = unwrap_return_type(return_type)
    if return_wrapper == 'none':
        return_wrapper = info.return_wrapper
    fn_name = fn.__name__
    return CallableType(
        params=tuple(params),
        param_names=tuple(p.name for p in info.params),
        param_kinds=tuple(p.kind for p in info.params),
        return_type=unwrapped_return,
        return_wrapper=return_wrapper,
        type_params=info.type_params,
        qualifiers=qualifiers,
        function_name=fn_name,
        param_has_default=[p.has_default for p in info.params],
        accepts_varargs=info.accepts_varargs,
        accepts_varkw=info.accepts_varkw,
    )


def _build_constructors(
    entries: tuple[ConstructorEntry, ...],
) -> tuple[BuiltConstructorEntry, ...]:
    built: list[BuiltConstructorEntry] = []
    for entry in _select_constructor_entries(entries):
        try:
            built.append(_build_constructor(entry))
        except UnsupportedVariadicParameterError:
            # Open-ended constructor params cannot be satisfied by DI.
            continue
    return tuple(built)


def _constructor_target_key(entry: ConstructorEntry) -> NormalizedType:
    return normalize_with_qualifier(entry.target_type, entry.provides)


def _select_constructor_entries(
    entries: tuple[ConstructorEntry, ...],
) -> tuple[ConstructorEntry, ...]:
    groups: list[tuple[NormalizedType, list[ConstructorEntry]]] = []

    for entry in entries:
        key = _constructor_target_key(entry)
        for existing_key, group in groups:
            if existing_key == key:
                group.append(entry)
                break
        else:
            groups.append((key, [entry]))

    selected: list[ConstructorEntry] = []
    for _, group in groups:
        overrides = [entry for entry in group if entry.operation == 'override']
        selected.extend(overrides or group)

    return tuple(selected)


def _build_constructor(entry: ConstructorEntry) -> BuiltConstructorEntry:
    return_type = normalize_with_qualifier(entry.target_type, entry.provides)
    callable_type = _build_callable_type(
        entry.constructor,
        return_type,
        entry.requires,
        allow_variadics=False,
        qualifiers=entry.provides,
    )
    return BuiltConstructorEntry(
        callable_type=callable_type,
        constructor=entry.constructor,
    )


def build_constructor_entry(entry: ConstructorEntry) -> BuiltConstructorEntry:
    return _build_constructor(entry)


def _build_methods(
    methods: dict[str, tuple[MethodEntry, ...]],
) -> dict[str, tuple[BuiltMethodEntry, ...]]:
    order = 0
    built: dict[str, tuple[BuiltMethodEntry, ...]] = {}
    for name, entries in methods.items():
        method_entries: list[BuiltMethodEntry] = []
        for entry in entries:
            method_entries.append(_build_method(entry, name, order))
            order += 1
        built[name] = tuple(method_entries)
    return built


def _build_method(entry: MethodEntry, method_name: str, order: int) -> BuiltMethodEntry:
    is_class_impl = entry.bound_to is not None
    method_func: Callable[..., object] | None = None
    output_qualifiers = _intersect_qualifiers(entry.provides, entry.requires)
    public_return_hint: object = typing.get_type_hints(entry.method).get(  # pyright: ignore[reportAny]
        'return', type(None)
    )
    public_callable_type = _build_callable_type(
        entry.method,
        normalize_with_qualifier(public_return_hint, output_qualifiers),
        entry.requires,
        skip_self=True,
        qualifiers=entry.requires,
    )

    if is_class_impl:
        # For class-based implementations, build callable from the actual
        # method (e.g. MongoUowTransition.with_write), not the class
        # constructor.  Constructor dependencies are resolved separately
        # via bound_to.
        method_func = cast(
            Callable[..., object], getattr(entry.implementation, method_name)
        )
        return_hint: object = typing.get_type_hints(method_func).get(  # pyright: ignore[reportAny]
            'return', type(None)
        )
        implementation_callable_type = _build_callable_type(
            method_func,
            normalize_with_qualifier(return_hint, output_qualifiers),
            entry.requires,
            skip_self=True,
            qualifiers=entry.requires,
        )
    else:
        impl_func = entry.implementation
        return_hint = typing.get_type_hints(impl_func).get(  # pyright: ignore[reportAny]
            'return', type(None)
        )
        implementation_callable_type = _build_callable_type(
            impl_func,
            normalize_with_qualifier(return_hint, output_qualifiers),
            entry.requires,
            qualifiers=entry.requires,
        )

    bound_to = (
        normalize_with_qualifier(entry.bound_to, entry.requires)
        if entry.bound_to is not None
        else None
    )
    implementation: type | Callable[..., object]
    if is_class_impl:
        # Runtime calls implementation(bound_instance, *args).  For class-based
        # impls this must be the unbound method so the call becomes
        # Class.method(instance, *args), not Class(instance, *args).
        assert method_func is not None, 'class-based method implementation must exist'
        implementation = method_func
    else:
        implementation = entry.implementation

    return BuiltMethodEntry(
        public_callable_type=public_callable_type,
        implementation_callable_type=implementation_callable_type,
        implementation=implementation,
        bound_to=bound_to,
        order=order,
    )
