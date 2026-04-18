"""Markers for type annotations."""

from typing import Annotated, Protocol, get_args, get_origin

from inlay._native import Qualifier

_CONTEXT_QUALIFIER_ATTR = '__context_qualifier__'


class LazyRef[T](Protocol):
    """Lazy reference to a resolvable type T.

    Use when you need T but can't receive it during construction
    (e.g., to break a dependency cycle).

    Guarantees:
    - If compilation succeeds, .get() will never fail after context is built
    - .get() MUST NOT be called during construction (raises LazyRefAccessError)
    """

    def get(self) -> T: ...


qual = Qualifier

UNQUALIFIED = Qualifier()


def qualifier[T](*tags: str) -> T:
    """Class decorator that attaches a qualifier to a type.

    When the context DSL normalizer encounters this type in an annotation,
    it automatically intersects the class qualifier with the current scope
    qualifier.

    Usage::

        @qualifier('transaction')
        class InMemoryTransaction(Transaction):
            ...

    This is equivalent to writing
    ``Annotated[InMemoryTransaction, qual('transaction')]``
    everywhere the type appears, but without modifying each use site.
    """
    q = Qualifier(*tags)

    def decorator(cls: T) -> T:
        setattr(cls, _CONTEXT_QUALIFIER_ATTR, q)  # noqa: B010
        return cls

    return decorator  # pyright: ignore[reportReturnType]


def extract_type_qualifier(tp: object) -> Qualifier:
    """Extract the effective qualifier from a type form.

    Checks (in order):
    1. ``Annotated[X, qual(...), ...]`` metadata — intersects all ``Qualifier``
       instances found in the annotation metadata.
    2. ``@qualifier(...)`` decorator on a class — reads ``__context_qualifier__``.
    3. Returns ``UNQUALIFIED`` if no qualifier is found.

    For ``Annotated`` types the result combines metadata qualifiers with any
    class-level qualifier on the base type (via recursion).
    """
    origin = get_origin(tp)
    if origin is Annotated:
        args = get_args(tp)
        base_qual = extract_type_qualifier(args[0])
        result = base_qual
        for item in args[1:]:
            if isinstance(item, Qualifier):
                result = result & item
        return result

    if isinstance(tp, type):
        class_qual = getattr(tp, _CONTEXT_QUALIFIER_ATTR, None)  # noqa: B009
        if isinstance(class_qual, Qualifier):
            return class_qual

    return UNQUALIFIED
