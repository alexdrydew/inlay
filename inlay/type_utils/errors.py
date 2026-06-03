"""Error types for type normalization."""

from typing import Self


class NormalizationError(Exception):
    """Raised when a type cannot be normalized."""


class MissingTypeAnnotationError(Exception):
    """Raised when a callable parameter lacks type annotation."""


class UnresolvedTypeAnnotationError(Exception):
    """Raised when a string annotation cannot be resolved at runtime."""

    @classmethod
    def from_name_error(cls, exc: NameError) -> Self:
        name = exc.name or '<unknown>'
        return cls(f'Could not resolve type annotation {name!r}')


class UnsupportedVariadicParameterError(Exception):
    """Raised when constructor/provider introspection sees *args or **kwargs."""
