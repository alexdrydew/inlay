"""Error types for type normalization."""


class NormalizationError(Exception):
    """Raised when a type cannot be normalized."""


class MissingTypeAnnotationError(Exception):
    """Raised when a callable parameter lacks type annotation."""


class UnresolvedTypeAnnotationError(Exception):
    """Raised when a string annotation cannot be resolved at runtime."""


class UnsupportedVariadicParameterError(Exception):
    """Raised when constructor/provider introspection sees *args or **kwargs."""
