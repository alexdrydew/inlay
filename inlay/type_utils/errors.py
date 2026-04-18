"""Error types for type normalization."""


class NormalizationError(Exception):
    """Raised when a type cannot be normalized."""


class MissingTypeAnnotationError(Exception):
    """Raised when a callable parameter lacks type annotation."""
