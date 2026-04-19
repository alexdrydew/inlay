"""Type normalization and introspection."""

from inlay.type_utils.errors import (
    MissingTypeAnnotationError,
    NormalizationError,
    UnsupportedVariadicParameterError,
)
from inlay.type_utils.markers import (
    UNQUALIFIED,
    LazyRef,
    extract_type_qualifier,
    qual,
    qualifier,
)
from inlay.type_utils.normalize import (
    CallableInfo,
    ParamInfo,
    ParamKind,
    WrapperKind,
    get_callable_info,
    normalize,
    normalize_callable,
    normalize_with_qualifier,
    unwrap_return_type,
)

__all__ = [
    'UNQUALIFIED',
    'CallableInfo',
    'LazyRef',
    'MissingTypeAnnotationError',
    'NormalizationError',
    'ParamInfo',
    'ParamKind',
    'UnsupportedVariadicParameterError',
    'WrapperKind',
    'extract_type_qualifier',
    'get_callable_info',
    'normalize',
    'normalize_callable',
    'normalize_with_qualifier',
    'qual',
    'qualifier',
    'unwrap_return_type',
]
