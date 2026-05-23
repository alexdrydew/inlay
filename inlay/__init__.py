from inlay._native import (
    CallableType,
    ClassType,
    Compiler,
    LazyRefType,
    ParamSpecType,
    PlainType,
    ProtocolBase,
    ProtocolMethod,
    ProtocolType,
    Qualifier,
    ResolutionError,
    RuleGraph,
    SentinelType,
    TypedDictType,
    TypeVarType,
    UnionType,
)
from inlay.compile import compile, compiled
from inlay.registry import (
    ConstructorEntry,
    MethodEntry,
    Registry,
)
from inlay.type_utils import (
    UNQUALIFIED,
    CallableInfo,
    LazyRef,
    MissingTypeAnnotationError,
    NormalizationError,
    ParamInfo,
    ParamKind,
    UnresolvedTypeAnnotationError,
    UnsupportedVariadicParameterError,
    extract_type_qualifier,
    get_callable_info,
    normalize,
    normalize_callable,
    normalize_with_qualifier,
    qual,
    qualifier,
)

type NormalizedType = (
    SentinelType
    | TypeVarType
    | ParamSpecType
    | PlainType
    | ProtocolType
    | TypedDictType
    | UnionType
    | CallableType
    | ClassType
    | LazyRefType
)

__all__ = [
    'CallableInfo',
    'CallableType',
    'ClassType',
    'Compiler',
    'ConstructorEntry',
    'LazyRef',
    'LazyRefType',
    'MethodEntry',
    'MissingTypeAnnotationError',
    'NormalizationError',
    'NormalizedType',
    'ParamInfo',
    'ParamKind',
    'ParamSpecType',
    'PlainType',
    'ProtocolMethod',
    'ProtocolBase',
    'ProtocolType',
    'Qualifier',
    'Registry',
    'ResolutionError',
    'RuleGraph',
    'SentinelType',
    'TypeVarType',
    'TypedDictType',
    'UnresolvedTypeAnnotationError',
    'UnsupportedVariadicParameterError',
    'UNQUALIFIED',
    'UnionType',
    'compile',
    'compiled',
    'get_callable_info',
    'normalize',
    'normalize_callable',
    'normalize_with_qualifier',
    'qual',
    'qualifier',
    'extract_type_qualifier',
]
