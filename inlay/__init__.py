from inlay._native import (
    CallableType,
    LazyRefType,
    ParamSpecType,
    PlainType,
    ProtocolType,
    Qualifier,
    Registry,
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
    RegistryBuilder,
)
from inlay.type_utils import (
    UNQUALIFIED,
    CallableInfo,
    ContextInject,
    LazyRef,
    MissingTypeAnnotationError,
    NormalizationError,
    ParamInfo,
    ParamKind,
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
    | LazyRefType
)

__all__ = [
    'CallableInfo',
    'CallableType',
    'ContextInject',
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
    'ProtocolType',
    'Qualifier',
    'Registry',
    'RegistryBuilder',
    'ResolutionError',
    'RuleGraph',
    'SentinelType',
    'TypeVarType',
    'TypedDictType',
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
