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
from inlay.compile import compile
from inlay.registry import (
    ConstructorEntry,
    DuplicateRegistrationError,
    HookEntry,
    MethodEntry,
    RegistryBuilder,
)
from inlay.type_utils import (
    UNQUALIFIED,
    CallableInfo,
    LazyRef,
    MissingTypeAnnotationError,
    NormalizationError,
    ParamInfo,
    ParamKind,
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
    'ConstructorEntry',
    'DuplicateRegistrationError',
    'HookEntry',
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
    'UNQUALIFIED',
    'UnionType',
    'compile',
    'get_callable_info',
    'normalize',
    'normalize_callable',
    'normalize_with_qualifier',
    'qual',
    'qualifier',
    'extract_type_qualifier',
]
