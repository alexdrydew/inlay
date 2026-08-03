from inlay._native import (
    CallableSignatureType,
    CallableType,
    CellType,
    ClassType,
    Compiler,
    ParamSpecType,
    PlainType,
    ProtocolBase,
    ProtocolMethod,
    ProtocolType,
    Qualifier,
    ReadCellType,
    ResolutionError,
    RuleGraph,
    SentinelType,
    TypedDictType,
    TypeVarType,
    UnionType,
)
from inlay.compile import compile, compiled, make_partial
from inlay.registry import (
    ConstructorEntry,
    MethodEntry,
    Registry,
)
from inlay.type_utils import (
    UNQUALIFIED,
    CallableInfo,
    Cell,
    MissingTypeAnnotationError,
    NormalizationError,
    ParamInfo,
    ParamKind,
    ReadCell,
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
    | CallableSignatureType
    | CallableType
    | ClassType
    | ReadCellType
    | CellType
)

__all__ = [
    'CallableInfo',
    'CallableSignatureType',
    'CallableType',
    'Cell',
    'CellType',
    'ClassType',
    'Compiler',
    'ConstructorEntry',
    'ReadCell',
    'ReadCellType',
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
    'make_partial',
    'get_callable_info',
    'normalize',
    'normalize_callable',
    'normalize_with_qualifier',
    'qual',
    'qualifier',
    'extract_type_qualifier',
]
