"""Shared normalized-type alias."""

from inlay._native import (
    CallableSignatureType,
    CallableType,
    CellType,
    ClassType,
    ParamSpecType,
    PlainType,
    ProtocolType,
    ReadCellType,
    SentinelType,
    TypedDictType,
    TypeVarType,
    UnionType,
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
