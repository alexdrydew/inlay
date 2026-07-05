"""Shared normalized-type alias."""

from inlay._native import (
    CallableSignatureType,
    CallableType,
    ClassType,
    LazyRefType,
    ParamSpecType,
    PlainType,
    ProtocolType,
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
    | LazyRefType
)
