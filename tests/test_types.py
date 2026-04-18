"""Tests for NormalizedType dataclasses."""

from typing import ParamSpec, TypeVar

from inlay import (
    CallableType,
    LazyRefType,
    ParamSpecType,
    PlainType,
    ProtocolType,
    SentinelType,
    TypedDictType,
    TypeVarType,
    UnionType,
    qual,
)


class TestPlainType:
    def test_simple_type(self) -> None:
        t = PlainType(origin=str, args=(), qualifiers=qual())

        assert t.origin is str
        assert t.args == ()
        assert t.qualifiers == qual()

    def test_generic_type(self) -> None:
        inner = PlainType(origin=str, args=(), qualifiers=qual())
        t = PlainType(origin=list, args=(inner,), qualifiers=qual())

        assert t.origin is list
        assert t.args == (inner,)

    def test_qualified_type(self) -> None:
        t = PlainType(origin=str, args=(), qualifiers=qual('read', 'primary'))

        assert t.qualifiers == qual('read', 'primary')

    def test_equality(self) -> None:
        t1 = PlainType(origin=str, args=(), qualifiers=qual())
        t2 = PlainType(origin=str, args=(), qualifiers=qual())
        t3 = PlainType(origin=int, args=(), qualifiers=qual())

        assert t1 == t2
        assert t1 != t3

    def test_different_qualifiers_not_equal(self) -> None:
        t1 = PlainType(origin=str, args=(), qualifiers=qual())
        t2 = PlainType(origin=str, args=(), qualifiers=qual('x'))

        assert t1 != t2


class TestSentinelType:
    def test_none(self) -> None:
        t = SentinelType(value=None, qualifiers=qual())

        assert t.value is None
        assert t.qualifiers == qual()

    def test_ellipsis(self) -> None:
        t = SentinelType(value=..., qualifiers=qual())

        assert t.value is ...
        assert t.qualifiers == qual()


class TestUnionType:
    def test_simple_union(self) -> None:
        str_type = PlainType(origin=str, args=(), qualifiers=qual())
        none_type = SentinelType(value=None, qualifiers=qual())
        t = UnionType(variants=(str_type, none_type), qualifiers=qual())

        assert str_type in t.variants
        assert none_type in t.variants
        assert len(t.variants) == 2

    def test_qualified_union(self) -> None:
        str_type = PlainType(origin=str, args=(), qualifiers=qual())
        int_type = PlainType(origin=int, args=(), qualifiers=qual())
        t = UnionType(variants=(str_type, int_type), qualifiers=qual('x'))

        assert t.qualifiers == qual('x')

    def test_order_preserved(self) -> None:
        str_type = PlainType(origin=str, args=(), qualifiers=qual())
        int_type = PlainType(origin=int, args=(), qualifiers=qual())

        t1 = UnionType(variants=(str_type, int_type), qualifiers=qual())
        t2 = UnionType(variants=(int_type, str_type), qualifiers=qual())

        assert t1 != t2
        assert t1.variants[0] == str_type
        assert t2.variants[0] == int_type


class TestCallableType:
    def test_no_params(self) -> None:
        return_type = PlainType(origin=str, args=(), qualifiers=qual())
        t = CallableType(
            params=(),
            param_names=(),
            param_kinds=(),
            return_type=return_type,
            return_wrapper='none',
            type_params=(),
            qualifiers=qual(),
        )

        assert t.params == ()
        assert t.param_names == ()
        assert t.param_kinds == ()
        assert t.return_type == return_type
        assert t.return_wrapper == 'none'

    def test_with_params(self) -> None:
        int_type = PlainType(origin=int, args=(), qualifiers=qual())
        str_type = PlainType(origin=str, args=(), qualifiers=qual())
        bool_type = PlainType(origin=bool, args=(), qualifiers=qual())
        t = CallableType(
            params=(int_type, str_type),
            param_names=('a', 'b'),
            param_kinds=('positional_or_keyword', 'keyword_only'),
            return_type=bool_type,
            return_wrapper='none',
            type_params=(),
            qualifiers=qual(),
        )

        assert t.params == (int_type, str_type)
        assert t.param_names == ('a', 'b')
        assert t.param_kinds == ('positional_or_keyword', 'keyword_only')
        assert t.return_type == bool_type


class TestTypeVarType:
    def test_simple_typevar(self) -> None:
        T = TypeVar('T')
        t = TypeVarType(typevar=T, qualifiers=qual())

        assert t.typevar is T
        assert t.qualifiers == qual()

    def test_qualified_typevar(self) -> None:
        T = TypeVar('T')
        t = TypeVarType(typevar=T, qualifiers=qual('x'))

        assert t.qualifiers == qual('x')

    def test_equality_by_typevar_identity(self) -> None:
        T = TypeVar('T')
        K = TypeVar('K')

        t1 = TypeVarType(typevar=T, qualifiers=qual())
        t2 = TypeVarType(typevar=T, qualifiers=qual())
        t3 = TypeVarType(typevar=K, qualifiers=qual())

        assert t1 == t2
        assert t1 != t3


class TestParamSpecType:
    def test_simple_paramspec(self) -> None:
        P = ParamSpec('P')
        t = ParamSpecType(paramspec=P, qualifiers=qual())

        assert t.paramspec is P
        assert t.qualifiers == qual()


class TestProtocolType:
    def test_construction(self) -> None:
        str_type = PlainType(origin=str, args=(), qualifiers=qual())
        int_type = PlainType(origin=int, args=(), qualifiers=qual())
        callable_type = CallableType(
            params=(int_type,),
            param_names=('x',),
            param_kinds=('positional_or_keyword',),
            return_type=str_type,
            return_wrapper='none',
            type_params=(),
            qualifiers=qual(),
        )

        t = ProtocolType(
            origin=object,
            type_params=(),
            methods={'do_thing': callable_type},
            attributes={'name': str_type},
            properties={'value': int_type},
            qualifiers=qual(),
        )

        assert t.methods == {'do_thing': callable_type}
        assert t.attributes == {'name': str_type}
        assert t.properties == {'value': int_type}
        assert t.qualifiers == qual()

    def test_qualified(self) -> None:
        t = ProtocolType(
            origin=object,
            type_params=(),
            methods={},
            attributes={},
            properties={},
            qualifiers=qual('read'),
        )

        assert t.qualifiers == qual('read')


class TestTypedDictType:
    def test_construction(self) -> None:
        str_type = PlainType(origin=str, args=(), qualifiers=qual())
        int_type = PlainType(origin=int, args=(), qualifiers=qual())

        t = TypedDictType(
            origin=object,
            type_params=(),
            attributes={'name': str_type, 'value': int_type},
            qualifiers=qual(),
        )

        assert t.attributes == {'name': str_type, 'value': int_type}
        assert t.qualifiers == qual()

    def test_equality(self) -> None:
        str_type = PlainType(origin=str, args=(), qualifiers=qual())

        t1 = TypedDictType(
            origin=object,
            type_params=(),
            attributes={'name': str_type},
            qualifiers=qual(),
        )
        t2 = TypedDictType(
            origin=object,
            type_params=(),
            attributes={'name': str_type},
            qualifiers=qual(),
        )

        assert t1 == t2


class TestLazyRefType:
    def test_construction(self) -> None:
        target = PlainType(origin=str, args=(), qualifiers=qual())
        t = LazyRefType(target=target, qualifiers=qual())

        assert t.target == target
        assert t.qualifiers == qual()


class TestNormalizedTypeVariants:
    def test_different_variants_not_equal(self) -> None:
        plain = PlainType(origin=str, args=(), qualifiers=qual())
        union = UnionType(variants=(plain,), qualifiers=qual())

        assert plain != union
