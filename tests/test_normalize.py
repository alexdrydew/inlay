"""Tests for normalize() function."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Annotated, NewType, ParamSpec, TypeVar

import pytest

from inlay import (
    CallableType,
    ClassType,
    LazyRef,
    LazyRefType,
    NormalizationError,
    ParamSpecType,
    PlainType,
    ProtocolMethod,
    ProtocolType,
    Qualifier,
    SentinelType,
    TypedDictType,
    TypeVarType,
    UnionType,
    normalize,
    qual,
)

type UserList = list[str]


class _UserForAlias:
    pass


type UserMapping = dict[str, _UserForAlias]

_T = TypeVar('_T')


def _plain(origin: type, qualifiers: Qualifier | None = None) -> PlainType:
    return PlainType(
        origin=origin,
        args=(),
        qualifiers=qual() if qualifiers is None else qualifiers,
    )


def _class(origin: type, qualifiers: Qualifier | None = None) -> ClassType:
    return ClassType(
        origin=origin,
        args=(),
        init_params=(),
        init_param_names=(),
        init_param_kinds=(),
        init_param_has_default=(),
        qualifiers=qual() if qualifiers is None else qualifiers,
    )


class TestNormalizeSimpleTypes:
    def test_normalize_str(self) -> None:
        result = normalize(str)

        assert isinstance(result, PlainType)
        assert result.origin is str
        assert result.args == ()
        assert result.qualifiers == qual()

    def test_normalize_int(self) -> None:
        assert normalize(int) == _plain(int)

    def test_normalize_bool(self) -> None:
        assert normalize(bool) == _plain(bool)

    def test_normalize_none(self) -> None:
        assert normalize(None) == SentinelType(value=None, qualifiers=qual())

    def test_normalize_none_type(self) -> None:
        assert normalize(type(None)) == SentinelType(value=None, qualifiers=qual())

    def test_normalize_ellipsis(self) -> None:
        assert normalize(...) == SentinelType(value=..., qualifiers=qual())

    def test_normalize_custom_class(self) -> None:
        class MyClass:
            pass

        result = normalize(MyClass)

        assert isinstance(result, ClassType)
        assert result.origin is MyClass
        assert result.args == ()
        assert result.init_params == ()
        assert result.qualifiers == qual()


class TestNormalizeGenericTypes:
    def test_normalize_list_str(self) -> None:
        assert normalize(list[str]) == PlainType(
            origin=list,
            args=(_plain(str),),
            qualifiers=qual(),
        )

    def test_normalize_dict_str_int(self) -> None:
        assert normalize(dict[str, int]) == PlainType(
            origin=dict,
            args=(
                _plain(str),
                _plain(int),
            ),
            qualifiers=qual(),
        )

    def test_normalize_nested_generic(self) -> None:
        inner = PlainType(
            origin=list,
            args=(_plain(str),),
            qualifiers=qual(),
        )
        assert normalize(list[list[str]]) == PlainType(
            origin=list, args=(inner,), qualifiers=qual()
        )

    def test_normalize_custom_generic(self) -> None:
        class Repository[T]:
            pass

        class User:
            pass

        result = normalize(Repository[User])

        assert isinstance(result, ClassType)
        assert result.origin is Repository
        assert result.args == (_class(User),)
        assert result.qualifiers == qual()


class TestNormalizeNewType:
    def test_normalize_newtype_preserves_identity(self) -> None:
        UserId = NewType('UserId', str)

        result = normalize(UserId)

        assert isinstance(result, PlainType)
        assert result.origin is UserId
        assert result.args == ()
        assert result.qualifiers == qual()

    def test_different_newtypes_not_equal(self) -> None:
        UserId = NewType('UserId', str)
        OrderId = NewType('OrderId', str)

        assert normalize(UserId) != normalize(OrderId)


class TestNormalizeUnion:
    def test_normalize_union_pipe_syntax(self) -> None:
        result = normalize(str | int)

        assert isinstance(result, UnionType)
        assert result.variants == (
            _plain(str),
            _plain(int),
        )
        assert result.qualifiers == qual()

    def test_normalize_optional(self) -> None:
        result = normalize(str | None)

        assert isinstance(result, UnionType)
        assert result.variants == (
            _plain(str),
            SentinelType(value=None, qualifiers=qual()),
        )

    def test_normalize_union_flattens_nested(self) -> None:
        result = normalize(str | int | bool)

        assert isinstance(result, UnionType)
        assert len(result.variants) == 3


class TestNormalizeCallable:
    def test_normalize_callable_no_args(self) -> None:
        result = normalize(Callable[[], str])

        assert result == CallableType(
            params=(),
            param_names=(),
            param_kinds=(),
            return_type=_plain(str),
            return_wrapper='none',
            type_params=(),
            qualifiers=qual(),
        )

    def test_normalize_callable_with_args(self) -> None:
        result = normalize(Callable[[int, str], bool])

        assert result == CallableType(
            params=(
                _plain(int),
                _plain(str),
            ),
            param_names=('_0', '_1'),
            param_kinds=('positional_or_keyword', 'positional_or_keyword'),
            return_type=_plain(bool),
            return_wrapper='none',
            type_params=(),
            qualifiers=qual(),
        )

    def test_normalize_callable_generic_return(self) -> None:
        result = normalize(Callable[[int], list[str]])

        expected_return = PlainType(
            origin=list,
            args=(_plain(str),),
            qualifiers=qual(),
        )
        assert result == CallableType(
            params=(_plain(int),),
            param_names=('_0',),
            param_kinds=('positional_or_keyword',),
            return_type=expected_return,
            return_wrapper='none',
            type_params=(),
            qualifiers=qual(),
        )


class TestNormalizeTypeVar:
    def test_normalize_typevar(self) -> None:
        T = TypeVar('T')
        assert normalize(T) == TypeVarType(typevar=T, qualifiers=qual())

    def test_normalize_typevar_in_generic(self) -> None:
        result = normalize(list[_T])  # pyright: ignore[reportGeneralTypeIssues]

        assert isinstance(result, PlainType)
        assert result.origin is list
        assert len(result.args) == 1
        assert isinstance(result.args[0], TypeVarType)
        assert result.args[0].typevar is _T


class TestNormalizeParamSpec:
    def test_normalize_paramspec(self) -> None:
        P = ParamSpec('P')
        assert normalize(P) == ParamSpecType(paramspec=P, qualifiers=qual())


class TestNormalizeAnnotated:
    def test_normalize_annotated_with_qualified(self) -> None:
        assert normalize(Annotated[str, qual('read')]) == PlainType(
            origin=str, args=(), qualifiers=qual('read')
        )

    def test_normalize_annotated_multiple_qualifiers(self) -> None:
        assert normalize(Annotated[str, qual('read', 'primary')]) == PlainType(
            origin=str, args=(), qualifiers=qual('read', 'primary')
        )

    def test_normalize_annotated_generic_with_qualified(self) -> None:
        assert normalize(Annotated[list[str], qual('cached')]) == PlainType(
            origin=list,
            args=(_plain(str, qual('cached')),),
            qualifiers=qual('cached'),
        )

    def test_normalize_annotated_without_qualified_preserves_type(self) -> None:
        class SomeMarker:
            pass

        assert normalize(Annotated[str, SomeMarker()]) == PlainType(
            origin=str, args=(), qualifiers=qual()
        )

    def test_normalize_annotated_union_with_qualified(self) -> None:
        result = normalize(Annotated[str | int, qual('x')])

        assert isinstance(result, UnionType)
        assert result.qualifiers == qual('x')


type _RecursiveList = int | list[_RecursiveList]


class TestNormalizeTypeAlias:
    def test_normalize_type_alias_expands(self) -> None:
        assert normalize(UserList) == PlainType(
            origin=list,
            args=(_plain(str),),
            qualifiers=qual(),
        )

    def test_normalize_type_alias_with_generic(self) -> None:
        assert normalize(UserMapping) == PlainType(
            origin=dict,
            args=(
                _plain(str),
                _class(_UserForAlias),
            ),
            qualifiers=qual(),
        )

    def test_normalize_recursive_type_alias_succeeds(self) -> None:
        result = normalize(_RecursiveList)
        assert isinstance(result, UnionType)


class TestNormalizeProtocol:
    def test_normalize_protocol_produces_protocol_type(self) -> None:
        from typing import Protocol

        class HasValue(Protocol):
            @property
            def value(self) -> str: ...

        result = normalize(HasValue)

        assert isinstance(result, ProtocolType)
        assert result.origin is HasValue
        assert 'value' in result.properties
        assert result.properties['value'] == PlainType(
            origin=str, args=(), qualifiers=qual()
        )

    def test_normalize_protocol_with_method(self) -> None:
        from typing import Protocol

        class HasMethod(Protocol):
            def do_thing(self, x: int) -> str: ...

        result = normalize(HasMethod)

        assert isinstance(result, ProtocolType)
        assert 'do_thing' in result.methods
        assert isinstance(result.methods['do_thing'], ProtocolMethod)
        assert isinstance(result.methods['do_thing'].callable, CallableType)
        assert result.methods['do_thing'].registration_protocol is result

    def test_inherited_protocol_method_preserves_registration_protocol(self) -> None:
        from typing import Protocol

        class Base(Protocol):
            def do_thing(self) -> str: ...

        class Child(Base, Protocol): ...

        base = normalize(Base)
        child = normalize(Child)

        assert isinstance(base, ProtocolType)
        assert isinstance(child, ProtocolType)
        registration_protocol = child.methods['do_thing'].registration_protocol
        assert isinstance(registration_protocol, ProtocolType)
        assert registration_protocol.origin is Base

    def test_ambiguous_inherited_protocol_method_raises(self) -> None:
        from typing import Protocol

        class First(Protocol):
            def do_thing(self) -> str: ...

        class Second(Protocol):
            def do_thing(self) -> str: ...

        class Child(First, Second, Protocol): ...

        with pytest.raises(NormalizationError, match='Ambiguous inherited'):
            _ = normalize(Child)

    def test_specialized_generic_base_property_substitutes_typevar(self) -> None:
        from typing import Protocol

        class Interface(Protocol):
            def use(self) -> None: ...

        class HasValue[ValueT: Interface = Interface](Protocol):
            @property
            def value(self) -> Annotated[ValueT, qual('slot')]: ...

        class SpecializedContext(HasValue[Interface], Protocol): ...

        result = normalize(SpecializedContext)

        assert isinstance(result, ProtocolType)
        value = result.properties['value']
        assert isinstance(value, ProtocolType)
        assert value.origin is Interface
        assert value.qualifiers == qual('slot')

    def test_nested_generic_base_property_substitutes_typevar(self) -> None:
        from typing import Protocol

        class Interface(Protocol):
            def use(self) -> None: ...

        class HasValue[ValueT: Interface = Interface](Protocol):
            @property
            def value(self) -> Annotated[ValueT, qual('slot')]: ...

        class Alias[AliasT: Interface](HasValue[AliasT], Protocol): ...

        class SpecializedContext(Alias[Interface], Protocol): ...

        result = normalize(SpecializedContext)

        assert isinstance(result, ProtocolType)
        value = result.properties['value']
        assert isinstance(value, ProtocolType)
        assert value.origin is Interface
        assert value.qualifiers == qual('slot')

    def test_old_style_specialized_generic_base_property_substitutes_typevar(
        self,
    ) -> None:
        from typing import Protocol

        class Interface(Protocol):
            def use(self) -> None: ...

        ValueT = TypeVar('ValueT', bound=Interface, covariant=True)

        class HasValue(Protocol[ValueT]):
            @property
            def value(self) -> Annotated[ValueT, qual('slot')]: ...

        class SpecializedContext(HasValue[Interface], Protocol): ...

        result = normalize(SpecializedContext)

        assert isinstance(result, ProtocolType)
        value = result.properties['value']
        assert isinstance(value, ProtocolType)
        assert value.origin is Interface
        assert value.qualifiers == qual('slot')


class TestNormalizeTypedDict:
    def test_normalize_typed_dict_produces_typed_dict_type(self) -> None:
        from typing import TypedDict

        class MyDict(TypedDict):
            name: str
            value: int

        result = normalize(MyDict)

        assert isinstance(result, TypedDictType)
        assert result.origin is MyDict
        assert result.attributes == {
            'name': _plain(str),
            'value': _plain(int),
        }


class TestNormalizeLazyRef:
    def test_normalize_lazy_ref(self) -> None:
        result = normalize(LazyRef[str])

        assert isinstance(result, LazyRefType)
        assert result.target == _plain(str)
        assert result.qualifiers == qual()

    def test_normalize_bare_lazy_ref_is_protocol(self) -> None:
        result = normalize(LazyRef)

        assert isinstance(result, ProtocolType)
        assert result.origin is LazyRef


class TestQualifierPropagation:
    """Qualifiers from a container must be intersected into its children."""

    def test_protocol_property_inherits_qualifiers(self) -> None:
        from typing import Protocol

        class HasClock(Protocol):
            @property
            def clock(self) -> str: ...

        result = normalize(Annotated[HasClock, qual('scoped')])

        assert isinstance(result, ProtocolType)
        assert result.qualifiers == qual('scoped')
        assert result.properties['clock'] == _plain(str, qual('scoped'))

    def test_protocol_attribute_inherits_qualifiers(self) -> None:
        from typing import Protocol

        class HasName(Protocol):
            name: str

        result = normalize(Annotated[HasName, qual('scoped')])

        assert isinstance(result, ProtocolType)
        assert result.attributes['name'] == _plain(str, qual('scoped'))

    def test_protocol_method_inherits_qualifiers(self) -> None:
        from typing import Protocol

        class HasAction(Protocol):
            def do_thing(self, x: int) -> str: ...

        result = normalize(Annotated[HasAction, qual('scoped')])

        assert isinstance(result, ProtocolType)
        protocol_method = result.methods['do_thing']
        assert isinstance(protocol_method, ProtocolMethod)
        method = protocol_method.callable
        assert method.qualifiers == qual('scoped')

    def test_protocol_method_params_inherit_qualifiers(self) -> None:
        from typing import Protocol

        class HasAction(Protocol):
            def do_thing(self, x: int, y: str) -> bool: ...

        result = normalize(Annotated[HasAction, qual('scoped')])

        assert isinstance(result, ProtocolType)
        protocol_method = result.methods['do_thing']
        assert isinstance(protocol_method, ProtocolMethod)
        method = protocol_method.callable
        assert method.params == (
            _plain(int, qual('scoped')),
            _plain(str, qual('scoped')),
        )

    def test_protocol_method_return_type_inherits_qualifiers(self) -> None:
        from typing import Protocol

        class HasAction(Protocol):
            def do_thing(self, x: int) -> str: ...

        result = normalize(Annotated[HasAction, qual('scoped')])

        assert isinstance(result, ProtocolType)
        protocol_method = result.methods['do_thing']
        assert isinstance(protocol_method, ProtocolMethod)
        method = protocol_method.callable
        assert method.return_type == _plain(str, qual('scoped'))

    def test_protocol_method_return_type_qualifiers_intersect(self) -> None:
        from typing import Protocol

        class HasAction(Protocol):
            def do_thing(self) -> Annotated[str, qual('read')]: ...

        result = normalize(Annotated[HasAction, qual('scoped')])

        assert isinstance(result, ProtocolType)
        protocol_method = result.methods['do_thing']
        assert isinstance(protocol_method, ProtocolMethod)
        method = protocol_method.callable
        assert method.return_type == _plain(str, qual('read') & qual('scoped'))

    def test_protocol_method_param_qualifiers_intersect(self) -> None:
        from typing import Protocol

        class HasAction(Protocol):
            def do_thing(self, x: Annotated[int, qual('read')]) -> str: ...

        result = normalize(Annotated[HasAction, qual('scoped')])

        assert isinstance(result, ProtocolType)
        protocol_method = result.methods['do_thing']
        assert isinstance(protocol_method, ProtocolMethod)
        method = protocol_method.callable
        assert method.params[0] == _plain(int, qual('read') & qual('scoped'))

    def test_protocol_method_no_propagation_when_unqualified(self) -> None:
        from typing import Protocol

        class HasAction(Protocol):
            def do_thing(self, x: int) -> str: ...

        result = normalize(HasAction)

        assert isinstance(result, ProtocolType)
        protocol_method = result.methods['do_thing']
        assert isinstance(protocol_method, ProtocolMethod)
        method = protocol_method.callable
        assert method.params[0] == _plain(int)
        assert method.return_type == _plain(str)

    def test_protocol_method_return_protocol_members_inherit_qualifiers(self) -> None:
        from typing import Protocol

        class Inner(Protocol):
            value: str

        class Outer(Protocol):
            def get_inner(self) -> Inner: ...

        result = normalize(Annotated[Outer, qual('scoped')])

        assert isinstance(result, ProtocolType)
        protocol_method = result.methods['get_inner']
        assert isinstance(protocol_method, ProtocolMethod)
        method = protocol_method.callable
        inner = method.return_type
        assert isinstance(inner, ProtocolType)
        assert inner.qualifiers == qual('scoped')
        assert inner.attributes['value'] == _plain(str, qual('scoped'))

    def test_protocol_member_qualifiers_intersect_with_own(self) -> None:
        from typing import Protocol

        class HasData(Protocol):
            data: Annotated[str, qual('read')]

        result = normalize(Annotated[HasData, qual('scoped')])

        assert isinstance(result, ProtocolType)
        assert result.attributes['data'] == _plain(str, qual('read') & qual('scoped'))

    def test_protocol_no_propagation_when_unqualified(self) -> None:
        from typing import Protocol

        class HasValue(Protocol):
            @property
            def value(self) -> str: ...

        result = normalize(HasValue)

        assert isinstance(result, ProtocolType)
        assert result.properties['value'] == _plain(str)

    def test_typeddict_attributes_inherit_qualifiers(self) -> None:
        from typing import TypedDict

        class MyDict(TypedDict):
            name: str
            value: int

        result = normalize(Annotated[MyDict, qual('scoped')])

        assert isinstance(result, TypedDictType)
        assert result.qualifiers == qual('scoped')
        assert result.attributes == {
            'name': _plain(str, qual('scoped')),
            'value': _plain(int, qual('scoped')),
        }

    def test_typeddict_attribute_qualifiers_intersect_with_own(self) -> None:
        from typing import TypedDict

        class MyDict(TypedDict):
            data: Annotated[str, qual('read')]

        result = normalize(Annotated[MyDict, qual('scoped')])

        assert isinstance(result, TypedDictType)
        assert result.attributes['data'] == _plain(str, qual('read') & qual('scoped'))

    @pytest.mark.parametrize(
        ('typ', 'expected_qualifiers', 'expected_variants'),
        [
            (
                Annotated[str | int, qual('x')],
                qual('x'),
                (
                    _plain(str, qual('x')),
                    _plain(int, qual('x')),
                ),
            ),
            (
                Annotated[Annotated[str, qual('a')] | int, qual('x')],
                qual('x'),
                (
                    _plain(str, qual('a') & qual('x')),
                    _plain(int, qual('x')),
                ),
            ),
            (
                str | int,
                qual(),
                (
                    _plain(str),
                    _plain(int),
                ),
            ),
        ],
    )
    def test_union_variant_qualifier_propagation(
        self,
        typ: object,
        expected_qualifiers: object,
        expected_variants: object,
    ) -> None:
        result = normalize(typ)

        assert isinstance(result, UnionType)
        assert result.qualifiers == expected_qualifiers
        assert result.variants == expected_variants

    @pytest.mark.parametrize(
        ('typ', 'expected_qualifiers', 'expected_target'),
        [
            (
                Annotated[LazyRef[str], qual('scoped')],
                qual('scoped'),
                _plain(str, qual('scoped')),
            ),
            (
                Annotated[LazyRef[Annotated[str, qual('read')]], qual('scoped')],
                qual('scoped'),
                _plain(str, qual('read') & qual('scoped')),
            ),
            (LazyRef[str], qual(), _plain(str)),
        ],
    )
    def test_lazy_ref_target_qualifier_propagation(
        self,
        typ: object,
        expected_qualifiers: object,
        expected_target: object,
    ) -> None:
        result = normalize(typ)

        assert isinstance(result, LazyRefType)
        assert result.qualifiers == expected_qualifiers
        assert result.target == expected_target


class TestDeepQualifierPropagation:
    """Qualifiers must propagate through nested compound types in a single pass."""

    def test_union_protocol_variant_members_inherit_qualifiers(self) -> None:
        from typing import Protocol

        class HasValue(Protocol):
            @property
            def value(self) -> str: ...

        result = normalize(Annotated[HasValue | int, qual('x')])

        assert isinstance(result, UnionType)
        protocol_variant = result.variants[0]
        assert isinstance(protocol_variant, ProtocolType)
        assert protocol_variant.qualifiers == qual('x')
        assert protocol_variant.properties['value'] == _plain(str, qual('x'))

    def test_union_typeddict_variant_attrs_inherit_qualifiers(self) -> None:
        from typing import TypedDict

        class MyDict(TypedDict):
            name: str

        result = normalize(Annotated[MyDict | int, qual('x')])

        assert isinstance(result, UnionType)
        td_variant = result.variants[0]
        assert isinstance(td_variant, TypedDictType)
        assert td_variant.qualifiers == qual('x')
        assert td_variant.attributes['name'] == _plain(str, qual('x'))

    def test_lazy_ref_protocol_target_members_inherit_qualifiers(self) -> None:
        from typing import Protocol

        class HasValue(Protocol):
            @property
            def value(self) -> str: ...

        result = normalize(Annotated[LazyRef[HasValue], qual('scoped')])

        assert isinstance(result, LazyRefType)
        target = result.target
        assert isinstance(target, ProtocolType)
        assert target.qualifiers == qual('scoped')
        assert target.properties['value'] == _plain(str, qual('scoped'))

    def test_nested_protocol_members_inherit_qualifiers(self) -> None:
        from typing import Protocol

        class Inner(Protocol):
            value: str

        class Outer(Protocol):
            inner: Inner

        result = normalize(Annotated[Outer, qual('admin')])

        assert isinstance(result, ProtocolType)
        inner_type = result.attributes['inner']
        assert isinstance(inner_type, ProtocolType)
        assert inner_type.qualifiers == qual('admin')
        assert inner_type.attributes['value'] == _plain(str, qual('admin'))

    def test_nested_protocol_own_qualifiers_intersect(self) -> None:
        from typing import Protocol

        class Inner(Protocol):
            value: str

        class Outer(Protocol):
            inner: Annotated[Inner, qual('read')]

        result = normalize(Annotated[Outer, qual('admin')])

        assert isinstance(result, ProtocolType)
        inner_type = result.attributes['inner']
        assert isinstance(inner_type, ProtocolType)
        assert inner_type.qualifiers == qual('read') & qual('admin')
        assert inner_type.attributes['value'] == _plain(
            str, qual('read') & qual('admin')
        )

    def test_union_protocol_variant_with_own_qualifiers(self) -> None:
        from typing import Protocol

        class HasValue(Protocol):
            value: Annotated[str, qual('read')]

        result = normalize(Annotated[HasValue | int, qual('x')])

        assert isinstance(result, UnionType)
        protocol_variant = result.variants[0]
        assert isinstance(protocol_variant, ProtocolType)
        assert protocol_variant.attributes['value'] == _plain(
            str, qual('read') & qual('x')
        )

    def test_lazy_ref_typeddict_target_attrs_inherit_qualifiers(self) -> None:
        from typing import TypedDict

        class MyDict(TypedDict):
            name: str

        result = normalize(Annotated[LazyRef[MyDict], qual('scoped')])

        assert isinstance(result, LazyRefType)
        target = result.target
        assert isinstance(target, TypedDictType)
        assert target.qualifiers == qual('scoped')
        assert target.attributes['name'] == _plain(str, qual('scoped'))

    def test_triple_nesting_protocol_propagation(self) -> None:
        from typing import Protocol

        class L0(Protocol):
            value: str

        class L1(Protocol):
            l0: L0

        class L2(Protocol):
            l1: L1

        result = normalize(Annotated[L2, qual('x')])

        assert isinstance(result, ProtocolType)
        l1_type = result.attributes['l1']
        assert isinstance(l1_type, ProtocolType)
        assert l1_type.qualifiers == qual('x')
        l0_type = l1_type.attributes['l0']
        assert isinstance(l0_type, ProtocolType)
        assert l0_type.qualifiers == qual('x')
        assert l0_type.attributes['value'] == _plain(str, qual('x'))

    def test_method_return_protocol_deep_propagation(self) -> None:
        from typing import Protocol

        class Inner(Protocol):
            value: str

        class Outer(Protocol):
            def transition(self) -> Inner: ...

        result = normalize(Annotated[Outer, qual('x')])

        assert isinstance(result, ProtocolType)
        protocol_method = result.methods['transition']
        assert isinstance(protocol_method, ProtocolMethod)
        method = protocol_method.callable
        inner = method.return_type
        assert isinstance(inner, ProtocolType)
        assert inner.qualifiers == qual('x')
        assert inner.attributes['value'] == _plain(str, qual('x'))

    def test_method_params_deep_propagation_into_protocol(self) -> None:
        from typing import Protocol

        class Dep(Protocol):
            value: str

        class Outer(Protocol):
            def action(self, dep: Dep) -> str: ...

        result = normalize(Annotated[Outer, qual('x')])

        assert isinstance(result, ProtocolType)
        protocol_method = result.methods['action']
        assert isinstance(protocol_method, ProtocolMethod)
        method = protocol_method.callable
        dep_param = method.params[0]
        assert isinstance(dep_param, ProtocolType)
        assert dep_param.qualifiers == qual('x')
        assert dep_param.attributes['value'] == _plain(str, qual('x'))


class TestNormalizeContextManager:
    def test_normalize_context_manager(self) -> None:
        assert normalize(AbstractContextManager[str]) == PlainType(
            origin=AbstractContextManager,
            args=(_plain(str),),
            qualifiers=qual(),
        )
