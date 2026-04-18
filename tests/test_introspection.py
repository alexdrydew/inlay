"""Tests for get_callable_info()."""

from inlay import (
    PlainType,
    get_callable_info,
    qual,
)


class TestGetCallableInfo:
    def test_simple_function(self) -> None:
        def my_func(_x: int, _y: str) -> bool:
            return True

        info = get_callable_info(my_func)

        assert len(info.params) == 2
        assert info.params[0].name == '_x'
        assert info.params[0].type == PlainType(origin=int, args=(), qualifiers=qual())
        assert info.params[0].has_default is False
        assert info.params[0].kind == 'positional_or_keyword'
        assert info.params[1].name == '_y'
        assert info.params[1].type == PlainType(origin=str, args=(), qualifiers=qual())
        assert info.params[1].kind == 'positional_or_keyword'
        assert info.return_type == PlainType(origin=bool, args=(), qualifiers=qual())

    def test_function_with_defaults(self) -> None:
        def my_func(_x: int, _y: str = 'default') -> bool:
            return True

        info = get_callable_info(my_func)

        assert info.params[0].has_default is False
        assert info.params[1].has_default is True

    def test_no_params(self) -> None:
        def my_func() -> str:
            return 'hello'

        info = get_callable_info(my_func)

        assert info.params == []
        assert info.return_type == PlainType(origin=str, args=(), qualifiers=qual())

    def test_method_excludes_self(self) -> None:
        class MyClass:
            def my_method(self, x: int) -> str:
                return str(x)

        info = get_callable_info(MyClass.my_method)

        assert len(info.params) == 1
        assert info.params[0].name == 'x'

    def test_generic_return_type(self) -> None:
        def my_func(_x: int) -> list[str]:
            return []

        info = get_callable_info(my_func)

        assert info.return_type == PlainType(
            origin=list,
            args=(PlainType(origin=str, args=(), qualifiers=qual()),),
            qualifiers=qual(),
        )
