"""make_partial public API tests."""

from collections.abc import Callable
from typing import Protocol, cast

import pytest

from inlay import Registry, ResolutionError, RuleGraph, compile, make_partial


class TestMakePartial:
    def test_source_public_args_and_di_params_feed_implementation(
        self, rules: RuleGraph
    ) -> None:
        class Source:
            def __init__(self, prefix: str) -> None:
                self.prefix: str = prefix

        class Dep:
            def __init__(self) -> None:
                self.suffix: str = '!'

        class Result:
            def __init__(self, value: str) -> None:
                self.value: str = value

        def public(count: int) -> Result: ...  # pyright: ignore[reportUnusedParameter]

        registry = Registry().register(Dep)(Dep).build(rules)

        @make_partial(Source, public, registry=registry)
        def build(source: Source, count: int, dep: Dep) -> Result:
            return Result((source.prefix * count) + dep.suffix)

        inner = build(Source('a'))

        assert inner(3).value == 'aaa!'

    def test_public_signature_controls_partial_args_not_impl_shape(
        self, rules: RuleGraph
    ) -> None:
        class Source:
            def __init__(self, prefix: str) -> None:
                self.prefix: str = prefix

        class Result:
            def __init__(self, value: str) -> None:
                self.value: str = value

        def public(count: int, label: str) -> Result: ...  # pyright: ignore[reportUnusedParameter]

        @make_partial(Source, public, registry=Registry().build(rules))
        def build(source: Source, count: int) -> Result:
            return Result(source.prefix * count)

        inner = build(Source('a'))

        with pytest.raises(TypeError, match='missing.*label'):
            inner(3)  # pyright: ignore[reportCallIssue]

    def test_none_implementation_return_resolves_public_return_type(
        self, rules: RuleGraph
    ) -> None:
        class Source:
            def __init__(self, prefix: str) -> None:
                self.prefix: str = prefix

        class Result:
            def __init__(self, source: Source, count: int) -> None:
                self.value: str = source.prefix * count

        def public(count: int) -> Result: ...  # pyright: ignore[reportUnusedParameter]

        seen: list[tuple[Source, int]] = []

        @make_partial(Source, public, registry=Registry().build(rules))
        def build(source: Source, count: int) -> None:
            seen.append((source, count))

        source = Source('b')

        result = build(source)(4)

        assert seen == [(source, 4)]
        assert result.value == 'bbbb'

    def test_autoresolved_protocol_source_shares_member_cache_with_impl_args(
        self, rules: RuleGraph
    ) -> None:
        class Dep:
            pass

        class Source(Protocol):
            @property
            def dep(self) -> Dep: ...

        class Result:
            def __init__(self, source_dep: Dep, injected_dep: Dep) -> None:
                self.source_dep: Dep = source_dep
                self.injected_dep: Dep = injected_dep

        def public() -> Result: ...

        calls: list[Dep] = []

        def make_dep() -> Dep:
            dep = Dep()
            calls.append(dep)
            return dep

        compiler = Registry().register(Dep)(make_dep).build(rules)
        source = compile(Source, compiler)
        source_dep = source.dep

        @make_partial(Source, public, registry=compiler)
        def build(source: Source, dep: Dep) -> Result:
            return Result(source.dep, dep)

        result = build(source)()

        assert result.source_dep is source_dep
        assert result.injected_dep is source_dep
        assert calls == [source_dep]

    def test_callable_return_binds_returned_callable_from_nested_context(
        self, rules: RuleGraph
    ) -> None:
        class OuterDep:
            pass

        class InnerDep:
            pass

        class OuterSource(Protocol):
            @property
            def outer_dep(self) -> OuterDep: ...

        class InnerSource(Protocol):
            @property
            def inner_dep(self) -> InnerDep: ...

        class OuterSourceImpl:
            def __init__(self, outer_dep: OuterDep) -> None:
                self.outer_dep: OuterDep = outer_dep

        class InnerSourceImpl:
            def __init__(self, inner_dep: InnerDep) -> None:
                self.inner_dep: InnerDep = inner_dep

        class Result:
            def __init__(self, outer_dep: OuterDep, inner_dep: InnerDep) -> None:
                self.outer_dep: OuterDep = outer_dep
                self.inner_dep: InnerDep = inner_dep

        def public(source: InnerSource) -> Result: ...  # pyright: ignore[reportUnusedParameter]

        events: list[str] = []

        @make_partial(OuterSource, public, registry=Registry().build(rules))
        def build(outer_dep: OuterDep) -> Callable[[InnerDep], Result]:
            events.append('outer')

            def inner(inner_dep: InnerDep) -> Result:
                events.append('inner')
                return Result(outer_dep, inner_dep)

            return inner

        outer_dep = OuterDep()
        inner_dep = InnerDep()

        inner = build(OuterSourceImpl(outer_dep))

        assert events == ['outer']

        result = inner(InnerSourceImpl(inner_dep))

        assert result.outer_dep is outer_dep
        assert result.inner_dep is inner_dep
        assert events == ['outer', 'inner']

    def test_callable_return_binds_two_returned_callable_levels(
        self, rules: RuleGraph
    ) -> None:
        class OuterDep:
            pass

        class MiddleDep:
            pass

        class InnerDep:
            pass

        class OuterSource(Protocol):
            @property
            def outer_dep(self) -> OuterDep: ...

        class MiddleSource(Protocol):
            @property
            def middle_dep(self) -> MiddleDep: ...

        class InnerSource(Protocol):
            @property
            def inner_dep(self) -> InnerDep: ...

        class OuterSourceImpl:
            def __init__(self, outer_dep: OuterDep) -> None:
                self.outer_dep: OuterDep = outer_dep

        class MiddleSourceImpl:
            def __init__(self, middle_dep: MiddleDep) -> None:
                self.middle_dep: MiddleDep = middle_dep

        class InnerSourceImpl:
            def __init__(self, inner_dep: InnerDep) -> None:
                self.inner_dep: InnerDep = inner_dep

        class Result:
            def __init__(
                self,
                outer_dep: OuterDep,
                middle_dep: MiddleDep,
                inner_dep: InnerDep,
            ) -> None:
                self.outer_dep: OuterDep = outer_dep
                self.middle_dep: MiddleDep = middle_dep
                self.inner_dep: InnerDep = inner_dep

        def public(_source: MiddleSource) -> Callable[[InnerSource], Result]: ...

        events: list[str] = []

        @make_partial(OuterSource, public, registry=Registry().build(rules))
        def build(
            outer_dep: OuterDep,
        ) -> Callable[[MiddleDep], Callable[[InnerDep], Result]]:
            events.append('outer')

            def middle(middle_dep: MiddleDep) -> Callable[[InnerDep], Result]:
                events.append('middle')

                def inner(inner_dep: InnerDep) -> Result:
                    events.append('inner')
                    return Result(outer_dep, middle_dep, inner_dep)

                return inner

            return middle

        outer_dep = OuterDep()
        middle_dep = MiddleDep()
        inner_dep = InnerDep()

        middle = build(OuterSourceImpl(outer_dep))

        assert events == ['outer']

        inner = middle(MiddleSourceImpl(middle_dep))

        assert events == ['outer', 'middle']

        result = inner(InnerSourceImpl(inner_dep))

        assert result.outer_dep is outer_dep
        assert result.middle_dep is middle_dep
        assert result.inner_dep is inner_dep
        assert events == ['outer', 'middle', 'inner']

    def test_union_callable_return_binds_returned_callable_variant(
        self, rules: RuleGraph
    ) -> None:
        class OuterDep:
            pass

        class InnerDep:
            pass

        class OuterSource(Protocol):
            @property
            def outer_dep(self) -> OuterDep: ...

        class InnerSource(Protocol):
            @property
            def inner_dep(self) -> InnerDep: ...

        class OuterSourceImpl:
            def __init__(self, outer_dep: OuterDep) -> None:
                self.outer_dep: OuterDep = outer_dep

        class InnerSourceImpl:
            def __init__(self, inner_dep: InnerDep) -> None:
                self.inner_dep: InnerDep = inner_dep

        class Result:
            def __init__(self, outer_dep: OuterDep, inner_dep: InnerDep | None) -> None:
                self.outer_dep: OuterDep = outer_dep
                self.inner_dep: InnerDep | None = inner_dep

        def public(
            _source: InnerSource,
        ) -> Result | Callable[[InnerSource], Result]: ...

        events: list[str] = []

        @make_partial(OuterSource, public, registry=Registry().build(rules))
        def build(
            outer_dep: OuterDep,
        ) -> Result | Callable[[InnerDep], Result]:
            events.append('outer')

            def inner(inner_dep: InnerDep) -> Result:
                events.append('inner')
                return Result(outer_dep, inner_dep)

            return inner

        outer_dep = OuterDep()
        inner_dep = InnerDep()

        result_or_inner = build(OuterSourceImpl(outer_dep))

        assert events == ['outer']
        assert callable(result_or_inner)

        inner = cast(Callable[[InnerSource], Result], result_or_inner)
        result = inner(InnerSourceImpl(inner_dep))

        assert result.outer_dep is outer_dep
        assert result.inner_dep is inner_dep
        assert events == ['outer', 'inner']

    def test_omitted_partial_returns_zero_arg_partial_from_impl_return(
        self, rules: RuleGraph
    ) -> None:
        class Source:
            def __init__(self, prefix: str) -> None:
                self.prefix: str = prefix

        class Dep:
            def __init__(self) -> None:
                self.suffix: str = '!'

        class Result:
            def __init__(self, value: str) -> None:
                self.value: str = value

        registry = Registry().register(Dep)(Dep).build(rules)

        @make_partial(Source, registry=registry)
        def build(source: Source, dep: Dep) -> Result:
            return Result(source.prefix + dep.suffix)

        inner = build(Source('a'))

        assert inner().value == 'a!'
        with pytest.raises(TypeError, match='takes 0 positional'):
            inner(1)  # pyright: ignore[reportCallIssue]

    def test_omitted_partial_source_scope_resolves_impl_params(
        self, rules: RuleGraph
    ) -> None:
        class Dep:
            pass

        class Source(Protocol):
            @property
            def dep(self) -> Dep: ...

        class Result:
            def __init__(self, dep: Dep) -> None:
                self.dep: Dep = dep

        calls: list[Dep] = []

        def make_dep() -> Dep:
            dep = Dep()
            calls.append(dep)
            return dep

        compiler = Registry().register(Dep)(make_dep).build(rules)
        source = compile(Source, compiler)
        source_dep = source.dep

        @make_partial(Source, registry=compiler)
        def build(dep: Dep) -> Result:
            return Result(dep)

        result = build(source)()

        assert result.dep is source_dep
        assert calls == [source_dep]

    def test_omitted_partial_unresolvable_impl_param_fails_at_decoration_time(
        self, rules: RuleGraph
    ) -> None:
        class Source:
            pass

        class Result:
            pass

        with pytest.raises(ResolutionError):

            @make_partial(Source, registry=Registry().build(rules))
            def build(missing: int) -> Result:  # pyright: ignore[reportUnusedFunction, reportUnusedParameter]
                return Result()

    def test_omitted_partial_requires_implementation_return_annotation(
        self, rules: RuleGraph
    ) -> None:
        class Source:
            pass

        with pytest.raises(
            TypeError,
            match='implementation must have a return annotation',
        ):

            @make_partial(Source, registry=Registry().build(rules))
            def build():  # pyright: ignore[reportUnusedFunction]
                pass

    def test_omitted_partial_preserves_async_implementation_wrapper(
        self, rules: RuleGraph
    ) -> None:
        import anyio

        class Source:
            pass

        class Result:
            pass

        @make_partial(Source, registry=Registry().build(rules))
        async def build() -> Result:
            return Result()

        async def run() -> None:
            assert isinstance(await build(Source())(), Result)

        anyio.run(run)

    def test_unresolvable_public_return_fails_at_decoration_time(
        self, rules: RuleGraph
    ) -> None:
        class Source:
            pass

        class Result:
            def __init__(self, missing: int) -> None:
                self.missing: int = missing

        def public() -> Result: ...

        with pytest.raises(ResolutionError):

            @make_partial(Source, public, registry=Registry().build(rules))
            def build(source: Source) -> None:  # pyright: ignore[reportUnusedFunction, reportUnusedParameter]
                pass

    def test_partial_requires_return_annotation(self, rules: RuleGraph) -> None:
        class Source:
            pass

        def public():
            pass

        with pytest.raises(TypeError, match='partial must have a return annotation'):
            _ = make_partial(Source, public, registry=Registry().build(rules))

    def test_implementation_requires_return_annotation(self, rules: RuleGraph) -> None:
        class Source:
            pass

        class Result:
            pass

        def public() -> Result: ...

        with pytest.raises(
            TypeError,
            match='implementation must have a return annotation',
        ):

            @make_partial(Source, public, registry=Registry().build(rules))
            def build(source: Source):  # pyright: ignore[reportUnusedFunction, reportUnusedParameter]
                pass

    def test_sync_partial_rejects_async_implementation(self, rules: RuleGraph) -> None:
        class Source:
            pass

        class Result:
            pass

        def public() -> Result: ...

        with pytest.raises(
            TypeError,
            match='incompatible partial implementation wrapper',
        ):

            @make_partial(Source, public, registry=Registry().build(rules))
            async def build(source: Source) -> Result:  # pyright: ignore[reportUnusedFunction, reportUnusedParameter]
                return Result()
