"""make_partial public API tests."""

from typing import Protocol

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
