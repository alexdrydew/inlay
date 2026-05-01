"""Protocol property resolution tests."""

import typing

from inlay import RegistryBuilder, RuleGraph, compile, normalize


class TestPropertySourceTypeVarFalsePositive:
    """Reproduce: parametric Protocol with a bare type variable property
    (e.g. HasTransaction[T] with transaction: T) matches ANY type request
    via cross_unify, producing false-positive property sources that can't
    actually be constructed.

    resolve_property should try resolving each candidate source and discard
    the ones that fail, rather than reporting ambiguity immediately.
    """

    def test_unresolvable_property_source_should_not_block_resolvable_one(
        self, rules: RuleGraph
    ) -> None:
        """Two protocols provide the same property type. One is available
        as a constant (seed param), the other requires construction that
        fails. The resolver should try both and use the resolvable one,
        not report ambiguity.
        """

        class Config:
            """Only obtainable from Storages.config property."""

            def __init__(self, x: int) -> None:
                self.x: int = x

        class _Unresolvable:
            pass

        class ResolvableSource(typing.Protocol):
            """Seed param - always available as constant."""

            @property
            def config(self) -> Config: ...

        class UnresolvableSource(typing.Protocol):
            """Also has config property, but can't be constructed."""

            @property
            def config(self) -> Config: ...

            @property
            def blocker(self) -> _Unresolvable: ...

        class Target(typing.Protocol):
            @property
            def config(self) -> Config: ...

        def factory(_good: ResolvableSource) -> Target: ...

        # Register a constructor for UnresolvableSource - it requires
        # _Unresolvable which has no constructor, so it will fail.
        def make_bad(_dep: _Unresolvable) -> UnresolvableSource: ...  # type: ignore[empty-body]

        registry = RegistryBuilder().register(UnresolvableSource)(make_bad)

        class StoragesImpl:
            @property
            def config(self) -> Config:
                return Config(x=42)

        compiled_factory = compile(factory, registry.build(), rules)
        result = compiled_factory(StoragesImpl())
        assert result.config.x == 42


class TestPropertyBasedProtocol:
    def test_property_resolved_via_constructor(self, rules: RuleGraph) -> None:
        """Protocol property type is resolved through the full pipeline."""

        class Config:
            pass

        class HasConfig(typing.Protocol):
            @property
            def config(self) -> Config: ...

        registry = RegistryBuilder().register(Config)(Config)
        native = registry.build()

        result = typing.cast(HasConfig, native.compile(rules, normalize(HasConfig)))

        assert isinstance(result.config, Config)

    def test_property_with_constructor_dependency(self, rules: RuleGraph) -> None:
        """Protocol property whose type itself has constructor dependencies."""

        class Db:
            pass

        class Repo:
            def __init__(self, db: Db) -> None:
                self.db: Db = db

        class HasRepo(typing.Protocol):
            @property
            def repo(self) -> Repo: ...

        registry = RegistryBuilder().register(Db)(Db).register(Repo)(Repo)
        native = registry.build()

        result = typing.cast(HasRepo, native.compile(rules, normalize(HasRepo)))

        assert isinstance(result.repo, Repo)
        assert isinstance(result.repo.db, Db)

    def test_property_and_method_together(self, rules: RuleGraph) -> None:
        """Protocol with both a property and a transition method."""

        class Config:
            pass

        class Child:
            pass

        class MyCtx(typing.Protocol):
            @property
            def config(self) -> Config: ...

            def create(self) -> Child: ...

        def create_impl() -> Child:
            return Child()

        registry = (
            RegistryBuilder()
            .register(Config)(Config)
            .register(Child)(Child)
            .register_method(MyCtx, MyCtx.create)(create_impl)
        )
        native = registry.build()

        result = typing.cast(MyCtx, native.compile(rules, normalize(MyCtx)))

        assert isinstance(result.config, Config)
        assert isinstance(result.create(), Child)

    def test_multiple_properties(self, rules: RuleGraph) -> None:
        """Protocol with multiple properties all resolved via constructors."""

        class A:
            pass

        class B:
            pass

        class HasBoth(typing.Protocol):
            @property
            def a(self) -> A: ...

            @property
            def b(self) -> B: ...

        registry = RegistryBuilder().register(A)(A).register(B)(B)
        native = registry.build()

        result = typing.cast(HasBoth, native.compile(rules, normalize(HasBoth)))

        assert isinstance(result.a, A)
        assert isinstance(result.b, B)

    def test_transition_target_with_property(self, rules: RuleGraph) -> None:
        """Transition produces a child context whose protocol has a property."""

        class Config:
            pass

        class ChildCtx(typing.Protocol):
            @property
            def config(self) -> Config: ...

        class ParentCtx(typing.Protocol):
            def enter(self) -> ChildCtx: ...

        registry = RegistryBuilder().register(Config)(Config)
        native = registry.build()

        result = typing.cast(ParentCtx, native.compile(rules, normalize(ParentCtx)))

        child = result.enter()
        assert isinstance(child.config, Config)
