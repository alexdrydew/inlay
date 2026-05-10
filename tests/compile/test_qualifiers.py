"""Qualifier propagation and scoping tests."""

import abc
import typing

import pytest

from inlay import RegistryBuilder, RuleGraph, compile, normalize


class TestExplicitAnyQualifier:
    def test_explicit_any_normalizes_without_ambient_qualifier(self) -> None:
        from typing import Annotated

        from inlay import ClassType, qual

        class Value:
            pass

        result = normalize(Annotated[Value, qual.ANY])

        assert isinstance(result, ClassType)
        assert result.qualifiers == qual.ANY

    def test_explicit_any_normalizes_to_ambient_qualifier(self) -> None:
        from typing import Annotated

        from inlay import ClassType, normalize_with_qualifier, qual

        class Value:
            pass

        result = normalize_with_qualifier(Annotated[Value, qual.ANY], qual('ns'))

        assert isinstance(result, ClassType)
        assert result.qualifiers == qual('ns')


class TestQualifierAnyMatching:
    def test_provider_any_matches_qualified_request(self, rules: RuleGraph) -> None:
        from typing import Annotated

        from inlay import qual

        class Dep:
            pass

        class AnyDep(Dep):
            pass

        registry = RegistryBuilder().register(Dep, qualifiers=qual.ANY)(AnyDep)

        result = registry.build().compile(rules, normalize(Annotated[Dep, qual('x')]))

        assert isinstance(result, AnyDep)

    def test_auto_method_param_any_matches_qualified_member(
        self, rules: RuleGraph
    ) -> None:
        from typing import Annotated

        from inlay import qual

        class NeedsX(typing.Protocol):
            @property
            def value(self) -> Annotated[int, qual('x')]: ...

        def root_any(
            value: Annotated[int, qual.ANY],
        ) -> NeedsX:
            raise AssertionError(value)

        factory = compile(root_any, RegistryBuilder().build(), rules)

        assert factory(1).value == 1

    def test_include_any_preserves_unqualified_provider_as_any(
        self, rules: RuleGraph
    ) -> None:
        from typing import Annotated

        from inlay import qual

        class Dep: ...

        class AnyDep(Dep): ...

        inner = RegistryBuilder().register(Dep)(AnyDep)
        registry = RegistryBuilder().include(inner, qualifiers=qual.ANY)

        result = registry.build().compile(rules, normalize(Annotated[Dep, qual('x')]))

        assert isinstance(result, AnyDep)

    def test_method_provides_any_matches_qualified_return(
        self, rules: RuleGraph
    ) -> None:
        from typing import Annotated

        from inlay import qual

        class Child(typing.Protocol):
            @property
            def value(self) -> Annotated[int, qual('x')]: ...

        class Root(typing.Protocol):
            def enter(self, value: int) -> Annotated[Child, qual('x')]: ...

        def enter(value: int) -> None: ...  # pyright: ignore[reportUnusedParameter]

        root = compile(
            Root,
            RegistryBuilder()
            .register_method(Root, Root.enter, provides=qual.ANY)(enter)
            .build(),
            rules,
        )

        assert root.enter(1).value == 1

    def test_requester_any_only_matches_provider_any(self, rules: RuleGraph) -> None:
        from typing import Annotated

        from inlay import qual

        class Dep(abc.ABC):
            @abc.abstractmethod
            def marker(self) -> None: ...

        class QualifiedDep(Dep):
            @typing.override
            def marker(self) -> None: ...

        registry = RegistryBuilder().register(Dep, qualifiers=qual('x'))(QualifiedDep)

        with pytest.raises(Exception) as exc_info:
            _ = registry.build().compile(rules, normalize(Annotated[Dep, qual.ANY]))

        assert type(exc_info.value).__name__ == 'ResolutionError'
        assert 'rules returned no match' in str(exc_info.value).lower()


class TestQualifierCompatibleAmbiguity:
    def test_broader_and_exact_constructors_are_ambiguous(
        self, rules: RuleGraph
    ) -> None:
        from inlay import qual

        class Job: ...

        class SharedJob(Job): ...

        class SpecificJob(Job): ...

        registry = (
            RegistryBuilder()
            .register(Job, qualifiers=qual('a') | qual('b'))(SharedJob)
            .register(Job, qualifiers=qual('a'))(SpecificJob)
        )

        with pytest.raises(Exception) as exc_info:
            _ = registry.build().compile(
                rules, normalize(typing.Annotated[Job, qual('a')])
            )

        assert type(exc_info.value).__name__ == 'ResolutionError'
        assert 'ambiguous constructor' in str(exc_info.value).lower()

    def test_broader_and_exact_constants_are_ambiguous(self, rules: RuleGraph) -> None:
        from inlay import qual

        class Job: ...

        def factory(
            _shared: typing.Annotated[Job, qual('a') | qual('b')],
            _specific: typing.Annotated[Job, qual('a')],
        ) -> typing.Annotated[Job, qual('a')]: ...

        with pytest.raises(Exception) as exc_info:
            _ = compile(factory, RegistryBuilder().build(), rules)

        assert type(exc_info.value).__name__ == 'ResolutionError'
        assert 'ambiguous constant' in str(exc_info.value).lower()

    def test_property_requested_name_preferred_over_other_compatible_member(
        self, rules: RuleGraph
    ) -> None:
        from inlay import qual

        class Value:
            def __init__(self, label: str) -> None:
                self.label: str = label

        class NamedSource(typing.Protocol):
            @property
            def value(self) -> typing.Annotated[Value, qual('a')]: ...

        class OtherSource(typing.Protocol):
            @property
            def other(self) -> typing.Annotated[Value, qual('a') | qual('b')]: ...

        class NamedImpl:
            @property
            def value(self) -> Value:
                return Value('named')

        class OtherImpl:
            @property
            def other(self) -> Value:
                return Value('other')

        class Root(typing.Protocol):
            @property
            def value(self) -> typing.Annotated[Value, qual('a')]: ...

        registry = (
            RegistryBuilder()
            .register(NamedSource)(NamedImpl)
            .register(OtherSource)(OtherImpl)
        )

        result = compile(Root, registry.build(), rules)

        assert result.value.label == 'named'

    def test_property_falls_back_when_requested_name_is_unresolvable(
        self, rules: RuleGraph
    ) -> None:
        from inlay import qual

        class Missing:
            def __init__(self, *_missing: object) -> None:
                pass

        class Value:
            def __init__(self, label: str) -> None:
                self.label: str = label

        class NamedSource(typing.Protocol):
            @property
            def value(self) -> typing.Annotated[Value, qual('a')]: ...

        class OtherSource(typing.Protocol):
            @property
            def other(self) -> typing.Annotated[Value, qual('a')]: ...

        def make_unresolvable(_missing: Missing) -> NamedSource: ...

        class OtherImpl:
            @property
            def other(self) -> Value:
                return Value('other')

        class Root(typing.Protocol):
            @property
            def value(self) -> typing.Annotated[Value, qual('a')]: ...

        registry = (
            RegistryBuilder()
            .register(NamedSource)(make_unresolvable)
            .register(OtherSource)(OtherImpl)
        )

        result = compile(Root, registry.build(), rules)

        assert result.value.label == 'other'

    def test_two_resolvable_same_name_properties_are_ambiguous(
        self, rules: RuleGraph
    ) -> None:
        from inlay import qual

        class Value: ...

        class SharedSource(typing.Protocol):
            @property
            def value(self) -> typing.Annotated[Value, qual('a') | qual('b')]: ...

        class SpecificSource(typing.Protocol):
            @property
            def value(self) -> typing.Annotated[Value, qual('a')]: ...

        class SharedImpl:
            @property
            def value(self) -> Value:
                return Value()

        class SpecificImpl:
            @property
            def value(self) -> Value:
                return Value()

        class Root(typing.Protocol):
            @property
            def value(self) -> typing.Annotated[Value, qual('a')]: ...

        registry = (
            RegistryBuilder()
            .register(SharedSource)(SharedImpl)
            .register(SpecificSource)(SpecificImpl)
        )

        with pytest.raises(Exception) as exc_info:
            _ = compile(Root, registry.build(), rules)

        assert type(exc_info.value).__name__ == 'ResolutionError'
        assert 'ambiguous property' in str(exc_info.value).lower()

    def test_attribute_requested_name_preferred_over_other_compatible_member(
        self, rules: RuleGraph
    ) -> None:
        from inlay import qual

        class Value:
            def __init__(self, label: str) -> None:
                self.label: str = label

        class NamedSource(typing.Protocol):
            value: typing.Annotated[Value, qual('a')]

        class OtherSource(typing.Protocol):
            other: typing.Annotated[Value, qual('a') | qual('b')]

        class NamedImpl:
            def __init__(self) -> None:
                self.value: Value = Value('named')

        class OtherImpl:
            def __init__(self) -> None:
                self.other: Value = Value('other')

        class Root(typing.Protocol):
            value: typing.Annotated[Value, qual('a')]

        registry = (
            RegistryBuilder()
            .register(NamedSource)(NamedImpl)
            .register(OtherSource)(OtherImpl)
        )

        result = compile(Root, registry.build(), rules)

        assert result.value.label == 'named'

    def test_attribute_falls_back_when_requested_name_is_unresolvable(
        self, rules: RuleGraph
    ) -> None:
        from inlay import qual

        class Missing:
            def __init__(self, *_missing: object) -> None:
                pass

        class Value:
            def __init__(self, label: str) -> None:
                self.label: str = label

        class NamedSource(typing.Protocol):
            value: typing.Annotated[Value, qual('a')]

        class OtherSource(typing.Protocol):
            other: typing.Annotated[Value, qual('a')]

        def make_unresolvable(_missing: Missing) -> NamedSource: ...

        class OtherImpl:
            def __init__(self) -> None:
                self.other: Value = Value('other')

        class Root(typing.Protocol):
            value: typing.Annotated[Value, qual('a')]

        registry = (
            RegistryBuilder()
            .register(NamedSource)(make_unresolvable)
            .register(OtherSource)(OtherImpl)
        )

        result = compile(Root, registry.build(), rules)

        assert result.value.label == 'other'

    def test_two_resolvable_same_name_attributes_are_ambiguous(
        self, rules: RuleGraph
    ) -> None:
        from inlay import qual

        class Value: ...

        class SharedSource(typing.Protocol):
            value: typing.Annotated[Value, qual('a') | qual('b')]

        class SpecificSource(typing.Protocol):
            value: typing.Annotated[Value, qual('a')]

        class SharedImpl:
            def __init__(self) -> None:
                self.value: Value = Value()

        class SpecificImpl:
            def __init__(self) -> None:
                self.value: Value = Value()

        class Root(typing.Protocol):
            value: typing.Annotated[Value, qual('a')]

        registry = (
            RegistryBuilder()
            .register(SharedSource)(SharedImpl)
            .register(SpecificSource)(SpecificImpl)
        )

        with pytest.raises(Exception) as exc_info:
            _ = compile(Root, registry.build(), rules)

        assert type(exc_info.value).__name__ == 'ResolutionError'
        assert 'ambiguous attribute' in str(exc_info.value).lower()

    def test_derived_same_source_same_name_attributes_are_not_ambiguous(
        self, rules: RuleGraph
    ) -> None:
        class Value: ...

        class Nested(typing.Protocol):
            value: Value

        class Source(typing.Protocol):
            nested: Nested
            value: Value

        class NestedImpl:
            def __init__(self) -> None:
                self.value: Value = Value()

        class SourceImpl:
            def __init__(self) -> None:
                self.nested: Nested = NestedImpl()
                self.value: Value = Value()

        class Root(typing.Protocol):
            value: Value

        registry = RegistryBuilder().register(Source)(SourceImpl)

        result = compile(Root, registry.build(), rules)

        assert isinstance(result.value, Value)


class TestRegisterFactoryQualifierAmbiguity:
    """Reproduce: register_factory with Annotated return type creates a
    constructor whose callable qualifier does NOT include the annotation
    qualifier. This causes ambiguity when two factories return the same
    base type with different annotation qualifiers.

    Root cause: cross_unify in lookup_constructors ignores qualifiers
    (only checks type structure), and resolve_constructor filters by
    the CALLABLE qualifier (not the return type qualifier).
    """

    def test_two_factories_same_type_different_return_qualifier(
        self, rules: RuleGraph
    ) -> None:
        """Two register_factory calls returning the same base type but
        with different Annotated qualifiers cause ambiguous constructor
        when a qualified request is made.
        """
        from typing import Annotated

        from inlay import qual

        class Executor[T]:
            def __init__(self, value: T) -> None:
                self.value: T = value

        class Ctx(typing.Protocol):
            @property
            def executor(self) -> Annotated[Executor[int], qual('branched')]: ...

        def provide_plain() -> Executor[int]:
            return Executor(1)

        def provide_branched() -> Annotated[Executor[int], qual('branched')]:
            return Executor(2)

        registry = (
            RegistryBuilder()
            .register_factory(provide_plain)
            .register_factory(provide_branched)
        )

        # Should resolve: Ctx.executor requests Executor[int]<branched>,
        # should match only provide_branched (not provide_plain)
        ctx = compile(Ctx, registry.build(), rules)
        assert ctx.executor.value == 2


class TestParametricFactoryQualifierBinding:
    def test_typevar_binding_propagates_across_qualifier_split(
        self, rules: RuleGraph
    ) -> None:
        """A parametric factory whose return type has a different qualifier
        than its parameters must still propagate TypeVar bindings.

        Pattern: factory(dep: Src[T]) -> Annotated[Tgt[T], qual('x')]
        The T in return gets qual('x'), T in param stays unqualified.
        Both must resolve via the same binding.
        """
        from typing import Annotated

        from inlay import qual

        class ConcreteVal:
            pass

        class Tgt:
            def __init__(self, val: ConcreteVal) -> None:
                self.val: ConcreteVal = val

        class Src:
            def __init__(self) -> None:
                pass

        def factory(_src: Src) -> Annotated[Tgt, qual('x')]:
            return Tgt(ConcreteVal())

        class Root(typing.Protocol):
            @property
            def tgt(self) -> Annotated[Tgt, qual('x')]: ...

        registry = (
            RegistryBuilder()
            .register(ConcreteVal)(ConcreteVal)
            .register(Tgt)(Tgt)
            .register(Src)(Src)
            .register_factory(factory)
        )

        result = compile(Root, registry.build(), rules)

        assert isinstance(result.tgt, Tgt)
        assert isinstance(result.tgt.val, ConcreteVal)

    def test_typevar_binding_propagates_with_include_qualifier(
        self, rules: RuleGraph
    ) -> None:
        """Real-world pattern: provide_transaction_executor[TxCtxT].

        factory(src: LazyRef[Proto[T]]) -> Annotated[Executor[T], qual('x')]
        After include(qual('mod')): return gets {x, mod}, param gets {mod}.
        T must still bind across the qualifier split.
        """
        from typing import Annotated

        from inlay import LazyRef, qual

        T = typing.TypeVar('T')

        class ConcreteCtx:
            pass

        class Executor[T_]:
            def __init__(self, val: object) -> None:
                self.val: object = val

        class Source[T_](typing.Protocol):
            def get(self) -> T_: ...

        class SourceImpl:
            def get(self) -> ConcreteCtx:
                return ConcreteCtx()

        def make_executor(
            src: LazyRef[Source[T]],
        ) -> Annotated[Executor[T], qual('x')]:
            return Executor(src)

        class Root(typing.Protocol):
            @property
            def executor(
                self,
            ) -> Annotated[Executor[ConcreteCtx], qual('x') & qual('mod')]: ...

        inner_registry = (
            RegistryBuilder()
            .register(Source[ConcreteCtx])(SourceImpl)
            .register_factory(make_executor)
        )

        registry = (
            RegistryBuilder()
            .register(ConcreteCtx)(ConcreteCtx)
            .include(inner_registry, qualifiers=qual('mod'))
        )

        result = compile(Root, registry.build(), rules)

        assert isinstance(result.executor, Executor)


class TestQualifiedTransitions:
    def test_child_param_shadows_parent_only_for_unnamed_lookup(
        self, rules: RuleGraph
    ) -> None:
        from typing import Annotated

        from inlay import qual

        class Value:
            def __init__(self, label: str) -> None:
                self.label: str = label

        class Child(typing.Protocol):
            a: Annotated[Value, qual('a')]
            value: Annotated[Value, qual('a')]

        class Root(typing.Protocol):
            def enter(
                self,
                b: Annotated[Value, qual('a') | qual('b')],
            ) -> Child: ...

        def make_root(a: Annotated[Value, qual('a')]) -> Root: ...  # pyright: ignore[reportUnusedParameter]

        root_factory = compile(make_root, RegistryBuilder().build(), rules)
        parent_value = Value('parent')
        child_value = Value('child')

        child = root_factory(parent_value).enter(child_value)

        assert child.a is parent_value
        assert child.value is child_value

    def test_transition_with_qualified_return_type(self, rules: RuleGraph) -> None:
        """Transition returning Annotated[ChildCtx, qual('x')] should resolve
        child members from the qualified scope.
        """
        from typing import Annotated

        from inlay import qual

        class Config:
            pass

        class ChildCtx(typing.Protocol):
            @property
            def config(self) -> Config: ...

        class ParentCtx(typing.Protocol):
            def enter(self) -> Annotated[ChildCtx, qual('scoped')]: ...

        registry = RegistryBuilder().register(Config, qualifiers=qual('scoped'))(Config)

        parent = compile(ParentCtx, registry.build(), rules)
        child = parent.enter()

        assert isinstance(child.config, Config)

    def test_qualified_transition_in_composed_protocol(self, rules: RuleGraph) -> None:
        """Composed protocol with multiple qualified transitions.

        Reproduces the real-world pattern: ApplicationContext has
        HasStoryContext, HasChatContext etc. each returning
        Annotated[ModuleContext, qual('module')].
        """
        from typing import Annotated

        from inlay import qual

        class ServiceA:
            pass

        class ServiceB:
            pass

        class CtxA(typing.Protocol):
            @property
            def svc(self) -> ServiceA: ...

        class CtxB(typing.Protocol):
            @property
            def svc(self) -> ServiceB: ...

        class HasCtxA(typing.Protocol):
            def with_a(self) -> Annotated[CtxA, qual('a')]: ...

        class HasCtxB(typing.Protocol):
            def with_b(self) -> Annotated[CtxB, qual('b')]: ...

        class RootCtx(HasCtxA, HasCtxB, typing.Protocol): ...

        registry = (
            RegistryBuilder()
            .register(ServiceA, qualifiers=qual('a'))(ServiceA)
            .register(ServiceB, qualifiers=qual('b'))(ServiceB)
        )

        root = compile(RootCtx, registry.build(), rules)

        child_a = root.with_a()
        assert isinstance(child_a.svc, ServiceA)

        child_b = root.with_b()
        assert isinstance(child_b.svc, ServiceB)


class TestQualifierPropagationBoundary:
    """Qualifier propagation rule:
    - ``qualifiers=`` is namespace shorthand and propagates to dependency
      resolution.
    - ``provides=`` qualifies only registered outputs.
    - ``requires=`` qualifies only source/input/dependency resolution.

    This applies uniformly to:
    - Constructor parameters
    - Property/attribute origins (qualifier on the member type must not
      flow up to the source protocol lookup)
    """

    def test_constructor_provides_qualifier_does_not_propagate_to_params(
        self, rules: RuleGraph
    ) -> None:
        """Constructor registered with provides=qual('x'): its parameter
        Dep is resolved unqualified, not with qual('x').
        """
        from typing import Annotated

        from inlay import qual

        class Dep:
            pass

        class Service:
            def __init__(self, dep: Dep) -> None:
                self.dep: Dep = dep

        native = (
            RegistryBuilder()
            .register(Dep)(Dep)
            .register(Service, provides=qual('x'))(Service)
        ).build()

        # If provides=qual('x') leaked into params, it would look for
        # Annotated[Dep, qual('x')] which is not registered -> error.
        result = native.compile(rules, normalize(Annotated[Service, qual('x')]))

        assert isinstance(result, Service)
        assert isinstance(result.dep, Dep)

    def test_constructor_namespace_qualifier_propagates_to_params(
        self, rules: RuleGraph
    ) -> None:
        """With qualifiers=qual('x'), constructor params resolve in the same
        qualified namespace.
        """
        from typing import Annotated

        from inlay import qual

        class Dep:
            pass

        class DepDefault(Dep):
            pass

        class DepX(Dep):
            pass

        class Service:
            def __init__(self, dep: Dep) -> None:
                self.dep: Dep = dep

        native = (
            RegistryBuilder()
            .register(Dep)(DepDefault)
            .register(Dep, qualifiers=qual('x'))(DepX)
            .register(Service, qualifiers=qual('x'))(Service)
        ).build()

        result = native.compile(rules, normalize(Annotated[Service, qual('x')]))

        assert isinstance(result, Service)
        assert isinstance(result.dep, DepX)

    def test_constructor_requires_qualifier_propagates_to_params_only(
        self, rules: RuleGraph
    ) -> None:
        """requires=qual('x') qualifies params without qualifying the target."""

        from inlay import qual

        class Dep:
            pass

        class DepX(Dep):
            pass

        class Service:
            def __init__(self, dep: Dep) -> None:
                self.dep: Dep = dep

        native = (
            RegistryBuilder()
            .register(Dep, provides=qual('x'))(DepX)
            .register(Service, requires=qual('x'))(Service)
        ).build()

        result = native.compile(rules, normalize(Service))

        assert isinstance(result, Service)
        assert isinstance(result.dep, DepX)

    def test_constructor_include_namespace_qualifier_propagates_to_params(
        self, rules: RuleGraph
    ) -> None:
        """include(qualifiers=qual('ns')) should propagate the namespace
        qualifier into constructor parameter resolution.
        """
        from typing import Annotated

        from inlay import qual

        class Dep:
            pass

        class Service:
            def __init__(self, dep: Dep) -> None:
                self.dep: Dep = dep

        inner = RegistryBuilder().register(Dep)(Dep).register(Service)(Service)
        native = RegistryBuilder().include(inner, qualifiers=qual('ns')).build()

        # Both Service and Dep get qual('ns') from include.
        # Namespace qualifier propagates to params, so param Dep is
        # resolved as Annotated[Dep, qual('ns')] -> matches.
        result = native.compile(rules, normalize(Annotated[Service, qual('ns')]))

        assert isinstance(result, Service)
        assert isinstance(result.dep, Dep)

    def test_constructor_combined_provides_and_namespace_requires(
        self, rules: RuleGraph
    ) -> None:
        """provides=qual('x') + include qualifier qual('ns'):
        only qual('ns') propagates to constructor params, while the target
        gets qual('x', 'ns').
        """
        from typing import Annotated

        from inlay import qual

        class Dep:
            pass

        class Service:
            def __init__(self, dep: Dep) -> None:
                self.dep: Dep = dep

        inner = (
            RegistryBuilder()
            .register(Dep)(Dep)
            .register(Service, provides=qual('x'))(Service)
        )
        native = RegistryBuilder().include(inner, qualifiers=qual('ns')).build()

        # Service has merged qualifier qual('x','ns').
        # Dep has qualifier qual('ns') (only namespace).
        # Param Dep should be resolved with qual('ns') (namespace only),
        # not qual('x','ns') (merged).
        result = native.compile(
            rules, normalize(Annotated[Service, qual('x') & qual('ns')])
        )

        assert isinstance(result, Service)
        assert isinstance(result.dep, Dep)

    def test_constructor_include_provides_only_does_not_propagate_to_params(
        self, rules: RuleGraph
    ) -> None:
        """include(provides=qual('out')) qualifies outputs only."""
        from typing import Annotated

        from inlay import qual

        class Dep:
            pass

        class Service:
            def __init__(self, dep: Dep) -> None:
                self.dep: Dep = dep

        inner = RegistryBuilder().register(Service)(Service)
        native = (
            RegistryBuilder()
            .register(Dep)(Dep)
            .include(inner, provides=qual('out'))
            .build()
        )

        result = native.compile(rules, normalize(Annotated[Service, qual('out')]))

        assert isinstance(result, Service)
        assert isinstance(result.dep, Dep)

    def test_constructor_include_requires_only_does_not_qualify_output(
        self, rules: RuleGraph
    ) -> None:
        """include(requires=qual('in')) qualifies params only."""

        from inlay import qual

        class Dep:
            pass

        class DepIn(Dep):
            pass

        class Service:
            def __init__(self, dep: Dep) -> None:
                self.dep: Dep = dep

        inner = RegistryBuilder().register(Service)(Service)
        native = (
            RegistryBuilder()
            .register(Dep, provides=qual('in'))(DepIn)
            .include(inner, requires=qual('in'))
            .build()
        )

        result = native.compile(rules, normalize(Service))

        assert isinstance(result, Service)
        assert isinstance(result.dep, DepIn)

    def test_property_source_resolves_via_constructor(self, rules: RuleGraph) -> None:
        """Baseline: a type resolved via property_source_rule when the
        source protocol has a registered constructor.
        """

        class Value:
            pass

        class Source(typing.Protocol):
            @property
            def value(self) -> Value: ...

        class SourceImpl:
            @property
            def value(self) -> Value:
                return Value()

        class Root(typing.Protocol):
            @property
            def value(self) -> Value: ...

        native = RegistryBuilder().register(Source)(SourceImpl).build()

        result = typing.cast(Root, native.compile(rules, normalize(Root)))

        assert isinstance(result.value, Value)

    def test_property_member_qualifier_does_not_propagate_to_source(
        self, rules: RuleGraph
    ) -> None:
        """Qualifier on the member type (Annotated[Value, qual('m')])
        must not propagate up to the source protocol resolution.
        """
        from typing import Annotated

        from inlay import qual

        class Value:
            pass

        class Source(typing.Protocol):
            @property
            def value(self) -> Annotated[Value, qual('m')]: ...

        class SourceImpl:
            @property
            def value(self) -> Value:
                return Value()

        class Root(typing.Protocol):
            @property
            def value(self) -> Annotated[Value, qual('m')]: ...

        native = RegistryBuilder().register(Source)(SourceImpl).build()

        result = typing.cast(Root, native.compile(rules, normalize(Root)))

        assert isinstance(result.value, Value)

    def test_attribute_member_qualifier_does_not_propagate_to_source(
        self, rules: RuleGraph
    ) -> None:
        """Same rule for attribute_source_rule: the qualifier on the
        attribute type must not propagate to the source resolution.
        """
        from typing import Annotated

        from inlay import qual

        class Value:
            pass

        class Source(typing.Protocol):
            value: Annotated[Value, qual('m')]

        class SourceImpl:
            def __init__(self) -> None:
                self.value: Value = Value()

        class Root(typing.Protocol):
            value: Annotated[Value, qual('m')]

        native = RegistryBuilder().register(Source)(SourceImpl).build()

        result = typing.cast(Root, native.compile(rules, normalize(Root)))

        assert isinstance(result.value, Value)

    def test_property_with_include_namespace_on_source(self, rules: RuleGraph) -> None:
        """Include namespace qualifier propagates correctly through
        property_source_rule resolution.
        """
        from typing import Annotated

        from inlay import qual

        class Value:
            pass

        class Source(typing.Protocol):
            @property
            def value(self) -> Annotated[Value, qual('m')]: ...

        class SourceImpl:
            @property
            def value(self) -> Value:
                return Value()

        class Root(typing.Protocol):
            @property
            def value(self) -> Annotated[Value, qual('m') & qual('ns')]: ...

        inner = RegistryBuilder().register(Source)(SourceImpl)
        native = RegistryBuilder().include(inner, qualifiers=qual('ns')).build()

        result = typing.cast(Root, native.compile(rules, normalize(Root)))

        assert isinstance(result.value, Value)

    def test_build_constructor_params_use_requires_qualifier(self) -> None:
        """_build_constructor normalizes params with requires, not provides."""

        from inlay import qual
        from inlay._native import CallableType, ClassType
        from inlay.registry import ConstructorEntry, build_constructor_entry

        class Dep:
            pass

        class Service:
            def __init__(self, dep: Dep) -> None: ...

        inner = (
            RegistryBuilder()
            .register(Dep)(Dep)
            .register(Service, provides=qual('x'))(Service)
        )
        registry = RegistryBuilder().include(inner, qualifiers=qual('ns'))

        # After include: Service provides = qual('x','ns'),
        #                Service requires = qual('ns').
        _ = registry.build()

        # Inspect the native registry's callable for Service.
        # The callable's return_type carries the full qualifier
        # (qual('x','ns')), but each param carries only the
        # namespace qualifier (qual('ns')).
        #
        # We verify indirectly: if the param had qual('x','ns'),
        # resolution with only qual('ns') Dep would fail.
        # The end-to-end constructor test above already proves this,
        # but here we verify the build output directly.
        entry = ConstructorEntry(
            constructor=Service,
            target_type=Service,
            provides=qual('x') & qual('ns'),
            requires=qual('ns'),
        )
        built_entry = build_constructor_entry(entry)
        ct = built_entry.callable_type
        assert isinstance(ct, CallableType)

        # Return type has merged qualifier
        assert ct.qualifiers == qual('x') & qual('ns')

        # Parameter 'dep' has only the namespace qualifier
        assert len(ct.params) == 1
        dep_param = ct.params[0]
        assert isinstance(dep_param, ClassType)
        assert dep_param.qualifiers == qual('ns')

    def test_protocol_member_annotation_qualifier_does_not_propagate_up(
        self,
    ) -> None:
        """When a protocol property has Annotated[T, qual('m')], the
        member qualifier stays on the member type; the protocol's own
        qualifier is unaffected.
        """
        from typing import Annotated

        from inlay import qual
        from inlay._native import ClassType, ProtocolType

        class Value:
            pass

        class Source(typing.Protocol):
            @property
            def value(self) -> Annotated[Value, qual('m')]: ...

        result = normalize(Source)

        assert isinstance(result, ProtocolType)
        # Protocol itself is unqualified
        assert result.qualifiers == qual()
        # Member type carries the annotation qualifier
        member = result.properties['value']
        assert isinstance(member, ClassType)
        assert member.origin is Value
        assert member.qualifiers == qual('m')

    def test_protocol_member_annotation_qualifier_intersects_with_context(
        self,
    ) -> None:
        """When a protocol is qualified (e.g. from include namespace),
        the member's annotation qualifier intersects with the context
        qualifier, but the context qualifier on the protocol itself
        does NOT include the annotation qualifier.
        """
        from typing import Annotated

        from inlay import qual
        from inlay._native import ClassType, ProtocolType

        class Value:
            pass

        class Source(typing.Protocol):
            @property
            def value(self) -> Annotated[Value, qual('m')]: ...

        result = normalize(Annotated[Source, qual('ns')])

        assert isinstance(result, ProtocolType)
        # Protocol has only the context qualifier
        assert result.qualifiers == qual('ns')
        # Member type has context intersection annotation
        member = result.properties['value']
        assert isinstance(member, ClassType)
        assert member.origin is Value
        assert member.qualifiers == qual('m') & qual('ns')
