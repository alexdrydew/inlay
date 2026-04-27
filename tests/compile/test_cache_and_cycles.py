"""Cache rollback and cycle termination tests."""

import typing

import pytest

from inlay import RegistryBuilder, RuleGraph, compile, normalize


class TestLazyRefCacheKeyCycles:
    @pytest.mark.skip(
        reason='TODO: fix lazy-ref constructor backreference cache-key cycle'
    )
    def test_constructor_backreference_does_not_hang_cache_key_computation(
        self,
        rules: RuleGraph,
    ) -> None:
        from inlay import LazyRef

        class A:
            b: LazyRef[B]

            def __init__(self, b: LazyRef[B]) -> None:
                self.b = b

        class B:
            a: A

            def __init__(self, a: A) -> None:
                self.a = a

        registry = RegistryBuilder().register(A)(A).register(B)(B).build()
        _ = compile(A, registry, rules)


class TestGrowingTypeTowerTermination:
    """Regression tests for growing type towers caused by parametric
    matching with bare TypeVars.

    A bare TypeVar property like `HasTransaction[T].transaction: T` matches
    ANY type via cross_unify, causing the monomorphization strategy to create
    HasTransaction[X] -> HasTransaction[HasTransaction[X]] -> ... chains.
    Each apply_bindings creates a new concrete key, so the cycle detector
    can't catch it.

    The resolver uses two mechanisms to guarantee termination:
    - Depth limit: early-out for a single growing path
    - Call budget: bounds total work when multiple tower-creating rules
      each try alternatives after depth-exceeded failures
    """

    def test_property_tower_terminates(self, rules: RuleGraph) -> None:
        """A bare TypeVar property creates an infinite tower.
        Resolution should fail (not hang) when the type has no other source.

        Without the depth/budget limits this would hang: the resolver
        creates HasValue[_Marker] -> HasValue[HasValue[_Marker]] -> ...
        endlessly via monomorphization.
        """

        class _Marker:
            pass

        class HasValue[T](typing.Protocol):
            @property
            def value(self) -> T: ...

        class Target(typing.Protocol):
            @property
            def marker(self) -> _Marker: ...

        # HasValue[T] registered - its property `value: T` is a bare
        # TypeVar that matches everything in the parametric index.
        def make_has_value(x: int) -> HasValue[int]: ...  # type: ignore[empty-body]

        registry = RegistryBuilder().register(HasValue[int])(make_has_value)

        # No seed parameter providing _Marker - forces resolution through
        # the parametric property chain which creates the tower.
        def factory() -> Target: ...

        with pytest.raises(Exception) as exc_info:
            compile(factory, registry.build(), rules)

        assert type(exc_info.value).__name__ == 'ResolutionError'

    def test_constructor_tower_terminates(self, rules: RuleGraph) -> None:
        """A constructor with a bare TypeVar return type matches everything.
        Resolution should fail (not hang) when the type has no other source.

        The constructor `unwrap(x: list[T]) -> T` creates a tower:
        need _Marker -> unwrap needs list[_Marker] -> unwrap needs
        list[list[_Marker]] -> ... each step wrapping in another list.
        """

        class _Marker:
            pass

        class Target(typing.Protocol):
            @property
            def marker(self) -> _Marker: ...

        def unwrap(x: list[object]) -> object: ...  # type: ignore[empty-body]

        registry = RegistryBuilder().register(object)(unwrap)

        def factory() -> Target: ...

        with pytest.raises(Exception) as exc_info:
            compile(factory, registry.build(), rules)

        assert type(exc_info.value).__name__ == 'ResolutionError'

    def test_constructor_found_after_property_tower(self, rules: RuleGraph) -> None:
        """Property tower fails (depth limit), but a constructor for the
        same type succeeds. MaxDepthExceeded is non-fatal, so match_first
        falls through to constructor_rule.
        """

        class Config:
            def __init__(self) -> None:
                self.x = 42

        class HasValue[T](typing.Protocol):
            @property
            def value(self) -> T: ...

        class Target(typing.Protocol):
            @property
            def config(self) -> Config: ...

        # HasValue[T] registered - bare TypeVar property creates tower for Config
        def make_has_value(x: int) -> HasValue[int]: ...  # type: ignore[empty-body]

        # But Config also has a direct constructor - should be found
        # after property_source fails with MaxDepthExceeded.
        def make_config() -> Config:
            return Config()

        registry = (
            RegistryBuilder()
            .register(HasValue[int])(make_has_value)
            .register(Config)(make_config)
        )

        # No seed parameter - Config must come from constructor,
        # not from a concrete property of some constant.
        def factory() -> Target: ...

        compiled_factory = compile(factory, registry.build(), rules)
        result = compiled_factory()
        assert result.config.x == 42


class TestLookupConstructorsConstantPoisoning:
    """lookup_constructors registers the constructor's concrete return type
    as a constant in the RegistryCache before the constructor's params are
    resolved. If the constructor later fails (unresolvable param), the
    phantom constant remains and poisons subsequent resolution.

    Scenario:
    - TargetImpl.__init__(Unresolvable) -> Target  (constructor will fail)
    - Target is a Protocol with property value: Value
    - Value is constructable (zero-arg)

    Expected: Target falls through to protocol_rule after the constructor
    fails, resolves value: Value via constructor_rule.

    Bug: lookup_constructors eagerly registers Target as a constant with
    source FnResult(TargetImpl). register_constant then indexes Value as
    a property source backed by that phantom constant. When protocol_rule
    resolves value: Value, constant_rule wins (phantom entry), emits a
    Constant node, and runtime crashes with "constant not found in scope".
    """

    def test_failed_constructor_does_not_poison_constants(
        self, rules: RuleGraph
    ) -> None:
        class Unresolvable: ...

        class Value: ...

        class Target(typing.Protocol):
            @property
            def value(self) -> Value: ...

        class TargetImpl:
            def __init__(self, x: Unresolvable) -> None: ...

            @property
            def value(self) -> Value:
                return Value()

        class Root(typing.Protocol):
            @property
            def target(self) -> Target: ...

        # given
        registry = RegistryBuilder().register(Value)(Value).register(Target)(TargetImpl)
        native = registry.build()

        # when
        result = typing.cast(Root, native.compile(rules, normalize(Root)))

        # then - Value resolved via constructor, not phantom constant
        assert isinstance(result.target.value, Value)


class TestLookupMethodsTransitionResultPoisoning:
    """A failed explicit method implementation must not poison fallback resolution.

    `resolve_method_impl` adds the method result as a transition-root constant
    before the bound instance is fully resolved. If that explicit method later
    fails and `auto_method` becomes the winning fallback, the phantom result
    source must not survive into the runtime graph.

    Otherwise the fallback child context reads from a result source that was
    never introduced at runtime and crashes with "source value not found in
    scope".
    """

    def test_failed_explicit_method_does_not_poison_fallback(
        self, rules: RuleGraph
    ) -> None:
        class Missing: ...

        class Value: ...

        class State(typing.TypedDict):
            value: Value

        class Child(typing.Protocol):
            @property
            def value(self) -> Value: ...

        class Root(typing.Protocol):
            def with_state(self) -> Child: ...

        @typing.final
        class WithStateImpl:
            def __init__(self, missing: Missing) -> None:
                self._missing = missing

            def with_state(self) -> State:
                return {'value': Value()}

        root = compile(
            Root,
            RegistryBuilder()
            .register(Value)(Value)
            .register_method(Root, method_name='with_state')(WithStateImpl)
            .build(),
            rules,
        )

        assert isinstance(root.with_state().value, Value)


class TestRollbackWithBackreference:
    def test_backreference_child_evicted_on_parent_rollback(
        self, rules: RuleGraph
    ) -> None:
        """When a constructor fails after a child resolved via LazyRef
        backreference, the child must be evicted from cache (its graph
        node was removed during rollback). The type then resolves via
        a fallback rule (protocol_rule).

        Resolution trace:
        1. Root -> property Target
        2. Target -> InProgress, tries constructor TargetImpl(ParamA, ParamB)
        3. ParamA -> constructor make_a(LazyRef[Target])
        4. LazyRef[Target] -> lazy_depth+1 -> hits InProgress -> backreference OK
        5. ParamA resolves ✓ (holds backreference to in-progress Target)
        6. ParamB -> fails (nothing provides it)
        7. TargetImpl constructor fails -> rollback evicts ParamA subtree
        8. match_first continues -> protocol_rule matches Target
        9. Target resolves as protocol (value: int via constant) ✓
        """
        from inlay import LazyRef

        class ParamA: ...

        class ParamB: ...

        class Value: ...

        class Target(typing.Protocol):
            @property
            def value(self) -> Value: ...

        class TargetImpl:
            def __init__(self, a: ParamA, b: ParamB) -> None:
                self.a = a
                self.b = b

            @property
            def value(self) -> Value:
                return Value()

        def make_a(t: LazyRef[Target]) -> ParamA:
            return ParamA()

        class Root(typing.Protocol):
            @property
            def target(self) -> Target: ...

        # given
        registry = (
            RegistryBuilder()
            .register(Value)(Value)
            .register(Target)(TargetImpl)
            .register_factory(make_a)
        )
        native = registry.build()

        # when
        result = typing.cast(Root, native.compile(rules, normalize(Root)))

        # then
        assert isinstance(result.target.value, Value)


class TestCrossTransitionCycleDetection:
    """Cyclic Protocol references through auto_method transition boundaries.

    InProgress markers don't propagate across with_transition scope
    boundaries (resolution_cache is replaced via mem::take). Child scopes
    re-resolve types that parent scopes are currently computing, causing
    redundant work proportional to max_transition_depth.
    """

    def test_three_protocol_cycle_through_transitions(self, rules: RuleGraph) -> None:
        """A.nested()->B, B.nested()->C, C.backref->A.

        Two transitions separate A from C. With cross-scope InProgress,
        C.backref resolves as a lazy backreference (lazy_depth=2 > 0).
        Without it, A is re-resolved in S2 from scratch, cycling until
        the budget is exhausted.

        Expected resolution (with cross-scope InProgress):
          S0: resolve(A, ld=0) -> InProgress(A,0) -> protocol -> nested
              -> auto_method -> transition S1
          S1: resolve(B, ld=1) -> protocol -> nested -> auto_method -> S2
          S2: resolve(C, ld=2) -> protocol -> backref: A
              -> cache finds InProgress(A,0) -> 0 < 2 -> backreference OK
          Total: 2 transitions.

        The solver-backed resolver keeps enough path context across
        transitions to terminate this cycle without the old per-scope
        cache reset behavior.
        """

        class A(typing.Protocol):
            def nested(self) -> B: ...

        class B(typing.Protocol):
            def nested(self) -> C: ...

        class C(typing.Protocol):
            @property
            def backref(self) -> A: ...

        class Root(typing.Protocol):
            @property
            def a(self) -> A: ...

        # given
        registry = RegistryBuilder()

        # when
        result = compile(Root, registry.build(), rules)

        # then
        assert result.a is not None
