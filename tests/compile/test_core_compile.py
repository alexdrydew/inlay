"""Core compile tests."""

import typing

from inlay import RegistryBuilder, RuleGraph, compile, normalize


class TestCompile:
    def test_zero_arg_constructor(self, rules: RuleGraph) -> None:
        """Compile a plain type backed by a zero-arg constructor."""

        class MyService:
            pass

        registry = RegistryBuilder().register(MyService)(MyService)
        native = registry.build()

        result = native.compile(rules, normalize(MyService))

        assert isinstance(result, MyService)

    def test_constructor_with_dependency(self, rules: RuleGraph) -> None:
        """Compile a type whose constructor depends on another constructable type."""

        class Config:
            pass

        class MyService:
            def __init__(self, config: Config) -> None:
                self.config: Config = config

        registry = (
            RegistryBuilder().register(Config)(Config).register(MyService)(MyService)
        )
        native = registry.build()

        result = native.compile(rules, normalize(MyService))

        assert isinstance(result, MyService)
        assert isinstance(result.config, Config)

    def test_method_hook_fires_on_transition(self, rules: RuleGraph) -> None:
        """Hook registered for a method fires when the transition is called."""

        class MyService:
            pass

        class MyContext(typing.Protocol):
            def create(self) -> MyService: ...

        def create_impl() -> MyService:
            return MyService()

        calls: list[object] = []

        def my_hook(service: MyService) -> None:
            calls.append(service)

        registry = (
            RegistryBuilder()
            .register(MyService)(MyService)
            .register_method(MyContext, method_name='create')(create_impl)
            .register_method_hook(MyContext, method_name='create')(my_hook)
        )
        native = registry.build()

        # when
        ctx = typing.cast(MyContext, native.compile(rules, normalize(MyContext)))
        result = ctx.create()

        # then
        assert isinstance(result, MyService)
        assert len(calls) == 1
        assert isinstance(calls[0], MyService)

    def test_constructor_param_prefers_named_transition_binding(
        self, rules: RuleGraph
    ) -> None:
        """Constructor params prefer matching transition binding names."""

        class Service:
            def __init__(self, branch_id: int) -> None:
                self.branch_id: int = branch_id

        class Child(typing.Protocol):
            @property
            def service(self) -> Service: ...

        @typing.final
        class PairTransition:
            def with_pair(self, branch_id: int, _session_id: int) -> int:
                return branch_id

        class Root(typing.Protocol):
            def with_pair(self, branch_id: int, session_id: int) -> Child: ...

        # given
        registry = (
            RegistryBuilder()
            .register(Service)(Service)
            .register_method(Root, method_name='with_pair')(PairTransition)
        )

        # when
        ctx = compile(Root, registry.build(), rules)
        child = ctx.with_pair(branch_id=3, session_id=7)

        # then
        assert isinstance(child.service, Service)
        assert child.service.branch_id == 3

    def test_method_hook_param_prefers_named_transition_binding(
        self, rules: RuleGraph
    ) -> None:
        """Hook params prefer matching transition binding names."""

        class MyService:
            pass

        class MyContext(typing.Protocol):
            def create(self, branch_id: int, session_id: int) -> MyService: ...

        def create_impl(_branch_id: int, _session_id: int) -> MyService:
            return MyService()

        calls: list[int] = []

        def my_hook(session_id: int) -> None:
            calls.append(session_id)

        # given
        registry = (
            RegistryBuilder()
            .register(MyService)(MyService)
            .register_method(MyContext, method_name='create')(create_impl)
            .register_method_hook(MyContext, method_name='create')(my_hook)
        )

        # when
        ctx = compile(MyContext, registry.build(), rules)
        result = ctx.create(branch_id=1, session_id=2)

        # then
        assert isinstance(result, MyService)
        assert calls == [2]


class TestCallableCompilation:
    def test_compile_callable_returns_factory(self, rules: RuleGraph) -> None:
        """Passing a callable to compile() returns a callable factory."""

        class MyService:
            pass

        def create_service() -> MyService:
            return MyService()

        registry = RegistryBuilder().register(MyService)(MyService)
        native = registry.build()

        factory = compile(create_service, native, rules)

        assert callable(factory)
        result = factory()
        assert isinstance(result, MyService)

    def test_compile_callable_resolves_params(self, rules: RuleGraph) -> None:
        """Callable parameters become caller-supplied seeds."""

        class Config:
            pass

        class MyService:
            def __init__(self, config: Config) -> None:
                self.config: Config = config

        def create_service(config: Config) -> MyService:
            return MyService(config)

        registry = (
            RegistryBuilder().register(Config)(Config).register(MyService)(MyService)
        )
        native = registry.build()

        factory = compile(create_service, native, rules)

        assert callable(factory)
        result = factory(config=Config())
        assert isinstance(result, MyService)
        assert isinstance(result.config, Config)

    def test_compile_callable_with_seed_params(self, rules: RuleGraph) -> None:
        """Callable params become runtime seeds on the returned factory."""

        class Config:
            pass

        class MyContext(typing.Protocol):
            @property
            def config(self) -> Config: ...

        def my_factory(name: str) -> MyContext:
            raise NotImplementedError(name)

        registry = RegistryBuilder().register(Config)(Config)
        native = registry.build()

        factory = compile(my_factory, native, rules)

        assert callable(factory)
        ctx = factory(name='hello')
        assert isinstance(ctx.config, Config)

    def test_compile_callable_mixes_seeds_and_resolved(self, rules: RuleGraph) -> None:
        """Factory with seed and registry-resolvable parameters."""

        class Config:
            pass

        class MyService:
            def __init__(self, config: Config) -> None:
                self.config: Config = config

        class MyContext(typing.Protocol):
            @property
            def service(self) -> MyService: ...

        def my_factory(name: str, config: Config) -> MyContext:
            raise NotImplementedError(f'{name}{config}')

        registry = (
            RegistryBuilder().register(Config)(Config).register(MyService)(MyService)
        )
        native = registry.build()

        factory = compile(my_factory, native, rules)

        assert callable(factory)
        ctx = factory(name='test', config=Config())
        assert isinstance(ctx.service, MyService)
        assert isinstance(ctx.service.config, Config)
