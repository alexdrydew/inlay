import argparse
import types
from time import perf_counter
from typing import Annotated, Literal, Protocol, TypedDict, cast, final

from inlay import LazyRef, RegistryBuilder, compile, qual
from inlay.default import default_rules

type Scenario = Literal['portable', 'env-sensitive']


@final
class Value:
    pass


class BranchState(TypedDict):
    branch_id: int


class CompiledRoot(Protocol):
    def with_ai_context(self) -> object: ...


def make_character_acl(character_context: type[object]) -> type[object]:
    def __init__(self, ctx_ref: object, branch_id: int | None = None) -> None:
        self.ctx_ref = ctx_ref
        self.branch_id = branch_id

    __init__.__annotations__ = {
        'ctx_ref': LazyRef[character_context],
        'branch_id': int | None,
        'return': None,
    }

    return types.new_class(
        'CharacterAcl', (), {}, lambda ns: ns.update({'__init__': __init__})
    )


def make_character_context(
    *, depth: int, value_count: int, acl_type: type[object] | None
) -> type[object]:
    next_context: type[object] | None = None

    for level in reversed(range(depth)):
        annotations: dict[str, object] = {
            f'value_{index}': Value for index in range(value_count)
        }
        if next_context is not None:
            annotations['next'] = next_context
        if acl_type is not None and level == 0:
            annotations['acl'] = acl_type

        namespace: dict[str, object] = {
            '__annotations__': annotations,
            '__module__': __name__,
        }
        next_context = types.new_class(
            f'CharacterContext{level}',
            (Protocol,),
            {},
            lambda ns, namespace=namespace: ns.update(namespace),
        )

    assert next_context is not None
    return next_context


def make_query_type(
    *,
    character_gateway: type[object],
    scenario: Scenario,
) -> type[object]:
    if scenario == 'env-sensitive':

        def __init__(
            self,
            character_ctx: object,
            branch_id: Annotated[int, qual('ai')],
        ) -> None:
            self.target = character_ctx
            self.branch_id = branch_id

        __init__.__annotations__ = {
            'character_ctx': Annotated[character_gateway, qual('character')],
            'branch_id': Annotated[int, qual('ai')],
            'return': None,
        }
    else:

        def __init__(
            self,
            character_ctx: object,
        ) -> None:
            self.target = character_ctx

        __init__.__annotations__ = {
            'character_ctx': Annotated[character_gateway, qual('character')],
            'return': None,
        }

    return types.new_class(
        'CharacterQuery', (), {}, lambda ns: ns.update({'__init__': __init__})
    )


def make_ai_branch_context(
    *,
    depth: int,
    value_count: int,
    query_count: int,
    query_type: type[object],
    scenario: Scenario,
) -> type[object]:
    next_context: type[object] | None = None

    for level in reversed(range(depth)):
        annotations: dict[str, object] = {
            f'value_{index}': Value for index in range(value_count)
        }
        for query_index in range(query_count):
            annotations[f'query_{query_index}'] = query_type
        if next_context is not None:
            annotations['next'] = next_context
        if scenario == 'env-sensitive':
            annotations['branch_id'] = int

        namespace: dict[str, object] = {
            '__annotations__': annotations,
            '__module__': __name__,
        }
        next_context = types.new_class(
            f'AiBranchContext{level}',
            (Protocol,),
            {},
            lambda ns, namespace=namespace: ns.update(namespace),
        )

    assert next_context is not None
    return next_context


def make_character_gateway(character_acl_type: type[object]) -> type[object]:
    def __init__(self, acl: object, value_0: Value) -> None:
        self.acl = acl
        self.value_0 = value_0

    __init__.__annotations__ = {
        'acl': character_acl_type,
        'value_0': Value,
        'return': None,
    }

    return types.new_class(
        'CharacterGateway', (), {}, lambda ns: ns.update({'__init__': __init__})
    )


def make_ai_context(
    ai_branch_context: type[object], branch_count: int
) -> tuple[type[object], list[str]]:
    method_names = [f'branch_{index}' for index in range(branch_count)]
    namespace: dict[str, object] = {'__module__': __name__}

    for method_name in method_names:

        def transition(self: object) -> object: ...

        transition.__name__ = method_name
        transition.__qualname__ = f'AiContext.{method_name}'
        transition.__annotations__ = {'return': ai_branch_context}
        namespace[method_name] = transition

    ai_context = types.new_class(
        'AiContext', (Protocol,), {}, lambda ns: ns.update(namespace)
    )
    return ai_context, method_names


def make_root_context(
    ai_context: type[object], character_context: type[object]
) -> type[object]:
    def with_ai_context(self: object) -> object: ...

    def with_character_context(
        self: object,
    ) -> object: ...

    with_ai_context.__annotations__ = {'return': Annotated[ai_context, qual('ai')]}
    with_character_context.__annotations__ = {
        'return': Annotated[character_context, qual('character')]
    }

    return types.new_class(
        'RootContext',
        (Protocol,),
        {},
        lambda ns: ns.update({
            '__module__': __name__,
            'with_ai_context': with_ai_context,
            'with_character_context': with_character_context,
        }),
    )


def make_ai_provider(method_names: list[str]) -> type[object]:
    namespace: dict[str, object] = {'__module__': __name__}

    def make_transition(branch_id: int):
        def transition(self: object) -> BranchState:
            return {'branch_id': branch_id}

        transition.__annotations__ = {'return': BranchState}
        return transition

    for index, method_name in enumerate(method_names):
        transition = make_transition(index)
        transition.__name__ = method_name
        transition.__qualname__ = f'AiProvider.{method_name}'
        namespace[method_name] = transition

    return types.new_class('AiProvider', (), {}, lambda ns: ns.update(namespace))


def build_registry(
    *,
    ai_context: type[object],
    method_names: list[str],
    query_type: type[object],
    character_acl_type: type[object],
    character_gateway: type[object],
) -> RegistryBuilder:
    ai_provider = make_ai_provider(method_names)

    character_registry = (
        RegistryBuilder()
        .register(Value)(Value)
        .register(character_acl_type)(character_acl_type)
        .register(character_gateway)(character_gateway)
    )
    ai_registry = RegistryBuilder()
    for method_name in method_names:
        ai_registry = ai_registry.register_method(ai_context, method_name=method_name)(
            ai_provider
        )

    return (
        RegistryBuilder()
        .register(query_type, qualifiers=qual('ai'))(query_type)
        .register(Value, qualifiers=qual('ai'))(Value)
        .include(character_registry, qualifiers=qual('character'))
        .include(ai_registry, qualifiers=qual('ai'))
    )


def run_once(
    *,
    scenario: Scenario,
    branches: int,
    depth: int,
    queries: int,
    values: int,
    character_depth: int,
    invoke: bool,
) -> None:
    character_context = make_character_context(
        depth=character_depth, value_count=values, acl_type=None
    )
    character_acl_type = make_character_acl(character_context)
    character_context = make_character_context(
        depth=character_depth,
        value_count=values,
        acl_type=character_acl_type,
    )
    character_gateway = make_character_gateway(character_acl_type)
    query_type = make_query_type(
        character_gateway=character_gateway,
        scenario=scenario,
    )
    ai_branch_context = make_ai_branch_context(
        depth=depth,
        value_count=values,
        query_count=queries,
        query_type=query_type,
        scenario=scenario,
    )
    ai_context, method_names = make_ai_context(ai_branch_context, branches)
    root_context = make_root_context(ai_context, character_context)

    registry_started = perf_counter()
    native = build_registry(
        ai_context=ai_context,
        method_names=method_names,
        query_type=query_type,
        character_acl_type=character_acl_type,
        character_gateway=character_gateway,
    ).build()
    registry_elapsed = perf_counter() - registry_started

    compile_started = perf_counter()
    root = cast(CompiledRoot, compile(root_context, native, default_rules()))
    compile_elapsed = perf_counter() - compile_started

    invoke_elapsed = None
    if invoke:
        invoke_started = perf_counter()
        ai_context_instance = root.with_ai_context()
        branch = getattr(ai_context_instance, method_names[0])()
        if values > 0:
            _ = branch.value_0
        for _ in range(depth - 1):
            branch = branch.next
        query = branch.query_0
        target = query.target
        if character_depth > 1:
            _ = target.next
        _ = target.acl
        if scenario == 'env-sensitive':
            _ = query.branch_id
        invoke_elapsed = perf_counter() - invoke_started

    print(
        ' '.join([
            'benchmark=cross_context_queries',
            f'scenario={scenario}',
            f'branches={branches}',
            f'ai_depth={depth}',
            f'character_depth={character_depth}',
            f'queries={queries}',
            f'values={values}',
            f'registry={registry_elapsed:.4f}s',
            f'compile={compile_elapsed:.4f}s',
            f'invoke={invoke_elapsed:.4f}s'
            if invoke_elapsed is not None
            else 'invoke=skipped',
        ])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scenario',
        choices=['portable', 'env-sensitive', 'all'],
        default='all',
    )
    parser.add_argument('--branches', type=int, default=32)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--queries', type=int, default=8)
    parser.add_argument('--values', type=int, default=16)
    parser.add_argument('--character-depth', type=int, default=2)
    parser.add_argument('--invoke', action='store_true')
    args = parser.parse_args()

    scenarios: list[Scenario]
    if args.scenario == 'all':
        scenarios = ['portable', 'env-sensitive']
    else:
        scenarios = [args.scenario]

    for scenario in scenarios:
        run_once(
            scenario=scenario,
            branches=args.branches,
            depth=args.depth,
            queries=args.queries,
            values=args.values,
            character_depth=args.character_depth,
            invoke=args.invoke,
        )


if __name__ == '__main__':
    main()
