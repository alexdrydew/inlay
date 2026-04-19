import argparse
import types
from time import perf_counter
from typing import Literal, Protocol, TypedDict, final

from inlay import RegistryBuilder, compile
from inlay.default import default_rules

type Scenario = Literal['portable', 'env-sensitive']


@final
class Value:
    pass


class BranchState(TypedDict):
    branch_id: int


def make_branch_context(
    *, depth: int, property_count: int, scenario: Scenario
) -> type[Protocol]:
    next_context: type[Protocol] | None = None

    for level in reversed(range(depth)):
        annotations: dict[str, object] = {
            f'value_{index}': Value for index in range(property_count)
        }
        if next_context is not None:
            annotations['next'] = next_context
        if scenario == 'env-sensitive':
            annotations['branch_id'] = int

        namespace: dict[str, object] = {
            '__annotations__': annotations,
            '__module__': __name__,
        }
        next_context = types.new_class(
            f'BranchContext{level}',
            (Protocol,),
            {},
            lambda ns, namespace=namespace: ns.update(namespace),
        )

    assert next_context is not None
    return next_context


def make_root_context(
    branch_context: type[Protocol], method_count: int
) -> tuple[type[Protocol], list[str]]:
    method_names = [f'branch_{index}' for index in range(method_count)]
    namespace: dict[str, object] = {'__module__': __name__}

    for method_name in method_names:

        def transition(self) -> branch_context: ...

        transition.__name__ = method_name
        transition.__qualname__ = method_name
        transition.__annotations__ = {'return': branch_context}
        namespace[method_name] = transition

    root_context = types.new_class(
        'RootContext', (Protocol,), {}, lambda ns: ns.update(namespace)
    )
    return root_context, method_names


def make_provider(method_names: list[str]) -> type[object]:
    namespace: dict[str, object] = {'__module__': __name__}

    def make_transition(branch_id: int):
        def transition(self) -> BranchState:
            return {'branch_id': branch_id}

        transition.__annotations__ = {'return': BranchState}
        return transition

    for index, method_name in enumerate(method_names):
        transition = make_transition(index)

        transition.__name__ = method_name
        transition.__qualname__ = f'Provider.{method_name}'
        namespace[method_name] = transition

    return types.new_class('Provider', (), {}, lambda ns: ns.update(namespace))


def build_registry(
    root_context: type[Protocol], method_names: list[str]
) -> RegistryBuilder:
    provider = make_provider(method_names)
    registry = RegistryBuilder().register(Value)(Value)
    for method_name in method_names:
        registry = registry.register_method(root_context, method_name=method_name)(
            provider
        )
    return registry


def run_once(
    *,
    scenario: Scenario,
    branch_count: int,
    depth: int,
    property_count: int,
    invoke: bool,
) -> None:
    branch_context = make_branch_context(
        depth=depth, property_count=property_count, scenario=scenario
    )
    root_context, method_names = make_root_context(branch_context, branch_count)

    registry_started = perf_counter()
    registry = build_registry(root_context, method_names)
    native = registry.build()
    registry_elapsed = perf_counter() - registry_started

    compile_started = perf_counter()
    root = compile(root_context, native, default_rules())
    compile_elapsed = perf_counter() - compile_started

    invoke_elapsed = None
    if invoke:
        invoke_started = perf_counter()
        branch = getattr(root, method_names[0])()
        if property_count > 0:
            _ = branch.value_0
        for _ in range(depth - 1):
            branch = branch.next
        if scenario == 'env-sensitive':
            _ = branch.branch_id
        invoke_elapsed = perf_counter() - invoke_started

    print(
        ' '.join([
            f'scenario={scenario}',
            f'branches={branch_count}',
            f'depth={depth}',
            f'properties={property_count}',
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
    parser.add_argument('--branches', type=int, default=16)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--properties', type=int, default=32)
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
            branch_count=args.branches,
            depth=args.depth,
            property_count=args.properties,
            invoke=args.invoke,
        )


if __name__ == '__main__':
    main()
