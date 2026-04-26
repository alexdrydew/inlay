import argparse
import sys
import types
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from inlay import Registry, compile
from inlay.default import default_rules

type Scenario = Literal['portable', 'env-sensitive']

TEMPLATE_DIR = Path(__file__).resolve().parent / 'templates'
TEMPLATE_NAME = 'cache_portability_context.py.j2'


def render_source(
    *,
    scenario: Scenario,
    transition_count: int,
    depth: int,
    property_count: int,
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(TEMPLATE_NAME)
    return template.render(
        scenario=scenario,
        transition_count=transition_count,
        depth=depth,
        property_count=property_count,
    )


def build_generated(
    *,
    scenario: Scenario,
    transition_count: int,
    depth: int,
    property_count: int,
) -> tuple[type[object], Registry, Callable[[object], None], str, str]:
    source = render_source(
        scenario=scenario,
        transition_count=transition_count,
        depth=depth,
        property_count=property_count,
    )
    module_name = '_cache_portability_' + '_'.join([
        scenario.replace('-', '_'),
        str(transition_count),
        str(depth),
        str(property_count),
    ])
    generated = types.ModuleType(module_name)
    sys.modules[module_name] = generated
    namespace = generated.__dict__
    exec(source, namespace)
    return (
        cast(type[object], namespace['RootContext']),
        cast(Registry, namespace['REGISTRY']),
        cast(Callable[[object], None], namespace['invoke']),
        source,
        module_name,
    )


def run_once(
    *,
    scenario: Scenario,
    transition_count: int,
    depth: int,
    property_count: int,
    invoke: bool,
    dump_generated: bool,
) -> None:
    assert transition_count > 0, 'Benchmark requires at least one transition'
    assert depth > 0, 'Benchmark requires positive context depth'

    registry_started = perf_counter()
    root_context, native, invoke_generated, source, module_name = build_generated(
        scenario=scenario,
        transition_count=transition_count,
        depth=depth,
        property_count=property_count,
    )
    registry_elapsed = perf_counter() - registry_started

    if dump_generated:
        print(source)

    compile_elapsed = 0.0
    invoke_elapsed: float | None = None
    try:
        compile_started = perf_counter()
        root = compile(root_context, native, default_rules())
        compile_elapsed = perf_counter() - compile_started

        if invoke:
            invoke_started = perf_counter()
            invoke_generated(root)
            invoke_elapsed = perf_counter() - invoke_started
    finally:
        del sys.modules[module_name]

    print(
        ' '.join([
            'benchmark=cache_portability',
            f'scenario={scenario}',
            f'transitions={transition_count}',
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
    _ = parser.add_argument(
        '--scenario',
        choices=['portable', 'env-sensitive', 'all'],
        default='all',
    )
    _ = parser.add_argument('--transitions', type=int, default=16)
    _ = parser.add_argument('--depth', type=int, default=3)
    _ = parser.add_argument('--properties', type=int, default=32)
    _ = parser.add_argument('--invoke', action='store_true')
    _ = parser.add_argument('--dump-generated', action='store_true')
    args = parser.parse_args()
    parsed = cast(dict[str, object], vars(args))
    parsed_scenario = cast(Scenario | Literal['all'], parsed['scenario'])

    scenarios: list[Scenario]
    if parsed_scenario == 'all':
        scenarios = ['portable', 'env-sensitive']
    else:
        scenarios = [parsed_scenario]

    for scenario in scenarios:
        run_once(
            scenario=scenario,
            transition_count=cast(int, parsed['transitions']),
            depth=cast(int, parsed['depth']),
            property_count=cast(int, parsed['properties']),
            invoke=cast(bool, parsed['invoke']),
            dump_generated=cast(bool, parsed['dump_generated']),
        )


if __name__ == '__main__':
    main()
