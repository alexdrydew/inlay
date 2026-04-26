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
TEMPLATE_NAME = 'cross_context_queries_context.py.j2'


def render_source(
    *,
    scenario: Scenario,
    transition_count: int,
    source_depth: int,
    queries: int,
    values: int,
    target_depth: int,
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
        source_depth=source_depth,
        queries=queries,
        values=values,
        target_depth=target_depth,
    )


def build_generated(
    *,
    scenario: Scenario,
    transition_count: int,
    source_depth: int,
    queries: int,
    values: int,
    target_depth: int,
) -> tuple[type[object], Registry, Callable[[object], None], str, str]:
    source = render_source(
        scenario=scenario,
        transition_count=transition_count,
        source_depth=source_depth,
        queries=queries,
        values=values,
        target_depth=target_depth,
    )
    module_name = '_cross_context_queries_' + '_'.join([
        scenario.replace('-', '_'),
        str(transition_count),
        str(source_depth),
        str(queries),
        str(values),
        str(target_depth),
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
    source_depth: int,
    queries: int,
    values: int,
    target_depth: int,
    invoke: bool,
    dump_generated: bool,
) -> None:
    assert transition_count > 0, 'Benchmark requires at least one transition'
    assert source_depth > 0, 'Benchmark requires positive source context depth'
    assert target_depth > 0, 'Benchmark requires positive target context depth'

    registry_started = perf_counter()
    root_context, native, invoke_generated, source, module_name = build_generated(
        scenario=scenario,
        transition_count=transition_count,
        source_depth=source_depth,
        queries=queries,
        values=values,
        target_depth=target_depth,
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
            'benchmark=cross_context_queries',
            f'scenario={scenario}',
            f'transitions={transition_count}',
            f'source_depth={source_depth}',
            f'target_depth={target_depth}',
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
    _ = parser.add_argument(
        '--scenario',
        choices=['portable', 'env-sensitive', 'all'],
        default='all',
    )
    _ = parser.add_argument('--transitions', type=int, default=32)
    _ = parser.add_argument('--source-depth', type=int, default=3)
    _ = parser.add_argument('--queries', type=int, default=8)
    _ = parser.add_argument('--values', type=int, default=16)
    _ = parser.add_argument('--target-depth', type=int, default=2)
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
            source_depth=cast(int, parsed['source_depth']),
            queries=cast(int, parsed['queries']),
            values=cast(int, parsed['values']),
            target_depth=cast(int, parsed['target_depth']),
            invoke=cast(bool, parsed['invoke']),
            dump_generated=cast(bool, parsed['dump_generated']),
        )


if __name__ == '__main__':
    main()
