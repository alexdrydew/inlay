import argparse
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal, cast

from inlay import Registry, compile
from inlay.default import default_rules

if TYPE_CHECKING:
    from jinja2 import Environment
else:
    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except ModuleNotFoundError as exc:  # pragma: no cover - benchmark bootstrap guard
        raise SystemExit(
            'This benchmark needs jinja2. Run it from the workspace root or install jinja2.'
        ) from exc

type Scenario = Literal['portable', 'env-sensitive']

TEMPLATE_DIR = Path(__file__).resolve().parent / 'templates'
TEMPLATE_NAME = 'density_context.py.j2'


def render_source(
    *,
    scenario: Scenario,
    density: int,
    hook_fanout: int,
    query_repeats: int,
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
        density=density,
        hook_fanout=hook_fanout,
        query_repeats=query_repeats,
        env_sensitive=scenario == 'env-sensitive',
    )


def build_generated(
    *,
    scenario: Scenario,
    density: int,
    hook_fanout: int,
    query_repeats: int,
) -> tuple[type[object], Registry, str]:
    source = render_source(
        scenario=scenario,
        density=density,
        hook_fanout=hook_fanout,
        query_repeats=query_repeats,
    )
    namespace: dict[str, object] = {}
    exec(source, namespace)
    root_type = cast(type[object], namespace['BenchmarkRoot'])
    registry = cast(Registry, namespace['REGISTRY'])
    return root_type, registry, source


def run_once(
    *,
    scenario: Scenario,
    density: int,
    hook_fanout: int,
    query_repeats: int,
    dump_generated: bool,
) -> None:
    generation_started = perf_counter()
    root_type, registry, source = build_generated(
        scenario=scenario,
        density=density,
        hook_fanout=hook_fanout,
        query_repeats=query_repeats,
    )
    generation_elapsed = perf_counter() - generation_started

    if dump_generated:
        print(source)

    compile_started = perf_counter()
    _ = compile(root_type, registry, default_rules())
    compile_elapsed = perf_counter() - compile_started

    print(
        ' '.join([
            'benchmark=density_template',
            f'scenario={scenario}',
            f'density={density}',
            f'hook_fanout={hook_fanout}',
            f'query_repeats={query_repeats}',
            f'generation={generation_elapsed:.4f}s',
            f'compile={compile_elapsed:.4f}s',
        ])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        '--scenario',
        choices=['portable', 'env-sensitive', 'all'],
        default='env-sensitive',
    )
    _ = parser.add_argument('--density', type=int, default=8)
    _ = parser.add_argument('--hook-fanout', type=int, default=2)
    _ = parser.add_argument('--query-repeats', type=int, default=6)
    _ = parser.add_argument('--dump-generated', action='store_true')
    args = parser.parse_args()

    scenarios: tuple[Scenario, ...]
    if args.scenario == 'all':
        scenarios = ('portable', 'env-sensitive')
    else:
        scenarios = (cast(Scenario, args.scenario),)

    for scenario in scenarios:
        run_once(
            scenario=scenario,
            density=args.density,
            hook_fanout=args.hook_fanout,
            query_repeats=args.query_repeats,
            dump_generated=args.dump_generated,
        )


if __name__ == '__main__':
    main()
