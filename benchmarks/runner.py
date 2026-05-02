import argparse
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from inlay import Registry, compile, normalize
from inlay.default import default_rules

type ArgKind = Literal['int', 'str']
type CompileMode = Literal['compile', 'registry_compile']
type BuildMetric = Literal['registry', 'generation', 'build_ms']
type CompileMetric = Literal['compile', 'compile_ms']
type ProductionModuleKind = Literal['entry', 'scope', 'write']
type ProductionExtraTransition = Literal['none', 'worker', 'flow']

TEMPLATE_DIR = Path(__file__).resolve().parent / 'tmplates'


@dataclass(frozen=True)
class BenchmarkArg:
    name: str
    cli: str
    kind: ArgKind
    default: int | str
    choices: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    args: tuple[BenchmarkArg, ...]
    output_fields: tuple[tuple[str, str], ...]
    positive_args: tuple[str, ...]
    compile_mode: CompileMode
    build_metric: BuildMetric
    compile_metric: CompileMetric
    invoke: bool
    catch_compile: bool
    scenario_arg: str | None
    scenario_all: str
    scenario_values: tuple[str, ...]


@dataclass(frozen=True)
class ProductionModuleConfig:
    name: str
    kind: ProductionModuleKind
    extra_transition: ProductionExtraTransition
    read_extras: tuple[str, ...]
    write_extras: tuple[str, ...]


@dataclass(frozen=True)
class ProductionTemplateModule:
    name: str
    title: str
    kind: ProductionModuleKind
    extra_transition: ProductionExtraTransition
    read_extras: tuple[str, ...]
    write_extras: tuple[str, ...]
    slot: int
    alt_slot: int


BENCHMARK_CONFIGS: dict[str, BenchmarkConfig] = {
    'cache_portability': BenchmarkConfig(
        name='cache_portability',
        args=(
            BenchmarkArg(
                name='scenario',
                cli='--scenario',
                kind='str',
                default='all',
                choices=('portable', 'env-sensitive', 'all'),
            ),
            BenchmarkArg(
                name='transition_count',
                cli='--transitions',
                kind='int',
                default=16,
                choices=(),
            ),
            BenchmarkArg(
                name='depth',
                cli='--depth',
                kind='int',
                default=3,
                choices=(),
            ),
            BenchmarkArg(
                name='property_count',
                cli='--properties',
                kind='int',
                default=32,
                choices=(),
            ),
        ),
        output_fields=(
            ('scenario', 'scenario'),
            ('transitions', 'transition_count'),
            ('depth', 'depth'),
            ('properties', 'property_count'),
        ),
        positive_args=('transition_count', 'depth'),
        compile_mode='compile',
        build_metric='registry',
        compile_metric='compile',
        invoke=True,
        catch_compile=False,
        scenario_arg='scenario',
        scenario_all='all',
        scenario_values=('portable', 'env-sensitive'),
    ),
    'cross_context_queries': BenchmarkConfig(
        name='cross_context_queries',
        args=(
            BenchmarkArg(
                name='scenario',
                cli='--scenario',
                kind='str',
                default='all',
                choices=('portable', 'env-sensitive', 'all'),
            ),
            BenchmarkArg(
                name='transition_count',
                cli='--transitions',
                kind='int',
                default=32,
                choices=(),
            ),
            BenchmarkArg(
                name='source_depth',
                cli='--source-depth',
                kind='int',
                default=3,
                choices=(),
            ),
            BenchmarkArg(
                name='queries',
                cli='--queries',
                kind='int',
                default=8,
                choices=(),
            ),
            BenchmarkArg(
                name='values',
                cli='--values',
                kind='int',
                default=16,
                choices=(),
            ),
            BenchmarkArg(
                name='target_depth',
                cli='--target-depth',
                kind='int',
                default=2,
                choices=(),
            ),
        ),
        output_fields=(
            ('scenario', 'scenario'),
            ('transitions', 'transition_count'),
            ('source_depth', 'source_depth'),
            ('target_depth', 'target_depth'),
            ('queries', 'queries'),
            ('values', 'values'),
        ),
        positive_args=('transition_count', 'source_depth', 'target_depth'),
        compile_mode='compile',
        build_metric='registry',
        compile_metric='compile',
        invoke=True,
        catch_compile=False,
        scenario_arg='scenario',
        scenario_all='all',
        scenario_values=('portable', 'env-sensitive'),
    ),
    'production_shape': BenchmarkConfig(
        name='production_shape',
        args=(
            BenchmarkArg(
                name='scenario',
                cli='--scenario',
                kind='str',
                default='env-sensitive',
                choices=('portable', 'env-sensitive', 'all'),
            ),
            BenchmarkArg(
                name='density',
                cli='--density',
                kind='int',
                default=4,
                choices=(),
            ),
            BenchmarkArg(
                name='handler_count',
                cli='--handlers',
                kind='int',
                default=4,
                choices=(),
            ),
            BenchmarkArg(
                name='shared_slots',
                cli='--shared-slots',
                kind='int',
                default=2,
                choices=(),
            ),
            BenchmarkArg(
                name='enabled_modules',
                cli='--modules',
                kind='str',
                default='alpha,beta,gamma,delta,epsilon',
                choices=(),
            ),
        ),
        output_fields=(
            ('scenario', 'scenario'),
            ('density', 'density'),
            ('handlers', 'handler_count'),
            ('shared_slots', 'shared_slots'),
            ('modules', 'active_modules'),
        ),
        positive_args=('density',),
        compile_mode='compile',
        build_metric='generation',
        compile_metric='compile',
        invoke=True,
        catch_compile=False,
        scenario_arg='scenario',
        scenario_all='all',
        scenario_values=('portable', 'env-sensitive'),
    ),
    'staged_neutral_cycle': BenchmarkConfig(
        name='staged_neutral_cycle',
        args=(
            BenchmarkArg(
                name='stage',
                cli='--stage',
                kind='str',
                default='relay',
                choices=(
                    'cycle',
                    'cycle-write',
                    'fork',
                    'open',
                    'hooks',
                    'relay',
                    'cluster',
                    'cross',
                    'tracks',
                ),
            ),
            BenchmarkArg(
                name='density',
                cli='--density',
                kind='int',
                default=4,
                choices=(),
            ),
            BenchmarkArg(
                name='handler_count',
                cli='--handlers',
                kind='int',
                default=4,
                choices=(),
            ),
            BenchmarkArg(
                name='shared_slots',
                cli='--shared-slots',
                kind='int',
                default=1,
                choices=(),
            ),
            BenchmarkArg(
                name='tracks',
                cli='--tracks',
                kind='int',
                default=4,
                choices=(),
            ),
        ),
        output_fields=(
            ('stage', 'stage'),
            ('density', 'density'),
            ('handlers', 'handler_count'),
            ('shared_slots', 'shared_slots'),
            ('tracks', 'tracks'),
        ),
        positive_args=(),
        compile_mode='registry_compile',
        build_metric='build_ms',
        compile_metric='compile_ms',
        invoke=False,
        catch_compile=True,
        scenario_arg=None,
        scenario_all='all',
        scenario_values=(),
    ),
}

PRODUCTION_MODULE_CONFIGS: tuple[ProductionModuleConfig, ...] = (
    ProductionModuleConfig(
        name='alpha',
        kind='entry',
        extra_transition='none',
        read_extras=(),
        write_extras=(),
    ),
    ProductionModuleConfig(
        name='beta',
        kind='entry',
        extra_transition='flow',
        read_extras=('BetaDeltaQueries', 'BetaEpsilonData'),
        write_extras=('BetaDeltaQueries', 'BetaEpsilonData'),
    ),
    ProductionModuleConfig(
        name='gamma',
        kind='entry',
        extra_transition='worker',
        read_extras=(
            'GammaAlphaQueries',
            'GammaDeltaQueries',
            'GammaEpsilonQueries',
        ),
        write_extras=('GammaAlphaQueries', 'GammaDeltaQueries'),
    ),
    ProductionModuleConfig(
        name='delta',
        kind='write',
        extra_transition='none',
        read_extras=(),
        write_extras=(),
    ),
    ProductionModuleConfig(
        name='epsilon',
        kind='scope',
        extra_transition='none',
        read_extras=(),
        write_extras=(),
    ),
)


def title(name: str) -> str:
    return ''.join(part.capitalize() for part in name.split('_'))


def int_value(values: dict[str, object], key: str) -> int:
    value = values[key]
    assert isinstance(value, int), f'Expected {key} to be an int'
    return value


def str_value(values: dict[str, object], key: str) -> str:
    value = values[key]
    assert isinstance(value, str), f'Expected {key} to be a string'
    return value


def enabled_modules(values: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        module.strip()
        for module in str_value(values, 'enabled_modules').split(',')
        if module.strip()
    )


def production_template_modules(
    values: dict[str, object],
) -> tuple[ProductionTemplateModule, ...]:
    requested_modules = set(enabled_modules(values))
    slot_count = max(int_value(values, 'shared_slots'), 1)
    modules: list[ProductionTemplateModule] = []
    for index, config in enumerate(PRODUCTION_MODULE_CONFIGS):
        if requested_modules and config.name not in requested_modules:
            continue
        modules.append(
            ProductionTemplateModule(
                name=config.name,
                title=title(config.name),
                kind=config.kind,
                extra_transition=config.extra_transition,
                read_extras=config.read_extras,
                write_extras=config.write_extras,
                slot=index % slot_count,
                alt_slot=(index + 1) % slot_count,
            )
        )
    return tuple(modules)


def stage_order(stage: str) -> int:
    match stage:
        case 'cycle':
            return 0
        case 'cycle-write':
            return 1
        case 'fork':
            return 2
        case 'open':
            return 3
        case 'hooks':
            return 4
        case 'relay':
            return 5
        case 'cluster' | 'tracks':
            return 6
        case 'cross':
            return 7
        case _:
            raise AssertionError(f'Unknown benchmark stage: {stage}')


def template_context(benchmark: str, values: dict[str, object]) -> dict[str, object]:
    context = dict(values)
    match benchmark:
        case 'production_shape':
            modules = production_template_modules(values)
            context['active_modules'] = modules
            context['active_module_names'] = tuple(module.name for module in modules)
            context['shared_slot_indices'] = tuple(
                range(max(int_value(values, 'shared_slots'), 1))
            )
        case 'staged_neutral_cycle':
            stage = str_value(values, 'stage')
            multi_track = stage in {'tracks', 'cross'}
            track_count = max(int_value(values, 'tracks'), 1) if multi_track else 1
            context['multi_track'] = multi_track
            context['track_count'] = track_count
            context['track_indices'] = tuple(range(track_count))
            context['shared_slot_indices'] = tuple(
                range(max(int_value(values, 'shared_slots'), 1))
            )
            context['stage_level'] = stage_order(stage)
        case _:
            pass
    return context


def env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def object_dict(value: object, message: str) -> dict[object, object]:
    assert isinstance(value, dict), message
    return cast(dict[object, object], value)


def add_benchmark_arg(parser: argparse.ArgumentParser, arg: BenchmarkArg) -> None:
    kind = int if arg.kind == 'int' else str
    if arg.choices:
        _ = parser.add_argument(
            arg.cli,
            dest=arg.name,
            type=kind,
            default=arg.default,
            choices=arg.choices,
        )
    else:
        _ = parser.add_argument(
            arg.cli,
            dest=arg.name,
            type=kind,
            default=arg.default,
        )


def parse_args(config: BenchmarkConfig, argv: list[str]) -> dict[str, object]:
    parser = argparse.ArgumentParser()
    for arg in config.args:
        add_benchmark_arg(parser, arg)
    if config.invoke:
        _ = parser.add_argument('--invoke', action='store_true')
    _ = parser.add_argument('--dump-generated', action='store_true')
    parsed = cast(dict[str, object], vars(parser.parse_args(argv)))
    for arg_name in config.positive_args:
        value = parsed[arg_name]
        assert isinstance(value, int), f'Expected {arg_name} to be an int'
        assert value > 0, f'Benchmark requires positive {arg_name}'
    return parsed


def scenario_runs(
    config: BenchmarkConfig, parsed: dict[str, object]
) -> tuple[dict[str, object], ...]:
    if config.scenario_arg is None:
        return (parsed,)
    scenario_value = parsed[config.scenario_arg]
    assert isinstance(scenario_value, str), 'Expected scenario value to be a string'
    if scenario_value != config.scenario_all:
        return (parsed,)
    return tuple(
        {**parsed, config.scenario_arg: scenario} for scenario in config.scenario_values
    )


def module_name(benchmark: str, values: dict[str, object]) -> str:
    parts = [benchmark.replace('-', '_')]
    for key, value in sorted(values.items()):
        if key in {'dump_generated', 'invoke'}:
            continue
        parts.append(f'{key}_{str(value).replace(",", "_").replace("-", "_")}')
    return '_benchmark_' + '_'.join(parts)


def exec_generated(
    *,
    benchmark: str,
    source: str,
    values: dict[str, object],
) -> tuple[dict[str, object], str]:
    name = module_name(benchmark, values)
    generated = types.ModuleType(name)
    sys.modules[name] = generated
    namespace = generated.__dict__
    namespace['__file__'] = str(TEMPLATE_DIR.parent / f'{benchmark}.py')
    exec(source, namespace)
    return namespace, name


def generated_source(namespace: dict[str, object], fallback: str) -> str:
    source = namespace.get('GENERATED_SOURCE')
    if isinstance(source, str):
        return source
    return fallback


def merge_output_values(
    values: dict[str, object], namespace: dict[str, object]
) -> dict[str, object]:
    merged = dict(values)
    output_values = namespace.get('OUTPUT_VALUES')
    if output_values is None:
        return merged
    config = object_dict(output_values, 'Expected OUTPUT_VALUES to be a mapping')
    for key, value in config.items():
        assert isinstance(key, str), 'Expected OUTPUT_VALUES keys to be strings'
        merged[key] = value
    return merged


def cleanup(namespace: dict[str, object], module: str) -> None:
    inner_module = namespace.get('MODULE_NAME')
    if isinstance(inner_module, str) and inner_module in sys.modules:
        del sys.modules[inner_module]
    if module in sys.modules:
        del sys.modules[module]


def format_time(metric: BuildMetric | CompileMetric, elapsed: float) -> str:
    match metric:
        case 'build_ms' | 'compile_ms':
            return f'{metric}={elapsed * 1000.0:.3f}'
        case 'registry' | 'generation' | 'compile':
            return f'{metric}={elapsed:.4f}s'


def format_value(value: object) -> str:
    if isinstance(value, tuple):
        tuple_value = cast(tuple[object, ...], value)
        return ','.join(str(item) for item in tuple_value)
    return str(value)


def output_parts(
    *,
    config: BenchmarkConfig,
    values: dict[str, object],
    build_elapsed: float,
    compile_elapsed: float,
    invoke_elapsed: float | None,
    outcome: str | None,
) -> list[str]:
    parts = [f'benchmark={config.name}']
    for label, name in config.output_fields:
        parts.append(f'{label}={format_value(values[name])}')
    parts.append(format_time(config.build_metric, build_elapsed))
    parts.append(format_time(config.compile_metric, compile_elapsed))
    if config.invoke:
        parts.append(
            f'invoke={invoke_elapsed:.4f}s'
            if invoke_elapsed is not None
            else 'invoke=skipped'
        )
    if outcome is not None:
        parts.append(f'outcome={outcome}')
    return parts


def compile_generated(
    *,
    config: BenchmarkConfig,
    namespace: dict[str, object],
) -> tuple[object | None, str | None]:
    target = namespace['TARGET']
    registry = cast(Registry, namespace['REGISTRY'])
    try:
        match config.compile_mode:
            case 'compile':
                return (
                    compile(cast(type[object], target), registry, default_rules()),
                    None,
                )
            case 'registry_compile':
                _ = registry.compile(default_rules(), normalize(target))
                return None, 'ok'
    except Exception as exc:  # noqa: BLE001
        if config.catch_compile:
            return None, type(exc).__name__
        raise


def run_once(
    *,
    template_name: str,
    config: BenchmarkConfig,
    values: dict[str, object],
) -> None:
    template = env().get_template(template_name)
    build_started = perf_counter()
    source = template.render(**template_context(config.name, values))
    namespace, generated_module = exec_generated(
        benchmark=config.name,
        source=source,
        values=values,
    )
    output_values = merge_output_values(values, namespace)
    build_elapsed = perf_counter() - build_started

    if cast(bool, values['dump_generated']):
        print(generated_source(namespace, source))

    compile_started = perf_counter()
    root, outcome = compile_generated(config=config, namespace=namespace)
    compile_elapsed = perf_counter() - compile_started

    invoke_elapsed: float | None = None
    try:
        if config.invoke and cast(bool, values['invoke']):
            invoke = cast(Callable[[object], None], namespace['invoke'])
            assert root is not None, 'Cannot invoke registry_compile benchmark'
            invoke_started = perf_counter()
            invoke(root)
            invoke_elapsed = perf_counter() - invoke_started
    finally:
        cleanup(namespace, generated_module)

    print(
        ' '.join(
            output_parts(
                config=config,
                values=output_values,
                build_elapsed=build_elapsed,
                compile_elapsed=compile_elapsed,
                invoke_elapsed=invoke_elapsed,
                outcome=outcome if config.catch_compile else None,
            )
        )
    )


def load_config(benchmark: str, template_name: str) -> BenchmarkConfig:
    try:
        _ = env().get_template(template_name)
    except TemplateNotFound as exc:
        raise SystemExit(
            f'benchmark not found: benchmarks/templates/{template_name}'
        ) from exc
    config = BENCHMARK_CONFIGS.get(benchmark)
    if config is None:
        raise SystemExit(f'benchmark config not found: {benchmark}')
    return config


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('BENCH=<name> is required')
    benchmark = sys.argv[1]
    template_name = f'{benchmark}.py.jinja'
    config = load_config(benchmark, template_name)
    parsed = parse_args(config, sys.argv[2:])
    for values in scenario_runs(config, parsed):
        run_once(template_name=template_name, config=config, values=values)


if __name__ == '__main__':
    main()
