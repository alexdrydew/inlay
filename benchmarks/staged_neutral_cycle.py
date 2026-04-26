import argparse
import sys
import types
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal, cast

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from inlay import Registry, normalize, qual
from inlay.default import default_rules

type Stage = Literal[
    'cycle',
    'cycle-write',
    'fork',
    'open',
    'hooks',
    'relay',
    'cluster',
    'cross',
    'tracks',
]

TEMPLATE_DIR = Path(__file__).resolve().parent / 'templates'
TEMPLATE_NAME = 'staged_neutral_cycle_context.py.j2'


@dataclass(frozen=True)
class TypedName:
    name: str
    type: str


@dataclass(frozen=True)
class ClassSpec:
    name: str
    params: tuple[TypedName, ...]


@dataclass(frozen=True)
class TypedDictSpec:
    name: str
    fields: tuple[TypedName, ...]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    params: tuple[TypedName, ...]
    return_type: str


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    fields: tuple[TypedName, ...]
    methods: tuple[MethodSpec, ...]


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    params: tuple[TypedName, ...]
    return_type: str
    body: tuple[str, ...]


@dataclass(frozen=True)
class MethodRegistrationSpec:
    protocol_name: str
    method_name: str
    function_name: str


@dataclass(frozen=True)
class TrackRegistrySpec:
    name: str
    common_classes: tuple[str, ...]
    methods: tuple[MethodRegistrationSpec, ...]
    hooks: tuple[MethodRegistrationSpec, ...]


@dataclass(frozen=True)
class GeneratedSpec:
    class_specs: tuple[ClassSpec, ...]
    typed_dict_specs: tuple[TypedDictSpec, ...]
    protocol_specs: tuple[ProtocolSpec, ...]
    function_specs: tuple[FunctionSpec, ...]
    shared_classes: tuple[str, ...]
    registries: tuple[TrackRegistrySpec, ...]
    multi_track: bool
    track_count: int


def typed_names(params: Iterable[tuple[str, str]]) -> tuple[TypedName, ...]:
    return tuple(TypedName(name, type_ref) for name, type_ref in params)


def method_spec(
    name: str,
    params: Iterable[tuple[str, str]],
    return_type: str,
) -> MethodSpec:
    return MethodSpec(name, typed_names(params), return_type)


def add_class(
    specs: list[ClassSpec],
    name: str,
    params: Iterable[tuple[str, str]] = (),
) -> None:
    specs.append(ClassSpec(name, typed_names(params)))


def add_typed_dict(
    specs: list[TypedDictSpec],
    name: str,
    fields: Iterable[tuple[str, str]] = (),
) -> None:
    specs.append(TypedDictSpec(name, typed_names(fields)))


def add_protocol(
    specs: list[ProtocolSpec],
    name: str,
    fields: Iterable[tuple[str, str]],
    methods: Iterable[tuple[str, Iterable[tuple[str, str]], str]],
) -> None:
    specs.append(
        ProtocolSpec(
            name,
            typed_names(fields),
            tuple(
                method_spec(method_name, params, return_type)
                for method_name, params, return_type in methods
            ),
        )
    )


def add_function(
    specs: list[FunctionSpec],
    name: str,
    params: Iterable[tuple[str, str]],
    return_type: str,
    body: Iterable[str],
) -> None:
    specs.append(FunctionSpec(name, typed_names(params), return_type, tuple(body)))


def env_int(module: str) -> str:
    return f"Annotated[int, qual('{module}')]"


def write_context(type_ref: str) -> str:
    return f"Annotated[{type_ref}, qual('write')]"


def stage_order(stage: Stage) -> int:
    return {
        'cycle': 0,
        'cycle-write': 1,
        'fork': 2,
        'open': 3,
        'hooks': 4,
        'relay': 5,
        'cluster': 6,
        'tracks': 6,
        'cross': 7,
    }[stage]


def build_track_spec(
    *,
    class_specs: list[ClassSpec],
    typed_dict_specs: list[TypedDictSpec],
    protocol_specs: list[ProtocolSpec],
    function_specs: list[FunctionSpec],
    prefix: str,
    module: str,
    track_index: int,
    track_count: int,
    stage: Stage,
    density: int,
    handler_count: int,
    slot: int,
    alt_slot: int,
) -> TrackRegistrySpec:
    level = stage_order(stage)
    function_prefix = prefix.lower() if prefix else 'main'
    registry_name = f'{prefix}_COMMON' if prefix else 'COMMON'
    has_write = level >= 1
    has_fork = level >= 2
    has_open = level >= 3
    has_hooks = level >= 4
    has_relay = level >= 5
    has_cluster = level >= 6
    has_cross = level >= 7

    if has_open:
        add_typed_dict(
            typed_dict_specs,
            f'{prefix}OpenState',
            [('open_id', env_int(module))],
        )
    if has_fork:
        add_typed_dict(
            typed_dict_specs,
            f'{prefix}ForkState',
            [('fork_id', env_int(module)), ('stamp', env_int(module))],
        )
    if has_write:
        add_typed_dict(typed_dict_specs, f'{prefix}WriteState')
    if has_cross:
        add_typed_dict(
            typed_dict_specs,
            f'{prefix}ExtraState',
            [('extra_id', env_int(module))],
        )

    if has_cross:
        add_class(
            class_specs,
            f'{prefix}BaseService',
            [('shared', f'BaseShared{slot}')],
        )

    if has_open:
        add_class(
            class_specs,
            f'{prefix}EntryValue',
            [('open_id', env_int(module)), ('shared', f'EntryShared{slot}')],
        )

    if has_fork:
        loop_params = [
            ('fork_id', env_int(module)),
            ('stamp', env_int(module)),
            ('shared', f'LoopShared{slot}'),
        ]
        if has_open:
            loop_params.insert(2, ('entry', f'{prefix}EntryValue'))
        add_class(class_specs, f'{prefix}LoopValue', loop_params)

    if has_cluster:
        add_class(
            class_specs,
            f'{prefix}Executor',
            [('read_ref', f'LazyRef[{prefix}Loop]'), ('shared', f'LoopShared{slot}')],
        )
        for handler in range(handler_count):
            add_class(
                class_specs,
                f'{prefix}Relay{handler}',
                [
                    ('shared', f'HookShared{alt_slot}'),
                    ('executor', f'{prefix}Executor'),
                ],
            )

        handler_ref_params: list[tuple[str, str]] = [('shared', f'LoopShared{slot}')]
        handler_ref_params.extend(
            (f'relay_{handler}', f'LazyRef[{prefix}Relay{handler}]')
            for handler in range(handler_count)
        )
        add_class(class_specs, f'{prefix}RelayRefs', handler_ref_params)
        add_class(
            class_specs,
            f'{prefix}Audit',
            [
                ('shared', f'HookShared{alt_slot}'),
                ('executor', f'{prefix}Executor'),
                ('refs', f'{prefix}RelayRefs'),
            ],
        )

        loop_service_params: list[tuple[str, str]] = [
            ('shared', f'LoopShared{slot}'),
            ('audit', f'{prefix}Audit'),
            ('executor', f'{prefix}Executor'),
            ('refs', f'{prefix}RelayRefs'),
        ]
        if has_open:
            loop_service_params.insert(0, ('entry', f'{prefix}EntryValue'))
        if has_fork:
            loop_service_params[0:0] = [
                ('fork_id', env_int(module)),
                ('stamp', env_int(module)),
            ]
        add_class(class_specs, f'{prefix}LoopService', loop_service_params)
        add_class(
            class_specs,
            f'{prefix}LoopWriteService',
            [
                ('loop', f'{prefix}LoopService'),
                ('shared', f'HookShared{alt_slot}'),
                ('refs', f'{prefix}RelayRefs'),
            ],
        )

    if has_cross:
        sibling_indices = [
            (track_index + offset) % max(track_count, 1) for offset in (1, 2)
        ]
        sibling_base_refs = [
            f"Annotated[T{sibling}BaseService, qual('m{sibling}')]"
            if track_count > 1
            else f"Annotated[{prefix}BaseService, qual('{module}')]"
            for sibling in sibling_indices
        ]
        add_class(
            class_specs,
            f'{prefix}QueryA',
            [
                ('open_id', env_int(module)),
                ('fork_id', env_int(module)),
                ('other_base', sibling_base_refs[0]),
                ('shared', f'HookShared{slot}'),
            ],
        )
        add_class(
            class_specs,
            f'{prefix}QueryB',
            [
                ('open_id', env_int(module)),
                ('fork_id', env_int(module)),
                ('other_base', sibling_base_refs[1]),
                ('shared', f'HookShared{alt_slot}'),
            ],
        )
        add_class(
            class_specs,
            f'{prefix}ExtraService',
            [
                ('extra_id', env_int(module)),
                ('query_a', f'{prefix}QueryA'),
                ('query_b', f'{prefix}QueryB'),
                ('shared', f'HookShared{slot}'),
            ],
        )

    loop_write_fields: list[tuple[str, str]] = []
    if has_cluster:
        loop_write_fields.extend([
            ('write', f'{prefix}LoopWriteService'),
            ('refs', f"'{prefix}RelayRefs'"),
        ])
    elif has_fork:
        loop_write_fields.append(('loop', f'{prefix}LoopValue'))
    if has_cluster:
        pass
    elif has_relay:
        loop_write_fields.append(('refs', f"'{prefix}RelayRefs'"))
    elif has_write:
        loop_write_fields.append(('back', f"'{prefix}BackRef'"))
    if has_cross:
        add_protocol(
            protocol_specs,
            f'{prefix}ExtraContext',
            [
                ('extra', f'{prefix}ExtraService'),
                ('query_a', f'{prefix}QueryA'),
                ('query_b', f'{prefix}QueryB'),
            ],
            [],
        )
        add_protocol(
            protocol_specs,
            f'{prefix}LoopWrite',
            loop_write_fields,
            [
                (
                    'enter_extra',
                    [('extra_id', env_int(module))],
                    write_context(f'{prefix}ExtraContext'),
                )
            ],
        )
    elif has_write:
        add_protocol(protocol_specs, f'{prefix}LoopWrite', loop_write_fields, [])

    loop_fields: list[tuple[str, str]] = []
    if has_cluster:
        loop_fields.extend([
            ('loop', f'{prefix}LoopService'),
            ('executor', f'{prefix}Executor'),
            ('audit', f'{prefix}Audit'),
            ('refs', f"'{prefix}RelayRefs'"),
        ])
    elif has_fork:
        loop_fields.append(('loop', f'{prefix}LoopValue'))
    if has_cluster:
        pass
    elif has_relay:
        loop_fields.extend([
            ('back', f"'{prefix}BackRef'"),
            ('cycle', f"'{prefix}Cycle'"),
            ('refs', f"'{prefix}RelayRefs'"),
        ])
    else:
        loop_fields.append(('back', f"'{prefix}BackRef'"))
    loop_methods: list[tuple[str, list[tuple[str, str]], str]] = []
    if has_write:
        loop_methods.append(('enter_write', [], write_context(f'{prefix}LoopWrite')))
    add_protocol(protocol_specs, f'{prefix}Loop', loop_fields, loop_methods)

    add_class(
        class_specs,
        f'{prefix}BackRef',
        [('target', f'LazyRef[{prefix}Loop]'), ('shared', f'LoopShared{slot}')],
    )

    if has_cluster:
        pass
    elif has_relay:
        for handler in range(handler_count):
            add_class(
                class_specs,
                f'{prefix}Relay{handler}',
                [('shared', f'HookShared{alt_slot}'), ('back', f'{prefix}BackRef')],
            )

        relay_params: list[tuple[str, str]] = [('shared', f'LoopShared{slot}')]
        relay_params.extend(
            (f'relay_{handler}', f'LazyRef[{prefix}Relay{handler}]')
            for handler in range(handler_count)
        )
        add_class(class_specs, f'{prefix}RelayRefs', relay_params)
        add_class(
            class_specs,
            f'{prefix}Cycle',
            [
                ('shared', f'LoopShared{slot}'),
                ('back', f'{prefix}BackRef'),
                ('refs', f'{prefix}RelayRefs'),
            ],
        )

    entry_fields: list[tuple[str, str]] = []
    if has_open:
        entry_fields.append(('entry', f'{prefix}EntryValue'))
    entry_methods: list[tuple[str, list[tuple[str, str]], str]] = []
    if has_fork:
        entry_methods.extend(
            (f'fork_{index}', [], f'{prefix}Loop') for index in range(density)
        )
    add_protocol(protocol_specs, f'{prefix}Entry', entry_fields, entry_methods)

    if has_open:
        add_protocol(
            protocol_specs,
            f'{prefix}Root',
            [],
            ((f'open_{index}', [], f'{prefix}Entry') for index in range(density)),
        )

    if has_open:
        for index in range(density):
            add_function(
                function_specs,
                f'{function_prefix}_open_{index}',
                [],
                f'{prefix}OpenState',
                [f"return {{'open_id': {index}}}"],
            )

    if has_fork:
        for index in range(density):
            add_function(
                function_specs,
                f'{function_prefix}_fork_{index}',
                [],
                f'{prefix}ForkState',
                [f"return {{'fork_id': {index}, 'stamp': {index}}}"],
            )

    if has_write:
        add_function(
            function_specs,
            f'{function_prefix}_enter_write',
            [],
            f'{prefix}WriteState',
            ['return {}'],
        )

    if has_cross:
        add_function(
            function_specs,
            f'{function_prefix}_enter_extra',
            [('extra_id', env_int(module))],
            f'{prefix}ExtraState',
            ["return {'extra_id': extra_id}"],
        )

    if has_hooks:
        for hook_index in range(handler_count):
            if has_open:
                for index in range(density):
                    add_function(
                        function_specs,
                        f'{function_prefix}_open_{index}_hook_{hook_index}',
                        [('shared', f'HookShared{slot}')],
                        'None',
                        ['return None'],
                    )
            if has_fork:
                for index in range(density):
                    add_function(
                        function_specs,
                        f'{function_prefix}_fork_{index}_hook_{hook_index}',
                        [('shared', f'HookShared{alt_slot}')],
                        'None',
                        ['return None'],
                    )
            if has_write:
                add_function(
                    function_specs,
                    f'{function_prefix}_enter_write_hook_{hook_index}',
                    [('shared', f'HookShared{slot}')],
                    'None',
                    ['return None'],
                )
            if has_cross:
                add_function(
                    function_specs,
                    f'{function_prefix}_enter_extra_hook_{hook_index}',
                    [
                        ('extra_id', env_int(module)),
                        ('shared', f'HookShared{alt_slot}'),
                    ],
                    'None',
                    ['return None'],
                )

    common_classes = [f'{prefix}BackRef']
    if has_open:
        common_classes.append(f'{prefix}EntryValue')
    if has_fork:
        common_classes.append(f'{prefix}LoopValue')
    if has_cluster:
        common_classes.extend(
            [
                f'{prefix}Executor',
                f'{prefix}RelayRefs',
                f'{prefix}Audit',
                f'{prefix}LoopService',
                f'{prefix}LoopWriteService',
            ]
            + [f'{prefix}Relay{handler}' for handler in range(handler_count)]
        )
    elif has_relay:
        common_classes.extend(
            [f'{prefix}RelayRefs', f'{prefix}Cycle']
            + [f'{prefix}Relay{handler}' for handler in range(handler_count)]
        )
    if has_cross:
        common_classes.extend([
            f'{prefix}BaseService',
            f'{prefix}QueryA',
            f'{prefix}QueryB',
            f'{prefix}ExtraService',
        ])

    methods: list[MethodRegistrationSpec] = []
    if has_open:
        methods.extend(
            MethodRegistrationSpec(
                f'{prefix}Root',
                f'open_{index}',
                f'{function_prefix}_open_{index}',
            )
            for index in range(density)
        )
    if has_fork:
        methods.extend(
            MethodRegistrationSpec(
                f'{prefix}Entry',
                f'fork_{index}',
                f'{function_prefix}_fork_{index}',
            )
            for index in range(density)
        )
    if has_write:
        methods.append(
            MethodRegistrationSpec(
                f'{prefix}Loop',
                'enter_write',
                f'{function_prefix}_enter_write',
            )
        )
    if has_cross:
        methods.append(
            MethodRegistrationSpec(
                f'{prefix}LoopWrite',
                'enter_extra',
                f'{function_prefix}_enter_extra',
            )
        )

    hooks: list[MethodRegistrationSpec] = []
    if has_hooks:
        for hook_index in range(handler_count):
            if has_open:
                hooks.extend(
                    MethodRegistrationSpec(
                        f'{prefix}Root',
                        f'open_{index}',
                        f'{function_prefix}_open_{index}_hook_{hook_index}',
                    )
                    for index in range(density)
                )
            if has_fork:
                hooks.extend(
                    MethodRegistrationSpec(
                        f'{prefix}Entry',
                        f'fork_{index}',
                        f'{function_prefix}_fork_{index}_hook_{hook_index}',
                    )
                    for index in range(density)
                )
            if has_write:
                hooks.append(
                    MethodRegistrationSpec(
                        f'{prefix}Loop',
                        'enter_write',
                        f'{function_prefix}_enter_write_hook_{hook_index}',
                    )
                )
            if has_cross:
                hooks.append(
                    MethodRegistrationSpec(
                        f'{prefix}LoopWrite',
                        'enter_extra',
                        f'{function_prefix}_enter_extra_hook_{hook_index}',
                    )
                )

    return TrackRegistrySpec(
        registry_name,
        tuple(common_classes),
        tuple(methods),
        tuple(hooks),
    )


def build_generated_spec(
    *,
    stage: Stage,
    density: int,
    handler_count: int,
    shared_slots: int,
    tracks: int,
) -> GeneratedSpec:
    class_specs: list[ClassSpec] = []
    typed_dict_specs: list[TypedDictSpec] = []
    protocol_specs: list[ProtocolSpec] = []
    function_specs: list[FunctionSpec] = []
    shared_classes: list[str] = []
    registries: list[TrackRegistrySpec] = []
    slot_count = max(shared_slots, 1)
    multi_track = stage in {'tracks', 'cross'}
    track_count = max(tracks, 1) if multi_track else 1

    for slot in range(slot_count):
        for class_name, params in (
            (f'BaseShared{slot}', []),
            (f'EntryShared{slot}', [('base', f'BaseShared{slot}')]),
            (f'LoopShared{slot}', [('base', f'BaseShared{slot}')]),
            (f'HookShared{slot}', [('base', f'BaseShared{slot}')]),
        ):
            add_class(class_specs, class_name, params)
            shared_classes.append(class_name)

    for track in range(track_count):
        registries.append(
            build_track_spec(
                class_specs=class_specs,
                typed_dict_specs=typed_dict_specs,
                protocol_specs=protocol_specs,
                function_specs=function_specs,
                prefix=f'T{track}' if multi_track else '',
                module=f'm{track}',
                track_index=track,
                track_count=track_count,
                stage='cluster' if stage == 'tracks' else stage,
                density=density,
                handler_count=handler_count,
                slot=track % slot_count,
                alt_slot=(track + 1) % slot_count,
            )
        )

    if multi_track:
        add_protocol(
            protocol_specs,
            'App',
            (
                (f'root_{track}', f"Annotated[T{track}Root, qual('m{track}')]")
                for track in range(track_count)
            ),
            [],
        )

    return GeneratedSpec(
        class_specs=tuple(class_specs),
        typed_dict_specs=tuple(typed_dict_specs),
        protocol_specs=tuple(protocol_specs),
        function_specs=tuple(function_specs),
        shared_classes=tuple(shared_classes),
        registries=tuple(registries),
        multi_track=multi_track,
        track_count=track_count,
    )


def render_source(
    *,
    stage: Stage,
    density: int,
    handler_count: int,
    shared_slots: int,
    tracks: int,
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(TEMPLATE_NAME)
    return template.render(
        spec=build_generated_spec(
            stage=stage,
            density=density,
            handler_count=handler_count,
            shared_slots=shared_slots,
            tracks=tracks,
        )
    )


def build_generated(
    *,
    stage: Stage,
    density: int,
    handler_count: int,
    shared_slots: int,
    tracks: int,
) -> tuple[object, Registry, str, str]:
    source = render_source(
        stage=stage,
        density=density,
        handler_count=handler_count,
        shared_slots=shared_slots,
        tracks=tracks,
    )
    module_name = '_staged_neutral_cycle_' + '_'.join([
        stage.replace('-', '_'),
        str(density),
        str(handler_count),
        str(shared_slots),
        str(tracks),
    ])
    generated = types.ModuleType(module_name)
    sys.modules[module_name] = generated
    namespace = generated.__dict__
    exec(source, namespace)
    target: object
    match stage:
        case 'cycle' | 'cycle-write':
            target = Annotated[cast(type[object], namespace['Loop']), qual('m0')]
        case 'fork':
            target = Annotated[cast(type[object], namespace['Entry']), qual('m0')]
        case 'open' | 'hooks' | 'relay' | 'cluster':
            target = Annotated[cast(type[object], namespace['Root']), qual('m0')]
        case 'cross' | 'tracks':
            target = cast(type[object], namespace['App'])
    return target, cast(Registry, namespace['REGISTRY']), source, module_name


def run_once(
    *,
    stage: Stage,
    density: int,
    handler_count: int,
    shared_slots: int,
    tracks: int,
    dump_generated: bool,
) -> None:
    build_started = perf_counter()
    target, registry, source, module_name = build_generated(
        stage=stage,
        density=density,
        handler_count=handler_count,
        shared_slots=shared_slots,
        tracks=tracks,
    )
    build_ms = (perf_counter() - build_started) * 1000.0

    if dump_generated:
        print(source)

    outcome = 'unknown'
    compile_started = perf_counter()
    try:
        _ = registry.compile(default_rules(), normalize(target))
        outcome = 'ok'
    except Exception as exc:  # noqa: BLE001
        outcome = type(exc).__name__
    finally:
        del sys.modules[module_name]
    compile_ms = (perf_counter() - compile_started) * 1000.0

    print(
        ' '.join([
            'benchmark=staged_neutral_cycle',
            f'stage={stage}',
            f'density={density}',
            f'handlers={handler_count}',
            f'shared_slots={shared_slots}',
            f'tracks={tracks}',
            f'build_ms={build_ms:.3f}',
            f'compile_ms={compile_ms:.3f}',
            f'outcome={outcome}',
        ])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        '--stage',
        choices=[
            'cycle',
            'cycle-write',
            'fork',
            'open',
            'hooks',
            'relay',
            'cluster',
            'cross',
            'tracks',
        ],
        default='relay',
    )
    _ = parser.add_argument('--density', type=int, default=4)
    _ = parser.add_argument('--handlers', type=int, default=4)
    _ = parser.add_argument('--shared-slots', type=int, default=1)
    _ = parser.add_argument('--tracks', type=int, default=4)
    _ = parser.add_argument('--dump-generated', action='store_true')
    args = parser.parse_args()
    parsed = cast(dict[str, object], vars(args))

    run_once(
        stage=cast(Stage, parsed['stage']),
        density=cast(int, parsed['density']),
        handler_count=cast(int, parsed['handlers']),
        shared_slots=cast(int, parsed['shared_slots']),
        tracks=cast(int, parsed['tracks']),
        dump_generated=cast(bool, parsed['dump_generated']),
    )


if __name__ == '__main__':
    main()
