import argparse
import sys
import types
from collections.abc import Callable
from time import perf_counter
from typing import Annotated, Literal, cast

from inlay import Registry, normalize
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


def env_int(module: str) -> str:
    return f"Annotated[int, qual('{module}')]"


def write_context(type_ref: str) -> str:
    return f"Annotated[{type_ref}, qual('write')]"


def emit_class(lines: list[str], name: str, params: list[tuple[str, str]]) -> None:
    lines.append(f'class {name}:')
    if not params:
        lines.append('    def __init__(self) -> None:')
        lines.append('        pass')
        lines.append('')
        return

    signature = ', '.join([
        'self',
        *[f'{name}: {type_ref}' for name, type_ref in params],
    ])
    lines.append(f'    def __init__({signature}) -> None:')
    for param_name, _ in params:
        lines.append(f'        self.{param_name} = {param_name}')
    lines.append('')


def emit_typed_dict(lines: list[str], name: str, fields: list[tuple[str, str]]) -> None:
    lines.append(f'class {name}(TypedDict):')
    if not fields:
        lines.append('    pass')
    else:
        for field_name, type_ref in fields:
            lines.append(f'    {field_name}: {type_ref}')
    lines.append('')


def emit_protocol(
    lines: list[str],
    name: str,
    fields: list[tuple[str, str]],
    methods: list[tuple[str, list[tuple[str, str]], str]],
) -> None:
    lines.append(f'class {name}(Protocol):')
    if not fields and not methods:
        lines.append('    pass')
        lines.append('')
        return
    for field_name, type_ref in fields:
        lines.append(f'    {field_name}: {type_ref}')
    for method_name, params, return_type in methods:
        signature = ', '.join([
            'self',
            *[f'{name}: {type_ref}' for name, type_ref in params],
        ])
        lines.append(f'    def {method_name}({signature}) -> {return_type}: ...')
    lines.append('')


def emit_function(
    lines: list[str],
    name: str,
    params: list[tuple[str, str]],
    return_type: str,
    body: list[str],
) -> None:
    signature = ', '.join(f'{name}: {type_ref}' for name, type_ref in params)
    lines.append(f'def {name}({signature}) -> {return_type}:')
    for line in body:
        lines.append(f'    {line}')
    lines.append('')


def emit_register_class(lines: list[str], registry_var: str, class_name: str) -> None:
    lines.append(
        f'{registry_var} = {registry_var}.register({class_name})({class_name})'
    )


def emit_register_method(
    lines: list[str],
    registry_var: str,
    protocol_name: str,
    method_name: str,
    function_name: str,
) -> None:
    lines.append(
        f'{registry_var} = {registry_var}.register_method('
        + f"{protocol_name}, method_name='{method_name}')({function_name})"
    )


def emit_register_hook(
    lines: list[str],
    registry_var: str,
    protocol_name: str,
    method_name: str,
    function_name: str,
) -> None:
    lines.append(
        f'{registry_var} = {registry_var}.register_method_hook('
        + f"{protocol_name}, method_name='{method_name}')({function_name})"
    )


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


def emit_track(
    lines: list[str],
    *,
    prefix: str,
    module: str,
    track_index: int,
    track_count: int,
    stage: Stage,
    density: int,
    handler_count: int,
    slot: int,
    alt_slot: int,
) -> None:
    level = stage_order(stage)
    function_prefix = prefix.lower() if prefix else 'main'
    registry_var = f'{prefix}_COMMON' if prefix else 'COMMON'
    has_write = level >= 1
    has_fork = level >= 2
    has_open = level >= 3
    has_hooks = level >= 4
    has_relay = level >= 5
    has_cluster = level >= 6
    has_cross = level >= 7

    if has_open:
        emit_typed_dict(lines, f'{prefix}OpenState', [('open_id', env_int(module))])
    if has_fork:
        emit_typed_dict(
            lines,
            f'{prefix}ForkState',
            [('fork_id', env_int(module)), ('stamp', env_int(module))],
        )
    if has_write:
        emit_typed_dict(lines, f'{prefix}WriteState', [])
    if has_cross:
        emit_typed_dict(lines, f'{prefix}ExtraState', [('extra_id', env_int(module))])

    if has_cross:
        emit_class(lines, f'{prefix}BaseService', [('shared', f'BaseShared{slot}')])

    if has_open:
        emit_class(
            lines,
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
        emit_class(lines, f'{prefix}LoopValue', loop_params)

    if has_cluster:
        emit_class(
            lines,
            f'{prefix}Executor',
            [('read_ref', f'LazyRef[{prefix}Loop]'), ('shared', f'LoopShared{slot}')],
        )
        for handler in range(handler_count):
            emit_class(
                lines,
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
        emit_class(lines, f'{prefix}RelayRefs', handler_ref_params)
        emit_class(
            lines,
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
        emit_class(lines, f'{prefix}LoopService', loop_service_params)
        emit_class(
            lines,
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
        emit_class(
            lines,
            f'{prefix}QueryA',
            [
                ('open_id', env_int(module)),
                ('fork_id', env_int(module)),
                ('other_base', sibling_base_refs[0]),
                ('shared', f'HookShared{slot}'),
            ],
        )
        emit_class(
            lines,
            f'{prefix}QueryB',
            [
                ('open_id', env_int(module)),
                ('fork_id', env_int(module)),
                ('other_base', sibling_base_refs[1]),
                ('shared', f'HookShared{alt_slot}'),
            ],
        )
        emit_class(
            lines,
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
        emit_protocol(
            lines,
            f'{prefix}ExtraContext',
            [
                ('extra', f'{prefix}ExtraService'),
                ('query_a', f'{prefix}QueryA'),
                ('query_b', f'{prefix}QueryB'),
            ],
            [],
        )
        emit_protocol(
            lines,
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
        emit_protocol(lines, f'{prefix}LoopWrite', loop_write_fields, [])

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
    emit_protocol(lines, f'{prefix}Loop', loop_fields, loop_methods)

    emit_class(
        lines,
        f'{prefix}BackRef',
        [('target', f'LazyRef[{prefix}Loop]'), ('shared', f'LoopShared{slot}')],
    )

    if has_cluster:
        pass
    elif has_relay:
        for handler in range(handler_count):
            emit_class(
                lines,
                f'{prefix}Relay{handler}',
                [('shared', f'HookShared{alt_slot}'), ('back', f'{prefix}BackRef')],
            )

        relay_params: list[tuple[str, str]] = [('shared', f'LoopShared{slot}')]
        relay_params.extend(
            (f'relay_{handler}', f'LazyRef[{prefix}Relay{handler}]')
            for handler in range(handler_count)
        )
        emit_class(lines, f'{prefix}RelayRefs', relay_params)
        emit_class(
            lines,
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
    emit_protocol(lines, f'{prefix}Entry', entry_fields, entry_methods)

    root_methods: list[tuple[str, list[tuple[str, str]], str]] = []
    if has_open:
        root_methods.extend(
            (f'open_{index}', [], f'{prefix}Entry') for index in range(density)
        )
        emit_protocol(lines, f'{prefix}Root', [], root_methods)

    if has_open:
        for index in range(density):
            emit_function(
                lines,
                f'{function_prefix}_open_{index}',
                [],
                f'{prefix}OpenState',
                [f"return {{'open_id': {index}}}"],
            )

    if has_fork:
        for index in range(density):
            emit_function(
                lines,
                f'{function_prefix}_fork_{index}',
                [],
                f'{prefix}ForkState',
                [f"return {{'fork_id': {index}, 'stamp': {index}}}"],
            )

    if has_write:
        emit_function(
            lines,
            f'{function_prefix}_enter_write',
            [],
            f'{prefix}WriteState',
            ['return {}'],
        )

    if has_cross:
        emit_function(
            lines,
            f'{function_prefix}_enter_extra',
            [('extra_id', env_int(module))],
            f'{prefix}ExtraState',
            ["return {'extra_id': extra_id}"],
        )

    if has_hooks:
        for hook_index in range(handler_count):
            if has_open:
                for index in range(density):
                    emit_function(
                        lines,
                        f'{function_prefix}_open_{index}_hook_{hook_index}',
                        [('shared', f'HookShared{slot}')],
                        'None',
                        ['return None'],
                    )
            if has_fork:
                for index in range(density):
                    emit_function(
                        lines,
                        f'{function_prefix}_fork_{index}_hook_{hook_index}',
                        [('shared', f'HookShared{alt_slot}')],
                        'None',
                        ['return None'],
                    )
            if has_write:
                emit_function(
                    lines,
                    f'{function_prefix}_enter_write_hook_{hook_index}',
                    [('shared', f'HookShared{slot}')],
                    'None',
                    ['return None'],
                )
            if has_cross:
                emit_function(
                    lines,
                    f'{function_prefix}_enter_extra_hook_{hook_index}',
                    [
                        ('extra_id', env_int(module)),
                        ('shared', f'HookShared{alt_slot}'),
                    ],
                    'None',
                    ['return None'],
                )

    lines.append(f'{registry_var} = RegistryBuilder()')
    common_classes = [
        f'{prefix}BackRef',
    ]
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
    for class_name in common_classes:
        emit_register_class(lines, registry_var, class_name)

    if has_open:
        for index in range(density):
            emit_register_method(
                lines,
                registry_var,
                f'{prefix}Root',
                f'open_{index}',
                f'{function_prefix}_open_{index}',
            )

    if has_fork:
        for index in range(density):
            emit_register_method(
                lines,
                registry_var,
                f'{prefix}Entry',
                f'fork_{index}',
                f'{function_prefix}_fork_{index}',
            )

    if has_write:
        emit_register_method(
            lines,
            registry_var,
            f'{prefix}Loop',
            'enter_write',
            f'{function_prefix}_enter_write',
        )

    if has_cross:
        emit_register_method(
            lines,
            registry_var,
            f'{prefix}LoopWrite',
            'enter_extra',
            f'{function_prefix}_enter_extra',
        )

    if has_hooks:
        for hook_index in range(handler_count):
            if has_open:
                for index in range(density):
                    emit_register_hook(
                        lines,
                        registry_var,
                        f'{prefix}Root',
                        f'open_{index}',
                        f'{function_prefix}_open_{index}_hook_{hook_index}',
                    )
            if has_fork:
                for index in range(density):
                    emit_register_hook(
                        lines,
                        registry_var,
                        f'{prefix}Entry',
                        f'fork_{index}',
                        f'{function_prefix}_fork_{index}_hook_{hook_index}',
                    )
            if has_write:
                emit_register_hook(
                    lines,
                    registry_var,
                    f'{prefix}Loop',
                    'enter_write',
                    f'{function_prefix}_enter_write_hook_{hook_index}',
                )
            if has_cross:
                emit_register_hook(
                    lines,
                    registry_var,
                    f'{prefix}LoopWrite',
                    'enter_extra',
                    f'{function_prefix}_enter_extra_hook_{hook_index}',
                )


def generate_source(
    *,
    stage: Stage,
    density: int,
    handler_count: int,
    shared_slots: int,
    tracks: int,
) -> str:
    lines = [
        'from typing import Annotated, Protocol, TypedDict',
        'from inlay import LazyRef, RegistryBuilder, qual',
        '',
    ]
    slot_count = max(shared_slots, 1)
    track_count = max(tracks, 1) if stage in {'tracks', 'cross'} else 1

    for slot in range(slot_count):
        emit_class(lines, f'BaseShared{slot}', [])
        emit_class(lines, f'EntryShared{slot}', [('base', f'BaseShared{slot}')])
        emit_class(lines, f'LoopShared{slot}', [('base', f'BaseShared{slot}')])
        emit_class(lines, f'HookShared{slot}', [('base', f'BaseShared{slot}')])

    for track in range(track_count):
        prefix = f'T{track}' if stage in {'tracks', 'cross'} else ''
        module = f'm{track}'
        emit_track(
            lines,
            prefix=prefix,
            module=module,
            track_index=track,
            track_count=track_count,
            stage='cluster' if stage == 'tracks' else stage,
            density=density,
            handler_count=handler_count,
            slot=track % slot_count,
            alt_slot=(track + 1) % slot_count,
        )

    if stage in {'tracks', 'cross'}:
        app_fields = [
            (f'root_{track}', f"Annotated[T{track}Root, qual('m{track}')]")
            for track in range(track_count)
        ]
        emit_protocol(lines, 'App', app_fields, [])
        lines.append('REGISTRY_BUILDER = RegistryBuilder()')
        for slot in range(slot_count):
            emit_register_class(lines, 'REGISTRY_BUILDER', f'BaseShared{slot}')
            emit_register_class(lines, 'REGISTRY_BUILDER', f'EntryShared{slot}')
            emit_register_class(lines, 'REGISTRY_BUILDER', f'LoopShared{slot}')
            emit_register_class(lines, 'REGISTRY_BUILDER', f'HookShared{slot}')
        for track in range(track_count):
            lines.append(
                'REGISTRY_BUILDER = REGISTRY_BUILDER.include('
                + f"T{track}_COMMON, qualifiers=qual('m{track}'))"
            )
            lines.append(
                'REGISTRY_BUILDER = REGISTRY_BUILDER.include('
                + f"T{track}_COMMON, qualifiers=qual('m{track}') & qual('write'))"
            )
        lines.append('REGISTRY = REGISTRY_BUILDER.build()')
        return '\n'.join(lines) + '\n'

    lines.append('REGISTRY_BUILDER = RegistryBuilder()')
    for slot in range(slot_count):
        emit_register_class(lines, 'REGISTRY_BUILDER', f'BaseShared{slot}')
        emit_register_class(lines, 'REGISTRY_BUILDER', f'EntryShared{slot}')
        emit_register_class(lines, 'REGISTRY_BUILDER', f'LoopShared{slot}')
        emit_register_class(lines, 'REGISTRY_BUILDER', f'HookShared{slot}')
    lines.append(
        "REGISTRY_BUILDER = REGISTRY_BUILDER.include(COMMON, qualifiers=qual('m0'))"
    )
    lines.append(
        'REGISTRY_BUILDER = REGISTRY_BUILDER.include('
        + "COMMON, qualifiers=qual('m0') & qual('write'))"
    )
    lines.append('REGISTRY = REGISTRY_BUILDER.build()')
    return '\n'.join(lines) + '\n'


def build_generated(
    *,
    stage: Stage,
    density: int,
    handler_count: int,
    shared_slots: int,
    tracks: int,
) -> tuple[object, Registry, str, str]:
    source = generate_source(
        stage=stage,
        density=density,
        handler_count=handler_count,
        shared_slots=shared_slots,
        tracks=tracks,
    )
    module_name = f'_staged_neutral_cycle_{stage}_{density}_{handler_count}_{tracks}'
    generated = types.ModuleType(module_name)
    sys.modules[module_name] = generated
    namespace = generated.__dict__
    exec(source, namespace)
    qual_fn = cast(Callable[[str], object], namespace['qual'])
    if stage == 'cycle':
        target: object = Annotated[cast(type[object], namespace['Loop']), qual_fn('m0')]
    elif stage == 'cycle-write':
        target = Annotated[cast(type[object], namespace['Loop']), qual_fn('m0')]
    elif stage == 'fork':
        target = Annotated[cast(type[object], namespace['Entry']), qual_fn('m0')]
    elif stage == 'open':
        target = Annotated[cast(type[object], namespace['Root']), qual_fn('m0')]
    elif stage == 'hooks':
        target = Annotated[cast(type[object], namespace['Root']), qual_fn('m0')]
    elif stage == 'relay':
        target = Annotated[cast(type[object], namespace['Root']), qual_fn('m0')]
    elif stage == 'cluster':
        target = Annotated[cast(type[object], namespace['Root']), qual_fn('m0')]
    elif stage == 'cross':
        target = cast(type[object], namespace['App'])
    else:
        target = cast(type[object], namespace['App'])
    registry = cast(Registry, namespace['REGISTRY'])
    return target, registry, source, module_name


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

    stage = cast(Stage, parsed['stage'])
    density = cast(int, parsed['density'])
    handler_count = cast(int, parsed['handlers'])
    shared_slots = cast(int, parsed['shared_slots'])
    tracks = cast(int, parsed['tracks'])
    dump_generated = cast(bool, parsed['dump_generated'])

    run_once(
        stage=stage,
        density=density,
        handler_count=handler_count,
        shared_slots=shared_slots,
        tracks=tracks,
        dump_generated=dump_generated,
    )


if __name__ == '__main__':
    main()
