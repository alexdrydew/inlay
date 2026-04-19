import argparse
from time import perf_counter
from typing import Literal

from inlay import compile
from inlay.default import default_rules

type Scenario = Literal['portable', 'env-sensitive']

SESSION_MODULES = ('chat',)
ALL_MODULES = SESSION_MODULES + ('story', 'character')


def title(name: str) -> str:
    return ''.join(part.capitalize() for part in name.split('_'))


def emit_class(lines: list[str], name: str, params: list[tuple[str, str]]) -> None:
    lines.append(f'class {name}:')
    if params:
        signature = ', '.join(['self', *[f'{n}: {t}' for n, t in params]])
        lines.append(f'    def __init__({signature}) -> None:')
        for param_name, _ in params:
            lines.append(f'        self.{param_name} = {param_name}')
    else:
        lines.append('    def __init__(self) -> None:')
        lines.append('        pass')
    lines.append('')


def emit_function(
    lines: list[str],
    name: str,
    params: list[tuple[str, str]],
    return_type: str,
    body: list[str],
) -> None:
    signature = ', '.join(f'{n}: {t}' for n, t in params)
    lines.append(f'def {name}({signature}) -> {return_type}:')
    for line in body:
        lines.append(f'    {line}')
    lines.append('')


def emit_typed_dict(
    lines: list[str],
    name: str,
    fields: list[tuple[str, str]],
) -> None:
    lines.append(f'class {name}(TypedDict):')
    if not fields:
        lines.append('    pass')
    else:
        for field_name, field_type in fields:
            lines.append(f'    {field_name}: {field_type}')
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
    for field_name, field_type in fields:
        lines.append(f'    {field_name}: {field_type}')
    for method_name, params, return_type in methods:
        signature = ', '.join(['self', *[f'{n}: {t}' for n, t in params]])
        lines.append(f'    def {method_name}({signature}) -> {return_type}: ...')
    lines.append('')


def emit_register_classes(
    lines: list[str],
    class_names: list[str],
    registry_var: str = 'registry',
) -> None:
    for class_name in class_names:
        lines.append(
            f'    {registry_var} = {registry_var}.register({class_name})({class_name})'
        )


def emit_register_method(
    lines: list[str],
    protocol_name: str,
    method_name: str,
    function_name: str,
    registry_var: str = 'registry',
) -> None:
    lines.append(
        f"    {registry_var} = {registry_var}.register_method({protocol_name}, method_name='{method_name}')({function_name})"
    )


def emit_register_hook(
    lines: list[str],
    protocol_name: str,
    method_name: str,
    function_name: str,
    registry_var: str = 'registry',
) -> None:
    lines.append(
        f"    {registry_var} = {registry_var}.register_method_hook({protocol_name}, method_name='{method_name}')({function_name})"
    )


def env_int(module: str) -> str:
    return f"Annotated[int, qual('{module}')]"


def env_str(module: str) -> str:
    return f"Annotated[str, qual('{module}')]"


def shared_base(index: int) -> str:
    return f'SharedBaseService{index}'


def shared_session(index: int) -> str:
    return f'SharedSessionService{index}'


def shared_branch(index: int) -> str:
    return f'SharedBranchService{index}'


def shared_message(index: int) -> str:
    return f'SharedMessageService{index}'


def base_name(module: str, index: int) -> str:
    return f'{title(module)}BaseService{index}'


def write_name(module: str, index: int) -> str:
    return f'{title(module)}WriteService{index}'


def session_name(module: str, index: int) -> str:
    return f'{title(module)}SessionService{index}'


def branch_name(module: str, index: int) -> str:
    return f'{title(module)}BranchService{index}'


def branch_write_name(module: str, index: int) -> str:
    return f'{title(module)}BranchWriteService{index}'


def message_name(module: str, index: int) -> str:
    return f'{title(module)}MessageService{index}'


def service_params(
    scenario: Scenario,
    module: str,
    kind: Literal['base', 'session', 'branch', 'branch_write', 'message'],
    index: int,
) -> list[tuple[str, str]]:
    params: list[tuple[str, str]] = [('base', base_name(module, index))]
    if kind == 'base':
        return params + [('shared', shared_base(index))]
    if kind == 'session':
        params.append(('shared', shared_session(index)))
        if scenario == 'env-sensitive':
            params.insert(0, ('session_id', env_int(module)))
        return params
    if kind == 'branch':
        params.append(('shared', shared_branch(index)))
        if scenario == 'env-sensitive':
            if module != 'character':
                params.insert(0, ('session_id', env_int(module)))
            params.insert(
                1 if module != 'character' else 0, ('branch_id', env_int(module))
            )
            params.insert(
                2 if module != 'character' else 1, ('vector_clock', env_int(module))
            )
        return params
    if kind == 'branch_write':
        params.append(('shared', shared_branch(index)))
        if scenario == 'env-sensitive':
            if module != 'character':
                params.insert(0, ('session_id', env_int(module)))
            params.insert(
                1 if module != 'character' else 0, ('branch_id', env_int(module))
            )
            params.insert(
                2 if module != 'character' else 1, ('vector_clock', env_int(module))
            )
        return params
    params.append(('shared', shared_message(index)))
    if scenario == 'env-sensitive':
        params = [
            ('message_id', env_int(module)),
            ('channel_name', env_str(module)),
            ('session_id', env_int(module)),
            ('branch_id', env_int(module)),
            ('vector_clock', env_int(module)),
            *params,
        ]
    return params


def generate_source(scenario: Scenario, fanout: int) -> str:
    lines: list[str] = [
        'from typing import Annotated, Protocol, TypedDict',
        'from inlay import LazyRef, RegistryBuilder, qual',
        '',
    ]

    for index in range(fanout):
        emit_class(lines, shared_base(index), [])
        shared_session_params = [('base', shared_base(index))]
        emit_class(lines, shared_session(index), shared_session_params)
        shared_branch_params = [('base', shared_base(index))]
        emit_class(lines, shared_branch(index), shared_branch_params)
        shared_message_params = [('base', shared_base(index))]
        emit_class(lines, shared_message(index), shared_message_params)

    for module in ALL_MODULES:
        prefix = title(module)
        emit_class(lines, f'{prefix}ModuleState', [])
        emit_class(lines, f'{prefix}WriteState', [])
        emit_class(lines, f'{prefix}HandlerState', [])
        for index in range(fanout):
            emit_class(lines, base_name(module, index), [])
            emit_class(
                lines,
                write_name(module, index),
                service_params(scenario, module, 'base', index),
            )
        if module in SESSION_MODULES:
            emit_typed_dict(
                lines, f'{prefix}SessionState', [('session_id', env_int(module))]
            )
            emit_typed_dict(
                lines,
                f'{prefix}BranchState',
                [('branch_id', env_int(module)), ('vector_clock', env_int(module))],
            )
            emit_typed_dict(
                lines,
                f'{prefix}MessageState',
                [
                    ('message_id', env_int(module)),
                    ('channel_name', env_str(module)),
                    ('session_id', env_int(module)),
                    ('branch_id', env_int(module)),
                    ('vector_clock', env_int(module)),
                ],
            )
            for index in range(fanout):
                emit_class(
                    lines,
                    session_name(module, index),
                    service_params(scenario, module, 'session', index),
                )
                emit_class(
                    lines,
                    branch_name(module, index),
                    service_params(scenario, module, 'branch', index),
                )
                emit_class(
                    lines,
                    branch_write_name(module, index),
                    service_params(scenario, module, 'branch_write', index),
                )
                emit_class(
                    lines,
                    message_name(module, index),
                    service_params(scenario, module, 'message', index),
                )
        elif module == 'character':
            emit_typed_dict(
                lines,
                'CharacterBranchState',
                [
                    ('branch_id', env_int('character')),
                    ('vector_clock', env_int('character')),
                ],
            )
            emit_typed_dict(
                lines,
                'CharacterMessageState',
                [
                    ('message_id', env_int('character')),
                    ('channel_name', env_str('character')),
                    ('session_id', env_int('character')),
                    ('branch_id', env_int('character')),
                    ('vector_clock', env_int('character')),
                ],
            )
            for index in range(fanout):
                emit_class(
                    lines,
                    branch_name(module, index),
                    service_params(scenario, module, 'branch', index),
                )
                emit_class(
                    lines,
                    branch_write_name(module, index),
                    service_params(scenario, module, 'branch_write', index),
                )
                emit_class(
                    lines,
                    message_name(module, index),
                    service_params(scenario, module, 'message', index),
                )

    for index in range(fanout):
        ai_game_params = [
            (
                'game_base',
                f"Annotated[{base_name('game', index)}, qual('game') ]".replace(
                    ' ]', ']'
                ),
            ),
            (
                'story_base',
                f"Annotated[{base_name('story', index)}, qual('story') ]".replace(
                    ' ]', ']'
                ),
            ),
            ('shared', shared_branch(index)),
        ]
        ai_story_params = [
            (
                'story_base',
                f"Annotated[{base_name('story', index)}, qual('story') ]".replace(
                    ' ]', ']'
                ),
            ),
            ('shared', shared_branch(index)),
        ]
        ai_character_params = [
            (
                'character_context',
                "Annotated[CharacterContext, qual('character') ]".replace(' ]', ']'),
            ),
            ('shared', shared_branch(index)),
        ]
        game_story_params = [
            (
                'story_base',
                f"Annotated[{base_name('story', index)}, qual('story') ]".replace(
                    ' ]', ']'
                ),
            ),
            ('shared', shared_branch(index)),
        ]
        game_character_params = [
            (
                'character_context',
                "Annotated[CharacterContext, qual('character') ]".replace(' ]', ']'),
            ),
            ('shared', shared_branch(index)),
        ]
        if scenario == 'env-sensitive':
            ai_game_params = [
                ('session_id', env_int('ai')),
                ('branch_id', env_int('ai')),
                *ai_game_params,
            ]
            ai_story_params = [
                ('session_id', env_int('ai')),
                ('branch_id', env_int('ai')),
                *ai_story_params,
            ]
            ai_character_params = [
                ('session_id', env_int('ai')),
                ('branch_id', env_int('ai')),
                *ai_character_params,
            ]
            game_story_params = [
                ('session_id', env_int('game')),
                ('branch_id', env_int('game')),
                *game_story_params,
            ]
            game_character_params = [
                ('session_id', env_int('game')),
                ('branch_id', env_int('game')),
                *game_character_params,
            ]
        emit_class(lines, f'AiGameQueries{index}', ai_game_params)
        emit_class(lines, f'AiStoryQueries{index}', ai_story_params)
        emit_class(lines, f'AiCharacterQueries{index}', ai_character_params)
        emit_class(lines, f'GameStoryQueries{index}', game_story_params)
        emit_class(lines, f'GameCharacterData{index}', game_character_params)
        agent_params = [
            ('base', base_name('ai', index)),
            ('shared', shared_branch(index)),
        ]
        tool_params = [
            ('base', base_name('ai', index)),
            ('shared', shared_branch(index)),
        ]
        if scenario == 'env-sensitive':
            agent_params = [
                ('session_id', env_int('ai')),
                ('branch_id', env_int('ai')),
                ('agent_id', env_int('ai')),
                *agent_params,
            ]
            tool_params = [
                ('session_id', env_int('ai')),
                ('branch_id', env_int('ai')),
                ('agent_id', env_int('ai')),
                ('tool_call_id', env_int('ai')),
                *tool_params,
            ]
        emit_class(lines, f'AiAgentService{index}', agent_params)
        emit_class(lines, f'AiToolService{index}', tool_params)

    emit_typed_dict(lines, 'AiAgentState', [('agent_id', env_int('ai'))])
    emit_typed_dict(lines, 'AiToolState', [('tool_call_id', env_int('ai'))])
    emit_class(lines, 'BenchmarkRootLoopExec', [('ref', 'LazyRef[BenchmarkRoot]')])

    def root_fields(module: str) -> list[tuple[str, str]]:
        return [
            *[(f'base_{index}', base_name(module, index)) for index in range(fanout)],
            *[(f'shared_base_{index}', shared_base(index)) for index in range(fanout)],
        ]

    def session_fields(module: str) -> list[tuple[str, str]]:
        return [
            *[
                (f'session_{index}', session_name(module, index))
                for index in range(fanout)
            ],
            *[
                (f'shared_session_{index}', shared_session(index))
                for index in range(fanout)
            ],
        ]

    def branch_fields(module: str, extras: list[str]) -> list[tuple[str, str]]:
        return [
            *[
                (f'branch_{index}', branch_name(module, index))
                for index in range(fanout)
            ],
            *[
                (f'shared_branch_{index}', shared_branch(index))
                for index in range(fanout)
            ],
            *[(f'extra_{index}', extras[index]) for index in range(fanout)],
        ]

    def message_fields(module: str, extras: list[str]) -> list[tuple[str, str]]:
        return [
            *[
                (f'message_{index}', message_name(module, index))
                for index in range(fanout)
            ],
            *[
                (f'shared_message_{index}', shared_message(index))
                for index in range(fanout)
            ],
            *[(f'extra_{index}', extras[index]) for index in range(fanout)],
        ]

    emit_protocol(
        lines,
        'StoryWriteContext',
        [
            *[
                (f'write_{index}', write_name('story', index))
                for index in range(fanout)
            ],
            *[(f'shared_base_{index}', shared_base(index)) for index in range(fanout)],
        ],
        [],
    )
    emit_protocol(
        lines,
        'StoryContext',
        root_fields('story'),
        [('with_write', [], 'StoryWriteContext')],
    )

    emit_protocol(
        lines,
        'CharacterWriteContext',
        [
            *[
                (f'write_{index}', write_name('character', index))
                for index in range(fanout)
            ],
            *[(f'shared_base_{index}', shared_base(index)) for index in range(fanout)],
        ],
        [],
    )
    emit_protocol(
        lines,
        'CharacterBranchedWriteContext',
        [
            *[
                (f'branch_write_{index}', branch_write_name('character', index))
                for index in range(fanout)
            ],
            *[
                (f'shared_branch_{index}', shared_branch(index))
                for index in range(fanout)
            ],
        ],
        [],
    )
    emit_protocol(
        lines,
        'CharacterBranchedReadContext',
        [
            *[
                (f'branch_{index}', branch_name('character', index))
                for index in range(fanout)
            ],
            *[
                (f'shared_branch_{index}', shared_branch(index))
                for index in range(fanout)
            ],
        ],
        [('with_branched_write', [], 'CharacterBranchedWriteContext')],
    )
    emit_protocol(
        lines,
        'CharacterMessageContext',
        [
            *[
                (f'message_{index}', message_name('character', index))
                for index in range(fanout)
            ],
            *[
                (f'shared_message_{index}', shared_message(index))
                for index in range(fanout)
            ],
        ],
        [('with_handler', [], 'CharacterBranchedWriteContext')],
    )
    emit_protocol(
        lines,
        'CharacterContext',
        root_fields('character'),
        [
            ('with_write', [], 'CharacterWriteContext'),
            (
                'with_branched_session',
                [('branch_id', 'int'), ('vector_clock', 'int')],
                'CharacterBranchedReadContext',
            ),
            (
                'with_message_scope',
                [
                    ('message_id', 'int'),
                    ('channel_name', 'str'),
                    ('session_id', 'int'),
                    ('branch_id', 'int'),
                    ('vector_clock', 'int'),
                ],
                'CharacterMessageContext',
            ),
        ],
    )

    for module in SESSION_MODULES:
        prefix = title(module)
        emit_protocol(
            lines,
            f'{prefix}WriteContext',
            [
                *[
                    (f'write_{index}', write_name(module, index))
                    for index in range(fanout)
                ],
                *[
                    (f'shared_base_{index}', shared_base(index))
                    for index in range(fanout)
                ],
            ],
            [],
        )
        emit_protocol(
            lines,
            f'{prefix}SessionWriteContext',
            [
                *[
                    (f'write_{index}', write_name(module, index))
                    for index in range(fanout)
                ],
                *[
                    (f'shared_session_{index}', shared_session(index))
                    for index in range(fanout)
                ],
            ],
            [],
        )
        session_methods = [
            ('with_write', [], f'{prefix}WriteContext'),
            (
                'with_branched_session',
                [('branch_id', 'int'), ('vector_clock', 'int')],
                f'{prefix}BranchedReadContext',
            ),
        ]
        session_methods.extend(
            (f'branch_{index}', [], f'{prefix}BranchedReadContext')
            for index in range(fanout)
        )
        emit_protocol(
            lines,
            f'{prefix}SessionReadContext',
            session_fields(module),
            session_methods,
        )
        branch_extras = [base_name(module, index) for index in range(fanout)]
        message_extras = [base_name(module, index) for index in range(fanout)]
        if module == 'game':
            branch_extras = [f'GameStoryQueries{index}' for index in range(fanout)]
            message_extras = [f'GameCharacterData{index}' for index in range(fanout)]
        if module == 'ai':
            branch_extras = [f'AiGameQueries{index}' for index in range(fanout)]
            message_extras = [f'AiCharacterQueries{index}' for index in range(fanout)]
        branch_write_methods: list[tuple[str, list[tuple[str, str]], str]] = []
        if module == 'ai':
            branch_write_methods.append((
                'with_agent',
                [('agent_id', 'int')],
                'AiAgentWriteContext',
            ))
        emit_protocol(
            lines,
            f'{prefix}BranchedWriteContext',
            [
                *[
                    (f'branch_write_{index}', branch_write_name(module, index))
                    for index in range(fanout)
                ],
                *[
                    (f'shared_branch_{index}', shared_branch(index))
                    for index in range(fanout)
                ],
                *[(f'extra_{index}', branch_extras[index]) for index in range(fanout)],
            ],
            branch_write_methods,
        )
        emit_protocol(
            lines,
            f'{prefix}BranchedReadContext',
            branch_fields(module, branch_extras),
            [('with_branched_write', [], f'{prefix}BranchedWriteContext')],
        )
        emit_protocol(
            lines,
            f'{prefix}MessageContext',
            message_fields(module, message_extras),
            [('with_handler', [], f'{prefix}BranchedWriteContext')],
        )
        root_methods = [
            ('with_write', [], f'{prefix}WriteContext'),
            ('with_session', [('session_id', 'int')], f'{prefix}SessionReadContext'),
        ]
        root_methods.extend(
            (f'route_{index}', [], f'{prefix}SessionReadContext')
            for index in range(fanout)
        )
        root_methods.append((
            'with_message_scope',
            [
                ('message_id', 'int'),
                ('channel_name', 'str'),
                ('session_id', 'int'),
                ('branch_id', 'int'),
                ('vector_clock', 'int'),
            ],
            f'{prefix}MessageContext',
        ))
        emit_protocol(lines, f'{prefix}Context', root_fields(module), root_methods)

    if 'ai' in SESSION_MODULES:
        emit_protocol(
            lines,
            'AiAgentWriteContext',
            [
                *[
                    (f'agent_{index}', f'AiAgentService{index}')
                    for index in range(fanout)
                ],
                *[
                    (f'shared_branch_{index}', shared_branch(index))
                    for index in range(fanout)
                ],
                *[
                    (f'query_{index}', f'AiStoryQueries{index}')
                    for index in range(fanout)
                ],
            ],
            [('with_tool', [('tool_call_id', 'int')], 'AiToolWriteContext')],
        )
        emit_protocol(
            lines,
            'AiToolWriteContext',
            [
                *[
                    (f'tool_{index}', f'AiToolService{index}')
                    for index in range(fanout)
                ],
                *[
                    (f'shared_branch_{index}', shared_branch(index))
                    for index in range(fanout)
                ],
                *[
                    (f'query_{index}', f'AiCharacterQueries{index}')
                    for index in range(fanout)
                ],
            ],
            [],
        )

    emit_protocol(
        lines,
        'AppContext',
        [],
        [
            (
                f'with_{module}_context',
                [],
                f"Annotated[{title(module)}Context, qual('{module}')]"
                if module != 'character'
                else "Annotated[CharacterContext, qual('character')]".replace(
                    'CharacterContext', 'CharacterContext'
                ),
            )
            for module in ALL_MODULES
        ],
    )
    emit_protocol(
        lines,
        'BenchmarkRoot',
        [('app', 'AppContext'), ('loop_exec', 'BenchmarkRootLoopExec')],
        [],
    )

    for module in ALL_MODULES:
        prefix = title(module)
        emit_function(
            lines,
            f'{module}_with_write',
            [],
            f'{prefix}WriteState',
            [f'return {prefix}WriteState()'],
        )
        if module in SESSION_MODULES:
            emit_function(
                lines,
                f'{module}_with_session',
                [('session_id', env_int(module))],
                f'{prefix}SessionState',
                ["return {'session_id': session_id}"],
            )
            for index in range(fanout):
                emit_function(
                    lines,
                    f'{module}_route_{index}',
                    [],
                    f'{prefix}SessionState',
                    [f"return {{'session_id': {index + 1}}}"],
                )
                emit_function(
                    lines,
                    f'{module}_branch_{index}',
                    [],
                    f'{prefix}BranchState',
                    [
                        f"return {{'branch_id': {index + 1}, 'vector_clock': {index + 1}}}"
                    ],
                )
            emit_function(
                lines,
                f'{module}_with_message_scope',
                [
                    ('message_id', env_int(module)),
                    ('channel_name', env_str(module)),
                    ('session_id', env_int(module)),
                    ('branch_id', env_int(module)),
                    ('vector_clock', env_int(module)),
                ],
                f'{prefix}MessageState',
                [
                    "return {'message_id': message_id, 'channel_name': channel_name, 'session_id': session_id, 'branch_id': branch_id, 'vector_clock': vector_clock}"
                ],
            )
            emit_function(
                lines,
                f'{module}_with_branched_session',
                [('branch_id', env_int(module)), ('vector_clock', env_int(module))],
                f'{prefix}BranchState',
                ["return {'branch_id': branch_id, 'vector_clock': vector_clock}"],
            )
            emit_function(
                lines,
                f'{module}_with_branched_write',
                [],
                f'{prefix}WriteState',
                [f'return {prefix}WriteState()'],
            )
            emit_function(
                lines,
                f'{module}_with_handler',
                [],
                f'{prefix}HandlerState',
                [f'return {prefix}HandlerState()'],
            )
        elif module == 'character':
            emit_function(
                lines,
                'character_with_branched_session',
                [
                    ('branch_id', env_int('character')),
                    ('vector_clock', env_int('character')),
                ],
                'CharacterBranchState',
                ["return {'branch_id': branch_id, 'vector_clock': vector_clock}"],
            )
            emit_function(
                lines,
                'character_with_message_scope',
                [
                    ('message_id', env_int('character')),
                    ('channel_name', env_str('character')),
                    ('session_id', env_int('character')),
                    ('branch_id', env_int('character')),
                    ('vector_clock', env_int('character')),
                ],
                'CharacterMessageState',
                [
                    "return {'message_id': message_id, 'channel_name': channel_name, 'session_id': session_id, 'branch_id': branch_id, 'vector_clock': vector_clock}"
                ],
            )
            emit_function(
                lines,
                'character_with_branched_write',
                [],
                'CharacterWriteState',
                ['return CharacterWriteState()'],
            )
            emit_function(
                lines,
                'character_with_handler',
                [],
                'CharacterHandlerState',
                ['return CharacterHandlerState()'],
            )
        else:
            emit_function(
                lines,
                'story_with_write',
                [],
                'StoryWriteState',
                ['return StoryWriteState()'],
            )
        emit_function(
            lines,
            f'app_with_{module}_context',
            [],
            f'{title(module)}ModuleState',
            [f'return {title(module)}ModuleState()'],
        )

    if 'ai' in SESSION_MODULES:
        emit_function(
            lines,
            'ai_with_agent',
            [('agent_id', env_int('ai'))],
            'AiAgentState',
            ["return {'agent_id': agent_id}"],
        )
        emit_function(
            lines,
            'ai_with_tool',
            [('tool_call_id', env_int('ai'))],
            'AiToolState',
            ["return {'tool_call_id': tool_call_id}"],
        )

    hook_specs: list[tuple[str, str, str, list[tuple[str, str]]]] = []
    for module in SESSION_MODULES:
        prefix = title(module)
        hook_specs.append((
            f'{prefix}Context',
            'with_session',
            f'{module}_with_session_hook',
            [('session_id', env_int(module)), ('shared', shared_session(0))],
        ))
        hook_specs.extend(
            (
                f'{prefix}Context',
                f'route_{index}',
                f'{module}_route_{index}_hook',
                [('shared', shared_session(0))],
            )
            for index in range(fanout)
        )
        hook_specs.append((
            f'{prefix}SessionReadContext',
            'with_branched_session',
            f'{module}_with_branched_session_hook',
            [
                ('branch_id', env_int(module)),
                ('vector_clock', env_int(module)),
                ('shared', shared_branch(0)),
            ],
        ))
        hook_specs.extend(
            (
                f'{prefix}SessionReadContext',
                f'branch_{index}',
                f'{module}_branch_{index}_hook',
                [('shared', shared_branch(0))],
            )
            for index in range(fanout)
        )
        hook_specs.append((
            f'{prefix}Context',
            'with_message_scope',
            f'{module}_with_message_scope_hook',
            [
                ('message_id', env_int(module)),
                ('session_id', env_int(module)),
                ('branch_id', env_int(module)),
                ('shared', shared_message(0)),
            ],
        ))
        hook_specs.append((
            f'{prefix}MessageContext',
            'with_handler',
            f'{module}_with_handler_hook',
            [('shared', shared_branch(0))],
        ))
    hook_specs.extend([
        (
            'CharacterContext',
            'with_branched_session',
            'character_with_branched_session_hook',
            [
                ('branch_id', env_int('character')),
                ('vector_clock', env_int('character')),
                ('shared', shared_branch(0)),
            ],
        ),
        (
            'CharacterContext',
            'with_message_scope',
            'character_with_message_scope_hook',
            [
                ('message_id', env_int('character')),
                ('session_id', env_int('character')),
                ('branch_id', env_int('character')),
                ('shared', shared_message(0)),
            ],
        ),
        (
            'CharacterMessageContext',
            'with_handler',
            'character_with_handler_hook',
            [('shared', shared_branch(0))],
        ),
        (
            'StoryContext',
            'with_write',
            'story_with_write_hook',
            [('shared', shared_base(0))],
        ),
    ])
    if 'ai' in SESSION_MODULES:
        hook_specs.extend([
            (
                'AiBranchedWriteContext',
                'with_agent',
                'ai_with_agent_hook',
                [('agent_id', env_int('ai')), ('shared', shared_branch(0))],
            ),
            (
                'AiAgentWriteContext',
                'with_tool',
                'ai_with_tool_hook',
                [('tool_call_id', env_int('ai')), ('shared', shared_branch(0))],
            ),
        ])
    for protocol_name, method_name, hook_name, params in hook_specs:
        emit_function(lines, hook_name, params, 'None', ['return None'])

    lines.append('def build_registry() -> RegistryBuilder:')
    lines.append('    registry = RegistryBuilder()')
    lines.append(
        '    registry = registry.register(BenchmarkRootLoopExec)(BenchmarkRootLoopExec)'
    )
    for module in ALL_MODULES:
        emit_register_method(
            lines, 'AppContext', f'with_{module}_context', f'app_with_{module}_context'
        )
    lines.append('')
    lines.append('    shared_registry = RegistryBuilder()')
    emit_register_classes(
        lines,
        [
            *(shared_base(index) for index in range(fanout)),
            *(shared_session(index) for index in range(fanout)),
            *(shared_branch(index) for index in range(fanout)),
            *(shared_message(index) for index in range(fanout)),
        ],
        'shared_registry',
    )
    lines.append('')
    for module in ALL_MODULES:
        prefix = title(module)
        lines.append(f'    {module}_registry = RegistryBuilder()')
        emit_register_classes(
            lines,
            [
                *(base_name(module, index) for index in range(fanout)),
                *(write_name(module, index) for index in range(fanout)),
            ],
            f'{module}_registry',
        )
        emit_register_method(
            lines,
            f'{prefix}Context' if module != 'character' else 'CharacterContext',
            'with_write',
            f'{module}_with_write',
            f'{module}_registry',
        )
        if module in SESSION_MODULES:
            emit_register_classes(
                lines,
                [
                    f'{prefix}SessionState',
                    f'{prefix}BranchState',
                    f'{prefix}MessageState',
                ],
                f'{module}_registry',
            )
            emit_register_classes(
                lines,
                [
                    *(session_name(module, index) for index in range(fanout)),
                    *(branch_name(module, index) for index in range(fanout)),
                    *(branch_write_name(module, index) for index in range(fanout)),
                    *(message_name(module, index) for index in range(fanout)),
                ],
                f'{module}_registry',
            )
            emit_register_method(
                lines,
                f'{prefix}Context',
                'with_session',
                f'{module}_with_session',
                f'{module}_registry',
            )
            for index in range(fanout):
                emit_register_method(
                    lines,
                    f'{prefix}Context',
                    f'route_{index}',
                    f'{module}_route_{index}',
                    f'{module}_registry',
                )
            emit_register_method(
                lines,
                f'{prefix}Context',
                'with_message_scope',
                f'{module}_with_message_scope',
                f'{module}_registry',
            )
            emit_register_method(
                lines,
                f'{prefix}SessionReadContext',
                'with_write',
                f'{module}_with_write',
                f'{module}_registry',
            )
            emit_register_method(
                lines,
                f'{prefix}SessionReadContext',
                'with_branched_session',
                f'{module}_with_branched_session',
                f'{module}_registry',
            )
            for index in range(fanout):
                emit_register_method(
                    lines,
                    f'{prefix}SessionReadContext',
                    f'branch_{index}',
                    f'{module}_branch_{index}',
                    f'{module}_registry',
                )
            emit_register_method(
                lines,
                f'{prefix}BranchedReadContext',
                'with_branched_write',
                f'{module}_with_branched_write',
                f'{module}_registry',
            )
            emit_register_method(
                lines,
                f'{prefix}MessageContext',
                'with_handler',
                f'{module}_with_handler',
                f'{module}_registry',
            )
        elif module == 'character':
            emit_register_classes(
                lines,
                ['CharacterBranchState', 'CharacterMessageState'],
                f'{module}_registry',
            )
            emit_register_classes(
                lines,
                [
                    *(branch_name(module, index) for index in range(fanout)),
                    *(branch_write_name(module, index) for index in range(fanout)),
                    *(message_name(module, index) for index in range(fanout)),
                ],
                f'{module}_registry',
            )
            emit_register_method(
                lines,
                'CharacterContext',
                'with_branched_session',
                'character_with_branched_session',
                f'{module}_registry',
            )
            emit_register_method(
                lines,
                'CharacterContext',
                'with_message_scope',
                'character_with_message_scope',
                f'{module}_registry',
            )
            emit_register_method(
                lines,
                'CharacterBranchedReadContext',
                'with_branched_write',
                'character_with_branched_write',
                f'{module}_registry',
            )
            emit_register_method(
                lines,
                'CharacterMessageContext',
                'with_handler',
                'character_with_handler',
                f'{module}_registry',
            )
        else:
            emit_register_method(
                lines,
                'StoryContext',
                'with_write',
                'story_with_write',
                f'{module}_registry',
            )
        if module == 'game':
            emit_register_classes(
                lines,
                [
                    *(f'GameStoryQueries{index}' for index in range(fanout)),
                    *(f'GameCharacterData{index}' for index in range(fanout)),
                ],
                f'{module}_registry',
            )
        if module == 'ai' and 'ai' in SESSION_MODULES:
            emit_register_classes(
                lines,
                [
                    'AiAgentState',
                    'AiToolState',
                    *(f'AiGameQueries{index}' for index in range(fanout)),
                    *(f'AiStoryQueries{index}' for index in range(fanout)),
                    *(f'AiCharacterQueries{index}' for index in range(fanout)),
                    *(f'AiAgentService{index}' for index in range(fanout)),
                    *(f'AiToolService{index}' for index in range(fanout)),
                ],
                f'{module}_registry',
            )
            emit_register_method(
                lines,
                'AiBranchedWriteContext',
                'with_agent',
                'ai_with_agent',
                f'{module}_registry',
            )
            emit_register_method(
                lines,
                'AiAgentWriteContext',
                'with_tool',
                'ai_with_tool',
                f'{module}_registry',
            )
        lines.append(
            f"    registry = registry.include(shared_registry, {module}_registry, qualifiers=qual('{module}'))"
        )
        lines.append('')
    for protocol_name, method_name, hook_name, _ in hook_specs:
        emit_register_hook(lines, protocol_name, method_name, hook_name)
    lines.append('    return registry')
    lines.append('')
    lines.append('REGISTRY = build_registry().build()')

    return '\n'.join(lines)


def build_generated(scenario: Scenario, fanout: int) -> tuple[type[object], object]:
    namespace: dict[str, object] = {}
    exec(generate_source(scenario, fanout), namespace)
    return namespace['BenchmarkRoot'], namespace['REGISTRY']


def run_once(*, scenario: Scenario, fanout: int, invoke: bool) -> None:
    generation_started = perf_counter()
    root_type, registry = build_generated(scenario, fanout)
    generation_elapsed = perf_counter() - generation_started

    compile_started = perf_counter()
    root = compile(root_type, registry, default_rules())
    compile_elapsed = perf_counter() - compile_started

    invoke_elapsed = None
    if invoke:
        invoke_started = perf_counter()
        app = root.app
        chat = app.with_chat_context()
        chat_session = chat.with_session(1)
        _ = chat_session.session_0
        chat_branch = chat_session.with_branched_session(1, 1)
        _ = chat_branch.with_branched_write().branch_write_0

        character = app.with_character_context()
        _ = character.with_branched_session(1, 1).branch_0

        story = app.with_story_context()
        _ = story.with_write().write_0
        invoke_elapsed = perf_counter() - invoke_started

    print(
        ' '.join([
            'benchmark=appish_codegen',
            f'scenario={scenario}',
            f'fanout={fanout}',
            f'generation={generation_elapsed:.4f}s',
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
        default='env-sensitive',
    )
    parser.add_argument('--fanout', type=int, default=2)
    parser.add_argument('--invoke', action='store_true')
    args = parser.parse_args()

    scenarios: list[Scenario]
    if args.scenario == 'all':
        scenarios = ['portable', 'env-sensitive']
    else:
        scenarios = [args.scenario]

    for scenario in scenarios:
        run_once(scenario=scenario, fanout=args.fanout, invoke=args.invoke)


if __name__ == '__main__':
    main()
