import argparse
import types
from time import perf_counter
from typing import Annotated, Literal, Protocol, final

from inlay import LazyRef, RegistryBuilder, compile, qual
from inlay.default import default_rules

type Scenario = Literal['portable', 'env-sensitive']

ROOT_METHODS = ('chat', 'game', 'ai', 'story', 'character')


def export[T](name: str, value: T) -> T:
    globals()[name] = value
    return value


def title(name: str) -> str:
    return ''.join(part.capitalize() for part in name.split('_'))


def make_class(
    name: str,
    params: list[tuple[str, object | str]],
) -> type[object]:
    param_names = [param_name for param_name, _ in params]

    def __init__(self, *args: object, **kwargs: object) -> None:
        values = dict(zip(param_names, args, strict=False))
        values.update(kwargs)
        for param_name in param_names:
            setattr(self, param_name, values.get(param_name))

    __init__.__annotations__ = {
        param_name: annotation for param_name, annotation in params
    } | {'return': None}
    namespace = {
        '__module__': __name__,
        '__init__': __init__,
    }
    return export(
        name,
        types.new_class(
            name, (), {}, lambda ns, namespace=namespace: ns.update(namespace)
        ),
    )


def make_protocol_method(
    owner: str,
    method_name: str,
    return_type: object,
    params: list[tuple[str, object]],
):
    def method(*args: object, **kwargs: object) -> object: ...

    method.__name__ = method_name
    method.__qualname__ = f'{owner}.{method_name}'
    method.__annotations__ = {
        param_name: annotation for param_name, annotation in params
    } | {'return': return_type}
    return method


def make_transition_impl(
    owner: str,
    method_name: str,
    state_type: type[object],
    params: list[tuple[str, object]],
):
    param_names = [param_name for param_name, _ in params]

    def implementation(*args: object, **kwargs: object) -> object:
        values = dict(zip(param_names, args[1:], strict=False))
        values.update(kwargs)
        return state_type(**values)

    implementation.__name__ = f'{owner}_{method_name}_impl'
    implementation.__qualname__ = implementation.__name__
    implementation.__annotations__ = {
        param_name: annotation for param_name, annotation in params
    } | {'return': state_type}
    return implementation


def make_bound_transition_class(
    name: str,
    method_name: str,
    state_type: type[object],
    ctor_params: list[tuple[str, object]],
) -> type[object]:
    param_names = [param_name for param_name, _ in ctor_params]

    def __init__(self, *args: object, **kwargs: object) -> None:
        values = dict(zip(param_names, args, strict=False))
        values.update(kwargs)
        for param_name in param_names:
            setattr(self, param_name, values.get(param_name))

    __init__.__annotations__ = {
        param_name: annotation for param_name, annotation in ctor_params
    } | {'return': None}

    def method(self) -> object:
        return state_type()

    method.__name__ = method_name
    method.__qualname__ = f'{name}.{method_name}'
    method.__annotations__ = {'return': state_type}

    return export(
        name,
        types.new_class(
            name,
            (),
            {},
            lambda ns: ns.update({
                '__module__': __name__,
                '__init__': __init__,
                method_name: method,
            }),
        ),
    )


def make_fixed_transition_impl(
    owner: str,
    method_name: str,
    state_type: type[object],
    values: dict[str, object],
):
    def implementation(*args: object, **kwargs: object) -> object:
        return state_type(**values)

    implementation.__name__ = f'{owner}_{method_name}_impl'
    implementation.__qualname__ = implementation.__name__
    implementation.__annotations__ = {'return': state_type}
    return implementation


def make_hook(
    owner: str,
    method_name: str,
    params: list[tuple[str, object]],
):
    def hook(*args: object, **kwargs: object) -> None:
        return None

    hook.__name__ = f'{owner}_{method_name}_hook'
    hook.__qualname__ = hook.__name__
    hook.__annotations__ = {
        param_name: annotation for param_name, annotation in params
    } | {'return': None}
    return hook


def make_protocol(
    name: str,
    fields: dict[str, object],
    methods: list[object],
) -> type[Protocol]:
    namespace: dict[str, object] = {
        '__module__': __name__,
        '__annotations__': fields,
    }
    for method in methods:
        namespace[method.__name__] = method
    return export(
        name,
        types.new_class(
            name, (Protocol,), {}, lambda ns, namespace=namespace: ns.update(namespace)
        ),
    )


def session_leaf_deps(
    module: str,
    base_type: type[object],
    scenario: Scenario,
) -> list[tuple[str, object]]:
    if scenario == 'portable':
        return [('base', base_type)]
    return [
        ('session_id', Annotated[int, qual(module)]),
        ('base', base_type),
    ]


def branch_leaf_deps(
    module: str,
    base_type: type[object],
    scenario: Scenario,
    *,
    include_session: bool,
) -> list[tuple[str, object]]:
    if scenario == 'portable':
        return [('base', base_type)]
    deps: list[tuple[str, object]] = []
    if include_session:
        deps.append(('session_id', Annotated[int, qual(module)]))
    deps.extend([
        ('branch_id', Annotated[int, qual(module)]),
        ('vector_clock', Annotated[int, qual(module)]),
        ('base', base_type),
    ])
    return deps


def message_leaf_deps(
    module: str,
    base_type: type[object],
    scenario: Scenario,
) -> list[tuple[str, object]]:
    if scenario == 'portable':
        return [('base', base_type)]
    return [
        ('message_id', Annotated[int, qual(module)]),
        ('channel_name', Annotated[str, qual(module)]),
        ('session_id', Annotated[int, qual(module)]),
        ('branch_id', Annotated[int, qual(module)]),
        ('vector_clock', Annotated[int, qual(module)]),
        ('base', base_type),
    ]


def branch_write_leaf_deps(
    module: str,
    base_type: type[object],
    scenario: Scenario,
    *,
    include_session: bool,
) -> list[tuple[str, object]]:
    deps = branch_leaf_deps(
        module,
        base_type,
        scenario,
        include_session=include_session,
    )
    return [('base_write', base_type)] if scenario == 'portable' else deps


def agent_leaf_deps(
    base_type: type[object],
    scenario: Scenario,
) -> list[tuple[str, object]]:
    if scenario == 'portable':
        return [('base', base_type)]
    return [
        ('session_id', Annotated[int, qual('ai')]),
        ('branch_id', Annotated[int, qual('ai')]),
        ('vector_clock', Annotated[int, qual('ai')]),
        ('agent_id', Annotated[int, qual('ai')]),
        ('base', base_type),
    ]


def tool_leaf_deps(
    base_type: type[object],
    scenario: Scenario,
) -> list[tuple[str, object]]:
    if scenario == 'portable':
        return [('base', base_type)]
    return [
        ('session_id', Annotated[int, qual('ai')]),
        ('branch_id', Annotated[int, qual('ai')]),
        ('vector_clock', Annotated[int, qual('ai')]),
        ('agent_id', Annotated[int, qual('ai')]),
        ('tool_call_id', Annotated[int, qual('ai')]),
        ('base', base_type),
    ]


def shared_session_deps(
    base_type: type[object],
    scenario: Scenario,
) -> list[tuple[str, object]]:
    if scenario == 'portable':
        return [('base', base_type)]
    return [('session_id', int), ('base', base_type)]


def shared_branch_deps(
    base_type: type[object],
    scenario: Scenario,
) -> list[tuple[str, object]]:
    if scenario == 'portable':
        return [('base', base_type)]
    return [
        ('branch_id', int),
        ('vector_clock', int),
        ('base', base_type),
    ]


def shared_message_deps(
    base_type: type[object],
    scenario: Scenario,
) -> list[tuple[str, object]]:
    if scenario == 'portable':
        return [('base', base_type)]
    return [
        ('message_id', int),
        ('channel_name', str),
        ('session_id', int),
        ('branch_id', int),
        ('vector_clock', int),
        ('base', base_type),
    ]


def shared_fields(
    families: dict[str, dict[str, object]],
    key: str,
    fanout: int,
) -> dict[str, object]:
    return {
        f'shared_{key}_{index}': families['shared'][key][index]  # type: ignore[index]
        for index in range(fanout)
    }


def replace_protocol_field(
    protocol: type[Protocol],
    field_name: str,
    field_type: object,
) -> None:
    protocol.__annotations__[field_name] = field_type


def make_self_loop_class(name: str, target_type: object) -> type[object]:
    return make_class(name, [('ref', LazyRef[target_type])])


def build_leaf_families(
    scenario: Scenario,
    fanout: int,
) -> dict[str, dict[str, object]]:
    shared_base = [
        make_class(f'SharedBaseService{index}', []) for index in range(fanout)
    ]
    families: dict[str, dict[str, object]] = {
        'shared': {
            'base': shared_base,
            'session': [
                make_class(
                    f'SharedSessionService{index}',
                    shared_session_deps(shared_base[index], scenario),
                )
                for index in range(fanout)
            ],
            'branch': [
                make_class(
                    f'SharedBranchService{index}',
                    shared_branch_deps(shared_base[index], scenario),
                )
                for index in range(fanout)
            ],
            'message': [
                make_class(
                    f'SharedMessageService{index}',
                    shared_message_deps(shared_base[index], scenario),
                )
                for index in range(fanout)
            ],
        }
    }

    for module in ROOT_METHODS:
        prefix = title(module)
        families[module] = {
            'module_state': make_class(f'{prefix}ModuleState', []),
            'write_state': make_class(f'{prefix}WriteState', []),
            'branch_write_state': make_class(f'{prefix}BranchedWriteState', []),
            'handler_state': make_class(f'{prefix}HandlerState', []),
            'base': [
                make_class(f'{prefix}BaseService{index}', []) for index in range(fanout)
            ],
            'write': [
                make_class(
                    f'{prefix}WriteService{index}',
                    [('base', families[module]['base'][index])]
                    if module in families
                    else [],
                )
                for index in range(fanout)
            ],
        }

    for module in ('chat', 'game', 'ai'):
        prefix = title(module)
        base_types: list[type[object]] = families[module]['base']  # type: ignore[assignment]
        families[module].update({
            'session_state': make_class(
                f'{prefix}SessionState',
                [('session_id', int)],
            ),
            'branch_state': make_class(
                f'{prefix}BranchState',
                [('branch_id', int), ('vector_clock', int)],
            ),
            'message_state': make_class(
                f'{prefix}MessageState',
                [
                    ('message_id', int),
                    ('channel_name', str),
                    ('session_id', int),
                    ('branch_id', int),
                    ('vector_clock', int),
                ],
            ),
            'session': [
                make_class(
                    f'{prefix}SessionService{index}',
                    session_leaf_deps(module, base_types[index], scenario),
                )
                for index in range(fanout)
            ],
            'branch': [
                make_class(
                    f'{prefix}BranchService{index}',
                    branch_leaf_deps(
                        module,
                        base_types[index],
                        scenario,
                        include_session=True,
                    ),
                )
                for index in range(fanout)
            ],
            'branch_write': [
                make_class(
                    f'{prefix}BranchWriteService{index}',
                    branch_write_leaf_deps(
                        module,
                        base_types[index],
                        scenario,
                        include_session=True,
                    ),
                )
                for index in range(fanout)
            ],
            'message': [
                make_class(
                    f'{prefix}MessageService{index}',
                    message_leaf_deps(module, base_types[index], scenario),
                )
                for index in range(fanout)
            ],
        })

    character_base: list[type[object]] = families['character']['base']  # type: ignore[assignment]
    families['character'].update({
        'branch_state': make_class(
            'CharacterBranchState',
            [('branch_id', int), ('vector_clock', int)],
        ),
        'message_state': make_class(
            'CharacterMessageState',
            [
                ('message_id', int),
                ('channel_name', str),
                ('session_id', int),
                ('branch_id', int),
                ('vector_clock', int),
            ],
        ),
        'branch': [
            make_class(
                f'CharacterBranchService{index}',
                branch_leaf_deps(
                    'character',
                    character_base[index],
                    scenario,
                    include_session=False,
                ),
            )
            for index in range(fanout)
        ],
        'branch_write': [
            make_class(
                f'CharacterBranchWriteService{index}',
                branch_write_leaf_deps(
                    'character',
                    character_base[index],
                    scenario,
                    include_session=False,
                ),
            )
            for index in range(fanout)
        ],
        'message': [
            make_class(
                f'CharacterMessageService{index}',
                message_leaf_deps('character', character_base[index], scenario),
            )
            for index in range(fanout)
        ],
    })

    ai_base: list[type[object]] = families['ai']['base']  # type: ignore[assignment]
    families['ai'].update({
        'agent_state': make_class('AiAgentState', [('agent_id', int)]),
        'tool_state': make_class('AiToolState', [('tool_call_id', int)]),
        'agent': [
            make_class(
                f'AiAgentService{index}',
                agent_leaf_deps(ai_base[index], scenario),
            )
            for index in range(fanout)
        ],
        'tool': [
            make_class(
                f'AiToolService{index}',
                tool_leaf_deps(ai_base[index], scenario),
            )
            for index in range(fanout)
        ],
    })

    return families


def build_appish_contexts(
    scenario: Scenario,
    fanout: int,
) -> tuple[type[Protocol], dict[str, dict[str, object]]]:
    families = build_leaf_families(scenario, fanout)

    context_names = {module: f'{title(module)}Context' for module in ROOT_METHODS}

    for module in ROOT_METHODS:
        prefix = title(module)
        families[module]['context_executor'] = make_class(
            f'{prefix}ContextExecutor',
            [('ctx_ref', f'LazyRef[{context_names[module]}]')],
        )

    story_write = make_protocol(
        'StoryWriteContext',
        {
            **{
                f'write_{index}': families['story']['write'][index]  # type: ignore[index]
                for index in range(fanout)
            },
            **shared_fields(families, 'base', fanout),
        },
        [],
    )
    story_context = make_protocol(
        'StoryContext',
        {
            **{
                f'base_{index}': families['story']['base'][index]  # type: ignore[index]
                for index in range(fanout)
            },
            **shared_fields(families, 'base', fanout),
            'loop_exec': object,
            'execute_in_transaction': families['story']['context_executor'],
        },
        [
            make_protocol_method('StoryContext', 'with_write', story_write, []),
        ],
    )

    character_branch_write = make_protocol(
        'CharacterBranchedWriteContext',
        {
            **{
                f'branch_write_{index}': families['character']['branch_write'][index]  # type: ignore[index]
                for index in range(fanout)
            },
            **shared_fields(families, 'branch', fanout),
        },
        [],
    )
    families['character']['branched_executor'] = make_class(
        'CharacterBranchedExecutor',
        [('ctx_ref', 'LazyRef[CharacterBranchedReadContext]')],
    )
    character_branch_read = make_protocol(
        'CharacterBranchedReadContext',
        {
            **{
                f'branch_{index}': families['character']['branch'][index]  # type: ignore[index]
                for index in range(fanout)
            },
            **shared_fields(families, 'branch', fanout),
            'loop_exec': object,
            'execute_in_branched_transaction': families['character'][
                'branched_executor'
            ],
        },
        [
            make_protocol_method(
                'CharacterBranchedReadContext',
                'with_branched_write',
                character_branch_write,
                [],
            )
        ],
    )
    character_write = make_protocol(
        'CharacterWriteContext',
        {
            **{
                f'write_{index}': families['character']['write'][index]  # type: ignore[index]
                for index in range(fanout)
            },
            **shared_fields(families, 'base', fanout),
        },
        [],
    )
    character_message = make_protocol(
        'CharacterMessageContext',
        {
            **{
                f'message_{index}': families['character']['message'][index]  # type: ignore[index]
                for index in range(fanout)
            },
            **shared_fields(families, 'message', fanout),
        },
        [
            make_protocol_method(
                'CharacterMessageContext',
                'with_handler',
                character_branch_write,
                [],
            )
        ],
    )
    character_context = make_protocol(
        'CharacterContext',
        {
            **{
                f'base_{index}': families['character']['base'][index]  # type: ignore[index]
                for index in range(fanout)
            },
            **shared_fields(families, 'base', fanout),
            'loop_exec': object,
            'execute_in_transaction': families['character']['context_executor'],
        },
        [
            make_protocol_method('CharacterContext', 'with_write', character_write, []),
            make_protocol_method(
                'CharacterContext',
                'with_branched_session',
                character_branch_read,
                [('branch_id', int), ('vector_clock', int)],
            ),
            make_protocol_method(
                'CharacterContext',
                'with_message_scope',
                character_message,
                [
                    ('message_id', int),
                    ('channel_name', str),
                    ('session_id', int),
                    ('branch_id', int),
                    ('vector_clock', int),
                ],
            ),
        ],
    )

    families['ai']['ai_game_queries'] = []
    families['ai']['ai_story_queries'] = []
    families['ai']['ai_character_queries'] = []
    families['game']['game_story_queries'] = []
    families['game']['game_character_data'] = []

    for index in range(fanout):
        ai_branch_type = families['ai']['branch'][index]  # type: ignore[index]
        game_branch_type = families['game']['branch'][index]  # type: ignore[index]
        ai_game_deps: list[tuple[str, object | str]]
        ai_story_deps: list[tuple[str, object | str]]
        ai_character_deps: list[tuple[str, object | str]]
        game_story_deps: list[tuple[str, object | str]]
        game_character_deps: list[tuple[str, object | str]]

        if scenario == 'portable':
            ai_game_deps = [
                (
                    'game_base',
                    f'Annotated[{title("game")}BaseService{index}, qual("game")]',
                ),
                (
                    'story_base',
                    f'Annotated[{title("story")}BaseService{index}, qual("story")]',
                ),
            ]
            ai_story_deps = [
                (
                    'story_base',
                    f'Annotated[{title("story")}BaseService{index}, qual("story")]',
                ),
            ]
            ai_character_deps = [
                ('character_context', 'Annotated[CharacterContext, qual("character")]'),
            ]
            game_story_deps = [
                (
                    'story_base',
                    f'Annotated[{title("story")}BaseService{index}, qual("story")]',
                ),
            ]
            game_character_deps = [
                ('character_context', 'Annotated[CharacterContext, qual("character")]'),
            ]
        else:
            ai_game_deps = [
                ('ai_branch', ai_branch_type),
                (
                    'game_base',
                    f'Annotated[{title("game")}BaseService{index}, qual("game")]',
                ),
                (
                    'story_base',
                    f'Annotated[{title("story")}BaseService{index}, qual("story")]',
                ),
            ]
            ai_story_deps = [
                ('ai_branch', ai_branch_type),
                (
                    'story_base',
                    f'Annotated[{title("story")}BaseService{index}, qual("story")]',
                ),
            ]
            ai_character_deps = [
                ('ai_branch', ai_branch_type),
                ('character_context', 'Annotated[CharacterContext, qual("character")]'),
            ]
            game_story_deps = [
                ('game_branch', game_branch_type),
                (
                    'story_base',
                    f'Annotated[{title("story")}BaseService{index}, qual("story")]',
                ),
            ]
            game_character_deps = [
                ('game_branch', game_branch_type),
                ('character_context', 'Annotated[CharacterContext, qual("character")]'),
            ]

        families['ai']['ai_game_queries'].append(
            make_class(f'AiGameQueries{index}', ai_game_deps)
        )
        families['ai']['ai_story_queries'].append(
            make_class(f'AiStoryQueries{index}', ai_story_deps)
        )
        families['ai']['ai_character_queries'].append(
            make_class(f'AiCharacterQueries{index}', ai_character_deps)
        )
        families['game']['game_story_queries'].append(
            make_class(f'GameStoryQueries{index}', game_story_deps)
        )
        families['game']['game_character_data'].append(
            make_class(f'GameCharacterData{index}', game_character_deps)
        )

    def session_module_protocols(
        module: str,
        *,
        extra_branch_fields: list[type[object]],
        extra_message_fields: list[type[object]],
        include_agent: bool,
    ) -> None:
        prefix = title(module)
        route_names = [f'route_{index}' for index in range(fanout)]
        branch_route_names = [f'branch_{index}' for index in range(fanout)]
        write_context = make_protocol(
            f'{prefix}WriteContext',
            {
                f'write_{index}': families[module]['write'][index]  # type: ignore[index]
                for index in range(fanout)
            }
            | shared_fields(families, 'base', fanout),
            [],
        )
        session_write = make_protocol(
            f'{prefix}SessionWriteContext',
            {
                f'write_{index}': families[module]['write'][index]  # type: ignore[index]
                for index in range(fanout)
            }
            | shared_fields(families, 'session', fanout),
            [
                make_protocol_method(
                    f'{prefix}SessionWriteContext',
                    'with_branched_session',
                    f'{prefix}BranchedWriteContext',
                    [('branch_id', int), ('vector_clock', int)],
                )
            ],
        )
        families[module]['session_executor'] = make_class(
            f'{prefix}SessionExecutor',
            [('ctx_ref', f'LazyRef[{prefix}SessionReadContext]')],
        )
        session_read = make_protocol(
            f'{prefix}SessionReadContext',
            {
                **{
                    f'session_{index}': families[module]['session'][index]  # type: ignore[index]
                    for index in range(fanout)
                },
                **shared_fields(families, 'session', fanout),
                'loop_exec': object,
                'execute_in_transaction': families[module]['session_executor'],
            },
            [
                make_protocol_method(
                    f'{prefix}SessionReadContext',
                    'with_write',
                    session_write,
                    [],
                ),
                make_protocol_method(
                    f'{prefix}SessionReadContext',
                    'with_branched_session',
                    f'{prefix}BranchedReadContext',
                    [('branch_id', int), ('vector_clock', int)],
                ),
                *[
                    make_protocol_method(
                        f'{prefix}SessionReadContext',
                        route_name,
                        f'{prefix}BranchedReadContext',
                        [],
                    )
                    for route_name in branch_route_names
                ],
            ],
        )
        families[module]['branched_executor'] = make_class(
            f'{prefix}BranchedExecutor',
            [('ctx_ref', f'LazyRef[{prefix}BranchedReadContext]')],
        )
        branched_write_name = f'{prefix}BranchedWriteContext'
        branched_write_methods = []
        if include_agent:
            branched_write_methods.append(
                make_protocol_method(
                    branched_write_name,
                    'with_agent',
                    'AiAgentWriteContext',
                    [('agent_id', int)],
                )
            )
        branched_write = make_protocol(
            branched_write_name,
            {
                **{
                    f'branch_write_{index}': families[module]['branch_write'][index]  # type: ignore[index]
                    for index in range(fanout)
                },
                **shared_fields(families, 'branch', fanout),
                **{
                    f'branch_extra_{index}': extra_branch_fields[index]
                    for index in range(fanout)
                },
            },
            branched_write_methods,
        )
        branched_read = make_protocol(
            f'{prefix}BranchedReadContext',
            {
                **{
                    f'branch_{index}': families[module]['branch'][index]  # type: ignore[index]
                    for index in range(fanout)
                },
                **shared_fields(families, 'branch', fanout),
                **{
                    f'branch_extra_{index}': extra_branch_fields[index]
                    for index in range(fanout)
                },
                'loop_exec': object,
                'execute_in_branched_transaction': families[module][
                    'branched_executor'
                ],
            },
            [
                make_protocol_method(
                    f'{prefix}BranchedReadContext',
                    'with_branched_write',
                    branched_write,
                    [],
                )
            ],
        )
        message_context = make_protocol(
            f'{prefix}MessageContext',
            {
                **{
                    f'message_{index}': families[module]['message'][index]  # type: ignore[index]
                    for index in range(fanout)
                },
                **shared_fields(families, 'message', fanout),
                **{
                    f'message_extra_{index}': extra_message_fields[index]
                    for index in range(fanout)
                },
            },
            [
                make_protocol_method(
                    f'{prefix}MessageContext',
                    'with_handler',
                    branched_write,
                    [],
                )
            ],
        )
        root_context = make_protocol(
            f'{prefix}Context',
            {
                **{
                    f'base_{index}': families[module]['base'][index]  # type: ignore[index]
                    for index in range(fanout)
                },
                **shared_fields(families, 'base', fanout),
                'loop_exec': object,
                'execute_in_transaction': families[module]['context_executor'],
            },
            [
                make_protocol_method(
                    f'{prefix}Context', 'with_write', write_context, []
                ),
                make_protocol_method(
                    f'{prefix}Context',
                    'with_session',
                    session_read,
                    [('session_id', int)],
                ),
                *[
                    make_protocol_method(
                        f'{prefix}Context', route_name, session_read, []
                    )
                    for route_name in route_names
                ],
                make_protocol_method(
                    f'{prefix}Context',
                    'with_message_scope',
                    message_context,
                    [
                        ('message_id', int),
                        ('channel_name', str),
                        ('session_id', int),
                        ('branch_id', int),
                        ('vector_clock', int),
                    ],
                ),
            ],
        )
        families[module].update({
            'context': root_context,
            'write_context': write_context,
            'session_read_context': session_read,
            'session_write_context': session_write,
            'branched_read_context': branched_read,
            'branched_write_context': branched_write,
            'message_context': message_context,
            'route_names': route_names,
            'branch_route_names': branch_route_names,
        })

    session_module_protocols(
        'chat',
        extra_branch_fields=families['chat']['base'],  # type: ignore[arg-type]
        extra_message_fields=families['chat']['base'],  # type: ignore[arg-type]
        include_agent=False,
    )
    session_module_protocols(
        'game',
        extra_branch_fields=families['game']['game_story_queries'],  # type: ignore[arg-type]
        extra_message_fields=families['game']['game_character_data'],  # type: ignore[arg-type]
        include_agent=False,
    )
    session_module_protocols(
        'ai',
        extra_branch_fields=families['ai']['ai_game_queries'],  # type: ignore[arg-type]
        extra_message_fields=families['ai']['ai_character_queries'],  # type: ignore[arg-type]
        include_agent=True,
    )

    ai_agent = make_protocol(
        'AiAgentWriteContext',
        {
            **{
                f'agent_{index}': families['ai']['agent'][index]  # type: ignore[index]
                for index in range(fanout)
            },
            **shared_fields(families, 'branch', fanout),
            **{
                f'agent_query_{index}': families['ai']['ai_story_queries'][index]  # type: ignore[index]
                for index in range(fanout)
            },
        },
        [
            make_protocol_method(
                'AiAgentWriteContext',
                'with_tool',
                'AiToolWriteContext',
                [('tool_call_id', int)],
            )
        ],
    )
    ai_tool = make_protocol(
        'AiToolWriteContext',
        {
            **{
                f'tool_{index}': families['ai']['tool'][index]  # type: ignore[index]
                for index in range(fanout)
            },
            **shared_fields(families, 'branch', fanout),
            **{
                f'tool_query_{index}': families['ai']['ai_character_queries'][index]  # type: ignore[index]
                for index in range(fanout)
            },
        },
        [],
    )
    families['ai'].update({
        'agent_context': ai_agent,
        'tool_context': ai_tool,
    })

    families['story'].update({
        'context': story_context,
        'write_context': story_write,
    })
    families['character'].update({
        'context': character_context,
        'write_context': character_write,
        'branched_read_context': character_branch_read,
        'branched_write_context': character_branch_write,
        'message_context': character_message,
    })

    families['story']['context_executor'] = make_class(
        'StoryContextExecutor',
        [('ctx_ref', LazyRef[story_context])],
    )
    families['story']['loop_exec'] = make_self_loop_class(
        'StoryLoopExec',
        story_context,
    )
    replace_protocol_field(story_context, 'loop_exec', families['story']['loop_exec'])
    families['story']['write_provider'] = make_bound_transition_class(
        'StoryWriteProvider',
        'with_write',
        families['story']['write_state'],
        [
            ('ctx_ref', LazyRef[story_context]),
            ('shared', families['shared']['base'][0]),
        ],
    )
    replace_protocol_field(story_context, 'self_ref', LazyRef[story_context])
    replace_protocol_field(
        story_context,
        'execute_in_transaction',
        families['story']['context_executor'],
    )
    families['character']['context_executor'] = make_class(
        'CharacterContextExecutor',
        [('ctx_ref', LazyRef[character_context])],
    )
    families['character']['loop_exec'] = make_self_loop_class(
        'CharacterLoopExec',
        character_context,
    )
    replace_protocol_field(
        character_context,
        'loop_exec',
        families['character']['loop_exec'],
    )
    families['character']['write_provider'] = make_bound_transition_class(
        'CharacterWriteProvider',
        'with_write',
        families['character']['write_state'],
        [
            ('ctx_ref', LazyRef[character_context]),
            ('shared', families['shared']['base'][0]),
        ],
    )
    replace_protocol_field(character_context, 'self_ref', LazyRef[character_context])
    replace_protocol_field(
        character_context,
        'execute_in_transaction',
        families['character']['context_executor'],
    )
    families['character']['branched_executor'] = make_class(
        'CharacterBranchedExecutor',
        [('ctx_ref', LazyRef[character_branch_read])],
    )
    families['character']['branch_loop_exec'] = make_self_loop_class(
        'CharacterBranchLoopExec',
        character_branch_read,
    )
    replace_protocol_field(
        character_branch_read,
        'loop_exec',
        families['character']['branch_loop_exec'],
    )
    families['character']['branched_write_provider'] = make_bound_transition_class(
        'CharacterBranchedWriteProvider',
        'with_branched_write',
        families['character']['branch_write_state'],
        [
            ('ctx_ref', LazyRef[character_branch_read]),
            ('shared', families['shared']['branch'][0]),
        ],
    )
    replace_protocol_field(
        character_branch_read,
        'self_ref',
        LazyRef[character_branch_read],
    )
    replace_protocol_field(
        character_branch_read,
        'execute_in_branched_transaction',
        families['character']['branched_executor'],
    )

    for module in ('chat', 'game', 'ai'):
        prefix = title(module)
        families[module]['context_executor'] = make_class(
            f'{prefix}ContextExecutor',
            [('ctx_ref', LazyRef[families[module]['context']])],
        )
        families[module]['loop_exec'] = make_self_loop_class(
            f'{prefix}LoopExec',
            families[module]['context'],
        )
        replace_protocol_field(
            families[module]['context'],
            'loop_exec',
            families[module]['loop_exec'],
        )
        families[module]['write_provider'] = make_bound_transition_class(
            f'{prefix}WriteProvider',
            'with_write',
            families[module]['write_state'],
            [
                ('ctx_ref', LazyRef[families[module]['context']]),
                ('shared', families['shared']['base'][0]),
            ],
        )
        replace_protocol_field(
            families[module]['context'],
            'self_ref',
            LazyRef[families[module]['context']],
        )
        replace_protocol_field(
            families[module]['context'],
            'execute_in_transaction',
            families[module]['context_executor'],
        )
        families[module]['session_executor'] = make_class(
            f'{prefix}SessionExecutor',
            [('ctx_ref', LazyRef[families[module]['session_read_context']])],
        )
        families[module]['session_loop_exec'] = make_self_loop_class(
            f'{prefix}SessionLoopExec',
            families[module]['session_read_context'],
        )
        replace_protocol_field(
            families[module]['session_read_context'],
            'loop_exec',
            families[module]['session_loop_exec'],
        )
        families[module]['session_write_provider'] = make_bound_transition_class(
            f'{prefix}SessionWriteProvider',
            'with_write',
            families[module]['write_state'],
            [
                ('ctx_ref', LazyRef[families[module]['session_read_context']]),
                ('shared', families['shared']['session'][0]),
            ],
        )
        replace_protocol_field(
            families[module]['session_read_context'],
            'self_ref',
            LazyRef[families[module]['session_read_context']],
        )
        replace_protocol_field(
            families[module]['session_read_context'],
            'execute_in_transaction',
            families[module]['session_executor'],
        )
        families[module]['branched_executor'] = make_class(
            f'{prefix}BranchedExecutor',
            [('ctx_ref', LazyRef[families[module]['branched_read_context']])],
        )
        families[module]['branch_loop_exec'] = make_self_loop_class(
            f'{prefix}BranchLoopExec',
            families[module]['branched_read_context'],
        )
        replace_protocol_field(
            families[module]['branched_read_context'],
            'loop_exec',
            families[module]['branch_loop_exec'],
        )
        families[module]['branched_write_provider'] = make_bound_transition_class(
            f'{prefix}BranchedWriteProvider',
            'with_branched_write',
            families[module]['branch_write_state'],
            [
                ('ctx_ref', LazyRef[families[module]['branched_read_context']]),
                ('session_shared', families['shared']['session'][0]),
                ('shared', families['shared']['branch'][0]),
            ],
        )
        replace_protocol_field(
            families[module]['branched_read_context'],
            'self_ref',
            LazyRef[families[module]['branched_read_context']],
        )
        replace_protocol_field(
            families[module]['branched_read_context'],
            'execute_in_branched_transaction',
            families[module]['branched_executor'],
        )

    for module in ROOT_METHODS:
        export(context_names[module], families[module]['context'])

    app_context = make_protocol(
        'AppContext',
        {'loop_exec': object},
        [
            make_protocol_method(
                'AppContext',
                f'with_{module}_context',
                Annotated[families[module]['context'], qual(module)],
                [],
            )
            for module in ROOT_METHODS
        ],
    )

    families['app'] = {'loop_exec': make_self_loop_class('AppLoopExec', app_context)}
    replace_protocol_field(app_context, 'loop_exec', families['app']['loop_exec'])

    benchmark_root = make_protocol(
        'BenchmarkRoot',
        {
            'app': app_context,
            'loop_exec': object,
        },
        [],
    )
    families['app']['root_loop_exec'] = make_self_loop_class(
        'BenchmarkRootLoopExec',
        benchmark_root,
    )
    replace_protocol_field(
        benchmark_root,
        'loop_exec',
        families['app']['root_loop_exec'],
    )

    export('AppContext', app_context)
    export('BenchmarkRoot', benchmark_root)
    return benchmark_root, families


def add_module_methods(
    registry: RegistryBuilder,
    families: dict[str, dict[str, object]],
) -> RegistryBuilder:
    registry = registry.register(families['app']['loop_exec'])(
        families['app']['loop_exec']
    )
    registry = registry.register(families['app']['root_loop_exec'])(
        families['app']['root_loop_exec']
    )
    for module in ROOT_METHODS:
        method_name = f'with_{module}_context'
        registry = registry.register_method(AppContext, method_name=method_name)(
            make_transition_impl(
                'AppContext', method_name, families[module]['module_state'], []
            )
        )
    return registry


def add_shared_registry(
    registry: RegistryBuilder,
    families: dict[str, dict[str, object]],
    fanout: int,
) -> RegistryBuilder:
    for key in ('base', 'session', 'branch', 'message'):
        for index in range(fanout):
            registry = registry.register(families['shared'][key][index])(
                families['shared'][key][index]
            )
    return registry


def add_story_registry(
    registry: RegistryBuilder,
    families: dict[str, dict[str, object]],
    fanout: int,
) -> RegistryBuilder:
    for index in range(fanout):
        registry = registry.register(families['story']['base'][index])(
            families['story']['base'][index]
        )
        registry = registry.register(families['story']['write'][index])(
            families['story']['write'][index]
        )
    registry = registry.register(families['story']['context_executor'])(
        families['story']['context_executor']
    )
    registry = registry.register(families['story']['loop_exec'])(
        families['story']['loop_exec']
    )
    registry = registry.register(families['story']['write_provider'])(
        families['story']['write_provider']
    )
    registry = registry.register_method(StoryContext, method_name='with_write')(
        families['story']['write_provider']
    )
    registry = registry.register_method_hook(StoryContext, method_name='with_write')(
        make_hook(
            'StoryContext',
            'with_write',
            [
                ('write_0', families['story']['write'][0]),
                ('shared_base_0', families['shared']['base'][0]),
            ],
        )
    )
    return registry


def add_character_registry(
    registry: RegistryBuilder,
    families: dict[str, dict[str, object]],
    fanout: int,
) -> RegistryBuilder:
    for index in range(fanout):
        registry = registry.register(families['character']['base'][index])(
            families['character']['base'][index]
        )
        registry = registry.register(families['character']['write'][index])(
            families['character']['write'][index]
        )
        registry = registry.register(families['character']['branch'][index])(
            families['character']['branch'][index]
        )
        registry = registry.register(families['character']['branch_write'][index])(
            families['character']['branch_write'][index]
        )
        registry = registry.register(families['character']['message'][index])(
            families['character']['message'][index]
        )
    registry = registry.register(families['character']['context_executor'])(
        families['character']['context_executor']
    )
    registry = registry.register(families['character']['loop_exec'])(
        families['character']['loop_exec']
    )
    registry = registry.register(families['character']['branched_executor'])(
        families['character']['branched_executor']
    )
    registry = registry.register(families['character']['branch_loop_exec'])(
        families['character']['branch_loop_exec']
    )
    registry = registry.register(families['character']['write_provider'])(
        families['character']['write_provider']
    )
    registry = registry.register(families['character']['branched_write_provider'])(
        families['character']['branched_write_provider']
    )
    registry = registry.register_method(CharacterContext, method_name='with_write')(
        families['character']['write_provider']
    )
    registry = registry.register_method(
        CharacterContext, method_name='with_branched_session'
    )(
        make_transition_impl(
            'CharacterContext',
            'with_branched_session',
            families['character']['branch_state'],
            [('branch_id', int), ('vector_clock', int)],
        )
    )
    registry = registry.register_method(
        CharacterContext, method_name='with_message_scope'
    )(
        make_transition_impl(
            'CharacterContext',
            'with_message_scope',
            families['character']['message_state'],
            [
                ('message_id', int),
                ('channel_name', str),
                ('session_id', int),
                ('branch_id', int),
                ('vector_clock', int),
            ],
        )
    )
    registry = registry.register_method(
        CharacterBranchedReadContext,
        method_name='with_branched_write',
    )(families['character']['branched_write_provider'])
    registry = registry.register_method(
        CharacterMessageContext, method_name='with_handler'
    )(
        make_transition_impl(
            'CharacterMessageContext',
            'with_handler',
            families['character']['handler_state'],
            [],
        )
    )
    for protocol, method_name, leaf, shared in [
        (
            CharacterContext,
            'with_write',
            families['character']['write'][0],
            families['shared']['base'][0],
        ),
        (
            CharacterContext,
            'with_branched_session',
            families['character']['branch'][0],
            families['shared']['branch'][0],
        ),
        (
            CharacterContext,
            'with_message_scope',
            families['character']['message'][0],
            families['shared']['message'][0],
        ),
        (
            CharacterBranchedReadContext,
            'with_branched_write',
            families['character']['branch_write'][0],
            families['shared']['branch'][0],
        ),
        (
            CharacterMessageContext,
            'with_handler',
            families['character']['branch_write'][0],
            families['shared']['branch'][0],
        ),
    ]:
        registry = registry.register_method_hook(protocol, method_name=method_name)(
            make_hook(
                protocol.__name__,
                method_name,
                [('dep', leaf), ('shared', shared)],
            )
        )
    return registry


def add_session_module_registry(
    registry: RegistryBuilder,
    module: str,
    families: dict[str, dict[str, object]],
    fanout: int,
    *,
    include_agent: bool,
) -> RegistryBuilder:
    spec = families[module]
    for key in ('base', 'write', 'session', 'branch', 'branch_write', 'message'):
        for index in range(fanout):
            registry = registry.register(spec[key][index])(spec[key][index])  # type: ignore[index]
    for key in ('context_executor', 'session_executor', 'branched_executor'):
        registry = registry.register(spec[key])(spec[key])
    for key in ('loop_exec', 'session_loop_exec', 'branch_loop_exec'):
        registry = registry.register(spec[key])(spec[key])
    for key in ('write_provider', 'session_write_provider', 'branched_write_provider'):
        registry = registry.register(spec[key])(spec[key])
    extra_keys = []
    if module == 'game':
        extra_keys = ['game_story_queries', 'game_character_data']
    if module == 'ai':
        extra_keys = ['ai_game_queries', 'ai_story_queries', 'ai_character_queries']
    for key in extra_keys:
        for index in range(fanout):
            registry = registry.register(spec[key][index])(spec[key][index])  # type: ignore[index]
    if include_agent:
        for key in ('agent', 'tool'):
            for index in range(fanout):
                registry = registry.register(spec[key][index])(spec[key][index])  # type: ignore[index]
    prefix = title(module)
    registry = registry.register_method(spec['context'], method_name='with_write')(
        spec['write_provider']
    )
    registry = registry.register_method(spec['context'], method_name='with_session')(
        make_transition_impl(
            f'{prefix}Context',
            'with_session',
            spec['session_state'],
            [('session_id', int)],
        )
    )
    for index, route_name in enumerate(spec['route_names']):
        registry = registry.register_method(spec['context'], method_name=route_name)(
            make_fixed_transition_impl(
                f'{prefix}Context',
                route_name,
                spec['session_state'],
                {'session_id': index + 1},
            )
        )
    registry = registry.register_method(
        spec['context'], method_name='with_message_scope'
    )(
        make_transition_impl(
            f'{prefix}Context',
            'with_message_scope',
            spec['message_state'],
            [
                ('message_id', int),
                ('channel_name', str),
                ('session_id', int),
                ('branch_id', int),
                ('vector_clock', int),
            ],
        )
    )
    registry = registry.register_method(
        spec['session_read_context'], method_name='with_write'
    )(spec['session_write_provider'])
    registry = registry.register_method(
        spec['session_read_context'], method_name='with_branched_session'
    )(
        make_transition_impl(
            f'{prefix}SessionReadContext',
            'with_branched_session',
            spec['branch_state'],
            [('branch_id', int), ('vector_clock', int)],
        )
    )
    for index, route_name in enumerate(spec['branch_route_names']):
        registry = registry.register_method(
            spec['session_read_context'], method_name=route_name
        )(
            make_fixed_transition_impl(
                f'{prefix}SessionReadContext',
                route_name,
                spec['branch_state'],
                {'branch_id': index + 1, 'vector_clock': index + 1},
            )
        )
    registry = registry.register_method(
        spec['session_write_context'], method_name='with_branched_session'
    )(
        make_transition_impl(
            f'{prefix}SessionWriteContext',
            'with_branched_session',
            spec['branch_state'],
            [('branch_id', int), ('vector_clock', int)],
        )
    )
    registry = registry.register_method(
        spec['branched_read_context'], method_name='with_branched_write'
    )(spec['branched_write_provider'])
    registry = registry.register_method(
        spec['message_context'], method_name='with_handler'
    )(
        make_transition_impl(
            f'{prefix}MessageContext',
            'with_handler',
            spec['handler_state'],
            [],
        )
    )
    if include_agent:
        registry = registry.register_method(
            spec['branched_write_context'], method_name='with_agent'
        )(
            make_transition_impl(
                f'{prefix}BranchedWriteContext',
                'with_agent',
                spec['agent_state'],
                [('agent_id', int)],
            )
        )
        registry = registry.register_method(
            AiAgentWriteContext, method_name='with_tool'
        )(
            make_transition_impl(
                'AiAgentWriteContext',
                'with_tool',
                spec['tool_state'],
                [('tool_call_id', int)],
            )
        )

    for protocol, method_name, leaf, shared in [
        (
            spec['context'],
            'with_write',
            spec['write'][0],
            families['shared']['base'][0],
        ),
        (
            spec['context'],
            'with_session',
            spec['session'][0],
            families['shared']['session'][0],
        ),
        (
            spec['context'],
            spec['route_names'][0],
            spec['session'][0],
            families['shared']['session'][0],
        ),
        (
            spec['context'],
            'with_message_scope',
            spec['message'][0],
            families['shared']['message'][0],
        ),
        (
            spec['session_read_context'],
            'with_write',
            spec['write'][0],
            families['shared']['base'][0],
        ),
        (
            spec['session_read_context'],
            'with_branched_session',
            spec['branch'][0],
            families['shared']['branch'][0],
        ),
        (
            spec['session_read_context'],
            spec['branch_route_names'][0],
            spec['branch'][0],
            families['shared']['branch'][0],
        ),
        (
            spec['session_write_context'],
            'with_branched_session',
            spec['branch_write'][0],
            families['shared']['branch'][0],
        ),
        (
            spec['branched_read_context'],
            'with_branched_write',
            spec['branch_write'][0],
            families['shared']['branch'][0],
        ),
        (
            spec['message_context'],
            'with_handler',
            spec['branch_write'][0],
            families['shared']['branch'][0],
        ),
    ]:
        registry = registry.register_method_hook(protocol, method_name=method_name)(
            make_hook(
                protocol.__name__,
                method_name,
                [('dep', leaf), ('shared', shared)],
            )
        )
    for route_name in spec['route_names'][1:]:
        registry = registry.register_method_hook(
            spec['context'], method_name=route_name
        )(
            make_hook(
                spec['context'].__name__,
                route_name,
                [
                    ('dep', spec['session'][0]),
                    ('shared', families['shared']['session'][0]),
                ],
            )
        )
    for route_name in spec['branch_route_names'][1:]:
        registry = registry.register_method_hook(
            spec['session_read_context'], method_name=route_name
        )(
            make_hook(
                spec['session_read_context'].__name__,
                route_name,
                [
                    ('dep', spec['branch'][0]),
                    ('shared', families['shared']['branch'][0]),
                ],
            )
        )
    if include_agent:
        registry = registry.register_method_hook(
            spec['branched_write_context'],
            method_name='with_agent',
        )(
            make_hook(
                'AiBranchedWriteContext',
                'with_agent',
                [
                    ('dep', spec['agent'][0]),
                    ('shared', families['shared']['branch'][0]),
                ],
            )
        )
        registry = registry.register_method_hook(
            AiAgentWriteContext, method_name='with_tool'
        )(
            make_hook(
                'AiAgentWriteContext',
                'with_tool',
                [('dep', spec['tool'][0]), ('shared', families['shared']['branch'][0])],
            )
        )
    return registry


def build_registry(
    families: dict[str, dict[str, object]],
    fanout: int,
) -> RegistryBuilder:
    registry = add_module_methods(RegistryBuilder(), families)
    shared_registry = add_shared_registry(RegistryBuilder(), families, fanout)

    chat_registry = add_session_module_registry(
        RegistryBuilder(),
        'chat',
        families,
        fanout,
        include_agent=False,
    )
    game_registry = add_session_module_registry(
        RegistryBuilder(),
        'game',
        families,
        fanout,
        include_agent=False,
    )
    ai_registry = add_session_module_registry(
        RegistryBuilder(),
        'ai',
        families,
        fanout,
        include_agent=True,
    )
    story_registry = add_story_registry(RegistryBuilder(), families, fanout)
    character_registry = add_character_registry(RegistryBuilder(), families, fanout)

    return (
        registry
        .include(shared_registry, chat_registry, qualifiers=qual('chat'))
        .include(shared_registry, game_registry, qualifiers=qual('game'))
        .include(shared_registry, ai_registry, qualifiers=qual('ai'))
        .include(shared_registry, story_registry, qualifiers=qual('story'))
        .include(shared_registry, character_registry, qualifiers=qual('character'))
    )


def run_once(
    *,
    scenario: Scenario,
    fanout: int,
    invoke: bool,
) -> None:
    root_context, families = build_appish_contexts(scenario, fanout)

    registry_started = perf_counter()
    native = build_registry(families, fanout).build()
    registry_elapsed = perf_counter() - registry_started

    compile_started = perf_counter()
    root = compile(root_context, native, default_rules())
    compile_elapsed = perf_counter() - compile_started

    invoke_elapsed = None
    if invoke:
        invoke_started = perf_counter()
        app = root.app
        ai_context = app.with_ai_context()
        ai_session = ai_context.with_session(1)
        ai_branch = ai_session.with_branched_session(2, 3)
        ai_write = ai_branch.with_branched_write()
        agent = ai_write.with_agent(4)
        tool = agent.with_tool(5)
        _ = tool.tool_0

        game_context = app.with_game_context()
        message_ctx = game_context.with_message_scope(10, 'game', 1, 2, 3)
        _ = message_ctx.message_extra_0

        character_context = app.with_character_context()
        character_branch = character_context.with_branched_session(2, 3)
        _ = character_branch.branch_0

        story_context = app.with_story_context()
        _ = story_context.with_write().write_0
        invoke_elapsed = perf_counter() - invoke_started

    print(
        ' '.join([
            'benchmark=appish_transitions',
            f'scenario={scenario}',
            f'fanout={fanout}',
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
        run_once(
            scenario=scenario,
            fanout=args.fanout,
            invoke=args.invoke,
        )


if __name__ == '__main__':
    main()
