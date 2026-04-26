import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol, cast

from inlay import Registry, compile
from inlay.default import default_rules

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ModuleNotFoundError as exc:  # pragma: no cover - benchmark bootstrap guard
    raise SystemExit(
        'This benchmark needs jinja2. Run it from the workspace root or install jinja2.'
    ) from exc

type Scenario = Literal['portable', 'env-sensitive']
type ModuleKind = Literal['session', 'branch', 'write']
type ExtraTransition = Literal['none', 'agent', 'saga']

TEMPLATE_DIR = Path(__file__).resolve().parent / 'templates'
TEMPLATE_NAME = 'production_shape_context.py.j2'


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
class HookSpec:
    protocol_name: str
    method_name: str
    function_name: str
    params: tuple[TypedName, ...]


@dataclass(frozen=True)
class MethodRegistrationSpec:
    protocol_name: str
    method_name: str
    function_name: str


@dataclass(frozen=True)
class ConstructorRegistrationSpec:
    type_name: str
    qualifiers: str
    provider_name: str


@dataclass(frozen=True)
class ModuleShape:
    name: str
    title: str
    kind: ModuleKind
    extra_transition: ExtraTransition
    slot: int
    alt_slot: int
    route_indices: tuple[int, ...]
    branch_indices: tuple[int, ...]
    handler_indices: tuple[int, ...]
    hook_indices: tuple[int, ...]

    @property
    def session(self) -> bool:
        return self.kind == 'session'

    @property
    def branch_only(self) -> bool:
        return self.kind == 'branch'

    @property
    def write_only(self) -> bool:
        return self.kind == 'write'

    @property
    def context(self) -> str:
        return f'{self.title}Context'

    @property
    def qualified_context(self) -> str:
        return f"Annotated[{self.context}, qual('{self.name}')]"

    @property
    def module_state(self) -> str:
        return f'{self.title}ModuleState'

    @property
    def write_state(self) -> str:
        return f'{self.title}WriteState'

    @property
    def base_service(self) -> str:
        return f'{self.title}BaseService'

    @property
    def write_service(self) -> str:
        return f'{self.title}WriteService'

    @property
    def write_context(self) -> str:
        return f'{self.title}WriteContext'

    @property
    def base_shared(self) -> str:
        return f'BaseShared{self.slot}'

    @property
    def base_shared_alt(self) -> str:
        return f'BaseShared{self.alt_slot}'

    @property
    def hook_shared(self) -> str:
        return f'HookShared{self.slot}'

    @property
    def hook_shared_alt(self) -> str:
        return f'HookShared{self.alt_slot}'

    @property
    def session_shared(self) -> str:
        return f'SessionShared{self.slot}'

    @property
    def branch_shared(self) -> str:
        return f'BranchShared{self.slot}'

    @property
    def branch_shared_alt(self) -> str:
        return f'BranchShared{self.alt_slot}'

    @property
    def message_shared(self) -> str:
        return f'MessageShared{self.alt_slot}'

    @property
    def session_state(self) -> str:
        return f'{self.title}SessionState'

    @property
    def branch_state(self) -> str:
        return f'{self.title}BranchState'

    @property
    def message_state(self) -> str:
        return f'{self.title}MessageState'

    @property
    def handler_state(self) -> str:
        return f'{self.title}HandlerState'

    @property
    def session_service(self) -> str:
        return f'{self.title}SessionService'

    @property
    def session_context(self) -> str:
        return f'{self.title}SessionReadContext'

    @property
    def branch_service(self) -> str:
        return f'{self.title}BranchService'

    @property
    def branch_audit(self) -> str:
        return f'{self.title}BranchAudit'

    @property
    def branch_context(self) -> str:
        return f'{self.title}BranchedReadContext'

    @property
    def branch_write_service(self) -> str:
        return f'{self.title}BranchWriteService'

    @property
    def branch_write_context(self) -> str:
        return f'{self.title}BranchedWriteContext'

    @property
    def message_service(self) -> str:
        return f'{self.title}MessageService'

    @property
    def message_context(self) -> str:
        return f'{self.title}MessageContext'

    @property
    def transaction_executor(self) -> str:
        return f'{self.title}TransactionExecutor'

    @property
    def handler_refs(self) -> str:
        return f'{self.title}HandlerRefs'

    @property
    def handler_classes(self) -> tuple[str, ...]:
        return tuple(f'{self.title}Handler{index}' for index in self.handler_indices)

    @property
    def ai_agent_state(self) -> str:
        return 'AiAgentState'

    @property
    def ai_tool_state(self) -> str:
        return 'AiToolState'

    @property
    def ai_agent_service(self) -> str:
        return 'AiAgentService'

    @property
    def ai_tool_service(self) -> str:
        return 'AiToolService'

    @property
    def ai_agent_context(self) -> str:
        return 'AiAgentWriteContext'

    @property
    def ai_tool_context(self) -> str:
        return 'AiToolWriteContext'

    @property
    def game_saga_state(self) -> str:
        return 'GameSagaState'

    @property
    def game_saga_service(self) -> str:
        return 'GameSagaService'

    @property
    def game_saga_context(self) -> str:
        return 'GameSagaWriteContext'

    @property
    def read_extras(self) -> tuple[str, ...]:
        match self.name:
            case 'ai':
                return ('AiGameQueries', 'AiStoryQueries', 'AiCharacterQueries')
            case 'game':
                return ('GameStoryQueries', 'GameCharacterData')
            case _:
                return ()

    @property
    def write_extras(self) -> tuple[str, ...]:
        match self.name:
            case 'ai':
                return ('AiGameQueries', 'AiStoryQueries')
            case 'game':
                return ('GameStoryQueries', 'GameCharacterData')
            case _:
                return ()


@dataclass(frozen=True)
class ModuleRegistrySpec:
    module: ModuleShape
    common_classes: tuple[str, ...]
    write_classes: tuple[str, ...]
    methods: tuple[MethodRegistrationSpec, ...]
    hooks: tuple[HookSpec, ...]


class BenchmarkBranchWriteContext(Protocol):
    branch_write: object


class BenchmarkBranchContext(Protocol):
    branch: object

    def with_branched_write(self) -> BenchmarkBranchWriteContext: ...


class BenchmarkSessionContext(Protocol):
    def branch_0(self) -> BenchmarkBranchContext: ...


class BenchmarkGameSagaContext(Protocol):
    saga: object


class BenchmarkGameBranchWriteContext(Protocol):
    def with_saga(self, saga_id: int) -> BenchmarkGameSagaContext: ...


class BenchmarkGameBranchContext(Protocol):
    branch: object

    def with_branched_write(self) -> BenchmarkGameBranchWriteContext: ...


class BenchmarkGameSessionContext(Protocol):
    def branch_0(self) -> BenchmarkGameBranchContext: ...


class BenchmarkAiToolContext(Protocol):
    tool: object


class BenchmarkAiAgentContext(Protocol):
    def with_tool(self, tool_call_id: int) -> BenchmarkAiToolContext: ...


class BenchmarkAiBranchWriteContext(Protocol):
    def with_agent(self, agent_id: int) -> BenchmarkAiAgentContext: ...


class BenchmarkAiBranchContext(Protocol):
    branch: object

    def with_branched_write(self) -> BenchmarkAiBranchWriteContext: ...


class BenchmarkAiSessionContext(Protocol):
    def branch_0(self) -> BenchmarkAiBranchContext: ...


class BenchmarkCharacterMessageContext(Protocol):
    def with_handler(self) -> BenchmarkBranchWriteContext: ...


class BenchmarkCharacterContext(Protocol):
    def with_message_scope(
        self,
        message_id: int,
        channel_name: str,
        session_id: int,
        branch_id: int,
        vector_clock: int,
    ) -> BenchmarkCharacterMessageContext: ...


class BenchmarkStoryWriteContext(Protocol):
    write: object


class BenchmarkStoryContext(Protocol):
    def with_write(self) -> BenchmarkStoryWriteContext: ...


class BenchmarkChatContext(Protocol):
    def route_0(self) -> BenchmarkSessionContext: ...


class BenchmarkGameContext(Protocol):
    def route_0(self) -> BenchmarkGameSessionContext: ...


class BenchmarkAiContext(Protocol):
    def route_0(self) -> BenchmarkAiSessionContext: ...


class BenchmarkAppContext(Protocol):
    def with_chat_context(self) -> BenchmarkChatContext: ...

    def with_game_context(self) -> BenchmarkGameContext: ...

    def with_ai_context(self) -> BenchmarkAiContext: ...

    def with_character_context(self) -> BenchmarkCharacterContext: ...

    def with_story_context(self) -> BenchmarkStoryContext: ...


class BenchmarkRootContext(Protocol):
    app: BenchmarkAppContext


def typed(name: str, type_ref: str) -> TypedName:
    return TypedName(name=name, type=type_ref)


def title(name: str) -> str:
    return ''.join(part.capitalize() for part in name.split('_'))


def env_int(module: str) -> str:
    return f"Annotated[int, qual('{module}')]"


def env_str(module: str) -> str:
    return f"Annotated[str, qual('{module}')]"


def write_context(type_ref: str) -> str:
    return f"Annotated[{type_ref}, qual('write')]"


def modules_for(
    density: int,
    handler_count: int,
    shared_slots: int,
    enabled_modules: tuple[str, ...],
) -> tuple[ModuleShape, ...]:
    slot_count = max(shared_slots, 1)
    indices = tuple(range(density))
    handler_indices = tuple(range(handler_count))
    definitions: tuple[tuple[str, ModuleKind, ExtraTransition], ...] = (
        ('chat', 'session', 'none'),
        ('game', 'session', 'saga'),
        ('ai', 'session', 'agent'),
        ('story', 'write', 'none'),
        ('character', 'branch', 'none'),
    )
    modules: list[ModuleShape] = []
    for index, (name, kind, extra_transition) in enumerate(definitions):
        if name not in enabled_modules:
            continue
        modules.append(
            ModuleShape(
                name=name,
                title=title(name),
                kind=kind,
                extra_transition=extra_transition,
                slot=index % slot_count,
                alt_slot=(index + 1) % slot_count,
                route_indices=indices,
                branch_indices=indices,
                handler_indices=handler_indices,
                hook_indices=indices,
            )
        )
    return tuple(modules)


def shared_classes(shared_slots: int) -> tuple[ClassSpec, ...]:
    classes: list[ClassSpec] = []
    for slot in range(max(shared_slots, 1)):
        classes.extend([
            ClassSpec(f'BaseShared{slot}', ()),
            ClassSpec(f'SessionShared{slot}', (typed('base', f'BaseShared{slot}'),)),
            ClassSpec(f'BranchShared{slot}', (typed('base', f'BaseShared{slot}'),)),
            ClassSpec(f'MessageShared{slot}', (typed('base', f'BaseShared{slot}'),)),
            ClassSpec(f'HookShared{slot}', (typed('base', f'BaseShared{slot}'),)),
        ])
    return tuple(classes)


def module_states(module: ModuleShape) -> tuple[ClassSpec, ...]:
    classes = [ClassSpec(module.module_state, ()), ClassSpec(module.write_state, ())]
    if module.session or module.branch_only:
        classes.append(ClassSpec(module.handler_state, ()))
    return tuple(classes)


def module_typed_dicts(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedDictSpec, ...]:
    typed_dicts: list[TypedDictSpec] = []
    if module.session:
        typed_dicts.extend([
            TypedDictSpec(
                module.session_state, (typed('session_id', env_int(module.name)),)
            ),
            TypedDictSpec(
                module.branch_state,
                (
                    typed('branch_id', env_int(module.name)),
                    typed('vector_clock', env_int(module.name)),
                ),
            ),
            TypedDictSpec(
                module.message_state,
                (
                    typed('message_id', env_int(module.name)),
                    typed('channel_name', env_str(module.name)),
                    typed('session_id', env_int(module.name)),
                    typed('branch_id', env_int(module.name)),
                    typed('vector_clock', env_int(module.name)),
                ),
            ),
        ])
    if module.branch_only:
        typed_dicts.extend([
            TypedDictSpec(
                module.branch_state,
                (
                    typed('branch_id', env_int(module.name)),
                    typed('vector_clock', env_int(module.name)),
                ),
            ),
            TypedDictSpec(
                module.message_state,
                (
                    typed('message_id', env_int(module.name)),
                    typed('channel_name', env_str(module.name)),
                    typed('session_id', env_int(module.name)),
                    typed('branch_id', env_int(module.name)),
                    typed('vector_clock', env_int(module.name)),
                ),
            ),
        ])
    match module.extra_transition:
        case 'agent':
            typed_dicts.extend([
                TypedDictSpec(
                    module.ai_agent_state, (typed('agent_id', env_int('ai')),)
                ),
                TypedDictSpec(
                    module.ai_tool_state,
                    (typed('tool_call_id', env_int('ai')),),
                ),
            ])
        case 'saga':
            typed_dicts.append(
                TypedDictSpec(
                    module.game_saga_state, (typed('saga_id', env_int('game')),)
                )
            )
        case 'none':
            pass
    return tuple(typed_dicts)


def base_service_params(module: ModuleShape) -> tuple[TypedName, ...]:
    return (typed('shared', module.base_shared),)


def write_service_params(module: ModuleShape) -> tuple[TypedName, ...]:
    return (
        typed('base', module.base_service),
        typed('shared', module.base_shared_alt),
    )


def transaction_executor_params(module: ModuleShape) -> tuple[TypedName, ...]:
    return (typed('shared', module.branch_shared),)


def handler_refs_params(module: ModuleShape) -> tuple[TypedName, ...]:
    return (typed('shared', module.branch_shared),)


def handler_params(module: ModuleShape) -> tuple[TypedName, ...]:
    return (
        typed('shared', module.hook_shared),
        typed('executor', module.transaction_executor),
    )


def branch_audit_params(module: ModuleShape) -> tuple[TypedName, ...]:
    return (
        typed('shared', module.branch_shared_alt),
        typed('executor', module.transaction_executor),
        typed('handler_refs', module.handler_refs),
    )


def session_service_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed('base', module.base_service),
        typed('shared', module.session_shared),
    ]
    if scenario == 'env-sensitive':
        params.insert(0, typed('session_id', env_int(module.name)))
    return tuple(params)


def branch_service_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed('base', module.base_service),
        typed('shared', module.branch_shared),
        typed('audit', module.branch_audit),
        typed('executor', module.transaction_executor),
        typed('handler_refs', module.handler_refs),
    ]
    if module.session:
        params.insert(1, typed('session', module.session_service))
    if scenario == 'env-sensitive':
        params = [
            typed('branch_id', env_int(module.name)),
            typed('vector_clock', env_int(module.name)),
            *params,
        ]
    return tuple(params)


def branch_write_service_params(module: ModuleShape) -> tuple[TypedName, ...]:
    return (
        typed('branch', module.branch_service),
        typed('shared', module.branch_shared_alt),
        typed('handler_refs', module.handler_refs),
    )


def message_service_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed('base', module.base_service),
        typed('shared', module.message_shared),
        typed('audit', module.branch_audit),
        typed('handler_refs', module.handler_refs),
    ]
    if scenario == 'env-sensitive':
        params = [
            typed('message_id', env_int(module.name)),
            typed('channel_name', env_str(module.name)),
            typed('session_id', env_int(module.name)),
            typed('branch_id', env_int(module.name)),
            typed('vector_clock', env_int(module.name)),
            *params,
        ]
    return tuple(params)


def ai_game_queries_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed(
            'game_base',
            "Annotated[GameBaseService, qual('game') ]".replace(' ]', ']'),
        ),
        typed(
            'story_base',
            "Annotated[StoryBaseService, qual('story') ]".replace(' ]', ']'),
        ),
        typed('shared', module.branch_shared),
    ]
    if scenario == 'env-sensitive':
        params = [
            typed('session_id', env_int('ai')),
            typed('branch_id', env_int('ai')),
            *params,
        ]
    return tuple(params)


def ai_story_queries_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed(
            'story_base',
            "Annotated[StoryBaseService, qual('story') ]".replace(' ]', ']'),
        ),
        typed('shared', module.branch_shared),
    ]
    if scenario == 'env-sensitive':
        params = [
            typed('session_id', env_int('ai')),
            typed('branch_id', env_int('ai')),
            *params,
        ]
    return tuple(params)


def ai_character_queries_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed(
            'character_base',
            "Annotated[CharacterBaseService, qual('character') ]".replace(' ]', ']'),
        ),
        typed('shared', module.branch_shared),
    ]
    if scenario == 'env-sensitive':
        params = [
            typed('session_id', env_int('ai')),
            typed('branch_id', env_int('ai')),
            *params,
        ]
    return tuple(params)


def ai_agent_service_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed('base', module.base_service),
        typed('shared', module.branch_shared_alt),
        typed('story_queries', 'AiStoryQueries'),
    ]
    if scenario == 'env-sensitive':
        params = [
            typed('session_id', env_int('ai')),
            typed('branch_id', env_int('ai')),
            typed('agent_id', env_int('ai')),
            *params,
        ]
    return tuple(params)


def ai_tool_service_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed('base', module.base_service),
        typed('shared', module.branch_shared_alt),
        typed('character_queries', 'AiCharacterQueries'),
    ]
    if scenario == 'env-sensitive':
        params = [
            typed('session_id', env_int('ai')),
            typed('branch_id', env_int('ai')),
            typed('agent_id', env_int('ai')),
            typed('tool_call_id', env_int('ai')),
            *params,
        ]
    return tuple(params)


def game_story_queries_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed(
            'story_base',
            "Annotated[StoryBaseService, qual('story') ]".replace(' ]', ']'),
        ),
        typed('shared', module.branch_shared),
    ]
    if scenario == 'env-sensitive':
        params = [
            typed('session_id', env_int('game')),
            typed('branch_id', env_int('game')),
            *params,
        ]
    return tuple(params)


def game_character_data_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed(
            'character_base',
            "Annotated[CharacterBaseService, qual('character') ]".replace(' ]', ']'),
        ),
        typed('shared', module.branch_shared),
    ]
    if scenario == 'env-sensitive':
        params = [
            typed('session_id', env_int('game')),
            typed('branch_id', env_int('game')),
            *params,
        ]
    return tuple(params)


def game_saga_service_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed('base', module.base_service),
        typed('shared', module.branch_shared_alt),
        typed('story_queries', 'GameStoryQueries'),
        typed('character_data', 'GameCharacterData'),
    ]
    if scenario == 'env-sensitive':
        params = [typed('saga_id', env_int('game')), *params]
    return tuple(params)


def module_service_classes(
    module: ModuleShape, scenario: Scenario
) -> tuple[ClassSpec, ...]:
    classes: list[ClassSpec] = [
        ClassSpec(module.base_service, base_service_params(module)),
        ClassSpec(module.write_service, write_service_params(module)),
    ]
    if module.session or module.branch_only:
        classes.append(
            ClassSpec(module.transaction_executor, transaction_executor_params(module))
        )
        classes.extend(
            ClassSpec(name, handler_params(module)) for name in module.handler_classes
        )
        classes.extend([
            ClassSpec(module.handler_refs, handler_refs_params(module)),
            ClassSpec(module.branch_audit, branch_audit_params(module)),
            ClassSpec(module.branch_service, branch_service_params(module, scenario)),
            ClassSpec(
                module.branch_write_service,
                branch_write_service_params(module),
            ),
            ClassSpec(module.message_service, message_service_params(module, scenario)),
        ])
    if module.session:
        classes.append(
            ClassSpec(module.session_service, session_service_params(module, scenario))
        )
    match module.extra_transition:
        case 'agent':
            classes.extend([
                ClassSpec('AiGameQueries', ai_game_queries_params(module, scenario)),
                ClassSpec('AiStoryQueries', ai_story_queries_params(module, scenario)),
                ClassSpec(
                    'AiCharacterQueries', ai_character_queries_params(module, scenario)
                ),
                ClassSpec(
                    module.ai_agent_service, ai_agent_service_params(module, scenario)
                ),
                ClassSpec(
                    module.ai_tool_service, ai_tool_service_params(module, scenario)
                ),
            ])
        case 'saga':
            classes.extend([
                ClassSpec(
                    'GameStoryQueries',
                    game_story_queries_params(module, scenario),
                ),
                ClassSpec(
                    'GameCharacterData',
                    game_character_data_params(module, scenario),
                ),
                ClassSpec(
                    module.game_saga_service,
                    game_saga_service_params(module, scenario),
                ),
            ])
        case 'none':
            pass
    return tuple(classes)


def module_protocols(module: ModuleShape) -> tuple[ProtocolSpec, ...]:
    protocols: list[ProtocolSpec] = [
        ProtocolSpec(
            module.write_context,
            (
                typed('write', module.write_service),
                typed('base', module.base_service),
            ),
            (),
        ),
        ProtocolSpec(
            module.context,
            (typed('base', module.base_service),),
            tuple(_context_methods(module)),
        ),
    ]
    if module.session:
        protocols.extend([
            ProtocolSpec(
                module.session_context,
                (typed('session', module.session_service),),
                tuple(_session_methods(module)),
            ),
            ProtocolSpec(
                module.branch_context,
                _branch_fields(module),
                (
                    MethodSpec(
                        'with_branched_write',
                        (),
                        write_context(module.branch_write_context),
                    ),
                ),
            ),
            ProtocolSpec(
                module.branch_write_context,
                _branch_write_fields(module),
                tuple(_branch_write_methods(module)),
            ),
            ProtocolSpec(
                module.message_context,
                (
                    typed('message', module.message_service),
                    typed('audit', module.branch_audit),
                ),
                (
                    MethodSpec(
                        'with_handler', (), write_context(module.branch_write_context)
                    ),
                ),
            ),
        ])
    if module.branch_only:
        protocols.extend([
            ProtocolSpec(
                module.branch_context,
                _branch_fields(module),
                (
                    MethodSpec(
                        'with_branched_write',
                        (),
                        write_context(module.branch_write_context),
                    ),
                ),
            ),
            ProtocolSpec(
                module.branch_write_context,
                _branch_write_fields(module),
                (),
            ),
            ProtocolSpec(
                module.message_context,
                (
                    typed('message', module.message_service),
                    typed('audit', module.branch_audit),
                ),
                (
                    MethodSpec(
                        'with_handler', (), write_context(module.branch_write_context)
                    ),
                ),
            ),
        ])
    match module.extra_transition:
        case 'agent':
            protocols.extend([
                ProtocolSpec(
                    module.ai_agent_context,
                    (
                        typed('agent', module.ai_agent_service),
                        typed('story_queries', 'AiStoryQueries'),
                    ),
                    (
                        MethodSpec(
                            'with_tool',
                            (typed('tool_call_id', 'int'),),
                            write_context(module.ai_tool_context),
                        ),
                    ),
                ),
                ProtocolSpec(
                    module.ai_tool_context,
                    (
                        typed('tool', module.ai_tool_service),
                        typed('character_queries', 'AiCharacterQueries'),
                    ),
                    (),
                ),
            ])
        case 'saga':
            protocols.append(
                ProtocolSpec(
                    module.game_saga_context,
                    (
                        typed('saga', module.game_saga_service),
                        typed('character_data', 'GameCharacterData'),
                    ),
                    (),
                )
            )
        case 'none':
            pass
    return tuple(protocols)


def _context_methods(module: ModuleShape) -> list[MethodSpec]:
    methods = [MethodSpec('with_write', (), write_context(module.write_context))]
    if module.session:
        methods.append(
            MethodSpec(
                'with_session',
                (typed('session_id', 'int'),),
                module.session_context,
            )
        )
        methods.extend(
            MethodSpec(f'route_{index}', (), module.session_context)
            for index in module.route_indices
        )
        methods.append(
            MethodSpec(
                'with_message_scope',
                (
                    typed('message_id', 'int'),
                    typed('channel_name', 'str'),
                    typed('session_id', 'int'),
                    typed('branch_id', 'int'),
                    typed('vector_clock', 'int'),
                ),
                module.message_context,
            )
        )
    if module.branch_only:
        methods.extend([
            MethodSpec(
                'with_branched_session',
                (typed('branch_id', 'int'), typed('vector_clock', 'int')),
                module.branch_context,
            ),
            MethodSpec(
                'with_message_scope',
                (
                    typed('message_id', 'int'),
                    typed('channel_name', 'str'),
                    typed('session_id', 'int'),
                    typed('branch_id', 'int'),
                    typed('vector_clock', 'int'),
                ),
                module.message_context,
            ),
        ])
    return methods


def _session_methods(module: ModuleShape) -> list[MethodSpec]:
    methods = [
        MethodSpec('with_write', (), write_context(module.write_context)),
        MethodSpec(
            'with_branched_session',
            (typed('branch_id', 'int'), typed('vector_clock', 'int')),
            module.branch_context,
        ),
    ]
    methods.extend(
        MethodSpec(f'branch_{index}', (), module.branch_context)
        for index in module.branch_indices
    )
    return methods


def _branch_fields(module: ModuleShape) -> tuple[TypedName, ...]:
    return (
        typed('branch', module.branch_service),
        typed('executor', module.transaction_executor),
        typed('audit', module.branch_audit),
        *tuple(
            typed(f'extra_{index}', extra)
            for index, extra in enumerate(module.read_extras)
        ),
    )


def _branch_write_fields(module: ModuleShape) -> tuple[TypedName, ...]:
    return (
        typed('branch_write', module.branch_write_service),
        typed('handler_refs', module.handler_refs),
        *tuple(
            typed(f'extra_{index}', extra)
            for index, extra in enumerate(module.write_extras)
        ),
    )


def _branch_write_methods(module: ModuleShape) -> list[MethodSpec]:
    match module.extra_transition:
        case 'agent':
            return [
                MethodSpec(
                    'with_agent',
                    (typed('agent_id', 'int'),),
                    write_context(module.ai_agent_context),
                )
            ]
        case 'saga':
            return [
                MethodSpec(
                    'with_saga',
                    (typed('saga_id', 'int'),),
                    write_context(module.game_saga_context),
                )
            ]
        case 'none':
            return []


def root_protocols(modules: tuple[ModuleShape, ...]) -> tuple[ProtocolSpec, ...]:
    return (
        ProtocolSpec(
            'AppContext',
            (),
            tuple(
                MethodSpec(f'with_{module.name}_context', (), module.qualified_context)
                for module in modules
            ),
        ),
        ProtocolSpec(
            'BenchmarkRoot',
            (typed('app', 'AppContext'),),
            (),
        ),
    )


def unique_names(*names: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(names))


def module_scopes(module: str) -> str:
    return f"qual('{module}') | (qual('{module}') & qual('write'))"


def module_functions(module: ModuleShape) -> tuple[FunctionSpec, ...]:
    functions = [
        FunctionSpec(
            f'app_with_{module.name}_context',
            (),
            module.module_state,
            (f'return {module.module_state}()',),
        ),
        FunctionSpec(
            f'{module.name}_with_write',
            (),
            module.write_state,
            (f'return {module.write_state}()',),
        ),
    ]
    if module.session:
        functions.extend([
            FunctionSpec(
                f'{module.name}_with_session',
                (typed('session_id', env_int(module.name)),),
                module.session_state,
                ("return {'session_id': session_id}",),
            ),
            FunctionSpec(
                f'{module.name}_with_branched_session',
                (
                    typed('branch_id', env_int(module.name)),
                    typed('vector_clock', env_int(module.name)),
                ),
                module.branch_state,
                ("return {'branch_id': branch_id, 'vector_clock': vector_clock}",),
            ),
            FunctionSpec(
                f'{module.name}_with_message_scope',
                (
                    typed('message_id', env_int(module.name)),
                    typed('channel_name', env_str(module.name)),
                    typed('session_id', env_int(module.name)),
                    typed('branch_id', env_int(module.name)),
                    typed('vector_clock', env_int(module.name)),
                ),
                module.message_state,
                (
                    "return {'message_id': message_id, 'channel_name': channel_name, "
                    + "'session_id': session_id, 'branch_id': branch_id, "
                    + "'vector_clock': vector_clock}",
                ),
            ),
            FunctionSpec(
                f'{module.name}_with_handler',
                (),
                module.handler_state,
                (f'return {module.handler_state}()',),
            ),
            FunctionSpec(
                f'{module.name}_with_branched_write',
                (),
                module.write_state,
                (f'return {module.write_state}()',),
            ),
        ])
        functions.extend(
            FunctionSpec(
                f'{module.name}_route_{index}',
                (),
                module.session_state,
                (f"return {{'session_id': {index + 1}}}",),
            )
            for index in module.route_indices
        )
        functions.extend(
            FunctionSpec(
                f'{module.name}_branch_{index}',
                (),
                module.branch_state,
                (f"return {{'branch_id': {index + 1}, 'vector_clock': {index + 1}}}",),
            )
            for index in module.branch_indices
        )
    if module.branch_only:
        functions.extend([
            FunctionSpec(
                f'{module.name}_with_branched_session',
                (
                    typed('branch_id', env_int(module.name)),
                    typed('vector_clock', env_int(module.name)),
                ),
                module.branch_state,
                ("return {'branch_id': branch_id, 'vector_clock': vector_clock}",),
            ),
            FunctionSpec(
                f'{module.name}_with_message_scope',
                (
                    typed('message_id', env_int(module.name)),
                    typed('channel_name', env_str(module.name)),
                    typed('session_id', env_int(module.name)),
                    typed('branch_id', env_int(module.name)),
                    typed('vector_clock', env_int(module.name)),
                ),
                module.message_state,
                (
                    "return {'message_id': message_id, 'channel_name': channel_name, "
                    + "'session_id': session_id, 'branch_id': branch_id, "
                    + "'vector_clock': vector_clock}",
                ),
            ),
            FunctionSpec(
                f'{module.name}_with_handler',
                (),
                module.handler_state,
                (f'return {module.handler_state}()',),
            ),
            FunctionSpec(
                f'{module.name}_with_branched_write',
                (),
                module.write_state,
                (f'return {module.write_state}()',),
            ),
        ])
    match module.extra_transition:
        case 'agent':
            functions.extend([
                FunctionSpec(
                    'ai_with_agent',
                    (typed('agent_id', env_int('ai')),),
                    module.ai_agent_state,
                    ("return {'agent_id': agent_id}",),
                ),
                FunctionSpec(
                    'ai_with_tool',
                    (typed('tool_call_id', env_int('ai')),),
                    module.ai_tool_state,
                    ("return {'tool_call_id': tool_call_id}",),
                ),
            ])
        case 'saga':
            functions.append(
                FunctionSpec(
                    'game_with_saga',
                    (typed('saga_id', env_int('game')),),
                    module.game_saga_state,
                    ("return {'saga_id': saga_id}",),
                )
            )
        case 'none':
            pass
    return tuple(functions)


def provider_body(return_type: str, params: tuple[TypedName, ...]) -> tuple[str, ...]:
    args = ', '.join(param.name for param in params)
    return (f'return {return_type}({args})',)


def root_provider_functions(
    modules: tuple[ModuleShape, ...],
    scenario: Scenario,
) -> tuple[FunctionSpec, ...]:
    functions: list[FunctionSpec] = []
    for module in modules:
        match module.extra_transition:
            case 'agent':
                ai_game_params = ai_game_queries_params(module, scenario)
                ai_story_params = ai_story_queries_params(module, scenario)
                ai_character_params = ai_character_queries_params(module, scenario)
                functions.extend([
                    FunctionSpec(
                        'provide_ai_game_queries',
                        ai_game_params,
                        'AiGameQueries',
                        provider_body('AiGameQueries', ai_game_params),
                    ),
                    FunctionSpec(
                        'provide_ai_story_queries',
                        ai_story_params,
                        'AiStoryQueries',
                        provider_body('AiStoryQueries', ai_story_params),
                    ),
                    FunctionSpec(
                        'provide_ai_character_queries',
                        ai_character_params,
                        'AiCharacterQueries',
                        provider_body('AiCharacterQueries', ai_character_params),
                    ),
                ])
            case 'saga':
                game_story_params = game_story_queries_params(module, scenario)
                game_character_params = game_character_data_params(module, scenario)
                functions.extend([
                    FunctionSpec(
                        'provide_game_story_queries',
                        game_story_params,
                        'GameStoryQueries',
                        provider_body('GameStoryQueries', game_story_params),
                    ),
                    FunctionSpec(
                        'provide_game_character_data',
                        game_character_params,
                        'GameCharacterData',
                        provider_body('GameCharacterData', game_character_params),
                    ),
                ])
            case 'none':
                pass
    return tuple(functions)


def app_hooks(modules: tuple[ModuleShape, ...]) -> tuple[HookSpec, ...]:
    hooks: list[HookSpec] = []
    for module in modules:
        for hook_index in module.hook_indices:
            hooks.append(
                HookSpec(
                    protocol_name='AppContext',
                    method_name=f'with_{module.name}_context',
                    function_name=f'app_with_{module.name}_context_hook_{hook_index}',
                    params=(typed('shared', module.hook_shared),),
                )
            )
    return tuple(hooks)


def module_hooks(module: ModuleShape) -> tuple[HookSpec, ...]:
    hooks: list[HookSpec] = []
    if module.session:
        for hook_index in module.hook_indices:
            hooks.append(
                HookSpec(
                    module.context,
                    'with_session',
                    f'{module.name}_with_session_hook_{hook_index}',
                    (
                        typed('session_id', env_int(module.name)),
                        typed('shared', module.hook_shared),
                    ),
                )
            )
            hooks.append(
                HookSpec(
                    module.session_context,
                    'with_branched_session',
                    f'{module.name}_with_branched_session_hook_{hook_index}',
                    (
                        typed('branch_id', env_int(module.name)),
                        typed('vector_clock', env_int(module.name)),
                        typed('shared', module.hook_shared_alt),
                    ),
                )
            )
            hooks.append(
                HookSpec(
                    module.context,
                    'with_message_scope',
                    f'{module.name}_with_message_scope_hook_{hook_index}',
                    (
                        typed('message_id', env_int(module.name)),
                        typed('session_id', env_int(module.name)),
                        typed('branch_id', env_int(module.name)),
                        typed('shared', module.hook_shared),
                    ),
                )
            )
            hooks.append(
                HookSpec(
                    module.message_context,
                    'with_handler',
                    f'{module.name}_with_handler_hook_{hook_index}',
                    (typed('shared', module.hook_shared_alt),),
                )
            )
        for route_index in module.route_indices:
            for hook_index in module.hook_indices:
                hooks.append(
                    HookSpec(
                        module.context,
                        f'route_{route_index}',
                        f'{module.name}_route_{route_index}_hook_{hook_index}',
                        (typed('shared', module.hook_shared),),
                    )
                )
        for branch_index in module.branch_indices:
            for hook_index in module.hook_indices:
                hooks.append(
                    HookSpec(
                        module.session_context,
                        f'branch_{branch_index}',
                        f'{module.name}_branch_{branch_index}_hook_{hook_index}',
                        (typed('shared', module.hook_shared_alt),),
                    )
                )
    if module.branch_only:
        for hook_index in module.hook_indices:
            hooks.extend([
                HookSpec(
                    module.context,
                    'with_branched_session',
                    f'{module.name}_with_branched_session_hook_{hook_index}',
                    (
                        typed('branch_id', env_int(module.name)),
                        typed('vector_clock', env_int(module.name)),
                        typed('shared', module.hook_shared),
                    ),
                ),
                HookSpec(
                    module.context,
                    'with_message_scope',
                    f'{module.name}_with_message_scope_hook_{hook_index}',
                    (
                        typed('message_id', env_int(module.name)),
                        typed('session_id', env_int(module.name)),
                        typed('branch_id', env_int(module.name)),
                        typed('shared', module.hook_shared_alt),
                    ),
                ),
                HookSpec(
                    module.message_context,
                    'with_handler',
                    f'{module.name}_with_handler_hook_{hook_index}',
                    (typed('shared', module.hook_shared),),
                ),
            ])
    match module.extra_transition:
        case 'agent':
            for hook_index in module.hook_indices:
                hooks.extend([
                    HookSpec(
                        module.branch_write_context,
                        'with_agent',
                        f'ai_with_agent_hook_{hook_index}',
                        (
                            typed('agent_id', env_int('ai')),
                            typed('shared', module.hook_shared),
                        ),
                    ),
                    HookSpec(
                        module.ai_agent_context,
                        'with_tool',
                        f'ai_with_tool_hook_{hook_index}',
                        (
                            typed('tool_call_id', env_int('ai')),
                            typed('shared', module.hook_shared_alt),
                        ),
                    ),
                ])
        case 'saga':
            for hook_index in module.hook_indices:
                hooks.append(
                    HookSpec(
                        module.branch_write_context,
                        'with_saga',
                        f'game_with_saga_hook_{hook_index}',
                        (
                            typed('saga_id', env_int('game')),
                            typed('shared', module.hook_shared),
                        ),
                    )
                )
        case 'none':
            pass
    return tuple(hooks)


def module_methods(module: ModuleShape) -> tuple[MethodRegistrationSpec, ...]:
    methods = [
        MethodRegistrationSpec(
            module.context, 'with_write', f'{module.name}_with_write'
        ),
    ]
    if module.session:
        methods.extend([
            MethodRegistrationSpec(
                module.context,
                'with_session',
                f'{module.name}_with_session',
            ),
            MethodRegistrationSpec(
                module.context,
                'with_message_scope',
                f'{module.name}_with_message_scope',
            ),
            MethodRegistrationSpec(
                module.session_context,
                'with_write',
                f'{module.name}_with_write',
            ),
            MethodRegistrationSpec(
                module.session_context,
                'with_branched_session',
                f'{module.name}_with_branched_session',
            ),
            MethodRegistrationSpec(
                module.branch_context,
                'with_branched_write',
                f'{module.name}_with_branched_write',
            ),
            MethodRegistrationSpec(
                module.message_context,
                'with_handler',
                f'{module.name}_with_handler',
            ),
        ])
        methods.extend(
            MethodRegistrationSpec(
                module.context,
                f'route_{index}',
                f'{module.name}_route_{index}',
            )
            for index in module.route_indices
        )
        methods.extend(
            MethodRegistrationSpec(
                module.session_context,
                f'branch_{index}',
                f'{module.name}_branch_{index}',
            )
            for index in module.branch_indices
        )
    if module.branch_only:
        methods.extend([
            MethodRegistrationSpec(
                module.context,
                'with_branched_session',
                f'{module.name}_with_branched_session',
            ),
            MethodRegistrationSpec(
                module.context,
                'with_message_scope',
                f'{module.name}_with_message_scope',
            ),
            MethodRegistrationSpec(
                module.message_context,
                'with_handler',
                f'{module.name}_with_handler',
            ),
            MethodRegistrationSpec(
                module.branch_context,
                'with_branched_write',
                f'{module.name}_with_branched_write',
            ),
        ])
    match module.extra_transition:
        case 'agent':
            methods.extend([
                MethodRegistrationSpec(
                    module.branch_write_context,
                    'with_agent',
                    'ai_with_agent',
                ),
                MethodRegistrationSpec(
                    module.ai_agent_context,
                    'with_tool',
                    'ai_with_tool',
                ),
            ])
        case 'saga':
            methods.append(
                MethodRegistrationSpec(
                    module.branch_write_context,
                    'with_saga',
                    'game_with_saga',
                )
            )
        case 'none':
            pass
    return tuple(methods)


def module_common_classes(module: ModuleShape) -> tuple[str, ...]:
    names = [
        module.module_state,
        module.write_state,
        module.base_service,
    ]
    if module.session or module.branch_only:
        names.extend([
            module.handler_state,
            module.branch_state,
            module.message_state,
            module.transaction_executor,
            *module.handler_classes,
            module.handler_refs,
            module.branch_audit,
            module.branch_service,
            module.message_service,
        ])
    if module.session:
        names.extend([module.session_state, module.session_service])
    match module.extra_transition:
        case 'agent':
            names.extend([module.ai_agent_state, module.ai_tool_state])
        case 'saga':
            names.extend([module.game_saga_state])
        case 'none':
            pass
    return unique_names(*names)


def module_write_classes(module: ModuleShape) -> tuple[str, ...]:
    names = [module.write_service]
    if module.session or module.branch_only:
        names.append(module.branch_write_service)
    match module.extra_transition:
        case 'agent':
            names.extend([module.ai_agent_service, module.ai_tool_service])
        case 'saga':
            names.append(module.game_saga_service)
        case 'none':
            pass
    return unique_names(*names)


def registries(modules: tuple[ModuleShape, ...]) -> tuple[ModuleRegistrySpec, ...]:
    return tuple(
        ModuleRegistrySpec(
            module=module,
            common_classes=module_common_classes(module),
            write_classes=module_write_classes(module),
            methods=module_methods(module),
            hooks=module_hooks(module),
        )
        for module in modules
    )


def root_constructor_registrations(
    modules: tuple[ModuleShape, ...],
) -> tuple[ConstructorRegistrationSpec, ...]:
    registrations: list[ConstructorRegistrationSpec] = []
    for module in modules:
        match module.extra_transition:
            case 'agent':
                registrations.extend([
                    ConstructorRegistrationSpec(
                        'AiGameQueries',
                        module_scopes('ai'),
                        'provide_ai_game_queries',
                    ),
                    ConstructorRegistrationSpec(
                        'AiStoryQueries',
                        module_scopes('ai'),
                        'provide_ai_story_queries',
                    ),
                    ConstructorRegistrationSpec(
                        'AiCharacterQueries',
                        module_scopes('ai'),
                        'provide_ai_character_queries',
                    ),
                ])
            case 'saga':
                registrations.extend([
                    ConstructorRegistrationSpec(
                        'GameStoryQueries',
                        module_scopes('game'),
                        'provide_game_story_queries',
                    ),
                    ConstructorRegistrationSpec(
                        'GameCharacterData',
                        module_scopes('game'),
                        'provide_game_character_data',
                    ),
                ])
            case 'none':
                pass
    return tuple(registrations)


def render_source(
    *,
    scenario: Scenario,
    density: int,
    handler_count: int,
    shared_slots: int,
    enabled_modules: tuple[str, ...],
) -> str:
    modules = modules_for(density, handler_count, shared_slots, enabled_modules)
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(TEMPLATE_NAME)
    class_specs = [*shared_classes(shared_slots)]
    typed_dict_specs: list[TypedDictSpec] = []
    protocol_specs = [*root_protocols(modules)]
    function_specs: list[FunctionSpec] = []
    for module in modules:
        class_specs.extend(module_states(module))
        typed_dict_specs.extend(module_typed_dicts(module, scenario))
        class_specs.extend(module_service_classes(module, scenario))
        protocol_specs.extend(module_protocols(module))
        function_specs.extend(module_functions(module))
    function_specs.extend(root_provider_functions(modules, scenario))
    rendered = template.render(
        class_specs=class_specs,
        typed_dict_specs=typed_dict_specs,
        protocol_specs=protocol_specs,
        function_specs=function_specs,
        global_classes=tuple(spec.name for spec in shared_classes(shared_slots)),
        root_constructor_registrations=root_constructor_registrations(modules),
        root_methods=tuple(
            MethodRegistrationSpec(
                'AppContext',
                f'with_{module.name}_context',
                f'app_with_{module.name}_context',
            )
            for module in modules
        ),
        root_hooks=app_hooks(modules),
        registries=registries(modules),
    )
    return rendered


def build_generated(
    *,
    scenario: Scenario,
    density: int,
    handler_count: int,
    shared_slots: int,
    enabled_modules: tuple[str, ...],
) -> tuple[type[object], Registry, str]:
    source = render_source(
        scenario=scenario,
        density=density,
        handler_count=handler_count,
        shared_slots=shared_slots,
        enabled_modules=enabled_modules,
    )
    namespace: dict[str, object] = {}
    exec(source, namespace)
    return (
        cast(type[object], namespace['BenchmarkRoot']),
        cast(Registry, namespace['REGISTRY']),
        source,
    )


def run_once(
    *,
    scenario: Scenario,
    density: int,
    handler_count: int,
    shared_slots: int,
    enabled_modules: tuple[str, ...],
    invoke: bool,
    dump_generated: bool,
) -> None:
    generation_started = perf_counter()
    root_type, registry, source = build_generated(
        scenario=scenario,
        density=density,
        handler_count=handler_count,
        shared_slots=shared_slots,
        enabled_modules=enabled_modules,
    )
    generation_elapsed = perf_counter() - generation_started

    if dump_generated:
        print(source)

    compile_started = perf_counter()
    root = cast(BenchmarkRootContext, compile(root_type, registry, default_rules()))
    compile_elapsed = perf_counter() - compile_started

    invoke_elapsed: float | None = None
    if invoke:
        invoke_started = perf_counter()
        app = root.app

        if 'chat' in enabled_modules:
            chat_branch = app.with_chat_context().route_0().branch_0()
            _ = chat_branch.branch
            _ = chat_branch.with_branched_write().branch_write

        if 'game' in enabled_modules:
            game_branch_write = (
                app.with_game_context().route_0().branch_0().with_branched_write()
            )
            _ = game_branch_write.with_saga(1).saga

        if 'ai' in enabled_modules:
            ai_branch_write = (
                app.with_ai_context().route_0().branch_0().with_branched_write()
            )
            _ = ai_branch_write.with_agent(1).with_tool(1).tool

        if 'character' in enabled_modules:
            character_write = (
                app
                .with_character_context()
                .with_message_scope(1, 'events', 1, 1, 1)
                .with_handler()
            )
            _ = character_write.branch_write

        if 'story' in enabled_modules:
            story_write = app.with_story_context().with_write()
            _ = story_write.write
        invoke_elapsed = perf_counter() - invoke_started

    print(
        ' '.join([
            'benchmark=production_shape',
            f'scenario={scenario}',
            f'density={density}',
            f'handlers={handler_count}',
            f'shared_slots={shared_slots}',
            f'modules={",".join(enabled_modules)}',
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
    parser.add_argument('--density', type=int, default=4)
    parser.add_argument('--handlers', type=int, default=4)
    parser.add_argument('--shared-slots', type=int, default=2)
    parser.add_argument(
        '--modules',
        default='chat,game,ai,story,character',
    )
    parser.add_argument('--invoke', action='store_true')
    parser.add_argument('--dump-generated', action='store_true')
    args = parser.parse_args()

    enabled_modules = tuple(
        module.strip()
        for module in args.modules.split(',')
        if module.strip() in {'chat', 'game', 'ai', 'story', 'character'}
    )
    assert enabled_modules, 'Expected at least one benchmark module'

    scenarios: list[Scenario]
    if args.scenario == 'all':
        scenarios = ['portable', 'env-sensitive']
    else:
        scenarios = [args.scenario]

    for scenario in scenarios:
        run_once(
            scenario=scenario,
            density=args.density,
            handler_count=args.handlers,
            shared_slots=args.shared_slots,
            enabled_modules=enabled_modules,
            invoke=args.invoke,
            dump_generated=args.dump_generated,
        )


if __name__ == '__main__':
    main()
