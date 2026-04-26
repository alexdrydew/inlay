import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol, cast

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from inlay import Registry, compile
from inlay.default import default_rules

type Scenario = Literal['portable', 'env-sensitive']
type ModuleKind = Literal['entry', 'scope', 'write']
type ExtraTransition = Literal['none', 'worker', 'flow']

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
    scope_indices: tuple[int, ...]
    handler_indices: tuple[int, ...]
    hook_indices: tuple[int, ...]

    @property
    def entry(self) -> bool:
        return self.kind == 'entry'

    @property
    def scope_only(self) -> bool:
        return self.kind == 'scope'

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
    def entry_shared(self) -> str:
        return f'EntryShared{self.slot}'

    @property
    def scope_shared(self) -> str:
        return f'ScopeShared{self.slot}'

    @property
    def scope_shared_alt(self) -> str:
        return f'ScopeShared{self.alt_slot}'

    @property
    def event_shared(self) -> str:
        return f'EventShared{self.alt_slot}'

    @property
    def entry_state(self) -> str:
        return f'{self.title}EntryState'

    @property
    def scope_state(self) -> str:
        return f'{self.title}ScopeState'

    @property
    def event_state(self) -> str:
        return f'{self.title}EventState'

    @property
    def handler_state(self) -> str:
        return f'{self.title}HandlerState'

    @property
    def entry_service(self) -> str:
        return f'{self.title}EntryService'

    @property
    def entry_context(self) -> str:
        return f'{self.title}EntryReadContext'

    @property
    def scope_service(self) -> str:
        return f'{self.title}ScopeService'

    @property
    def scope_audit(self) -> str:
        return f'{self.title}ScopeAudit'

    @property
    def scope_context(self) -> str:
        return f'{self.title}ScopedReadContext'

    @property
    def scope_write_service(self) -> str:
        return f'{self.title}ScopeWriteService'

    @property
    def scope_write_context(self) -> str:
        return f'{self.title}ScopedWriteContext'

    @property
    def event_service(self) -> str:
        return f'{self.title}EventService'

    @property
    def event_context(self) -> str:
        return f'{self.title}EventContext'

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
    def worker_state(self) -> str:
        return 'WorkerState'

    @property
    def worker_step_state(self) -> str:
        return 'WorkerStepState'

    @property
    def worker_service(self) -> str:
        return 'WorkerService'

    @property
    def worker_step_service(self) -> str:
        return 'WorkerStepService'

    @property
    def worker_context(self) -> str:
        return 'WorkerWriteContext'

    @property
    def worker_step_context(self) -> str:
        return 'WorkerStepWriteContext'

    @property
    def flow_state(self) -> str:
        return 'FlowState'

    @property
    def flow_service(self) -> str:
        return 'FlowService'

    @property
    def flow_context(self) -> str:
        return 'FlowWriteContext'

    @property
    def read_extras(self) -> tuple[str, ...]:
        match self.name:
            case 'gamma':
                return ('GammaAlphaQueries', 'GammaDeltaQueries', 'GammaEpsilonQueries')
            case 'beta':
                return ('BetaDeltaQueries', 'BetaEpsilonData')
            case _:
                return ()

    @property
    def write_extras(self) -> tuple[str, ...]:
        match self.name:
            case 'gamma':
                return ('GammaAlphaQueries', 'GammaDeltaQueries')
            case 'beta':
                return ('BetaDeltaQueries', 'BetaEpsilonData')
            case _:
                return ()


@dataclass(frozen=True)
class ModuleRegistrySpec:
    module: ModuleShape
    common_classes: tuple[str, ...]
    write_classes: tuple[str, ...]
    methods: tuple[MethodRegistrationSpec, ...]
    hooks: tuple[HookSpec, ...]


class BenchmarkScopeWriteContext(Protocol):
    scope_write: object


class BenchmarkScopeContext(Protocol):
    scope: object

    def with_scoped_write(self) -> BenchmarkScopeWriteContext: ...


class BenchmarkEntryContext(Protocol):
    def transition_0(self) -> BenchmarkScopeContext: ...


class BenchmarkFlowContext(Protocol):
    flow: object


class BenchmarkBetaScopeWriteContext(Protocol):
    def with_flow(self, flow_id: int) -> BenchmarkFlowContext: ...


class BenchmarkBetaScopeContext(Protocol):
    scope: object

    def with_scoped_write(self) -> BenchmarkBetaScopeWriteContext: ...


class BenchmarkBetaEntryContext(Protocol):
    def transition_0(self) -> BenchmarkBetaScopeContext: ...


class BenchmarkWorkerStepContext(Protocol):
    step: object


class BenchmarkWorkerContext(Protocol):
    def with_step(self, step_id: int) -> BenchmarkWorkerStepContext: ...


class BenchmarkGammaScopeWriteContext(Protocol):
    def with_worker(self, worker_id: int) -> BenchmarkWorkerContext: ...


class BenchmarkGammaScopeContext(Protocol):
    scope: object

    def with_scoped_write(self) -> BenchmarkGammaScopeWriteContext: ...


class BenchmarkGammaEntryContext(Protocol):
    def transition_0(self) -> BenchmarkGammaScopeContext: ...


class BenchmarkEventContext(Protocol):
    event: object

    def with_handler(self) -> BenchmarkScopeWriteContext: ...


class BenchmarkEpsilonContext(Protocol):
    def with_event_scope(
        self,
        event_id: int,
        group_name: str,
        entry_id: int,
        scope_id: int,
        revision: int,
    ) -> BenchmarkEventContext: ...


class BenchmarkWriteContext(Protocol):
    write: object


class BenchmarkDeltaContext(Protocol):
    def with_write(self) -> BenchmarkWriteContext: ...


class BenchmarkAlphaContext(Protocol):
    def with_write(self) -> BenchmarkWriteContext: ...

    def route_0(self) -> BenchmarkEntryContext: ...


class BenchmarkBetaContext(Protocol):
    def with_write(self) -> BenchmarkWriteContext: ...

    def route_0(self) -> BenchmarkBetaEntryContext: ...


class BenchmarkGammaContext(Protocol):
    def with_write(self) -> BenchmarkWriteContext: ...

    def route_0(self) -> BenchmarkGammaEntryContext: ...


class BenchmarkAppContext(Protocol):
    def with_alpha_context(self) -> BenchmarkAlphaContext: ...

    def with_beta_context(self) -> BenchmarkBetaContext: ...

    def with_gamma_context(self) -> BenchmarkGammaContext: ...

    def with_epsilon_context(self) -> BenchmarkEpsilonContext: ...

    def with_delta_context(self) -> BenchmarkDeltaContext: ...


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
        ('alpha', 'entry', 'none'),
        ('beta', 'entry', 'flow'),
        ('gamma', 'entry', 'worker'),
        ('delta', 'write', 'none'),
        ('epsilon', 'scope', 'none'),
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
                scope_indices=indices,
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
            ClassSpec(f'EntryShared{slot}', (typed('base', f'BaseShared{slot}'),)),
            ClassSpec(f'ScopeShared{slot}', (typed('base', f'BaseShared{slot}'),)),
            ClassSpec(f'EventShared{slot}', (typed('base', f'BaseShared{slot}'),)),
            ClassSpec(f'HookShared{slot}', (typed('base', f'BaseShared{slot}'),)),
        ])
    return tuple(classes)


def module_states(module: ModuleShape) -> tuple[ClassSpec, ...]:
    classes = [ClassSpec(module.module_state, ()), ClassSpec(module.write_state, ())]
    if module.entry or module.scope_only:
        classes.append(ClassSpec(module.handler_state, ()))
    return tuple(classes)


def module_typed_dicts(module: ModuleShape) -> tuple[TypedDictSpec, ...]:
    typed_dicts: list[TypedDictSpec] = []
    if module.entry:
        typed_dicts.extend([
            TypedDictSpec(
                module.entry_state, (typed('entry_id', env_int(module.name)),)
            ),
            TypedDictSpec(
                module.scope_state,
                (
                    typed('scope_id', env_int(module.name)),
                    typed('revision', env_int(module.name)),
                ),
            ),
            TypedDictSpec(
                module.event_state,
                (
                    typed('event_id', env_int(module.name)),
                    typed('group_name', env_str(module.name)),
                    typed('entry_id', env_int(module.name)),
                    typed('scope_id', env_int(module.name)),
                    typed('revision', env_int(module.name)),
                ),
            ),
        ])
    if module.scope_only:
        typed_dicts.extend([
            TypedDictSpec(
                module.scope_state,
                (
                    typed('scope_id', env_int(module.name)),
                    typed('revision', env_int(module.name)),
                ),
            ),
            TypedDictSpec(
                module.event_state,
                (
                    typed('event_id', env_int(module.name)),
                    typed('group_name', env_str(module.name)),
                    typed('entry_id', env_int(module.name)),
                    typed('scope_id', env_int(module.name)),
                    typed('revision', env_int(module.name)),
                ),
            ),
        ])
    match module.extra_transition:
        case 'worker':
            typed_dicts.extend([
                TypedDictSpec(
                    module.worker_state,
                    (typed('worker_id', env_int('gamma')),),
                ),
                TypedDictSpec(
                    module.worker_step_state,
                    (typed('step_id', env_int('gamma')),),
                ),
            ])
        case 'flow':
            typed_dicts.append(
                TypedDictSpec(module.flow_state, (typed('flow_id', env_int('beta')),))
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
    return (typed('shared', module.scope_shared),)


def handler_refs_params(module: ModuleShape) -> tuple[TypedName, ...]:
    return (typed('shared', module.scope_shared),)


def handler_params(module: ModuleShape) -> tuple[TypedName, ...]:
    return (
        typed('shared', module.hook_shared),
        typed('executor', module.transaction_executor),
    )


def scope_audit_params(module: ModuleShape) -> tuple[TypedName, ...]:
    return (
        typed('shared', module.scope_shared_alt),
        typed('executor', module.transaction_executor),
        typed('handler_refs', module.handler_refs),
    )


def entry_service_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed('base', module.base_service),
        typed('shared', module.entry_shared),
    ]
    if scenario == 'env-sensitive':
        params.insert(0, typed('entry_id', env_int(module.name)))
    return tuple(params)


def scope_service_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed('base', module.base_service),
        typed('shared', module.scope_shared),
        typed('audit', module.scope_audit),
        typed('executor', module.transaction_executor),
        typed('handler_refs', module.handler_refs),
    ]
    if module.entry:
        params.insert(1, typed('entry', module.entry_service))
    if scenario == 'env-sensitive':
        params = [
            typed('scope_id', env_int(module.name)),
            typed('revision', env_int(module.name)),
            *params,
        ]
    return tuple(params)


def scope_write_service_params(module: ModuleShape) -> tuple[TypedName, ...]:
    return (
        typed('scope', module.scope_service),
        typed('shared', module.scope_shared_alt),
        typed('handler_refs', module.handler_refs),
    )


def event_service_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed('base', module.base_service),
        typed('shared', module.event_shared),
        typed('audit', module.scope_audit),
        typed('handler_refs', module.handler_refs),
    ]
    if scenario == 'env-sensitive':
        params = [
            typed('event_id', env_int(module.name)),
            typed('group_name', env_str(module.name)),
            typed('entry_id', env_int(module.name)),
            typed('scope_id', env_int(module.name)),
            typed('revision', env_int(module.name)),
            *params,
        ]
    return tuple(params)


def gamma_alpha_queries_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed(
            'alpha_base',
            "Annotated[AlphaBaseService, qual('alpha') ]".replace(' ]', ']'),
        ),
        typed(
            'delta_base',
            "Annotated[DeltaBaseService, qual('delta') ]".replace(' ]', ']'),
        ),
        typed('shared', module.scope_shared),
    ]
    if scenario == 'env-sensitive':
        params = [
            typed('entry_id', env_int('gamma')),
            typed('scope_id', env_int('gamma')),
            *params,
        ]
    return tuple(params)


def gamma_delta_queries_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed(
            'delta_base',
            "Annotated[DeltaBaseService, qual('delta') ]".replace(' ]', ']'),
        ),
        typed('shared', module.scope_shared),
    ]
    if scenario == 'env-sensitive':
        params = [
            typed('entry_id', env_int('gamma')),
            typed('scope_id', env_int('gamma')),
            *params,
        ]
    return tuple(params)


def gamma_epsilon_queries_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed(
            'epsilon_base',
            "Annotated[EpsilonBaseService, qual('epsilon') ]".replace(' ]', ']'),
        ),
        typed('shared', module.scope_shared),
    ]
    if scenario == 'env-sensitive':
        params = [
            typed('entry_id', env_int('gamma')),
            typed('scope_id', env_int('gamma')),
            *params,
        ]
    return tuple(params)


def worker_service_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed('base', module.base_service),
        typed('shared', module.scope_shared_alt),
        typed('delta_queries', 'GammaDeltaQueries'),
    ]
    if scenario == 'env-sensitive':
        params = [
            typed('entry_id', env_int('gamma')),
            typed('scope_id', env_int('gamma')),
            typed('worker_id', env_int('gamma')),
            *params,
        ]
    return tuple(params)


def worker_step_service_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed('base', module.base_service),
        typed('shared', module.scope_shared_alt),
        typed('epsilon_queries', 'GammaEpsilonQueries'),
    ]
    if scenario == 'env-sensitive':
        params = [
            typed('entry_id', env_int('gamma')),
            typed('scope_id', env_int('gamma')),
            typed('worker_id', env_int('gamma')),
            typed('step_id', env_int('gamma')),
            *params,
        ]
    return tuple(params)


def beta_delta_queries_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed(
            'delta_base',
            "Annotated[DeltaBaseService, qual('delta') ]".replace(' ]', ']'),
        ),
        typed('shared', module.scope_shared),
    ]
    if scenario == 'env-sensitive':
        params = [
            typed('entry_id', env_int('beta')),
            typed('scope_id', env_int('beta')),
            *params,
        ]
    return tuple(params)


def beta_epsilon_data_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed(
            'epsilon_base',
            "Annotated[EpsilonBaseService, qual('epsilon') ]".replace(' ]', ']'),
        ),
        typed('shared', module.scope_shared),
    ]
    if scenario == 'env-sensitive':
        params = [
            typed('entry_id', env_int('beta')),
            typed('scope_id', env_int('beta')),
            *params,
        ]
    return tuple(params)


def flow_service_params(
    module: ModuleShape, scenario: Scenario
) -> tuple[TypedName, ...]:
    params: list[TypedName] = [
        typed('base', module.base_service),
        typed('shared', module.scope_shared_alt),
        typed('delta_queries', 'BetaDeltaQueries'),
        typed('epsilon_data', 'BetaEpsilonData'),
    ]
    if scenario == 'env-sensitive':
        params = [typed('flow_id', env_int('beta')), *params]
    return tuple(params)


def module_service_classes(
    module: ModuleShape, scenario: Scenario
) -> tuple[ClassSpec, ...]:
    classes: list[ClassSpec] = [
        ClassSpec(module.base_service, base_service_params(module)),
        ClassSpec(module.write_service, write_service_params(module)),
    ]
    if module.entry or module.scope_only:
        classes.append(
            ClassSpec(module.transaction_executor, transaction_executor_params(module))
        )
        classes.extend(
            ClassSpec(name, handler_params(module)) for name in module.handler_classes
        )
        classes.extend([
            ClassSpec(module.handler_refs, handler_refs_params(module)),
            ClassSpec(module.scope_audit, scope_audit_params(module)),
            ClassSpec(module.scope_service, scope_service_params(module, scenario)),
            ClassSpec(
                module.scope_write_service,
                scope_write_service_params(module),
            ),
            ClassSpec(module.event_service, event_service_params(module, scenario)),
        ])
    if module.entry:
        classes.append(
            ClassSpec(module.entry_service, entry_service_params(module, scenario))
        )
    match module.extra_transition:
        case 'worker':
            classes.extend([
                ClassSpec(
                    'GammaAlphaQueries',
                    gamma_alpha_queries_params(module, scenario),
                ),
                ClassSpec(
                    'GammaDeltaQueries',
                    gamma_delta_queries_params(module, scenario),
                ),
                ClassSpec(
                    'GammaEpsilonQueries',
                    gamma_epsilon_queries_params(module, scenario),
                ),
                ClassSpec(
                    module.worker_service,
                    worker_service_params(module, scenario),
                ),
                ClassSpec(
                    module.worker_step_service,
                    worker_step_service_params(module, scenario),
                ),
            ])
        case 'flow':
            classes.extend([
                ClassSpec(
                    'BetaDeltaQueries',
                    beta_delta_queries_params(module, scenario),
                ),
                ClassSpec(
                    'BetaEpsilonData',
                    beta_epsilon_data_params(module, scenario),
                ),
                ClassSpec(
                    module.flow_service,
                    flow_service_params(module, scenario),
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
    if module.entry:
        protocols.extend([
            ProtocolSpec(
                module.entry_context,
                (typed('entry', module.entry_service),),
                tuple(_entry_methods(module)),
            ),
            ProtocolSpec(
                module.scope_context,
                _scope_fields(module),
                (
                    MethodSpec(
                        'with_scoped_write',
                        (),
                        write_context(module.scope_write_context),
                    ),
                ),
            ),
            ProtocolSpec(
                module.scope_write_context,
                _scope_write_fields(module),
                tuple(_scope_write_methods(module)),
            ),
            ProtocolSpec(
                module.event_context,
                (
                    typed('event', module.event_service),
                    typed('audit', module.scope_audit),
                ),
                (
                    MethodSpec(
                        'with_handler', (), write_context(module.scope_write_context)
                    ),
                ),
            ),
        ])
    if module.scope_only:
        protocols.extend([
            ProtocolSpec(
                module.scope_context,
                _scope_fields(module),
                (
                    MethodSpec(
                        'with_scoped_write',
                        (),
                        write_context(module.scope_write_context),
                    ),
                ),
            ),
            ProtocolSpec(
                module.scope_write_context,
                _scope_write_fields(module),
                (),
            ),
            ProtocolSpec(
                module.event_context,
                (
                    typed('event', module.event_service),
                    typed('audit', module.scope_audit),
                ),
                (
                    MethodSpec(
                        'with_handler', (), write_context(module.scope_write_context)
                    ),
                ),
            ),
        ])
    match module.extra_transition:
        case 'worker':
            protocols.extend([
                ProtocolSpec(
                    module.worker_context,
                    (
                        typed('worker', module.worker_service),
                        typed('delta_queries', 'GammaDeltaQueries'),
                    ),
                    (
                        MethodSpec(
                            'with_step',
                            (typed('step_id', 'int'),),
                            write_context(module.worker_step_context),
                        ),
                    ),
                ),
                ProtocolSpec(
                    module.worker_step_context,
                    (
                        typed('step', module.worker_step_service),
                        typed('epsilon_queries', 'GammaEpsilonQueries'),
                    ),
                    (),
                ),
            ])
        case 'flow':
            protocols.append(
                ProtocolSpec(
                    module.flow_context,
                    (
                        typed('flow', module.flow_service),
                        typed('epsilon_data', 'BetaEpsilonData'),
                    ),
                    (),
                )
            )
        case 'none':
            pass
    return tuple(protocols)


def _context_methods(module: ModuleShape) -> list[MethodSpec]:
    methods = [MethodSpec('with_write', (), write_context(module.write_context))]
    if module.entry:
        methods.append(
            MethodSpec(
                'with_entry',
                (typed('entry_id', 'int'),),
                module.entry_context,
            )
        )
        methods.extend(
            MethodSpec(f'route_{index}', (), module.entry_context)
            for index in module.route_indices
        )
        methods.append(
            MethodSpec(
                'with_event_scope',
                (
                    typed('event_id', 'int'),
                    typed('group_name', 'str'),
                    typed('entry_id', 'int'),
                    typed('scope_id', 'int'),
                    typed('revision', 'int'),
                ),
                module.event_context,
            )
        )
    if module.scope_only:
        methods.extend([
            MethodSpec(
                'with_scoped_entry',
                (typed('scope_id', 'int'), typed('revision', 'int')),
                module.scope_context,
            ),
            MethodSpec(
                'with_event_scope',
                (
                    typed('event_id', 'int'),
                    typed('group_name', 'str'),
                    typed('entry_id', 'int'),
                    typed('scope_id', 'int'),
                    typed('revision', 'int'),
                ),
                module.event_context,
            ),
        ])
    return methods


def _entry_methods(module: ModuleShape) -> list[MethodSpec]:
    methods = [
        MethodSpec('with_write', (), write_context(module.write_context)),
        MethodSpec(
            'with_scoped_entry',
            (typed('scope_id', 'int'), typed('revision', 'int')),
            module.scope_context,
        ),
    ]
    methods.extend(
        MethodSpec(f'transition_{index}', (), module.scope_context)
        for index in module.scope_indices
    )
    return methods


def _scope_fields(module: ModuleShape) -> tuple[TypedName, ...]:
    return (
        typed('scope', module.scope_service),
        typed('executor', module.transaction_executor),
        typed('audit', module.scope_audit),
        *tuple(
            typed(f'extra_{index}', extra)
            for index, extra in enumerate(module.read_extras)
        ),
    )


def _scope_write_fields(module: ModuleShape) -> tuple[TypedName, ...]:
    return (
        typed('scope_write', module.scope_write_service),
        typed('handler_refs', module.handler_refs),
        *tuple(
            typed(f'extra_{index}', extra)
            for index, extra in enumerate(module.write_extras)
        ),
    )


def _scope_write_methods(module: ModuleShape) -> list[MethodSpec]:
    match module.extra_transition:
        case 'worker':
            return [
                MethodSpec(
                    'with_worker',
                    (typed('worker_id', 'int'),),
                    write_context(module.worker_context),
                )
            ]
        case 'flow':
            return [
                MethodSpec(
                    'with_flow',
                    (typed('flow_id', 'int'),),
                    write_context(module.flow_context),
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
    if module.entry:
        functions.extend([
            FunctionSpec(
                f'{module.name}_with_entry',
                (typed('entry_id', env_int(module.name)),),
                module.entry_state,
                ("return {'entry_id': entry_id}",),
            ),
            FunctionSpec(
                f'{module.name}_with_scoped_entry',
                (
                    typed('scope_id', env_int(module.name)),
                    typed('revision', env_int(module.name)),
                ),
                module.scope_state,
                ("return {'scope_id': scope_id, 'revision': revision}",),
            ),
            FunctionSpec(
                f'{module.name}_with_event_scope',
                (
                    typed('event_id', env_int(module.name)),
                    typed('group_name', env_str(module.name)),
                    typed('entry_id', env_int(module.name)),
                    typed('scope_id', env_int(module.name)),
                    typed('revision', env_int(module.name)),
                ),
                module.event_state,
                (
                    "return {'event_id': event_id, 'group_name': group_name, "
                    + "'entry_id': entry_id, 'scope_id': scope_id, "
                    + "'revision': revision}",
                ),
            ),
            FunctionSpec(
                f'{module.name}_with_handler',
                (),
                module.handler_state,
                (f'return {module.handler_state}()',),
            ),
            FunctionSpec(
                f'{module.name}_with_scoped_write',
                (),
                module.write_state,
                (f'return {module.write_state}()',),
            ),
        ])
        functions.extend(
            FunctionSpec(
                f'{module.name}_route_{index}',
                (),
                module.entry_state,
                (f"return {{'entry_id': {index + 1}}}",),
            )
            for index in module.route_indices
        )
        functions.extend(
            FunctionSpec(
                f'{module.name}_transition_{index}',
                (),
                module.scope_state,
                (f"return {{'scope_id': {index + 1}, 'revision': {index + 1}}}",),
            )
            for index in module.scope_indices
        )
    if module.scope_only:
        functions.extend([
            FunctionSpec(
                f'{module.name}_with_scoped_entry',
                (
                    typed('scope_id', env_int(module.name)),
                    typed('revision', env_int(module.name)),
                ),
                module.scope_state,
                ("return {'scope_id': scope_id, 'revision': revision}",),
            ),
            FunctionSpec(
                f'{module.name}_with_event_scope',
                (
                    typed('event_id', env_int(module.name)),
                    typed('group_name', env_str(module.name)),
                    typed('entry_id', env_int(module.name)),
                    typed('scope_id', env_int(module.name)),
                    typed('revision', env_int(module.name)),
                ),
                module.event_state,
                (
                    "return {'event_id': event_id, 'group_name': group_name, "
                    + "'entry_id': entry_id, 'scope_id': scope_id, "
                    + "'revision': revision}",
                ),
            ),
            FunctionSpec(
                f'{module.name}_with_handler',
                (),
                module.handler_state,
                (f'return {module.handler_state}()',),
            ),
            FunctionSpec(
                f'{module.name}_with_scoped_write',
                (),
                module.write_state,
                (f'return {module.write_state}()',),
            ),
        ])
    match module.extra_transition:
        case 'worker':
            functions.extend([
                FunctionSpec(
                    'gamma_with_worker',
                    (typed('worker_id', env_int('gamma')),),
                    module.worker_state,
                    ("return {'worker_id': worker_id}",),
                ),
                FunctionSpec(
                    'gamma_with_step',
                    (typed('step_id', env_int('gamma')),),
                    module.worker_step_state,
                    ("return {'step_id': step_id}",),
                ),
            ])
        case 'flow':
            functions.append(
                FunctionSpec(
                    'beta_with_flow',
                    (typed('flow_id', env_int('beta')),),
                    module.flow_state,
                    ("return {'flow_id': flow_id}",),
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
            case 'worker':
                gamma_alpha_params = gamma_alpha_queries_params(module, scenario)
                gamma_delta_params = gamma_delta_queries_params(module, scenario)
                gamma_epsilon_params = gamma_epsilon_queries_params(module, scenario)
                functions.extend([
                    FunctionSpec(
                        'provide_gamma_alpha_queries',
                        gamma_alpha_params,
                        'GammaAlphaQueries',
                        provider_body('GammaAlphaQueries', gamma_alpha_params),
                    ),
                    FunctionSpec(
                        'provide_gamma_delta_queries',
                        gamma_delta_params,
                        'GammaDeltaQueries',
                        provider_body('GammaDeltaQueries', gamma_delta_params),
                    ),
                    FunctionSpec(
                        'provide_gamma_epsilon_queries',
                        gamma_epsilon_params,
                        'GammaEpsilonQueries',
                        provider_body('GammaEpsilonQueries', gamma_epsilon_params),
                    ),
                ])
            case 'flow':
                beta_delta_params = beta_delta_queries_params(module, scenario)
                beta_epsilon_params = beta_epsilon_data_params(module, scenario)
                functions.extend([
                    FunctionSpec(
                        'provide_beta_delta_queries',
                        beta_delta_params,
                        'BetaDeltaQueries',
                        provider_body('BetaDeltaQueries', beta_delta_params),
                    ),
                    FunctionSpec(
                        'provide_beta_epsilon_data',
                        beta_epsilon_params,
                        'BetaEpsilonData',
                        provider_body('BetaEpsilonData', beta_epsilon_params),
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
    if module.entry:
        for hook_index in module.hook_indices:
            hooks.append(
                HookSpec(
                    module.context,
                    'with_entry',
                    f'{module.name}_with_entry_hook_{hook_index}',
                    (
                        typed('entry_id', env_int(module.name)),
                        typed('shared', module.hook_shared),
                    ),
                )
            )
            hooks.append(
                HookSpec(
                    module.entry_context,
                    'with_scoped_entry',
                    f'{module.name}_with_scoped_entry_hook_{hook_index}',
                    (
                        typed('scope_id', env_int(module.name)),
                        typed('revision', env_int(module.name)),
                        typed('shared', module.hook_shared_alt),
                    ),
                )
            )
            hooks.append(
                HookSpec(
                    module.context,
                    'with_event_scope',
                    f'{module.name}_with_event_scope_hook_{hook_index}',
                    (
                        typed('event_id', env_int(module.name)),
                        typed('entry_id', env_int(module.name)),
                        typed('scope_id', env_int(module.name)),
                        typed('shared', module.hook_shared),
                    ),
                )
            )
            hooks.append(
                HookSpec(
                    module.event_context,
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
        for scope_index in module.scope_indices:
            for hook_index in module.hook_indices:
                hooks.append(
                    HookSpec(
                        module.entry_context,
                        f'transition_{scope_index}',
                        f'{module.name}_transition_{scope_index}_hook_{hook_index}',
                        (typed('shared', module.hook_shared_alt),),
                    )
                )
    if module.scope_only:
        for hook_index in module.hook_indices:
            hooks.extend([
                HookSpec(
                    module.context,
                    'with_scoped_entry',
                    f'{module.name}_with_scoped_entry_hook_{hook_index}',
                    (
                        typed('scope_id', env_int(module.name)),
                        typed('revision', env_int(module.name)),
                        typed('shared', module.hook_shared),
                    ),
                ),
                HookSpec(
                    module.context,
                    'with_event_scope',
                    f'{module.name}_with_event_scope_hook_{hook_index}',
                    (
                        typed('event_id', env_int(module.name)),
                        typed('entry_id', env_int(module.name)),
                        typed('scope_id', env_int(module.name)),
                        typed('shared', module.hook_shared_alt),
                    ),
                ),
                HookSpec(
                    module.event_context,
                    'with_handler',
                    f'{module.name}_with_handler_hook_{hook_index}',
                    (typed('shared', module.hook_shared),),
                ),
            ])
    match module.extra_transition:
        case 'worker':
            for hook_index in module.hook_indices:
                hooks.extend([
                    HookSpec(
                        module.scope_write_context,
                        'with_worker',
                        f'gamma_with_worker_hook_{hook_index}',
                        (
                            typed('worker_id', env_int('gamma')),
                            typed('shared', module.hook_shared),
                        ),
                    ),
                    HookSpec(
                        module.worker_context,
                        'with_step',
                        f'gamma_with_step_hook_{hook_index}',
                        (
                            typed('step_id', env_int('gamma')),
                            typed('shared', module.hook_shared_alt),
                        ),
                    ),
                ])
        case 'flow':
            for hook_index in module.hook_indices:
                hooks.append(
                    HookSpec(
                        module.scope_write_context,
                        'with_flow',
                        f'beta_with_flow_hook_{hook_index}',
                        (
                            typed('flow_id', env_int('beta')),
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
    if module.entry:
        methods.extend([
            MethodRegistrationSpec(
                module.context,
                'with_entry',
                f'{module.name}_with_entry',
            ),
            MethodRegistrationSpec(
                module.context,
                'with_event_scope',
                f'{module.name}_with_event_scope',
            ),
            MethodRegistrationSpec(
                module.entry_context,
                'with_write',
                f'{module.name}_with_write',
            ),
            MethodRegistrationSpec(
                module.entry_context,
                'with_scoped_entry',
                f'{module.name}_with_scoped_entry',
            ),
            MethodRegistrationSpec(
                module.scope_context,
                'with_scoped_write',
                f'{module.name}_with_scoped_write',
            ),
            MethodRegistrationSpec(
                module.event_context,
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
                module.entry_context,
                f'transition_{index}',
                f'{module.name}_transition_{index}',
            )
            for index in module.scope_indices
        )
    if module.scope_only:
        methods.extend([
            MethodRegistrationSpec(
                module.context,
                'with_scoped_entry',
                f'{module.name}_with_scoped_entry',
            ),
            MethodRegistrationSpec(
                module.context,
                'with_event_scope',
                f'{module.name}_with_event_scope',
            ),
            MethodRegistrationSpec(
                module.event_context,
                'with_handler',
                f'{module.name}_with_handler',
            ),
            MethodRegistrationSpec(
                module.scope_context,
                'with_scoped_write',
                f'{module.name}_with_scoped_write',
            ),
        ])
    match module.extra_transition:
        case 'worker':
            methods.extend([
                MethodRegistrationSpec(
                    module.scope_write_context,
                    'with_worker',
                    'gamma_with_worker',
                ),
                MethodRegistrationSpec(
                    module.worker_context,
                    'with_step',
                    'gamma_with_step',
                ),
            ])
        case 'flow':
            methods.append(
                MethodRegistrationSpec(
                    module.scope_write_context,
                    'with_flow',
                    'beta_with_flow',
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
    if module.entry or module.scope_only:
        names.extend([
            module.handler_state,
            module.scope_state,
            module.event_state,
            module.transaction_executor,
            *module.handler_classes,
            module.handler_refs,
            module.scope_audit,
            module.scope_service,
            module.event_service,
        ])
    if module.entry:
        names.extend([module.entry_state, module.entry_service])
    match module.extra_transition:
        case 'worker':
            names.extend([module.worker_state, module.worker_step_state])
        case 'flow':
            names.extend([module.flow_state])
        case 'none':
            pass
    return unique_names(*names)


def module_write_classes(module: ModuleShape) -> tuple[str, ...]:
    names = [module.write_service]
    if module.entry or module.scope_only:
        names.append(module.scope_write_service)
    match module.extra_transition:
        case 'worker':
            names.extend([module.worker_service, module.worker_step_service])
        case 'flow':
            names.append(module.flow_service)
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
            case 'worker':
                registrations.extend([
                    ConstructorRegistrationSpec(
                        'GammaAlphaQueries',
                        module_scopes('gamma'),
                        'provide_gamma_alpha_queries',
                    ),
                    ConstructorRegistrationSpec(
                        'GammaDeltaQueries',
                        module_scopes('gamma'),
                        'provide_gamma_delta_queries',
                    ),
                    ConstructorRegistrationSpec(
                        'GammaEpsilonQueries',
                        module_scopes('gamma'),
                        'provide_gamma_epsilon_queries',
                    ),
                ])
            case 'flow':
                registrations.extend([
                    ConstructorRegistrationSpec(
                        'BetaDeltaQueries',
                        module_scopes('beta'),
                        'provide_beta_delta_queries',
                    ),
                    ConstructorRegistrationSpec(
                        'BetaEpsilonData',
                        module_scopes('beta'),
                        'provide_beta_epsilon_data',
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
        typed_dict_specs.extend(module_typed_dicts(module))
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

        if 'alpha' in enabled_modules:
            alpha_scope = app.with_alpha_context().route_0().transition_0()
            _ = alpha_scope.scope
            if scenario == 'portable':
                alpha_scope_write = alpha_scope.with_scoped_write()
                _ = alpha_scope_write.scope_write
            else:
                _ = app.with_alpha_context().with_write().write

        if 'beta' in enabled_modules:
            if scenario == 'portable':
                beta_scope_write = (
                    app.with_beta_context().route_0().transition_0().with_scoped_write()
                )
                beta_flow = beta_scope_write.with_flow(1)
                _ = beta_flow.flow
            else:
                _ = app.with_beta_context().with_write().write

        if 'gamma' in enabled_modules:
            if scenario == 'portable':
                gamma_scope_write = (
                    app
                    .with_gamma_context()
                    .route_0()
                    .transition_0()
                    .with_scoped_write()
                )
                gamma_step = gamma_scope_write.with_worker(1).with_step(1)
                _ = gamma_step.step
            else:
                _ = app.with_gamma_context().with_write().write

        if 'epsilon' in enabled_modules:
            epsilon_event = app.with_epsilon_context().with_event_scope(
                1, 'events', 1, 1, 1
            )
            _ = epsilon_event.event
            if scenario == 'portable':
                epsilon_write = epsilon_event.with_handler()
                _ = epsilon_write.scope_write

        if 'delta' in enabled_modules:
            delta_write = app.with_delta_context().with_write()
            _ = delta_write.write
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
    _ = parser.add_argument(
        '--scenario',
        choices=['portable', 'env-sensitive', 'all'],
        default='env-sensitive',
    )
    _ = parser.add_argument('--density', type=int, default=4)
    _ = parser.add_argument('--handlers', type=int, default=4)
    _ = parser.add_argument('--shared-slots', type=int, default=2)
    _ = parser.add_argument(
        '--modules',
        default='alpha,beta,gamma,delta,epsilon',
    )
    _ = parser.add_argument('--invoke', action='store_true')
    _ = parser.add_argument('--dump-generated', action='store_true')
    args = parser.parse_args()
    parsed = cast(dict[str, object], vars(args))
    modules_arg = cast(str, parsed['modules'])

    enabled_modules = tuple(
        module.strip()
        for module in modules_arg.split(',')
        if module.strip() in {'alpha', 'beta', 'gamma', 'delta', 'epsilon'}
    )
    assert enabled_modules, 'Expected at least one benchmark module'

    parsed_scenario = cast(Scenario | Literal['all'], parsed['scenario'])
    scenarios: list[Scenario]
    if parsed_scenario == 'all':
        scenarios = ['portable', 'env-sensitive']
    else:
        scenarios = [parsed_scenario]

    for scenario in scenarios:
        run_once(
            scenario=scenario,
            density=cast(int, parsed['density']),
            handler_count=cast(int, parsed['handlers']),
            shared_slots=cast(int, parsed['shared_slots']),
            enabled_modules=enabled_modules,
            invoke=cast(bool, parsed['invoke']),
            dump_generated=cast(bool, parsed['dump_generated']),
        )


if __name__ == '__main__':
    main()
