# Benchmarks

Controlled local benchmarks for `inlay` cache portability behavior.

The main entrypoint is `cache_portability.py`. It constructs a root protocol
with many explicit transition methods that all return the same child protocol.

`hostish_cross_context.py` models a more application-like graph:

- `RootContext -> with_ai_context() / with_character_context()`
- `AiContext` has many explicit branch methods
- each AI branch child contains many query adapters
- query adapters depend on sibling `CharacterContext`

`appish_transitions.py` pushes further toward the real service shape:

- `AppContext -> with_chat_context() / with_game_context() / ...`
- module contexts expose `with_write()`, `with_session()`,
  `with_branched_session()`, and `with_message_scope()`-style transitions
- session modules also expose repeated explicit route/branch methods that return
  the same nested context types under different transition environments
- nested contexts include lazy backrefs via `LazyRef[...]`-based executor helper
  properties
- AI and Game contexts include cross-module adapters that depend on sibling
  Story/Character/Game contexts

`appish_codegen.py` uses a different generation strategy:

- it emits Python source code with real method signatures and executes it with
  `exec()`
- it keeps an app-shaped root plus repeated session/branch transitions
- it includes an explicit root-level `LazyRef[...]` self-cycle, which produces
  nonzero `active_ancestor_lazy_hits` and `fixpoint_reruns`

This benchmark is useful when you specifically want to exercise a real lazy
cycle instead of only repeated cache reuse.

`staged_neutral_cycle.py` is a neutral staged ladder for isolating the compile
explosion ingredients:

- `cycle`: pure lazy self-cycle
- `cycle-write`: add nested write-qualified child context
- `fork`: add one value-carrying transition before the recursive loop
- `open`: add a second value-carrying transition before the recursive loop
- `hooks`: add repeated method hooks to the transition chain
- `relay`: add sibling recursive fan-out on top of `hooks`
- `cluster`: route the fan-out through shared executor/audit/service families
- `tracks`: duplicate the `cluster` stage across multiple qualified tracks with
  shared slot families
- `cross`: add saga-like nested write contexts with cross-track env-sensitive
  adapters

This benchmark is useful when you want to identify the first structural change
that introduces lazy hits, fixpoint reruns, and eventually superlinear compile
time growth.

The benchmark is still synthetic, but it is meant to be closer to the real
monolithic app graph than `hostish_cross_context.py`.

Two scenarios are supported:

- `portable`: the child protocol does not read from the explicit transition
  result source, so cache reuse across transition environments should be safe.
- `env-sensitive`: the child protocol reads `branch_id` from the explicit
  transition result source, so cache reuse across transition environments must
  be rejected.

Usage:

```bash
uv run python benchmarks/cache_portability.py --branches 8 --properties 24
uv run python benchmarks/cache_portability.py --scenario portable --branches 32 --depth 4 --properties 64
uv run python benchmarks/cache_portability.py --scenario env-sensitive --branches 32 --depth 4 --properties 64 --invoke
uv run python benchmarks/hostish_cross_context.py --branches 32 --depth 3 --queries 8 --values 16 --character-depth 2
uv run python benchmarks/appish_transitions.py --fanout 4
uv run python benchmarks/appish_transitions.py --scenario all --fanout 8 --invoke
uv run python benchmarks/appish_codegen.py --fanout 1 --invoke
INLAY_COMPILE_STATS=1 INLAY_SOLVER_STATS=1 uv run python benchmarks/appish_codegen.py --scenario all --fanout 2
uv run python benchmarks/staged_neutral_cycle.py --stage open --density 4 --handlers 1
INLAY_COMPILE_STATS=1 INLAY_SOLVER_STATS=1 uv run python benchmarks/staged_neutral_cycle.py --stage relay --density 1 --handlers 1
```

Output includes:

- registry build time
- compile time
- optional first transition invocation time

The benchmark is intentionally synthetic. It is meant to compare cache
portability behavior under controlled conditions, not to mirror the full game
service graph exactly.
