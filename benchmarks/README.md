# Benchmarks

Controlled local benchmarks for `inlay` compile and transition scaling. Each
top-level benchmark should have one primary scaling behavior; avoid adding a new
benchmark if an existing one already isolates the same behavior.

## Primary Benchmarks

| Benchmark | Scaling behavior | Primary knobs | Non-goal |
| --- | --- | --- | --- |
| `cache_portability` | Reuse or rejection of cached child context compilation across many explicit transitions returning the same protocol | `--transitions`, `--depth`, `--properties`, `--scenario` | Application-shaped module graphs, lazy cycles |
| `cross_context_queries` | Qualified sibling-context query fanout from one module into another | `--transitions`, `--queries`, `--source-depth`, `--target-depth`, `--values`, `--scenario` | Entry/write transition ladders |
| `staged_neutral_cycle` | Structural threshold from a lazy self-cycle to fork/open/hooks/relay/cluster/tracks/cross-track recursion | `--stage`, `--density`, `--handlers`, `--shared-slots`, `--tracks` | Production naming or service fidelity |
| `production_shape` | End-to-end production-like module graph with entry scopes, writes, hooks, worker flows, and cross-module adapters | `--density`, `--handlers`, `--shared-slots`, `--modules`, `--scenario` | Lazy recursive cycle behavior |

## Scenarios

`portable` means the child context does not read from the explicit transition
result source, so reuse across transition environments should be safe.

`env-sensitive` means providers read transition result values, so reuse across
transition environments must be rejected or scoped more narrowly.

## Usage

```bash
make bench BENCH=cache_portability ARGS="--scenario all --transitions 16 --depth 3 --properties 32"
make bench BENCH=cross_context_queries ARGS="--scenario env-sensitive --transitions 32 --source-depth 3 --queries 8 --values 16 --target-depth 2"
make bench BENCH=staged_neutral_cycle ARGS="--stage open --density 4 --handlers 1"
make bench BENCH=staged_neutral_cycle ARGS="--stage relay --density 1 --handlers 1"
make bench BENCH=production_shape ARGS="--scenario env-sensitive --density 4 --handlers 4 --shared-slots 2"
```

Add `--invoke` where supported to include first-transition execution time after
compilation.

## Perfetto Tracing

```bash
make bench BENCH=production_shape TRACE=/tmp/inlay-production-shape.pftrace ARGS="--density 2 --handlers 2 --invoke"

make perfetto-install
make perfetto-query TRACE=/tmp/inlay-production-shape.pftrace
make perfetto-query TRACE=/tmp/inlay-production-shape.pftrace SQL="SELECT EXTRACT_ARG(arg_set_id, 'debug.outcome') AS outcome, COUNT(*) AS count FROM slice WHERE name = 'solver.goal_outcome' GROUP BY outcome ORDER BY count DESC"
```

`helpers/query_trace.py` is an optional Python trace-query helper. It is not a
benchmark and is intentionally outside the top-level `make bench BENCH=<name>`
entrypoint set.

## Retired Duplicates

`appish_transitions.py`, `appish_codegen.py`, and `density_template.py` were
removed because they overlapped with `production_shape`, `staged_neutral_cycle`,
or `cross_context_queries` without isolating a distinct scaling behavior.

## Adding Benchmarks

New benchmarks should print `benchmark=<name>`, expose the smallest set of knobs
needed for the target scaling behavior, and include a tiny smoke-run command in
this README. If a benchmark combines multiple behaviors, add the missing knob or
stage to an existing benchmark instead of creating another overlapping file.
