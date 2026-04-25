Run local benchmarks from `packages/inlay` with `make bench BENCH=<name> ARGS="..."`; for traced runs, add `TRACE=/tmp/<name>.pftrace` and the target will rebuild with `perfetto-tracing` enabled.
Query traces natively with Perfetto via `make perfetto-install` once and then `make perfetto-query TRACE=/tmp/<name>.pftrace [SQL="SELECT ..."]`.
For tracing, put stable context fields on spans and do not duplicate them on child events unless the event can occur outside that span.
Model function outcomes as instrumented return/error attributes via `ret`/`err`; do not create separate outcome events.
Keep tracing-gated field computation inside tracing macros: prefer inline `solver_span_record!` / `solver_event!` field expressions over `#[cfg(feature = "tracing")]` locals or blocks when the values are only used for tracing.
