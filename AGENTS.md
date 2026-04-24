Run local benchmarks from `packages/inlay` with `make bench BENCH=<name> ARGS="..."`; for traced runs, add `TRACE=/tmp/<name>.pftrace` and the target will rebuild with `perfetto-tracing` enabled.
Query traces natively with Perfetto via `make perfetto-install` once and then `make perfetto-query TRACE=/tmp/<name>.pftrace [SQL="SELECT ..."]`.
