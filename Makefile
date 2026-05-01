PERFETTO_DIR := .local/perfetto
TRACE_PROCESSOR := $(PERFETTO_DIR)/trace_processor
SQL ?= SELECT name, COUNT(*) AS count, ROUND(SUM(dur) / 1e6, 3) AS total_ms, ROUND(AVG(dur) / 1e3, 3) AS avg_us FROM slice WHERE name GLOB 'solver.*' OR name GLOB 'inlay*' GROUP BY name ORDER BY total_ms DESC, count DESC, name LIMIT 25
BASEDPYRIGHT_ARGS ?=
CARGO_TEST_ARGS ?=
PYTEST_ARGS ?=
RUFF_ARGS ?= check .

.PHONY: basedpyright bench perfetto-install perfetto-query ruff test test-python test-rust

basedpyright:
	uv run basedpyright $(BASEDPYRIGHT_ARGS)

ruff:
	uv run ruff $(RUFF_ARGS)

test: test-rust test-python

test-rust:
	cargo test $(CARGO_TEST_ARGS)
	cargo test --manifest-path crates/solver/Cargo.toml --features example $(CARGO_TEST_ARGS)

test-python:
	uv run pytest $(PYTEST_ARGS)

bench:
	@test -n "$(BENCH)" || (printf '%s\n' 'BENCH=<name> is required' >&2; exit 1)
	@test -f "benchmarks/templates/$(BENCH).py.jinja" || (printf 'benchmark not found: benchmarks/templates/%s.py.jinja\n' "$(BENCH)" >&2; exit 1)
	@if [ -n "$(TRACE)" ]; then \
		MATURIN_PEP517_ARGS="--features perfetto-tracing" \
		INLAY_PERFETTO_TRACE_PATH="$(TRACE)" \
		uv run --reinstall-package inlay python benchmarks/runner.py "$(BENCH)" $(ARGS); \
	else \
		uv run python benchmarks/runner.py "$(BENCH)" $(ARGS); \
	fi

perfetto-install:
	@mkdir -p "$(PERFETTO_DIR)"
	@if [ ! -x "$(TRACE_PROCESSOR)" ]; then \
		curl -fsSL https://get.perfetto.dev/trace_processor -o "$(TRACE_PROCESSOR)"; \
		chmod +x "$(TRACE_PROCESSOR)"; \
	fi
	@"$(TRACE_PROCESSOR)" --version

perfetto-query: perfetto-install
	@test -n "$(TRACE)" || (printf '%s\n' 'TRACE=/path/to/trace.pftrace is required' >&2; exit 1)
	@"$(TRACE_PROCESSOR)" -Q "$(SQL)" "$(TRACE)"
