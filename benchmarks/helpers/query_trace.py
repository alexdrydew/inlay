import argparse
import shutil
from pathlib import Path

try:
    from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig
except ModuleNotFoundError as exc:  # pragma: no cover - benchmark bootstrap guard
    raise SystemExit(
        'This helper needs the `perfetto` dev dependency. Run `uv sync --dev`.'
    ) from exc


DEFAULT_SQL = """
    SELECT
        name,
        COUNT(*) AS count,
        ROUND(SUM(dur) / 1e6, 3) AS total_ms,
        ROUND(AVG(dur) / 1e3, 3) AS avg_us
    FROM slice
    WHERE name GLOB 'solver.*' OR name = 'compile'
    GROUP BY name
    ORDER BY total_ms DESC, count DESC, name
    LIMIT 25
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run Perfetto SQL queries against an inlay benchmark trace.'
    )
    parser.add_argument('trace', type=Path, help='Path to a .pftrace file')
    parser.add_argument('--sql', help='Inline Perfetto SQL query to run')
    parser.add_argument(
        '--timeout',
        type=int,
        default=10,
        help='Trace processor startup timeout in seconds',
    )
    parser.add_argument(
        '--tp-bin',
        help='Optional path to a trace_processor_shell binary',
    )
    return parser.parse_args()


def resolve_sql(args: argparse.Namespace) -> str:
    if args.sql is not None:
        return args.sql
    return DEFAULT_SQL


def format_cell(value: object) -> str:
    if value is None:
        return ''
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def print_rows(result: TraceProcessor.QueryResultIterator) -> None:
    print('\t'.join(result.column_names))
    for row in result:
        print(
            '\t'.join(
                format_cell(getattr(row, column)) for column in result.column_names
            )
        )


def main() -> None:
    args = parse_args()
    sql = resolve_sql(args)
    bin_path = args.tp_bin or shutil.which('trace_processor_shell')
    config = TraceProcessorConfig(bin_path=bin_path, load_timeout=args.timeout)

    with TraceProcessor(trace=str(args.trace), config=config) as tp:
        print_rows(tp.query(sql))


if __name__ == '__main__':
    main()
