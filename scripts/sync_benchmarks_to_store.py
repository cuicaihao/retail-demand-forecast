"""Retired V2 result importer.

Historical single-cutoff results are immutable under
data/benchmarks/archive_single_cutoff_28d/. Current results must be produced by
an explicit V4 benchmark and reviewed with scripts/publish_benchmark.py.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "This V2 importer is retired. Run scripts/run_benchmark.py with an "
        "explicit --benchmark, then publish the validated run with "
        "scripts/publish_benchmark.py."
    )


if __name__ == "__main__":
    main()
