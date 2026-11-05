"""Validate and publish one lightweight V4 benchmark snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_forecast.benchmark_protocol import load_benchmark_spec
from retail_forecast.benchmark_registry import (
    PUBLISHED_DIR,
    benchmark_result_root,
    generate_v4_summaries,
    validate_v4_model_result,
)

MODEL_FILES = (
    "predictions.csv",
    "metrics_period.csv",
    "metrics_annual.csv",
    "selected_config.json",
    "manifest.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(source_root: Path, benchmark_id: str) -> Path:
    spec = load_benchmark_spec(benchmark_id)
    source = benchmark_result_root(source_root, benchmark_id)
    destination = benchmark_result_root(PUBLISHED_DIR, benchmark_id)
    if destination.exists():
        raise FileExistsError(
            f"Published snapshot already exists: {destination}. "
            "Published benchmark IDs are immutable."
        )
    models = sorted(path for path in (source / "models").iterdir() if path.is_dir())
    if not models:
        raise ValueError(f"No model results found under {source / 'models'}")
    for model_path in models:
        validate_v4_model_result(model_path, spec, require_all_scenarios=True)

    destination.mkdir(parents=True)
    for model_path in models:
        target = destination / "models" / model_path.name
        target.mkdir(parents=True)
        for filename in MODEL_FILES:
            shutil.copy2(model_path / filename, target / filename)
    generate_v4_summaries(base_dir=PUBLISHED_DIR, spec=spec)

    files = sorted(path for path in destination.rglob("*") if path.is_file())
    release = {
        "benchmark_id": benchmark_id,
        "protocol_hash": spec.protocol_hash,
        "published_at": datetime.now(UTC).isoformat(),
        "model_ids": [path.name for path in models],
        "files": {str(path.relative_to(destination)): _sha256(path) for path in files},
    }
    (destination / "release_manifest.json").write_text(
        json.dumps(release, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    destination = publish(args.source_root, args.benchmark)
    print(f"Published immutable snapshot: {destination}")


if __name__ == "__main__":
    main()
