"""Publish an immutable comparison of two reviewed training-history releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from publish_benchmark import _publishing_source_provenance, _validate_release_id
from validate_benchmark_release import validate_release

from retail_forecast.benchmark_protocol import load_benchmark_spec
from retail_forecast.benchmark_registry import PUBLISHED_DIR
from retail_forecast.history_comparison import (
    build_training_history_comparison,
    summarize_training_history_comparison,
)

COMPARISON_SCHEMA_VERSION = "training-history-comparison-release-v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison_id(main_benchmark_id: str, ablation_benchmark_id: str) -> str:
    return f"{main_benchmark_id}--vs--{ablation_benchmark_id}"


def published_comparison_root(comparison: str, release_id: str) -> Path:
    return PUBLISHED_DIR / "comparisons" / comparison / "releases" / release_id


def _source_descriptor(path: Path, release: dict) -> dict:
    return {
        "benchmark_id": release["benchmark_id"],
        "release_id": release["release_id"],
        "protocol_hash": release["protocol_hash"],
        "release_manifest_sha256": _sha256(path / "release_manifest.json"),
        "relative_path": (
            f"{release['benchmark_id']}/releases/{release['release_id']}"
        ),
    }


def publish_comparison(
    *,
    main_release_path: Path,
    fixed_window_release_path: Path,
    release_id: str,
) -> Path:
    """Validate two model releases and atomically publish their paired comparison."""
    release_id = _validate_release_id(release_id)
    provenance = _publishing_source_provenance()
    main_release = validate_release(main_release_path)
    fixed_window_release = validate_release(fixed_window_release_path)
    main_spec = load_benchmark_spec(main_release["benchmark_id"])
    fixed_window_spec = load_benchmark_spec(fixed_window_release["benchmark_id"])

    main_frame = pd.read_csv(main_release_path / "summary/scenario_leaderboard.csv")
    fixed_window_frame = pd.read_csv(
        fixed_window_release_path / "summary/scenario_leaderboard.csv"
    )
    comparison = build_training_history_comparison(
        main_frame,
        fixed_window_frame,
        main_spec=main_spec,
        fixed_window_spec=fixed_window_spec,
    )
    summary = summarize_training_history_comparison(comparison)
    identifier = comparison_id(main_spec.benchmark_id, fixed_window_spec.benchmark_id)
    destination = published_comparison_root(identifier, release_id)
    if destination.exists():
        raise FileExistsError(
            f"Published comparison already exists: {destination}. "
            "Release IDs are immutable."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{release_id}-", dir=destination.parent))
    try:
        comparison.to_csv(staging / "training_history_comparison.csv", index=False)
        summary.to_csv(staging / "scenario_summary.csv", index=False)
        files = sorted(path for path in staging.rglob("*") if path.is_file())
        manifest = {
            "release_schema_version": COMPARISON_SCHEMA_VERSION,
            "release_id": release_id,
            "comparison_id": identifier,
            "published_at": datetime.now(UTC).isoformat(),
            "published_by_source_provenance": provenance,
            "main_source_release": _source_descriptor(main_release_path, main_release),
            "fixed_window_source_release": _source_descriptor(
                fixed_window_release_path, fixed_window_release
            ),
            "fixed_window_weeks": fixed_window_spec.train_weeks,
            "model_ids": sorted(main_release["model_ids"]),
            "scenarios": list(main_spec.scenarios),
            "files": {str(path.relative_to(staging)): _sha256(path) for path in files},
        }
        (staging / "release_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-release", type=Path, required=True)
    parser.add_argument("--fixed-window-release", type=Path, required=True)
    parser.add_argument("--release", required=True)
    args = parser.parse_args()
    destination = publish_comparison(
        main_release_path=args.main_release,
        fixed_window_release_path=args.fixed_window_release,
        release_id=args.release,
    )
    print(f"Published immutable training-history comparison: {destination}")


if __name__ == "__main__":
    main()
