"""Recompute a published benchmark's file hashes and V4 contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_forecast.benchmark_protocol import load_benchmark_spec
from retail_forecast.benchmark_registry import validate_v4_model_result
from retail_forecast.history_comparison import (
    build_training_history_comparison,
    summarize_training_history_comparison,
)

COMPARISON_SCHEMA_VERSION = "training-history-comparison-release-v2"


def _validate_comparison_release(path: Path, release: dict) -> None:
    if release.get("release_id") != path.name:
        raise ValueError("Comparison release ID does not match its directory")
    if path.parent.name != "releases" or path.parent.parent.name != release.get(
        "comparison_id"
    ):
        raise ValueError("Comparison ID does not match its directory")
    provenance = release.get("published_by_source_provenance")
    if not isinstance(provenance, dict) or provenance.get("git_dirty"):
        raise ValueError("Comparison releases require clean publishing provenance")
    if set(release["files"]) != {
        "training_history_comparison.csv",
        "scenario_summary.csv",
    }:
        raise ValueError("Unexpected comparison release file inventory")

    published_root = path.parents[3]
    source_records = []
    for key in ("main_source_release", "fixed_window_source_release"):
        descriptor = release.get(key)
        if not isinstance(descriptor, dict):
            raise TypeError(f"Comparison release is missing {key}")
        source_path = published_root / descriptor["relative_path"]
        source_release = validate_release(source_path)
        source_hash = hashlib.sha256(
            (source_path / "release_manifest.json").read_bytes()
        ).hexdigest()
        if source_hash != descriptor["release_manifest_sha256"]:
            raise ValueError(f"{key} manifest hash does not match")
        for field in ("benchmark_id", "release_id", "protocol_hash"):
            if source_release[field] != descriptor[field]:
                raise ValueError(f"{key} {field} does not match")
        source_records.append((source_path, source_release))

    (main_path, main_release), (fixed_path, fixed_release) = source_records
    main_spec = load_benchmark_spec(main_release["benchmark_id"])
    fixed_spec = load_benchmark_spec(fixed_release["benchmark_id"])
    if release.get("fixed_window_weeks") != fixed_spec.train_weeks:
        raise ValueError("Comparison fixed-window length does not match its protocol")
    expected = build_training_history_comparison(
        pd.read_csv(main_path / "summary/scenario_leaderboard.csv"),
        pd.read_csv(fixed_path / "summary/scenario_leaderboard.csv"),
        main_spec=main_spec,
        fixed_window_spec=fixed_spec,
    )
    actual = pd.read_csv(path / "training_history_comparison.csv")
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12)
    expected_summary = summarize_training_history_comparison(expected)
    actual_summary = pd.read_csv(path / "scenario_summary.csv")
    pd.testing.assert_frame_equal(
        actual_summary, expected_summary, check_exact=False, rtol=1e-12
    )
    if release["model_ids"] != sorted(main_release["model_ids"]):
        raise ValueError("Comparison model inventory does not match its source")
    if release["scenarios"] != list(main_spec.scenarios):
        raise ValueError("Comparison scenario inventory does not match its source")


def validate_release(path: Path) -> dict:
    manifest_path = path / "release_manifest.json"
    release = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed_files = set(release["files"])
    actual_files = {
        str(file_path.relative_to(path))
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path != manifest_path
    }
    if listed_files != actual_files:
        raise ValueError(
            "Release file inventory mismatch: "
            f"missing={sorted(listed_files - actual_files)}, "
            f"unexpected={sorted(actual_files - listed_files)}"
        )
    for relative, expected in release["files"].items():
        file_path = path / relative
        if not file_path.is_file():
            raise ValueError(f"Release file is missing: {relative}")
        actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Release hash mismatch: {relative}")
    if release.get("release_schema_version") == COMPARISON_SCHEMA_VERSION:
        _validate_comparison_release(path, release)
        return release
    if release.get("protocol_hash"):
        spec = load_benchmark_spec(release["benchmark_id"])
        if release["protocol_hash"] != spec.protocol_hash:
            raise ValueError("Release protocol hash does not match declaration")
        if release.get("release_schema_version") != "benchmark-release-v1":
            raise ValueError("Unsupported V4 release schema")
        if release.get("release_id") != path.name:
            raise ValueError("Release ID does not match its directory")
        publishing_provenance = release.get("published_by_source_provenance")
        if not isinstance(publishing_provenance, dict) or publishing_provenance.get(
            "git_dirty"
        ):
            raise ValueError("V4 releases require clean publishing provenance")
        model_provenance = release.get("model_source_provenance")
        if not isinstance(model_provenance, dict):
            raise TypeError("V4 releases require per-model source provenance")
        actual_model_ids = sorted(
            model_path.name
            for model_path in (path / "models").iterdir()
            if model_path.is_dir()
        )
        if release["model_ids"] != actual_model_ids:
            raise ValueError("Release model inventory does not match its directories")
        if sorted(model_provenance) != actual_model_ids:
            raise ValueError("Release provenance inventory does not match its models")
        for model_id in release["model_ids"]:
            model_manifest = validate_v4_model_result(
                path / "models" / model_id,
                spec,
                require_current_runtime=False,
            )
            if model_manifest["source_provenance"] != model_provenance[model_id]:
                raise ValueError(
                    f"{model_id}: source provenance does not match the release"
                )
            if model_manifest["source_provenance"]["git_dirty"]:
                raise ValueError(f"{model_id}: published model provenance is dirty")
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    args = parser.parse_args()
    release = validate_release(args.release)
    print(
        f"Validated {release.get('benchmark_id', release.get('comparison_id'))}"
        f"/{release.get('release_id', 'legacy')}: "
        f"{len(release['model_ids'])} models, {len(release['files'])} files"
    )


if __name__ == "__main__":
    main()
