"""Validate and publish one lightweight V4 benchmark snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from validate_benchmark_release import validate_release

from retail_forecast.benchmark_protocol import load_benchmark_spec
from retail_forecast.benchmark_registry import (
    PUBLISHED_DIR,
    benchmark_result_root,
    current_source_provenance,
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
RELEASE_ID_PATTERN = re.compile(r"^\d{8}-[a-z0-9]+(?:-[a-z0-9]+)*$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def published_release_root(benchmark_id: str, release_id: str) -> Path:
    return PUBLISHED_DIR / benchmark_id / "releases" / release_id


def _validate_release_id(release_id: str) -> str:
    if not RELEASE_ID_PATTERN.fullmatch(release_id):
        raise ValueError(
            "release_id must use YYYYMMDD-lowercase-kebab-case, "
            "for example 20261106-first-stage-v1"
        )
    return release_id


def publish(
    source_root: Path,
    benchmark_id: str,
    release_id: str,
    model_ids: list[str] | None = None,
    base_release_id: str | None = None,
) -> Path:
    spec = load_benchmark_spec(benchmark_id)
    release_id = _validate_release_id(release_id)
    source_provenance = current_source_provenance()
    if source_provenance["git_dirty"]:
        raise ValueError("Publishing requires a clean Git worktree")
    if source_provenance["git_commit"] == "unknown":
        raise ValueError("Publishing requires a resolvable Git commit")
    source = benchmark_result_root(source_root, benchmark_id)
    destination = published_release_root(benchmark_id, release_id)
    if destination.exists():
        raise FileExistsError(
            f"Published release already exists: {destination}. Release IDs are immutable."
        )
    available_source = {
        path.name: path for path in (source / "models").iterdir() if path.is_dir()
    }
    if not available_source:
        raise ValueError(f"No model results found under {source / 'models'}")
    requested_source = sorted(model_ids or available_source)
    if missing := sorted(set(requested_source) - set(available_source)):
        raise ValueError(f"Requested models are missing from the source run: {missing}")
    selected_models: dict[str, Path] = {}
    if base_release_id:
        base_release_path = published_release_root(benchmark_id, base_release_id)
        base_release = validate_release(base_release_path)
        if base_release.get("benchmark_id") != benchmark_id:
            raise ValueError("Base release belongs to a different benchmark")
        if base_release.get("protocol_hash") != spec.protocol_hash:
            raise ValueError("Base release uses a different benchmark protocol")
        selected_models.update(
            {
                model_id: base_release_path / "models" / model_id
                for model_id in base_release["model_ids"]
            }
        )
    for model_id in requested_source:
        model_path = available_source[model_id]
        manifest = validate_v4_model_result(
            model_path,
            spec,
            require_all_scenarios=True,
            require_current_runtime=True,
        )
        if manifest["source_provenance"] != source_provenance:
            raise ValueError(
                f"{model_path}: model provenance does not match the clean publishing tree"
            )
        selected_models[model_id] = model_path
    requested = sorted(selected_models)

    releases_root = destination.parent
    releases_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{release_id}-", dir=releases_root))
    try:
        for model_id in requested:
            model_path = selected_models[model_id]
            target = staging / "models" / model_id
            target.mkdir(parents=True)
            for filename in MODEL_FILES:
                shutil.copy2(model_path / filename, target / filename)
        generate_v4_summaries(
            result_root=staging,
            spec=spec,
            require_current_runtime=False,
        )

        files = sorted(path for path in staging.rglob("*") if path.is_file())
        release = {
            "release_schema_version": "benchmark-release-v1",
            "release_id": release_id,
            "benchmark_id": benchmark_id,
            "protocol_hash": spec.protocol_hash,
            "published_at": datetime.now(UTC).isoformat(),
            "published_by_source_provenance": source_provenance,
            "base_release_id": base_release_id,
            "model_ids": requested,
            "model_source_provenance": {
                model_id: json.loads(
                    (staging / "models" / model_id / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                )["source_provenance"]
                for model_id in requested
            },
            "files": {str(path.relative_to(staging)): _sha256(path) for path in files},
        }
        (staging / "release_manifest.json").write_text(
            json.dumps(release, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--models",
        help="Optional comma-separated release subset; defaults to all source models.",
    )
    parser.add_argument(
        "--base-release",
        help="Optional prior release whose reviewed models are carried forward.",
    )
    args = parser.parse_args()
    model_ids = (
        [model.strip() for model in args.models.split(",") if model.strip()]
        if args.models
        else None
    )
    destination = publish(
        args.source_root,
        args.benchmark,
        args.release,
        model_ids=model_ids,
        base_release_id=args.base_release,
    )
    print(f"Published immutable snapshot: {destination}")


if __name__ == "__main__":
    main()
