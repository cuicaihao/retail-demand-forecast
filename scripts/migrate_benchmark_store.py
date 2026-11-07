"""Migrate mixed live V2/V3 results into isolated historical snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "data" / "benchmarks"
LIVE_RESULTS = BENCHMARKS / "results"
ARCHIVE_V2 = BENCHMARKS / "archive_single_cutoff_28d"
V3_ID = "2026-retail-week-24w-v3"
V3_DESTINATION = BENCHMARKS / "published" / V3_ID


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same_payload(left: Path, right: Path) -> bool:
    names = ("manifest.json", "metrics.csv", "predictions.csv")
    return all(
        (left / name).is_file()
        and (right / name).is_file()
        and _sha256(left / name) == _sha256(right / name)
        for name in names
    )


def migration_inventory() -> tuple[list[Path], list[Path]]:
    duplicate_v2: list[Path] = []
    v3_models: list[Path] = []
    for path in sorted(item for item in LIVE_RESULTS.iterdir() if item.is_dir()):
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        protocol = manifest.get("protocol_version")
        if protocol == "retail-benchmark-v2":
            archived = ARCHIVE_V2 / path.name
            if not _same_payload(path, archived):
                raise ValueError(f"V2 live result is not identical to archive: {path}")
            duplicate_v2.append(path)
        elif protocol == "retail-benchmark-v3-week-aligned":
            v3_models.append(path)
        else:
            raise ValueError(f"Unknown live benchmark protocol in {path}: {protocol}")
    return duplicate_v2, v3_models


def apply_migration() -> Path:
    duplicate_v2, v3_models = migration_inventory()
    if V3_DESTINATION.exists():
        raise FileExistsError(f"Migration destination already exists: {V3_DESTINATION}")
    (V3_DESTINATION / "models").mkdir(parents=True)
    for source in v3_models:
        shutil.copytree(source, V3_DESTINATION / "models" / source.name)
    if (BENCHMARKS / "summary").is_dir():
        shutil.copytree(BENCHMARKS / "summary", V3_DESTINATION / "summary")

    files = sorted(path for path in V3_DESTINATION.rglob("*") if path.is_file())
    release = {
        "benchmark_id": V3_ID,
        "protocol_version": "retail-benchmark-v3-week-aligned",
        "status": "historical-fixed-24-week-snapshot",
        "warning": "Not comparable with V4 expanding-history results.",
        "migrated_at": datetime.now(UTC).isoformat(),
        "model_ids": [path.name for path in v3_models],
        "files": {
            str(path.relative_to(V3_DESTINATION)): _sha256(path) for path in files
        },
    }
    (V3_DESTINATION / "release_manifest.json").write_text(
        json.dumps(release, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for path in [*duplicate_v2, *v3_models]:
        shutil.rmtree(path)
    if (BENCHMARKS / "summary").is_dir():
        shutil.rmtree(BENCHMARKS / "summary")
    return V3_DESTINATION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    duplicate_v2, v3_models = migration_inventory()
    print(f"Verified archived V2 duplicates: {len(duplicate_v2)}")
    print(f"V3 models ready for immutable snapshot: {len(v3_models)}")
    if args.apply:
        print(f"Migrated to: {apply_migration()}")
    else:
        print("Dry run only; pass --apply to migrate.")


if __name__ == "__main__":
    main()
