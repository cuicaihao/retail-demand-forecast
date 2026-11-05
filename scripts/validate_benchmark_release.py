"""Recompute a published benchmark's file hashes and V4 contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_forecast.benchmark_protocol import load_benchmark_spec
from retail_forecast.benchmark_registry import validate_v4_model_result


def validate_release(path: Path) -> dict:
    manifest_path = path / "release_manifest.json"
    release = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in release["files"].items():
        file_path = path / relative
        if not file_path.is_file():
            raise ValueError(f"Release file is missing: {relative}")
        actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Release hash mismatch: {relative}")
    if release.get("protocol_hash"):
        spec = load_benchmark_spec(release["benchmark_id"])
        if release["protocol_hash"] != spec.protocol_hash:
            raise ValueError("Release protocol hash does not match declaration")
        for model_id in release["model_ids"]:
            validate_v4_model_result(path / "models" / model_id, spec)
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    args = parser.parse_args()
    release = validate_release(args.release)
    print(
        f"Validated {release['benchmark_id']}: "
        f"{len(release['model_ids'])} models, {len(release['files'])} files"
    )


if __name__ == "__main__":
    main()
