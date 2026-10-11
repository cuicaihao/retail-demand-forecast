from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a run before review or promotion."
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    runs_root = (ROOT / "runs").resolve()
    if runs_root not in run_dir.parents:
        raise SystemExit(f"run_dir must be below {runs_root}")
    if not run_dir.is_dir():
        raise SystemExit(f"missing run directory: {run_dir}")

    manifests = sorted(run_dir.rglob("*manifest.json"))
    if not manifests:
        raise SystemExit("no manifest found")
    teaching = ROOT / "data" / "teaching"
    failures: list[str] = []
    for manifest_path in manifests:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = record.get("input_sha256") or record.get("hashes") or {}
        for name, digest in expected.items():
            source = teaching / name
            if not source.is_file() or sha256(source) != digest:
                failures.append(f"{manifest_path}: input mismatch for {name}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"validated {len(manifests)} manifest(s) in {run_dir}")


if __name__ == "__main__":
    main()
