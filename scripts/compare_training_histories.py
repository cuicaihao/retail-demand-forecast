"""Build a strict expanding-history versus fixed-window ablation table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_forecast.benchmark_protocol import load_benchmark_spec
from retail_forecast.benchmark_registry import generate_training_history_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--main-benchmark", required=True)
    parser.add_argument("--ablation-benchmark", required=True)
    args = parser.parse_args()
    output = generate_training_history_ablation(
        base_dir=args.source_root,
        main_spec=load_benchmark_spec(args.main_benchmark),
        ablation_spec=load_benchmark_spec(args.ablation_benchmark),
    )
    print(f"Training-history comparison written to {output}")


if __name__ == "__main__":
    main()
