"""Plot paired WAPE changes for a published training-history comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from validate_benchmark_release import validate_release

from retail_forecast.benchmark_registry import PUBLISHED_DIR

IMAGE_ROOT = ROOT / "images" / "benchmarks" / "comparisons"


def _display_name(value: str) -> str:
    return (
        value.replace("Xgboost", "XGBoost")
        .replace("Lightgbm", "LightGBM")
        .replace("Catboost", "CatBoost")
    )


def plot_comparison(comparison_id: str, release_id: str) -> Path:
    release_root = (
        PUBLISHED_DIR / "comparisons" / comparison_id / "releases" / release_id
    )
    validate_release(release_root)
    comparison = pd.read_csv(release_root / "training_history_ablation.csv")
    comparison = comparison[comparison.comparison_role.eq("breadth")].copy()
    comparison["model_name"] = comparison["model_name"].map(_display_name)
    comparison["delta_pp"] = comparison["WAPE_delta_24w_minus_expanding"] * 100

    scenarios = ["grocery", "wine", "garden", "luxury"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    for ax, scenario in zip(axes.flat, scenarios, strict=True):
        frame = comparison[comparison.scenario.eq(scenario)].sort_values("delta_pp")
        colors = ["#2166AC" if value >= 0 else "#B2182B" for value in frame.delta_pp]
        bars = ax.barh(frame.model_name, frame.delta_pp, color=colors, alpha=0.88)
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_title(scenario.capitalize(), fontweight="bold")
        ax.set_xlabel("24-week WAPE minus expanding-history WAPE (percentage points)")
        ax.grid(True, axis="x", linestyle=":", alpha=0.55)
        offset = max(abs(frame.delta_pp).max() * 0.02, 0.08)
        for bar, value in zip(bars, frame.delta_pp, strict=True):
            ax.text(
                value + (offset if value >= 0 else -offset),
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.2f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8,
            )
        if scenario == "luxury":
            ax.text(
                0.01,
                0.01,
                "WAPE view only; consult event recall and F1 tables.",
                transform=ax.transAxes,
                fontsize=8,
                color="#555555",
            )

    fig.suptitle(
        "Training-History Ablation: 24 Weeks vs Expanding History\n"
        "Positive values favor expanding history; representative breadth configurations only",
        fontsize=14,
        fontweight="bold",
    )
    output = IMAGE_ROOT / comparison_id / release_id
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "webp"):
        path = output / f"fig-01-training-history-wape-delta.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--release", required=True)
    args = parser.parse_args()
    output = plot_comparison(args.comparison, args.release)
    print(f"Rendered training-history comparison figure: {output}")


if __name__ == "__main__":
    main()
