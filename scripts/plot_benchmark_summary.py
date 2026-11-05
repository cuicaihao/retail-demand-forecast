"""
Academic and Blog Publication Visualization Suite for Retail Demand Forecasting Benchmark.

Generates:
1. fig-01-retail-week-leaderboard-all-scenarios (.png & .webp)
2. fig-02-13-period-wape-trajectories (.png & .webp)
3. fig-03-gbdt-tweedie-ablation-deepdive (.png & .webp)
4. fig-04-intermittent-zero-recall (.png & .webp)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_DIR = ROOT / "data" / "benchmarks" / "published"
IMAGE_DIR = ROOT / "images" / "benchmarks"


def set_academic_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.edgecolor": "#CCCCCC",
            "axes.linewidth": 0.8,
            "grid.color": "#E5E5E5",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.autolayout": False,
        }
    )


def save_dual_format(fig: plt.Figure, base_filename: str):
    """Save figure in both high-res PNG and modern WebP formats."""
    for ext in ["png", "webp"]:
        out_path = IMAGE_DIR / f"{base_filename}.{ext}"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"Saved: {out_path}")


def plot_fig01_annual_leaderboard(df_scenario: pd.DataFrame):
    """Fig 1: Cross-family annual WAPE comparison across all 4 scenarios."""
    print("Generating Fig 1: Annual Leaderboard by Scenario...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    axes = axes.flatten()

    scenarios = ["grocery", "wine", "garden", "luxury"]
    sc_titles = {
        "grocery": "Grocery (High-frequency FMCG, Continuous ~110 units)",
        "wine": "Wine (Poisson Count Pulses & Promos ~10 units)",
        "garden": "Garden (Annual Sinusoidal Seasonality ~65 units)",
        "luxury": "Luxury Handbags (94% Zero-Inflated Intermittent)",
    }

    palette = {
        "Seasonal Naive (7d)": "#7F7F7F",
        "Moving Average (14d)": "#A6A6A6",
        "Prophet": "#9467BD",
        "Regression + ARIMA (1,0,1)": "#BCBD22",
        "Decision Tree": "#8C564B",
        "Random Forest": "#E377C2",
        "LightGBM Default (MSE)": "#1F77B4",
        "XGBoost Default (MSE)": "#2CA02C",
        "CatBoost Default (MAE)": "#17BECF",
        "LightGBM Tweedie (p=1.8)": "#FF7F0E",
        "XGBoost Tweedie (p=1.8)": "#D62728",
    }

    for idx, sc in enumerate(scenarios):
        ax = axes[idx]
        sub = df_scenario[
            (df_scenario["scenario"] == sc)
            & (df_scenario["comparison_role"] == "breadth")
        ].copy()
        sub["WAPE_pct"] = sub["WAPE"] * 100
        sub = sub.sort_values("WAPE_pct", ascending=True)

        colors = [palette.get(m, "#4A90E2") for m in sub["model_name"]]
        bars = ax.barh(
            sub["model_name"], sub["WAPE_pct"], color=colors, alpha=0.88, height=0.68
        )

        ax.set_title(sc_titles[sc], fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("WAPE (%) [Lower is Better]", fontsize=9)
        ax.grid(True, axis="x", linestyle=":", alpha=0.6)

        max_val = sub["WAPE_pct"].max()
        ax.set_xlim(0, max_val * 1.15)

        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + (max_val * 0.015),
                bar.get_y() + bar.get_height() / 2,
                f"{w:.2f}%",
                va="center",
                fontsize=8.5,
                fontweight="bold"
                if "Tweedie" in bar.get_label() or w == sub["WAPE_pct"].min()
                else "normal",
            )

    fig.suptitle(
        "Melbourne Retail Demand Forecasting 2026 Retail-Week Benchmark: Model Families & Objectives",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    save_dual_format(fig, "fig-01-retail-week-leaderboard-all-scenarios")
    plt.close(fig)


def plot_fig02_trajectories(df_period: pd.DataFrame):
    """Fig 2: 13-period WAPE stability across 2026."""
    print("Generating Fig 2: 13-Period Trajectories...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes = axes.flatten()

    scenarios = ["grocery", "wine", "garden", "luxury"]
    key_models = [
        ("Seasonal Naive (7d)", "#7F7F7F", "--", "o"),
        ("Prophet", "#9467BD", "-.", "^"),
        ("Regression + ARIMA (1,0,1)", "#BCBD22", "-", "P"),
        ("LightGBM Default (MSE)", "#1F77B4", "-", "s"),
        ("XGBoost Default (MSE)", "#2CA02C", "-", "v"),
        ("XGBoost Tweedie (p=1.8)", "#D62728", "-", "D"),
    ]

    for idx, sc in enumerate(scenarios):
        ax = axes[idx]
        sub = df_period[df_period["scenario"] == sc]

        for model_name, color, ls, marker in key_models:
            m_sub = sub[sub["model_name"] == model_name].sort_values("period")
            if not m_sub.empty:
                wape_pct = m_sub["WAPE"] * 100
                ax.plot(
                    m_sub["period"],
                    wape_pct,
                    label=model_name,
                    color=color,
                    linestyle=ls,
                    marker=marker,
                    linewidth=1.8,
                    markersize=5,
                    alpha=0.9,
                )

        ax.set_title(
            f"2026 Periodic Stability: {sc.upper()}", fontsize=11, fontweight="bold"
        )
        ax.set_ylabel("Period WAPE (%)", fontsize=9)
        ax.set_xlabel("4-Week Fiscal Period (P01 to P13)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Cross-Seasonal Generalization: 13 Non-Overlapping 4-Week Periods Across 2026",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    save_dual_format(fig, "fig-02-13-period-wape-trajectories")
    plt.close(fig)


def plot_fig03_tweedie_ablation(df_scenario: pd.DataFrame, df_ablation: pd.DataFrame):
    """Fig 3: GBDT Tweedie variance power p ablation deep dive."""
    print("Generating Fig 3: GBDT Tweedie Ablation...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    xgb_default = df_scenario[df_scenario["model_id"] == "xgboost_default"]
    xgb_tweedie = df_ablation[
        df_ablation["model_id"].str.startswith("xgboost_tweedie_")
    ]
    xgb = pd.concat([xgb_default, xgb_tweedie], ignore_index=True)
    label_by_id = {
        "xgboost_default": "XGBoost Default (MSE)",
        "xgboost_tweedie_p11": "XGBoost Tweedie (p=1.1)",
        "xgboost_tweedie_p12": "XGBoost Tweedie (p=1.2)",
        "xgboost_tweedie_p15": "XGBoost Tweedie (p=1.5)",
        "xgboost_tweedie_p18": "XGBoost Tweedie (p=1.8)",
    }
    expected_ids = set(label_by_id)
    for scenario_name in ("grocery", "luxury"):
        actual_ids = set(xgb.loc[xgb["scenario"] == scenario_name, "model_id"])
        if actual_ids != expected_ids:
            raise ValueError(
                f"Incomplete XGBoost ablation for {scenario_name}: "
                f"expected {sorted(expected_ids)}, got {sorted(actual_ids)}"
            )
    xgb["plot_label"] = xgb["model_id"].map(label_by_id)

    # Luxury scenario
    xgb_lux = xgb[xgb["scenario"] == "luxury"].sort_values("WAPE")
    xgb_lux["WAPE_pct"] = xgb_lux["WAPE"] * 100

    bars1 = ax1.bar(
        xgb_lux["plot_label"],
        xgb_lux["WAPE_pct"],
        color=[
            "#D62728" if "p=1.8" in m else "#2CA02C" if "Default" in m else "#FF7F0E"
            for m in xgb_lux["plot_label"]
        ],
        alpha=0.88,
        width=0.55,
    )
    ax1.set_title(
        "Luxury Handbags: 94% Zero-Inflated Demand (WAPE %)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_ylabel("Pooled retail-week WAPE (%) [Lower is Better]", fontsize=9)
    ax1.tick_params(axis="x", rotation=25)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 2.5,
            f"{h:.1f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    # Grocery scenario
    xgb_groc = xgb[xgb["scenario"] == "grocery"].sort_values("WAPE")
    xgb_groc["WAPE_pct"] = xgb_groc["WAPE"] * 100

    bars2 = ax2.bar(
        xgb_groc["plot_label"],
        xgb_groc["WAPE_pct"],
        color=[
            "#D62728" if "p=1.2" in m else "#2CA02C" if "Default" in m else "#FF7F0E"
            for m in xgb_groc["plot_label"]
        ],
        alpha=0.88,
        width=0.55,
    )
    ax2.set_title(
        "Grocery FMCG: Continuous Demand (WAPE %)", fontsize=11, fontweight="bold"
    )
    ax2.set_ylabel("Pooled retail-week WAPE (%) [Lower is Better]", fontsize=9)
    ax2.tick_params(axis="x", rotation=25)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.1,
            f"{h:.2f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle(
        "Ablation Deep Dive: Impact of Tweedie Variance Power p on XGBoost (2026 Walk-Forward)",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    save_dual_format(fig, "fig-03-gbdt-tweedie-ablation-deepdive")
    plt.close(fig)


def plot_fig04_zero_recall(df_scenario: pd.DataFrame):
    """Fig 4: Explicit zero-day recall on intermittent demand."""
    print("Generating Fig 4: Zero Recall Comparison...")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    lux = df_scenario[df_scenario["scenario"] == "luxury"].copy()
    lux = lux.dropna(subset=["ZeroRecallAt05"]).sort_values(
        "ZeroRecallAt05", ascending=True
    )
    lux["Zero_pct"] = lux["ZeroRecallAt05"] * 100

    colors = [
        "#D62728"
        if "Tweedie (p=1.8)" in m
        else "#2CA02C"
        if "Default" in m
        else "#4A90E2"
        for m in lux["model_name"]
    ]
    bars = ax.barh(
        lux["model_name"], lux["Zero_pct"], color=colors, alpha=0.88, height=0.6
    )

    ax.set_title(
        "Zero-Demand Recall on Luxury Handbags (Prediction < 0.5 when Sales = 0)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("ZeroRecall@0.5 (%) [Higher is Better]", fontsize=9)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax.set_xlim(0, 108)

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%",
            va="center",
            fontsize=8.5,
            fontweight="bold" if w > 80 else "normal",
        )

    save_dual_format(fig, "fig-04-intermittent-zero-recall")
    plt.close(fig)


def main():
    global IMAGE_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--release", required=True)
    args = parser.parse_args()
    set_academic_style()
    summary_dir = PUBLISHED_DIR / args.benchmark / "releases" / args.release / "summary"
    IMAGE_DIR = ROOT / "images" / "benchmarks" / args.benchmark / args.release
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    sc_file = summary_dir / "scenario_leaderboard.csv"
    per_file = summary_dir / "period_trajectories.csv"
    ablation_file = summary_dir / "configuration_ablations.csv"

    if not sc_file.is_file():
        print(
            f"Error: {sc_file} not found. Publish benchmark {args.benchmark!r} first."
        )
        return

    df_scenario = pd.read_csv(sc_file)
    df_period = pd.read_csv(per_file) if per_file.is_file() else pd.DataFrame()
    df_ablation = (
        pd.read_csv(ablation_file) if ablation_file.is_file() else pd.DataFrame()
    )

    plot_fig01_annual_leaderboard(df_scenario)
    if not df_period.empty:
        plot_fig02_trajectories(df_period)
    if df_ablation.empty:
        raise FileNotFoundError(f"Required ablation summary not found: {ablation_file}")
    plot_fig03_tweedie_ablation(df_scenario, df_ablation)
    plot_fig04_zero_recall(df_scenario)
    print(f"\nFigures rendered to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
