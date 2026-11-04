"""
Benchmark Registry and Standard Results Store for Retail Demand Forecasting.

This module provides:
1. Standard model results contract (predictions.csv, metrics_period.csv, metrics_annual.csv, manifest.json, model_report.md).
2. Week-aligned Rolling Window Generator (Mode '4week' non-overlapping & Mode 'weekly' rolling).
3. Model Configuration & Registry.
4. Global Multi-Model Aggregator & Leaderboard synthesis (0 retraining cost).
"""

from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    INPUT_HASHES,
    ROOT,
)

# Canonical Persistent Benchmark Store
BENCHMARK_DIR = ROOT / "data" / "benchmarks"
RESULTS_DIR = BENCHMARK_DIR / "results"
SUMMARY_DIR = BENCHMARK_DIR / "summary"


# =====================================================================
# 1. 统一接口与配置契约 (Unified Interfaces & Config Contract)
# =====================================================================


@dataclass
class ForecastOutput:
    """Standardized output container for a forecasting model on an evaluation horizon."""

    point_forecast: np.ndarray
    q10: np.ndarray | None = None
    q90: np.ndarray | None = None
    latency_ms: float = 0.0
    extra_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Standardized model specification and hyperparameter container."""

    model_id: str  # e.g. "xgboost_tweedie_p18", "seasonal_naive"
    model_name: str  # e.g. "XGBoost Tweedie (p=1.8)", "Seasonal Naive"
    family: (
        str  # "baseline", "tree_gbdt", "statistical", "neural", "foundation", "automl"
    )
    variant: str = "default"  # "default", "tweedie_p11", "lossguide", etc.
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""


class BaseRetailForecaster(ABC):
    """Abstract base class that every benchmark model should implement."""

    def __init__(self, name: str, family: str):
        self.name = name
        self.family = family

    @abstractmethod
    def fit(self, history: pd.DataFrame, scenario: str) -> None:
        """Fit model on historical observations."""

    @abstractmethod
    def predict(self, future: pd.DataFrame, scenario: str) -> ForecastOutput:
        """Generate forecasts for future period."""


# =====================================================================
# 2. 周对齐前向滚动窗口生成器 (Week-Aligned Rolling Window Generator)
# =====================================================================


@dataclass
class RollingFold:
    """A single evaluation fold strictly aligned to Monday-Sunday complete weeks."""

    fold_id: int  # 1..13 or 1..50
    period_name: str  # "P01", "P02", ... or "Step01", ...
    train_start: pd.Timestamp  # Monday
    train_end: pd.Timestamp  # Sunday
    test_start: pd.Timestamp  # Monday
    test_end: pd.Timestamp  # Sunday
    train_weeks: int
    eval_weeks: int
    n_test_days: int


def generate_week_aligned_folds(
    mode: str = "4week",
    train_weeks: int = 24,
    horizon_days: int = 28,
    target_year: int = 2026,
    include_p14: bool = False,
) -> list[RollingFold]:
    """
    Generate strictly Monday-to-Sunday week-aligned evaluation folds.

    Mode '4week' (Non-overlapping blocks):
      - 13 non-overlapping periods of 28 days (4 complete weeks) = 364 calendar days.
      - P01 (W1-W4): 2025-12-29 (Mon) to 2026-01-25 (Sun).
      - P13 (W49-W52): 2026-11-30 (Mon) to 2026-12-27 (Sun).
      - If include_p14=True: P14 (W53-W56): 2026-12-28 (Mon) to 2027-01-24 (Sun).
      - Prior training window: exactly `train_weeks` complete weeks (e.g. 24 weeks = 168 days).

    Mode 'weekly' (Rolling 1-week stepping):
      - 50 rolling steps, each stepping 1 week (7 days) forward, predicting future 28 days.
      - Step 01: 2025-12-29 to 2026-01-25.
      - Step 50: 2026-12-07 to 2027-01-03 (covers through 2026-12-31).
    """
    w1_start = pd.Timestamp("2025-12-29")  # Monday
    assert w1_start.day_name() == "Monday", "W1 start must be Monday"

    eval_weeks = horizon_days // 7
    folds = []

    if mode == "4week":
        num_periods = 14 if include_p14 else 13
        for k in range(num_periods):
            test_start = w1_start + pd.Timedelta(days=k * horizon_days)
            test_end = test_start + pd.Timedelta(days=horizon_days - 1)
            train_end = test_start - pd.Timedelta(days=1)
            train_start = (
                train_end - pd.Timedelta(weeks=train_weeks) + pd.Timedelta(days=1)
            )

            assert test_start.day_name() == "Monday"
            assert test_end.day_name() == "Sunday"
            assert train_start.day_name() == "Monday"
            assert train_end.day_name() == "Sunday"

            folds.append(
                RollingFold(
                    fold_id=k + 1,
                    period_name=f"P{k + 1:02d}",
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_weeks=train_weeks,
                    eval_weeks=eval_weeks,
                    n_test_days=horizon_days,
                )
            )

    elif mode == "weekly":
        num_steps = 50
        for s in range(num_steps):
            test_start = w1_start + pd.Timedelta(weeks=s)
            test_end = test_start + pd.Timedelta(days=horizon_days - 1)
            train_end = test_start - pd.Timedelta(days=1)
            train_start = (
                train_end - pd.Timedelta(weeks=train_weeks) + pd.Timedelta(days=1)
            )

            assert test_start.day_name() == "Monday"
            assert test_end.day_name() == "Sunday"

            folds.append(
                RollingFold(
                    fold_id=s + 1,
                    period_name=f"Step{s + 1:02d}",
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_weeks=train_weeks,
                    eval_weeks=eval_weeks,
                    n_test_days=horizon_days,
                )
            )
    else:
        raise ValueError(f"Unsupported mode: '{mode}'. Expected '4week' or 'weekly'.")

    return folds


# =====================================================================
# 3. 评测计算工具 (Standard Metrics Engine)
# =====================================================================


def compute_standard_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    q10: np.ndarray | pd.Series | None = None,
    q90: np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    """Compute standard retail benchmark metrics aligned across the entire repository."""
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    assert len(y) == len(p), f"Length mismatch: {len(y)} vs {len(p)}"

    n = len(y)
    if n == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "WAPE": np.nan,
            "Bias": np.nan,
            "ZeroAccuracy": np.nan,
            "Coverage80": np.nan,
            "n": 0,
        }

    err = p - y
    abs_err = np.abs(err)
    total_sales = float(y.sum())

    mae = float(abs_err.mean())
    rmse = float(np.sqrt((err**2).mean()))
    wape = float(abs_err.sum() / total_sales) if total_sales > 0 else np.nan
    bias = float(err.sum() / total_sales) if total_sales > 0 else np.nan

    # Zero accuracy for intermittent evaluation (percentage of y==0 where pred < 0.05)
    zero_mask = y == 0
    if zero_mask.sum() > 0:
        zero_accuracy = float((p[zero_mask] < 0.05).mean())
    else:
        zero_accuracy = np.nan

    coverage = np.nan
    if q10 is not None and q90 is not None:
        ql = np.asarray(q10, dtype=float).reshape(-1)
        qh = np.asarray(q90, dtype=float).reshape(-1)
        in_interval = (y >= ql) & (y <= qh)
        coverage = float(in_interval.mean())

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
        "Bias": bias,
        "ZeroAccuracy": zero_accuracy,
        "Coverage80": coverage,
        "n": n,
    }


# =====================================================================
# 4. 独立存储契约与单模型报告 (Persistence & Single-Model Reporting)
# =====================================================================


def get_benchmark_dirs(eval_mode: str = "4week") -> tuple[Path, Path]:
    """
    Return (results_dir, summary_dir) for a specific evaluation mode.
    - '4week': canonical 'data/benchmarks/results/' and 'data/benchmarks/summary/'
    - other modes (e.g. 'weekly'): isolated 'data/benchmarks/results_<mode>/'
    """
    if eval_mode == "4week":
        return RESULTS_DIR, SUMMARY_DIR
    res_dir = BENCHMARK_DIR / f"results_{eval_mode}"
    sum_dir = BENCHMARK_DIR / f"summary_{eval_mode}"
    return res_dir, sum_dir


def get_model_result_dir(model_id: str, eval_mode: str = "4week") -> Path:
    """Return the dedicated storage path for a model's benchmark results under the given eval mode."""
    results_dir, _ = get_benchmark_dirs(eval_mode)
    model_dir = results_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _get_current_git_commit() -> str:
    """Retrieve current Git commit hash for provenance."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def generate_single_model_report(
    model_config: ModelConfig,
    metrics_annual_df: pd.DataFrame,
    metrics_period_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate self-contained markdown validation report for a single model."""
    lines = [
        f"# 零售需求预测模型单体验证报告: {model_config.model_name}",
        "",
        f"> **Model ID**: `{model_config.model_id}`  ",
        f"> **Model Family**: `{model_config.family}`  ",
        f"> **Variant**: `{model_config.variant}`  ",
        f"> **Generated At**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        "",
        "## 一、 2026 全年综合表现 (Annual Summary)",
        "",
        "| 场景 (Scenario) | 真实总销量 | 全年 WAPE (%) | 全年 MAE | 全年 RMSE | 全年 Bias (%) | 零值命中率 | 周期 WAPE 均值 | 周期 WAPE 标准差 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for _, r in metrics_annual_df.iterrows():
        tot = f"{r.get('Total_Units', 0):,.1f}"
        wape_str = f"**{r['WAPE'] * 100:.2f}%**" if pd.notna(r["WAPE"]) else "N/A"
        bias_str = f"{r['Bias'] * 100:+.2f}%" if pd.notna(r["Bias"]) else "N/A"
        zero_str = (
            f"{r['ZeroAccuracy'] * 100:.1f}%"
            if pd.notna(r.get("ZeroAccuracy"))
            else "-"
        )
        p_mean = (
            f"{r['Period_WAPE_Mean'] * 100:.2f}%"
            if pd.notna(r.get("Period_WAPE_Mean"))
            else "-"
        )
        p_std = (
            f"{r['Period_WAPE_Std'] * 100:.2f}%"
            if pd.notna(r.get("Period_WAPE_Std"))
            else "-"
        )
        lines.append(
            f"| `{r['scenario']}` | {tot} | {wape_str} | {r['MAE']:.4f} | {r['RMSE']:.4f} | {bias_str} | {zero_str} | {p_mean} | {p_std} |"
        )

    lines.extend(
        [
            "",
            "## 二、 周期稳定性分析 (Period Volatility)",
            "下表展示模型在 2026 年各周期（4 周分块或周滑动步进）的 WAPE 演变轨迹：",
            "",
        ]
    )

    if not metrics_period_df.empty and "period" in metrics_period_df.columns:
        p_pivot = metrics_period_df.pivot(
            index="period", columns="scenario", values="WAPE"
        )
        p_pivot = p_pivot * 100
        p_pivot = p_pivot.reset_index()
        lines.append(
            "| 周期 (Period) | "
            + " | ".join([f"`{c}` (%)" for c in p_pivot.columns if c != "period"])
            + " |"
        )
        lines.append(
            "| :--- | "
            + " | ".join([":---" for c in p_pivot.columns if c != "period"])
            + " |"
        )
        for _, row in p_pivot.iterrows():
            vals = [
                f"{row[c]:.2f}%" if pd.notna(row[c]) else "-"
                for c in p_pivot.columns
                if c != "period"
            ]
            lines.append(f"| **{row['period']}** | " + " | ".join(vals) + " |")

    lines.extend(
        [
            "",
            "## 三、 模型配置与超参数 (Hyperparameters)",
            "```json",
            json.dumps(model_config.params, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_rolling_benchmark_results(
    model_config: ModelConfig,
    predictions_df: pd.DataFrame,
    metrics_period_df: pd.DataFrame,
    metrics_annual_df: pd.DataFrame,
    eval_mode: str = "4week",
    train_weeks: int = 24,
    extra_manifest: dict[str, Any] | None = None,
) -> Path:
    """
    Persist standard 5-contract benchmark results for a single model:
    1. predictions.csv: Full horizon predictions per fold.
    2. metrics_period.csv: Metrics across all periods.
    3. metrics_annual.csv: Cumulative annual benchmark summary.
    4. manifest.json: Metadata, input hashes, Git commit.
    5. model_report.md: Standalone markdown validation report.
    """
    out_dir = get_model_result_dir(model_config.model_id, eval_mode=eval_mode)

    predictions_df.to_csv(out_dir / "predictions.csv", index=False)
    metrics_period_df.to_csv(out_dir / "metrics_period.csv", index=False)
    metrics_annual_df.to_csv(out_dir / "metrics_annual.csv", index=False)

    # Also save legacy metrics.csv for backwards compatibility
    metrics_annual_df.to_csv(out_dir / "metrics.csv", index=False)

    manifest = {
        "model_id": model_config.model_id,
        "model_name": model_config.model_name,
        "family": model_config.family,
        "variant": model_config.variant,
        "params": model_config.params,
        "eval_mode": eval_mode,
        "train_weeks": train_weeks,
        "updated_at": datetime.now(UTC).isoformat(),
        "git_commit": _get_current_git_commit(),
        "input_hashes": INPUT_HASHES,
        "protocol_version": "retail-benchmark-v3-week-aligned",
    }
    if extra_manifest:
        manifest.update(extra_manifest)

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Generate standalone report
    generate_single_model_report(
        model_config=model_config,
        metrics_annual_df=metrics_annual_df,
        metrics_period_df=metrics_period_df,
        output_path=out_dir / "model_report.md",
    )

    print(f"[{model_config.model_id}] Results successfully persisted to {out_dir}")
    return out_dir


def save_benchmark_results(
    model_key: str,
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    manifest_meta: dict[str, Any] | None = None,
) -> Path:
    """Legacy compatibility persistence function."""
    out_dir = get_model_result_dir(model_key)
    predictions_df.to_csv(out_dir / "predictions.csv", index=False)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    meta = {
        "model_key": model_key,
        "updated_at": datetime.now(UTC).isoformat(),
        "input_hashes": INPUT_HASHES,
        "protocol_version": "retail-benchmark-v2",
    }
    if manifest_meta:
        meta.update(manifest_meta)

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[{model_key}] Legacy results saved to {out_dir}")
    return out_dir


# =====================================================================
# 5. 全模型聚合与全景排行榜 (Global Aggregator & Leaderboard)
# =====================================================================


def load_all_benchmark_results(
    eval_mode: str = "4week",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all available model predictions and metrics from the benchmark result store.
    Returns (all_annual_df, all_period_df, all_predictions_df).
    """
    res_dir, _ = get_benchmark_dirs(eval_mode)
    if not res_dir.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_annual = []
    all_period = []
    all_preds = []

    for model_path in sorted(res_dir.iterdir()):
        if not model_path.is_dir() or model_path.name.startswith("."):
            continue

        ann_file = model_path / "metrics_annual.csv"
        leg_file = model_path / "metrics.csv"
        per_file = model_path / "metrics_period.csv"
        pred_file = model_path / "predictions.csv"

        if ann_file.is_file():
            df_a = pd.read_csv(ann_file)
            if "model_id" not in df_a.columns:
                df_a["model_id"] = model_path.name
            all_annual.append(df_a)
        elif leg_file.is_file():
            df_l = pd.read_csv(leg_file)
            if "model_id" not in df_l.columns:
                df_l["model_id"] = model_path.name
            all_annual.append(df_l)

        if per_file.is_file():
            df_p = pd.read_csv(per_file)
            if "model_id" not in df_p.columns:
                df_p["model_id"] = model_path.name
            all_period.append(df_p)

        if pred_file.is_file():
            df_pred = pd.read_csv(pred_file)
            if "model_id" not in df_pred.columns:
                df_pred["model_id"] = model_path.name
            all_preds.append(df_pred)

    df_annual = (
        pd.concat(all_annual, ignore_index=True) if all_annual else pd.DataFrame()
    )
    df_period = (
        pd.concat(all_period, ignore_index=True) if all_period else pd.DataFrame()
    )
    df_preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    return df_annual, df_period, df_preds


def generate_leaderboard(
    eval_mode: str = "4week", output_dir: Path | None = None
) -> pd.DataFrame:
    """
    Generate unified multi-model leaderboard from persisted benchmark results.
    Ranks models by mean annual test WAPE, MAE, RMSE across all 4 scenarios.
    """
    df_annual, df_period, _ = load_all_benchmark_results(eval_mode=eval_mode)
    if df_annual.empty:
        print(
            f"No benchmark results found to generate leaderboard for mode: {eval_mode}."
        )
        return pd.DataFrame()

    _, sum_dir = get_benchmark_dirs(eval_mode)
    out = output_dir or sum_dir
    out.mkdir(parents=True, exist_ok=True)

    # Model identifier column
    model_col = "model_name" if "model_name" in df_annual.columns else "model"
    if model_col not in df_annual.columns:
        df_annual[model_col] = df_annual["model_id"]

    # 1. 场景维度排行榜 (Scenario Leaderboard)
    scenario_summary = (
        df_annual.groupby(["scenario", model_col], as_index=False)
        .agg(
            WAPE=("WAPE", "mean"),
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            Bias=("Bias", "mean"),
            ZeroAccuracy=(
                "ZeroAccuracy",
                "mean" if "ZeroAccuracy" in df_annual.columns else "first",
            ),
            Period_WAPE_Mean=(
                "Period_WAPE_Mean",
                "mean" if "Period_WAPE_Mean" in df_annual.columns else "first",
            ),
            Period_WAPE_Std=(
                "Period_WAPE_Std",
                "mean" if "Period_WAPE_Std" in df_annual.columns else "first",
            ),
        )
        .sort_values(["scenario", "WAPE"])
    )
    scenario_summary.to_csv(out / "scenario_leaderboard.csv", index=False)

    # 2. 全景综合天梯榜 (Global Leaderboard)
    global_leaderboard = (
        df_annual.groupby(model_col, as_index=False)
        .agg(
            Mean_WAPE=("WAPE", "mean"),
            Mean_MAE=("MAE", "mean"),
            Mean_RMSE=("RMSE", "mean"),
            Mean_Bias=("Bias", "mean"),
            Mean_ZeroAccuracy=(
                "ZeroAccuracy",
                "mean" if "ZeroAccuracy" in df_annual.columns else "first",
            ),
            Stability_Std=(
                "Period_WAPE_Std",
                "mean" if "Period_WAPE_Std" in df_annual.columns else "first",
            ),
            Scenarios_Evaluated=("scenario", "nunique"),
        )
        .sort_values("Mean_WAPE")
    )
    global_leaderboard.to_csv(out / "annual_leaderboard.csv", index=False)
    global_leaderboard.to_csv(out / "global_leaderboard.csv", index=False)

    # 3. 周期演变大表 (Period Trajectories)
    if not df_period.empty:
        df_period.to_csv(out / "period_trajectories.csv", index=False)

    print("\n" + "=" * 80)
    print("RETAIL DEMAND FORECASTING BENCHMARK - GLOBAL ANNUAL LEADERBOARD (2026)")
    print("=" * 80)
    print(global_leaderboard.to_string(index=False))
    print(f"\nLeaderboards successfully saved to: {out}")
    return global_leaderboard
