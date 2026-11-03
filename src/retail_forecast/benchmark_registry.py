"""
Benchmark Registry and Standard Results Store for Retail Demand Forecasting.

This module provides:
1. Standard model results contract (predictions.csv, metrics.csv, manifest.json).
2. Model Registry: lists all available models and allows running individual or subsets of models.
3. Aggregator: joins all registered model outputs into a unified leaderboard and comparison tables.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .common import (
    CPU_THREADS,
    DATA_DIR,
    ENCODER_LENGTH,
    HORIZON,
    INPUT_HASHES,
    ROOT,
    SCENARIOS,
    SEED,
    TRAIN_END,
    datasets,
    metrics,
    sales_prediction,
    valid_labels,
    windows,
    vic_holidays,
)

# Canonical Persistent Benchmark Store
BENCHMARK_DIR = ROOT / "data" / "benchmarks"
RESULTS_DIR = BENCHMARK_DIR / "results"
SUMMARY_DIR = BENCHMARK_DIR / "summary"


# =====================================================================
# 1. 统一输出数据结构与接口契约 (Standard Results Contract)
# =====================================================================


@dataclass
class ForecastOutput:
    """Standardized output container for a forecasting model on a scenario split."""

    point_forecast: np.ndarray
    q10: Optional[np.ndarray] = None
    q90: Optional[np.ndarray] = None
    latency_ms: float = 0.0
    extra_details: Dict[str, Any] = field(default_factory=dict)


class BaseRetailForecaster(ABC):
    """Abstract base class that every benchmark model should implement."""

    def __init__(self, name: str, family: str):
        self.name = name
        self.family = family

    @abstractmethod
    def fit(self, history: pd.DataFrame, scenario: str) -> None:
        """Fit model on historical observations."""
        pass

    @abstractmethod
    def predict(self, future: pd.DataFrame, scenario: str) -> ForecastOutput:
        """Generate forecasts for future period."""
        pass


# =====================================================================
# 2. 评测计算与格式化工具 (Evaluation Metrics)
# =====================================================================


def compute_standard_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    q10: Optional[Union[np.ndarray, pd.Series]] = None,
    q90: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Dict[str, float]:
    """Compute benchmark metrics aligned with the repo-wide protocol."""
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
        "Coverage80": coverage,
        "n": n,
    }


# =====================================================================
# 3. 基准结果保存与持久化 (Persistence)
# =====================================================================


def get_model_result_dir(model_key: str) -> Path:
    """Return the dedicated storage path for a model's benchmark results."""
    model_dir = RESULTS_DIR / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def save_benchmark_results(
    model_key: str,
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    manifest_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Persist standard benchmark results into data/benchmarks/results/<model_key>/.
    predictions_df columns: [scenario, model, split, forecast_origin, ds, y, prediction, (q10, q90)]
    metrics_df columns: [scenario, model, split, (forecast_origin), MAE, RMSE, WAPE, Bias, (Coverage80)]
    """
    out_dir = get_model_result_dir(model_key)

    predictions_df.to_csv(out_dir / "predictions.csv", index=False)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    meta = {
        "model_key": model_key,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_hashes": INPUT_HASHES,
        "protocol_version": "retail-benchmark-v2",
    }
    if manifest_meta:
        meta.update(manifest_meta)

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[{model_key}] Results successfully saved to {out_dir}")
    return out_dir


# =====================================================================
# 4. 全模型聚合与全景排行榜 (Global Aggregator & Leaderboard)
# =====================================================================


def load_all_benchmark_results() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all available model predictions and metrics from data/benchmarks/results/.
    Returns (all_metrics_df, all_predictions_df).
    """
    if not RESULTS_DIR.exists():
        return pd.DataFrame(), pd.DataFrame()

    all_metrics = []
    all_predictions = []

    for model_path in sorted(RESULTS_DIR.iterdir()):
        if not model_path.is_dir():
            continue
        m_file = model_path / "metrics.csv"
        p_file = model_path / "predictions.csv"

        if m_file.is_file():
            df_m = pd.read_csv(m_file)
            if "model_key" not in df_m.columns:
                df_m["model_key"] = model_path.name
            all_metrics.append(df_m)

        if p_file.is_file():
            df_p = pd.read_csv(p_file, parse_dates=["ds"])
            if "model_key" not in df_p.columns:
                df_p["model_key"] = model_path.name
            all_predictions.append(df_p)

    df_metrics = (
        pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    )
    df_preds = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame()
    )
    return df_metrics, df_preds


def generate_leaderboard(output_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Generate unified multi-model leaderboard from persisted benchmark results.
    Ranks models by mean test WAPE, MAE, RMSE across all 4 scenarios.
    """
    df_metrics, _ = load_all_benchmark_results()
    if df_metrics.empty:
        print("No benchmark results found to generate leaderboard.")
        return pd.DataFrame()

    out = output_dir or SUMMARY_DIR
    out.mkdir(parents=True, exist_ok=True)

    # 过滤 test 测试集
    test_metrics = df_metrics[df_metrics["split"] == "test"].copy()

    # 场景维度排行
    scenario_summary = (
        test_metrics.groupby(["scenario", "model"], as_index=False)
        .agg(
            WAPE=("WAPE", "mean"),
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            Bias=("Bias", "mean"),
        )
        .sort_values(["scenario", "WAPE"])
    )
    scenario_summary.to_csv(out / "scenario_leaderboard.csv", index=False)

    # 跨场景全局综合排行 (Global Leaderboard)
    global_leaderboard = (
        test_metrics.groupby("model", as_index=False)
        .agg(
            Mean_WAPE=("WAPE", "mean"),
            Mean_MAE=("MAE", "mean"),
            Mean_RMSE=("RMSE", "mean"),
            Mean_Bias=("Bias", "mean"),
            Scenarios_Evaluated=("scenario", "nunique"),
        )
        .sort_values("Mean_WAPE")
    )
    global_leaderboard.to_csv(out / "global_leaderboard.csv", index=False)

    print("\n" + "=" * 70)
    print("RETAIL DEMAND FORECASTING BENCHMARK - GLOBAL LEADERBOARD (TEST 28-DAY)")
    print("=" * 70)
    print(global_leaderboard.to_string(index=False))
    print(f"\nLeaderboards saved to: {out}")
    return global_leaderboard
