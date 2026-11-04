"""
Script to extract and register existing historical experiment results
into data/benchmarks/results/<model_key>/.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from retail_forecast.benchmark_registry import (
    RESULTS_DIR,
    SUMMARY_DIR,
    generate_leaderboard,
    save_benchmark_results,
)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def sync_core_models():
    comparison_dir = (
        ROOT / "runs" / "20261020-all-models-summary" / "all_model_comparison"
    )
    if not comparison_dir.is_dir():
        print(f"Warning: {comparison_dir} not found.")
        return

    pred_file = comparison_dir / "predictions.csv"
    summary_file = comparison_dir / "model_comparison.csv"
    if not (pred_file.is_file() and summary_file.is_file()):
        return

    print("Syncing core models from all_model_comparison...")
    df_preds = pd.read_csv(pred_file, parse_dates=["ds"])
    df_metrics = pd.read_csv(summary_file)

    for model_name in df_preds["model"].unique():
        model_key = slugify(model_name)
        sub_preds = df_preds[df_preds["model"] == model_name].copy()
        sub_metrics = df_metrics[df_metrics["model"] == model_name].copy()

        save_benchmark_results(
            model_key=model_key,
            predictions_df=sub_preds,
            metrics_df=sub_metrics,
            manifest_meta={
                "model_name": model_name,
                "source": "runs/20261020-all-models-summary",
            },
        )


def sync_timesfm():
    tsfm_dir = ROOT / "runs" / "20261103-timesfm-validation"
    pred_file = tsfm_dir / "bq_timesfm_predictions.csv"
    benchmark_file = tsfm_dir / "exp1_full_history_benchmark.csv"

    if not (pred_file.is_file() and benchmark_file.is_file()):
        return

    print("Syncing TimesFM 3.0 results...")
    df_bq = pd.read_csv(pred_file, parse_dates=["ds"])
    df_exp1 = pd.read_csv(benchmark_file)

    # 提取全历史基准
    sub_preds = df_bq[df_bq["experiment_id"] == "exp1_full_history"].copy()
    sub_preds["split"] = "test"
    sub_preds["forecast_origin"] = pd.Timestamp("2025-12-03")

    # 格式化指标
    tsfm_metrics = df_exp1[df_exp1["Model"].str.contains("TimesFM")].copy()
    formatted_metrics = []
    for _, row in tsfm_metrics.iterrows():
        formatted_metrics.append(
            {
                "scenario": row["Scenario"],
                "model": "TimesFM 3.0 (BigQuery)",
                "split": "test",
                "MAE": row["MAE"],
                "RMSE": row["RMSE"],
                "WAPE": row["WAPE (%)"] / 100.0,
                "Bias": row["Bias (%)"] / 100.0,
                "Coverage80": row["Coverage 80%"] / 100.0
                if not pd.isna(row["Coverage 80%"])
                else np.nan,
                "n": 28,
            }
        )

    save_benchmark_results(
        model_key="timesfm",
        predictions_df=sub_preds,
        metrics_df=pd.DataFrame(formatted_metrics),
        manifest_meta={
            "model_name": "TimesFM 3.0 (BigQuery)",
            "foundation_model": True,
            "source": "runs/20261103-timesfm-validation",
        },
    )


def sync_autogluon():
    ag_dir = ROOT / "runs" / "20261020-autogluon-validation" / "autogluon_timeseries"
    pred_file = ag_dir / "predictions.csv"
    metrics_file = ag_dir / "aggregate_metrics.csv"

    if not (pred_file.is_file() and metrics_file.is_file()):
        return

    print("Syncing AutoGluon-TimeSeries results...")
    df_preds = pd.read_csv(pred_file, parse_dates=["ds"])
    df_metrics = pd.read_csv(metrics_file)

    save_benchmark_results(
        model_key="autogluon",
        predictions_df=df_preds,
        metrics_df=df_metrics,
        manifest_meta={
            "model_name": "AutoGluon Ensemble",
            "automl": True,
            "source": "runs/20261020-autogluon-validation",
        },
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    sync_core_models()
    sync_timesfm()
    sync_autogluon()

    print("\nGenerating comprehensive global leaderboard...")
    generate_leaderboard()


if __name__ == "__main__":
    main()
