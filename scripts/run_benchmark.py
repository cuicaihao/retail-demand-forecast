"""
Unified Benchmark Runner for Retail Demand Forecasting.

Supports:
1. Running all models or any specific subset of models:
   --models prophet,lightgbm
2. Running single model independently without re-running expensive models:
   --models lightgbm
3. Refreshing leaderboard only (zero retraining):
   --compare-only
4. Listing all supported models:
   --list
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from retail_forecast.benchmark_registry import (
    BENCHMARK_DIR,
    RESULTS_DIR,
    SUMMARY_DIR,
    compute_standard_metrics,
    generate_leaderboard,
    save_benchmark_results,
)
from retail_forecast.common import (
    DATA_DIR,
    HORIZON,
    SCENARIOS,
    datasets,
    sales_prediction,
    valid_labels,
    windows,
)


def run_seasonal_naive_benchmark():
    print("\n>>> Running Seasonal Naive Benchmark...")
    rows_preds = []
    rows_metrics = []

    for sc_name in SCENARIOS:
        df = datasets[sc_name]
        full_hist = df.iloc[:-HORIZON].copy()
        future = df.iloc[-HORIZON:].copy()
        fut_valid = valid_labels(future)
        y_true = future.loc[fut_valid, "y"].values

        # 7-day seasonal naive
        naive_pattern = full_hist.iloc[-7:]["y"].values
        pred = np.tile(naive_pattern, int(np.ceil(HORIZON / 7)))[:HORIZON]
        p_val = sales_prediction(future, pred)

        score = compute_standard_metrics(y_true, p_val[fut_valid])
        rows_metrics.append(
            {
                "scenario": sc_name,
                "model": "Seasonal naive",
                "split": "test",
                "forecast_origin": full_hist["ds"].iloc[-1],
                "MAE": score["MAE"],
                "RMSE": score["RMSE"],
                "WAPE": score["WAPE"],
                "Bias": score["Bias"],
                "n": score["n"],
            }
        )

        sub_df = future.copy()
        sub_df["model"] = "Seasonal naive"
        sub_df["split"] = "test"
        sub_df["forecast_origin"] = full_hist["ds"].iloc[-1]
        sub_df["prediction"] = p_val
        rows_preds.append(sub_df)

    save_benchmark_results(
        model_key="seasonal_naive",
        predictions_df=pd.concat(rows_preds, ignore_index=True),
        metrics_df=pd.DataFrame(rows_metrics),
        manifest_meta={"model_name": "Seasonal naive", "type": "baseline"},
    )


def run_prophet_benchmark():
    print("\n>>> Running Prophet Benchmark...")
    from retail_forecast.prophet_benchmark import fit_predict_prophet

    rows_preds = []
    rows_metrics = []

    for sc_name in SCENARIOS:
        df = datasets[sc_name]
        full_hist = df.iloc[:-HORIZON].copy()
        future = df.iloc[-HORIZON:].copy()
        fut_valid = valid_labels(future)
        y_true = future.loc[fut_valid, "y"].values

        clean_hist = full_hist.copy()
        if sc_name == "grocery":
            bad_mask = (~clean_hist["is_open"]) | clean_hist["stockout"]
            clean_hist.loc[bad_mask, "y"] = np.nan
            clean_hist["y"] = clean_hist["y"].interpolate().bfill().ffill()

        p_pred, q10, q90, lat = fit_predict_prophet(clean_hist, future, sc_name)
        score = compute_standard_metrics(
            y_true, p_pred[fut_valid], q10[fut_valid], q90[fut_valid]
        )

        rows_metrics.append(
            {
                "scenario": sc_name,
                "model": "Prophet",
                "split": "test",
                "forecast_origin": full_hist["ds"].iloc[-1],
                "MAE": score["MAE"],
                "RMSE": score["RMSE"],
                "WAPE": score["WAPE"],
                "Bias": score["Bias"],
                "Coverage80": score["Coverage80"],
                "latency_ms": lat,
                "n": score["n"],
            }
        )

        sub_df = future.copy()
        sub_df["model"] = "Prophet"
        sub_df["split"] = "test"
        sub_df["forecast_origin"] = full_hist["ds"].iloc[-1]
        sub_df["prediction"] = p_pred
        sub_df["q10"] = q10
        sub_df["q90"] = q90
        rows_preds.append(sub_df)

    save_benchmark_results(
        model_key="prophet",
        predictions_df=pd.concat(rows_preds, ignore_index=True),
        metrics_df=pd.DataFrame(rows_metrics),
        manifest_meta={"model_name": "Prophet", "type": "statistical_additive"},
    )


def run_tree_model_benchmark(model_name: str):
    print(f"\n>>> Running Tree Benchmark for {model_name}...")
    from retail_forecast.trees import fit_predict_tree_model

    rows_preds = []
    rows_metrics = []

    for sc_name in SCENARIOS:
        df = datasets[sc_name]
        full_hist = df.iloc[:-HORIZON].copy()
        future = df.iloc[-HORIZON:].copy()
        fut_valid = valid_labels(future)
        y_true = future.loc[fut_valid, "y"].values

        clean_hist = full_hist.copy()
        if sc_name == "grocery":
            bad_mask = (~clean_hist["is_open"]) | clean_hist["stockout"]
            clean_hist.loc[bad_mask, "y"] = np.nan
            clean_hist["y"] = clean_hist["y"].interpolate().bfill().ffill()

        t0 = time.perf_counter()
        p_pred = fit_predict_tree_model(
            clean_hist, future, sc_name, model_name=model_name
        )
        lat = (time.perf_counter() - t0) * 1000

        score = compute_standard_metrics(y_true, p_pred[fut_valid])
        rows_metrics.append(
            {
                "scenario": sc_name,
                "model": model_name,
                "split": "test",
                "forecast_origin": full_hist["ds"].iloc[-1],
                "MAE": score["MAE"],
                "RMSE": score["RMSE"],
                "WAPE": score["WAPE"],
                "Bias": score["Bias"],
                "latency_ms": lat,
                "n": score["n"],
            }
        )

        sub_df = future.copy()
        sub_df["model"] = model_name
        sub_df["split"] = "test"
        sub_df["forecast_origin"] = full_hist["ds"].iloc[-1]
        sub_df["prediction"] = p_pred
        rows_preds.append(sub_df)

    slug = model_name.lower().replace(" ", "_")
    save_benchmark_results(
        model_key=slug,
        predictions_df=pd.concat(rows_preds, ignore_index=True),
        metrics_df=pd.DataFrame(rows_metrics),
        manifest_meta={"model_name": model_name, "type": "tree_gbdt"},
    )


def run_timesfm_benchmark():
    print("\n>>> Running TimesFM Benchmark (Syncing from Cloud / Runs)...")
    from scripts.sync_benchmarks_to_store import sync_timesfm

    sync_timesfm()


AVAILABLE_RUNNERS = {
    "seasonal_naive": run_seasonal_naive_benchmark,
    "prophet": run_prophet_benchmark,
    "lightgbm": lambda: run_tree_model_benchmark("LightGBM"),
    "xgboost": lambda: run_tree_model_benchmark("XGBoost"),
    "catboost": lambda: run_tree_model_benchmark("CatBoost"),
    "random_forest": lambda: run_tree_model_benchmark("Random Forest"),
    "decision_tree": lambda: run_tree_model_benchmark("Decision Tree"),
    "timesfm": run_timesfm_benchmark,
}


def main():
    parser = argparse.ArgumentParser(
        description="Unified Retail Demand Forecasting Benchmark Runner."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model keys to run (e.g., 'prophet,lightgbm,timesfm').",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all natively runnable benchmark models.",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only aggregate all existing results in data/benchmarks/ and refresh the global leaderboard.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all supported model keys in the benchmark runner.",
    )
    args = parser.parse_args()

    if args.list:
        print("Supported Benchmark Model Keys:")
        for k in sorted(AVAILABLE_RUNNERS.keys()):
            print(f"  - {k}")
        print("\nStored / Offline Models (in data/benchmarks/results/):")
        if RESULTS_DIR.exists():
            for p in sorted(RESULTS_DIR.iterdir()):
                if p.is_dir():
                    print(f"  * {p.name}")
        return

    if args.compare_only:
        print(
            "Refreshing global benchmark leaderboard from data/benchmarks/results/..."
        )
        generate_leaderboard()
        return

    models_to_run = []
    if args.all:
        models_to_run = list(AVAILABLE_RUNNERS.keys())
    elif args.models:
        for m in args.models.split(","):
            m = m.strip().lower()
            if m:
                if m not in AVAILABLE_RUNNERS:
                    print(
                        f"Warning: Unknown model '{m}'. Use --list to view supported models."
                    )
                else:
                    models_to_run.append(m)

    if not models_to_run:
        print(
            "No models selected to run. Use --models <keys>, --all, or --compare-only."
        )
        return

    print(f"Executing benchmarks for: {', '.join(models_to_run)}")
    for m in models_to_run:
        t0 = time.time()
        AVAILABLE_RUNNERS[m]()
        print(f"[{m}] Finished in {time.time() - t0:.2f}s")

    print("\nRefreshing global leaderboard...")
    generate_leaderboard()


if __name__ == "__main__":
    main()
