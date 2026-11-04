"""
Unified Week-Aligned Rolling Benchmark Runner for Retail Demand Forecasting.

Supports:
1. Running all models, families, or specific models/variants:
   --models seasonal_naive,lightgbm_default,xgboost_tweedie_p18
2. Running single model independently:
   --models seasonal_naive
3. Running entire model families:
   --family baseline
   --family tree
4. Evaluation modes:
   --mode 4week (default: 13 non-overlapping 28-day periods, 364 days total)
   --mode weekly (50 rolling steps, each predicting future 28 days)
5. Configurable training window:
   --train-weeks 24 (default: 24 complete weeks = 168 days)
6. Refreshing global leaderboard only (0 re-training, milliseconds):
   --compare-only
7. Listing all supported models & stored models:
   --list
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from retail_forecast.baselines import (
    MovingAverageForecaster,
    SeasonalNaiveForecaster,
)
from retail_forecast.benchmark_registry import (
    BENCHMARK_DIR,
    RESULTS_DIR,
    SUMMARY_DIR,
    ModelConfig,
    RollingFold,
    compute_standard_metrics,
    generate_leaderboard,
    generate_week_aligned_folds,
    get_model_result_dir,
    save_rolling_benchmark_results,
)
from retail_forecast.common import (
    DATA_DIR,
    HORIZON,
    INPUT_HASHES,
    SCENARIOS,
    datasets,
    sales_prediction,
    valid_labels,
)
from retail_forecast.trees import fit_predict_tree_model, tune_model

# =====================================================================
# 模型变种注册字典 (Model Registry & Configuration Matrix)
# =====================================================================

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # 1. Baseline Family
    "seasonal_naive": ModelConfig(
        model_id="seasonal_naive",
        model_name="Seasonal Naive (7d)",
        family="baseline",
        variant="default",
        description="7-day lag seasonal repeat tiled over 28-day horizon",
    ),
    "moving_average_14d": ModelConfig(
        model_id="moving_average_14d",
        model_name="Moving Average (14d)",
        family="baseline",
        variant="default",
        params={"window_days": 14},
        description="Rolling 14-day average of historical open sales",
    ),
    # 2. Tree / GBDT Family: LightGBM
    "lightgbm_default": ModelConfig(
        model_id="lightgbm_default",
        model_name="LightGBM Default (MSE)",
        family="tree_gbdt",
        variant="default",
        params={},
        description="LightGBM with default L2 (Mean Squared Error) objective",
    ),
    "lightgbm_tweedie_p11": ModelConfig(
        model_id="lightgbm_tweedie_p11",
        model_name="LightGBM Tweedie (p=1.1)",
        family="tree_gbdt",
        variant="tweedie_p11",
        params={"objective": "tweedie", "tweedie_variance_power": 1.1},
        description="LightGBM with Tweedie loss (variance power p=1.1, strong Poisson count mode)",
    ),
    "lightgbm_tweedie_p12": ModelConfig(
        model_id="lightgbm_tweedie_p12",
        model_name="LightGBM Tweedie (p=1.2)",
        family="tree_gbdt",
        variant="tweedie_p12",
        params={"objective": "tweedie", "tweedie_variance_power": 1.2},
        description="LightGBM with Tweedie loss (variance power p=1.2, FMCG continuous optimal)",
    ),
    "lightgbm_tweedie_p15": ModelConfig(
        model_id="lightgbm_tweedie_p15",
        model_name="LightGBM Tweedie (p=1.5)",
        family="tree_gbdt",
        variant="tweedie_p15",
        params={"objective": "tweedie", "tweedie_variance_power": 1.5},
        description="LightGBM with Tweedie loss (variance power p=1.5, compound Poisson-gamma)",
    ),
    "lightgbm_tweedie_p18": ModelConfig(
        model_id="lightgbm_tweedie_p18",
        model_name="LightGBM Tweedie (p=1.8)",
        family="tree_gbdt",
        variant="tweedie_p18",
        params={"objective": "tweedie", "tweedie_variance_power": 1.8},
        description="LightGBM with Tweedie loss (variance power p=1.8, zero-inflated Gamma mode)",
    ),
    # 3. Tree / GBDT Family: XGBoost
    "xgboost_default": ModelConfig(
        model_id="xgboost_default",
        model_name="XGBoost Default (MSE)",
        family="tree_gbdt",
        variant="default",
        params={},
        description="XGBoost with default reg:squarederror objective",
    ),
    "xgboost_tweedie_p11": ModelConfig(
        model_id="xgboost_tweedie_p11",
        model_name="XGBoost Tweedie (p=1.1)",
        family="tree_gbdt",
        variant="tweedie_p11",
        params={"objective": "reg:tweedie", "tweedie_variance_power": 1.1},
        description="XGBoost with Tweedie loss (p=1.1)",
    ),
    "xgboost_tweedie_p12": ModelConfig(
        model_id="xgboost_tweedie_p12",
        model_name="XGBoost Tweedie (p=1.2)",
        family="tree_gbdt",
        variant="tweedie_p12",
        params={"objective": "reg:tweedie", "tweedie_variance_power": 1.2},
        description="XGBoost with Tweedie loss (p=1.2)",
    ),
    "xgboost_tweedie_p15": ModelConfig(
        model_id="xgboost_tweedie_p15",
        model_name="XGBoost Tweedie (p=1.5)",
        family="tree_gbdt",
        variant="tweedie_p15",
        params={"objective": "reg:tweedie", "tweedie_variance_power": 1.5},
        description="XGBoost with Tweedie loss (p=1.5)",
    ),
    "xgboost_tweedie_p18": ModelConfig(
        model_id="xgboost_tweedie_p18",
        model_name="XGBoost Tweedie (p=1.8)",
        family="tree_gbdt",
        variant="tweedie_p18",
        params={"objective": "reg:tweedie", "tweedie_variance_power": 1.8},
        description="XGBoost with Tweedie loss (p=1.8, optimal for luxury zero-inflation)",
    ),
    # 4. Tree / GBDT Family: CatBoost
    "catboost_default": ModelConfig(
        model_id="catboost_default",
        model_name="CatBoost Default (MAE)",
        family="tree_gbdt",
        variant="default",
        params={},
        description="CatBoost with default MAE loss function",
    ),
    "catboost_tweedie_p11": ModelConfig(
        model_id="catboost_tweedie_p11",
        model_name="CatBoost Tweedie (p=1.1)",
        family="tree_gbdt",
        variant="tweedie_p11",
        params={"loss_function": "Tweedie:variance_power=1.1"},
        description="CatBoost with Tweedie loss function (p=1.1)",
    ),
    "catboost_tweedie_p15": ModelConfig(
        model_id="catboost_tweedie_p15",
        model_name="CatBoost Tweedie (p=1.5)",
        family="tree_gbdt",
        variant="tweedie_p15",
        params={"loss_function": "Tweedie:variance_power=1.5"},
        description="CatBoost with Tweedie loss function (p=1.5)",
    ),
    "catboost_tweedie_p18": ModelConfig(
        model_id="catboost_tweedie_p18",
        model_name="CatBoost Tweedie (p=1.8)",
        family="tree_gbdt",
        variant="tweedie_p18",
        params={"loss_function": "Tweedie:variance_power=1.8"},
        description="CatBoost with Tweedie loss function (p=1.8)",
    ),
    # 5. Classical Trees
    "random_forest": ModelConfig(
        model_id="random_forest",
        model_name="Random Forest",
        family="tree_gbdt",
        variant="default",
        params={},
        description="Scikit-Learn Random Forest Regressor",
    ),
    "decision_tree": ModelConfig(
        model_id="decision_tree",
        model_name="Decision Tree",
        family="tree_gbdt",
        variant="default",
        params={},
        description="Scikit-Learn Decision Tree Regressor",
    ),
    # 6. Statistical Models
    "prophet": ModelConfig(
        model_id="prophet",
        model_name="Prophet",
        family="statistical",
        variant="default",
        params={},
        description="Meta Prophet with weekly/yearly seasonality and VIC holiday regressors",
    ),
}

# Aliases for convenience
MODEL_ALIASES = {
    "lightgbm": "lightgbm_default",
    "xgboost": "xgboost_default",
    "catboost": "catboost_default",
}


# =====================================================================
# 核心评测执行引擎 (Core Execution Pipeline for a Single Model)
# =====================================================================


def _is_cache_valid(
    model_config: ModelConfig, eval_mode: str, train_weeks: int
) -> bool:
    """Check if model results already exist and match current input hashes & params."""
    res_dir = get_model_result_dir(model_config.model_id, eval_mode=eval_mode)
    manifest_path = res_dir / "manifest.json"
    ann_path = res_dir / "metrics_annual.csv"
    per_path = res_dir / "metrics_period.csv"
    pred_path = res_dir / "predictions.csv"

    if not (
        manifest_path.is_file()
        and ann_path.is_file()
        and per_path.is_file()
        and pred_path.is_file()
    ):
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("eval_mode") != eval_mode:
            return False
        if manifest.get("train_weeks") != train_weeks:
            return False
        if manifest.get("params") != model_config.params:
            return False
        if manifest.get("input_hashes") != INPUT_HASHES:
            return False
        return True
    except Exception:
        return False


def run_single_model_benchmark(
    model_config: ModelConfig,
    eval_mode: str = "4week",
    train_weeks: int = 24,
    scenarios: Optional[List[str]] = None,
    force: bool = False,
) -> Path:
    """
    Execute full walk-forward benchmark for a single model configuration.
    Idempotent: automatically skips if valid cache exists and force is False.
    """
    scenarios = scenarios or list(SCENARIOS.keys())

    if not force and _is_cache_valid(model_config, eval_mode, train_weeks):
        print(
            f"[{model_config.model_id}] Up-to-date benchmark cache found. Skipping execution. (Use --force to recompute)"
        )
        return get_model_result_dir(model_config.model_id, eval_mode=eval_mode)

    folds = generate_week_aligned_folds(
        mode=eval_mode, train_weeks=train_weeks, horizon_days=HORIZON
    )
    print(f"\n========================================================")
    print(f"Executing Benchmark: {model_config.model_name}")
    print(f"Model ID: {model_config.model_id} | Family: {model_config.family}")
    print(
        f"Evaluation Mode: {eval_mode} ({len(folds)} folds) | Train History: {train_weeks} weeks"
    )
    print(f"Scenarios: {scenarios}")
    print(f"========================================================\n")

    all_preds_rows = []
    all_period_metrics = []
    annual_summary_rows = []

    # Pre-tune base tree models if necessary
    tree_base_params_cache = {}
    if model_config.family == "tree_gbdt":
        model_name_map = {
            "lightgbm": "LightGBM",
            "xgboost": "XGBoost",
            "catboost": "CatBoost",
            "random_forest": "Random Forest",
            "decision_tree": "Decision Tree",
        }
        # Find base family name
        base_family_name = None
        for k, v in model_name_map.items():
            if model_config.model_id.startswith(k):
                base_family_name = v
                break

        if base_family_name:
            for sc in scenarios:
                best_p, _, _ = tune_model(sc, base_family_name)
                tree_base_params_cache[sc] = best_p

    prophet_params_cache = {}
    total_start_time = time.time()

    for sc in scenarios:
        df = datasets[sc]
        sc_period_wapes = []
        sc_y_true_all = []
        sc_y_pred_all = []

        print(f"  >>> Evaluating Scenario: {sc.upper()}...")

        for fold in folds:
            # Slicing history and future
            hist = df[(df.ds >= fold.train_start) & (df.ds <= fold.train_end)].copy()
            future = df[(df.ds >= fold.test_start) & (df.ds <= fold.test_end)].copy()
            fut_valid = valid_labels(future)
            y_true = future.loc[fut_valid, "y"].values

            # Grocery cleaning for training
            if sc == "grocery":
                bad_mask = (~hist["is_open"]) | hist["stockout"]
                hist.loc[bad_mask, "y"] = np.nan
                hist["y"] = hist["y"].interpolate().bfill().ffill()

            # Execute model
            t0 = time.perf_counter()
            p_pred = None
            q10_pred = None
            q90_pred = None

            if model_config.model_id == "seasonal_naive":
                forecaster = SeasonalNaiveForecaster(lag_days=7)
                forecaster.fit(hist, sc)
                out = forecaster.predict(future, sc)
                p_pred = out.point_forecast

            elif model_config.model_id == "moving_average_14d":
                forecaster = MovingAverageForecaster(window_days=14)
                forecaster.fit(hist, sc)
                out = forecaster.predict(future, sc)
                p_pred = out.point_forecast

            elif model_config.family == "tree_gbdt":
                # Combine base tuned params with variant specific params
                base_p = tree_base_params_cache.get(sc, {}).copy()
                base_p.update(model_config.params)
                p_pred = fit_predict_tree_model(
                    history=hist,
                    future=future,
                    scenario=sc,
                    model_name=base_family_name,
                    params=base_p,
                )

            elif model_config.model_id == "prophet":
                from retail_forecast.prophet_benchmark import (
                    fit_prophet,
                    predict_prophet,
                    tune_prophet,
                )

                if sc not in prophet_params_cache:
                    p_params, _, _ = tune_prophet(sc)
                    prophet_params_cache[sc] = p_params
                p_params = prophet_params_cache[sc]
                p_model, _ = fit_prophet(hist, sc, p_params)
                p_raw = predict_prophet(p_model, future, sc)
                p_pred = sales_prediction(future, p_raw)
                q10_pred = None
                q90_pred = None

            else:
                raise NotImplementedError(
                    f"Runner for model '{model_config.model_id}' is not yet implemented."
                )

            lat_ms = (time.perf_counter() - t0) * 1000

            # Score period metrics
            score = compute_standard_metrics(
                y_true=y_true,
                y_pred=p_pred[fut_valid],
                q10=q10_pred[fut_valid] if q10_pred is not None else None,
                q90=q90_pred[fut_valid] if q90_pred is not None else None,
            )

            sc_period_wapes.append(score["WAPE"])
            sc_y_true_all.extend(y_true)
            sc_y_pred_all.extend(p_pred[fut_valid])

            all_period_metrics.append(
                {
                    "scenario": sc,
                    "model_id": model_config.model_id,
                    "model_name": model_config.model_name,
                    "family": model_config.family,
                    "fold": fold.fold_id,
                    "period": fold.period_name,
                    "test_start": str(fold.test_start.date()),
                    "test_end": str(fold.test_end.date()),
                    "MAE": score["MAE"],
                    "RMSE": score["RMSE"],
                    "WAPE": score["WAPE"],
                    "Bias": score["Bias"],
                    "ZeroAccuracy": score["ZeroAccuracy"],
                    "Coverage80": score["Coverage80"],
                    "latency_ms": lat_ms,
                    "n": score["n"],
                }
            )

            # Store predictions
            sub_pred = (
                future[["ds", "y", "is_open", "stockout"]].copy()
                if "stockout" in future.columns
                else future[["ds", "y", "is_open"]].copy()
            )
            sub_pred["scenario"] = sc
            sub_pred["model_id"] = model_config.model_id
            sub_pred["fold"] = fold.fold_id
            sub_pred["period"] = fold.period_name
            sub_pred["prediction"] = p_pred
            sub_pred["q10"] = q10_pred if q10_pred is not None else np.nan
            sub_pred["q90"] = q90_pred if q90_pred is not None else np.nan
            all_preds_rows.append(sub_pred)

        # Compute full-horizon cumulative metrics for scenario
        y_all = np.array(sc_y_true_all)
        p_all = np.array(sc_y_pred_all)
        ann_score = compute_standard_metrics(y_all, p_all)

        wapes = np.array(sc_period_wapes)
        annual_summary_rows.append(
            {
                "scenario": sc,
                "model_id": model_config.model_id,
                "model_name": model_config.model_name,
                "family": model_config.family,
                "MAE": ann_score["MAE"],
                "RMSE": ann_score["RMSE"],
                "WAPE": ann_score["WAPE"],
                "Bias": ann_score["Bias"],
                "ZeroAccuracy": ann_score["ZeroAccuracy"],
                "Period_WAPE_Mean": float(np.nanmean(wapes)),
                "Period_WAPE_Std": float(np.nanstd(wapes)),
                "Total_Units": float(y_all.sum()),
                "n": ann_score["n"],
            }
        )
        print(
            f"     -> Cumulative WAPE: {ann_score['WAPE'] * 100:.2f}% | Period WAPE Mean: {np.nanmean(wapes) * 100:.2f}% ± {np.nanstd(wapes) * 100:.2f}%"
        )

    preds_df = pd.concat(all_preds_rows, ignore_index=True)
    period_df = pd.DataFrame(all_period_metrics)
    annual_df = pd.DataFrame(annual_summary_rows)

    elapsed = time.time() - total_start_time
    print(f"\n[{model_config.model_id}] Evaluation finished in {elapsed:.2f}s")

    out_dir = save_rolling_benchmark_results(
        model_config=model_config,
        predictions_df=preds_df,
        metrics_period_df=period_df,
        metrics_annual_df=annual_df,
        eval_mode=eval_mode,
        train_weeks=train_weeks,
        extra_manifest={"elapsed_seconds": elapsed},
    )
    return out_dir


# =====================================================================
# CLI 命令行入口 (Main CLI Command Parser)
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Unified Week-Aligned Rolling Benchmark Runner for Retail Demand Forecasting."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model IDs to execute (e.g. 'seasonal_naive,xgboost_default,xgboost_tweedie_p18').",
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        choices=["baseline", "tree_gbdt", "statistical", "all"],
        help="Execute all models within a specified model family.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="4week",
        choices=["4week", "weekly"],
        help="Evaluation window mode: '4week' (13 non-overlapping 28-day blocks) or 'weekly' (50 rolling steps).",
    )
    parser.add_argument(
        "--train-weeks",
        type=int,
        default=24,
        help="Number of historical complete weeks used for training (default: 24 weeks = 168 days).",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Comma-separated scenarios to evaluate (default: all 4: 'grocery,wine,garden,luxury').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if cached results match input hashes.",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only aggregate all existing results in data/benchmarks/results/ and refresh the global leaderboard.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered benchmark models and stored results.",
    )

    args = parser.parse_args()

    if args.list:
        print("=" * 70)
        print("REGISTERED BENCHMARK MODELS:")
        print("=" * 70)
        for mid, cfg in sorted(MODEL_REGISTRY.items()):
            print(f"  [{cfg.family:<12}] {mid:<24} : {cfg.description}")
        print("\nSTORED MODEL RESULTS (in data/benchmarks/results/):")
        if RESULTS_DIR.exists():
            for p in sorted(RESULTS_DIR.iterdir()):
                if p.is_dir() and not p.name.startswith("."):
                    m_file = p / "metrics_annual.csv"
                    status = (
                        "Week-Aligned Rolling v3"
                        if m_file.exists()
                        else "Legacy 28-Day"
                    )
                    print(f"  * {p.name:<26} ({status})")
        return

    if args.compare_only:
        print(f"Refreshing benchmark leaderboard for mode '{args.mode}'...")
        generate_leaderboard(eval_mode=args.mode)
        return

    # Resolve target models
    target_models = []
    if args.family:
        if args.family == "all":
            target_models = list(MODEL_REGISTRY.keys())
        else:
            target_models = [
                mid for mid, cfg in MODEL_REGISTRY.items() if cfg.family == args.family
            ]
    elif args.models:
        for m in args.models.split(","):
            m = m.strip().lower()
            m = MODEL_ALIASES.get(m, m)
            if m in MODEL_REGISTRY:
                target_models.append(m)
            else:
                print(
                    f"Warning: Unknown model ID '{m}'. Run with --list to view valid models."
                )

    if not target_models:
        print(
            "No models selected. Use --models <keys>, --family <name>, or --compare-only."
        )
        return

    scenarios = (
        [s.strip().lower() for s in args.scenarios.split(",") if s.strip()]
        if args.scenarios
        else list(SCENARIOS.keys())
    )

    print(
        f"Selected models for benchmark: {target_models} across scenarios: {scenarios}"
    )
    for mid in target_models:
        cfg = MODEL_REGISTRY[mid]
        run_single_model_benchmark(
            model_config=cfg,
            eval_mode=args.mode,
            train_weeks=args.train_weeks,
            scenarios=scenarios,
            force=args.force,
        )

    print(f"\nRefreshing global leaderboard for mode '{args.mode}'...")
    generate_leaderboard(eval_mode=args.mode)


if __name__ == "__main__":
    main()
