"""Run one explicit, versioned retail-demand benchmark protocol."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from retail_forecast.benchmark_adapters import (
    baseline_factory,
    prophet_factory,
    resolve_model_parameters,
    tree_factory,
)
from retail_forecast.benchmark_protocol import list_benchmark_specs, load_benchmark_spec
from retail_forecast.benchmark_registry import (
    ModelConfig,
    compute_standard_metrics,
    generate_v4_summaries,
    model_result_dir,
    save_v4_benchmark_results,
    validate_v4_model_result,
)
from retail_forecast.common import ROOT as PACKAGE_ROOT
from retail_forecast.common import SCENARIOS, datasets, valid_labels


def _config(
    model_id: str,
    model_name: str,
    *,
    toolkit: str,
    family: str,
    factory,
    variant: str = "default",
    role: str = "ablation",
    group: str | None = None,
    params: dict | None = None,
    description: str = "",
) -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        model_name=model_name,
        toolkit=toolkit,
        family=family,
        variant=variant,
        comparison_role=role,
        config_group=group or toolkit,
        adapter_factory=factory,
        params=params or {},
        description=description,
    )


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "seasonal_naive": _config(
        "seasonal_naive",
        "Seasonal Naive (7d)",
        toolkit="seasonal_naive",
        family="baseline",
        factory=baseline_factory,
        role="breadth",
        params={"lag_days": 7},
        description="Repeat the most recent complete week.",
    ),
    "moving_average_14d": _config(
        "moving_average_14d",
        "Moving Average (14d)",
        toolkit="moving_average",
        family="baseline",
        factory=baseline_factory,
        role="breadth",
        params={"window_days": 14},
        description="Mean of the latest 14 open trading days.",
    ),
    "decision_tree": _config(
        "decision_tree",
        "Decision Tree",
        toolkit="decision_tree",
        family="tree_classical",
        factory=tree_factory,
        role="breadth",
    ),
    "random_forest": _config(
        "random_forest",
        "Random Forest",
        toolkit="random_forest",
        family="tree_classical",
        factory=tree_factory,
        role="breadth",
    ),
    "prophet": _config(
        "prophet",
        "Prophet",
        toolkit="prophet",
        family="statistical",
        factory=prophet_factory,
        role="breadth",
    ),
}


def _register_gbdt_matrix() -> None:
    toolkits = {
        "lightgbm": {
            "default_name": "LightGBM Default (MSE)",
            "default": {},
            "tweedie": lambda power: {
                "objective": "tweedie",
                "tweedie_variance_power": power,
            },
        },
        "xgboost": {
            "default_name": "XGBoost Default (MSE)",
            "default": {},
            "tweedie": lambda power: {
                "objective": "reg:tweedie",
                "tweedie_variance_power": power,
            },
        },
        "catboost": {
            "default_name": "CatBoost Default (MAE)",
            "default": {},
            "tweedie": lambda power: {
                "loss_function": f"Tweedie:variance_power={power}"
            },
        },
    }
    for toolkit, values in toolkits.items():
        default_id = f"{toolkit}_default"
        MODEL_REGISTRY[default_id] = _config(
            default_id,
            values["default_name"],
            toolkit=toolkit,
            family="tree_gbdt",
            factory=tree_factory,
            role="breadth",
            params=values["default"],
            description=f"Representative default objective for {toolkit}.",
        )
        for suffix, power in (("p11", 1.1), ("p12", 1.2), ("p15", 1.5), ("p18", 1.8)):
            model_id = f"{toolkit}_tweedie_{suffix}"
            MODEL_REGISTRY[model_id] = _config(
                model_id,
                f"{toolkit.title()} Tweedie (p={power:.1f})",
                toolkit=toolkit,
                family="tree_gbdt",
                factory=tree_factory,
                variant=f"tweedie_{suffix}",
                role="ablation",
                params=values["tweedie"](power),
                description="Controlled Tweedie variance-power ablation.",
            )


_register_gbdt_matrix()

MODEL_ALIASES = {
    "lightgbm": "lightgbm_default",
    "xgboost": "xgboost_default",
    "catboost": "catboost_default",
}


def _seasonal_scale(history: pd.DataFrame, lag: int = 7) -> float | None:
    labels = history.loc[valid_labels(history), ["ds", "y"]].copy()
    current = labels.set_index("ds").y
    aligned = pd.concat(
        [current, current.shift(freq=pd.Timedelta(days=lag))], axis=1
    ).dropna()
    if aligned.empty:
        return None
    value = float(np.abs(aligned.iloc[:, 0] - aligned.iloc[:, 1]).mean())
    return value if value > 0 else None


def _prepare_history(history: pd.DataFrame, scenario: str) -> pd.DataFrame:
    clean = history.copy()
    if scenario == "grocery":
        invalid = (~clean.is_open) | clean.stockout
        clean.loc[invalid, "y"] = np.nan
        clean["y"] = clean.y.interpolate().bfill().ffill()
    return clean


def _cache_valid(
    path: Path,
    spec,
    config: ModelConfig,
    scenarios: list[str],
    resolved_params: dict[str, dict],
) -> bool:
    if not path.is_dir():
        return False
    try:
        manifest = validate_v4_model_result(path, spec, require_all_scenarios=False)
    except (ValueError, FileNotFoundError, json.JSONDecodeError):
        return False
    return (
        manifest.get("model_id") == config.model_id
        and manifest.get("scenarios") == scenarios
        and manifest.get("resolved_params_by_scenario") == resolved_params
    )


def run_model(
    *,
    config: ModelConfig,
    spec,
    run_root: Path,
    scenarios: list[str],
    force: bool,
) -> Path:
    selection_dir = run_root / spec.benchmark_id / "selection"
    resolved_params, selection_record = resolve_model_parameters(
        config, spec, selection_dir
    )
    output_path = model_result_dir(run_root, spec.benchmark_id, config.model_id)
    if not force and _cache_valid(
        output_path, spec, config, scenarios, resolved_params
    ):
        print(f"[{config.model_id}] compatible cache found; skipping")
        return output_path

    folds = spec.folds(min(frame.ds.min() for frame in datasets.values()))
    prediction_parts: list[pd.DataFrame] = []
    period_rows: list[dict] = []
    annual_rows: list[dict] = []
    started = time.perf_counter()

    for scenario in scenarios:
        frame = datasets[scenario]
        pooled_y: list[float] = []
        pooled_prediction: list[float] = []
        pooled_q10: list[float] = []
        pooled_q90: list[float] = []
        fold_scales: list[tuple[int, float]] = []
        fold_wapes: list[float] = []
        for fold in folds:
            history = frame[frame.ds.between(fold.train_start, fold.train_end)].copy()
            future = frame[frame.ds.between(fold.test_start, fold.test_end)].copy()
            if len(future) != fold.n_test_days:
                raise ValueError(
                    f"{scenario}/{fold.period_name}: expected {fold.n_test_days} future rows"
                )
            scale = _seasonal_scale(history)
            history = _prepare_history(history, scenario)
            factory = config.adapter_factory
            if factory is None:
                raise RuntimeError(
                    f"{config.model_id} has no registered adapter factory"
                )
            adapter = factory(config, scenario, resolved_params[scenario])
            fold_started = time.perf_counter()
            adapter.fit(history, scenario)
            output = adapter.predict(future, scenario)
            latency_ms = (time.perf_counter() - fold_started) * 1000

            prediction = np.asarray(output.point_forecast, dtype=float)
            if len(prediction) != len(future) or not np.isfinite(prediction).all():
                raise ValueError(
                    f"{scenario}/{fold.period_name}: invalid point forecast"
                )
            valid = valid_labels(future).to_numpy()
            y = future.loc[valid, "y"].to_numpy(float)
            pred = prediction[valid]
            q10 = (
                np.asarray(output.q10, float)[valid] if output.q10 is not None else None
            )
            q90 = (
                np.asarray(output.q90, float)[valid] if output.q90 is not None else None
            )
            metrics = compute_standard_metrics(
                y, pred, q10=q10, q90=q90, seasonal_scale=scale
            )
            fold_wapes.append(metrics["WAPE"])
            if scale is not None:
                fold_scales.append((metrics["n"], scale))
            pooled_y.extend(y)
            pooled_prediction.extend(pred)
            if q10 is not None and q90 is not None:
                pooled_q10.extend(q10)
                pooled_q90.extend(q90)
            period_rows.append(
                {
                    "scenario": scenario,
                    "model_id": config.model_id,
                    "model_name": config.model_name,
                    "toolkit": config.toolkit,
                    "family": config.family,
                    "variant": config.variant,
                    "comparison_role": config.comparison_role,
                    "fold": fold.fold_id,
                    "period": fold.period_name,
                    "test_start": str(fold.test_start.date()),
                    "test_end": str(fold.test_end.date()),
                    **metrics,
                    "latency_ms": latency_ms,
                }
            )

            stored = future[["ds", "y", "is_open", "stockout"]].copy()
            stored["scenario"] = scenario
            stored["model_id"] = config.model_id
            stored["fold"] = fold.fold_id
            stored["period"] = fold.period_name
            stored["forecast_origin"] = fold.train_end
            stored["prediction"] = prediction
            stored["q10"] = output.q10 if output.q10 is not None else np.nan
            stored["q90"] = output.q90 if output.q90 is not None else np.nan
            prediction_parts.append(stored)

        annual_scale = (
            sum(n * scale for n, scale in fold_scales) / sum(n for n, _ in fold_scales)
            if fold_scales
            else None
        )
        annual = compute_standard_metrics(
            np.asarray(pooled_y),
            np.asarray(pooled_prediction),
            q10=np.asarray(pooled_q10) if pooled_q10 else None,
            q90=np.asarray(pooled_q90) if pooled_q90 else None,
            seasonal_scale=annual_scale,
        )
        annual_rows.append(
            {
                "scenario": scenario,
                "model_id": config.model_id,
                "model_name": config.model_name,
                "toolkit": config.toolkit,
                "family": config.family,
                "variant": config.variant,
                "comparison_role": config.comparison_role,
                "config_group": config.config_group,
                **annual,
                "Fold_WAPE_Mean": float(np.nanmean(fold_wapes)),
                "Fold_WAPE_Std": float(np.nanstd(fold_wapes)),
            }
        )
        print(f"[{config.model_id}] {scenario}: pooled WAPE={annual['WAPE']:.4f}")

    return save_v4_benchmark_results(
        base_dir=run_root,
        spec=spec,
        model_config=config,
        predictions_df=pd.concat(prediction_parts, ignore_index=True),
        metrics_period_df=pd.DataFrame(period_rows),
        metrics_annual_df=pd.DataFrame(annual_rows),
        folds=folds,
        scenarios=scenarios,
        resolved_params={scenario: resolved_params[scenario] for scenario in scenarios},
        selection_record=selection_record,
        elapsed_seconds=time.perf_counter() - started,
    )


def _validate_run_id(value: str) -> str:
    pattern = r"\d{8}-[a-z0-9]+(?:-[a-z0-9]+)*-(?:benchmark|validation|experiment)"
    if not re.fullmatch(pattern, value):
        raise ValueError(
            "run ID must match YYYYMMDD-[model]-[type] in lowercase kebab-case"
        )
    return value


def _default_run_id(target_models: list[str]) -> str:
    token = target_models[0] if len(target_models) == 1 else "all-models"
    return f"{datetime.now(UTC):%Y%m%d}-{token}-benchmark"


def _select_models(args: argparse.Namespace) -> list[str]:
    if args.family:
        return [
            model_id
            for model_id, config in MODEL_REGISTRY.items()
            if args.family == "all" or config.family == args.family
        ]
    if args.models:
        selected = []
        for raw in args.models.split(","):
            model_id = MODEL_ALIASES.get(raw.strip().lower(), raw.strip().lower())
            if model_id not in MODEL_REGISTRY:
                raise ValueError(f"Unknown model ID: {model_id}")
            selected.append(model_id)
        return selected
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", help="Explicit benchmark ID from data/benchmarks/protocols."
    )
    parser.add_argument("--models", help="Comma-separated registered model IDs.")
    parser.add_argument(
        "--family",
        choices=["baseline", "tree_classical", "tree_gbdt", "statistical", "all"],
    )
    parser.add_argument(
        "--scenarios", help="Comma-separated subset for local diagnostics only."
    )
    parser.add_argument("--run-id", help="Local runs/ directory name.")
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Root containing <benchmark-id>/models for --compare-only.",
    )
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("BENCHMARK PROTOCOLS")
        for spec in list_benchmark_specs():
            print(f"  {spec.benchmark_id}: {spec.label}")
        print("\nREGISTERED MODELS")
        for model_id, config in sorted(MODEL_REGISTRY.items()):
            print(
                f"  [{config.family:<14}] {model_id:<28} "
                f"toolkit={config.toolkit:<14} role={config.comparison_role}"
            )
        return

    if not args.benchmark:
        parser.error("--benchmark is required for execution and aggregation")
    spec = load_benchmark_spec(args.benchmark)
    if args.compare_only:
        if args.source_root is None:
            parser.error(
                "--compare-only requires --source-root; implicit latest-result scans are forbidden"
            )
        generate_v4_summaries(base_dir=args.source_root, spec=spec)
        return

    target_models = _select_models(args)
    if not target_models:
        parser.error("select --models or --family")
    scenarios = (
        [value.strip() for value in args.scenarios.split(",")]
        if args.scenarios
        else list(spec.scenarios)
    )
    unknown_scenarios = set(scenarios) - set(spec.scenarios) - set(SCENARIOS)
    if unknown_scenarios:
        parser.error(f"unknown scenarios: {sorted(unknown_scenarios)}")

    run_id = _validate_run_id(args.run_id or _default_run_id(target_models))
    run_root = PACKAGE_ROOT / "runs" / run_id
    for model_id in target_models:
        run_model(
            config=MODEL_REGISTRY[model_id],
            spec=spec,
            run_root=run_root,
            scenarios=scenarios,
            force=args.force,
        )
    if set(scenarios) == set(spec.scenarios):
        generate_v4_summaries(base_dir=run_root, spec=spec)
        print(f"Compatible summaries written under {run_root / spec.benchmark_id}")
    else:
        print(
            "Partial-scenario diagnostics completed; release aggregation intentionally skipped."
        )


if __name__ == "__main__":
    main()
