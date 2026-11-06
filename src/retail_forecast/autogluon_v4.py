"""Budgeted AutoGluon suite for the V4 benchmark protocols."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .benchmark_registry import (
    ModelConfig,
    compute_standard_metrics,
    save_v4_benchmark_results,
)
from .common import CPU_THREADS, INPUT_HASHES, SEED, datasets, sales_prediction
from .common import valid_labels as retail_valid_labels

AUTOML_TIME_LIMIT_SECONDS = 240
AUTOML_QUANTILES = (0.1, 0.5, 0.9)
AUTOML_MODEL_LIBRARY = (
    "SeasonalNaive",
    "Zero",
    "ETS",
    "Theta",
    "AutoARIMA",
    "Croston",
    "ADIDA",
    "IMAPA",
    "NPTS",
    "DirectTabular",
    "RecursiveTabular",
    "DeepAR",
    "DLinear",
    "SimpleFeedForward",
)
AUTOML_COMPONENT_IDS = {
    "SeasonalNaive": "ag-seasonal-naive",
    "Zero": "ag-zero",
    "ETS": "ag-ets",
    "Theta": "ag-theta",
    "AutoARIMA": "ag-auto-arima",
    "Croston": "ag-croston",
    "ADIDA": "ag-adida",
    "IMAPA": "ag-imapa",
    "NPTS": "ag-npts",
    "DirectTabular": "ag-direct-tabular",
    "RecursiveTabular": "ag-recursive-tabular",
    "DeepAR": "ag-deepar",
    "DLinear": "ag-dlinear",
    "SimpleFeedForward": "ag-simple-feedforward",
    "WeightedEnsemble": "ag-weighted-ensemble",
}


def automl_model_configs() -> dict[str, ModelConfig]:
    """Declare the AutoML comparison family without adding it to the breadth table."""
    configs = {}
    for component, model_id in AUTOML_COMPONENT_IDS.items():
        model_name = (
            "AG Weighted Ensemble"
            if component == "WeightedEnsemble"
            else f"AG {component}"
        )
        configs[model_id] = ModelConfig(
            model_id=model_id,
            model_name=model_name,
            toolkit="autogluon_timeseries",
            family="automl",
            variant=component.lower(),
            comparison_role="automl",
            config_group="autogluon_timeseries",
            params={
                "time_limit_seconds": AUTOML_TIME_LIMIT_SECONDS,
                "prediction_length": 28,
                "num_val_windows": 1,
                "refit_full": True,
                "enable_ensemble": True,
            },
            description=(
                "Budgeted AutoGluon weighted ensemble."
                if component == "WeightedEnsemble"
                else f"Component trained inside the budgeted AutoGluon suite: {component}."
            ),
        )
    return configs


def known_covariate_names(scenario: str) -> list[str]:
    """Return only covariates known at the forecast origin."""
    from .common import SCENARIOS

    names = ["is_open"]
    if "is_closed_holiday" in datasets[scenario].columns:
        names.append("is_closed_holiday")
    names.extend(SCENARIOS[scenario]["features"])
    return list(dict.fromkeys(names))


def autogluon_frame(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Convert one scenario without exposing future or diagnostic truth."""
    result = pd.DataFrame(
        {
            "item_id": scenario,
            "timestamp": pd.to_datetime(frame.ds),
            "target": frame.y.where(retail_valid_labels(frame)).astype(float),
        }
    )
    for column in known_covariate_names(scenario):
        result[column] = frame[column].astype(float).to_numpy()
    forbidden = {
        "y",
        "stockout",
        "conditional_mean",
        "latent_demand",
        "occurrence_probability",
        "positive_demand",
    }
    if forbidden.intersection(result.columns):
        raise AssertionError("AutoGluon input exposes forbidden target diagnostics")
    return result


def _as_timeseries(frame: pd.DataFrame, scenario: str):
    from autogluon.timeseries import TimeSeriesDataFrame

    return TimeSeriesDataFrame.from_data_frame(
        autogluon_frame(frame, scenario),
        id_column="item_id",
        timestamp_column="timestamp",
    )


def _future_covariates(frame: pd.DataFrame, scenario: str):
    from autogluon.timeseries import TimeSeriesDataFrame

    columns = ["item_id", "timestamp", *known_covariate_names(scenario)]
    return TimeSeriesDataFrame.from_data_frame(
        autogluon_frame(frame, scenario)[columns],
        id_column="item_id",
        timestamp_column="timestamp",
    )


def _suite_identity(spec: Any, scenario: str, fold: Any, history: pd.DataFrame) -> dict:
    payload = {
        "benchmark_id": spec.benchmark_id,
        "protocol_hash": spec.protocol_hash,
        "scenario": scenario,
        "fold": fold.manifest_record(),
        "history_start": str(history.ds.min().date()),
        "history_end": str(history.ds.max().date()),
        "history_rows": len(history),
        "input_hashes": INPUT_HASHES,
        "model_library": list(AUTOML_MODEL_LIBRARY),
        "time_limit_seconds": AUTOML_TIME_LIMIT_SECONDS,
        "quantiles": list(AUTOML_QUANTILES),
        "autogluon_timeseries": importlib.metadata.version("autogluon-timeseries"),
        "random_seed": SEED,
        "refit_full": True,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "identity_sha256": hashlib.sha256(encoded).hexdigest()}


def _resolve_component_models(model_names: list[str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for prefix, model_id in AUTOML_COMPONENT_IDS.items():
        matches = [
            name
            for name in model_names
            if name == prefix or name.startswith(prefix + "_")
        ]
        if matches:
            resolved[model_id] = matches[-1]
    return resolved


def _fit_or_load_predictor(
    *,
    cache_path: Path,
    identity: dict,
    history: pd.DataFrame,
    scenario: str,
):
    from autogluon.timeseries import TimeSeriesPredictor

    identity_path = cache_path / "suite_identity.json"
    if identity_path.is_file():
        stored = json.loads(identity_path.read_text(encoding="utf-8"))
        if stored == identity:
            predictor = TimeSeriesPredictor.load(cache_path)
            return predictor, 0.0, True
        raise ValueError(f"AutoGluon cache identity mismatch: {cache_path}")
    if cache_path.exists() and any(cache_path.iterdir()):
        raise ValueError(f"Unvalidated AutoGluon cache exists: {cache_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    predictor = TimeSeriesPredictor(
        target="target",
        known_covariates_names=known_covariate_names(scenario),
        prediction_length=28,
        freq="D",
        eval_metric="MAE",
        path=cache_path,
        verbosity=1,
        quantile_levels=list(AUTOML_QUANTILES),
        cache_predictions=True,
    )
    started = time.perf_counter()
    predictor.fit(
        _as_timeseries(history, scenario),
        time_limit=AUTOML_TIME_LIMIT_SECONDS,
        hyperparameters={name: {} for name in AUTOML_MODEL_LIBRARY},
        num_val_windows=1,
        refit_every_n_windows=1,
        refit_full=True,
        enable_ensemble=True,
        random_seed=SEED,
        verbosity=1,
    )
    elapsed = time.perf_counter() - started
    identity_path.write_text(
        json.dumps(identity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return predictor, elapsed, False


def _forecast_components(
    predictor,
    history: pd.DataFrame,
    future: pd.DataFrame,
    scenario: str,
    fit_seconds: float,
) -> tuple[pd.DataFrame, dict[str, str]]:
    resolved = _resolve_component_models(list(predictor.model_names()))
    if "ag-weighted-ensemble" not in resolved:
        raise RuntimeError(
            f"AutoGluon did not produce a weighted ensemble for {scenario}; "
            "the formal AutoML result is incomplete"
        )
    rows = []
    history_ts = _as_timeseries(history, scenario)
    covariates = _future_covariates(future, scenario)
    for model_id, internal_name in resolved.items():
        started = time.perf_counter()
        forecast = predictor.predict(
            history_ts,
            known_covariates=covariates,
            model=internal_name,
            random_seed=SEED,
        ).reset_index()
        inference_seconds = time.perf_counter() - started
        if len(forecast) != len(future):
            raise ValueError(f"{model_id}: AutoGluon forecast length mismatch")
        point = sales_prediction(future, forecast["mean"].to_numpy(float))
        q10 = sales_prediction(future, np.maximum(forecast["0.1"].to_numpy(float), 0))
        q90 = sales_prediction(future, np.maximum(forecast["0.9"].to_numpy(float), 0))
        for index, source in future.reset_index(drop=True).iterrows():
            rows.append(
                {
                    "model_id": model_id,
                    "internal_model": internal_name,
                    "ds": source.ds,
                    "y": source.y,
                    "is_open": source.is_open,
                    "stockout": source.stockout,
                    "prediction": point[index],
                    "q10": q10[index],
                    "q90": q90[index],
                    "fit_seconds": fit_seconds,
                    "inference_seconds": inference_seconds,
                }
            )
    return pd.DataFrame(rows), resolved


def _seasonal_scale(history: pd.DataFrame, lag: int = 7) -> float | None:
    labels = history.loc[retail_valid_labels(history), ["ds", "y"]].copy()
    current = labels.set_index("ds").y
    aligned = pd.concat(
        [current, current.shift(freq=pd.Timedelta(days=lag))], axis=1
    ).dropna()
    if aligned.empty:
        return None
    value = float(np.abs(aligned.iloc[:, 0] - aligned.iloc[:, 1]).mean())
    return value if value > 0 else None


def _save_component_result(
    *,
    model_id: str,
    predictions: pd.DataFrame,
    spec: Any,
    run_root: Path,
    folds: list[Any],
    scenarios: list[str],
    availability: dict[str, Any],
) -> Path:
    config = automl_model_configs()[model_id]
    period_rows = []
    annual_rows = []
    for scenario in scenarios:
        scenario_rows = predictions[predictions.scenario.eq(scenario)].copy()
        pooled_y: list[float] = []
        pooled_prediction: list[float] = []
        pooled_q10: list[float] = []
        pooled_q90: list[float] = []
        fold_wapes: list[float] = []
        fold_scales: list[tuple[int, float]] = []
        for fold in folds:
            rows = scenario_rows[scenario_rows.fold.eq(fold.fold_id)].copy()
            valid = retail_valid_labels(rows).to_numpy()
            history = datasets[scenario][
                datasets[scenario].ds.between(fold.train_start, fold.train_end)
            ]
            scale = _seasonal_scale(history)
            metrics = compute_standard_metrics(
                rows.loc[valid, "y"],
                rows.loc[valid, "prediction"],
                q10=rows.loc[valid, "q10"],
                q90=rows.loc[valid, "q90"],
                seasonal_scale=scale,
            )
            fold_wapes.append(metrics["WAPE"])
            fold_scales.append((metrics["n"], scale or 0.0))
            pooled_y.extend(rows.loc[valid, "y"].to_numpy(float))
            pooled_prediction.extend(rows.loc[valid, "prediction"].to_numpy(float))
            pooled_q10.extend(rows.loc[valid, "q10"].to_numpy(float))
            pooled_q90.extend(rows.loc[valid, "q90"].to_numpy(float))
            period_rows.append(
                {
                    "scenario": scenario,
                    "model_id": model_id,
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
                    "latency_ms": float(
                        (rows.fit_seconds.iloc[0] + rows.inference_seconds.iloc[0])
                        * 1000
                    ),
                }
            )
        annual_scale = (
            sum(n * scale for n, scale in fold_scales) / sum(n for n, _ in fold_scales)
            if fold_scales
            else None
        )
        annual = compute_standard_metrics(
            pooled_y,
            pooled_prediction,
            q10=np.asarray(pooled_q10),
            q90=np.asarray(pooled_q90),
            seasonal_scale=annual_scale,
        )
        annual_rows.append(
            {
                "scenario": scenario,
                "model_id": model_id,
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
    stored = predictions[
        [
            "ds",
            "y",
            "is_open",
            "stockout",
            "scenario",
            "model_id",
            "fold",
            "period",
            "forecast_origin",
            "prediction",
            "q10",
            "q90",
        ]
    ].copy()
    resolved_params = {scenario: dict(config.params) for scenario in scenarios}
    selection_record = {
        "model_id": model_id,
        "toolkit": config.toolkit,
        "selection_policy": "budgeted-automl-history-only-v1",
        "selection_end": str(spec.selection_end.date()),
        "formal_test_labels_used_for_selection": False,
        "candidate_library": list(AUTOML_MODEL_LIBRARY),
        "time_limit_seconds_per_fit": AUTOML_TIME_LIMIT_SECONDS,
        "resolved_params_by_scenario": resolved_params,
        "component_availability": availability,
        "input_hashes": INPUT_HASHES,
    }
    return save_v4_benchmark_results(
        base_dir=run_root,
        spec=spec,
        model_config=config,
        predictions_df=stored,
        metrics_period_df=pd.DataFrame(period_rows),
        metrics_annual_df=pd.DataFrame(annual_rows),
        folds=folds,
        scenarios=scenarios,
        resolved_params=resolved_params,
        selection_record=selection_record,
        elapsed_seconds=float(predictions.fit_seconds.sum() / 28),
    )


def run_autogluon_suite(
    *,
    spec: Any,
    run_root: Path,
    scenarios: list[str],
) -> list[str]:
    """Fit AutoGluon once per scenario/fold and materialize complete ag-* models."""
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(CPU_THREADS))
    folds = spec.folds(min(frame.ds.min() for frame in datasets.values()))
    cache_root = run_root / "autogluon-cache" / spec.benchmark_id
    prediction_parts = []
    inventories = []
    for scenario in scenarios:
        frame = datasets[scenario]
        for fold in folds:
            history = frame[frame.ds.between(fold.train_start, fold.train_end)].copy()
            future = frame[frame.ds.between(fold.test_start, fold.test_end)].copy()
            identity = _suite_identity(spec, scenario, fold, history)
            cache_path = cache_root / scenario / f"fold-{fold.fold_id:02d}"
            print(
                f"[automl] {spec.benchmark_id} {scenario}/{fold.period_name}: "
                f"{AUTOML_TIME_LIMIT_SECONDS}s budget",
                flush=True,
            )
            predictor, fit_seconds, reused = _fit_or_load_predictor(
                cache_path=cache_path,
                identity=identity,
                history=history,
                scenario=scenario,
            )
            fold_predictions, resolved = _forecast_components(
                predictor, history, future, scenario, fit_seconds
            )
            fold_predictions["scenario"] = scenario
            fold_predictions["fold"] = fold.fold_id
            fold_predictions["period"] = fold.period_name
            fold_predictions["forecast_origin"] = fold.train_end
            prediction_parts.append(fold_predictions)
            inventories.append(
                {
                    "scenario": scenario,
                    "fold": fold.fold_id,
                    "period": fold.period_name,
                    "cache_reused": reused,
                    "fit_seconds": fit_seconds,
                    "trained_internal_models": list(predictor.model_names()),
                    "resolved_components": resolved,
                }
            )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    expected_fold_count = len(scenarios) * len(folds)
    availability_counts = (
        predictions[["model_id", "scenario", "fold"]]
        .drop_duplicates()
        .groupby("model_id")
        .size()
        .to_dict()
    )
    complete = sorted(
        model_id
        for model_id, count in availability_counts.items()
        if count == expected_fold_count
    )
    if "ag-weighted-ensemble" not in complete:
        raise RuntimeError(
            "ag-weighted-ensemble is not complete across all requested folds"
        )
    availability = {
        "required_scenario_folds": expected_fold_count,
        "counts": availability_counts,
        "complete_model_ids": complete,
        "inventories": inventories,
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "suite_manifest.json").write_text(
        json.dumps(availability, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for model_id in complete:
        model_predictions = predictions[predictions.model_id.eq(model_id)].copy()
        _save_component_result(
            model_id=model_id,
            predictions=model_predictions,
            spec=spec,
            run_root=run_root,
            folds=folds,
            scenarios=scenarios,
            availability=availability,
        )
        annual_path = (
            run_root / spec.benchmark_id / "models" / model_id / "metrics_annual.csv"
        )
        print(f"[{model_id}] wrote {annual_path}")
    return complete
