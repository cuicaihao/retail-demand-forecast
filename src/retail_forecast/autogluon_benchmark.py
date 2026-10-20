from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (
    CPU_THREADS,
    DATA_DIR,
    HORIZON,
    INPUT_HASHES,
    RUNS_DIR,
    SCENARIOS,
    SEED,
    datasets,
    metrics,
    sales_prediction,
    valid_labels,
    vic_holidays,
    windows,
)

AUTOGLOUON_MODEL_LIBRARY = (
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
EXCLUDED_FOUNDATION_MODELS = (
    "Chronos",
    "Chronos2",
    "Toto",
    "Toto2",
    "PatchTST",
    "TemporalFusionTransformer",
)
QUANTILES = (0.1, 0.5, 0.9)
DEFAULT_VALIDATION_SECONDS = int(os.environ.get("AUTOGLOUON_VALIDATION_SECONDS", "90"))
DEFAULT_FINAL_SECONDS = int(os.environ.get("AUTOGLOUON_FINAL_SECONDS", "240"))
DEFAULT_IMPORTANCE_SECONDS = int(os.environ.get("AUTOGLOUON_IMPORTANCE_SECONDS", "60"))
OLD_COMPARISON = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "all-models-summary-validation-20261020"
    / "all_model_comparison"
)


def known_covariate_names(scenario: str) -> list[str]:
    """Return only variables genuinely known throughout the forecast horizon."""
    names = ["is_open"]
    if "is_closed_holiday" in datasets[scenario].columns:
        names.append("is_closed_holiday")
    names.extend(SCENARIOS[scenario]["features"])
    return list(dict.fromkeys(names))


def autogluon_frame(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Convert a scenario frame without exposing diagnostic or future-only truth."""
    result = pd.DataFrame(
        {
            "item_id": scenario,
            "timestamp": pd.to_datetime(frame.ds),
            "target": frame.y.where(valid_labels(frame)).astype(float),
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
    assert not forbidden.intersection(result.columns)
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


def model_library() -> dict[str, dict]:
    return {name: {} for name in AUTOGLOUON_MODEL_LIBRARY}


def fit_predictor(
    history: pd.DataFrame,
    scenario: str,
    path: Path,
    time_limit: int,
):
    from autogluon.timeseries import TimeSeriesPredictor

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path)
    predictor = TimeSeriesPredictor(
        target="target",
        known_covariates_names=known_covariate_names(scenario),
        prediction_length=HORIZON,
        freq="D",
        eval_metric="MAE",
        path=path,
        verbosity=1,
        quantile_levels=list(QUANTILES),
        cache_predictions=True,
    )
    started = time.perf_counter()
    predictor.fit(
        _as_timeseries(history, scenario),
        time_limit=time_limit,
        hyperparameters=model_library(),
        num_val_windows=1,
        refit_every_n_windows=1,
        refit_full=False,
        enable_ensemble=True,
        random_seed=SEED,
        verbosity=1,
    )
    return predictor, time.perf_counter() - started


def _selected_model(predictor) -> str:
    info = predictor.info()
    return str(info.get("best_model") or predictor.model_best)


def _forecast(
    predictor, history: pd.DataFrame, future: pd.DataFrame, scenario: str, model=None
):
    forecast = predictor.predict(
        _as_timeseries(history, scenario),
        known_covariates=_future_covariates(future, scenario),
        model=model,
        random_seed=SEED,
    ).reset_index()
    assert len(forecast) == len(future)
    return forecast


def _prediction_table(
    predictor,
    history: pd.DataFrame,
    future: pd.DataFrame,
    scenario: str,
    split: str,
    regime: str,
    selected_model: str,
) -> pd.DataFrame:
    forecast = _forecast(predictor, history, future, scenario, selected_model)
    result = future.copy()
    result["raw_prediction"] = forecast["mean"].to_numpy(float)
    result["prediction"] = sales_prediction(result, result.raw_prediction)
    for quantile, output_name in [("0.1", "q10"), ("0.5", "q50"), ("0.9", "q90")]:
        values = np.maximum(forecast[quantile].to_numpy(float), 0)
        result[output_name] = np.where(result.is_open, values, 0)
    result["scenario"] = scenario
    result["model"] = "AutoGluon Ensemble"
    result["selected_model"] = selected_model
    result["split"] = split
    result["regime"] = regime
    result["forecast_origin"] = history.ds.iloc[-1]
    result["horizon_day"] = np.arange(1, len(result) + 1)
    return result


def _extract_ensemble_weights(predictor, scenario: str, regime: str) -> list[dict]:
    info = predictor.info()
    selected = _selected_model(predictor)
    weights = info.get("model_info", {}).get(selected, {}).get("model_weights", {})
    if not weights and "WeightedEnsemble" in info.get("model_info", {}):
        weights = info["model_info"]["WeightedEnsemble"].get("model_weights", {})
    return [
        {
            "scenario": scenario,
            "regime": regime,
            "selected_model": selected,
            "component": component,
            "weight": weight,
        }
        for component, weight in weights.items()
    ]


def _model_inventory(
    predictor, scenario: str, regime: str
) -> tuple[list[dict], list[dict]]:
    trained = set(predictor.model_names())
    leaderboard = predictor.leaderboard(silent=True).copy()
    leaderboard.insert(0, "regime", regime)
    leaderboard.insert(0, "scenario", scenario)
    rows = leaderboard.to_dict("records")
    failures = []
    for model in AUTOGLOUON_MODEL_LIBRARY:
        if not any(name == model or name.startswith(model + "_") for name in trained):
            failures.append(
                {
                    "scenario": scenario,
                    "regime": regime,
                    "model": model,
                    "status": "not_trained",
                    "reason": "failed, unavailable, or skipped after the time limit",
                }
            )
    return rows, failures


def _business_slices(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rolling = predictions.loc[predictions.split.eq("validation")]
    promo_columns = {
        "grocery": "discount",
        "wine": "promo_case_buy",
        "garden": "is_end_of_autumn_clearance",
        "luxury": "discount",
    }
    for scenario, group in rolling.groupby("scenario"):
        promotion = group[promo_columns[scenario]].gt(0)
        holiday = group.ds.isin(vic_holidays.ds)
        for condition, mask in [
            ("promotion", promotion),
            ("non_promotion", ~promotion),
            ("holiday", holiday),
            ("ordinary", ~promotion & ~holiday),
        ]:
            selected = group.loc[mask]
            if len(selected) and valid_labels(selected).any():
                rows.append(
                    {
                        "scenario": scenario,
                        "model": "AutoGluon Ensemble",
                        "condition": condition,
                        **metrics(selected, selected.prediction),
                    }
                )
    return pd.DataFrame(rows)


def _aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, split), group in predictions.groupby(["scenario", "split"]):
        rows.append(
            {
                "scenario": scenario,
                "model": "AutoGluon Ensemble",
                "split": split,
                **metrics(group, group.prediction),
            }
        )
    return pd.DataFrame(rows)


def _comparison(metrics_table: pd.DataFrame, old_dir: Path) -> pd.DataFrame:
    old = pd.read_csv(old_dir / "model_comparison.csv")
    old = old.loc[old.split.isin(["validation", "test"])].copy()
    current = metrics_table.copy()
    combined = pd.concat([old, current], ignore_index=True, sort=False)
    validation = (
        combined.loc[combined.split.eq("validation")]
        .set_index(["scenario", "model"])
        .WAPE
    )
    test = combined.loc[combined.split.eq("test")].set_index(["scenario", "model"]).WAPE
    gap = (test - validation).rename("test_minus_validation_WAPE").reset_index()
    return combined.merge(gap, on=["scenario", "model"], how="left")


def _feature_importance(predictor, history: pd.DataFrame, scenario: str, seconds: int):
    try:
        result = predictor.feature_importance(
            data=_as_timeseries(history, scenario),
            model=_selected_model(predictor),
            metric="MAE",
            features=known_covariate_names(scenario),
            time_limit=seconds,
            method="permutation",
            num_iterations=3,
            random_seed=SEED,
            include_confidence_band=True,
            confidence_level=0.95,
        ).reset_index(names="feature")
        result.insert(0, "scenario", scenario)
        result.insert(1, "model", "AutoGluon Ensemble")
        return result, None
    except Exception as exc:  # noqa: BLE001 - export explainability failures.
        return pd.DataFrame(), {
            "scenario": scenario,
            "regime": "final",
            "model": "feature_importance",
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def component_test_diagnostics(
    predictor,
    history: pd.DataFrame,
    future: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    """Score final-fit components without using test scores for model selection."""
    rows = []
    for model_name in predictor.model_names():
        forecast = _forecast(predictor, history, future, scenario, model_name)
        prediction = sales_prediction(future, forecast["mean"].to_numpy(float))
        rows.append(
            {
                "scenario": scenario,
                "component": model_name,
                **metrics(future, prediction),
            }
        )
    return pd.DataFrame(rows)


def _plot_results(predictions: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, scenario in zip(axes.flat, SCENARIOS):
        group = predictions.loc[
            predictions.scenario.eq(scenario) & predictions.split.eq("test")
        ]
        ax.fill_between(
            group.ds, group.q10, group.q90, alpha=0.2, label="10%–90% interval"
        )
        ax.plot(group.ds, group.y, "ko-", ms=3, label="Observed")
        ax.plot(group.ds, group.prediction, label="AutoGluon Ensemble")
        ax.set(title=f"{scenario} — final 28-day test", ylabel="Units")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
    fig.savefig(out / "final_test_predictions.png", dpi=150)
    plt.close(fig)


def plot_comparison(comparison: pd.DataFrame, weights: pd.DataFrame, out: Path) -> None:
    test = comparison.loc[comparison.split.eq("test")].copy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for ax, scenario in zip(axes.flat, SCENARIOS):
        group = test.loc[test.scenario.eq(scenario)].sort_values("WAPE")
        colors = [
            "#d55e00" if model == "AutoGluon Ensemble" else "#5790fc"
            for model in group.model
        ]
        ax.barh(group.model, group.WAPE * 100, color=colors)
        ax.invert_yaxis()
        ax.set(title=f"{scenario} — test WAPE", xlabel="WAPE (%)")
    fig.savefig(out / "comparison_with_all_models.png", dpi=150)
    plt.close(fig)

    final = weights.loc[weights.regime.eq("final")].copy()
    if len(final):
        pivot = final.pivot_table(
            index="component", columns="scenario", values="weight", fill_value=0
        )
        pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
        axes = pivot.plot.barh(
            subplots=True, layout=(2, 2), figsize=(14, 10), legend=False
        )
        for ax in np.asarray(axes).flat:
            ax.set_xlabel("Ensemble weight")
        plt.gcf().tight_layout()
        plt.gcf().savefig(out / "final_ensemble_weights.png", dpi=150)
        plt.close(plt.gcf())


def run_autogluon_benchmark(
    family: str = "autogluon_timeseries",
    validation_seconds: int = DEFAULT_VALIDATION_SECONDS,
    final_seconds: int = DEFAULT_FINAL_SECONDS,
    importance_seconds: int = DEFAULT_IMPORTANCE_SECONDS,
    old_comparison_dir: Path = OLD_COMPARISON,
):
    """Run leakage-safe AutoGluon experiments for the four fixed scenarios."""
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(CPU_THREADS))
    out = RUNS_DIR / family
    out.mkdir(parents=True, exist_ok=False)
    predictor_root = out / "predictors"
    metric_rows: list[dict] = []
    prediction_tables: list[pd.DataFrame] = []
    leaderboard_rows: list[dict] = []
    weight_rows: list[dict] = []
    fit_rows: list[dict] = []
    failure_rows: list[dict] = []
    importance_tables: list[pd.DataFrame] = []
    component_tables: list[pd.DataFrame] = []
    final_predictors = {}

    for scenario, frame in datasets.items():
        for window in windows(scenario):
            split = "test" if window["split"] == "test" else "validation"
            regime = window["regime"]
            origin = window["origin"]
            history = frame.iloc[: origin + 1].copy()
            future = frame.iloc[origin + 1 : origin + HORIZON + 1].copy()
            assert history.ds.max() < future.ds.min()
            time_limit = final_seconds if split == "test" else validation_seconds
            print(f"AutoGluon {scenario}/{regime} ({time_limit}s budget)", flush=True)
            predictor, elapsed = fit_predictor(
                history,
                scenario,
                predictor_root / scenario / regime,
                time_limit,
            )
            selected = _selected_model(predictor)
            table = _prediction_table(
                predictor, history, future, scenario, split, regime, selected
            )
            prediction_tables.append(table)
            metric_rows.append(
                {
                    "scenario": scenario,
                    "model": "AutoGluon Ensemble",
                    "split": split,
                    "regime": regime,
                    **metrics(table, table.prediction),
                }
            )
            leaderboard, missing = _model_inventory(predictor, scenario, regime)
            leaderboard_rows.extend(leaderboard)
            failure_rows.extend(missing)
            weight_rows.extend(_extract_ensemble_weights(predictor, scenario, regime))
            info = predictor.info()
            fit_rows.append(
                {
                    "scenario": scenario,
                    "regime": regime,
                    "split": split,
                    "training_rows": len(history),
                    "time_limit_seconds": time_limit,
                    "fit_seconds": elapsed,
                    "selected_model": selected,
                    "models_trained": len(predictor.model_names()),
                    "best_validation_score": info.get("best_model_score_val"),
                }
            )
            if split == "test":
                final_predictors[scenario] = predictor
                component_tables.append(
                    component_test_diagnostics(predictor, history, future, scenario)
                )
                importance, error = _feature_importance(
                    predictor, history, scenario, importance_seconds
                )
                if len(importance):
                    importance_tables.append(importance)
                if error:
                    failure_rows.append(error)

        pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False)
        pd.concat(prediction_tables, ignore_index=True).to_csv(
            out / "predictions.csv", index=False
        )
        pd.DataFrame(leaderboard_rows).to_csv(out / "leaderboard.csv", index=False)
        pd.DataFrame(fit_rows).to_csv(out / "fit_details.csv", index=False)

    predictions = pd.concat(prediction_tables, ignore_index=True)
    window_metrics = pd.DataFrame(metric_rows)
    aggregate = _aggregate_metrics(predictions)
    comparison = _comparison(aggregate, Path(old_comparison_dir))
    weights = pd.DataFrame(weight_rows)
    failures = pd.DataFrame(failure_rows)
    importance = (
        pd.concat(importance_tables, ignore_index=True)
        if importance_tables
        else pd.DataFrame(columns=["scenario", "model", "feature", "importance"])
    )
    slices = _business_slices(predictions)
    component_diagnostics = pd.concat(component_tables, ignore_index=True)

    window_metrics.to_csv(out / "metrics.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(leaderboard_rows).to_csv(out / "leaderboard.csv", index=False)
    weights.to_csv(out / "ensemble_weights.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(out / "fit_details.csv", index=False)
    slices.to_csv(out / "business_slices.csv", index=False)
    importance.to_csv(out / "feature_importance.csv", index=False)
    failures.to_csv(out / "model_failures.csv", index=False)
    component_diagnostics.to_csv(out / "component_test_diagnostics.csv", index=False)
    comparison.to_csv(out / "comparison_with_all_models.csv", index=False)
    aggregate.to_csv(out / "aggregate_metrics.csv", index=False)
    _plot_results(predictions, out)
    plot_comparison(comparison, weights, out)

    for name, digest in INPUT_HASHES.items():
        assert hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() == digest
    old_manifest = json.loads((Path(old_comparison_dir) / "manifest.json").read_text())
    assert old_manifest["hashes"] == INPUT_HASHES
    manifest = {
        "family": family,
        "model": "AutoGluon Ensemble",
        "autogluon_version": "1.5.0",
        "protocol_version": "retail-benchmark-v2-autogluon-v1",
        "hashes": INPUT_HASHES,
        "seed": SEED,
        "cpu_threads": CPU_THREADS,
        "prediction_length": HORIZON,
        "validation_seconds_per_window": validation_seconds,
        "final_seconds_per_scenario": final_seconds,
        "importance_seconds_per_scenario": importance_seconds,
        "models_requested": list(AUTOGLOUON_MODEL_LIBRARY),
        "models_excluded": list(EXCLUDED_FOUNDATION_MODELS),
        "windows": {scenario: windows(scenario) for scenario in SCENARIOS},
        "comparison_source": str(Path(old_comparison_dir).resolve()),
        "train_hindcast": "not reported because AutoGluon has no directly comparable in-sample hindcast",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {
        "metrics": window_metrics,
        "aggregate_metrics": aggregate,
        "predictions": predictions,
        "leaderboard": pd.DataFrame(leaderboard_rows),
        "ensemble_weights": weights,
        "fit_details": pd.DataFrame(fit_rows),
        "business_slices": slices,
        "feature_importance": importance,
        "model_failures": failures,
        "component_test_diagnostics": component_diagnostics,
        "comparison": comparison,
        "output_dir": out,
        "predictors": final_predictors,
    }
