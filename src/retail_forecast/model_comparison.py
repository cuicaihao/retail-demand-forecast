from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import *

CORE_MODELS = [
    "Prophet",
    "ARIMA",
    "Decision Tree",
    "Random Forest",
    "LightGBM",
    "XGBoost",
    "CatBoost",
    "MLP",
    "LSTM",
    "TFT",
]
FAMILY_DIRECTORIES = {
    "Decision Tree": "tree_decision_tree",
    "Random Forest": "tree_random_forest",
    "LightGBM": "tree_lightgbm",
    "XGBoost": "tree_xgboost",
    "CatBoost": "tree_catboost",
    "MLP": "neural_mlp",
    "LSTM": "neural_lstm",
    "TFT": "neural_tft",
    "ARIMA": "arima",
}


def _manifest_matches(path: Path):
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return False
    record = json.loads(manifest.read_text())
    return (record.get("hashes") or record.get("input_sha256")) == INPUT_HASHES


def latest_compatible(directory: str, required=("metrics.csv", "predictions.csv")):
    candidates = sorted(
        RUNS_DIR.parent.glob(f"*/{directory}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if _manifest_matches(path) and all(
            (path / name).is_file() for name in required
        ):
            return path
    raise FileNotFoundError(f"No hash-compatible result found for {directory}")


def _load_prophet():
    paths = [
        latest_compatible("prophet_retail"),
        latest_compatible("prophet_luxury"),
    ]
    manifests = [json.loads((path / "manifest.json").read_text()) for path in paths]
    scenarios = [scenario for record in manifests for scenario in record["scenarios"]]
    if sorted(scenarios) != sorted(SCENARIOS) or len(scenarios) != len(set(scenarios)):
        raise ValueError("Prophet result parts must cover each scenario exactly once")
    return paths


def _baseline_tables():
    metric_rows, prediction_tables = [], []
    for scenario, frame in datasets.items():
        for window in windows(scenario):
            origin = window["origin"]
            history = frame.iloc[: origin + 1]
            future = frame.iloc[origin + 1 : origin + 1 + HORIZON].copy()
            future["prediction"] = sales_prediction(
                future, seasonal_naive(history, future)
            )
            future["scenario"] = scenario
            future["model"] = "Seasonal naive"
            future["split"] = "test" if window["split"] == "test" else "validation"
            future["regime"] = window["regime"]
            future["forecast_origin"] = history.ds.iloc[-1]
            prediction_tables.append(future)
            metric_rows.append(
                {
                    "scenario": scenario,
                    "model": "Seasonal naive",
                    "split": future["split"].iloc[0],
                    "regime": window["regime"],
                    **metrics(future, future.prediction),
                }
            )
        blocks = []
        for start in training_blocks():
            history = frame.iloc[:start]
            block = frame.iloc[start : start + HORIZON].copy()
            block["prediction"] = sales_prediction(
                block, seasonal_naive(history, block)
            )
            blocks.append(block)
        hindcast = pd.concat(blocks, ignore_index=True)
        hindcast["scenario"] = scenario
        hindcast["model"] = "Seasonal naive"
        hindcast["split"] = "train_hindcast"
        hindcast["regime"] = "training"
        prediction_tables.append(hindcast)
        metric_rows.append(
            {
                "scenario": scenario,
                "model": "Seasonal naive",
                "split": "train_hindcast",
                "regime": "training",
                **metrics(hindcast, hindcast.prediction),
            }
        )
    return pd.DataFrame(metric_rows), pd.concat(prediction_tables, ignore_index=True)


def load_all_results():
    sources = {"Prophet": _load_prophet()}
    for model, directory in FAMILY_DIRECTORIES.items():
        sources[model] = [latest_compatible(directory)]
    (
        metrics_tables,
        prediction_tables,
        timing_tables,
        tuning_tables,
        explanation_tables,
    ) = (
        [],
        [],
        [],
        [],
        [],
    )
    for model, paths in sources.items():
        for path in paths:
            metrics_table = pd.read_csv(path / "metrics.csv")
            predictions = pd.read_csv(
                path / "predictions.csv", parse_dates=["ds", "forecast_origin"]
            )
            if model == "ARIMA":
                metrics_table = metrics_table.loc[
                    metrics_table.model.eq("Regression + ARIMA")
                ].copy()
                predictions = predictions.loc[
                    predictions.model.eq("Regression + ARIMA")
                ].copy()
                metrics_table["model"] = "ARIMA"
                predictions["model"] = "ARIMA"
            metrics_table["split"] = metrics_table.split.replace(
                {"rolling": "validation"}
            )
            predictions["split"] = predictions.split.replace({"rolling": "validation"})
            metrics_tables.append(metrics_table)
            prediction_tables.append(predictions)
            fit_path = path / "fit_details.csv"
            if fit_path.is_file():
                fit = pd.read_csv(fit_path)
                if model == "ARIMA":
                    fit = fit.loc[fit.model.eq("Regression + ARIMA")].copy()
                    fit["model"] = "ARIMA"
                timing_tables.append(fit)
            tuning_path = path / "tuning_results.csv"
            if tuning_path.is_file():
                tuning_tables.append(pd.read_csv(tuning_path))
            for filename in ["explanations.csv", "feature_importance.csv"]:
                explanation_path = path / filename
                if explanation_path.is_file():
                    explanation_tables.append(pd.read_csv(explanation_path))
                    break
    baseline_metrics, baseline_predictions = _baseline_tables()
    metrics_tables.append(baseline_metrics)
    prediction_tables.append(baseline_predictions)
    return {
        "metrics": pd.concat(metrics_tables, ignore_index=True),
        "predictions": pd.concat(prediction_tables, ignore_index=True),
        "timing": pd.concat(timing_tables, ignore_index=True),
        "tuning": pd.concat(tuning_tables, ignore_index=True),
        "explanations": pd.concat(explanation_tables, ignore_index=True),
    }, sources


def summarize_all(tables):
    metrics_table = tables["metrics"]
    summary = metrics_table.groupby(["scenario", "model", "split"], as_index=False).agg(
        MAE=("MAE", "mean"),
        RMSE=("RMSE", "mean"),
        WAPE=("WAPE", "mean"),
        Bias=("Bias", "mean"),
        n=("n", "sum"),
    )
    baseline = summary.loc[summary.model.eq("Seasonal naive")][
        ["scenario", "split", "WAPE"]
    ].rename(columns={"WAPE": "naive_WAPE"})
    summary = summary.merge(baseline, on=["scenario", "split"], how="left")
    summary["improvement_vs_naive"] = 1 - summary.WAPE / summary.naive_WAPE
    summary = summary.drop(columns="naive_WAPE")
    train = (
        summary.loc[summary.split.eq("train_hindcast")]
        .set_index(["scenario", "model"])
        .WAPE
    )
    test = summary.loc[summary.split.eq("test")].set_index(["scenario", "model"]).WAPE
    gap = (test - train).rename("test_minus_train_WAPE").reset_index()
    return summary.merge(gap, on=["scenario", "model"], how="left")


def export_all_comparison(tables, sources):
    out = RUNS_DIR / "all_model_comparison"
    out.mkdir(parents=True, exist_ok=False)
    summary = summarize_all(tables)
    summary.to_csv(out / "model_comparison.csv", index=False)
    tables["predictions"].to_csv(out / "predictions.csv", index=False)
    tables["timing"].to_csv(out / "fit_details.csv", index=False)
    tuning_time = (
        tables["tuning"]
        .groupby(["scenario", "model"], as_index=False)
        .tuning_seconds.sum()
        .rename(columns={"tuning_seconds": "total_tuning_seconds"})
    )
    tuning_time.to_csv(out / "tuning_time.csv", index=False)
    tables["explanations"].to_csv(out / "explanations.csv", index=False)
    slices = []
    validation_predictions = tables["predictions"].loc[
        tables["predictions"].split.eq("validation")
    ]
    for (scenario, model), group in validation_predictions.groupby(
        ["scenario", "model"]
    ):
        promotion_column = {
            "grocery": "discount",
            "wine": "promo_case_buy",
            "garden": "is_end_of_autumn_clearance",
            "luxury": "discount",
        }[scenario]
        promotion = group[promotion_column].gt(0)
        holiday = group.ds.isin(vic_holidays.ds)
        for condition, mask in [
            ("promotion", promotion),
            ("non_promotion", ~promotion),
            ("holiday", holiday),
            ("ordinary", ~promotion & ~holiday),
        ]:
            selected = group.loc[mask]
            if valid_labels(selected).any():
                slices.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "condition": condition,
                        **metrics(selected, selected.prediction),
                    }
                )
    pd.DataFrame(slices).to_csv(out / "business_slices.csv", index=False)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "family": "all_model_comparison",
                "models": CORE_MODELS,
                "baseline": "Seasonal naive",
                "protocol_version": "retail-benchmark-v2",
                "hashes": INPUT_HASHES,
                "sources": {
                    model: [str(path) for path in paths]
                    for model, paths in sources.items()
                },
            },
            indent=2,
        )
    )
    return summary, out


def plot_all_comparison(tables, summary, out):
    display_order = [*CORE_MODELS, "Seasonal naive"]
    fig, axes = plt.subplots(
        1, len(SCENARIOS), figsize=(24, 5), constrained_layout=True
    )
    for ax, scenario in zip(axes, SCENARIOS):
        table = (
            summary.loc[summary.scenario.eq(scenario)]
            .pivot(index="split", columns="model", values="WAPE")
            .reindex(
                index=["train_hindcast", "validation", "test"],
                columns=display_order,
            )
            * 100
        )
        image = ax.imshow(table.to_numpy(), aspect="auto", cmap="YlOrRd")
        ax.set(
            title=scenario,
            xticks=range(len(display_order)),
            xticklabels=display_order,
            yticks=range(3),
            yticklabels=["Train", "Validation", "Test"],
        )
        ax.tick_params(axis="x", rotation=55)
        fig.colorbar(image, ax=ax, shrink=0.7)
    fig.savefig(out / "all_model_metrics.png", dpi=150)
    plt.show()
    plt.close(fig)

    predictions = tables["predictions"]
    for scenario in SCENARIOS:
        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
        group = predictions.loc[
            (predictions.scenario == scenario) & predictions.split.eq("test")
        ]
        first = group.loc[group.model.eq(CORE_MODELS[0])]
        ax.plot(first.ds, first.y, "ko-", ms=3, label="Observed")
        for model in CORE_MODELS:
            values = group.loc[group.model.eq(model)]
            ax.plot(values.ds, values.prediction, label=model)
        ax.set(title=f"{scenario} final 28-day test", ylabel="Units")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(ncol=4)
        fig.savefig(out / f"{scenario}_test_predictions.png", dpi=150)
        plt.show()
        plt.close(fig)
