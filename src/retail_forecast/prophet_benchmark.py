from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json

from .common import *

PROTOCOL_VERSION = "retail-benchmark-v2"
PROPHET_CANDIDATES = [
    {
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 1.0,
        "seasonality_mode": "additive",
    },
    {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "seasonality_mode": "additive",
    },
    {
        "changepoint_prior_scale": 0.1,
        "seasonality_prior_scale": 10.0,
        "seasonality_mode": "additive",
    },
    {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "seasonality_mode": "multiplicative",
    },
]


def prophet_frame(frame: pd.DataFrame, scenario: str, include_y: bool = True):
    columns = ["ds", *SCENARIOS[scenario]["features"]]
    if include_y:
        columns.insert(1, "y")
    return frame[columns].copy()


def new_prophet(scenario: str, parameters: dict):
    model = Prophet(
        holidays=vic_holidays[["ds", "holiday"]],
        weekly_seasonality=3,
        yearly_seasonality=5,
        daily_seasonality=False,
        interval_width=0.8,
        **parameters,
    )
    for feature in SCENARIOS[scenario]["features"]:
        model.add_regressor(feature, standardize=True)
    return model


def fit_prophet(history, scenario, parameters):
    started = time.perf_counter()
    valid = valid_labels(history)
    training = prophet_frame(history.loc[valid], scenario)
    model = new_prophet(scenario, parameters).fit(training)
    return model, {
        "training_rows": int(valid.sum()),
        "setup_seconds": time.perf_counter() - started,
    }


def predict_prophet(model, future, scenario):
    features = prophet_frame(future, scenario, include_y=False)
    changed = future.copy()
    changed["y"] = 12345.0
    pd.testing.assert_frame_equal(
        features, prophet_frame(changed, scenario, include_y=False)
    )
    return model.predict(features)["yhat"].to_numpy(float)


def _prediction_frame(future, raw, scenario, split, regime, origin):
    result = future.copy()
    result["raw_prediction"] = np.asarray(raw, float)
    result["prediction"] = sales_prediction(result, raw)
    result["scenario"] = scenario
    result["model"] = "Prophet"
    result["split"] = split
    result["regime"] = regime
    result["forecast_origin"] = origin
    result["horizon_day"] = np.arange(1, len(result) + 1)
    return result


def tune_prophet(scenario: str):
    frame = datasets[scenario]
    rows, cached = [], {}
    for candidate_id, parameters in enumerate(PROPHET_CANDIDATES):
        predictions, elapsed = [], 0.0
        for window in windows(scenario)[:-1]:
            origin = window["origin"]
            history = frame.iloc[: origin + 1]
            future = frame.iloc[origin + 1 : origin + 1 + HORIZON]
            model, detail = fit_prophet(history, scenario, parameters)
            elapsed += detail["setup_seconds"]
            predictions.append(
                _prediction_frame(
                    future,
                    predict_prophet(model, future, scenario),
                    scenario,
                    "validation",
                    window["regime"],
                    history.ds.iloc[-1],
                )
            )
        joined = pd.concat(predictions, ignore_index=True)
        score = metrics(joined, joined.prediction)
        rows.append(
            {
                "scenario": scenario,
                "model": "Prophet",
                "candidate_id": candidate_id,
                "parameters": json.dumps(parameters, sort_keys=True),
                "validation_MAE": score["MAE"],
                "validation_RMSE": score["RMSE"],
                "validation_WAPE": score["WAPE"],
                "tuning_seconds": elapsed,
            }
        )
        cached[candidate_id] = joined
    table = pd.DataFrame(rows).sort_values(
        ["validation_MAE", "validation_RMSE", "candidate_id"]
    )
    best_id = int(table.iloc[0].candidate_id)
    return PROPHET_CANDIDATES[best_id], table, cached[best_id]


def run_prophet_benchmark(scenarios, family: str):
    out = RUNS_DIR / family
    out.mkdir(parents=True, exist_ok=False)
    metric_rows, predictions, tuning_tables = [], [], []
    parameter_rows, fit_rows, explanation_rows = [], [], []
    for scenario in scenarios:
        frame = datasets[scenario]
        print(f"Prophet: tuning {scenario}", flush=True)
        parameters, tuning, validation = tune_prophet(scenario)
        tuning_tables.append(tuning)
        parameter_rows.append(
            {
                "scenario": scenario,
                "model": "Prophet",
                "parameters": json.dumps(parameters, sort_keys=True),
            }
        )
        predictions.append(validation)
        for regime, group in validation.groupby("regime", sort=False):
            metric_rows.append(
                {
                    "scenario": scenario,
                    "model": "Prophet",
                    "split": "validation",
                    "regime": regime,
                    **metrics(group, group.prediction),
                }
            )

        origin = windows(scenario)[-1]["origin"]
        history = frame.iloc[: origin + 1]
        future = frame.iloc[origin + 1 : origin + 1 + HORIZON]
        model, detail = fit_prophet(history, scenario, parameters)
        fit_rows.append({"scenario": scenario, "model": "Prophet", **detail})
        test_forecast = model.predict(prophet_frame(future, scenario, include_y=False))
        test = _prediction_frame(
            future,
            test_forecast.yhat,
            scenario,
            "test",
            "final",
            history.ds.iloc[-1],
        )
        predictions.append(test)
        metric_rows.append(
            {
                "scenario": scenario,
                "model": "Prophet",
                "split": "test",
                "regime": "final",
                **metrics(test, test.prediction),
            }
        )
        blocks = []
        for start in training_blocks():
            past = frame.iloc[:start]
            block = frame.iloc[start : start + HORIZON]
            blocks.append(
                _prediction_frame(
                    block,
                    predict_prophet(model, block, scenario),
                    scenario,
                    "train_hindcast",
                    "training",
                    past.ds.iloc[-1],
                )
            )
        hindcast = pd.concat(blocks, ignore_index=True)
        predictions.append(hindcast)
        metric_rows.append(
            {
                "scenario": scenario,
                "model": "Prophet",
                "split": "train_hindcast",
                "regime": "training",
                **metrics(hindcast, hindcast.prediction),
            }
        )
        component_names = [
            "trend",
            "weekly",
            "yearly",
            "holidays",
            *SCENARIOS[scenario]["features"],
        ]
        for name in component_names:
            if name in test_forecast:
                explanation_rows.append(
                    {
                        "scenario": scenario,
                        "model": "Prophet",
                        "feature": name,
                        "importance_type": "mean_absolute_component",
                        "importance": float(test_forecast[name].abs().mean()),
                    }
                )
        (out / f"{scenario}_model.json").write_text(model_to_json(model))

    tables = {
        "metrics": pd.DataFrame(metric_rows),
        "predictions": pd.concat(predictions, ignore_index=True),
        "tuning_results": pd.concat(tuning_tables, ignore_index=True),
        "selected_parameters": pd.DataFrame(parameter_rows),
        "fit_details": pd.DataFrame(fit_rows),
        "explanations": pd.DataFrame(explanation_rows),
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "family": family,
                "model": "Prophet",
                "scenarios": list(scenarios),
                "protocol_version": PROTOCOL_VERSION,
                "hashes": INPUT_HASHES,
                "seed": SEED,
                "selection_metric": "pooled rolling validation MAE",
            },
            indent=2,
        )
    )
    return (*tables.values(), out)
