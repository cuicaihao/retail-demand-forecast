"""Forecast adapters and pre-evaluation parameter selection for V4 benchmarks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .arima import fit_arima, forecast_arima
from .baselines import MovingAverageForecaster, SeasonalNaiveForecaster
from .benchmark_registry import BaseRetailForecaster, ForecastOutput, ModelConfig
from .common import INPUT_HASHES, datasets, sales_prediction, windows
from .neural import (
    NEURAL_CANDIDATES,
    fit_neural_config,
    predict_neural_config,
    tune_neural,
)
from .prophet_benchmark import fit_prophet, predict_prophet, tune_prophet
from .trees import fit_model, predict_model, tune_model


class TreeForecasterAdapter(BaseRetailForecaster):
    """Adapt the existing origin-frozen tree implementation to the V4 interface."""

    def __init__(self, model_name: str, params: dict[str, Any]):
        super().__init__(name=model_name, family="tree_gbdt")
        self.model_name = model_name
        self.params = params
        self.bundle: dict[str, Any] | None = None
        self.history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame, scenario: str) -> None:
        self.history = history.copy()
        self.bundle, _ = fit_model(history, scenario, self.model_name, self.params)

    def predict(self, future: pd.DataFrame, scenario: str) -> ForecastOutput:
        if self.bundle is None or self.history is None:
            raise RuntimeError("Tree adapter must be fitted before prediction")
        raw = predict_model(self.bundle, self.history, future, scenario)
        return ForecastOutput(point_forecast=sales_prediction(future, raw))


class ProphetForecasterAdapter(BaseRetailForecaster):
    """Adapt Prophet while keeping all future target values outside inference."""

    def __init__(self, params: dict[str, Any]):
        super().__init__(name="Prophet", family="statistical")
        self.params = params
        self.model: Any = None

    def fit(self, history: pd.DataFrame, scenario: str) -> None:
        self.model, _ = fit_prophet(history, scenario, self.params)

    def predict(self, future: pd.DataFrame, scenario: str) -> ForecastOutput:
        if self.model is None:
            raise RuntimeError("Prophet adapter must be fitted before prediction")
        raw = predict_prophet(self.model, future, scenario)
        return ForecastOutput(point_forecast=sales_prediction(future, raw))


class ARIMAForecasterAdapter(BaseRetailForecaster):
    """Statsmodels regression with ARIMA errors and known-future covariates."""

    def __init__(self, params: dict[str, Any]):
        super().__init__(name="Regression + ARIMA", family="statistical")
        self.params = params
        self.bundle: dict[str, Any] | None = None
        self.history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame, scenario: str) -> None:
        self.history = history.copy()
        self.bundle, details = fit_arima(
            history, scenario, "Regression + ARIMA", self.params
        )
        if not details["converged"]:
            raise RuntimeError(f"ARIMA did not converge for {scenario}")

    def predict(self, future: pd.DataFrame, scenario: str) -> ForecastOutput:
        if self.bundle is None or self.history is None:
            raise RuntimeError("ARIMA adapter must be fitted before prediction")
        raw, interval = forecast_arima(
            self.bundle, self.history, future, scenario, alpha=0.2
        )
        return ForecastOutput(
            point_forecast=sales_prediction(future, raw),
            q10=sales_prediction(future, interval[:, 0]),
            q90=sales_prediction(future, interval[:, 1]),
        )


class NeuralForecasterAdapter(BaseRetailForecaster):
    """Origin-frozen MLP, LSTM, or TFT trained independently for every fold."""

    def __init__(self, model_name: str, params: dict[str, Any]):
        super().__init__(name=model_name, family="neural")
        self.model_name = model_name
        self.params = params
        self.bundle: dict[str, Any] | None = None
        self.history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame, scenario: str) -> None:
        self.history = history.copy()
        self.bundle, _ = fit_neural_config(
            history,
            scenario,
            self.model_name,
            self.params,
            checkpoint_tag=f"v4_{self.model_name.lower()}_{scenario}",
            save_checkpoint=False,
        )

    def predict(self, future: pd.DataFrame, scenario: str) -> ForecastOutput:
        if self.bundle is None or self.history is None:
            raise RuntimeError("Neural adapter must be fitted before prediction")
        raw = predict_neural_config(
            self.bundle,
            self.history,
            future,
            scenario,
            self.model_name,
        )
        return ForecastOutput(point_forecast=sales_prediction(future, raw))


TREE_TOOLKITS = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}


def baseline_factory(
    config: ModelConfig, scenario: str, params: dict[str, Any]
) -> BaseRetailForecaster:
    del scenario
    if config.model_id == "seasonal_naive":
        return SeasonalNaiveForecaster(lag_days=int(params.get("lag_days", 7)))
    if config.model_id == "moving_average_14d":
        return MovingAverageForecaster(window_days=int(params.get("window_days", 14)))
    raise KeyError(f"Unknown baseline adapter: {config.model_id}")


def tree_factory(
    config: ModelConfig, scenario: str, params: dict[str, Any]
) -> BaseRetailForecaster:
    del scenario
    return TreeForecasterAdapter(TREE_TOOLKITS[config.toolkit], params)


def prophet_factory(
    config: ModelConfig, scenario: str, params: dict[str, Any]
) -> BaseRetailForecaster:
    del config, scenario
    return ProphetForecasterAdapter(params)


def arima_factory(
    config: ModelConfig, scenario: str, params: dict[str, Any]
) -> BaseRetailForecaster:
    del config, scenario
    return ARIMAForecasterAdapter(params)


def neural_factory(
    config: ModelConfig, scenario: str, params: dict[str, Any]
) -> BaseRetailForecaster:
    del scenario
    return NeuralForecasterAdapter(config.model_name, params)


def _selection_windows_end(scenario: str) -> pd.Timestamp:
    frame = datasets[scenario]
    ends = []
    for window in windows(scenario)[:-1]:
        origin = window["origin"]
        ends.append(frame.iloc[origin + 28].ds)
    return max(ends)


def _selection_identity(config: ModelConfig) -> str:
    if config.family == "neural":
        return config.model_id
    return config.toolkit


def _selection_cache_path(cache_dir: Path, selection_identity: str) -> Path:
    return cache_dir / f"{selection_identity}.json"


def _selection_cache_key(spec: Any, config: ModelConfig) -> str:
    selection_identity = _selection_identity(config)
    value = {
        "selection_policy": spec.selection_policy,
        "selection_end": str(spec.selection_end.date()),
        "selection_identity": selection_identity,
        "input_hashes": INPUT_HASHES,
    }
    if config.family == "neural":
        value["candidate_grid"] = NEURAL_CANDIDATES[config.model_name]
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _select_base_params(
    config: ModelConfig, spec: Any, cache_dir: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Select base capacity only on fixed windows before formal evaluation."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    selection_identity = _selection_identity(config)
    cache_path = _selection_cache_path(cache_dir, selection_identity)
    cache_key = _selection_cache_key(spec, config)
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("selection_cache_key") == cache_key:
            return cached["base_params_by_scenario"], cached

    selection_details: dict[str, Any] = {}
    if config.family == "baseline":
        base = {scenario: {} for scenario in spec.scenarios}
    elif config.family in {"tree_classical", "tree_gbdt"}:
        model_name = TREE_TOOLKITS[config.toolkit]
        base = {}
        for scenario in spec.scenarios:
            if _selection_windows_end(scenario) > spec.selection_end:
                raise ValueError(
                    f"{scenario} parameter selection crosses formal evaluation boundary"
                )
            selected, _, _ = tune_model(scenario, model_name)
            base[scenario] = selected
    elif config.model_id == "arima":
        base = {scenario: {} for scenario in spec.scenarios}
    elif config.toolkit == "prophet":
        base = {}
        for scenario in spec.scenarios:
            if _selection_windows_end(scenario) > spec.selection_end:
                raise ValueError(
                    f"{scenario} parameter selection crosses formal evaluation boundary"
                )
            selected, _, _ = tune_prophet(scenario)
            base[scenario] = selected
    elif config.family == "neural":
        base = {}
        for scenario in spec.scenarios:
            if _selection_windows_end(scenario) > spec.selection_end:
                raise ValueError(
                    f"{scenario} parameter selection crosses formal evaluation boundary"
                )
            selected, table, _ = tune_neural(scenario, config.model_name)
            base[scenario] = selected
            selection_details[scenario] = json.loads(table.to_json(orient="records"))
    else:
        raise NotImplementedError(
            f"No V4 parameter selection adapter for toolkit {config.toolkit!r}"
        )

    record = {
        "selection_cache_key": cache_key,
        "selection_policy": spec.selection_policy,
        "selection_end": str(spec.selection_end.date()),
        "toolkit": config.toolkit,
        "selection_identity": selection_identity,
        "input_hashes": INPUT_HASHES,
        "candidate_grid": NEURAL_CANDIDATES.get(config.model_name),
        "selection_details_by_scenario": selection_details,
        "base_params_by_scenario": base,
    }
    cache_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return base, record


def resolve_model_parameters(
    config: ModelConfig, spec: Any, cache_dir: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Freeze complete scenario parameters before any formal fold is evaluated."""
    base_params, selection = _select_base_params(config, spec, cache_dir)
    resolved: dict[str, dict[str, Any]] = {}
    for scenario in spec.scenarios:
        merged = dict(base_params[scenario])
        merged.update(config.params)
        resolved[scenario] = merged
    record = {
        **selection,
        "model_id": config.model_id,
        "declared_variant_params": config.params,
        "resolved_params_by_scenario": resolved,
    }
    return resolved, record
