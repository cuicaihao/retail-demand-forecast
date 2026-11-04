"""
Baseline Forecasting Models for Retail Demand Forecasting Benchmark.

Includes:
1. SeasonalNaiveForecaster: 7-day lag seasonal repeat tiled over the forecast horizon.
2. MovingAverageForecaster: Simple rolling moving average over recent historical window.
3. ZeroDemandForecaster: Pure zero-demand baseline for intermittent demand comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .benchmark_registry import BaseRetailForecaster, ForecastOutput


class SeasonalNaiveForecaster(BaseRetailForecaster):
    """
    7-day seasonal naive baseline.
    Repeats the most recent full week (7 calendar days) across the forecast horizon.
    Zeros out predictions on store closure days (is_open == False).
    """

    def __init__(self, lag_days: int = 7):
        super().__init__(name="Seasonal naive", family="baseline")
        self.lag_days = lag_days
        self._pattern: np.ndarray | None = None

    def fit(self, history: pd.DataFrame, scenario: str) -> None:
        assert len(history) >= self.lag_days, "History too short for seasonal naive"
        # Extract the last lag_days of sales
        self._pattern = history.iloc[-self.lag_days :]["y"].values.astype(float)

    def predict(self, future: pd.DataFrame, scenario: str) -> ForecastOutput:
        assert self._pattern is not None, "Model must be fitted before predict"
        horizon = len(future)
        reps = int(np.ceil(horizon / self.lag_days))
        tiled = np.tile(self._pattern, reps)[:horizon]

        # Apply store closure mask (0 on closed days, >=0 on open days)
        if "is_open" in future.columns:
            preds = np.where(future["is_open"], np.maximum(tiled, 0.0), 0.0)
        else:
            preds = np.maximum(tiled, 0.0)

        return ForecastOutput(point_forecast=preds)


class MovingAverageForecaster(BaseRetailForecaster):
    """
    Rolling moving average baseline over the last K open trading days.
    """

    def __init__(self, window_days: int = 14):
        super().__init__(name=f"Moving Average ({window_days}d)", family="baseline")
        self.window_days = window_days
        self._mean_val: float = 0.0

    def fit(self, history: pd.DataFrame, scenario: str) -> None:
        open_history = (
            history[history["is_open"]] if "is_open" in history.columns else history
        )
        sample = open_history.tail(self.window_days)["y"].values
        self._mean_val = float(np.mean(sample)) if len(sample) > 0 else 0.0

    def predict(self, future: pd.DataFrame, scenario: str) -> ForecastOutput:
        horizon = len(future)
        raw = np.full(horizon, self._mean_val, dtype=float)
        if "is_open" in future.columns:
            preds = np.where(future["is_open"], raw, 0.0)
        else:
            preds = raw
        return ForecastOutput(point_forecast=preds)
