"""TimesFM 3.0 adapters for the frozen V4 retail benchmark protocols."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .benchmark_registry import BaseRetailForecaster, ForecastOutput, ModelConfig
from .common import make_features, sales_prediction, valid_labels

TIMESFM_CHECKPOINT = "google/timesfm-3.0-pytorch"
TIMESFM_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
TIMESFM_DEVICE = "cpu"
TIMESFM_BATCH_SIZE = 1
TIMESFM_HISTORY_IMPUTATION = "mask-invalid-then-forward-fill-backfill"


def timesfm_context(history: pd.DataFrame) -> np.ndarray:
    """Build a regular daily target context without diagnostic truth columns."""
    context = history["y"].where(valid_labels(history)).ffill().bfill()
    if context.isna().any():
        raise ValueError("TimesFM requires at least one valid historical label")
    values = context.to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("TimesFM context contains non-finite values")
    return values


def timesfm_known_covariates(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Return only calendar and business inputs known at the forecast origin."""
    covariates = make_features(frame, scenario).reset_index(drop=True)
    covariates.insert(0, "is_open", frame["is_open"].astype(float).to_numpy())
    forbidden = {
        "y",
        "stockout",
        "conditional_mean",
        "latent_demand",
        "occurrence_probability",
        "positive_demand",
    }
    if forbidden.intersection(covariates.columns):
        raise AssertionError("TimesFM covariates expose target or diagnostic truth")
    if not np.isfinite(covariates.to_numpy(float)).all():
        raise ValueError("TimesFM covariates contain non-finite values")
    return covariates.astype(np.float32)


def timesfm_covariate_matrix(
    history: pd.DataFrame, future: pd.DataFrame, scenario: str
) -> np.ndarray:
    """Create the TimesFM 3.0 (variates, context + horizon) input matrix."""
    combined = pd.concat([history, future], ignore_index=True)
    matrix = timesfm_known_covariates(combined, scenario).to_numpy(np.float32).T
    expected_length = len(history) + len(future)
    if matrix.ndim != 2 or matrix.shape[1] != expected_length:
        raise ValueError(
            "TimesFM covariates must have shape (variates, context + horizon)"
        )
    return matrix


@lru_cache(maxsize=2)
def load_timesfm_model(
    checkpoint: str = TIMESFM_CHECKPOINT,
    revision: str = TIMESFM_REVISION,
    device: str = TIMESFM_DEVICE,
    per_core_batch_size: int = TIMESFM_BATCH_SIZE,
):
    """Load one process-wide checkpoint so folds never reload the 1.2 GB weights."""
    import timesfm

    return timesfm.TimesFM3Forecaster.from_pretrained(
        checkpoint,
        revision=revision,
        device=device,
        per_core_batch_size=per_core_batch_size,
    )


class TimesFMV4Forecaster(BaseRetailForecaster):
    """Zero-shot TimesFM 3.0 with optional origin-known future covariates."""

    def __init__(self, params: dict[str, Any]):
        super().__init__(name="TimesFM 3.0", family="foundation_model")
        self.params = params
        self.history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame, scenario: str) -> None:
        del scenario
        self.history = history.copy()
        timesfm_context(self.history)

    def predict(self, future: pd.DataFrame, scenario: str) -> ForecastOutput:
        if self.history is None:
            raise RuntimeError("TimesFM adapter must be fitted before prediction")
        context = timesfm_context(self.history)
        use_covariates = bool(self.params.get("use_covariates", False))
        covariates = (
            timesfm_covariate_matrix(self.history, future, scenario)
            if use_covariates
            else None
        )
        model = load_timesfm_model(
            checkpoint=str(self.params.get("checkpoint", TIMESFM_CHECKPOINT)),
            revision=str(self.params.get("revision", TIMESFM_REVISION)),
            device=str(self.params.get("device", TIMESFM_DEVICE)),
            per_core_batch_size=int(
                self.params.get("per_core_batch_size", TIMESFM_BATCH_SIZE)
            ),
        )
        started = time.perf_counter()
        result = model.predict(
            context=context,
            horizon=len(future),
            past_future_covariates=covariates,
            return_quantiles=True,
            sort_quantiles=True,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        point = np.asarray(result.forecast, dtype=float).reshape(-1)
        quantiles = np.asarray(result.quantiles, dtype=float)
        if quantiles.shape != (len(future), 9):
            raise ValueError(
                f"Unexpected TimesFM quantile shape {quantiles.shape}; expected "
                f"({len(future)}, 9)"
            )
        return ForecastOutput(
            point_forecast=sales_prediction(future, point),
            q10=sales_prediction(future, quantiles[:, 0]),
            q90=sales_prediction(future, quantiles[:, 8]),
            latency_ms=latency_ms,
            extra_details={
                "checkpoint": self.params.get("checkpoint", TIMESFM_CHECKPOINT),
                "revision": self.params.get("revision", TIMESFM_REVISION),
                "device": self.params.get("device", TIMESFM_DEVICE),
                "use_covariates": use_covariates,
                "context_length": len(context),
                "covariate_channels": 0 if covariates is None else covariates.shape[0],
            },
        )


def timesfm_factory(
    config: ModelConfig, scenario: str, params: dict[str, Any]
) -> BaseRetailForecaster:
    del config, scenario
    return TimesFMV4Forecaster(params)
