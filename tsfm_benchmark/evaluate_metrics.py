from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q_low: np.ndarray | None = None,
    q_high: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute demand forecasting metrics aligned with retail-demand-forecast benchmarks.
    y_true: 1D array of observed ground truth
    y_pred: 1D array of point forecast (e.g., mean or median)
    q_low: 1D array of 10th percentile (P10)
    q_high: 1D array of 90th percentile (P90)
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    assert len(y) == len(p), f"Length mismatch: {len(y)} vs {len(p)}"

    n = len(y)
    if n == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "WAPE": np.nan,
            "Bias": np.nan,
            "Coverage80": np.nan,
            "MeanWidth80": np.nan,
            "n": 0,
        }

    err = p - y
    abs_err = np.abs(err)
    total_sales = float(y.sum())

    mae = float(abs_err.mean())
    rmse = float(np.sqrt((err**2).mean()))
    wape = float(abs_err.sum() / total_sales) if total_sales > 0 else np.nan
    bias = float(err.sum() / total_sales) if total_sales > 0 else np.nan

    coverage = np.nan
    mean_width = np.nan
    if q_low is not None and q_high is not None:
        ql = np.asarray(q_low, dtype=float).reshape(-1)
        qh = np.asarray(q_high, dtype=float).reshape(-1)
        assert len(ql) == n and len(qh) == n
        in_interval = (y >= ql) & (y <= qh)
        coverage = float(in_interval.mean())
        mean_width = float((qh - ql).mean())

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
        "Bias": bias,
        "Coverage80": coverage,
        "MeanWidth80": mean_width,
        "n": n,
    }
