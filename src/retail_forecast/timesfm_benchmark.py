from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

# =====================================================================
# 关键：在 macOS Apple Silicon 上，PyTorch / TimesFM 必须在 LightGBM (libomp)
# 之前加载，避免 OpenMP 动态库初始化顺序冲突引发的段错误 (Exit code 139)
# =====================================================================
import timesfm
from lightgbm import LGBMRegressor
from prophet import Prophet

from .common import (
    CPU_THREADS,
    SCENARIOS,
    SEED,
    make_features,
    sales_prediction,
    valid_labels,
    vic_holidays,
)

# =====================================================================
# 1. 经典基线与传统模型
# =====================================================================


def predict_seasonal_naive(history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """7天季节朴素基线 (向前顺延同星期有效值)"""
    h = history.loc[valid_labels(history), ["ds", "y"]].copy()
    h["weekday"] = h.ds.dt.dayofweek
    v = h.groupby("weekday").tail(1).set_index("weekday").y
    assert len(v) == 7
    raw = future.ds.dt.dayofweek.map(v).to_numpy(float)
    return sales_prediction(future, raw)


def fit_predict_prophet(
    history: pd.DataFrame, future: pd.DataFrame, scenario: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Prophet with VIC holidays & promo regressors (带 80% 预测区间)"""
    t0 = time.perf_counter()
    valid = valid_labels(history)
    m = Prophet(
        holidays=vic_holidays[["ds", "holiday"]],
        weekly_seasonality=3,
        yearly_seasonality=5 if len(history) >= 365 else False,
        daily_seasonality=False,
        interval_width=0.8,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )
    for col in SCENARIOS[scenario]["features"]:
        m.add_regressor(col, standardize=True)

    df_train = history.loc[valid, ["ds", "y", *SCENARIOS[scenario]["features"]]].copy()
    m.fit(df_train)

    df_future = future[["ds", *SCENARIOS[scenario]["features"]]].copy()
    fcst = m.predict(df_future)
    elapsed = (time.perf_counter() - t0) * 1000

    y_pred = sales_prediction(future, fcst["yhat"].to_numpy(float))
    q10 = sales_prediction(future, fcst["yhat_lower"].to_numpy(float))
    q90 = sales_prediction(future, fcst["yhat_upper"].to_numpy(float))
    return y_pred, q10, q90, elapsed


def fit_predict_lightgbm(
    history: pd.DataFrame,
    future: pd.DataFrame,
    scenario: str,
    objective: str = "regression",
) -> tuple[np.ndarray, float]:
    """LightGBM tabular baseline (或 Tweedie for luxury)"""
    t0 = time.perf_counter()
    y_hist = history["y"].where(valid_labels(history)).ffill().bfill().to_numpy(float)

    def extract_row(past_y: np.ndarray, fut_row: pd.DataFrame) -> dict:
        feats = make_features(fut_row, scenario).iloc[0].to_dict()
        feats["lag_1"] = past_y[-1]
        feats["lag_7"] = past_y[-7] if len(past_y) >= 7 else past_y[-1]
        feats["lag_14"] = past_y[-14] if len(past_y) >= 14 else past_y[-1]
        feats["rolling_mean_7"] = (
            past_y[-7:].mean() if len(past_y) >= 7 else past_y.mean()
        )
        feats["rolling_mean_28"] = (
            past_y[-28:].mean() if len(past_y) >= 28 else past_y.mean()
        )
        return feats

    rows_x, rows_y = [], []
    step = 7 if len(y_hist) > 300 else 1
    min_len = 28 if len(y_hist) >= 60 else 14
    for idx in range(min_len, len(history), step):
        if not valid_labels(history.iloc[[idx]]).values[0]:
            continue
        past_slice = y_hist[:idx]
        fut_slice = history.iloc[[idx]]
        rows_x.append(extract_row(past_slice, fut_slice))
        rows_y.append(history["y"].iloc[idx])

    if not rows_x:
        return np.zeros(len(future)), 0.0

    X_train = pd.DataFrame(rows_x)
    y_train = np.array(rows_y, dtype=float)

    params: dict[str, Any] = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "random_state": SEED,
        "n_jobs": CPU_THREADS,
        "verbosity": -1,
    }
    if objective == "tweedie":
        params.update({"objective": "tweedie", "tweedie_variance_power": 1.5})
    else:
        params.update({"objective": "regression_l1"})

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)

    curr_y = y_hist.copy()
    preds = []
    for step_i in range(len(future)):
        fut_row = future.iloc[[step_i]]
        row_feat = extract_row(curr_y, fut_row)
        pred_val = model.predict(pd.DataFrame([row_feat]))[0]
        preds.append(pred_val)
        curr_y = np.append(curr_y, pred_val)

    elapsed = (time.perf_counter() - t0) * 1000
    return sales_prediction(future, np.array(preds)), elapsed


def predict_croston_sba(
    history: pd.DataFrame, future: pd.DataFrame
) -> tuple[np.ndarray, float]:
    """Croston's Syntetos-Boylan Approximation (SBA) for intermittent demand"""
    t0 = time.perf_counter()
    y_valid = history.loc[valid_labels(history), "y"].to_numpy(float)
    non_zeros = np.where(y_valid > 0)[0]
    if len(non_zeros) < 2:
        return np.zeros(len(future)), 0.0

    intervals = np.diff(non_zeros)
    demands = y_valid[non_zeros[1:]]

    alpha = 0.1
    z_hat = float(demands[0])
    p_hat = float(intervals[0])
    for d, p in zip(demands[1:], intervals[1:]):
        z_hat += alpha * (d - z_hat)
        p_hat += alpha * (p - p_hat)

    forecast_rate = (z_hat / p_hat) * (1.0 - alpha / 2.0) if p_hat > 0 else 0.0
    elapsed = (time.perf_counter() - t0) * 1000
    raw = np.full(len(future), forecast_rate)
    return sales_prediction(future, raw), elapsed


# =====================================================================
# 2. 官方 TimesFM 2.5 与 3.0 受控预测引擎
# =====================================================================


class TimesFMBenchmarkEngine:
    """
    统一封装官方 Google TimesFM 2.5 (200M) 与 TimesFM 3.0 (330M)：
    - TimesFM 2.5: 支持原生 compile / forecast 及 forecast_with_covariates (xreg + timesfm)
    - TimesFM 3.0: 支持原生 Lookahead 协变量 (past_future_covariates) 端到端 Variate Attention 混合
    """

    def __init__(self):
        print("Initializing Official TimesFM 2.5 (200M)...", flush=True)
        self.tfm_25 = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
        cfg_25 = timesfm.ForecastConfig(
            max_context=1068, max_horizon=28, per_core_batch_size=1
        )
        self.tfm_25.compile(cfg_25)

        print("Initializing Official TimesFM 3.0 (330M)...", flush=True)
        self.tfm_30 = timesfm.TimesFM3Forecaster.from_pretrained(
            "google/timesfm-3.0-pytorch"
        )
        print("Both TimesFM 2.5 and 3.0 successfully initialized!", flush=True)

    def predict_25_univariate(
        self, history: pd.DataFrame, future: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """TimesFM 2.5 单变量零样本预测"""
        t0 = time.perf_counter()
        context_y = (
            history["y"]
            .where(valid_labels(history))
            .ffill()
            .bfill()
            .to_numpy(np.float32)
        )
        point, quantiles = self.tfm_25.forecast(horizon=len(future), inputs=[context_y])
        elapsed = (time.perf_counter() - t0) * 1000

        mean = point[0]
        # quantiles shape: (1, 28, 10), index 1 is P10, index 9 is P90
        q10 = quantiles[0, :, 1]
        q90 = quantiles[0, :, 9]
        return (
            sales_prediction(future, mean),
            sales_prediction(future, q10),
            sales_prediction(future, q90),
            elapsed,
        )

    def predict_25_xreg(
        self,
        history: pd.DataFrame,
        future: pd.DataFrame,
        scenario: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """TimesFM 2.5 + XReg (官方外挂 L2 岭回归正交解耦: xreg + timesfm)"""
        t0 = time.perf_counter()
        context_y = (
            history["y"]
            .where(valid_labels(history))
            .ffill()
            .bfill()
            .to_numpy(np.float32)
        )

        full_df = pd.concat([history, future], ignore_index=True)
        cov_df = make_features(full_df, scenario)

        # 构造 dynamic_numerical_covariates 字典
        dynamic_covariates = {
            col: [cov_df[col].to_numpy(np.float32)] for col in cov_df.columns
        }

        point, quantiles = self.tfm_25.forecast_with_covariates(
            inputs=[context_y],
            dynamic_numerical_covariates=dynamic_covariates,
            xreg_mode="xreg + timesfm",
            ridge=1.0,
            force_on_cpu=True,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        mean = point[0]
        q10 = quantiles[0, :, 1]
        q90 = quantiles[0, :, 9]
        return (
            sales_prediction(future, mean),
            sales_prediction(future, q10),
            sales_prediction(future, q90),
            elapsed,
        )

    def predict_30_univariate(
        self, history: pd.DataFrame, future: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """TimesFM 3.0 原生单变量零样本预测"""
        t0 = time.perf_counter()
        context_y = (
            history["y"]
            .where(valid_labels(history))
            .ffill()
            .bfill()
            .to_numpy(np.float32)
        )
        out = self.tfm_30.predict(
            context=context_y, horizon=len(future), return_quantiles=True
        )
        elapsed = (time.perf_counter() - t0) * 1000

        mean = out.forecast
        # quantiles shape: (28, 9), index 0 is P10, index 8 is P90
        q10 = out.quantiles[:, 0]
        q90 = out.quantiles[:, 8]
        return (
            sales_prediction(future, mean),
            sales_prediction(future, q10),
            sales_prediction(future, q90),
            elapsed,
        )

    def predict_30_native(
        self,
        history: pd.DataFrame,
        future: pd.DataFrame,
        scenario: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """TimesFM 3.0 原生 Lookahead 动态协变量注意力端到端混合"""
        t0 = time.perf_counter()
        context_y = (
            history["y"]
            .where(valid_labels(history))
            .ffill()
            .bfill()
            .to_numpy(np.float32)
        )

        full_df = pd.concat([history, future], ignore_index=True)
        cov_df = make_features(full_df, scenario)

        # 协变量矩阵: shape (total_len, num_features)
        cov_matrix = cov_df.to_numpy(np.float32)

        out = self.tfm_30.predict(
            context=context_y,
            horizon=len(future),
            past_future_covariates=cov_matrix,
            return_quantiles=True,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        mean = out.forecast
        q10 = out.quantiles[:, 0]
        q90 = out.quantiles[:, 8]
        return (
            sales_prediction(future, mean),
            sales_prediction(future, q10),
            sales_prediction(future, q90),
            elapsed,
        )


# =====================================================================
# 3. 统一指标评估工具
# =====================================================================


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    q_low: np.ndarray | pd.Series | None = None,
    q_high: np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    """
    计算与 retail-demand-forecast 基准一致的零售需求预测量化指标。
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
