from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 引入基础配置与基准模块
from retail_forecast.common import (
    DATA_DIR,
    HORIZON,
    SCENARIOS,
    datasets,
    metrics,
    valid_labels,
    windows,
)
from retail_forecast.timesfm_benchmark import (
    TimesFMBenchmarkEngine,
    fit_predict_lightgbm,
    fit_predict_prophet,
    predict_croston_sba,
    predict_seasonal_naive,
)
from tsfm_benchmark.evaluate_metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "images" / "tsfm"
IMG_DIR.mkdir(parents=True, exist_ok=True)
ARTICLES_DIR = ROOT / "articles"


def main():
    print("=" * 70)
    print("Starting TimesFM 2.5 / 3.0 Controlled Experiments on Apple Silicon M4")
    print("=" * 70)

    engine = TimesFMBenchmarkEngine()

    # -----------------------------------------------------------------
    # 实验 1: 四场景全历史 1,068 天基准对决 (Full-History Benchmark)
    # -----------------------------------------------------------------
    print("\n>>> Running Experiment 1: Full-History Benchmark (Context = 1,068d)...")
    exp1_rows = []
    exp1_predictions = {}

    for sc_name in ["grocery", "wine", "garden", "luxury"]:
        df = datasets[sc_name]
        history = df.iloc[:-HORIZON].copy()
        future = df.iloc[-HORIZON:].copy()
        valid_fut = valid_labels(future)
        y_true = future.loc[valid_fut, "y"].values

        # 针对 grocery 场景，去偏清洁输入
        clean_hist = history.copy()
        if sc_name == "grocery":
            bad_mask = (~clean_hist["is_open"]) | clean_hist["stockout"]
            clean_hist.loc[bad_mask, "y"] = np.nan
            clean_hist["y"] = clean_hist["y"].interpolate().bfill().ffill()

        models_to_run = [
            "Seasonal Naive",
            "Prophet (With Regressors)",
            "LightGBM",
            "TimesFM 2.5 (Univariate)",
            "TimesFM 2.5 + XReg",
            "TimesFM 3.0 (Univariate)",
            "TimesFM 3.0 (Native Covariates)",
        ]
        if sc_name == "luxury":
            models_to_run.extend(["Croston / SBA", "LightGBM (Tweedie)"])

        exp1_predictions[sc_name] = {"history": history, "future": future, "preds": {}}

        for m_name in models_to_run:
            t_start = time.perf_counter()
            q10, q90 = None, None

            if m_name == "Seasonal Naive":
                p = predict_seasonal_naive(history, future)
                elapsed = (time.perf_counter() - t_start) * 1000
            elif m_name == "Prophet (With Regressors)":
                p, q10, q90, elapsed = fit_predict_prophet(clean_hist, future, sc_name)
            elif m_name == "LightGBM":
                p, elapsed = fit_predict_lightgbm(
                    clean_hist, future, sc_name, objective="regression"
                )
            elif m_name == "LightGBM (Tweedie)":
                p, elapsed = fit_predict_lightgbm(
                    clean_hist, future, sc_name, objective="tweedie"
                )
            elif m_name == "Croston / SBA":
                p, elapsed = predict_croston_sba(history, future)
            elif m_name == "TimesFM 2.5 (Univariate)":
                p, q10, q90, elapsed = engine.predict_25_univariate(clean_hist, future)
            elif m_name == "TimesFM 2.5 + XReg":
                p, q10, q90, elapsed = engine.predict_25_xreg(
                    clean_hist, future, sc_name
                )
            elif m_name == "TimesFM 3.0 (Univariate)":
                p, q10, q90, elapsed = engine.predict_30_univariate(clean_hist, future)
            elif m_name == "TimesFM 3.0 (Native Covariates)":
                p, q10, q90, elapsed = engine.predict_30_native(
                    clean_hist, future, sc_name
                )

            exp1_predictions[sc_name]["preds"][m_name] = {
                "pred": p,
                "q10": q10,
                "q90": q90,
            }

            # 评估指标
            m_res = compute_metrics(
                y_true=y_true,
                y_pred=p[valid_fut],
                q_low=q10[valid_fut] if q10 is not None else None,
                q_high=q90[valid_fut] if q90 is not None else None,
            )
            exp1_rows.append(
                {
                    "Scenario": sc_name,
                    "Model": m_name,
                    "WAPE (%)": m_res["WAPE"] * 100,
                    "MAE": m_res["MAE"],
                    "RMSE": m_res["RMSE"],
                    "Bias (%)": m_res["Bias"] * 100,
                    "Coverage 80%": m_res["Coverage80"] * 100
                    if not np.isnan(m_res["Coverage80"])
                    else np.nan,
                    "Latency (ms)": elapsed,
                }
            )
            print(
                f"  [{sc_name:7s}] {m_name:32s} | WAPE: {m_res['WAPE'] * 100:6.2f}% | MAE: {m_res['MAE']:6.2f} | Latency: {elapsed:5.1f}ms"
            )

    df_exp1 = pd.DataFrame(exp1_rows)
    df_exp1.to_csv(ROOT / "runs" / "exp1_full_history_benchmark.csv", index=False)

    # -----------------------------------------------------------------
    # 实验 2: 缺货删失污染实验 (Stockout Censoring Sensitivity)
    # -----------------------------------------------------------------
    print("\n>>> Running Experiment 2: Stockout Censoring Sensitivity (Grocery)...")
    exp2_rows = []
    g_df = datasets["grocery"]
    g_hist = g_df.iloc[:-HORIZON].copy()
    g_future = g_df.iloc[-HORIZON:].copy()
    g_valid_fut = valid_labels(g_future)
    y_g_true = g_future.loc[g_valid_fut, "y"].values

    # Group A: 原始未去偏输入 (含24个截断至20的缺货日)
    raw_hist = g_hist.copy()

    # Group B: 修复插补去偏输入
    imputed_hist = g_hist.copy()
    bad_mask = (~imputed_hist["is_open"]) | imputed_hist["stockout"]
    imputed_hist.loc[bad_mask, "y"] = np.nan
    imputed_hist["y"] = imputed_hist["y"].interpolate().bfill().ffill()

    exp2_preds = {}
    for group_name, input_h in [
        ("Group A (Raw Censored)", raw_hist),
        ("Group B (Unbiased Imputed)", imputed_hist),
    ]:
        p_25_xreg, q10_25_x, q90_25_x, _ = engine.predict_25_xreg(
            input_h, g_future, "grocery"
        )
        p_30_nat, q10_30_n, q90_30_n, _ = engine.predict_30_native(
            input_h, g_future, "grocery"
        )

        exp2_preds[group_name] = {
            "25_xreg": (p_25_xreg, q10_25_x, q90_25_x),
            "30_native": (p_30_nat, q10_30_n, q90_30_n),
        }

        for m_lbl, (pred_arr, ql, qh) in [
            ("TimesFM 2.5 + XReg", (p_25_xreg, q10_25_x, q90_25_x)),
            ("TimesFM 3.0 (Native Covariates)", (p_30_nat, q10_30_n, q90_30_n)),
        ]:
            met = compute_metrics(
                y_g_true, pred_arr[g_valid_fut], ql[g_valid_fut], qh[g_valid_fut]
            )
            exp2_rows.append(
                {
                    "Input Condition": group_name,
                    "Model": m_lbl,
                    "WAPE (%)": met["WAPE"] * 100,
                    "Bias (%)": met["Bias"] * 100,
                    "MAE": met["MAE"],
                    "RMSE": met["RMSE"],
                    "Coverage 80%": met["Coverage80"] * 100,
                }
            )
            print(
                f"  {group_name} | {m_lbl:32s} | Bias: {met['Bias'] * 100:6.2f}% | WAPE: {met['WAPE'] * 100:6.2f}%"
            )

    df_exp2 = pd.DataFrame(exp2_rows)
    df_exp2.to_csv(ROOT / "runs" / "exp2_stockout_censoring.csv", index=False)

    # -----------------------------------------------------------------
    # 实验 3: 冷启动历史截断退化测试 (Cold-Start Truncation Curve)
    # -----------------------------------------------------------------
    print("\n>>> Running Experiment 3: Cold-Start Context Truncation Curve...")
    lengths = [1068, 365, 90, 56, 28, 14]
    exp3_rows = []

    for sc_name in ["grocery", "wine", "garden"]:
        df = datasets[sc_name]
        full_hist = df.iloc[:-HORIZON].copy()
        future = df.iloc[-HORIZON:].copy()
        valid_fut = valid_labels(future)
        y_true = future.loc[valid_fut, "y"].values

        for L in lengths:
            short_hist = full_hist.iloc[-L:].copy()
            if sc_name == "grocery":
                bad_m = (~short_hist["is_open"]) | short_hist["stockout"]
                short_hist.loc[bad_m, "y"] = np.nan
                short_hist["y"] = short_hist["y"].interpolate().bfill().ffill()

            p_prophet, _, _, _ = fit_predict_prophet(short_hist, future, sc_name)
            p_lgb, _ = fit_predict_lightgbm(short_hist, future, sc_name)
            p_25_xreg, _, _, _ = engine.predict_25_xreg(short_hist, future, sc_name)
            p_30_nat, _, _, _ = engine.predict_30_native(short_hist, future, sc_name)

            for m_lbl, p_val in [
                ("Prophet", p_prophet),
                ("LightGBM", p_lgb),
                ("TimesFM 2.5 + XReg", p_25_xreg),
                ("TimesFM 3.0 Native", p_30_nat),
            ]:
                score = compute_metrics(y_true, p_val[valid_fut])
                exp3_rows.append(
                    {
                        "Scenario": sc_name,
                        "Context Length (d)": L,
                        "Model": m_lbl,
                        "WAPE (%)": score["WAPE"] * 100,
                        "MAE": score["MAE"],
                    }
                )
        print(f"  Completed Cold-Start scans for [{sc_name}] across {lengths} days")

    df_exp3 = pd.DataFrame(exp3_rows)
    df_exp3.to_csv(ROOT / "runs" / "exp3_cold_start_curve.csv", index=False)

    # -----------------------------------------------------------------
    # 图表绘制 (按规范生成 300 DPI 出版级图表及 .webp)
    # -----------------------------------------------------------------
    print("\n>>> Generating Publication-Ready Figures in images/tsfm/...")

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    # 图 1: 四场景 28 天预测与区间对比 (2 x 2 四宫格)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300)
    scenarios_meta = [
        ("grocery", "Grocery (High-Freq FMCG)", axes[0, 0]),
        ("wine", "Wine (Pulse Promotions)", axes[0, 1]),
        ("garden", "Garden Supplies (Strong Annual Seasonality)", axes[1, 0]),
        ("luxury", "Luxury Handbag (93.6% Zero Intermittent)", axes[1, 1]),
    ]

    for sc_name, title, ax in scenarios_meta:
        hist = exp1_predictions[sc_name]["history"]
        fut = exp1_predictions[sc_name]["future"]
        preds = exp1_predictions[sc_name]["preds"]

        tail_hist = hist.iloc[-14:]
        ds_all = pd.concat([tail_hist["ds"], fut["ds"]])
        y_all = pd.concat([tail_hist["y"], fut["y"]]).values

        ax.plot(
            ds_all,
            y_all,
            "o-",
            color="#111111",
            ms=3.5,
            lw=1.2,
            label="Ground Truth (y)",
            zorder=5,
        )
        ax.plot(
            fut["ds"],
            preds["Prophet (With Regressors)"]["pred"],
            "--",
            color="#1f77b4",
            lw=1.5,
            label="Prophet",
        )
        ax.plot(
            fut["ds"],
            preds["LightGBM"]["pred"],
            "-.",
            color="#2ca02c",
            lw=1.5,
            label="LightGBM",
        )

        # TimesFM 2.5 + XReg
        p25 = preds["TimesFM 2.5 + XReg"]["pred"]
        ax.plot(
            fut["ds"], p25, "-", color="#ff7f0e", lw=1.6, label="TimesFM 2.5 + XReg"
        )

        # TimesFM 3.0 Native
        p30 = preds["TimesFM 3.0 (Native Covariates)"]["pred"]
        q10_30 = preds["TimesFM 3.0 (Native Covariates)"]["q10"]
        q90_30 = preds["TimesFM 3.0 (Native Covariates)"]["q90"]
        ax.plot(
            fut["ds"], p30, "-", color="#d62728", lw=2.0, label="TimesFM 3.0 Native"
        )
        if q10_30 is not None and q90_30 is not None:
            ax.fill_between(
                fut["ds"],
                q10_30,
                q90_30,
                color="#d62728",
                alpha=0.15,
                label="TimesFM 3.0 P10–P90",
            )

        ax.axvline(hist["ds"].iloc[-1], color="#888888", linestyle=":", lw=1.2)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        if ax == axes[0, 0]:
            ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig1_path = IMG_DIR / "fig-tsfm-01-four-scenarios-comparison.png"
    fig.savefig(fig1_path, dpi=300)
    fig.savefig(IMG_DIR / "fig-tsfm-01-four-scenarios-comparison.webp", dpi=300)
    plt.close(fig)
    print(f"  Saved {fig1_path}")

    # 图 2: 冷启动退化曲线 (3 x 1 纵向堆叠，严格适配单栏博客排版)
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True, dpi=300)
    plot_scs = [
        ("grocery", "Grocery Demand", axes[0]),
        ("wine", "Wine Demand", axes[1]),
        ("garden", "Garden Demand", axes[2]),
    ]

    for sc_name, title, ax in plot_scs:
        sub_df = df_exp3[df_exp3["Scenario"] == sc_name]
        for m_lbl, col, ls, marker in [
            ("Prophet", "#1f77b4", "--", "s"),
            ("LightGBM", "#2ca02c", "-.", "^"),
            ("TimesFM 2.5 + XReg", "#ff7f0e", "-", "o"),
            ("TimesFM 3.0 Native", "#d62728", "-", "D"),
        ]:
            m_data = sub_df[sub_df["Model"] == m_lbl].sort_values("Context Length (d)")
            ax.plot(
                m_data["Context Length (d)"],
                m_data["WAPE (%)"],
                linestyle=ls,
                color=col,
                marker=marker,
                lw=1.8,
                ms=5,
                label=m_lbl,
            )

        ax.set_title(f"Cold-Start Degradation: {title}", fontsize=11, fontweight="bold")
        ax.set_ylabel("28-day WAPE (%)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xscale("log")
        ax.set_xticks(lengths)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        if ax == axes[0]:
            ax.legend(fontsize=8.5, loc="upper right")

    axes[-1].set_xlabel("Context Length L (Days before test period)", fontsize=10)
    plt.tight_layout()
    fig2_path = IMG_DIR / "fig-tsfm-02-coldstart-degradation-curve.png"
    fig.savefig(fig2_path, dpi=300)
    fig.savefig(IMG_DIR / "fig-tsfm-02-coldstart-degradation-curve.webp", dpi=300)
    plt.close(fig)
    print(f"  Saved {fig2_path}")

    # 图 3: 缺货删失污染诊断图 (2 x 1 上下垂直对比，单栏博客友好)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), dpi=300)
    g_hist = datasets["grocery"].iloc[:-HORIZON]
    g_fut = datasets["grocery"].iloc[-HORIZON:]

    # 上层: 删失点与真实潜在需求对比 (展示最后 120 天历史)
    view_slice = g_hist.iloc[-120:]
    axes[0].plot(
        view_slice["ds"],
        view_slice["latent_demand"],
        color="#444444",
        lw=1.2,
        label="Unconstrained Latent Demand",
    )
    axes[0].plot(
        view_slice["ds"],
        view_slice["y"],
        color="#1f77b4",
        lw=1.4,
        label="Observed Sales (y)",
    )
    stockout_pts = view_slice[view_slice["stockout"]]
    axes[0].scatter(
        stockout_pts["ds"],
        stockout_pts["y"],
        color="#d62728",
        s=40,
        zorder=6,
        label="Stockout Censored (= 20 units)",
    )
    axes[0].set_title(
        "(A) Historical Censoring: Unconstrained Latent Demand vs. Censored Observations",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].set_ylabel("Units", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8.5, loc="upper left")

    # 下层: Raw vs. Imputed 在 TimesFM 3.0 上的预测带下移对比
    raw_p, raw_ql, raw_qh = exp2_preds["Group A (Raw Censored)"]["30_native"]
    imp_p, imp_ql, imp_qh = exp2_preds["Group B (Unbiased Imputed)"]["30_native"]

    axes[1].plot(
        g_fut["ds"],
        g_fut["y"],
        "ko-",
        ms=3.5,
        label="Actual Observed (Test Ground Truth)",
    )
    axes[1].plot(
        g_fut["ds"],
        raw_p,
        "--",
        color="#d62728",
        lw=1.8,
        label="TimesFM 3.0 (Raw Censored Input)",
    )
    axes[1].fill_between(
        g_fut["ds"],
        raw_ql,
        raw_qh,
        color="#d62728",
        alpha=0.12,
        label="Raw Input P10–P90 Band",
    )

    axes[1].plot(
        g_fut["ds"],
        imp_p,
        "-",
        color="#2ca02c",
        lw=2.0,
        label="TimesFM 3.0 (Unbiased Imputed Input)",
    )
    axes[1].fill_between(
        g_fut["ds"],
        imp_ql,
        imp_qh,
        color="#2ca02c",
        alpha=0.15,
        label="Imputed Input P10–P90 Band",
    )

    axes[1].set_title(
        "(B) Forecast Bias: Systemic Downward Shift Caused by Censored History",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].set_ylabel("Predicted Units", fontsize=10)
    axes[1].set_xlabel("Forecast Horizon (December 2025)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8.5, loc="upper left")

    plt.tight_layout()
    fig3_path = IMG_DIR / "fig-tsfm-03-stockout-censoring-impact.png"
    fig.savefig(fig3_path, dpi=300)
    fig.savefig(IMG_DIR / "fig-tsfm-03-stockout-censoring-impact.webp", dpi=300)
    plt.close(fig)
    print(f"  Saved {fig3_path}")

    print("\nAll experiments and figure generation completed successfully!")


if __name__ == "__main__":
    main()
