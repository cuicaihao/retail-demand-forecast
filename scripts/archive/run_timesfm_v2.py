from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retail_forecast.archive.timesfm_v2 import (
    compute_metrics,
    fit_predict_lightgbm,
    fit_predict_prophet,
    predict_seasonal_naive,
)
from retail_forecast.common import (
    HORIZON,
    datasets,
    sales_prediction,
    valid_labels,
)

DEFAULT_RUNS_DIR = ROOT / "runs" / "20261103-timesfm-validation"
DEFAULT_IMG_DIR = ROOT / "images" / "tsfm"


def run_benchmark(
    csv_path: Path,
    output_dir: Path,
    img_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading BigQuery TimesFM predictions from: {csv_path}")
    df_bq = pd.read_csv(csv_path, parse_dates=["ds"])

    # 1. 评估实验 1: 全历史基准 (Exp 1: Full History Benchmark)
    print("\nEvaluating Experiment 1 (Full History Benchmark)...")
    exp1_rows = []
    exp1_preds = {}

    for sc_name in ["grocery", "wine", "garden", "luxury"]:
        df_sc = datasets[sc_name]
        history = df_sc.iloc[:-HORIZON].copy()
        future = df_sc.iloc[-HORIZON:].copy()
        valid_fut = valid_labels(future)
        y_true = future.loc[valid_fut, "y"].values

        clean_hist = history.copy()
        if sc_name == "grocery":
            bad_mask = (~clean_hist["is_open"]) | clean_hist["stockout"]
            clean_hist.loc[bad_mask, "y"] = np.nan
            clean_hist["y"] = clean_hist["y"].interpolate().bfill().ffill()

        exp1_preds[sc_name] = {"history": history, "future": future, "preds": {}}

        # 本地基准模型
        p_naive = predict_seasonal_naive(history, future)
        p_prophet, q10_prophet, q90_prophet, lat_prophet = fit_predict_prophet(
            clean_hist, future, sc_name
        )
        p_lgb, lat_lgb = fit_predict_lightgbm(clean_hist, future, sc_name)

        exp1_preds[sc_name]["preds"]["Seasonal Naive"] = {
            "pred": p_naive,
            "q10": None,
            "q90": None,
        }
        exp1_preds[sc_name]["preds"]["Prophet"] = {
            "pred": p_prophet,
            "q10": q10_prophet,
            "q90": q90_prophet,
        }
        exp1_preds[sc_name]["preds"]["LightGBM"] = {
            "pred": p_lgb,
            "q10": None,
            "q90": None,
        }

        # BigQuery TimesFM 预测
        bq_sub = df_bq[
            (df_bq["scenario"] == sc_name)
            & (df_bq["experiment_id"] == "exp1_full_history")
        ].sort_values("ds")
        assert len(bq_sub) == HORIZON, (
            f"Expected {HORIZON} days for {sc_name}, got {len(bq_sub)}"
        )

        p_bq = sales_prediction(future, bq_sub["prediction"].values)
        q10_bq = sales_prediction(future, bq_sub["q10"].values)
        q90_bq = sales_prediction(future, bq_sub["q90"].values)
        exp1_preds[sc_name]["preds"]["TimesFM (BigQuery)"] = {
            "pred": p_bq,
            "q10": q10_bq,
            "q90": q90_bq,
        }

        for m_name, p_val, ql, qh, lat in [
            ("Seasonal Naive", p_naive, None, None, 5.0),
            (
                "Prophet (With Regressors)",
                p_prophet,
                q10_prophet,
                q90_prophet,
                lat_prophet,
            ),
            ("LightGBM", p_lgb, None, None, lat_lgb),
            ("TimesFM (BigQuery)", p_bq, q10_bq, q90_bq, 45.0),
        ]:
            met = compute_metrics(
                y_true,
                p_val[valid_fut],
                ql[valid_fut] if ql is not None else None,
                qh[valid_fut] if qh is not None else None,
            )
            exp1_rows.append(
                {
                    "Scenario": sc_name,
                    "Model": m_name,
                    "WAPE (%)": met["WAPE"] * 100,
                    "MAE": met["MAE"],
                    "RMSE": met["RMSE"],
                    "Bias (%)": met["Bias"] * 100,
                    "Coverage 80%": met["Coverage80"] * 100
                    if not np.isnan(met["Coverage80"])
                    else np.nan,
                    "Latency (ms)": lat,
                }
            )
            print(
                f"  [{sc_name:7s}] {m_name:32s} | WAPE: {met['WAPE'] * 100:6.2f}% | MAE: {met['MAE']:6.2f}"
            )

    df_exp1 = pd.DataFrame(exp1_rows)
    df_exp1.to_csv(output_dir / "exp1_full_history_benchmark.csv", index=False)

    # 2. 评估实验 2: 缺货删失污染实验 (Exp 2: Stockout Censoring Sensitivity)
    print("\nEvaluating Experiment 2 (Stockout Censoring Sensitivity)...")
    exp2_rows = []
    g_df = datasets["grocery"]
    g_future = g_df.iloc[-HORIZON:].copy()
    g_valid = valid_labels(g_future)
    y_g_true = g_future.loc[g_valid, "y"].values

    # Clean (Imputed) BQ TimesFM
    bq_clean = df_bq[
        (df_bq["scenario"] == "grocery")
        & (df_bq["experiment_id"] == "exp1_full_history")
    ].sort_values("ds")
    p_bq_clean = sales_prediction(g_future, bq_clean["prediction"].values)
    q10_bq_c = sales_prediction(g_future, bq_clean["q10"].values)
    q90_bq_c = sales_prediction(g_future, bq_clean["q90"].values)

    # Raw Censored BQ TimesFM
    bq_raw = df_bq[
        (df_bq["scenario"] == "grocery")
        & (df_bq["experiment_id"] == "exp2_raw_censored")
    ].sort_values("ds")
    p_bq_raw = sales_prediction(g_future, bq_raw["prediction"].values)
    q10_bq_r = sales_prediction(g_future, bq_raw["q10"].values)
    q90_bq_r = sales_prediction(g_future, bq_raw["q90"].values)

    for cond, (pred, ql, qh) in [
        ("Group A (Raw Censored Input)", (p_bq_raw, q10_bq_r, q90_bq_r)),
        ("Group B (Unbiased Imputed Input)", (p_bq_clean, q10_bq_c, q90_bq_c)),
    ]:
        met = compute_metrics(y_g_true, pred[g_valid], ql[g_valid], qh[g_valid])
        exp2_rows.append(
            {
                "Input Condition": cond,
                "Model": "TimesFM (BigQuery)",
                "WAPE (%)": met["WAPE"] * 100,
                "Bias (%)": met["Bias"] * 100,
                "MAE": met["MAE"],
                "RMSE": met["RMSE"],
                "Coverage 80%": met["Coverage80"] * 100,
            }
        )
        print(
            f"  {cond} | Bias: {met['Bias'] * 100:6.2f}% | WAPE: {met['WAPE'] * 100:6.2f}%"
        )

    df_exp2 = pd.DataFrame(exp2_rows)
    df_exp2.to_csv(output_dir / "exp2_stockout_censoring.csv", index=False)

    # 3. 评估实验 3: 冷启动截断曲线 (Exp 3: Cold-Start Truncation Curve)
    print("\nEvaluating Experiment 3 (Cold-Start Truncation Curve)...")
    exp3_rows = []
    lengths = [1068, 365, 90, 56, 28, 14]

    for sc_name in ["grocery", "wine", "garden"]:
        df_sc = datasets[sc_name]
        full_hist = df_sc.iloc[:-HORIZON].copy()
        future = df_sc.iloc[-HORIZON:].copy()
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

            # 获取 BQ TimesFM 在该长度下的预测
            exp_tag = "exp1_full_history" if L == 1068 else f"exp3_L{L}"
            bq_row = df_bq[
                (df_bq["scenario"] == sc_name) & (df_bq["experiment_id"] == exp_tag)
            ].sort_values("ds")
            p_bq_l = sales_prediction(future, bq_row["prediction"].values)

            for m_lbl, p_val in [
                ("Prophet", p_prophet),
                ("LightGBM", p_lgb),
                ("TimesFM (BigQuery)", p_bq_l),
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

    df_exp3 = pd.DataFrame(exp3_rows)
    df_exp3.to_csv(output_dir / "exp3_cold_start_curve.csv", index=False)

    # 4. 生成出版级图表
    print(f"\nGenerating figures in {img_dir}...")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    # 图 1: 四场景四宫格
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300)
    scenarios_meta = [
        ("grocery", "Grocery (High-Freq FMCG)", axes[0, 0]),
        ("wine", "Wine (Pulse Promotions)", axes[0, 1]),
        ("garden", "Garden Supplies (Strong Annual Seasonality)", axes[1, 0]),
        ("luxury", "Luxury Handbag (93.6% Zero Intermittent)", axes[1, 1]),
    ]
    for sc_name, title, ax in scenarios_meta:
        hist = exp1_preds[sc_name]["history"]
        fut = exp1_preds[sc_name]["future"]
        preds = exp1_preds[sc_name]["preds"]

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
            preds["Prophet"]["pred"],
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

        p_bq = preds["TimesFM (BigQuery)"]["pred"]
        q10_bq = preds["TimesFM (BigQuery)"]["q10"]
        q90_bq = preds["TimesFM (BigQuery)"]["q90"]
        ax.plot(
            fut["ds"], p_bq, "-", color="#d62728", lw=2.0, label="TimesFM (BigQuery)"
        )
        if q10_bq is not None and q90_bq is not None:
            ax.fill_between(
                fut["ds"],
                q10_bq,
                q90_bq,
                color="#d62728",
                alpha=0.15,
                label="TimesFM P10–P90",
            )

        ax.axvline(hist["ds"].iloc[-1], color="#888888", linestyle=":", lw=1.2)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        if ax == axes[0, 0]:
            ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(img_dir / "fig-tsfm-01-four-scenarios-comparison.png", dpi=300)
    fig.savefig(img_dir / "fig-tsfm-01-four-scenarios-comparison.webp", dpi=300)
    plt.close(fig)

    # 图 2: 冷启动退化曲线 (3 x 1 纵向堆叠)
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
            ("TimesFM (BigQuery)", "#d62728", "-", "o"),
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
    fig.savefig(img_dir / "fig-tsfm-02-coldstart-degradation-curve.png", dpi=300)
    fig.savefig(img_dir / "fig-tsfm-02-coldstart-degradation-curve.webp", dpi=300)
    plt.close(fig)

    # 图 3: 缺货删失污染图 (2 x 1 上下垂直结构)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), dpi=300)
    g_hist = datasets["grocery"].iloc[:-HORIZON]
    g_fut = datasets["grocery"].iloc[-HORIZON:]

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

    axes[1].plot(
        g_fut["ds"],
        g_fut["y"],
        "ko-",
        ms=3.5,
        label="Actual Observed (Test Ground Truth)",
    )
    axes[1].plot(
        g_fut["ds"],
        p_bq_raw,
        "--",
        color="#d62728",
        lw=1.8,
        label="TimesFM (Raw Censored Input)",
    )
    axes[1].fill_between(
        g_fut["ds"],
        q10_bq_r,
        q90_bq_r,
        color="#d62728",
        alpha=0.12,
        label="Raw Input P10–P90 Band",
    )
    axes[1].plot(
        g_fut["ds"],
        p_bq_clean,
        "-",
        color="#2ca02c",
        lw=2.0,
        label="TimesFM (Unbiased Imputed Input)",
    )
    axes[1].fill_between(
        g_fut["ds"],
        q10_bq_c,
        q90_bq_c,
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
    fig.savefig(img_dir / "fig-tsfm-03-stockout-censoring-impact.png", dpi=300)
    fig.savefig(img_dir / "fig-tsfm-03-stockout-censoring-impact.webp", dpi=300)
    plt.close(fig)

    print(
        f"\nAll benchmark evaluations completed successfully!\n"
        f"Metrics saved to: {output_dir}\n"
        f"Figures saved to: {img_dir}\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process BigQuery TimesFM predictions, evaluate benchmark metrics, and generate figures."
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_RUNS_DIR / "bq_timesfm_predictions.csv"),
        help="Path to BigQuery predictions CSV (default: runs/20261103-timesfm-validation/bq_timesfm_predictions.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RUNS_DIR),
        help="Directory to save benchmark metrics CSVs",
    )
    parser.add_argument(
        "--img-dir",
        default=str(DEFAULT_IMG_DIR),
        help="Directory to save generated figure plots",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Predictions CSV not found: {csv_path}. Please download it from BigQuery first."
        )

    run_benchmark(
        csv_path=csv_path,
        output_dir=Path(args.output_dir),
        img_dir=Path(args.img_dir),
    )


if __name__ == "__main__":
    main()
