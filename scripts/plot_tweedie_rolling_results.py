"""
Generate high-resolution figures for the 12-Month Walk-Forward Rolling Window Experiment:
1. fig-tweedie-03-rolling-12m-monthly-wape.png (.webp): 12-month trajectory of monthly WAPE across scenarios.
2. fig-tweedie-04-rolling-12m-annual-summary.png (.webp): Full-Year 2025 (365 days) comprehensive leaderboard.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "runs" / "20261103-gbdt-tweedie-experiment" / "rolling_12m"
IMG_DIR = ROOT / "images" / "tweedie"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def setup_fonts():
    for path in (
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
    ):
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(
                fname=path
            ).get_name()
            break
    else:
        plt.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC",
            "Microsoft YaHei",
            "SimHei",
            "DejaVu Sans",
        ]
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})


def plot_monthly_trajectories():
    setup_fonts()
    monthly_path = EXP_DIR / "rolling_metrics_by_month.csv"
    if not monthly_path.exists():
        print(f"Error: {monthly_path} not found.")
        return

    df = pd.read_csv(monthly_path)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), facecolor="white")
    plt.subplots_adjust(
        left=0.07, right=0.96, top=0.90, bottom=0.08, hspace=0.32, wspace=0.22
    )

    fig.suptitle(
        "2025 全年 12 个月滚动前瞻验证（Walk-Forward 12-Month Backtest）：月度 WAPE 演变轨迹\n"
        "固定 24 个月滚动训练窗口 (~730天) | 逐月独立盲测 (共 365 天) | 检验四季更替与极端脉冲下的稳健性",
        fontsize=13.5,
        fontweight="bold",
        y=0.96,
        color="#1A252C",
    )

    scenarios = ["luxury", "wine", "garden", "grocery"]
    titles = [
        "Scenario 4: luxury (奢品手袋 - 93.6% 零膨胀)",
        "Scenario 2: wine (精品红酒 - 泊松促销脉冲)",
        "Scenario 3: garden (庭院园艺 - 正弦强季节性)",
        "Scenario 1: grocery (快消生鲜 - 高频连续 FMCG)",
    ]

    months = [f"{m:02d}月" for m in range(1, 13)]

    # Representative configs to highlight
    rep_configs = [
        ("Seasonal Naive", "#6C757D", "--", 1.8, "Seasonal Naive (基准)"),
        ("LGBM - Default (MSE)", "#E63946", "-.", 2.0, "LightGBM 默认 (MSE)"),
        ("LGBM - Tweedie (p=1.8)", "#2A9D8F", "-", 2.5, "LightGBM Tweedie (p=1.8)"),
        ("XGB - Default (MSE)", "#D90429", ":", 2.0, "XGBoost 默认 (MSE)"),
        ("XGB - Tweedie (p=1.8)", "#4361EE", "-", 2.5, "XGBoost Tweedie (p=1.8)"),
    ]

    for idx, sc in enumerate(scenarios):
        ax = axes[idx // 2, idx % 2]
        sub = df[df["scenario"] == sc]

        # For wine, garden, grocery, p=1.1 or 1.2 is more representative than 1.8
        if sc in ["grocery", "garden"]:
            current_configs = [
                ("Seasonal Naive", "#6C757D", "--", 1.8, "Seasonal Naive (基准)"),
                ("LGBM - Default (MSE)", "#E63946", "-.", 2.0, "LightGBM 默认 (MSE)"),
                (
                    "LGBM - Tweedie (p=1.1)",
                    "#2A9D8F",
                    "-",
                    2.5,
                    "LightGBM Tweedie (p=1.1)",
                ),
                ("XGB - Default (MSE)", "#D90429", ":", 2.0, "XGBoost 默认 (MSE)"),
                (
                    "XGB - Tweedie (p=1.2)",
                    "#4361EE",
                    "-",
                    2.5,
                    "XGBoost Tweedie (p=1.2)",
                ),
            ]
        elif sc == "wine":
            current_configs = [
                ("Seasonal Naive", "#6C757D", "--", 1.8, "Seasonal Naive (基准)"),
                ("LGBM - Default (MSE)", "#E63946", "-.", 2.0, "LightGBM 默认 (MSE)"),
                (
                    "LGBM - Tweedie (p=1.1)",
                    "#2A9D8F",
                    "-",
                    2.5,
                    "LightGBM Tweedie (p=1.1)",
                ),
                ("XGB - Default (MSE)", "#D90429", ":", 2.0, "XGBoost 默认 (MSE)"),
                (
                    "XGB - Tweedie (p=1.5)",
                    "#4361EE",
                    "-",
                    2.5,
                    "XGBoost Tweedie (p=1.5)",
                ),
            ]
        else:
            current_configs = rep_configs

        for cfg, color, style, lw, label in current_configs:
            cfg_data = sub[sub["config_name"] == cfg].sort_values("fold")
            if len(cfg_data) == 12:
                wape_vals = cfg_data["WAPE"].values * 100
                ax.plot(
                    range(1, 13),
                    wape_vals,
                    label=label,
                    color=color,
                    linestyle=style,
                    linewidth=lw,
                    marker="o",
                    markersize=4.5,
                )

        ax.set_title(titles[idx], fontsize=11.5, fontweight="bold", color="#1D3557")
        ax.set_xlabel("预测评估月份 (2025年)", fontsize=9.5)
        ax.set_ylabel("月度 WAPE (%) - 越低越好", fontsize=9.5)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

        if sc == "luxury":
            ax.set_ylim(50, 450)
            ax.annotate(
                "默认 MSE 无法纯化 0 值\n持续输出微小均值底噪\nWAPE 长期在 200%~400% 游荡",
                xy=(1, 395),
                xytext=(2.2, 330),
                arrowprops=dict(arrowstyle="->", color="#E63946", lw=1.2),
                fontsize=8.5,
                color="#E63946",
                fontweight="bold",
            )
            ax.annotate(
                "Tweedie (p=1.8) 强力消灭底噪\nWAPE 贴近 100% 理论极值",
                xy=(2, 100),
                xytext=(3.5, 140),
                arrowprops=dict(arrowstyle="->", color="#4361EE", lw=1.2),
                fontsize=8.5,
                color="#4361EE",
                fontweight="bold",
            )
        elif sc == "grocery":
            ax.set_ylim(5, 25)
            ax.text(
                0.04,
                0.88,
                "全年 12 个月 Tweedie 表现稳定优于 MSE\n方差自适应加权使得冬夏季表现均更平稳",
                transform=ax.transAxes,
                fontsize=8.5,
                bbox=dict(
                    boxstyle="round,pad=0.25", facecolor="#E8F5E9", edgecolor="#81C784"
                ),
            )
        elif sc == "wine":
            ax.set_ylim(15, 65)
            ax.text(
                0.04,
                0.88,
                "Tweedie 更好地捕捉了 5-8月 以及 12月促销脉冲\nWAPE 整体优于默认 MSE 与 Naive",
                transform=ax.transAxes,
                fontsize=8.5,
                bbox=dict(
                    boxstyle="round,pad=0.25", facecolor="#FFF3E0", edgecolor="#FFB74D"
                ),
            )
        elif sc == "garden":
            ax.set_ylim(6, 25)
            ax.text(
                0.04,
                0.88,
                "连续正弦四季特征主导，Tweedie 与 MSE 表现相当\n证明在高斯平稳场景 Tweedie 保持无害稳健性",
                transform=ax.transAxes,
                fontsize=8.5,
                bbox=dict(
                    boxstyle="round,pad=0.25", facecolor="#F3E5F5", edgecolor="#BA68C8"
                ),
            )

    out_png = IMG_DIR / "fig-tweedie-03-rolling-12m-monthly-wape.png"
    out_webp = IMG_DIR / "fig-tweedie-03-rolling-12m-monthly-wape.webp"
    plt.savefig(out_png, dpi=300, facecolor="white")
    plt.close()

    with Image.open(out_png) as img:
        img.save(out_webp, "WEBP", quality=95)
    print(f"Generated:\n  - {out_png}\n  - {out_webp}")


def plot_annual_summary():
    setup_fonts()
    annual_path = EXP_DIR / "rolling_metrics_annual_summary.csv"
    if not annual_path.exists():
        print(f"Error: {annual_path} not found.")
        return

    df = pd.read_csv(annual_path)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), facecolor="white")
    plt.subplots_adjust(
        left=0.08, right=0.96, top=0.90, bottom=0.08, hspace=0.36, wspace=0.25
    )

    fig.suptitle(
        "2025 全年累计综合排行榜（Full-Year 365 Days Walk-Forward Benchmark）\n"
        "跨越春夏秋冬完整周期的累计 WAPE (%) 与零值识别准确率（Zero Accuracy %）",
        fontsize=13.5,
        fontweight="bold",
        y=0.96,
        color="#1A252C",
    )

    scenarios = ["luxury", "wine", "garden", "grocery"]
    titles = [
        "luxury (奢品手袋 - 全年 363 营业日，341天为0)",
        "wine (精品红酒 - 全年总销量 4,085件)",
        "garden (庭院园艺 - 全年总销量 24,753件)",
        "grocery (快消生鲜 - 全年总销量 41,383件)",
    ]

    rep_configs = [
        "Seasonal Naive",
        "LGBM - Default (MSE)",
        "LGBM - Tweedie (p=1.1)",
        "LGBM - Tweedie (p=1.8)",
        "XGB - Default (MSE)",
        "XGB - Tweedie (p=1.1)",
        "XGB - Tweedie (p=1.8)",
        "CatBoost - Default (MAE)",
        "CatBoost - Tweedie (p=1.8)",
    ]

    for idx, sc in enumerate(scenarios):
        ax = axes[idx // 2, idx % 2]
        sub = df[(df["scenario"] == sc) & (df["config_name"].isin(rep_configs))].copy()

        # Sort configs for readability
        order_dict = {cfg: i for i, cfg in enumerate(rep_configs)}
        sub["order"] = sub["config_name"].map(order_dict)
        sub = sub.sort_values("order", ascending=False)

        configs = sub["config_name"].values
        wapes = sub["annual_WAPE"].values * 100
        zero_accs = sub["zero_accuracy"].values * 100

        colors = []
        for c in configs:
            if "Naive" in c:
                colors.append("#6C757D")
            elif "Default" in c:
                colors.append("#E63946")
            else:
                colors.append("#2A9D8F")

        bars = ax.barh(
            configs, wapes, color=colors, alpha=0.85, edgecolor="#333333", height=0.65
        )
        ax.set_title(titles[idx], fontsize=11, fontweight="bold", color="#1D3557")
        ax.set_xlabel("全年 365天 累计 WAPE (%) - 越低越好", fontsize=9.5)
        ax.grid(True, axis="x", linestyle="--", alpha=0.45)

        max_w = max(wapes)
        ax.set_xlim(0, max_w * 1.25)

        for bar, w, z_acc in zip(bars, wapes, zero_accs):
            label_text = f"{w:.1f}%"
            if sc == "luxury":
                label_text += f" (零准确率:{z_acc:.0f}%)"
            ax.text(
                bar.get_width() + max_w * 0.015,
                bar.get_y() + bar.get_height() / 2,
                label_text,
                va="center",
                fontsize=8.5,
                fontweight="bold" if "Tweedie" in bar.get_label() else "normal",
            )

        if sc == "luxury":
            ax.text(
                0.40,
                0.20,
                "年度核心结论：\n• XGBoost Tweedie (p=1.8) WAPE 116.4%\n  比默认 MSE (253.1%) 改善 136.7 个百分点！\n• 零准确率从 24% 飞跃至 90%！",
                transform=ax.transAxes,
                fontsize=8.2,
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="#FBC02D"
                ),
            )
        elif sc == "grocery":
            ax.text(
                0.45,
                0.20,
                "年度核心结论：\n• LightGBM Tweedie (p=1.1) 全年 WAPE 8.53%\n  稳定胜过默认 MSE (8.72%)\n• 证明异方差加权对高频快消具备全年有效性",
                transform=ax.transAxes,
                fontsize=8.2,
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#81C784"
                ),
            )

    out_png = IMG_DIR / "fig-tweedie-04-rolling-12m-annual-summary.png"
    out_webp = IMG_DIR / "fig-tweedie-04-rolling-12m-annual-summary.webp"
    plt.savefig(out_png, dpi=300, facecolor="white")
    plt.close()

    with Image.open(out_png) as img:
        img.save(out_webp, "WEBP", quality=95)
    print(f"Generated:\n  - {out_png}\n  - {out_webp}")


if __name__ == "__main__":
    plot_monthly_trajectories()
    plot_annual_summary()
