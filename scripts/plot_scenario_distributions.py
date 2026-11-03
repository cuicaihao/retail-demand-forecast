"""
Generate high-resolution comparison chart showing probability distributions
of daily sales records vs. clean/latent demand across the 4 retail scenarios,
illustrating the impact of store closures and stockout censoring.
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
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


def plot_scenario_distributions():
    setup_fonts()

    # Load data
    scenarios = ["grocery", "wine", "garden", "luxury"]
    dfs = {
        s: pd.read_csv(ROOT / "data" / "teaching" / f"{s}_scenario_data.csv")
        for s in scenarios
    }

    fig, axes = plt.subplots(2, 4, figsize=(16.5, 10.2), facecolor="white")
    plt.subplots_adjust(
        left=0.06, right=0.97, top=0.88, bottom=0.08, hspace=0.36, wspace=0.28
    )

    fig.suptitle(
        "四类零售场景的日销售与真实需求分布对比：闭店与缺货截断对分布形态的影响\n"
        "Upper Panel: 原始观测销售记录 $y$（包含闭店归零与缺货截断伪峰） | Lower Panel: 消除闭店与缺货截断后的真实营业需求分布（Clean / Latent Demand）",
        fontsize=13.5,
        fontweight="bold",
        y=0.96,
        color="#1A252C",
    )

    # Column Titles
    titles = [
        "Scenario 1: grocery (快消生鲜)\n高频连续型需求",
        "Scenario 2: wine (精品红酒)\n低频泊松脉冲型需求",
        "Scenario 3: garden (庭院园艺)\n强季节性平稳型需求",
        "Scenario 4: luxury (奢品配饰)\n极度间歇性零膨胀需求",
    ]

    # Row 1: Raw Sales Distribution
    # -------------------------------------------------------------
    # 1.1 Grocery Raw
    # -------------------------------------------------------------
    ax = axes[0, 0]
    df = dfs["grocery"]
    # Separate closed (y=0), stockout (y=20), and normal
    normal_y = df[(df["is_open"]) & (~df["stockout"])]["y"]
    stockout_y = df[df["stockout"]]["y"]
    closed_y = df[~df["is_open"]]["y"]

    bins = np.linspace(0, 200, 41)
    ax.hist(
        normal_y,
        bins=bins,
        color="#2A6F97",
        alpha=0.85,
        label=f"正常营业销售 ({len(normal_y)}天)",
    )
    ax.hist(
        [0],
        bins=bins,
        weights=[len(closed_y)],
        color="#E63946",
        label=f"法定闭店日 ({len(closed_y)}天, y=0)",
    )
    ax.hist(
        [20],
        bins=bins,
        weights=[len(stockout_y)],
        color="#E76F51",
        label=f"缺货人工截断 ({len(stockout_y)}天, y=20)",
    )

    ax.set_title(titles[0], fontsize=11, fontweight="bold", color="#1D3557", pad=10)
    ax.set_xlabel("日销售量 $y$", fontsize=9.5)
    ax.set_ylabel("频数 (天数)", fontsize=9.5)
    ax.set_xlim(-5, 205)
    ax.set_ylim(0, 160)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.2, framealpha=0.9)

    ax.annotate(
        "闭店(6天)",
        xy=(0, 6),
        xytext=(15, 35),
        arrowprops=dict(arrowstyle="->", color="#E63946", lw=1.2),
        fontsize=8.5,
        fontweight="bold",
        color="#E63946",
    )
    ax.annotate(
        "缺货上限(24天)",
        xy=(20, 24),
        xytext=(35, 65),
        arrowprops=dict(arrowstyle="->", color="#E76F51", lw=1.2),
        fontsize=8.5,
        fontweight="bold",
        color="#E76F51",
    )
    ax.text(
        0.05,
        0.92,
        "Row 1: 原始观测值 (含人为失真)",
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color="#6C757D",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8F9FA", edgecolor="#CED4DA"),
    )

    # -------------------------------------------------------------
    # 1.2 Wine Raw
    # -------------------------------------------------------------
    ax = axes[0, 1]
    df = dfs["wine"]
    closed_y = df[~df["is_open"]]["y"]
    open_zeros = df[(df["is_open"]) & (df["y"] == 0)]["y"]
    open_pos = df[(df["is_open"]) & (df["y"] > 0)]["y"]

    bins = np.arange(0, 42, 1) - 0.5
    ax.hist(
        open_pos,
        bins=bins,
        color="#457B9D",
        alpha=0.85,
        label=f"营业正销量 ({len(open_pos)}天)",
    )
    ax.bar(
        [0],
        [len(closed_y)],
        width=0.8,
        color="#E63946",
        label=f"闭店归零 ({len(closed_y)}天)",
    )
    ax.bar(
        [0],
        [len(open_zeros)],
        bottom=[len(closed_y)],
        width=0.8,
        color="#F4A261",
        label=f"营业自然无销售 ({len(open_zeros)}天)",
    )

    ax.set_title(titles[1], fontsize=11, fontweight="bold", color="#1D3557", pad=10)
    ax.set_xlabel("日销售量 $y$", fontsize=9.5)
    ax.set_ylabel("频数 (天数)", fontsize=9.5)
    ax.set_xlim(-1, 40)
    ax.set_ylim(0, 140)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.2, framealpha=0.9)

    ax.annotate(
        "0销售合计 20天\n(闭店6 + 自然14)",
        xy=(0, 20),
        xytext=(4, 50),
        arrowprops=dict(arrowstyle="->", color="#D95F02", lw=1.2),
        fontsize=8.5,
        fontweight="bold",
        color="#D95F02",
    )

    # -------------------------------------------------------------
    # 1.3 Garden Raw
    # -------------------------------------------------------------
    ax = axes[0, 2]
    df = dfs["garden"]
    closed_y = df[~df["is_open"]]["y"]
    open_y = df[df["is_open"]]["y"]

    bins = np.linspace(0, 170, 35)
    ax.hist(
        open_y,
        bins=bins,
        color="#3D5A80",
        alpha=0.85,
        label=f"正常营业销售 ({len(open_y)}天)",
    )
    ax.hist(
        [0],
        bins=bins,
        weights=[len(closed_y)],
        color="#E63946",
        label=f"闭店归零 ({len(closed_y)}天)",
    )

    ax.set_title(titles[2], fontsize=11, fontweight="bold", color="#1D3557", pad=10)
    ax.set_xlabel("日销售量 $y$", fontsize=9.5)
    ax.set_ylabel("频数 (天数)", fontsize=9.5)
    ax.set_xlim(-5, 175)
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.2, framealpha=0.9)

    ax.annotate(
        "闭店(6天)",
        xy=(0, 6),
        xytext=(15, 30),
        arrowprops=dict(arrowstyle="->", color="#E63946", lw=1.2),
        fontsize=8.5,
        fontweight="bold",
        color="#E63946",
    )

    # -------------------------------------------------------------
    # 1.4 Luxury Raw
    # -------------------------------------------------------------
    ax = axes[0, 3]
    df = dfs["luxury"]
    closed_y = df[~df["is_open"]]["y"]
    open_zeros = df[(df["is_open"]) & (df["y"] == 0)]["y"]
    open_pos = df[(df["is_open"]) & (df["y"] > 0)]["y"]

    # Bar chart for discrete values 0, 1, 2, 3...
    unique_vals, counts = np.unique(open_pos, return_counts=True)
    ax.bar(
        [0],
        [len(closed_y)],
        color="#E63946",
        width=0.6,
        label=f"闭店归零 ({len(closed_y)}天)",
    )
    ax.bar(
        [0],
        [len(open_zeros)],
        bottom=[len(closed_y)],
        color="#7209B7",
        width=0.6,
        label=f"营业零销售 ({len(open_zeros)}天, 93.6%)",
    )
    ax.bar(
        unique_vals,
        counts,
        color="#4361EE",
        width=0.6,
        label=f"实际成交日 ({len(open_pos)}天, 6.4%)",
    )

    ax.set_title(titles[3], fontsize=11, fontweight="bold", color="#1D3557", pad=10)
    ax.set_xlabel("日销售量 $y$", fontsize=9.5)
    ax.set_ylabel("频数 (天数)", fontsize=9.5)
    ax.set_xlim(-1, 10.5)
    ax.set_ylim(0, 1150)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.2, framealpha=0.9)

    ax.text(
        0,
        1040,
        f"0 销售合计\n1,026 天 (93.6%)",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#7209B7",
    )

    # Row 2: Clean / Latent Demand Distribution
    # -------------------------------------------------------------
    # 2.1 Grocery Clean
    # -------------------------------------------------------------
    ax = axes[1, 0]
    df = dfs["grocery"]
    open_df = df[df["is_open"]]
    latent_clean = open_df["latent_demand"]

    bins = np.linspace(60, 200, 36)
    ax.hist(
        latent_clean,
        bins=bins,
        color="#2A9D8F",
        alpha=0.85,
        edgecolor="#21867A",
        label="真实潜在需求 (Uncensored)",
    )

    ax.set_title(
        "grocery: 消除闭店与缺货截断后的真实需求",
        fontsize=10.5,
        fontweight="bold",
        color="#1D3557",
        pad=8,
    )
    ax.set_xlabel("真实日需求量 $y^*$", fontsize=9.5)
    ax.set_ylabel("频数 (天数)", fontsize=9.5)
    ax.set_xlim(55, 205)
    ax.set_ylim(0, 150)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.2, framealpha=0.9)

    ax.text(
        0.05,
        0.90,
        "Row 2: 真实有效需求 (去除外生干扰)",
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color="#2A9D8F",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#E8F5E9", edgecolor="#81C784"),
    )
    stat_str = (
        "营业天数: 1,090 天\n"
        "真实零销售率: 0.00% (0天)\n"
        "均值 $\\mu = 111.59$, 标准差 $\\sigma = 13.82$\n"
        "形态: 对称高斯分布 (同方差)"
    )
    ax.text(
        0.05,
        0.58,
        stat_str,
        transform=ax.transAxes,
        fontsize=8.2,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CED4DA", alpha=0.95
        ),
    )
    ax.text(
        0.05,
        0.08,
        "结论: 零点由闭店引起，非零膨胀；\n缺货截断掩盖了高斯均值，标准回归适配。",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#1B4332",
        style="italic",
    )

    # -------------------------------------------------------------
    # 2.2 Wine Clean
    # -------------------------------------------------------------
    ax = axes[1, 1]
    df = dfs["wine"]
    open_df = df[df["is_open"]]
    open_y = open_df["y"]

    bins = np.arange(0, 42, 1) - 0.5
    ax.hist(
        open_y,
        bins=bins,
        color="#264653",
        alpha=0.85,
        edgecolor="#1D3557",
        label="营业日销售 (Open Days)",
    )

    ax.set_title(
        "wine: 消除闭店后的真实营业需求",
        fontsize=10.5,
        fontweight="bold",
        color="#1D3557",
        pad=8,
    )
    ax.set_xlabel("真实日需求量 $y$", fontsize=9.5)
    ax.set_ylabel("频数 (天数)", fontsize=9.5)
    ax.set_xlim(-1, 40)
    ax.set_ylim(0, 140)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.2, framealpha=0.9)

    stat_str = (
        "营业天数: 1,090 天\n"
        "真实零销售率: 1.28% (仅14天)\n"
        "均值 $\\mu = 10.38$, 方差 $\\sigma^2 = 18.23$\n"
        "形态: 泊松脉冲计数 (轻微过度分散)"
    )
    ax.text(
        0.42,
        0.66,
        stat_str,
        transform=ax.transAxes,
        fontsize=8.2,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CED4DA", alpha=0.95
        ),
    )
    ax.text(
        0.05,
        0.08,
        "结论: 零值仅占 1.28%，无显著零膨胀；\n长尾为促销脉冲，适配 Poisson / Tweedie(p≈1.1)。",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#1B4332",
        style="italic",
    )

    # -------------------------------------------------------------
    # 2.3 Garden Clean
    # -------------------------------------------------------------
    ax = axes[1, 2]
    df = dfs["garden"]
    open_df = df[df["is_open"]]
    open_y = open_df["y"]

    bins = np.linspace(0, 170, 35)
    ax.hist(
        open_y,
        bins=bins,
        color="#3A5A40",
        alpha=0.85,
        edgecolor="#283D2C",
        label="营业日销售 (Open Days)",
    )

    ax.set_title(
        "garden: 消除闭店后的真实营业需求",
        fontsize=10.5,
        fontweight="bold",
        color="#1D3557",
        pad=8,
    )
    ax.set_xlabel("真实日需求量 $y$", fontsize=9.5)
    ax.set_ylabel("频数 (天数)", fontsize=9.5)
    ax.set_xlim(-5, 175)
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.2, framealpha=0.9)

    stat_str = (
        "营业天数: 1,090 天\n"
        "真实零销售率: 0.00% (严格0天)\n"
        "均值 $\\mu = 64.92$, 标准差 $\\sigma = 22.84$\n"
        "形态: 宽幅连续周期分布"
    )
    ax.text(
        0.42,
        0.66,
        stat_str,
        transform=ax.transAxes,
        fontsize=8.2,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CED4DA", alpha=0.95
        ),
    )
    ax.text(
        0.05,
        0.08,
        "结论: 营业日零销售率为零；\n宽分布由夏冬正弦季节性驱动，标准回归完胜。",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#1B4332",
        style="italic",
    )

    # -------------------------------------------------------------
    # 2.4 Luxury Clean
    # -------------------------------------------------------------
    ax = axes[1, 3]
    df = dfs["luxury"]
    open_df = df[df["is_open"]]
    open_zeros = open_df[open_df["y"] == 0]["y"]
    open_pos = open_df[open_df["y"] > 0]["y"]

    unique_vals, counts = np.unique(open_pos, return_counts=True)
    ax.bar(
        [0],
        [len(open_zeros)],
        color="#5A189A",
        width=0.6,
        label=f"真实营业零销售 ({len(open_zeros)}天, 93.58%)",
    )
    ax.bar(
        unique_vals,
        counts,
        color="#06D6A0",
        width=0.6,
        label=f"真实正销售 ({len(open_pos)}天, 6.42%)",
    )

    ax.set_title(
        "luxury: 消除闭店后的真实营业需求",
        fontsize=10.5,
        fontweight="bold",
        color="#1D3557",
        pad=8,
    )
    ax.set_xlabel("真实日需求量 $y$", fontsize=9.5)
    ax.set_ylabel("频数 (天数)", fontsize=9.5)
    ax.set_xlim(-1, 10.5)
    ax.set_ylim(0, 1150)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.2, framealpha=0.9)

    ax.text(
        0,
        1035,
        f"依然有 1,020 天\n严格为 0 (93.58%)",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#5A189A",
    )
    stat_str = (
        "营业天数: 1,090 天\n"
        "真实零销售率: 93.58% (1,020天)\n"
        "正值条件均值: 1.30 (VIP峰值可达10)\n"
        "形态: 复合泊松-伽马 (Tweedie $1 < p < 2$)"
    )
    ax.text(
        0.32,
        0.48,
        stat_str,
        transform=ax.transAxes,
        fontsize=8.2,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CED4DA", alpha=0.95
        ),
    )
    ax.text(
        0.05,
        0.08,
        "核心结论: 零销售是内在经济机制，非外生闭店；\n具有绝对零点塌缩质点，Tweedie 理论最优！",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#7B2CBF",
        fontweight="bold",
    )

    # Save to PNG and WebP
    out_png = IMG_DIR / "fig-tweedie-02-scenario-distributions-censoring.png"
    out_webp = IMG_DIR / "fig-tweedie-02-scenario-distributions-censoring.webp"

    plt.savefig(out_png, dpi=300, facecolor="white")
    plt.close()

    # Convert to high-quality WebP
    with Image.open(out_png) as img:
        img.save(out_webp, "WEBP", quality=95)

    print(f"Successfully generated:\n  - {out_png}\n  - {out_webp}")


if __name__ == "__main__":
    plot_scenario_distributions()
