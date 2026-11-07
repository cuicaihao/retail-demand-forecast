"""
Generate high-resolution distribution spectrum chart showing the relationship between
Tweedie variance power parameter p and probability density/mass functions.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy import stats

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


def simulate_compound_poisson_gamma(
    mu: float, p: float, phi: float = 1.0, n_samples: int = 300000, seed: int = 42
):
    rng = np.random.default_rng(seed)
    lam = (mu ** (2 - p)) / (phi * (2 - p))
    alpha = (2 - p) / (p - 1)
    beta = phi * (p - 1) * (mu ** (p - 1))
    scale = 1.0 / beta

    n_events = rng.poisson(lam, size=n_samples)
    y = np.zeros(n_samples)
    positive_mask = n_events > 0
    pos_counts = n_events[positive_mask]

    # Each event draws Gamma(alpha, scale) -> sum of k Gammas is Gamma(k * alpha, scale)
    y[positive_mask] = rng.gamma(shape=pos_counts * alpha, scale=scale)
    prob_zero = np.exp(-lam)
    return y, prob_zero, lam, alpha, scale


def plot_tweedie_spectrum():
    setup_fonts()
    fig = plt.figure(figsize=(15.2, 10.8), facecolor="white")

    # Grid specification: Row 0 for Continuum Axis, Rows 1-2 for 6 Density Subplots
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[0.8, 1.2, 1.2],
        hspace=0.38,
        wspace=0.25,
        left=0.06,
        right=0.96,
        top=0.92,
        bottom=0.06,
    )

    # -------------------------------------------------------------
    # 0. Header Title
    # -------------------------------------------------------------
    fig.suptitle(
        "Tweedie 指数分散族（EDM）：方差幂指数 $p$ 与分布形态演变全景图谱\n"
        r"$\mathrm{Var}(Y) = \phi \cdot \mu^p$ —— 从连续对称、离散计数、零膨胀复合泊松-伽马，到连续正偏厚尾",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # -------------------------------------------------------------
    # 1. Continuum Spectrum Axis (Top Banner)
    # -------------------------------------------------------------
    ax_bar = fig.add_subplot(gs[0, :])
    ax_bar.set_xlim(-0.3, 3.3)
    ax_bar.set_ylim(-0.7, 1.3)
    ax_bar.axis("off")

    # Main horizontal axis line
    ax_bar.annotate(
        "",
        xy=(3.25, 0.4),
        xytext=(-0.25, 0.4),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="#2C3E50"),
    )
    ax_bar.text(
        3.28,
        0.4,
        "幂参数 $p$",
        fontsize=11,
        fontweight="bold",
        va="center",
        color="#2C3E50",
    )

    # Regions & Anchors
    anchors = [
        (
            0.0,
            "p = 0\n正态分布 (Normal)",
            "#2980B9",
            "Var(Y) = φ\n连续实数 (-∞, +∞)\n经典加性噪声 (Garden/Grocery)",
            0.8,
        ),
        (
            0.5,
            "0 < p < 1\n不存在 EDM",
            "#95A5A6",
            "数学上不存在\n对应有效概率分布",
            0.0,
        ),
        (
            1.0,
            "p = 1\n泊松分布 (Poisson)",
            "#27AE60",
            "Var(Y) = φμ\n离散非负整数 {0,1,2...}\n低频计数 (Wine脉冲)",
            0.8,
        ),
        (
            1.5,
            "1 < p < 2\n复合泊松-伽马 (Compound Poisson-Gamma)",
            "#D35400",
            "★ 核心重点：零膨胀半连续分布 [0, +∞)\nN ~ Poisson(λ) 且 Z ~ Gamma(α, β)\n90%+零需求/间歇性需求 (Luxury手袋/精算纯保费)",
            0.8,
        ),
        (
            2.0,
            "p = 2\n伽马分布 (Gamma)",
            "#8E44AD",
            "Var(Y) = φμ²\n严格正连续 (0, +∞)\n恒正乘性右偏",
            0.8,
        ),
        (
            3.0,
            "p = 3\n逆高斯分布 (Inv-Gaussian)",
            "#C0392B",
            "Var(Y) = φμ³\n极端右偏厚尾 (0, +∞)\n高危巨额索赔/长尾等待",
            0.8,
        ),
    ]

    # Draw shaded range for 1 < p < 2
    ax_bar.axvspan(1.0, 2.0, ymin=0.45, ymax=0.85, color="#FAD7A0", alpha=0.5, zorder=1)
    ax_bar.axvspan(
        0.05,
        0.95,
        ymin=0.5,
        ymax=0.75,
        color="#E5E7E9",
        alpha=0.6,
        linestyle=":",
        zorder=1,
    )

    for p_val, title, col, desc, y_pos in anchors:
        if p_val == 0.5:
            ax_bar.text(
                p_val,
                0.4,
                "× 不存在",
                ha="center",
                va="center",
                color="#7F8C8D",
                fontsize=9,
                fontweight="bold",
            )
            continue
        # Tick marker
        ax_bar.plot([p_val, p_val], [0.32, 0.48], lw=3, color=col)
        # Title box
        ax_bar.text(
            p_val,
            0.65,
            title,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=col,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, lw=1.2),
        )
        # Description
        ax_bar.text(
            p_val, 0.15, desc, ha="center", va="top", fontsize=8.2, color="#34495E"
        )

    # -------------------------------------------------------------
    # 2. Six Distribution Subplots (2 rows x 3 cols)
    # -------------------------------------------------------------

    # (a) p = 0: Normal Distribution
    ax_a = fig.add_subplot(gs[1, 0])
    x_norm = np.linspace(2, 18, 500)
    y_norm = stats.norm.pdf(x_norm, loc=10, scale=2.5)
    ax_a.plot(x_norm, y_norm, color="#2980B9", lw=2.2)
    ax_a.fill_between(x_norm, 0, y_norm, color="#2980B9", alpha=0.25)
    ax_a.axvline(10, color="#2980B9", linestyle="--", alpha=0.7, label="期望 μ=10")
    ax_a.set_title(
        "(a) p = 0: 正态高斯分布 (Normal)\nVar(Y) = φ, 支持集 (-∞, +∞)",
        fontsize=10.5,
        fontweight="bold",
        color="#2980B9",
    )
    ax_a.set_xlabel("需求 y")
    ax_a.set_ylabel("概率密度 f(y)")
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.text(
        0.05,
        0.85,
        "• 对称钟形曲线\n• 允许负值产生\n• 方差与均值解耦 (同方差)",
        transform=ax_a.transAxes,
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7"),
    )

    # (b) p = 1.0: Poisson Distribution
    ax_b = fig.add_subplot(gs[1, 1])
    k_pois = np.arange(0, 12)
    pmf_pois = stats.poisson.pmf(k_pois, mu=3.5)
    ax_b.bar(
        k_pois, pmf_pois, color="#27AE60", width=0.55, alpha=0.8, edgecolor="#1E8449"
    )
    ax_b.set_title(
        "(b) p = 1.0: 泊松分布 (Poisson)\nVar(Y) = φμ, 支持集 {0, 1, 2, ...}",
        fontsize=10.5,
        fontweight="bold",
        color="#27AE60",
    )
    ax_b.set_xlabel("需求件数 k")
    ax_b.set_ylabel("概率质量 P(Y=k)")
    ax_b.set_xticks(k_pois)
    ax_b.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax_b.text(
        0.48,
        0.85,
        "• 严格离散计数\n• 方差随均值线性增长\n• P(Y=0) = e^(-μ) ≈ 3.0%",
        transform=ax_b.transAxes,
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7"),
    )

    # (c) p = 1.2: Tweedie Compound Poisson-Gamma (Near Poisson)
    ax_c = fig.add_subplot(gs[1, 2])
    sim_c, p0_c, _lam_c, _, _ = simulate_compound_poisson_gamma(mu=1.8, p=1.2, phi=1.0)
    pos_c = sim_c[sim_c > 0]
    ax_c.bar(
        [0.0],
        [p0_c],
        width=0.25,
        color="#E67E22",
        edgecolor="#B9770E",
        zorder=4,
        label=f"零点质点 P(0)={p0_c:.1%}",
    )
    kde_c = stats.gaussian_kde(pos_c)
    x_c = np.linspace(0.1, 7.5, 400)
    # Density scaled by P(Y>0)
    ax_c.plot(
        x_c,
        kde_c(x_c) * (1 - p0_c),
        color="#D35400",
        lw=2.2,
        label="连续正偏密度 (Y>0)",
    )
    ax_c.fill_between(x_c, 0, kde_c(x_c) * (1 - p0_c), color="#E67E22", alpha=0.25)
    ax_c.set_title(
        "(c) p = 1.2: Tweedie 复合过程 (偏泊松)\nVar(Y) = φ·μ^1.2, 零点质点 + 离散向长尾",
        fontsize=10.5,
        fontweight="bold",
        color="#D35400",
    )
    ax_c.set_xlabel("需求 y")
    ax_c.set_ylabel("概率质量/密度")
    ax_c.grid(True, linestyle=":", alpha=0.6)
    ax_c.legend(loc="upper right", fontsize=8.2, framealpha=0.9)
    ax_c.text(
        0.05,
        0.60,
        f"• 均值 μ = 1.8\n• 零点概率 P(0) = {p0_c:.1%}\n• 峰值靠左，离散事件频发",
        transform=ax_c.transAxes,
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7"),
    )

    # (d) p = 1.6: Tweedie Compound Poisson-Gamma (Near Gamma, Extreme Zero-Inflation)
    ax_d = fig.add_subplot(gs[2, 0])
    sim_d, p0_d, _lam_d, _, _ = simulate_compound_poisson_gamma(mu=0.35, p=1.6, phi=1.0)
    pos_d = sim_d[sim_d > 0]
    ax_d.bar(
        [0.0],
        [p0_d],
        width=0.25,
        color="#D35400",
        edgecolor="#935116",
        zorder=4,
        label=f"零点质点 P(0)={p0_d:.1%}",
    )
    if len(pos_d) > 10:
        kde_d = stats.gaussian_kde(pos_d)
        x_d = np.linspace(0.05, 4.0, 400)
        ax_d.plot(
            x_d,
            kde_d(x_d) * (1 - p0_d),
            color="#BA4A00",
            lw=2.2,
            label="连续正值长尾 (Y>0)",
        )
        ax_d.fill_between(x_d, 0, kde_d(x_d) * (1 - p0_d), color="#D35400", alpha=0.25)
    ax_d.set_title(
        "(d) p = 1.6: Tweedie 复合过程 (重度零膨胀)\nVar(Y) = φ·μ^1.6, 典型 Luxury 间歇需求",
        fontsize=10.5,
        fontweight="bold",
        color="#BA4A00",
    )
    ax_d.set_xlabel("需求 y")
    ax_d.set_ylabel("概率质量/密度")
    ax_d.grid(True, linestyle=":", alpha=0.6)
    ax_d.legend(loc="upper right", fontsize=8.2, framealpha=0.9)
    ax_d.text(
        0.05,
        0.60,
        f"• 均值 μ = 0.35\n• 零点概率 P(0) = {p0_d:.1%}\n• 绝大多数归零，偶现高价脉冲",
        transform=ax_d.transAxes,
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7"),
    )

    # (e) p = 2.0: Gamma Distribution
    ax_e = fig.add_subplot(gs[2, 1])
    x_gam = np.linspace(0.01, 12, 500)
    y_gam = stats.gamma.pdf(x_gam, a=2.2, scale=1.8)
    ax_e.plot(x_gam, y_gam, color="#8E44AD", lw=2.2)
    ax_e.fill_between(x_gam, 0, y_gam, color="#8E44AD", alpha=0.25)
    ax_e.set_title(
        "(e) p = 2.0: 伽马分布 (Gamma)\nVar(Y) = φμ², 严格正连续支持集 (0, +∞)",
        fontsize=10.5,
        fontweight="bold",
        color="#8E44AD",
    )
    ax_e.set_xlabel("需求 y")
    ax_e.set_ylabel("概率密度 f(y)")
    ax_e.grid(True, linestyle=":", alpha=0.6)
    ax_e.text(
        0.48,
        0.85,
        "• 无零点塌缩质点 P(0)=0\n• 严格恒正 (Y > 0)\n• 变异系数 CV 为常数 (乘性误差)",
        transform=ax_e.transAxes,
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7"),
    )

    # (f) p = 3.0: Inverse Gaussian Distribution
    ax_f = fig.add_subplot(gs[2, 2])
    x_inv = np.linspace(0.01, 10, 500)
    # scipy's invgauss parameterization: mu_scipy = mu / lambda
    y_inv = stats.invgauss.pdf(x_inv, mu=2.0 / 1.5, scale=1.5)
    ax_f.plot(x_inv, y_inv, color="#C0392B", lw=2.2)
    ax_f.fill_between(x_inv, 0, y_inv, color="#C0392B", alpha=0.25)
    ax_f.set_title(
        "(f) p = 3.0: 逆高斯分布 (Inverse Gaussian)\nVar(Y) = φμ³, 极端右偏超厚尾 (0, +∞)",
        fontsize=10.5,
        fontweight="bold",
        color="#C0392B",
    )
    ax_f.set_xlabel("需求 y")
    ax_f.set_ylabel("概率密度 f(y)")
    ax_f.grid(True, linestyle=":", alpha=0.6)
    ax_f.text(
        0.48,
        0.85,
        "• 陡峭起峰，极度平缓衰减\n• 方差随均值立方飙升\n• 巨灾大额索赔/极端脉冲",
        transform=ax_f.transAxes,
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7"),
    )

    # Save outputs
    png_path = IMG_DIR / "fig-tweedie-01-distribution-spectrum.png"
    webp_path = IMG_DIR / "fig-tweedie-01-distribution-spectrum.webp"
    fig.savefig(png_path, dpi=200)
    fig.savefig(webp_path, dpi=200)
    plt.close(fig)
    print(f"Generated chart:\n  - {png_path}\n  - {webp_path}")


if __name__ == "__main__":
    plot_tweedie_spectrum()
