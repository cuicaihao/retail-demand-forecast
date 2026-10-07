"""Plot the saved teaching-data snapshot without fitting or regenerating data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "generated" / "prophet" / "teaching_data_overview.png"


def main():
    # Use a local Chinese font; no font downloads or additional dependencies.
    for path in (
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
    ):
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    else:
        plt.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"
        ]
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})

    data = pd.read_csv(
        ROOT / "data" / "teaching" / "grocery_scenario_data.csv", parse_dates=["ds"]
    ).sort_values("ds")
    assert not data["ds"].duplicated().any()
    assert data["ds"].min() == pd.Timestamp("2023-01-01")
    assert data["ds"].max() == pd.Timestamp("2025-12-31")
    cutoff = data["ds"].iloc[-29]
    test_start = data["ds"].iloc[-28]
    assert cutoff == pd.Timestamp("2025-12-03")
    assert test_start == pd.Timestamp("2025-12-04")
    assert data.loc[~data["is_open"], "y"].eq(0).all()

    blue, grey, red, purple = "#176A96", "#85929D", "#C35B27", "#7A447D"
    fig, axes = plt.subplots(2, 1, figsize=(13.6, 8.6))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.075, right=0.97, top=0.77, bottom=0.10, hspace=0.43)
    fig.suptitle("主案例教学数据：需求如何变成销售记录", x=0.075, y=0.965,
                 ha="left", fontsize=19, fontweight="bold", color="#233747")
    fig.text(0.075, 0.915, "VIC 门店商品合成数据  ·  2023—2025 年  ·  最后 28 天留作测试",
             color="#526574", fontsize=11)

    handles = [
        Line2D([], [], color=grey, lw=1.2, label="实际销售 y（合成记录）"),
        Line2D([], [], color=blue, lw=1.5, label="教学设定的无噪声营业需求"),
        Line2D([], [], color=red, marker="x", linestyle="none", label="营业日缺货"),
        Line2D([], [], color=purple, marker="s", linestyle="none", label="闭店零销售"),
        Patch(facecolor="#F6DDA8", alpha=0.65, label="测试期：12/04—12/31"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.068, 0.887),
               ncol=3, frameon=False, columnspacing=2.0, handlelength=2.5)

    views = [data, data.loc[data["ds"] >= "2025-11-01"]]
    titles = ["A  三年全貌：年内起伏与销售约束", "B  末段放大：2025 年 11—12 月"]
    for index, (ax, frame, title) in enumerate(zip(axes, views, titles)):
        ax.set_facecolor("white")
        ax.axvspan(test_start, data["ds"].max() + pd.Timedelta(hours=12),
                   color="#F6DDA8", alpha=0.55, zorder=0)
        ax.plot(frame["ds"], frame["y"], color=grey, alpha=0.80,
                lw=0.65 if index == 0 else 1.3, zorder=2)
        ax.plot(frame["ds"], frame["conditional_mean"], color=blue,
                lw=0.75 if index == 0 else 1.6, alpha=0.90, zorder=3)
        stockouts = frame.loc[frame["stockout"]]
        closed = frame.loc[~frame["is_open"]]
        ax.scatter(stockouts["ds"], stockouts["y"], marker="x", s=27,
                   color=red, linewidths=1.2, zorder=5)
        ax.scatter(closed["ds"], closed["y"], marker="s", s=30,
                   color=purple, edgecolors="white", linewidths=0.6, zorder=5)
        ax.axvline(cutoff, color="#8A6D40", linestyle="--", lw=1.1, zorder=4)
        ax.set_title(title, loc="left", fontsize=12, pad=13, color="#233747")
        ax.set_ylabel("件 / 天", labelpad=10)
        ax.set_ylim(-10, 210)
        ax.set_yticks([0, 50, 100, 150, 200])
        ax.set_xlim(frame["ds"].min(), frame["ds"].max() + pd.Timedelta(hours=12))
        ax.grid(axis="y", color="#E6EBEF", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#CDD6DD")
        ax.tick_params(colors="#526574", length=3)
        if index == 0:
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.text(pd.Timestamp("2025-11-26"), 196, "训练截至 12/03",
                    ha="right", color="#795D31", fontsize=10)
            ax.text(pd.Timestamp("2025-12-17"), 196, "28 天测试期",
                    ha="center", color="#795D31", fontsize=10)
            ax.annotate("圣诞节闭店：销售为 0", xy=(pd.Timestamp("2025-12-25"), 0),
                        xytext=(pd.Timestamp("2025-12-08"), 38), color=purple,
                        arrowprops={"arrowstyle": "->", "color": purple}, fontsize=10)
            ax.text(0.01, 0.04, "这一局部窗口没有营业日缺货", transform=ax.transAxes,
                    color="#657785", fontsize=9)

    fig.text(0.075, 0.035, "蓝线来自数据生成公式；销售记录另外受到随机波动、缺货和营业状态的影响。",
             color="#526574", fontsize=10)
    fig.savefig(OUTPUT, dpi=180, facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
