"""
scripts/plot_teaching_scenarios.py
----------------------------------
Regenerates the 4 core scenario demand figures and the ADI-CV demand profile chart
in `images/` using the 5-year canonical teaching dataset (2022-07-04 to 2027-06-27).

Generated visual assets:
1. images/grocery-demand.png / .webp
2. images/wine-demand.png / .webp
3. images/garden-demand.png / .webp
4. images/luxury-handbag-demand.png / .webp
5. images/retail-demand-profile-adi-cv.png / .webp
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "teaching"
IMAGE_DIR = REPO_ROOT / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def plot_grocery():
    print("Generating grocery-demand (5-year)...")
    df = pd.read_csv(
        DATA_DIR / "grocery_scenario_data.csv", parse_dates=["ds"]
    ).sort_values("ds")

    fig, (ax_all, ax_zoom) = plt.subplots(2, 1, figsize=(14, 8))
    marker_styles = {
        "is_closed_holiday": ("s", "#D9534F", 42),
        "stockout": ("^", "#E69F00", 42),
    }

    def add_state_markers(axis, frame):
        for column, (marker, color, size) in marker_styles.items():
            mask = frame[column].astype(bool)
            axis.scatter(
                frame.loc[mask, "ds"],
                frame.loc[mask, "y"],
                marker=marker,
                color=color,
                s=size,
                alpha=0.75,
                label=f"{column}=True",
                zorder=3,
            )

    ax_all.plot(df["ds"], df["y"], color="#2878B5", linewidth=0.7, label="y (sales)")
    add_state_markers(ax_all, df)
    ax_all.set_title(
        "Synthetic Grocery Sales (5-Year: 2022-2027) & Operating-State Markers",
        fontsize=12,
        fontweight="bold",
    )
    ax_all.set_ylabel("Observed Sales (units)")
    ax_all.legend(loc="upper left", ncol=3, framealpha=0.9)
    ax_all.grid(True, alpha=0.3)

    # Zoom window: 2023-03-18 to 2023-04-07
    zoom_start = pd.Timestamp("2023-03-18")
    zoom_end = pd.Timestamp("2023-04-07")
    zoom_data = df.loc[df["ds"].between(zoom_start, zoom_end)]
    ax_zoom.plot(
        zoom_data["ds"],
        zoom_data["y"],
        color="#2878B5",
        lw=1.5,
        label="y (sales)",
        zorder=2,
    )
    add_state_markers(ax_zoom, zoom_data)
    ax_discount = ax_zoom.twinx()
    ax_discount.plot(
        zoom_data["ds"],
        zoom_data["discount"],
        color="#C69200",
        marker="+",
        linestyle="--",
        label="discount",
    )
    ax_zoom.set_title(
        "Zoom: Mar 18 - Apr 7, 2023, Sales and Discount Promotion",
        fontsize=11,
        fontweight="bold",
    )
    ax_zoom.set_ylabel("Observed Sales (units)")
    ax_discount.set_ylabel("Discount")
    ax_zoom.set_xlabel("Date (ds)")
    handles, labels = ax_zoom.get_legend_handles_labels()
    extra_handles, extra_labels = ax_discount.get_legend_handles_labels()
    ax_zoom.legend(
        handles + extra_handles,
        labels + extra_labels,
        loc="upper right",
        framealpha=0.9,
    )
    ax_zoom.grid(True, alpha=0.3)

    fig.tight_layout()
    for fmt in ("png", "webp"):
        fig.savefig(
            IMAGE_DIR / f"grocery-demand.{fmt}",
            dpi=160,
            bbox_inches="tight",
            format=fmt,
        )
    plt.close(fig)
    print("  -> Saved grocery-demand.png & .webp")


def plot_wine():
    print("Generating wine-demand (5-year)...")
    df = pd.read_csv(
        DATA_DIR / "wine_scenario_data.csv", parse_dates=["ds"]
    ).sort_values("ds")

    fig, (ax_all, ax_zoom) = plt.subplots(2, 1, figsize=(14, 8))
    wine_marker_styles = {
        "is_closed_holiday": ("s", "#D9534F", 42),
        "stockout": ("^", "#E69F00", 42),
    }

    def add_wine_state_markers(axis, frame):
        closed = ~frame["is_open"]
        axis.scatter(
            frame.loc[closed, "ds"],
            frame.loc[closed, "y"],
            marker=wine_marker_styles["is_closed_holiday"][0],
            color=wine_marker_styles["is_closed_holiday"][1],
            s=wine_marker_styles["is_closed_holiday"][2],
            alpha=0.75,
            label="is_closed_holiday=True",
            zorder=3,
        )
        stockout = frame["stockout"].astype(bool)
        if stockout.any():
            axis.scatter(
                frame.loc[stockout, "ds"],
                frame.loc[stockout, "y"],
                marker=wine_marker_styles["stockout"][0],
                color=wine_marker_styles["stockout"][1],
                s=wine_marker_styles["stockout"][2],
                alpha=0.75,
                label="stockout=True",
                zorder=3,
            )

    ax_all.plot(df["ds"], df["y"], color="#2878B5", linewidth=0.7, label="y (sales)")
    add_wine_state_markers(ax_all, df)
    ax_all.set_title(
        "Synthetic Wine Sales (5-Year: 2022-2027) & Operating-State Markers",
        fontsize=12,
        fontweight="bold",
    )
    ax_all.set_ylabel("Observed Sales (bottles)")
    ax_all.legend(loc="upper left", ncol=3, framealpha=0.9)
    ax_all.grid(True, alpha=0.3)

    # Zoom window: 2023-03-18 to 2023-04-07
    zoom_start = pd.Timestamp("2023-03-18")
    zoom_end = pd.Timestamp("2023-04-07")
    zoom_data = df.loc[df["ds"].between(zoom_start, zoom_end)]
    ax_zoom.plot(
        zoom_data["ds"],
        zoom_data["y"],
        color="#2878B5",
        lw=1.5,
        label="y (sales)",
        zorder=2,
    )
    add_wine_state_markers(ax_zoom, zoom_data)
    ax_promo = ax_zoom.twinx()
    ax_promo.plot(
        zoom_data["ds"],
        zoom_data["promo_case_buy"],
        color="#C69200",
        marker="o",
        linestyle="--",
        label="promo_case_buy",
    )
    ax_zoom.set_title(
        "Zoom: Mar 18 - Apr 7, 2023, Sales and Case-Buy Promotion",
        fontsize=11,
        fontweight="bold",
    )
    ax_zoom.set_ylabel("Observed Sales (bottles)")
    ax_promo.set_ylabel("promo_case_buy")
    ax_promo.set_yticks([0, 1], labels=["False", "True"])
    ax_zoom.set_xlabel("Date (ds)")
    handles, labels = ax_zoom.get_legend_handles_labels()
    extra_handles, extra_labels = ax_promo.get_legend_handles_labels()
    ax_zoom.legend(
        handles + extra_handles,
        labels + extra_labels,
        loc="upper right",
        ncol=3,
        framealpha=0.9,
    )
    ax_zoom.grid(True, alpha=0.3)

    fig.tight_layout()
    for fmt in ("png", "webp"):
        fig.savefig(
            IMAGE_DIR / f"wine-demand.{fmt}", dpi=160, bbox_inches="tight", format=fmt
        )
    plt.close(fig)
    print("  -> Saved wine-demand.png & .webp")


def plot_garden():
    print("Generating garden-demand (5-year)...")
    df = pd.read_csv(
        DATA_DIR / "garden_scenario_data.csv", parse_dates=["ds"]
    ).sort_values("ds")

    fig, (ax_all, ax_zoom) = plt.subplots(2, 1, figsize=(14, 8))

    ax_all.plot(df["ds"], df["y"], color="#2878B5", linewidth=0.7, label="y (sales)")
    closed = ~df["is_open"]
    ax_all.scatter(
        df.loc[closed, "ds"],
        df.loc[closed, "y"],
        marker="s",
        color="#D9534F",
        s=42,
        alpha=0.75,
        label="is_closed_holiday=True",
        zorder=3,
    )
    stockout = df["stockout"].astype(bool)
    if stockout.any():
        ax_all.scatter(
            df.loc[stockout, "ds"],
            df.loc[stockout, "y"],
            marker="^",
            color="#E69F00",
            s=42,
            alpha=0.75,
            label="stockout=True",
            zorder=3,
        )
    ax_all.set_title(
        "Synthetic Garden Sales (5-Year: 2022-2027) & Annual Weather Seasonality",
        fontsize=12,
        fontweight="bold",
    )
    ax_all.set_ylabel("Observed Sales (units)")
    ax_all.legend(loc="upper left", ncol=3, framealpha=0.9)
    ax_all.grid(True, alpha=0.3)

    # Zoom window: 2023-05-18 to 2023-06-07
    zoom_start = pd.Timestamp("2023-05-18")
    zoom_end = pd.Timestamp("2023-06-07")
    zoom_data = df.loc[df["ds"].between(zoom_start, zoom_end)]
    ax_zoom.plot(
        zoom_data["ds"],
        zoom_data["y"],
        color="#2878B5",
        lw=1.5,
        label="y (sales)",
        zorder=2,
    )
    zoom_closed = ~zoom_data["is_open"]
    ax_zoom.scatter(
        zoom_data.loc[zoom_closed, "ds"],
        zoom_data.loc[zoom_closed, "y"],
        marker="s",
        color="#D9534F",
        s=48,
        alpha=0.75,
        label="is_closed_holiday=True",
        zorder=3,
    )
    ax_event = ax_zoom.twinx()
    ax_event.plot(
        zoom_data["ds"],
        zoom_data["is_end_of_autumn_clearance"],
        color="#C69200",
        marker="o",
        linestyle="--",
        label="is_end_of_autumn_clearance",
    )
    ax_zoom.set_title(
        "Zoom: May 18 - Jun 7, 2023, Sales and Autumn Clearance",
        fontsize=11,
        fontweight="bold",
    )
    ax_zoom.set_ylabel("Observed Sales (units)")
    ax_event.set_ylabel("is_end_of_autumn_clearance")
    ax_event.set_yticks([0, 1], labels=["False", "True"])
    ax_zoom.set_xlabel("Date (ds)")
    handles, labels = ax_zoom.get_legend_handles_labels()
    extra_handles, extra_labels = ax_event.get_legend_handles_labels()
    ax_zoom.legend(
        handles + extra_handles,
        labels + extra_labels,
        loc="upper right",
        ncol=3,
        framealpha=0.9,
    )
    ax_zoom.grid(True, alpha=0.3)

    fig.tight_layout()
    for fmt in ("png", "webp"):
        fig.savefig(
            IMAGE_DIR / f"garden-demand.{fmt}", dpi=160, bbox_inches="tight", format=fmt
        )
    plt.close(fig)
    print("  -> Saved garden-demand.png & .webp")


def plot_luxury():
    print("Generating luxury-handbag-demand (5-year)...")
    df = pd.read_csv(
        DATA_DIR / "luxury_scenario_data.csv", parse_dates=["ds"]
    ).sort_values("ds")

    fig, (ax_all, ax_zoom) = plt.subplots(2, 1, figsize=(14, 8))

    positive_sales = df["y"] > 0
    ax_all.step(
        df["ds"],
        df["y"],
        where="mid",
        color="#2878B5",
        linewidth=0.7,
        label="sales (units)",
        zorder=2,
    )
    ax_all.scatter(
        df.loc[positive_sales, "ds"],
        df.loc[positive_sales, "y"],
        marker="o",
        color="#2878B5",
        s=26,
        label="positive sales",
        zorder=3,
    )
    vip_mask = df["is_vip_event"].astype(bool)
    ax_all.scatter(
        df.loc[vip_mask, "ds"],
        df.loc[vip_mask, "y"],
        marker="D",
        color="#D9534F",
        s=28,
        label="VIP event",
        zorder=4,
    )
    ax_all.set_title(
        "Luxury Handbag: Intermittent Daily Sales (5-Year: 2022-2027, 93.6% Zero Sales)",
        fontsize=12,
        fontweight="bold",
    )
    ax_all.set_ylabel("Sales (units)")
    ax_all.legend(loc="upper left", ncol=3, framealpha=0.9)
    ax_all.grid(True, alpha=0.3)

    # Zoom window: 2023-12-01 to 2023-12-21
    zoom_start = pd.Timestamp("2023-12-01")
    zoom_end = pd.Timestamp("2023-12-21")
    zoom_data = df.loc[df["ds"].between(zoom_start, zoom_end)]
    zoom_positive = zoom_data["y"] > 0
    ax_zoom.step(
        zoom_data["ds"],
        zoom_data["y"],
        where="mid",
        color="#2878B5",
        lw=1.5,
        label="sales",
        zorder=2,
    )
    ax_zoom.scatter(
        zoom_data.loc[zoom_positive, "ds"],
        zoom_data.loc[zoom_positive, "y"],
        marker="o",
        color="#2878B5",
        s=40,
        label="positive sales",
        zorder=3,
    )
    zoom_vip = zoom_data["is_vip_event"].astype(bool)
    ax_zoom.scatter(
        zoom_data.loc[zoom_vip, "ds"],
        zoom_data.loc[zoom_vip, "y"],
        marker="D",
        color="#D9534F",
        s=44,
        label="VIP event",
        zorder=4,
    )
    ax_probability = ax_zoom.twinx()
    ax_probability.plot(
        zoom_data["ds"],
        zoom_data["occurrence_probability"],
        color="#C69200",
        marker="o",
        linestyle="--",
        label="occurrence_probability",
    )
    ax_zoom.set_title(
        "Zoom: Dec 1-21, 2023, Sales and Occurrence Probability",
        fontsize=11,
        fontweight="bold",
    )
    ax_zoom.set_ylabel("Sales (units)")
    ax_probability.set_ylabel("Occurrence Probability")
    ax_zoom.set_xlabel("Date (ds)")
    handles, labels = ax_zoom.get_legend_handles_labels()
    extra_handles, extra_labels = ax_probability.get_legend_handles_labels()
    ax_zoom.legend(
        handles + extra_handles,
        labels + extra_labels,
        loc="upper left",
        ncol=2,
        framealpha=0.9,
    )
    ax_zoom.grid(True, alpha=0.3)

    fig.tight_layout()
    for fmt in ("png", "webp"):
        fig.savefig(
            IMAGE_DIR / f"luxury-handbag-demand.{fmt}",
            dpi=160,
            bbox_inches="tight",
            format=fmt,
        )
    plt.close(fig)
    print("  -> Saved luxury-handbag-demand.png & .webp")


def plot_adi_cv():
    print("Generating retail-demand-profile-adi-cv (5-year)...")
    diagnostics = pd.read_csv(DATA_DIR / "scenario_diagnostics.csv")

    frequent_scenarios = diagnostics[
        diagnostics["scenario"].isin(["grocery", "wine", "garden"])
    ]
    luxury_scenario = diagnostics[diagnostics["scenario"] == "luxury"]
    features = diagnostics[["ADI", "CV_plus_squared"]].to_numpy(dtype=float)
    labels = diagnostics["scenario"].eq("luxury").astype(int).to_numpy()

    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=10.0, class_weight="balanced", random_state=0),
    )
    svm.fit(features, labels)

    scaler = svm.named_steps["standardscaler"]
    classifier = svm.named_steps["svc"]
    raw_coefficients = classifier.coef_[0] / scaler.scale_
    raw_intercept = classifier.intercept_[0] - np.dot(raw_coefficients, scaler.mean_)
    svm_slope = -raw_coefficients[0] / raw_coefficients[1]
    svm_intercept = -raw_intercept / raw_coefficients[1]
    svm_equation = (
        rf"$\mathrm{{CV}}_+^2 = {svm_slope:.4f}\,\mathrm{{ADI}} {svm_intercept:+.4f}$"
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    adi_limit = max(18.0, diagnostics["ADI"].max() * 1.15)
    cv_limit = max(1.1, diagnostics["CV_plus_squared"].max() * 1.15)
    grid_adi, grid_cv = np.meshgrid(
        np.linspace(0, adi_limit, 300),
        np.linspace(0, cv_limit, 300),
    )
    grid_features = np.column_stack([grid_adi.ravel(), grid_cv.ravel()])
    decision_values = svm.decision_function(grid_features).reshape(grid_adi.shape)
    decision_limit = np.abs(decision_values).max()
    decision_norm = TwoSlopeNorm(vmin=-decision_limit, vcenter=0, vmax=decision_limit)
    decision_cmap = LinearSegmentedColormap.from_list(
        "svm_decision_regions", ["#BFDCEC", "#F7F8FA", "#F1C3C6"]
    )
    heatmap = ax.contourf(
        grid_adi,
        grid_cv,
        decision_values,
        levels=41,
        cmap=decision_cmap,
        norm=decision_norm,
        alpha=0.8,
        extend="both",
        zorder=0,
    )

    ax.scatter(
        frequent_scenarios["ADI"],
        frequent_scenarios["CV_plus_squared"],
        color="#2878B5",
        marker="o",
        s=80,
        label="Grocery / Wine / Garden",
        zorder=3,
    )
    ax.scatter(
        luxury_scenario["ADI"],
        luxury_scenario["CV_plus_squared"],
        color="#D9534F",
        marker="D",
        s=95,
        label="Luxury Handbag",
        zorder=4,
    )

    scenario_labels = {
        "grocery": "Grocery",
        "wine": "Wine",
        "garden": "Garden",
        "luxury": "Luxury",
    }
    label_offsets = {
        "grocery": (8, -12),
        "wine": (8, 6),
        "garden": (8, -2),
        "luxury": (8, 0),
    }
    for row in diagnostics.itertuples(index=False):
        ax.annotate(
            scenario_labels[row.scenario],
            (row.ADI, row.CV_plus_squared),
            xytext=label_offsets[row.scenario],
            textcoords="offset points",
            fontweight="bold",
        )

    ax.contour(
        grid_adi,
        grid_cv,
        decision_values,
        levels=[0],
        colors=["#6A4C93"],
        linestyles=["-."],
        linewidths=1.7,
    )
    ax.axvline(
        1.32, color="0.4", linestyle="--", linewidth=1, label="ADI reference: 1.32"
    )
    ax.axhline(
        0.49,
        color="0.55",
        linestyle=":",
        linewidth=1,
        label="CV+ squared reference: 0.49",
    )
    ax.text(
        0.03,
        0.95,
        "Frequent / Lower variability",
        transform=ax.transAxes,
        va="top",
        color="#2878B5",
        fontweight="bold",
    )
    ax.text(
        0.68,
        0.95,
        "Intermittent / Lumpy",
        transform=ax.transAxes,
        va="top",
        color="#D9534F",
        fontweight="bold",
    )

    formula_x = 8.0
    formula_y = svm_slope * formula_x + svm_intercept
    ax.annotate(
        svm_equation,
        xy=(formula_x, formula_y),
        xytext=(formula_x - 2.2, formula_y + 0.14),
        textcoords="data",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.25",
            "fc": "white",
            "ec": "#6A4C93",
            "alpha": 0.95,
        },
        arrowprops={"arrowstyle": "->", "color": "#6A4C93", "linewidth": 1.0},
    )

    ax.set_xlim(0, adi_limit)
    ax.set_ylim(0, cv_limit)
    ax.set_xlabel("ADI (Average Demand Interval)")
    ax.set_ylabel("CV+ squared (Positive-demand variability)")
    ax.set_title(
        "Retail Scenarios: Demand Profiles in ADI - CV² Space (5-Year)",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.25)
    handles, _legend_labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [], [], color="#6A4C93", linestyle="-.", label="Linear SVM boundary (n=4)"
        )
    )
    ax.legend(handles=handles, loc="center right", framealpha=0.9)
    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("SVM decision score")
    fig.tight_layout()

    for fmt in ("png", "webp"):
        fig.savefig(
            IMAGE_DIR / f"retail-demand-profile-adi-cv.{fmt}",
            dpi=160,
            bbox_inches="tight",
            format=fmt,
        )
    plt.close(fig)
    print("  -> Saved retail-demand-profile-adi-cv.png & .webp")


def main():
    plot_grocery()
    plot_wine()
    plot_garden()
    plot_luxury()
    plot_adi_cv()
    print("\nAll 5 scenario data figures successfully regenerated with 5-year data!")


if __name__ == "__main__":
    main()
