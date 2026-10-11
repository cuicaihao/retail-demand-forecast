from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "teaching"
HORIZON = 28
ENCODER_LENGTH = 84
SEED = 42
CPU_THREADS = 2
RUN_ID = os.environ.get("RETAIL_FORECAST_RUN_ID") or datetime.now(
    timezone.utc
).strftime("%Y%m%dT%H%M%S%fZ")
RUNS_DIR = ROOT / "runs" / RUN_ID
THREAD_LIMITER = threadpool_limits(limits=CPU_THREADS)
SCENARIOS = {
    "grocery": {"features": ["discount"], "label": "Grocery"},
    "wine": {
        "features": ["promo_case_buy", "is_december_peak", "is_party_event"],
        "label": "Wine",
    },
    "garden": {
        "features": ["is_long_weekend_diy", "is_end_of_autumn_clearance"],
        "label": "Garden",
    },
    "luxury": {
        "features": [
            "is_saturday",
            "is_sunday",
            "is_december_peak",
            "is_vip_event",
            "discount",
        ],
        "label": "Luxury",
    },
}
datasets = {}
for name, c in SCENARIOS.items():
    f = (
        pd.read_csv(DATA_DIR / f"{name}_scenario_data.csv", parse_dates=["ds"])
        .sort_values("ds")
        .reset_index(drop=True)
    )
    assert f.ds.is_unique and f.ds.diff().dropna().eq(pd.Timedelta(days=1)).all()
    assert f.y.notna().all() and f.y.ge(0).all()
    f["time_idx"] = np.arange(len(f))
    datasets[name] = f
TRAIN_END = len(datasets["grocery"]) - HORIZON - 1
cutoff = datasets["grocery"].ds.iloc[TRAIN_END]
vic_holidays = pd.read_csv(DATA_DIR / "vic_holidays.csv", parse_dates=["ds"])
holiday_names = sorted(vic_holidays.holiday.unique())
holiday_dates = {
    n: pd.DatetimeIndex(vic_holidays.loc[vic_holidays.holiday.eq(n), "ds"])
    for n in holiday_names
}
WINDOW_STARTS = {
    "grocery": [
        ("ordinary", "2025-02-03"),
        ("peak", "2025-04-07"),
        ("promotion", "2025-05-19"),
        ("low", "2025-10-06"),
    ],
    "wine": [
        ("ordinary", "2025-03-03"),
        ("peak", "2025-07-07"),
        ("promotion", "2025-08-04"),
        ("low", "2025-02-03"),
    ],
    "garden": [
        ("ordinary", "2025-02-03"),
        ("peak", "2025-04-07"),
        ("promotion", "2025-05-19"),
        ("low", "2025-10-06"),
    ],
    "luxury": [
        ("ordinary", "2025-04-07"),
        ("peak", "2024-12-02"),
        ("promotion", "2025-06-16"),
        ("low", "2025-10-06"),
    ],
}
INPUT_HASHES = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA_DIR.glob("*.csv")
}


def valid_labels(f):
    return f.is_open & ~f.stockout & f.y.notna()


def masked_y(f):
    return f.y.where(valid_labels(f)).astype(float)


def make_features(frame, scenario):
    days = (frame.ds - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
    X = pd.DataFrame(index=frame.index)
    X["time_years"] = (frame.ds - pd.Timestamp("2023-01-01")) / pd.Timedelta(
        days=365.25
    )
    for label, period, order in [("weekly", 7, 3), ("yearly", 365.25, 5)]:
        for k in range(1, order + 1):
            a = 2 * np.pi * k * days / period
            X[f"{label}_sin_{k}"] = np.sin(a)
            X[f"{label}_cos_{k}"] = np.cos(a)
    for i, n in enumerate(holiday_names):
        X[f"holiday_{i:02d}"] = frame.ds.isin(holiday_dates[n]).astype(float)
    for col in SCENARIOS[scenario]["features"]:
        X[col] = frame[col].astype(float)
    assert not {
        "y",
        "latent_demand",
        "conditional_mean",
        "stockout",
        "is_open",
    }.intersection(X)
    return X.astype(float)


feature_names = {n: list(make_features(f, n)) for n, f in datasets.items()}


def seasonal_naive(history, future):
    h = history.loc[valid_labels(history), ["ds", "y"]].copy()
    h["weekday"] = h.ds.dt.dayofweek
    v = h.groupby("weekday").tail(1).set_index("weekday").y
    assert len(v) == 7
    return future.ds.dt.dayofweek.map(v).to_numpy(float)


def sales_prediction(f, raw):
    raw = np.asarray(raw, float).reshape(-1)
    assert len(raw) == len(f) and np.isfinite(raw).all()
    return np.where(f.is_open, np.maximum(raw, 0), 0)


def metrics(f, prediction):
    m = valid_labels(f).to_numpy()
    y = f.y.to_numpy(float)[m]
    p = np.asarray(prediction, float)[m]
    assert len(y) > 0
    total = y.sum()  # 间歇性需求窗口可能没有销量，相对指标记为缺失
    return {
        "MAE": float(abs(p - y).mean()),
        "RMSE": float(np.sqrt(((p - y) ** 2).mean())),
        "WAPE": float(abs(p - y).sum() / total) if total > 0 else np.nan,
        "Bias": float((p - y).sum() / total) if total > 0 else np.nan,
        "n": len(y),
    }


def windows(scenario):
    f = datasets[scenario]
    out = []
    seen = set()
    for regime, date in WINDOW_STARTS[scenario]:
        start = int(f.index[f.ds.eq(pd.Timestamp(date))][0])
        assert start + HORIZON - 1 <= TRAIN_END
        dates = set(range(start, start + HORIZON))
        assert not seen.intersection(dates)
        seen.update(dates)
        out.append({"split": "rolling", "regime": regime, "origin": start - 1})
    return out + [{"split": "test", "regime": "final", "origin": TRAIN_END}]


def training_blocks():
    return range(TRAIN_END - 5 * HORIZON + 1, TRAIN_END + 1, HORIZON)


def run_family(family, names, fit, predict):
    out = RUNS_DIR / family
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    preds = []
    fits = []
    finals = {}
    for scenario, f in datasets.items():
        for window in windows(scenario):
            origin = window["origin"]
            history = f.iloc[: origin + 1].copy()
            future = f.iloc[origin + 1 : origin + 1 + HORIZON].copy()
            for name in names:
                print(f"{family} {scenario}/{window['regime']}/{name}", flush=True)
                t = time.perf_counter()
                model, detail = (
                    (history, {})
                    if name == "Seasonal naive"
                    else fit(history, scenario, name, window)
                )
                seconds = time.perf_counter() - t
                raw = (
                    seasonal_naive(history, future)
                    if name == "Seasonal naive"
                    else predict(model, history, future, scenario, name)
                )
                g = future.copy()
                g["raw_prediction"] = raw
                g["prediction"] = sales_prediction(g, raw)
                g["scenario"] = scenario
                g["model"] = name
                g["split"] = window["split"]
                g["regime"] = window["regime"]
                g["forecast_origin"] = history.ds.iloc[-1]
                g["horizon_day"] = np.arange(1, HORIZON + 1)
                preds.append(g)
                rows.append(
                    {
                        "scenario": scenario,
                        "model": name,
                        "split": window["split"],
                        "regime": window["regime"],
                        **metrics(g, g.prediction),
                    }
                )
                fits.append(
                    {
                        "scenario": scenario,
                        "model": name,
                        "regime": window["regime"],
                        "setup_seconds": seconds,
                        **detail,
                    }
                )
                if window["split"] == "test":
                    finals[(scenario, name)] = model
                    blocks = []
                    for start in training_blocks():
                        h = f.iloc[:start].copy()
                        b = f.iloc[start : start + HORIZON].copy()
                        r = (
                            seasonal_naive(h, b)
                            if name == "Seasonal naive"
                            else predict(model, h, b, scenario, name)
                        )
                        b["raw_prediction"] = r
                        b["prediction"] = sales_prediction(b, r)
                        b["forecast_origin"] = h.ds.iloc[-1]
                        b["horizon_day"] = np.arange(1, HORIZON + 1)
                        blocks.append(b)
                    tr = pd.concat(blocks, ignore_index=True)
                    tr["scenario"] = scenario
                    tr["model"] = name
                    tr["split"] = "train_hindcast"
                    tr["regime"] = "training"
                    preds.append(tr)
                    rows.append(
                        {
                            "scenario": scenario,
                            "model": name,
                            "split": "train_hindcast",
                            "regime": "training",
                            **metrics(tr, tr.prediction),
                        }
                    )
                pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
                pd.DataFrame(fits).to_csv(out / "fit_details.csv", index=False)
    p = pd.concat(preds, ignore_index=True)
    m = pd.DataFrame(rows)
    details = pd.DataFrame(fits)
    assert len(m) == len(SCENARIOS) * len(names) * 6
    for _, g in p.groupby(["scenario", "split", "regime"]):
        sets = [set(x.loc[valid_labels(x), "ds"]) for _, x in g.groupby("model")]
        assert all(s == sets[0] for s in sets)
    assert p.loc[p.split.eq("rolling"), "ds"].max() <= cutoff
    p.to_csv(out / "predictions.csv", index=False)
    summary = []
    for (scenario, name), g in p.loc[p.split.eq("rolling")].groupby(
        ["scenario", "model"]
    ):
        summary.append(
            {
                "scenario": scenario,
                "model": name,
                **metrics(g, g.prediction),
                "mean_window_WAPE": float(
                    m.loc[
                        (m.scenario == scenario)
                        & (m.model == name)
                        & m.split.eq("rolling"),
                        "WAPE",
                    ].mean()
                ),
            }
        )
    summary = pd.DataFrame(summary)
    base = summary.loc[summary.model.eq("Seasonal naive")].set_index("scenario").WAPE
    summary["improvement_vs_naive"] = 1 - summary.WAPE / summary.scenario.map(base)
    summary.to_csv(out / "rolling_summary.csv", index=False)
    slices = []
    for (scenario, name), g in p.loc[p.split.eq("rolling")].groupby(
        ["scenario", "model"]
    ):
        promo = g[
            {
                "grocery": "discount",
                "wine": "promo_case_buy",
                "garden": "is_end_of_autumn_clearance",
                "luxury": "discount",
            }[scenario]
        ].gt(0)
        holiday = g.ds.isin(vic_holidays.ds)
        for condition, mask in [
            ("promotion", promo),
            ("non_promotion", ~promo),
            ("holiday", holiday),
            ("ordinary", ~promo & ~holiday),
        ]:
            selected = g.loc[mask]
            if valid_labels(selected).any():
                slices.append(
                    {
                        "scenario": scenario,
                        "model": name,
                        "condition": condition,
                        **metrics(selected, selected.prediction),
                    }
                )
    pd.DataFrame(slices).to_csv(out / "business_slices.csv", index=False)
    for name, digest in INPUT_HASHES.items():
        assert hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() == digest
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "family": family,
                "cutoff": str(cutoff.date()),
                "hashes": INPUT_HASHES,
                "models": names,
                "seed": SEED,
                "windows": WINDOW_STARTS,
                "train": "in-sample hindcast",
                "rolling": "fresh fit per origin; TFT may restore verified origin-specific checkpoint",
                "test": "single final 28-day holdout",
            },
            indent=2,
        )
    )
    return m, p, details, summary, finals


def plot_family(family, names, m, p, summary):
    out = RUNS_DIR / family
    colors = dict(zip(names, plt.get_cmap("tab10").colors))
    fig, axes = plt.subplots(
        1, len(SCENARIOS), figsize=(22, 5), constrained_layout=True
    )
    for ax, scenario in zip(axes, SCENARIOS):
        table = (
            m.loc[(m.scenario == scenario) & m.split.eq("rolling")]
            .pivot(index="regime", columns="model", values="WAPE")
            .reindex(index=["ordinary", "peak", "promotion", "low"], columns=names)
            * 100
        )
        im = ax.imshow(table.to_numpy(), aspect="auto", cmap="YlOrRd")
        ax.set(
            xticks=range(len(names)),
            xticklabels=names,
            yticks=range(4),
            yticklabels=table.index,
            title=scenario + " rolling WAPE (%)",
        )
        ax.tick_params(axis="x", rotation=45)
        for i in range(4):
            for j in range(len(names)):
                ax.text(
                    j,
                    i,
                    f"{table.iloc[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(out / "rolling_heatmap.png", dpi=150)
    plt.show()
    plt.close(fig)
    for scenario in SCENARIOS:
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
        for split, label in [
            ("train_hindcast", "Train in-sample"),
            ("rolling", "Rolling validation"),
            ("test", "Final test"),
        ]:
            if split == "rolling":
                v = summary.loc[summary.scenario.eq(scenario)].set_index("model").WAPE
            else:
                v = (
                    m.loc[(m.scenario == scenario) & m.split.eq(split)]
                    .set_index("model")
                    .WAPE
                )
            axes[0].plot(names, v.reindex(names) * 100, "o-", label=label)
        axes[0].set(title=scenario, ylabel="WAPE (%)")
        axes[0].legend()
        g = p.loc[(p.scenario == scenario) & p.split.eq("test")]
        first = g.loc[g.model == names[0]]
        axes[1].plot(first.ds, first.y, "ko-", ms=3, label="Observed")
        for name in names:
            v = g.loc[g.model == name]
            axes[1].plot(v.ds, v.prediction, label=name, color=colors[name])
        axes[1].set(ylabel="Units", title="Final 28-day test")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].legend(ncol=3)
        fig.savefig(out / f"{scenario}_comparison.png", dpi=150)
        plt.show()
        plt.close(fig)
    return colors
