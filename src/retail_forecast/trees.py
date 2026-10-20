from __future__ import annotations

import json
import pickle
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from .common import *

MODEL_ORDER = ["Decision Tree", "Random Forest", "LightGBM", "XGBoost", "CatBoost"]
MODEL_SLUGS = {
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "LightGBM": "lightgbm",
    "XGBoost": "xgboost",
    "CatBoost": "catboost",
}

# Small deterministic teaching grids expose the main capacity controls without
# turning the tutorial into an expensive AutoML exercise.
PARAMETER_GRIDS = {
    "Decision Tree": [
        {"max_depth": 3, "min_samples_leaf": 20, "ccp_alpha": 0.0},
        {"max_depth": 5, "min_samples_leaf": 10, "ccp_alpha": 0.0},
        {"max_depth": 8, "min_samples_leaf": 5, "ccp_alpha": 0.0},
        {"max_depth": None, "min_samples_leaf": 10, "ccp_alpha": 0.01},
    ],
    "Random Forest": [
        {"max_depth": 6, "min_samples_leaf": 10, "max_features": "sqrt"},
        {"max_depth": 10, "min_samples_leaf": 5, "max_features": 0.7},
        {"max_depth": None, "min_samples_leaf": 5, "max_features": 0.7},
        {"max_depth": None, "min_samples_leaf": 2, "max_features": 1.0},
    ],
    "LightGBM": [
        {
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 20,
            "reg_lambda": 1.0,
        },
        {
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 10,
            "reg_lambda": 1.0,
        },
        {
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 10,
            "reg_lambda": 5.0,
        },
        {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 5,
            "reg_lambda": 1.0,
        },
    ],
    "XGBoost": [
        {
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 10,
            "reg_lambda": 1.0,
        },
        {
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
        },
        {
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_child_weight": 5,
            "reg_lambda": 5.0,
        },
        {
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 2,
            "reg_lambda": 1.0,
        },
    ],
    "CatBoost": [
        {"learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 3.0},
        {"learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 5.0},
        {"learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 5.0},
        {"learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 10.0},
    ],
}


def origin_features(
    history: pd.DataFrame, future: pd.DataFrame, scenario: str
) -> pd.DataFrame:
    """Build direct multi-step features using only target data known at origin."""
    assert history.ds.max() < future.ds.min()
    values = masked_y(history).ffill()
    assert values.tail(28).notna().all()
    X = make_features(future, scenario).copy()
    for label, value in {
        "origin_last": values.iloc[-1],
        "origin_lag7": values.iloc[-8],
        "origin_mean7": values.tail(7).mean(),
        "origin_mean28": values.tail(28).mean(),
        "origin_valid_fraction28": float(valid_labels(history.tail(28)).mean()),
    }.items():
        X[label] = float(value)
    X["horizon_day"] = np.arange(1, len(future) + 1)
    return X.reset_index(drop=True).astype(float)


def origin_training(
    history: pd.DataFrame, scenario: str
) -> tuple[pd.DataFrame, pd.Series]:
    xs, ys = [], []
    for origin in range(ENCODER_LENGTH - 1, len(history) - HORIZON, 7):
        past = history.iloc[: origin + 1]
        future = history.iloc[origin + 1 : origin + 1 + HORIZON]
        valid = valid_labels(future).to_numpy()
        xs.append(origin_features(past, future, scenario).loc[valid])
        ys.append(future.loc[valid, "y"].reset_index(drop=True))
    if not xs:
        raise ValueError("History is too short to construct origin training rows")
    return pd.concat(xs, ignore_index=True), pd.concat(ys, ignore_index=True)


def _new_estimator(model_name: str, params: dict, n_estimators: int | None = None):
    common = {"random_state": SEED}
    if model_name == "Decision Tree":
        return DecisionTreeRegressor(**common, **params)
    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=300, n_jobs=CPU_THREADS, **common, **params
        )
    if model_name == "LightGBM":
        return LGBMRegressor(
            n_estimators=n_estimators or 600,
            n_jobs=CPU_THREADS,
            verbosity=-1,
            **common,
            **params,
        )
    if model_name == "XGBoost":
        return XGBRegressor(
            n_estimators=n_estimators or 600,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=CPU_THREADS,
            eval_metric="mae",
            subsample=1.0,
            colsample_bytree=1.0,
            **common,
            **params,
        )
    if model_name == "CatBoost":
        return CatBoostRegressor(
            iterations=n_estimators or 600,
            loss_function="MAE",
            random_seed=SEED,
            thread_count=CPU_THREADS,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
    raise KeyError(model_name)


def fit_model(history, scenario, model_name, parameter_config):
    """Fit one model; booster early stopping stays strictly inside history."""
    started = time.perf_counter()
    if model_name in {"Decision Tree", "Random Forest"}:
        Xf, yf = origin_training(history, scenario)
        model = _new_estimator(model_name, parameter_config).fit(Xf, yf)
        rounds = getattr(model, "n_estimators", 1)
        internal_mae = np.nan
    else:
        core, validation = history.iloc[:-HORIZON], history.iloc[-HORIZON:]
        assert core.ds.max() < validation.ds.min()
        Xc, yc = origin_training(core, scenario)
        valid = valid_labels(validation).to_numpy()
        Xv = origin_features(core, validation, scenario).loc[valid]
        yv = validation.loc[valid, "y"].reset_index(drop=True)
        candidate = _new_estimator(model_name, parameter_config)
        if model_name == "LightGBM":
            candidate.fit(
                Xc,
                yc,
                eval_set=[(Xv, yv)],
                eval_metric="mae",
                callbacks=[early_stopping(40, verbose=False), log_evaluation(0)],
            )
            rounds = int(candidate.best_iteration_ or 600)
        elif model_name == "XGBoost":
            candidate.set_params(early_stopping_rounds=40)
            candidate.fit(Xc, yc, eval_set=[(Xv, yv)], verbose=False)
            rounds = int(candidate.best_iteration + 1)
        else:
            candidate.fit(Xc, yc, eval_set=(Xv, yv), early_stopping_rounds=40)
            rounds = int(candidate.get_best_iteration() + 1)
        internal_mae = float(np.abs(candidate.predict(Xv) - yv).mean())
        Xf, yf = origin_training(history, scenario)
        model = _new_estimator(model_name, parameter_config, rounds).fit(Xf, yf)
    return {"model": model, "columns": list(Xf.columns)}, {
        "selected_rounds": int(rounds),
        "internal_validation_MAE": internal_mae,
        "training_rows": len(yf),
        "setup_seconds": time.perf_counter() - started,
    }


def predict_model(bundle, history, future, scenario):
    features = origin_features(history, future, scenario)
    changed = future.copy()
    changed["y"] = 12345.0
    pd.testing.assert_frame_equal(features, origin_features(history, changed, scenario))
    assert list(features) == bundle["columns"]
    return bundle["model"].predict(features)


def _prediction_frame(future, raw, scenario, model_name, split, regime, origin):
    result = future.copy()
    result["raw_prediction"] = np.asarray(raw, float)
    result["prediction"] = sales_prediction(result, raw)
    result["scenario"] = scenario
    result["model"] = model_name
    result["split"] = split
    result["regime"] = regime
    result["forecast_origin"] = origin
    result["horizon_day"] = np.arange(1, len(result) + 1)
    return result


def tune_model(scenario: str, model_name: str):
    frame = datasets[scenario]
    candidates, prediction_cache = [], {}
    for candidate_id, params in enumerate(PARAMETER_GRIDS[model_name]):
        fold_frames, seconds, rounds = [], 0.0, []
        for window in windows(scenario)[:-1]:
            origin = window["origin"]
            history = frame.iloc[: origin + 1].copy()
            future = frame.iloc[origin + 1 : origin + 1 + HORIZON].copy()
            bundle, detail = fit_model(history, scenario, model_name, params)
            fold_frames.append(
                _prediction_frame(
                    future,
                    predict_model(bundle, history, future, scenario),
                    scenario,
                    model_name,
                    "validation",
                    window["regime"],
                    history.ds.iloc[-1],
                )
            )
            seconds += detail["setup_seconds"]
            rounds.append(detail["selected_rounds"])
        joined = pd.concat(fold_frames, ignore_index=True)
        score = metrics(joined, joined.prediction)
        candidates.append(
            {
                "scenario": scenario,
                "model": model_name,
                "candidate_id": candidate_id,
                "parameters": json.dumps(params, sort_keys=True),
                "validation_MAE": score["MAE"],
                "validation_RMSE": score["RMSE"],
                "validation_WAPE": score["WAPE"],
                "mean_selected_rounds": float(np.mean(rounds)),
                "tuning_seconds": seconds,
            }
        )
        prediction_cache[candidate_id] = joined
    table = pd.DataFrame(candidates).sort_values(
        ["validation_MAE", "validation_RMSE", "candidate_id"]
    )
    best_id = int(table.iloc[0].candidate_id)
    return PARAMETER_GRIDS[model_name][best_id], table, prediction_cache[best_id]


def _intrinsic_importance(bundle, model_name):
    model = bundle["model"]
    if model_name == "LightGBM":
        return model.booster_.feature_importance(importance_type="gain").astype(float)
    if model_name == "CatBoost":
        return model.get_feature_importance().astype(float)
    return np.asarray(model.feature_importances_, float)


def _permutation_importance(scenario, model_name, params):
    frame = datasets[scenario]
    rng = np.random.default_rng(SEED)
    baseline_error, permuted_error, count = 0.0, None, 0
    for window in windows(scenario)[:-1]:
        origin = window["origin"]
        history = frame.iloc[: origin + 1].copy()
        future = frame.iloc[origin + 1 : origin + 1 + HORIZON].copy()
        bundle, _ = fit_model(history, scenario, model_name, params)
        X = origin_features(history, future, scenario)
        valid = valid_labels(future).to_numpy()
        y = future.loc[valid, "y"].to_numpy(float)
        base = sales_prediction(future, bundle["model"].predict(X))[valid]
        baseline_error += np.abs(base - y).sum()
        count += len(y)
        if permuted_error is None:
            permuted_error = np.zeros(X.shape[1])
        for index, column in enumerate(X):
            shuffled = X.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            prediction = sales_prediction(future, bundle["model"].predict(shuffled))[
                valid
            ]
            permuted_error[index] += np.abs(prediction - y).sum()
    return (permuted_error - baseline_error) / count


def run_model_benchmark(model_name: str):
    """Tune, evaluate and export one model family across all four datasets."""
    slug = MODEL_SLUGS[model_name]
    out = RUNS_DIR / f"tree_{slug}"
    out.mkdir(parents=True, exist_ok=False)
    metric_rows, predictions, tuning_tables = [], [], []
    selected_rows, importance_rows, final_models = [], [], {}
    for scenario, frame in datasets.items():
        print(f"{model_name}: tuning {scenario}", flush=True)
        params, tuning, validation_predictions = tune_model(scenario, model_name)
        tuning_tables.append(tuning)
        selected_rows.append(
            {
                "scenario": scenario,
                "model": model_name,
                "parameters": json.dumps(params, sort_keys=True),
            }
        )
        predictions.append(validation_predictions)
        for regime, group in validation_predictions.groupby("regime", sort=False):
            metric_rows.append(
                {
                    "scenario": scenario,
                    "model": model_name,
                    "split": "validation",
                    "regime": regime,
                    **metrics(group, group.prediction),
                }
            )

        origin = windows(scenario)[-1]["origin"]
        history = frame.iloc[: origin + 1].copy()
        future = frame.iloc[origin + 1 : origin + 1 + HORIZON].copy()
        final_bundle, detail = fit_model(history, scenario, model_name, params)
        final_models[scenario] = final_bundle
        test = _prediction_frame(
            future,
            predict_model(final_bundle, history, future, scenario),
            scenario,
            model_name,
            "test",
            "final",
            history.ds.iloc[-1],
        )
        predictions.append(test)
        metric_rows.append(
            {
                "scenario": scenario,
                "model": model_name,
                "split": "test",
                "regime": "final",
                **metrics(test, test.prediction),
                **detail,
            }
        )

        blocks = []
        for start in training_blocks():
            past = frame.iloc[:start].copy()
            block = frame.iloc[start : start + HORIZON].copy()
            blocks.append(
                _prediction_frame(
                    block,
                    predict_model(final_bundle, past, block, scenario),
                    scenario,
                    model_name,
                    "train_hindcast",
                    "training",
                    past.ds.iloc[-1],
                )
            )
        hindcast = pd.concat(blocks, ignore_index=True)
        predictions.append(hindcast)
        metric_rows.append(
            {
                "scenario": scenario,
                "model": model_name,
                "split": "train_hindcast",
                "regime": "training",
                **metrics(hindcast, hindcast.prediction),
            }
        )

        native = _intrinsic_importance(final_bundle, model_name)
        permutation = _permutation_importance(scenario, model_name, params)
        for feature, intrinsic, comparable in zip(
            final_bundle["columns"], native, permutation
        ):
            importance_rows.extend(
                [
                    {
                        "scenario": scenario,
                        "model": model_name,
                        "feature": feature,
                        "importance_type": "native",
                        "importance": float(intrinsic),
                    },
                    {
                        "scenario": scenario,
                        "model": model_name,
                        "feature": feature,
                        "importance_type": "permutation_MAE_increase",
                        "importance": float(comparable),
                    },
                ]
            )
        with (out / f"{scenario}_model.pkl").open("wb") as handle:
            pickle.dump(final_bundle, handle)

    m = pd.DataFrame(metric_rows)
    p = pd.concat(predictions, ignore_index=True)
    tuning = pd.concat(tuning_tables, ignore_index=True)
    selected = pd.DataFrame(selected_rows)
    importance = pd.DataFrame(importance_rows)
    m.to_csv(out / "metrics.csv", index=False)
    p.to_csv(out / "predictions.csv", index=False)
    tuning.to_csv(out / "tuning_results.csv", index=False)
    selected.to_csv(out / "selected_parameters.csv", index=False)
    importance.to_csv(out / "feature_importance.csv", index=False)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "slug": slug,
                "cutoff": str(cutoff.date()),
                "test_dates": [
                    str((cutoff + pd.Timedelta(days=1)).date()),
                    str(datasets["grocery"].ds.max().date()),
                ],
                "hashes": INPUT_HASHES,
                "seed": SEED,
                "selection_metric": "pooled rolling validation MAE",
                "feature_protocol": "origin-frozen history plus known-future calendar/business features",
            },
            indent=2,
        )
    )
    for name, digest in INPUT_HASHES.items():
        assert hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() == digest
    return m, p, tuning, selected, importance, final_models, out


def seasonal_naive_diagnostics():
    """Return the common non-fitted reference on validation and test windows."""
    rows = []
    for scenario, frame in datasets.items():
        for window in windows(scenario):
            origin = window["origin"]
            history = frame.iloc[: origin + 1]
            future = frame.iloc[origin + 1 : origin + 1 + HORIZON].copy()
            future["prediction"] = sales_prediction(
                future, seasonal_naive(history, future)
            )
            rows.append(
                {
                    "scenario": scenario,
                    "model": "Seasonal naive",
                    "split": "test" if window["split"] == "test" else "validation",
                    "regime": window["regime"],
                    **metrics(future, future.prediction),
                }
            )
    return pd.DataFrame(rows)


def plot_model_benchmark(model_name, metrics_table, predictions, importance, out):
    fig, axes = plt.subplots(
        1, len(SCENARIOS), figsize=(20, 4), constrained_layout=True
    )
    for ax, scenario in zip(axes, SCENARIOS):
        table = (
            metrics_table.loc[
                (metrics_table.scenario == scenario)
                & metrics_table.split.eq("validation")
            ]
            .set_index("regime")
            .reindex(["ordinary", "peak", "promotion", "low"])[["WAPE"]]
            * 100
        )
        image = ax.imshow(table.to_numpy(), aspect="auto", cmap="YlOrRd")
        ax.set(
            title=f"{scenario} validation WAPE (%)",
            yticks=range(4),
            yticklabels=table.index,
            xticks=[0],
            xticklabels=[model_name],
        )
        for row, value in enumerate(table.WAPE):
            ax.text(
                0,
                row,
                "NA" if pd.isna(value) else f"{value:.1f}",
                ha="center",
                va="center",
            )
        fig.colorbar(image, ax=ax, shrink=0.7)
    fig.savefig(out / "validation_heatmap.png", dpi=150)
    plt.show()
    plt.close(fig)

    for scenario in SCENARIOS:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
        group = metrics_table.loc[metrics_table.scenario.eq(scenario)]
        summary = [
            group.loc[group.split.eq(split), "WAPE"].mean() * 100
            for split in ["train_hindcast", "validation", "test"]
        ]
        axes[0].plot(["Train hindcast", "Validation", "Test"], summary, "o-")
        axes[0].set(title=f"{scenario} — {model_name}", ylabel="WAPE (%)")
        test = predictions.loc[
            (predictions.scenario == scenario) & predictions.split.eq("test")
        ]
        axes[1].plot(test.ds, test.y, "ko-", ms=3, label="Observed")
        axes[1].plot(test.ds, test.prediction, label=model_name)
        axes[1].set(title="Final 28-day test", ylabel="Units")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].legend()
        fig.savefig(out / f"{scenario}_performance.png", dpi=150)
        plt.show()
        plt.close(fig)

        top = (
            importance.loc[
                (importance.scenario == scenario)
                & importance.importance_type.eq("permutation_MAE_increase")
            ]
            .nlargest(12, "importance")
            .sort_values("importance")
        )
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.barh(top.feature, top.importance)
        ax.set(
            title=f"{scenario} — {model_name}",
            xlabel="Validation MAE increase after permutation",
        )
        fig.savefig(out / f"{scenario}_importance.png", dpi=150)
        plt.show()
        plt.close(fig)


def _compatible_result_directory(model_name: str) -> Path:
    slug = MODEL_SLUGS[model_name]
    candidates = sorted(
        RUNS_DIR.parent.glob(f"*/tree_{slug}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    required = [
        "metrics.csv",
        "predictions.csv",
        "feature_importance.csv",
        "selected_parameters.csv",
        "tuning_results.csv",
    ]
    for path in candidates:
        manifest_path = path / "manifest.json"
        if (
            manifest_path.is_file()
            and all((path / name).is_file() for name in required)
            and json.loads(manifest_path.read_text()).get("hashes") == INPUT_HASHES
        ):
            return path
    raise FileNotFoundError(
        f"No hash-compatible run found for {model_name}; execute its notebook first"
    )


def load_comparison_results():
    tables = {
        key: []
        for key in ["metrics", "predictions", "importance", "parameters", "tuning"]
    }
    sources = {}
    for model_name in MODEL_ORDER:
        path = _compatible_result_directory(model_name)
        sources[model_name] = path
        tables["metrics"].append(pd.read_csv(path / "metrics.csv"))
        tables["predictions"].append(
            pd.read_csv(path / "predictions.csv", parse_dates=["ds", "forecast_origin"])
        )
        tables["importance"].append(pd.read_csv(path / "feature_importance.csv"))
        tables["parameters"].append(pd.read_csv(path / "selected_parameters.csv"))
        tables["tuning"].append(pd.read_csv(path / "tuning_results.csv"))
    return {
        key: pd.concat(parts, ignore_index=True) for key, parts in tables.items()
    }, sources


def export_comparison(tables: dict):
    out = RUNS_DIR / "tree_comparison"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    metrics_table = tables["metrics"]
    aggregate = metrics_table.groupby(
        ["scenario", "model", "split"], as_index=False
    ).agg(
        MAE=("MAE", "mean"),
        RMSE=("RMSE", "mean"),
        WAPE=("WAPE", "mean"),
        Bias=("Bias", "mean"),
        n=("n", "sum"),
    )
    final_fit = (
        metrics_table.loc[metrics_table.split.eq("test")]
        .set_index(["scenario", "model"])
        .setup_seconds.rename("final_fit_seconds")
        .reset_index()
    )
    tuning_time = (
        tables["tuning"]
        .groupby(["scenario", "model"], as_index=False)
        .tuning_seconds.sum()
        .rename(columns={"tuning_seconds": "total_tuning_seconds"})
    )
    aggregate = aggregate.merge(final_fit, on=["scenario", "model"], how="left")
    aggregate = aggregate.merge(tuning_time, on=["scenario", "model"], how="left")
    train = (
        aggregate.loc[aggregate.split.eq("train_hindcast")]
        .set_index(["scenario", "model"])
        .WAPE
    )
    test = (
        aggregate.loc[aggregate.split.eq("test")].set_index(["scenario", "model"]).WAPE
    )
    gap = (test - train).rename("test_minus_train_WAPE").reset_index()
    aggregate = aggregate.merge(gap, on=["scenario", "model"], how="left")
    aggregate.to_csv(out / "model_comparison.csv", index=False)
    tables["importance"].to_csv(out / "feature_importance.csv", index=False)
    tables["parameters"].to_csv(out / "selected_parameters.csv", index=False)
    (out / "manifest.json").write_text(
        json.dumps(
            {"models": MODEL_ORDER, "hashes": INPUT_HASHES, "seed": SEED}, indent=2
        )
    )
    return aggregate, out


def plot_comparison(tables: dict, aggregate: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(
        1, len(SCENARIOS), figsize=(22, 5), constrained_layout=True
    )
    for ax, scenario in zip(axes, SCENARIOS):
        table = (
            aggregate.loc[
                (aggregate.scenario == scenario)
                & aggregate.split.isin(["train_hindcast", "validation", "test"])
            ]
            .pivot(index="split", columns="model", values="WAPE")
            .reindex(
                index=["train_hindcast", "validation", "test"], columns=MODEL_ORDER
            )
            * 100
        )
        image = ax.imshow(table.to_numpy(), aspect="auto", cmap="YlOrRd")
        ax.set(
            title=scenario,
            xticks=range(len(MODEL_ORDER)),
            xticklabels=MODEL_ORDER,
            yticks=range(3),
            yticklabels=["Train", "Validation", "Test"],
        )
        ax.tick_params(axis="x", rotation=45)
        for row in range(3):
            for col in range(len(MODEL_ORDER)):
                value = table.iloc[row, col]
                ax.text(
                    col,
                    row,
                    "NA" if pd.isna(value) else f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        fig.colorbar(image, ax=ax, shrink=0.7)
    fig.savefig(out / "five_model_metrics.png", dpi=150)
    plt.show()
    plt.close(fig)

    predictions = tables["predictions"]
    for scenario in SCENARIOS:
        fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
        group = predictions.loc[
            (predictions.scenario == scenario) & predictions.split.eq("test")
        ]
        first = group.loc[group.model.eq(MODEL_ORDER[0])]
        ax.plot(first.ds, first.y, "ko-", ms=3, label="Observed")
        for model_name in MODEL_ORDER:
            values = group.loc[group.model.eq(model_name)]
            ax.plot(values.ds, values.prediction, label=model_name)
        ax.set(title=f"{scenario} final 28-day test", ylabel="Units")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(ncol=3)
        fig.savefig(out / f"{scenario}_test_comparison.png", dpi=150)
        plt.show()
        plt.close(fig)

    importance = tables["importance"]
    for scenario in SCENARIOS:
        group = importance.loc[
            (importance.scenario == scenario)
            & importance.importance_type.eq("permutation_MAE_increase")
        ]
        features = group.groupby("feature").importance.mean().nlargest(12).index
        table = (
            group.loc[group.feature.isin(features)]
            .pivot(index="feature", columns="model", values="importance")
            .reindex(columns=MODEL_ORDER)
            .fillna(0)
        )
        table = table.loc[table.mean(axis=1).sort_values().index]
        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        table.plot.barh(ax=ax)
        ax.set(
            title=f"{scenario} comparable feature importance",
            xlabel="Validation MAE increase after permutation",
        )
        fig.savefig(out / f"{scenario}_importance_comparison.png", dpi=150)
        plt.show()
        plt.close(fig)
