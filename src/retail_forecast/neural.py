import json
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import *
from .tft import fit_tft, predict_tft, restore_tft

NAMES = ["MLP", "LSTM", "TFT", "Seasonal naive"]
NEURAL_MAX_EPOCHS = 35
NEURAL_BATCH_SIZE = 64
WINDOW_STRIDE = 7
TFT_MODE = "train"  # Historical checkpoints are opt-in because their manifests use an older data snapshot.
torch.set_num_threads(CPU_THREADS)


class WindowNetwork(nn.Module):
    def __init__(self, kind, features, hidden=32):
        super().__init__()
        self.kind = kind
        self.encoder = (
            nn.Sequential(nn.Linear(ENCODER_LENGTH, hidden), nn.ReLU())
            if kind == "MLP"
            else nn.LSTM(features + 1, hidden, batch_first=True)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + features, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, past, future):
        if self.kind == "MLP":
            context = self.encoder(past[:, :, 0])
        else:
            _, (h, _) = self.encoder(past)
            context = h[-1]
        context = context[:, None, :].expand(-1, future.shape[1], -1)
        return self.head(torch.cat([context, future], dim=-1)).squeeze(-1)


def preprocessor(history, scenario):
    y = history.loc[valid_labels(history), "y"]
    mean = float(y.mean())
    std = max(float(y.std()), 1.0)
    return {
        "mean": mean,
        "std": std,
        "scaler": StandardScaler().fit(make_features(history, scenario)),
    }


def inputs(history, future, scenario, prep):
    y = (
        masked_y(history)
        .ffill()
        .fillna(prep["mean"])
        .tail(ENCODER_LENGTH)
        .to_numpy(float)
    )
    assert len(y) == ENCODER_LENGTH
    p = prep["scaler"].transform(make_features(history.tail(ENCODER_LENGTH), scenario))
    f = prep["scaler"].transform(make_features(future, scenario))
    return np.column_stack([(y - prep["mean"]) / prep["std"], p]).astype(
        "float32"
    ), f.astype("float32")


def training_tensors(history, scenario, prep):
    past = []
    future = []
    targets = []
    weights = []
    for origin in range(ENCODER_LENGTH - 1, len(history) - HORIZON, WINDOW_STRIDE):
        h = history.iloc[: origin + 1]
        f = history.iloc[origin + 1 : origin + 1 + HORIZON]
        a, b = inputs(h, f, scenario, prep)
        past.append(a)
        future.append(b)
        targets.append(
            ((f.y.to_numpy(float) - prep["mean"]) / prep["std"]).astype("float32")
        )
        weights.append(valid_labels(f).to_numpy(dtype="float32"))
    return tuple(
        torch.from_numpy(np.stack(a)) for a in [past, future, targets, weights]
    )


def masked_mse(p, y, w):
    return (((p - y) ** 2) * w).sum() / w.sum().clamp_min(1.0)


def train_epochs(
    model, tensors, epochs, validation=None, learning_rate=0.003, weight_decay=0.001
):
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=NEURAL_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    logs = []
    best = float("inf")
    best_epoch = 1
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for p, f, y, w in loader:
            opt.zero_grad()
            loss = masked_mse(model(p, f), y, w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if validation is not None:
            model.eval()
            with torch.no_grad():
                v = float(
                    masked_mse(
                        model(validation[0], validation[1]),
                        validation[2],
                        validation[3],
                    )
                )
            row["validation_loss"] = v
            if v < best:
                best = v
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
        logs.append(row)
        if validation is not None and stale >= 8:
            break
    return best_epoch, logs


def fit_neural(history, scenario, name, window):
    if name == "TFT":
        if TFT_MODE == "checkpoint":
            return restore_tft(history, scenario, window["regime"])
        bundle, details = fit_tft(
            history,
            scenario,
            train_end=len(history) - 1,
            checkpoint_tag=f"{scenario}_{window['regime']}",
        )
        details["mode"] = "retrained"
        return bundle, details
    core = history.iloc[:-HORIZON]
    v = history.iloc[-HORIZON:]
    prep = preprocessor(core, scenario)
    train = training_tensors(core, scenario, prep)
    a, b = inputs(core, v, scenario, prep)
    y = ((v.y.to_numpy(float) - prep["mean"]) / prep["std"]).astype("float32")
    w = valid_labels(v).to_numpy(dtype="float32")
    validation = tuple(torch.from_numpy(x)[None] for x in [a, b, y, w])
    torch.manual_seed(SEED)
    candidate = WindowNetwork(name, len(feature_names[scenario]))
    epochs, selection = train_epochs(candidate, train, NEURAL_MAX_EPOCHS, validation)
    prep = preprocessor(history, scenario)
    torch.manual_seed(SEED)
    model = WindowNetwork(name, len(feature_names[scenario]))
    _, _final_log = train_epochs(
        model, training_tensors(history, scenario, prep), epochs
    )
    model.eval()
    out = RUNS_DIR / "neural"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{scenario}_{window['regime']}_{name}"
    pd.DataFrame(selection).to_csv(out / f"{tag}_selection.csv", index=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "kind": name,
            "feature_count": len(feature_names[scenario]),
            "mean": prep["mean"],
            "std": prep["std"],
            "scaler_mean": prep["scaler"].mean_.tolist(),
            "scaler_scale": prep["scaler"].scale_.tolist(),
            "feature_names": feature_names[scenario],
        },
        out / f"{tag}.pt",
    )
    return {"model": model, "prep": prep}, {
        "selected_epochs": epochs,
        "mode": "new_training",
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_windows": len(train[0]),
        "validation_MSE_scaled": min(r["validation_loss"] for r in selection),
    }


def predict_neural(bundle, history, future, scenario, name):
    if name == "TFT":
        f = pd.concat([history, future])
        origin = len(history) - 1
        raw = predict_tft(bundle, f, scenario, origin)
        changed = predict_tft(bundle, f, scenario, origin, placeholder=12345.0)
        np.testing.assert_allclose(raw, changed, rtol=1e-6, atol=1e-5)
        return raw
    a, b = inputs(history, future, scenario, bundle["prep"])
    changed = future.copy()
    changed["y"] = 12345.0
    ca, cb = inputs(history, changed, scenario, bundle["prep"])
    np.testing.assert_array_equal(a, ca)
    np.testing.assert_array_equal(b, cb)
    with torch.no_grad():
        raw = (
            bundle["model"](torch.from_numpy(a)[None], torch.from_numpy(b)[None])
            .numpy()
            .reshape(-1)
        )
    return raw * bundle["prep"]["std"] + bundle["prep"]["mean"]


def neural_curves():
    out = RUNS_DIR / "neural"
    fig, axes = plt.subplots(
        1, len(SCENARIOS), figsize=(20, 4), constrained_layout=True
    )
    for ax, scenario in zip(axes, SCENARIOS):
        for kind in ["MLP", "LSTM"]:
            g = pd.read_csv(out / f"{scenario}_final_{kind}_selection.csv")
            ax.plot(g.epoch, np.sqrt(g.validation_loss), label=kind + " validation")
        ax.set(title=scenario, xlabel="Epoch", ylabel="Scaled validation RMSE")
        ax.legend()
    fig.savefig(out / "learning_curves.png", dpi=150)
    plt.show()
    plt.close(fig)


NEURAL_MODEL_ORDER = ["MLP", "LSTM", "TFT"]
NEURAL_CANDIDATES = {
    "MLP": [
        {"hidden": 16, "learning_rate": 0.001, "weight_decay": 0.001},
        {"hidden": 32, "learning_rate": 0.001, "weight_decay": 0.001},
        {"hidden": 16, "learning_rate": 0.003, "weight_decay": 0.0001},
        {"hidden": 32, "learning_rate": 0.003, "weight_decay": 0.001},
    ],
    "LSTM": [
        {"hidden": 16, "learning_rate": 0.001, "weight_decay": 0.001},
        {"hidden": 32, "learning_rate": 0.001, "weight_decay": 0.001},
        {"hidden": 16, "learning_rate": 0.003, "weight_decay": 0.0001},
        {"hidden": 32, "learning_rate": 0.003, "weight_decay": 0.001},
    ],
    "TFT": [
        {
            "hidden_size": 8,
            "attention_head_size": 1,
            "hidden_continuous_size": 8,
            "dropout": 0.1,
            "learning_rate": 0.005,
            "max_epochs": 12,
        },
        {
            "hidden_size": 16,
            "attention_head_size": 2,
            "hidden_continuous_size": 8,
            "dropout": 0.1,
            "learning_rate": 0.005,
            "max_epochs": 12,
        },
        {
            "hidden_size": 16,
            "attention_head_size": 2,
            "hidden_continuous_size": 8,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "max_epochs": 12,
        },
    ],
}


def fit_neural_config(history, scenario, model_name, config, checkpoint_tag):
    started = time.perf_counter()
    if model_name == "TFT":
        bundle, detail = fit_tft(
            history,
            scenario,
            train_end=len(history) - 1,
            checkpoint_tag=checkpoint_tag,
            config=config,
        )
        detail["setup_seconds"] = time.perf_counter() - started
        detail["parameters"] = sum(p.numel() for p in bundle["model"].parameters())
        return bundle, detail

    core, validation_frame = history.iloc[:-HORIZON], history.iloc[-HORIZON:]
    prep = preprocessor(core, scenario)
    train = training_tensors(core, scenario, prep)
    past, future = inputs(core, validation_frame, scenario, prep)
    target = ((validation_frame.y.to_numpy(float) - prep["mean"]) / prep["std"]).astype(
        "float32"
    )
    weight = valid_labels(validation_frame).to_numpy(dtype="float32")
    validation = tuple(
        torch.from_numpy(value)[None] for value in [past, future, target, weight]
    )
    torch.manual_seed(SEED)
    candidate = WindowNetwork(
        model_name, len(feature_names[scenario]), hidden=config["hidden"]
    )
    epochs, selection = train_epochs(
        candidate,
        train,
        NEURAL_MAX_EPOCHS,
        validation,
        config["learning_rate"],
        config["weight_decay"],
    )
    prep = preprocessor(history, scenario)
    torch.manual_seed(SEED)
    model = WindowNetwork(
        model_name, len(feature_names[scenario]), hidden=config["hidden"]
    )
    train_epochs(
        model,
        training_tensors(history, scenario, prep),
        epochs,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    model.eval()
    return {"model": model, "prep": prep}, {
        "selected_epochs": epochs,
        "validation_MSE_scaled": min(row["validation_loss"] for row in selection),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_windows": len(train[0]),
        "setup_seconds": time.perf_counter() - started,
    }


def predict_neural_config(bundle, history, future, scenario, model_name):
    if model_name == "TFT":
        full = pd.concat([history, future])
        origin = len(history) - 1
        raw = predict_tft(bundle, full, scenario, origin)
        changed = predict_tft(bundle, full, scenario, origin, placeholder=12345.0)
        np.testing.assert_allclose(raw, changed, rtol=1e-6, atol=1e-5)
        return raw
    return predict_neural(bundle, history, future, scenario, model_name)


def _neural_prediction_frame(future, raw, scenario, model_name, split, regime, origin):
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


def tune_neural(scenario, model_name):
    frame = datasets[scenario]
    rows, cached = [], {}
    for candidate_id, config in enumerate(NEURAL_CANDIDATES[model_name]):
        predictions, elapsed, epochs = [], 0.0, []
        for window in windows(scenario)[:-1]:
            origin = window["origin"]
            history = frame.iloc[: origin + 1].copy()
            future = frame.iloc[origin + 1 : origin + 1 + HORIZON].copy()
            tag = f"tune_{model_name.lower()}_{scenario}_{candidate_id}_{window['regime']}"
            bundle, detail = fit_neural_config(
                history, scenario, model_name, config, tag
            )
            raw = predict_neural_config(bundle, history, future, scenario, model_name)
            predictions.append(
                _neural_prediction_frame(
                    future,
                    raw,
                    scenario,
                    model_name,
                    "validation",
                    window["regime"],
                    history.ds.iloc[-1],
                )
            )
            elapsed += detail["setup_seconds"]
            epochs.append(detail["selected_epochs"])
        joined = pd.concat(predictions, ignore_index=True)
        score = metrics(joined, joined.prediction)
        rows.append(
            {
                "scenario": scenario,
                "model": model_name,
                "candidate_id": candidate_id,
                "parameters": json.dumps(config, sort_keys=True),
                "validation_MAE": score["MAE"],
                "validation_RMSE": score["RMSE"],
                "validation_WAPE": score["WAPE"],
                "mean_selected_epochs": float(np.mean(epochs)),
                "tuning_seconds": elapsed,
            }
        )
        cached[candidate_id] = joined
    table = pd.DataFrame(rows).sort_values(
        ["validation_MAE", "validation_RMSE", "candidate_id"]
    )
    best_id = int(table.iloc[0].candidate_id)
    return NEURAL_CANDIDATES[model_name][best_id], table, cached[best_id]


def _neural_permutation_sensitivity(scenario, model_name, config):
    frame = datasets[scenario]
    rng = np.random.default_rng(SEED)
    baseline_error, shuffled_error, count, columns = 0.0, None, 0, None
    for window in windows(scenario)[:-1]:
        origin = window["origin"]
        history = frame.iloc[: origin + 1].copy()
        future = frame.iloc[origin + 1 : origin + 1 + HORIZON].copy()
        tag = f"importance_{model_name.lower()}_{scenario}_{window['regime']}"
        bundle, _ = fit_neural_config(history, scenario, model_name, config, tag)
        raw = predict_neural_config(bundle, history, future, scenario, model_name)
        valid = valid_labels(future).to_numpy()
        target = future.loc[valid, "y"].to_numpy(float)
        baseline_error += np.abs(sales_prediction(future, raw)[valid] - target).sum()
        count += len(target)
        columns = ["calendar_date", *SCENARIOS[scenario]["features"]]
        if shuffled_error is None:
            shuffled_error = np.zeros(len(columns))
        for index, column in enumerate(columns):
            changed = future.copy()
            if column != "calendar_date":
                changed[column] = rng.permutation(changed[column].to_numpy())
            else:
                # Calendar/Fourier/holiday inputs are jointly derived from ds.
                changed["ds"] = rng.permutation(changed.ds.to_numpy())
            prediction = predict_neural_config(
                bundle, history, changed, scenario, model_name
            )
            shuffled_error[index] += np.abs(
                sales_prediction(future, prediction)[valid] - target
            ).sum()
    return columns, (shuffled_error - baseline_error) / count


def run_neural_benchmark(model_name):
    slug = model_name.lower()
    out = RUNS_DIR / f"neural_{slug}"
    out.mkdir(parents=True, exist_ok=False)
    metrics_rows, prediction_tables, tuning_tables = [], [], []
    selected_rows, fit_rows, sensitivity_rows = [], [], []
    for scenario, frame in datasets.items():
        print(f"{model_name}: tuning {scenario}", flush=True)
        config, tuning, validation = tune_neural(scenario, model_name)
        tuning_tables.append(tuning)
        selected_rows.append(
            {
                "scenario": scenario,
                "model": model_name,
                "parameters": json.dumps(config, sort_keys=True),
            }
        )
        prediction_tables.append(validation)
        for regime, group in validation.groupby("regime", sort=False):
            metrics_rows.append(
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
        bundle, detail = fit_neural_config(
            history, scenario, model_name, config, f"final_{slug}_{scenario}"
        )
        fit_rows.append({"scenario": scenario, "model": model_name, **detail})
        test = _neural_prediction_frame(
            future,
            predict_neural_config(bundle, history, future, scenario, model_name),
            scenario,
            model_name,
            "test",
            "final",
            history.ds.iloc[-1],
        )
        prediction_tables.append(test)
        metrics_rows.append(
            {
                "scenario": scenario,
                "model": model_name,
                "split": "test",
                "regime": "final",
                **metrics(test, test.prediction),
            }
        )
        blocks = []
        for start in training_blocks():
            past = frame.iloc[:start].copy()
            block = frame.iloc[start : start + HORIZON].copy()
            blocks.append(
                _neural_prediction_frame(
                    block,
                    predict_neural_config(bundle, past, block, scenario, model_name),
                    scenario,
                    model_name,
                    "train_hindcast",
                    "training",
                    past.ds.iloc[-1],
                )
            )
        hindcast = pd.concat(blocks, ignore_index=True)
        prediction_tables.append(hindcast)
        metrics_rows.append(
            {
                "scenario": scenario,
                "model": model_name,
                "split": "train_hindcast",
                "regime": "training",
                **metrics(hindcast, hindcast.prediction),
            }
        )
        columns, sensitivity = _neural_permutation_sensitivity(
            scenario, model_name, config
        )
        sensitivity_rows.extend(
            {
                "scenario": scenario,
                "model": model_name,
                "feature": feature,
                "importance_type": "permutation_MAE_increase",
                "importance": float(value),
            }
            for feature, value in zip(columns, sensitivity)
        )
        if model_name == "TFT":
            torch.save(
                {"state_dict": bundle["model"].state_dict(), "config": config},
                out / f"{scenario}_model.ckpt",
            )
        else:
            torch.save(bundle["model"].state_dict(), out / f"{scenario}_model.pt")
    tables = {
        "metrics": pd.DataFrame(metrics_rows),
        "predictions": pd.concat(prediction_tables, ignore_index=True),
        "tuning_results": pd.concat(tuning_tables, ignore_index=True),
        "selected_parameters": pd.DataFrame(selected_rows),
        "fit_details": pd.DataFrame(fit_rows),
        "explanations": pd.DataFrame(sensitivity_rows),
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "family": f"neural_{slug}",
                "model": model_name,
                "scenarios": list(SCENARIOS),
                "protocol_version": "retail-benchmark-v2",
                "hashes": INPUT_HASHES,
                "seed": SEED,
                "selection_metric": "pooled rolling validation MAE",
            },
            indent=2,
        )
    )
    return (*tables.values(), out)
