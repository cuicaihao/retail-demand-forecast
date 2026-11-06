"""Strict V4 model interfaces, metrics, persistence, and aggregation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import INPUT_HASHES, ROOT, SEED, datasets
from .history_comparison import build_training_history_comparison

BENCHMARK_DIR = ROOT / "data" / "benchmarks"
PUBLISHED_DIR = BENCHMARK_DIR / "published"


@dataclass
class ForecastOutput:
    """Standard output from one fitted adapter."""

    point_forecast: np.ndarray
    q10: np.ndarray | None = None
    q90: np.ndarray | None = None
    latency_ms: float = 0.0
    extra_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Declared model identity and comparison role."""

    model_id: str
    model_name: str
    family: str
    toolkit: str = "unknown"
    variant: str = "default"
    comparison_role: str = "ablation"
    config_group: str = "default"
    adapter_factory: (
        Callable[[ModelConfig, str, dict[str, Any]], BaseRetailForecaster] | None
    ) = field(default=None, repr=False, compare=False)
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""


class BaseRetailForecaster(ABC):
    """Interface implemented by every V4 model adapter."""

    def __init__(self, name: str, family: str):
        self.name = name
        self.family = family

    @abstractmethod
    def fit(self, history: pd.DataFrame, scenario: str) -> None:
        """Fit using observations available at the forecast origin."""

    @abstractmethod
    def predict(self, future: pd.DataFrame, scenario: str) -> ForecastOutput:
        """Predict without reading future targets or diagnostic ground truth."""


def compute_standard_metrics(
    y_true: np.ndarray | pd.Series | list[float],
    y_pred: np.ndarray | pd.Series | list[float],
    q10: np.ndarray | pd.Series | None = None,
    q90: np.ndarray | pd.Series | None = None,
    seasonal_scale: float | None = None,
    event_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute pooled retail and intermittent-demand metrics."""
    y = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y) != len(prediction):
        raise ValueError(f"Length mismatch: {len(y)} != {len(prediction)}")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require finite observations and predictions")
    n = len(y)
    if n == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "WAPE": np.nan,
            "Bias": np.nan,
            "WAPE_Denominator": 0.0,
            "MASE": np.nan,
            "ZeroRecallAt05": np.nan,
            "DemandRecallAt05": np.nan,
            "DemandEventF1At05": np.nan,
            "PositiveDemandMAE": np.nan,
            "Coverage80": np.nan,
            "MeanIntervalWidth80": np.nan,
            "PinballLoss80": np.nan,
            "n": 0,
        }

    error = prediction - y
    absolute_error = np.abs(error)
    denominator = float(y.sum())
    mae = float(absolute_error.mean())
    zero_mask = y == 0
    demand_mask = y > 0
    predicted_demand = prediction >= event_threshold
    true_positive = int((predicted_demand & demand_mask).sum())
    false_positive = int((predicted_demand & zero_mask).sum())
    false_negative = int(((~predicted_demand) & demand_mask).sum())
    f1_denominator = 2 * true_positive + false_positive + false_negative

    coverage = np.nan
    interval_width = np.nan
    pinball = np.nan
    if (q10 is None) != (q90 is None):
        raise ValueError("q10 and q90 must be supplied together")
    if q10 is not None and q90 is not None:
        low = np.asarray(q10, dtype=float).reshape(-1)
        high = np.asarray(q90, dtype=float).reshape(-1)
        if len(low) != n or len(high) != n:
            raise ValueError("Prediction interval length does not match observations")
        if (high < low).any():
            raise ValueError("q90 must be greater than or equal to q10")
        coverage = float(((y >= low) & (y <= high)).mean())
        interval_width = float((high - low).mean())
        low_error = y - low
        high_error = y - high
        low_loss = np.maximum(0.1 * low_error, -0.9 * low_error)
        high_loss = np.maximum(0.9 * high_error, -0.1 * high_error)
        pinball = float((low_loss.mean() + high_loss.mean()) / 2)

    return {
        "MAE": mae,
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "WAPE": float(absolute_error.sum() / denominator)
        if denominator > 0
        else np.nan,
        "Bias": float(error.sum() / denominator) if denominator > 0 else np.nan,
        "WAPE_Denominator": denominator,
        "MASE": float(mae / seasonal_scale)
        if seasonal_scale is not None and seasonal_scale > 0
        else np.nan,
        "ZeroRecallAt05": float((~predicted_demand[zero_mask]).mean())
        if zero_mask.any()
        else np.nan,
        "DemandRecallAt05": float(predicted_demand[demand_mask].mean())
        if demand_mask.any()
        else np.nan,
        "DemandEventF1At05": float(2 * true_positive / f1_denominator)
        if f1_denominator
        else np.nan,
        "PositiveDemandMAE": float(absolute_error[demand_mask].mean())
        if demand_mask.any()
        else np.nan,
        "Coverage80": coverage,
        "MeanIntervalWidth80": interval_width,
        "PinballLoss80": pinball,
        "n": n,
    }


V4_PREDICTION_COLUMNS = {
    "ds",
    "y",
    "is_open",
    "stockout",
    "scenario",
    "model_id",
    "fold",
    "period",
    "forecast_origin",
    "prediction",
    "q10",
    "q90",
}
V4_PERIOD_COLUMNS = {
    "scenario",
    "model_id",
    "model_name",
    "toolkit",
    "family",
    "variant",
    "comparison_role",
    "fold",
    "period",
    "test_start",
    "test_end",
    "WAPE",
    "WAPE_Denominator",
    "n",
}


def benchmark_result_root(base_dir: Path, benchmark_id: str) -> Path:
    return Path(base_dir) / benchmark_id


def model_result_dir(base_dir: Path, benchmark_id: str, model_id: str) -> Path:
    return benchmark_result_root(base_dir, benchmark_id) / "models" / model_id


def _git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _source_tree_sha256() -> str:
    """Hash benchmark code and environment declarations, including dirty files."""
    paths = [
        *sorted((ROOT / "src" / "retail_forecast").rglob("*.py")),
        ROOT / "scripts" / "run_benchmark.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def current_source_provenance() -> dict[str, Any]:
    """Describe the exact source state used by a local benchmark execution."""
    status = _git_output("status", "--porcelain", "--untracked-files=normal")
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": status not in {"", "unknown"},
        "source_tree_sha256": _source_tree_sha256(),
    }


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _software_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for distribution in (
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "catboost",
        "prophet",
        "autogluon-timeseries",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _valid_label_provenance(predictions: pd.DataFrame) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    for scenario, scenario_rows in predictions.groupby("scenario", sort=True):
        fold_records = {}
        all_dates: list[str] = []
        for fold, rows in scenario_rows.groupby("fold", sort=True):
            valid = (
                rows.is_open.astype(bool) & ~rows.stockout.astype(bool) & rows.y.notna()
            )
            dates = rows.loc[valid, "ds"].dt.strftime("%Y-%m-%d").tolist()
            all_dates.extend(dates)
            fold_records[str(fold)] = {
                "count": len(dates),
                "dates_sha256": _json_sha256(dates),
            }
        provenance[scenario] = {
            "count": len(all_dates),
            "dates_sha256": _json_sha256(all_dates),
            "folds": fold_records,
        }
    return provenance


def save_v4_benchmark_results(
    *,
    base_dir: Path,
    spec: Any,
    model_config: ModelConfig,
    predictions_df: pd.DataFrame,
    metrics_period_df: pd.DataFrame,
    metrics_annual_df: pd.DataFrame,
    folds: list[Any],
    scenarios: list[str],
    resolved_params: dict[str, dict[str, Any]],
    selection_record: dict[str, Any],
    elapsed_seconds: float,
) -> Path:
    """Persist one result and validate it before returning."""
    output = model_result_dir(base_dir, spec.benchmark_id, model_config.model_id)
    output.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output / "predictions.csv", index=False)
    metrics_period_df.to_csv(output / "metrics_period.csv", index=False)
    metrics_annual_df.to_csv(output / "metrics_annual.csv", index=False)
    selected_path = output / "selected_config.json"
    selected_path.write_text(
        json.dumps(selection_record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    stored_predictions = predictions_df.copy()
    stored_predictions["ds"] = pd.to_datetime(stored_predictions.ds)
    valid_label_provenance = _valid_label_provenance(stored_predictions)
    source_provenance = current_source_provenance()
    manifest = {
        "result_schema_version": "benchmark-result-v4",
        "benchmark_id": spec.benchmark_id,
        "protocol_version": spec.protocol_version,
        "protocol_hash": spec.protocol_hash,
        "metrics_version": spec.metrics_version,
        "model_id": model_config.model_id,
        "model_name": model_config.model_name,
        "toolkit": model_config.toolkit,
        "family": model_config.family,
        "variant": model_config.variant,
        "comparison_role": model_config.comparison_role,
        "config_group": model_config.config_group,
        "declared_params": model_config.params,
        "resolved_params_by_scenario": resolved_params,
        "selected_config_sha256": hashlib.sha256(
            selected_path.read_bytes()
        ).hexdigest(),
        "scenarios": scenarios,
        "folds": [fold.manifest_record() for fold in folds],
        "valid_label_counts": {
            scenario: valid_label_provenance[scenario]["count"]
            for scenario in scenarios
        },
        "valid_label_provenance": valid_label_provenance,
        "input_hashes": INPUT_HASHES,
        "software_versions": _software_versions(),
        "random_seed": SEED,
        "git_commit": source_provenance["git_commit"],
        "source_provenance": source_provenance,
        "updated_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed_seconds,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validate_v4_model_result(output, spec, require_all_scenarios=False)
    return output


def validate_v4_model_result(
    path: Path,
    spec: Any,
    *,
    require_all_scenarios: bool = True,
    require_current_runtime: bool = True,
) -> dict[str, Any]:
    """Fail closed for incomplete or incompatible results."""
    required = {
        "manifest.json",
        "selected_config.json",
        "predictions.csv",
        "metrics_period.csv",
        "metrics_annual.csv",
    }
    if missing := sorted(name for name in required if not (path / name).is_file()):
        raise ValueError(f"{path}: missing required files: {missing}")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "result_schema_version": "benchmark-result-v4",
        "benchmark_id": spec.benchmark_id,
        "protocol_version": spec.protocol_version,
        "protocol_hash": spec.protocol_hash,
        "metrics_version": spec.metrics_version,
        "input_hashes": INPUT_HASHES,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"{path}: incompatible {key}: {manifest.get(key)!r} != {value!r}"
            )
    required_provenance = {
        "software_versions": dict,
        "random_seed": int,
        "source_provenance": dict,
    }
    for key, expected_type in required_provenance.items():
        if not isinstance(manifest.get(key), expected_type):
            raise TypeError(f"{path}: missing or invalid {key}")
    source_provenance = manifest["source_provenance"]
    required_source_fields = {
        "git_commit": str,
        "git_dirty": bool,
        "source_tree_sha256": str,
    }
    for key, expected_type in required_source_fields.items():
        if not isinstance(source_provenance.get(key), expected_type):
            raise TypeError(f"{path}: missing or invalid source_provenance.{key}")
    if manifest.get("git_commit") != source_provenance["git_commit"]:
        raise ValueError(f"{path}: git_commit does not match source provenance")
    if require_current_runtime:
        runtime_expected = {
            "software_versions": _software_versions(),
            "random_seed": SEED,
            "source_provenance": current_source_provenance(),
        }
        for key, value in runtime_expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"{path}: incompatible {key}: {manifest.get(key)!r} != {value!r}"
                )
    selected_path = path / "selected_config.json"
    selected_hash = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    if manifest.get("selected_config_sha256") != selected_hash:
        raise ValueError(f"{path}: selected configuration hash mismatch")

    scenarios = tuple(manifest.get("scenarios", []))
    if not scenarios or not set(scenarios).issubset(spec.scenarios):
        raise ValueError(f"{path}: invalid scenario set {scenarios}")
    if require_all_scenarios and set(scenarios) != set(spec.scenarios):
        raise ValueError(f"{path}: partial scenario result cannot enter a release")

    predictions = pd.read_csv(
        path / "predictions.csv", parse_dates=["ds", "forecast_origin"]
    )
    period = pd.read_csv(path / "metrics_period.csv")
    annual = pd.read_csv(path / "metrics_annual.csv")
    if missing := sorted(V4_PREDICTION_COLUMNS - set(predictions.columns)):
        raise ValueError(f"{path}: prediction columns missing: {missing}")
    if missing := sorted(V4_PERIOD_COLUMNS - set(period.columns)):
        raise ValueError(f"{path}: period metric columns missing: {missing}")
    if predictions.duplicated(["scenario", "fold", "ds"]).any():
        raise ValueError(f"{path}: duplicate scenario/fold/date prediction rows")
    if set(predictions.scenario) != set(scenarios):
        raise ValueError(f"{path}: prediction scenarios do not match manifest")
    if set(annual.scenario) != set(scenarios) or annual.scenario.duplicated().any():
        raise ValueError(f"{path}: annual metrics must contain one row per scenario")
    valid_label_provenance = _valid_label_provenance(predictions)
    if manifest.get("valid_label_provenance") != valid_label_provenance:
        raise ValueError(f"{path}: valid label date set mismatch")
    expected_counts = {
        scenario: record["count"] for scenario, record in valid_label_provenance.items()
    }
    if manifest.get("valid_label_counts") != expected_counts:
        raise ValueError(f"{path}: valid label counts mismatch")

    folds = spec.folds(min(frame.ds.min() for frame in datasets.values()))
    if manifest.get("folds") != [fold.manifest_record() for fold in folds]:
        raise ValueError(f"{path}: fold contract mismatch")
    for scenario in scenarios:
        for fold in folds:
            rows = predictions[
                predictions.scenario.eq(scenario) & predictions.fold.eq(fold.fold_id)
            ]
            expected_dates = pd.date_range(fold.test_start, fold.test_end, freq="D")
            if len(rows) != fold.n_test_days or not pd.DatetimeIndex(rows.ds).equals(
                expected_dates
            ):
                raise ValueError(
                    f"{path}: incomplete dates for {scenario}/{fold.period_name}"
                )
            if not rows.forecast_origin.eq(fold.train_end).all():
                raise ValueError(
                    f"{path}: forecast origin mismatch for {scenario}/{fold.period_name}"
                )
    return manifest


def _fold_distribution(values: pd.Series) -> dict[str, float]:
    clean = values.dropna().to_numpy(float)
    if len(clean) == 0:
        return {
            "Fold_WAPE_Median": np.nan,
            "Fold_WAPE_IQR": np.nan,
            "Fold_WAPE_Mean_CI95_Low": np.nan,
            "Fold_WAPE_Mean_CI95_High": np.nan,
        }
    rng = np.random.default_rng(42)
    bootstrap = rng.choice(clean, size=(4000, len(clean)), replace=True).mean(axis=1)
    return {
        "Fold_WAPE_Median": float(np.median(clean)),
        "Fold_WAPE_IQR": float(np.quantile(clean, 0.75) - np.quantile(clean, 0.25)),
        "Fold_WAPE_Mean_CI95_Low": float(np.quantile(bootstrap, 0.025)),
        "Fold_WAPE_Mean_CI95_High": float(np.quantile(bootstrap, 0.975)),
    }


def _paired_fold_wins(period: pd.DataFrame) -> pd.DataFrame:
    """Describe paired fold wins without claiming statistical significance."""
    rows: list[dict[str, Any]] = []
    for scenario, group in period.groupby("scenario"):
        pivot = group.pivot(index="fold", columns="model_id", values="WAPE")
        for model_a, model_b in combinations(sorted(pivot.columns), 2):
            paired = pivot[[model_a, model_b]].dropna()
            delta = paired[model_a] - paired[model_b]
            rows.append(
                {
                    "scenario": scenario,
                    "model_a": model_a,
                    "model_b": model_b,
                    "paired_folds": len(paired),
                    "model_a_wins": int((delta < 0).sum()),
                    "model_b_wins": int((delta > 0).sum()),
                    "ties": int((delta == 0).sum()),
                    "median_wape_delta_a_minus_b": float(delta.median()),
                }
            )
    return pd.DataFrame(rows)


def generate_v4_summaries(
    *,
    spec: Any,
    base_dir: Path | None = None,
    result_root: Path | None = None,
    require_all_scenarios: bool = True,
    require_current_runtime: bool = True,
) -> dict[str, pd.DataFrame]:
    """Aggregate only models compatible with one explicit benchmark ID."""
    if (base_dir is None) == (result_root is None):
        raise ValueError("Provide exactly one of base_dir or result_root")
    root = (
        benchmark_result_root(base_dir, spec.benchmark_id)
        if base_dir is not None
        else Path(result_root)
    )
    models_dir = root / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(f"No model results found under {models_dir}")
    annual_parts: list[pd.DataFrame] = []
    period_parts: list[pd.DataFrame] = []
    manifests = []
    for path in sorted(item for item in models_dir.iterdir() if item.is_dir()):
        manifests.append(
            validate_v4_model_result(
                path,
                spec,
                require_all_scenarios=require_all_scenarios,
                require_current_runtime=require_current_runtime,
            )
        )
        annual_parts.append(pd.read_csv(path / "metrics_annual.csv"))
        period_parts.append(pd.read_csv(path / "metrics_period.csv"))
    if not annual_parts:
        raise ValueError(f"No compatible model results found under {models_dir}")

    annual = pd.concat(annual_parts, ignore_index=True)
    period = pd.concat(period_parts, ignore_index=True)
    distribution_rows = [
        {
            "scenario": scenario,
            "model_id": model_id,
            **_fold_distribution(group.WAPE),
        }
        for (scenario, model_id), group in period.groupby(["scenario", "model_id"])
    ]
    scenario = annual.merge(
        pd.DataFrame(distribution_rows), on=["scenario", "model_id"], how="left"
    ).sort_values(["scenario", "WAPE", "MAE"])
    outputs = {
        "scenario_leaderboard": scenario,
        "breadth_scorecard": scenario[scenario.comparison_role.eq("breadth")].copy(),
        "configuration_ablations": scenario[
            scenario.comparison_role.eq("ablation")
        ].copy(),
        "automl_scorecard": scenario[scenario.comparison_role.eq("automl")].copy(),
        "capability_matrix": scenario.pivot(
            index="model_name", columns="scenario", values="WAPE"
        ).reset_index(),
        "period_trajectories": period,
        "paired_fold_wins": _paired_fold_wins(period),
    }
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(summary_dir / f"{name}.csv", index=False)
    summary_manifest = {
        "benchmark_id": spec.benchmark_id,
        "protocol_hash": spec.protocol_hash,
        "model_ids": sorted(manifest["model_id"] for manifest in manifests),
        "source_manifest_hashes": {
            manifest["model_id"]: _json_sha256(manifest) for manifest in manifests
        },
        "outputs": {
            f"{name}.csv": hashlib.sha256(
                (summary_dir / f"{name}.csv").read_bytes()
            ).hexdigest()
            for name in outputs
        },
    }
    (summary_dir / "summary_manifest.json").write_text(
        json.dumps(summary_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return outputs


def generate_training_history_comparison(
    *,
    base_dir: Path,
    main_spec: Any,
    fixed_window_spec: Any,
) -> Path:
    """Compare two protocols that differ only in their training-history policy."""
    summaries = {}
    manifest_hashes: dict[str, dict[str, str]] = {}
    model_sets = []
    for label, spec in (("main", main_spec), ("fixed_window", fixed_window_spec)):
        generated = generate_v4_summaries(base_dir=base_dir, spec=spec)
        summaries[label] = generated["scenario_leaderboard"]
        summary_manifest_path = (
            benchmark_result_root(base_dir, spec.benchmark_id)
            / "summary"
            / "summary_manifest.json"
        )
        summary_manifest = json.loads(summary_manifest_path.read_text(encoding="utf-8"))
        manifest_hashes[label] = summary_manifest["source_manifest_hashes"]
        model_sets.append(set(summary_manifest["model_ids"]))
    comparison = build_training_history_comparison(
        summaries["main"],
        summaries["fixed_window"],
        main_spec=main_spec,
        fixed_window_spec=fixed_window_spec,
    )

    comparison_id = f"{main_spec.benchmark_id}--vs--{fixed_window_spec.benchmark_id}"
    output_dir = Path(base_dir) / "comparisons" / comparison_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_history_comparison.csv"
    comparison.to_csv(output_path, index=False)
    manifest = {
        "comparison_schema_version": "training-history-comparison-v2",
        "main_benchmark_id": main_spec.benchmark_id,
        "main_protocol_hash": main_spec.protocol_hash,
        "fixed_window_benchmark_id": fixed_window_spec.benchmark_id,
        "fixed_window_protocol_hash": fixed_window_spec.protocol_hash,
        "fixed_window_weeks": fixed_window_spec.train_weeks,
        "model_ids": sorted(model_sets[0]),
        "source_manifest_hashes": manifest_hashes,
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_dir
