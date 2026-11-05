from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_benchmark_report import replace_marked_block
from publish_benchmark import (
    _contains_only_release_outputs,
    _publishing_source_provenance,
    _validate_release_id,
    publish,
    published_release_root,
)
from run_benchmark import MODEL_REGISTRY
from validate_benchmark_release import validate_release

from retail_forecast.benchmark_adapters import (
    TREE_TOOLKITS,
    ARIMAForecasterAdapter,
    _selection_windows_end,
)
from retail_forecast.benchmark_protocol import load_benchmark_spec
from retail_forecast.benchmark_registry import (
    ModelConfig,
    _paired_fold_wins,
    compute_standard_metrics,
    save_v4_benchmark_results,
    validate_v4_model_result,
)
from retail_forecast.common import datasets, valid_labels
from retail_forecast.history_comparison import (
    METRIC_COLUMNS,
    build_training_history_comparison,
    summarize_training_history_comparison,
)


class BenchmarkV4ProtocolTests(unittest.TestCase):
    def test_classical_and_gbdt_toolkits_share_the_tree_adapter_matrix(self) -> None:
        self.assertEqual(
            set(TREE_TOOLKITS),
            {"decision_tree", "random_forest", "lightgbm", "xgboost", "catboost"},
        )

    def test_registry_has_symmetric_first_stage_gbdt_ablations(self) -> None:
        self.assertEqual(len(MODEL_REGISTRY), 21)
        expected_variants = {
            "default",
            "tweedie_p11",
            "tweedie_p12",
            "tweedie_p15",
            "tweedie_p18",
        }
        for toolkit in ("lightgbm", "xgboost", "catboost"):
            variants = {
                config.variant
                for config in MODEL_REGISTRY.values()
                if config.toolkit == toolkit
            }
            self.assertEqual(variants, expected_variants)

    def test_arima_is_a_predeclared_breadth_configuration(self) -> None:
        config = MODEL_REGISTRY["arima"]
        self.assertEqual(config.toolkit, "statsmodels")
        self.assertEqual(config.family, "statistical")
        self.assertEqual(config.comparison_role, "breadth")
        self.assertEqual(config.params["order"], [1, 0, 1])
        self.assertTrue(config.params["use_exog"])

    def test_arima_prediction_does_not_read_future_targets(self) -> None:
        frame = datasets["grocery"]
        history = frame.iloc[:400].copy()
        future = frame.iloc[400:428].copy()
        adapter = ARIMAForecasterAdapter(MODEL_REGISTRY["arima"].params)
        adapter.fit(history, "grocery")

        expected = adapter.predict(future, "grocery")
        changed = future.copy()
        changed["y"] = changed["y"] + 1_000_000
        for column in ("conditional_mean", "latent_demand"):
            if column in changed:
                changed[column] = changed[column] + 1_000_000
        actual = adapter.predict(changed, "grocery")

        np.testing.assert_allclose(actual.point_forecast, expected.point_forecast)
        np.testing.assert_allclose(actual.q10, expected.q10)
        np.testing.assert_allclose(actual.q90, expected.q90)
        self.assertTrue(np.isfinite(actual.point_forecast).all())
        self.assertTrue((actual.q10 <= actual.q90).all())

    def test_parameter_selection_windows_precede_formal_evaluation(self) -> None:
        spec = load_benchmark_spec("2026-retail-week-expanding-v4")
        for scenario in spec.scenarios:
            self.assertLessEqual(_selection_windows_end(scenario), spec.selection_end)

    def test_expanding_and_104_week_specs_share_formal_folds(self) -> None:
        expanding = load_benchmark_spec("2026-retail-week-expanding-v4")
        sliding = load_benchmark_spec("2026-retail-week-104w-rolling-v4")
        data_start = datasets["grocery"].ds.min()
        expanding_folds = expanding.folds(data_start)
        sliding_folds = sliding.folds(data_start)
        self.assertEqual(len(expanding_folds), 13)
        self.assertEqual(
            [(fold.test_start, fold.test_end) for fold in expanding_folds],
            [(fold.test_start, fold.test_end) for fold in sliding_folds],
        )
        self.assertEqual(expanding_folds[0].train_start, data_start)
        self.assertEqual(
            (sliding_folds[0].train_end - sliding_folds[0].train_start).days + 1,
            728,
        )
        evaluation_dates = [
            date
            for fold in expanding_folds
            for date in pd.date_range(fold.test_start, fold.test_end, freq="D")
        ]
        self.assertEqual(len(evaluation_dates), 364)
        self.assertEqual(len(set(evaluation_dates)), 364)
        self.assertLess(expanding.selection_end, expanding.evaluation_start)

    def test_metrics_are_pooled_and_event_definitions_are_explicit(self) -> None:
        result = compute_standard_metrics(
            np.array([0.0, 0.0, 1.0, 2.0]),
            np.array([0.0, 0.6, 0.4, 2.0]),
            q10=np.array([0.0, 0.0, 0.0, 1.0]),
            q90=np.array([0.2, 1.0, 1.2, 3.0]),
            seasonal_scale=0.5,
        )
        self.assertAlmostEqual(result["MAE"], 0.3)
        self.assertAlmostEqual(result["WAPE"], 0.4)
        self.assertEqual(result["WAPE_Denominator"], 3.0)
        self.assertAlmostEqual(result["MASE"], 0.6)
        self.assertAlmostEqual(result["ZeroRecallAt05"], 0.5)
        self.assertAlmostEqual(result["DemandRecallAt05"], 0.5)
        self.assertAlmostEqual(result["DemandEventF1At05"], 0.5)
        self.assertAlmostEqual(result["PositiveDemandMAE"], 0.3)
        self.assertAlmostEqual(result["Coverage80"], 1.0)
        self.assertAlmostEqual(result["MeanIntervalWidth80"], 1.1)

    def test_zero_sales_denominator_is_missing_not_infinite(self) -> None:
        result = compute_standard_metrics(np.zeros(3), np.ones(3))
        self.assertTrue(np.isnan(result["WAPE"]))
        self.assertTrue(np.isnan(result["Bias"]))
        self.assertEqual(result["WAPE_Denominator"], 0.0)

    def test_paired_fold_wins_use_matching_folds(self) -> None:
        period = pd.DataFrame(
            {
                "scenario": ["grocery"] * 6,
                "fold": [0, 1, 2, 0, 1, 2],
                "model_id": ["a", "a", "a", "b", "b", "b"],
                "WAPE": [0.1, 0.3, 0.2, 0.2, 0.1, 0.2],
            }
        )
        result = _paired_fold_wins(period).iloc[0]
        self.assertEqual(result["paired_folds"], 3)
        self.assertEqual(result["model_a_wins"], 1)
        self.assertEqual(result["model_b_wins"], 1)
        self.assertEqual(result["ties"], 1)

    def test_report_marker_replacement_preserves_authored_prose(self) -> None:
        original = (
            "before\n<!-- BENCHMARK_BLOCK:provenance:START -->\nold\n"
            "<!-- BENCHMARK_BLOCK:provenance:END -->\nafter\n"
        )
        changed = replace_marked_block(original, "provenance", "new")
        self.assertTrue(changed.startswith("before\n"))
        self.assertTrue(changed.endswith("\nafter\n"))
        self.assertIn("\nnew\n", changed)

    def test_release_ids_and_paths_are_versioned_below_benchmark(self) -> None:
        release_id = _validate_release_id("20261106-first-stage-v1")
        path = published_release_root("2026-retail-week-expanding-v4", release_id)
        self.assertEqual(path.name, release_id)
        self.assertEqual(path.parent.name, "releases")
        self.assertEqual(path.parent.parent.name, "2026-retail-week-expanding-v4")
        with self.assertRaisesRegex(ValueError, "YYYYMMDD"):
            _validate_release_id("latest")

    def test_publish_cleanliness_allows_only_pending_release_outputs(self) -> None:
        provenance = {
            "git_commit": "a" * 40,
            "git_dirty": True,
            "source_tree_sha256": "b" * 64,
        }
        with (
            patch(
                "publish_benchmark.current_source_provenance",
                return_value=provenance,
            ),
            patch(
                "publish_benchmark.subprocess.run",
                return_value=SimpleNamespace(
                    stdout=(
                        "?? data/benchmarks/published/example/releases/20261106-v1/\n"
                    )
                ),
            ),
            patch(
                "publish_benchmark._contains_only_release_outputs",
                return_value=True,
            ),
        ):
            result = _publishing_source_provenance()
            self.assertFalse(result["git_dirty"])
            self.assertEqual(len(result["pending_release_outputs"]), 1)
        with (
            patch(
                "publish_benchmark.current_source_provenance",
                return_value=provenance,
            ),
            patch(
                "publish_benchmark.subprocess.run",
                return_value=SimpleNamespace(stdout=" M README.md\n"),
            ),
            self.assertRaisesRegex(ValueError, "README.md"),
        ):
            _publishing_source_provenance()

    def test_comparison_release_paths_are_accepted_as_pending_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending = (
                root
                / "data/benchmarks/published/comparisons/a--vs--b/releases/20261106-test-v1"
            )
            pending.mkdir(parents=True)
            (pending / "scenario_summary.csv").write_text("scenario\n")
            with patch("publish_benchmark.ROOT", root):
                self.assertTrue(
                    _contains_only_release_outputs(
                        "data/benchmarks/published/comparisons/a--vs--b/"
                        "releases/20261106-test-v1/"
                    )
                )

    def test_training_history_comparison_is_paired_and_scenario_specific(self) -> None:
        expanding = load_benchmark_spec("2026-retail-week-expanding-v4")
        sliding = load_benchmark_spec("2026-retail-week-104w-rolling-v4")
        identity = {
            "scenario": "grocery",
            "model_id": "test_model",
            "model_name": "Test Model",
            "toolkit": "test",
            "family": "test",
            "variant": "default",
            "comparison_role": "breadth",
            "config_group": "test",
        }
        metrics = {metric: np.nan for metric in METRIC_COLUMNS}
        metrics.update({"WAPE": 0.1, "MAE": 1.0, "n": 10})
        main = pd.DataFrame([{**identity, **metrics}])
        fixed_window_metrics = {**metrics, "WAPE": 0.2, "MAE": 2.0}
        fixed_window = pd.DataFrame([{**identity, **fixed_window_metrics}])
        comparison = build_training_history_comparison(
            main,
            fixed_window,
            main_spec=expanding,
            fixed_window_spec=sliding,
        )
        self.assertAlmostEqual(
            comparison.loc[0, "WAPE_delta_fixed_window_minus_expanding"], 0.1
        )
        self.assertEqual(comparison.loc[0, "fixed_window_weeks"], 104)
        self.assertEqual(comparison.loc[0, "WAPE_winner"], "expanding")
        summary = summarize_training_history_comparison(comparison).iloc[0]
        self.assertEqual(summary["expanding_wins"], 1)
        self.assertEqual(summary["fixed_window_wins"], 0)


class BenchmarkV4ResultValidationTests(unittest.TestCase):
    def _write_valid_result(self, root: Path, scenarios: list[str] | None = None):
        spec = load_benchmark_spec("2026-retail-week-expanding-v4")
        scenarios = scenarios or list(spec.scenarios)
        folds = spec.folds(datasets["grocery"].ds.min())
        predictions = []
        period_rows = []
        annual_rows = []
        for scenario in scenarios:
            frame = datasets[scenario]
            scenario_y = []
            for fold in folds:
                future = frame[frame.ds.between(fold.test_start, fold.test_end)].copy()
                stored = future[["ds", "y", "is_open", "stockout"]].copy()
                stored["scenario"] = scenario
                stored["model_id"] = "test_model"
                stored["fold"] = fold.fold_id
                stored["period"] = fold.period_name
                stored["forecast_origin"] = fold.train_end
                stored["prediction"] = future.y.to_numpy(float)
                stored["q10"] = np.nan
                stored["q90"] = np.nan
                predictions.append(stored)
                valid = valid_labels(future)
                y = future.loc[valid, "y"].to_numpy(float)
                scenario_y.extend(y)
                metrics = compute_standard_metrics(y, y)
                period_rows.append(
                    {
                        "scenario": scenario,
                        "model_id": "test_model",
                        "model_name": "Test Model",
                        "toolkit": "test",
                        "family": "test",
                        "variant": "default",
                        "comparison_role": "breadth",
                        "fold": fold.fold_id,
                        "period": fold.period_name,
                        "test_start": str(fold.test_start.date()),
                        "test_end": str(fold.test_end.date()),
                        **metrics,
                    }
                )
            annual_rows.append(
                {
                    "scenario": scenario,
                    "model_id": "test_model",
                    "model_name": "Test Model",
                    "toolkit": "test",
                    "family": "test",
                    "variant": "default",
                    "comparison_role": "breadth",
                    "config_group": "test",
                    **compute_standard_metrics(scenario_y, scenario_y),
                }
            )
        config = ModelConfig(
            model_id="test_model",
            model_name="Test Model",
            family="test",
            toolkit="test",
            comparison_role="breadth",
        )
        path = save_v4_benchmark_results(
            base_dir=root,
            spec=spec,
            model_config=config,
            predictions_df=pd.concat(predictions, ignore_index=True),
            metrics_period_df=pd.DataFrame(period_rows),
            metrics_annual_df=pd.DataFrame(annual_rows),
            folds=folds,
            scenarios=scenarios,
            resolved_params={scenario: {} for scenario in scenarios},
            selection_record={"model_id": "test_model", "resolved_params": {}},
            elapsed_seconds=0.0,
        )
        return spec, path

    def test_valid_result_rejects_partial_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec, path = self._write_valid_result(Path(directory), ["grocery"])
            validate_v4_model_result(path, spec, require_all_scenarios=False)
            with self.assertRaisesRegex(ValueError, "partial scenario"):
                validate_v4_model_result(path, spec, require_all_scenarios=True)

    def test_protocol_change_invalidates_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec, path = self._write_valid_result(Path(directory))
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["protocol_hash"] = "changed"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "protocol_hash"):
                validate_v4_model_result(path, spec)

    def test_software_version_change_invalidates_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec, path = self._write_valid_result(Path(directory))
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["software_versions"]["pandas"] = "changed"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "software_versions"):
                validate_v4_model_result(path, spec)
            validate_v4_model_result(path, spec, require_current_runtime=False)

    def test_source_change_invalidates_cache_but_not_historical_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec, path = self._write_valid_result(Path(directory))
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["source_provenance"]["source_tree_sha256"] = "changed"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "source_provenance"):
                validate_v4_model_result(path, spec)
            validate_v4_model_result(path, spec, require_current_runtime=False)

    def test_valid_label_date_set_change_invalidates_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec, path = self._write_valid_result(Path(directory))
            predictions_path = path / "predictions.csv"
            predictions = pd.read_csv(predictions_path)
            index = predictions.index[
                predictions.is_open.astype(bool) & ~predictions.stockout.astype(bool)
            ][0]
            predictions.loc[index, "is_open"] = False
            predictions.to_csv(predictions_path, index=False)
            with self.assertRaisesRegex(ValueError, "valid label date set"):
                validate_v4_model_result(path, spec)

    def test_missing_or_duplicate_date_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec, path = self._write_valid_result(Path(directory))
            predictions_path = path / "predictions.csv"
            predictions = pd.read_csv(predictions_path)
            predictions = pd.concat([predictions.iloc[:-1], predictions.iloc[[0]]])
            predictions.to_csv(predictions_path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate|incomplete"):
                validate_v4_model_result(path, spec)

    def test_versioned_release_is_immutable_and_self_validating(self) -> None:
        provenance = {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "source_tree_sha256": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "retail_forecast.benchmark_registry.current_source_provenance",
                return_value=provenance,
            ):
                spec, _ = self._write_valid_result(root)
            published = root / "published"
            release_id = "20261106-test-v1"
            with (
                patch("publish_benchmark.PUBLISHED_DIR", published),
                patch(
                    "publish_benchmark._publishing_source_provenance",
                    return_value=provenance,
                ),
            ):
                destination = publish(
                    root,
                    spec.benchmark_id,
                    release_id,
                    model_ids=["test_model"],
                )
                release = validate_release(destination)
                self.assertEqual(release["release_id"], release_id)
                self.assertEqual(release["model_ids"], ["test_model"])
                inherited = publish(
                    root,
                    spec.benchmark_id,
                    "20261106-test-v2",
                    model_ids=["test_model"],
                    base_release_id=release_id,
                )
                inherited_release = validate_release(inherited)
                self.assertEqual(inherited_release["base_release_id"], release_id)
                with self.assertRaises(FileExistsError):
                    publish(
                        root,
                        spec.benchmark_id,
                        release_id,
                        model_ids=["test_model"],
                    )


if __name__ == "__main__":
    unittest.main()
