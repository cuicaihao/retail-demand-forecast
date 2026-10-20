from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_forecast.autogluon_benchmark import (
    AUTOGLOUON_MODEL_LIBRARY,
    EXCLUDED_FOUNDATION_MODELS,
    autogluon_frame,
    known_covariate_names,
)
from retail_forecast.common import SCENARIOS, datasets, valid_labels
from retail_forecast.model_comparison import CORE_MODELS
from retail_forecast.neural import NEURAL_CANDIDATES, inputs, preprocessor
from retail_forecast.prophet_benchmark import PROPHET_CANDIDATES, prophet_frame


class ModelBenchmarkProtocolTests(unittest.TestCase):
    def test_prophet_candidates_and_scenario_features(self):
        self.assertEqual(len(PROPHET_CANDIDATES), 4)
        for scenario, frame in datasets.items():
            columns = prophet_frame(frame.head(5), scenario).columns
            self.assertIn("ds", columns)
            self.assertIn("y", columns)
            self.assertTrue(set(SCENARIOS[scenario]["features"]).issubset(columns))
            self.assertNotIn("conditional_mean", columns)
            self.assertNotIn("latent_demand", columns)

    def test_neural_candidate_budget(self):
        self.assertEqual(len(NEURAL_CANDIDATES["MLP"]), 4)
        self.assertEqual(len(NEURAL_CANDIDATES["LSTM"]), 4)
        self.assertEqual(len(NEURAL_CANDIDATES["TFT"]), 3)

    def test_neural_future_targets_do_not_change_inputs(self):
        frame = datasets["grocery"]
        history, future = frame.iloc[:500], frame.iloc[500:528].copy()
        prep = preprocessor(history, "grocery")
        first = inputs(history, future, "grocery", prep)
        future["y"] = 99999.0
        future["conditional_mean"] = -99999.0
        second = inputs(history, future, "grocery", prep)
        pd.testing.assert_frame_equal(pd.DataFrame(first[0]), pd.DataFrame(second[0]))
        pd.testing.assert_frame_equal(pd.DataFrame(first[1]), pd.DataFrame(second[1]))

    def test_core_model_order(self):
        self.assertEqual(len(CORE_MODELS), 10)
        self.assertEqual(CORE_MODELS[0], "Prophet")
        self.assertEqual(CORE_MODELS[-1], "TFT")

    def test_autogluon_model_budget(self):
        self.assertIn("SeasonalNaive", AUTOGLOUON_MODEL_LIBRARY)
        self.assertIn("AutoARIMA", AUTOGLOUON_MODEL_LIBRARY)
        self.assertIn("DirectTabular", AUTOGLOUON_MODEL_LIBRARY)
        self.assertIn("DeepAR", AUTOGLOUON_MODEL_LIBRARY)
        self.assertTrue(
            {"Chronos", "Toto", "PatchTST", "TemporalFusionTransformer"}.issubset(
                EXCLUDED_FOUNDATION_MODELS
            )
        )

    def test_autogluon_future_targets_do_not_change_inputs(self):
        future = datasets["luxury"].iloc[500:528].copy()
        first = autogluon_frame(future, "luxury").drop(columns="target")
        future["y"] = 99999.0
        future["conditional_mean"] = -99999.0
        future["latent_demand"] = 99999.0
        future["occurrence_probability"] = 1.0
        second = autogluon_frame(future, "luxury").drop(columns="target")
        pd.testing.assert_frame_equal(first, second)

    def test_autogluon_covariates_are_known_and_scenario_specific(self):
        forbidden = {
            "y",
            "target",
            "stockout",
            "conditional_mean",
            "latent_demand",
            "occurrence_probability",
            "positive_demand",
        }
        for scenario, frame in datasets.items():
            names = known_covariate_names(scenario)
            self.assertFalse(forbidden.intersection(names))
            self.assertIn("is_open", names)
            converted = autogluon_frame(frame.head(10), scenario)
            self.assertEqual(
                converted.target.isna().sum(), (~valid_labels(frame.head(10))).sum()
            )


if __name__ == "__main__":
    unittest.main()
