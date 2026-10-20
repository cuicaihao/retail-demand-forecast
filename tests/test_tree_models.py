from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_forecast.common import DATA_DIR, INPUT_HASHES, datasets, valid_labels
from retail_forecast.trees import (
    MODEL_ORDER,
    PARAMETER_GRIDS,
    fit_model,
    origin_features,
    predict_model,
)


class TreeModelProtocolTests(unittest.TestCase):
    def test_all_requested_models_have_candidate_grids(self) -> None:
        self.assertEqual(
            MODEL_ORDER,
            ["Decision Tree", "Random Forest", "LightGBM", "XGBoost", "CatBoost"],
        )
        self.assertTrue(all(len(PARAMETER_GRIDS[name]) >= 4 for name in MODEL_ORDER))

    def test_origin_features_ignore_future_targets_and_diagnostic_truth(self) -> None:
        frame = datasets["grocery"]
        history, future = frame.iloc[:500], frame.iloc[500:528].copy()
        original = origin_features(history, future, "grocery")
        future["y"] = 99999.0
        future["conditional_mean"] = -99999.0
        changed = origin_features(history, future, "grocery")
        pd.testing.assert_frame_equal(original, changed)
        forbidden = {"y", "conditional_mean", "latent_demand", "stockout", "is_open"}
        self.assertFalse(forbidden.intersection(original.columns))

    def test_decision_tree_fit_is_reproducible_and_finite(self) -> None:
        frame = datasets["wine"]
        history, future = frame.iloc[:700], frame.iloc[700:728]
        params = PARAMETER_GRIDS["Decision Tree"][0]
        first, _ = fit_model(history, "wine", "Decision Tree", params)
        second, _ = fit_model(history, "wine", "Decision Tree", params)
        a = predict_model(first, history, future, "wine")
        b = predict_model(second, history, future, "wine")
        np.testing.assert_allclose(a, b)
        self.assertTrue(np.isfinite(a).all())
        self.assertGreater(valid_labels(future).sum(), 0)

    def test_input_hashes_remain_current(self) -> None:
        actual = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in DATA_DIR.glob("*.csv")
        }
        self.assertEqual(actual, INPUT_HASHES)


if __name__ == "__main__":
    unittest.main()
