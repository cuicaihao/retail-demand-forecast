"""Strict comparison helpers for training-history benchmark protocols."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

IDENTITY_COLUMNS = [
    "scenario",
    "model_id",
    "model_name",
    "toolkit",
    "family",
    "variant",
    "comparison_role",
    "config_group",
]

METRIC_COLUMNS = [
    "MAE",
    "RMSE",
    "WAPE",
    "Bias",
    "MASE",
    "ZeroRecallAt05",
    "DemandRecallAt05",
    "DemandEventF1At05",
    "PositiveDemandMAE",
    "Coverage80",
    "MeanIntervalWidth80",
    "PinballLoss80",
    "n",
]

COMPARABLE_SPEC_FIELDS = (
    "protocol_version",
    "evaluation_start",
    "evaluation_end",
    "horizon_days",
    "step_days",
    "selection_end",
    "selection_policy",
    "scenarios",
    "valid_label_rule",
    "metrics_version",
)


def validate_history_specs(main_spec: Any, ablation_spec: Any) -> None:
    """Require protocols to differ only in their training-history policy."""
    mismatches = [
        field_name
        for field_name in COMPARABLE_SPEC_FIELDS
        if getattr(main_spec, field_name) != getattr(ablation_spec, field_name)
    ]
    if mismatches:
        raise ValueError(
            "Training-history protocols differ outside history policy: "
            + ", ".join(mismatches)
        )
    if main_spec.training_policy == ablation_spec.training_policy:
        raise ValueError(
            "Training-history ablation requires different history policies"
        )
    if main_spec.training_policy != "expanding":
        raise ValueError("The main training-history protocol must be expanding")
    if ablation_spec.training_policy != "sliding" or ablation_spec.train_weeks != 24:
        raise ValueError("The ablation protocol must use a fixed 24-week window")


def build_training_history_comparison(
    main: pd.DataFrame,
    ablation: pd.DataFrame,
    *,
    main_spec: Any,
    ablation_spec: Any,
) -> pd.DataFrame:
    """Build one paired expanding-history versus 24-week result table."""
    validate_history_specs(main_spec, ablation_spec)
    main_ids = set(main["model_id"])
    ablation_ids = set(ablation["model_id"])
    if main_ids != ablation_ids:
        raise ValueError(
            "Training-history model sets differ: "
            f"missing from main={sorted(ablation_ids - main_ids)}, "
            f"missing from ablation={sorted(main_ids - ablation_ids)}"
        )

    required = set(IDENTITY_COLUMNS + METRIC_COLUMNS)
    for label, frame in (("main", main), ("ablation", ablation)):
        if missing := sorted(required - set(frame.columns)):
            raise ValueError(f"{label} summary is missing columns: {missing}")

    comparison = main[IDENTITY_COLUMNS + METRIC_COLUMNS].merge(
        ablation[IDENTITY_COLUMNS + METRIC_COLUMNS],
        on=IDENTITY_COLUMNS,
        how="outer",
        validate="one_to_one",
        suffixes=("_expanding", "_24w"),
        indicator=True,
    )
    if not comparison._merge.eq("both").all():
        raise ValueError("Training-history scenario/model identities do not match")
    comparison = comparison.drop(columns="_merge")
    for metric in METRIC_COLUMNS:
        comparison[f"{metric}_delta_24w_minus_expanding"] = (
            comparison[f"{metric}_24w"] - comparison[f"{metric}_expanding"]
        )
    wape_delta = comparison["WAPE_delta_24w_minus_expanding"]
    comparison["WAPE_winner"] = np.select(
        [wape_delta > 0, wape_delta < 0],
        [main_spec.training_policy, "sliding_24w"],
        default="tie",
    )
    return comparison.sort_values(["scenario", "model_id"]).reset_index(drop=True)


def summarize_training_history_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired WAPE wins without averaging across demand scenarios."""
    rows = []
    for scenario, frame in comparison.groupby("scenario", sort=True):
        winners = frame["WAPE_winner"].value_counts()
        rows.append(
            {
                "scenario": scenario,
                "model_count": len(frame),
                "expanding_wins": int(winners.get("expanding", 0)),
                "sliding_24w_wins": int(winners.get("sliding_24w", 0)),
                "ties": int(winners.get("tie", 0)),
                "median_WAPE_delta_24w_minus_expanding": float(
                    frame["WAPE_delta_24w_minus_expanding"].median()
                ),
                "breadth_median_WAPE_delta_24w_minus_expanding": float(
                    frame.loc[
                        frame["comparison_role"].eq("breadth"),
                        "WAPE_delta_24w_minus_expanding",
                    ].median()
                ),
            }
        )
    return pd.DataFrame(rows)
