"""Generate benchmark-owned Markdown blocks without overwriting authored prose."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_forecast.benchmark_protocol import load_benchmark_spec
from retail_forecast.benchmark_registry import PUBLISHED_DIR, benchmark_result_root

GENERATED_REPORTS = ROOT / "docs" / "reports" / "generated"


def _format_value(column: str, value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if column in {
        "WAPE",
        "Bias",
        "ZeroRecallAt05",
        "DemandRecallAt05",
        "DemandEventF1At05",
        "Coverage80",
        "standard_WAPE",
        "automl_WAPE",
        "standard_Event_F1",
        "automl_Event_F1",
        "garden",
        "grocery",
        "luxury",
        "wine",
    }:
        return f"{value * 100:.2f}%"
    if column == "WAPE_Delta_AutoML_Minus_Standard":
        return f"{value * 100:+.2f} pp"
    return f"{value:.4f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    labels = {
        "model_name": "模型",
        "toolkit": "工具包",
        "variant": "配置",
        "WAPE": "WAPE",
        "MAE": "MAE",
        "RMSE": "RMSE",
        "Bias": "Bias",
        "MASE": "MASE",
        "ZeroRecallAt05": "ZeroRecall@0.5",
        "DemandRecallAt05": "DemandRecall@0.5",
        "DemandEventF1At05": "Event F1@0.5",
        "PositiveDemandMAE": "Positive-demand MAE",
        "Fold_WAPE_Median": "Fold WAPE median",
        "Fold_WAPE_IQR": "Fold WAPE IQR",
        "Coverage80": "Coverage80",
        "MeanIntervalWidth80": "80% interval width",
        "PinballLoss80": "80% pinball loss",
        "scenario": "场景",
        "standard_model_name": "观察到的最佳普通广度模型",
        "standard_WAPE": "普通模型 WAPE",
        "automl_model_name": "AutoML 输出",
        "automl_WAPE": "AutoML WAPE",
        "WAPE_Delta_AutoML_Minus_Standard": "差值 (AutoML-普通)",
        "standard_Event_F1": "普通模型 Event F1",
        "automl_Event_F1": "AutoML Event F1",
        "garden": "Garden WAPE",
        "grocery": "Grocery WAPE",
        "luxury": "Luxury WAPE",
        "wine": "Wine WAPE",
    }
    lines = [
        "| " + " | ".join(labels.get(column, column) for column in columns) + " |",
        "| " + " | ".join(":---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(
                str(value) if isinstance(value, str) else _format_value(column, value)
            )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _normalise_display_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep acronym casing stable in generated presentation blocks."""
    result = frame.copy()
    result["model_name"] = (
        result["model_name"]
        .str.replace("Xgboost", "XGBoost", regex=False)
        .str.replace("Lightgbm", "LightGBM", regex=False)
        .str.replace("Catboost", "CatBoost", regex=False)
    )
    return result


def replace_marked_block(document: str, block_name: str, content: str) -> str:
    """Replace only a generator-owned region and preserve all surrounding prose."""
    start = f"<!-- BENCHMARK_BLOCK:{block_name}:START -->"
    end = f"<!-- BENCHMARK_BLOCK:{block_name}:END -->"
    if document.count(start) != 1 or document.count(end) != 1:
        raise ValueError(f"Expected exactly one marker pair for {block_name!r}")
    before, remainder = document.split(start, 1)
    _, after = remainder.split(end, 1)
    return f"{before}{start}\n{content.rstrip()}\n{end}{after}"


def replace_available_blocks(document: str, blocks: dict[str, str]) -> str:
    """Replace blocks present in a document while rejecting half-defined markers."""
    result = document
    for name, content in blocks.items():
        start = f"<!-- BENCHMARK_BLOCK:{name}:START -->"
        end = f"<!-- BENCHMARK_BLOCK:{name}:END -->"
        if (start in result) != (end in result):
            raise ValueError(f"Incomplete marker pair for {name!r}")
        if start in result:
            result = replace_marked_block(result, name, content)
    return result


def generate_blocks(benchmark_id: str, release_id: str) -> Path:
    spec = load_benchmark_spec(benchmark_id)
    root = benchmark_result_root(PUBLISHED_DIR, benchmark_id) / "releases" / release_id
    release_path = root / "release_manifest.json"
    if not release_path.is_file():
        raise FileNotFoundError(f"Benchmark is not published: {release_path}")
    summary = root / "summary"
    breadth = _normalise_display_names(pd.read_csv(summary / "breadth_scorecard.csv"))
    ablations = _normalise_display_names(
        pd.read_csv(summary / "configuration_ablations.csv")
    )
    automl = _normalise_display_names(pd.read_csv(summary / "automl_scorecard.csv"))
    output = GENERATED_REPORTS / benchmark_id / release_id
    output.mkdir(parents=True, exist_ok=True)

    general_columns = [
        "model_name",
        "toolkit",
        "variant",
        "WAPE",
        "MAE",
        "RMSE",
        "Bias",
        "MASE",
        "Fold_WAPE_Median",
        "Fold_WAPE_IQR",
    ]
    luxury_columns = [
        "model_name",
        "toolkit",
        "variant",
        "WAPE",
        "MAE",
        "ZeroRecallAt05",
        "DemandRecallAt05",
        "DemandEventF1At05",
        "PositiveDemandMAE",
    ]
    blocks = []
    for scenario_name in spec.scenarios:
        subset = breadth[breadth.scenario.eq(scenario_name)]
        columns = luxury_columns if scenario_name == "luxury" else general_columns
        blocks.extend(
            [
                f"## {scenario_name}",
                "",
                _markdown_table(subset, columns),
                "",
            ]
        )
    tables = "\n".join(blocks).rstrip() + "\n"
    (output / "scenario_leaderboards.md").write_text(tables, encoding="utf-8")

    ablation_groups = sorted(ablations.config_group.dropna().unique())
    representatives = breadth[breadth.config_group.isin(ablation_groups)]
    configuration = pd.concat([representatives, ablations], ignore_index=True)
    ablation_blocks = []
    for scenario_name in spec.scenarios:
        columns = luxury_columns if scenario_name == "luxury" else general_columns
        for config_group in ablation_groups:
            subset = configuration[
                configuration.scenario.eq(scenario_name)
                & configuration.config_group.eq(config_group)
            ].sort_values("WAPE")
            ablation_blocks.extend(
                [
                    f"## {scenario_name} / {config_group}",
                    "",
                    _markdown_table(subset, columns),
                    "",
                ]
            )
    ablation_tables = "\n".join(ablation_blocks).rstrip() + "\n"
    (output / "configuration_ablations.md").write_text(
        ablation_tables, encoding="utf-8"
    )

    automl_general_columns = [
        "model_name",
        "WAPE",
        "MAE",
        "Coverage80",
        "MeanIntervalWidth80",
        "PinballLoss80",
        "Fold_WAPE_Median",
        "Fold_WAPE_IQR",
    ]
    automl_luxury_columns = [
        "model_name",
        "WAPE",
        "MAE",
        "ZeroRecallAt05",
        "DemandRecallAt05",
        "DemandEventF1At05",
        "PositiveDemandMAE",
        "Coverage80",
        "MeanIntervalWidth80",
        "PinballLoss80",
    ]
    automl_blocks = []
    for scenario_name in spec.scenarios:
        subset = automl[automl.scenario.eq(scenario_name)].sort_values("WAPE")
        columns = (
            automl_luxury_columns
            if scenario_name == "luxury"
            else automl_general_columns
        )
        automl_blocks.extend(
            [
                f"## {scenario_name}",
                "",
                _markdown_table(subset, columns),
                "",
            ]
        )
    automl_tables = "\n".join(automl_blocks).rstrip() + "\n"
    (output / "automl_scorecard.md").write_text(automl_tables, encoding="utf-8")

    best_standard = (
        breadth.sort_values(["scenario", "WAPE"]).groupby("scenario").head(1)
    )
    ensemble = automl[automl.model_id.eq("ag-weighted-ensemble")]
    comparison = best_standard.merge(
        ensemble,
        on="scenario",
        suffixes=("_standard", "_automl"),
        validate="one_to_one",
    )
    comparison = comparison.assign(
        standard_model_name=comparison.model_name_standard,
        standard_WAPE=comparison.WAPE_standard,
        automl_model_name=comparison.model_name_automl,
        automl_WAPE=comparison.WAPE_automl,
        WAPE_Delta_AutoML_Minus_Standard=(
            comparison.WAPE_automl - comparison.WAPE_standard
        ),
        standard_Event_F1=comparison.DemandEventF1At05_standard,
        automl_Event_F1=comparison.DemandEventF1At05_automl,
    )
    comparison_columns = [
        "scenario",
        "standard_model_name",
        "standard_WAPE",
        "automl_model_name",
        "automl_WAPE",
        "WAPE_Delta_AutoML_Minus_Standard",
        "standard_Event_F1",
        "automl_Event_F1",
    ]
    (output / "automl_vs_standard.md").write_text(
        _markdown_table(comparison, comparison_columns) + "\n", encoding="utf-8"
    )

    capability = (
        breadth.pivot(
            index=["model_name", "toolkit"], columns="scenario", values="WAPE"
        )
        .reset_index()
        .sort_values("model_name")
    )
    capability_columns = [
        "model_name",
        "toolkit",
        *[scenario for scenario in spec.scenarios if scenario in capability.columns],
    ]
    (output / "capability_matrix.md").write_text(
        _markdown_table(capability, capability_columns) + "\n", encoding="utf-8"
    )

    timesfm = pd.concat(
        [
            breadth[breadth.config_group.eq("timesfm_3_0")],
            ablations[ablations.config_group.eq("timesfm_3_0")],
        ],
        ignore_index=True,
    ).sort_values(["scenario", "variant"])
    timesfm_columns = [
        "scenario",
        "model_name",
        "variant",
        "WAPE",
        "Coverage80",
        "MeanIntervalWidth80",
        "PinballLoss80",
        "DemandEventF1At05",
    ]
    (output / "timesfm_ablation.md").write_text(
        _markdown_table(timesfm, timesfm_columns) + "\n", encoding="utf-8"
    )

    release_digest = hashlib.sha256(release_path.read_bytes()).hexdigest()
    provenance = (
        f"> Benchmark ID: `{benchmark_id}`<br>\n"
        f"> Release ID: `{release_id}`<br>\n"
        f"> Protocol hash: `{spec.protocol_hash}`<br>\n"
        f"> Release manifest SHA-256: `{release_digest}`<br>\n"
        "> Fold uncertainty intervals are descriptive; they are not claims of statistical significance.\n"
    )
    (output / "provenance.md").write_text(provenance, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument(
        "--sync-target",
        type=Path,
        action="append",
        default=[],
        help="Markdown file containing all generated benchmark marker pairs.",
    )
    args = parser.parse_args()
    output = generate_blocks(args.benchmark, args.release)
    blocks = {
        "scenario_leaderboards": (output / "scenario_leaderboards.md").read_text(),
        "configuration_ablations": (output / "configuration_ablations.md").read_text(),
        "automl_scorecard": (output / "automl_scorecard.md").read_text(),
        "automl_vs_standard": (output / "automl_vs_standard.md").read_text(),
        "capability_matrix": (output / "capability_matrix.md").read_text(),
        "timesfm_ablation": (output / "timesfm_ablation.md").read_text(),
        "provenance": (output / "provenance.md").read_text(),
    }
    for target in args.sync_target:
        document = target.read_text(encoding="utf-8")
        document = replace_available_blocks(document, blocks)
        target.write_text(document, encoding="utf-8")
    print(f"Generated benchmark-owned report blocks: {output}")


if __name__ == "__main__":
    main()
