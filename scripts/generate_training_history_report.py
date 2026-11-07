"""Generate controlled Markdown blocks from a published history comparison."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_benchmark_report import replace_available_blocks
from validate_benchmark_release import validate_release

from retail_forecast.benchmark_registry import PUBLISHED_DIR

GENERATED_REPORTS = ROOT / "docs" / "reports" / "generated" / "comparisons"


def _display_names(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["model_name"] = (
        result["model_name"]
        .str.replace("Xgboost", "XGBoost", regex=False)
        .str.replace("Lightgbm", "LightGBM", regex=False)
        .str.replace("Catboost", "CatBoost", regex=False)
    )
    return result


def _markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label, _ in columns) + " |",
        "| " + " | ".join(":---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column, _, formatter in columns:
            value = row[column]
            if formatter == "percent":
                values.append(f"{value * 100:.2f}%")
            elif formatter == "points":
                values.append(f"{value * 100:+.2f} pp")
            elif formatter == "integer":
                values.append(str(int(value)))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate_blocks(comparison_id: str, release_id: str) -> Path:
    release_root = (
        PUBLISHED_DIR / "comparisons" / comparison_id / "releases" / release_id
    )
    manifest = validate_release(release_root)
    comparison = _display_names(
        pd.read_csv(release_root / "training_history_comparison.csv")
    )
    summary = pd.read_csv(release_root / "scenario_summary.csv")
    fixed_window_weeks = int(manifest["fixed_window_weeks"])
    fixed_window_label = f"{fixed_window_weeks} 周"
    output = GENERATED_REPORTS / comparison_id / release_id
    output.mkdir(parents=True, exist_ok=True)

    summary_columns = [
        ("scenario", "场景", "text"),
        ("model_count", "配置数", "integer"),
        ("expanding_wins", "扩展历史胜", "integer"),
        ("fixed_window_wins", f"{fixed_window_label}胜", "integer"),
        ("ties", "平局", "integer"),
        (
            "median_WAPE_delta_fixed_window_minus_expanding",
            f"WAPE 差值中位数 ({fixed_window_weeks}w-expanding)",
            "points",
        ),
        (
            "breadth_median_WAPE_delta_fixed_window_minus_expanding",
            "广度配置 WAPE 差值中位数",
            "points",
        ),
    ]
    (output / "history_policy_summary.md").write_text(
        _markdown_table(summary, summary_columns) + "\n", encoding="utf-8"
    )

    detail_blocks = []
    breadth = comparison[comparison.comparison_role.eq("breadth")]
    common_columns = [
        ("model_name", "代表模型", "text"),
        ("toolkit", "工具包", "text"),
        ("WAPE_expanding", "扩展历史 WAPE", "percent"),
        ("WAPE_fixed_window", f"{fixed_window_label} WAPE", "percent"),
        ("WAPE_delta_fixed_window_minus_expanding", "差值", "points"),
        ("WAPE_winner", "WAPE 胜者", "text"),
    ]
    luxury_columns = common_columns + [
        ("DemandEventF1At05_expanding", "扩展历史 Event F1", "percent"),
        (
            "DemandEventF1At05_fixed_window",
            f"{fixed_window_label} Event F1",
            "percent",
        ),
    ]
    for scenario in manifest["scenarios"]:
        frame = breadth[breadth.scenario.eq(scenario)].sort_values(
            "WAPE_delta_fixed_window_minus_expanding", ascending=False
        )
        columns = luxury_columns if scenario == "luxury" else common_columns
        detail_blocks.extend(
            [f"## {scenario}", "", _markdown_table(frame, columns), ""]
        )
    (output / "history_policy_breadth.md").write_text(
        "\n".join(detail_blocks).rstrip() + "\n", encoding="utf-8"
    )

    ablation_blocks = []
    ablations = comparison[comparison.comparison_role.eq("ablation")]
    ablation_columns = [("model_name", "消融配置", "text"), *common_columns[1:]]
    for scenario in manifest["scenarios"]:
        frame = ablations[ablations.scenario.eq(scenario)].sort_values(
            ["config_group", "WAPE_delta_fixed_window_minus_expanding"],
            ascending=[True, False],
        )
        columns = (
            ablation_columns + luxury_columns[len(common_columns) :]
            if scenario == "luxury"
            else ablation_columns
        )
        ablation_blocks.extend(
            [f"## {scenario}", "", _markdown_table(frame, columns), ""]
        )
    (output / "history_policy_ablations.md").write_text(
        "\n".join(ablation_blocks).rstrip() + "\n", encoding="utf-8"
    )

    automl_blocks = []
    automl = comparison[comparison.comparison_role.eq("automl")]
    for scenario in manifest["scenarios"]:
        frame = automl[automl.scenario.eq(scenario)].sort_values(
            "WAPE_delta_fixed_window_minus_expanding", ascending=False
        )
        columns = luxury_columns if scenario == "luxury" else common_columns
        automl_blocks.extend(
            [f"## {scenario}", "", _markdown_table(frame, columns), ""]
        )
    (output / "history_policy_automl.md").write_text(
        "\n".join(automl_blocks).rstrip() + "\n", encoding="utf-8"
    )

    manifest_path = release_root / "release_manifest.json"
    provenance = (
        f"> Comparison ID: `{comparison_id}`<br>\n"
        f"> Release ID: `{release_id}`<br>\n"
        f"> Main source: `{manifest['main_source_release']['benchmark_id']}` / "
        f"`{manifest['main_source_release']['release_id']}`<br>\n"
        f"> Fixed-window source: "
        f"`{manifest['fixed_window_source_release']['benchmark_id']}` / "
        f"`{manifest['fixed_window_source_release']['release_id']}`<br>\n"
        f"> Comparison release manifest SHA-256: "
        f"`{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}`<br>\n"
        "> WAPE differences are paired descriptive results, not statistical significance claims.\n"
    )
    (output / "provenance.md").write_text(provenance, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument(
        "--sync-target",
        type=Path,
        action="append",
        default=[],
        help="Markdown file containing all training-history marker pairs.",
    )
    args = parser.parse_args()
    output = generate_blocks(args.comparison, args.release)
    blocks = {
        "history_policy_summary": (output / "history_policy_summary.md").read_text(),
        "history_policy_breadth": (output / "history_policy_breadth.md").read_text(),
        "history_policy_ablations": (
            output / "history_policy_ablations.md"
        ).read_text(),
        "history_policy_automl": (output / "history_policy_automl.md").read_text(),
        "history_provenance": (output / "provenance.md").read_text(),
    }
    for target in args.sync_target:
        document = target.read_text(encoding="utf-8")
        document = replace_available_blocks(document, blocks)
        target.write_text(document, encoding="utf-8")
    print(f"Generated training-history report blocks: {output}")


if __name__ == "__main__":
    main()
