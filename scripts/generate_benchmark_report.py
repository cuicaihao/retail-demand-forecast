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
    }:
        return f"{value * 100:.2f}%"
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


def replace_marked_block(document: str, block_name: str, content: str) -> str:
    """Replace only a generator-owned region and preserve all surrounding prose."""
    start = f"<!-- BENCHMARK_BLOCK:{block_name}:START -->"
    end = f"<!-- BENCHMARK_BLOCK:{block_name}:END -->"
    if document.count(start) != 1 or document.count(end) != 1:
        raise ValueError(f"Expected exactly one marker pair for {block_name!r}")
    before, remainder = document.split(start, 1)
    _, after = remainder.split(end, 1)
    return f"{before}{start}\n{content.rstrip()}\n{end}{after}"


def generate_blocks(benchmark_id: str) -> Path:
    spec = load_benchmark_spec(benchmark_id)
    root = benchmark_result_root(PUBLISHED_DIR, benchmark_id)
    release_path = root / "release_manifest.json"
    if not release_path.is_file():
        raise FileNotFoundError(f"Benchmark is not published: {release_path}")
    scenario = pd.read_csv(root / "summary" / "scenario_leaderboard.csv")
    output = GENERATED_REPORTS / benchmark_id
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
        subset = scenario[scenario.scenario.eq(scenario_name)]
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

    release_digest = hashlib.sha256(release_path.read_bytes()).hexdigest()
    provenance = (
        f"> Benchmark ID: `{benchmark_id}`  \n"
        f"> Protocol hash: `{spec.protocol_hash}`  \n"
        f"> Release manifest SHA-256: `{release_digest}`  \n"
        "> Fold uncertainty intervals are descriptive; they are not claims of statistical significance.\n"
    )
    (output / "provenance.md").write_text(provenance, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument(
        "--sync-target",
        type=Path,
        action="append",
        default=[],
        help="Markdown file containing scenario_leaderboards and provenance marker pairs.",
    )
    args = parser.parse_args()
    output = generate_blocks(args.benchmark)
    blocks = {
        "scenario_leaderboards": (output / "scenario_leaderboards.md").read_text(),
        "provenance": (output / "provenance.md").read_text(),
    }
    for target in args.sync_target:
        document = target.read_text(encoding="utf-8")
        for name, content in blocks.items():
            document = replace_marked_block(document, name, content)
        target.write_text(document, encoding="utf-8")
    print(f"Generated benchmark-owned report blocks: {output}")


if __name__ == "__main__":
    main()
