"""Generate the current README benchmark-result block from two V4 releases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_benchmark_report import replace_marked_block
from validate_benchmark_release import validate_release

SCENARIOS = ("grocery", "wine", "garden", "luxury")


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _fold_summary(row: pd.Series) -> str:
    return f"{_percent(row['Fold_WAPE_Mean'])} ± {_percent(row['Fold_WAPE_Std'])}"


def _ranking_table(frame: pd.DataFrame) -> str:
    lines = [
        "| 场景 | 排名 | 模型 | Pooled WAPE | 13-fold WAPE 均值 ± 标准差 | Event F1@0.5 |",
        "| :--- | ---: | :--- | ---: | ---: | ---: |",
    ]
    for scenario in SCENARIOS:
        ranked = frame[frame.scenario.eq(scenario)].nsmallest(3, "WAPE")
        if len(ranked) != 3:
            raise ValueError(f"Expected at least three results for {scenario!r}")
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            lines.append(
                "| "
                + " | ".join(
                    [
                        scenario.title(),
                        str(rank),
                        str(row["model_name"]),
                        _percent(row["WAPE"]),
                        _fold_summary(row),
                        _percent(row["DemandEventF1At05"]),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def _timesfm_table(breadth: pd.DataFrame, ablations: pd.DataFrame) -> str:
    frame = pd.concat(
        [
            breadth[breadth.config_group.eq("timesfm_3_0")],
            ablations[ablations.config_group.eq("timesfm_3_0")],
        ],
        ignore_index=True,
    )
    lines = [
        "| 场景 | 配置 | Pooled WAPE | 13-fold WAPE 均值 ± 标准差 | Coverage80 | Event F1@0.5 |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for scenario in SCENARIOS:
        selected = frame[frame.scenario.eq(scenario)].sort_values("variant")
        if set(selected.variant) != {"known_covariates", "univariate"}:
            raise ValueError(f"Incomplete TimesFM configurations for {scenario!r}")
        for _, row in selected.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        scenario.title(),
                        str(row["variant"]),
                        _percent(row["WAPE"]),
                        _fold_summary(row),
                        _percent(row["Coverage80"]),
                        _percent(row["DemandEventF1At05"]),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def _load_summary(release_root: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    manifest = validate_release(release_root)
    summary = release_root / "summary"
    frames = {
        "breadth": pd.read_csv(summary / "breadth_scorecard.csv"),
        "automl": pd.read_csv(summary / "automl_scorecard.csv"),
        "ablations": pd.read_csv(summary / "configuration_ablations.csv"),
    }
    return manifest, frames


def generate_block(main_release: Path, production_release: Path) -> str:
    main_manifest, main = _load_summary(main_release)
    production_manifest, production = _load_summary(production_release)
    if main_manifest["release_id"] != production_manifest["release_id"]:
        raise ValueError("README sources must use the same release ID")
    if main_manifest["model_ids"] != production_manifest["model_ids"]:
        raise ValueError("README sources must have identical model inventories")

    release_id = main_manifest["release_id"]
    sections = [
        f"当前审核 release 为 `{release_id}`，包含 26 个普通配置和 15 个独立 AutoML 配置。",
        "前三名统一按 pooled WAPE 排名；13-fold 均值与标准差只描述月度稳定性。",
        "Luxury 必须联合 Event F1 阅读，约 100% WAPE 可能只是接近全零预测。",
        "",
        "### 普通工具包广度榜",
        "",
        "每个工具包只有一个预先声明的代表配置参与排名；AutoGluon 不进入本表。",
        "TimesFM 单变量配置是其广度代表，因此达到前三名时会在本表出现。",
        "",
        "#### 主榜：扩展历史",
        "",
        _ranking_table(main["breadth"]),
        "",
        "#### 生产模拟：固定 104 周训练、预测 4 周",
        "",
        _ranking_table(production["breadth"]),
        "",
        "### AutoGluon 独立 AutoML 对比",
        "",
        "AutoGluon 的 15 个组件和 ensemble 只在 AutoML 层内部排名，不占据普通广度榜名次。",
        "",
        "#### 主榜：扩展历史",
        "",
        _ranking_table(main["automl"]),
        "",
        "#### 生产模拟：固定 104 周训练、预测 4 周",
        "",
        _ranking_table(production["automl"]),
        "",
        "### TimesFM 3.0 配置结果",
        "",
        "`univariate` 是广度代表；`known_covariates` 是已知未来业务变量消融。",
        "Coverage80 是经验区间覆盖率，不是库存服务水平保证。",
        "",
        "#### 主榜：扩展历史",
        "",
        _timesfm_table(main["breadth"], main["ablations"]),
        "",
        "#### 生产模拟：固定 104 周训练、预测 4 周",
        "",
        _timesfm_table(production["breadth"], production["ablations"]),
        "",
        "完整能力矩阵、配置消融和历史策略比较见",
        "[V4 发布摘要](docs/reports/2026-retail-week-v4-release-summary.md)；完整证据见",
        "[V4 完整报告](docs/reports/2026-retail-week-v4-first-stage-report.md)。",
    ]
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-release", type=Path, required=True)
    parser.add_argument("--production-release", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=ROOT / "README.md")
    args = parser.parse_args()

    block = generate_block(args.main_release, args.production_release)
    document = args.target.read_text(encoding="utf-8")
    document = replace_marked_block(document, "current_results", block)
    args.target.write_text(document, encoding="utf-8")
    print(f"Updated current benchmark results in {args.target}")


if __name__ == "__main__":
    main()
