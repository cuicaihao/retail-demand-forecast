from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "35-autogluon-timeseries-tutorial.ipynb"


def notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
    }
    nb.cells = [
        nbf.v4.new_markdown_cell(
            """# AutoGluon TimeSeries：自动模型选择、集成与零售需求预测

本 Notebook 使用与前面十个模型完全相同的四套数据、四个滚动验证窗口和最终 28 天测试期，回答三个问题：

1. AutoGluon 在固定 CPU 预算下会选择哪些统计、表格和轻量神经模型？
2. Weighted Ensemble 是否比已有 Prophet、树模型和神经网络更稳定？
3. ensemble 权重、组件排行榜和 permutation importance 能解释什么，又不能解释什么？

官方参考：[TimeSeries 快速入门](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-quick-start.html)、[模型库](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html)、[特征重要性](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.feature_importance.html)。本项目因 Torch 2.8 兼容性固定使用 AutoGluon 1.5.0。"""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. 实验边界

- 四个场景分别训练，不让不同量纲的数据互相借力。
- 每个场景的四个验证起点各使用 90 秒，最终拟合使用 240 秒，总预算约 10 分钟。
- 排除 Chronos、Toto、PatchTST 和 AutoGluon TFT，不下载基础模型权重，也不与已有 TFT 重复消耗预算。
- 闭店和缺货标签在训练目标中记为缺失；未来只提供营业状态、节假日和业务计划变量。
- AutoGluon 没有与旧协议直接可比的训练回看，因此只比较 validation 和 test。"""
        ),
        nbf.v4.new_code_cell(
            """import hashlib
import os
import sys
from pathlib import Path

def find_repo_root():
    configured = os.environ.get("RETAIL_FORECAST_ROOT")
    candidates = [Path(configured).expanduser()] if configured else [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "pyproject.toml").is_file() and (candidate / "notebooks").is_dir():
            return candidate
    raise RuntimeError("找不到仓库根目录；请从仓库内启动 Jupyter，或设置 RETAIL_FORECAST_ROOT")

REPO_ROOT = find_repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))
from IPython.display import Image, display

from retail_forecast.autogluon_benchmark import (
    AUTOGLOUON_MODEL_LIBRARY,
    EXCLUDED_FOUNDATION_MODELS,
    autogluon_frame,
    known_covariate_names,
    run_autogluon_benchmark,
)
from retail_forecast.common import DATA_DIR, INPUT_HASHES, SCENARIOS, datasets

print("Requested model library:", AUTOGLOUON_MODEL_LIBRARY)
print("Excluded high-cost models:", EXCLUDED_FOUNDATION_MODELS)"""
        ),
        nbf.v4.new_markdown_cell(
            """## 2. 数据适配与防泄漏检查

`TimeSeriesDataFrame` 使用 `item_id / timestamp / target` 长表。这里每次只有一个场景、一个 item；业务变量作为已知未来协变量。AutoGluon 会自动构造时间特征和目标滞后，因此不再手工加入未来不可得的目标统计。"""
        ),
        nbf.v4.new_code_cell(
            """for scenario, frame in datasets.items():
    converted = autogluon_frame(frame.head(10), scenario)
    print(scenario, known_covariate_names(scenario), list(converted.columns))
    assert not {"y", "stockout", "conditional_mean", "latent_demand", "occurrence_probability", "positive_demand"}.intersection(converted.columns)
    display(converted.head(2))"""
        ),
        nbf.v4.new_markdown_cell("## 3. 运行四场景 AutoGluon benchmark"),
        nbf.v4.new_code_cell(
            """result = run_autogluon_benchmark()
out = result["output_dir"]
display(result["aggregate_metrics"].round(4))
display(result["fit_details"].round(3))
print("Exports:", out.resolve())"""
        ),
        nbf.v4.new_markdown_cell("## 4. 组件排行榜、失败记录与 ensemble 权重"),
        nbf.v4.new_code_cell(
            """display(result["leaderboard"].sort_values(["scenario", "regime", "score_val"], ascending=[True, True, False]).round(4))
display(result["ensemble_weights"].sort_values(["scenario", "regime", "weight"], ascending=[True, True, False]).round(4))
display(result["model_failures"])
display(result["component_test_diagnostics"].sort_values(["scenario", "MAE"]).round(4))

for scenario in SCENARIOS:
    print("Final ensemble:", scenario)
    display(result["ensemble_weights"].loc[(result["ensemble_weights"].scenario == scenario) & (result["ensemble_weights"].regime == "final")].sort_values("weight", ascending=False).round(4))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 5. 预测区间与解释性

Permutation importance 表示扰动某个已知协变量后 MAE 的增加；负值意味着该特征可能有害。它是全局、事后的关联解释，不是 Prophet 的结构分解，也不能说明单日预测的因果来源。Ensemble 权重解释模型组合，但权重高不代表该组件在所有业务窗口都最好。"""
        ),
        nbf.v4.new_code_cell(
            """display(result["feature_importance"].sort_values(["scenario", "importance"], ascending=[True, False]).round(4))
display(result["business_slices"].sort_values(["scenario", "condition", "MAE"]).round(4))
display(Image(filename=str(out / "final_test_predictions.png")))"""
        ),
        nbf.v4.new_markdown_cell("## 6. 与原十种模型比较"),
        nbf.v4.new_code_cell(
            """comparison = result["comparison"]
for scenario in SCENARIOS:
    print(scenario)
    display(comparison.loc[(comparison.scenario == scenario) & comparison.split.isin(["validation", "test"])].sort_values(["split", "MAE"])[["model", "split", "MAE", "RMSE", "WAPE", "Bias", "n", "test_minus_validation_WAPE"]].round(4))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 7. 如何评价 AutoGluon

优势是统一接口、自动模型比较、概率区间、失败容错和集成；代价是依赖较重、预算会改变候选模型完成度、内部选择比单模型更黑盒。对于单条、约三年的日序列，统计和表格组件通常比全局深度模型更容易得到稳定估计。最终结论必须同时查看外部滚动验证、最终测试、ensemble 组成和业务切片，不能只引用内部 leaderboard。"""
        ),
        nbf.v4.new_code_cell(
            """required = {"metrics.csv", "aggregate_metrics.csv", "predictions.csv", "leaderboard.csv", "ensemble_weights.csv", "fit_details.csv", "business_slices.csv", "feature_importance.csv", "model_failures.csv", "component_test_diagnostics.csv", "comparison_with_all_models.csv", "manifest.json"}
assert required.issubset({path.name for path in out.iterdir()})
for name, digest in INPUT_HASHES.items():
    assert hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() == digest
print("AutoGluon benchmark validation complete")"""
        ),
    ]
    for index, cell in enumerate(nb.cells, start=1):
        cell.id = f"autogluon-{index:02d}"
    return nb


def main():
    nbf.write(notebook(), OUTPUT)


if __name__ == "__main__":
    main()
