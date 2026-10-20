from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
MODELS = [
    ("32a-decision-tree-tutorial.ipynb", "Decision Tree", "决策树"),
    ("32b-random-forest-tutorial.ipynb", "Random Forest", "随机森林"),
    ("32c-lightgbm-tutorial.ipynb", "LightGBM", "LightGBM"),
    ("32d-xgboost-tutorial.ipynb", "XGBoost", "XGBoost"),
    ("32e-catboost-tutorial.ipynb", "CatBoost", "CatBoost"),
]


SETUP = """# 首次使用请在仓库根目录运行：uv sync --group dev
from pathlib import Path
import os, sys, inspect

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
import pandas as pd
from IPython.display import display, Code
from retail_forecast.common import *
from retail_forecast.trees import *

display(pd.DataFrame([
    {
        "scenario": name,
        "days": len(frame),
        "start": frame.ds.min(),
        "end": frame.ds.max(),
        "open_zero_rate": float(frame.loc[frame.is_open, "y"].eq(0).mean()),
        "stockout_days": int(frame.stockout.sum()),
    }
    for name, frame in datasets.items()
]))"""


def model_notebook(model_name: str, chinese_name: str):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"""# 用 {chinese_name} 做零售需求预测

本 Notebook 使用商超、红酒、庭院和奢侈品手袋四套固定 CSV。每套数据独立使用训练期内的四个滚动窗口选择超参数，再用全部测试前数据训练，最后仅评估一次 2025-12-04 至 2025-12-31 的 28 天测试集。

所有结果都是当前数据、特征工程和有限调参预算的联合结果，不能解释为模型家族的普遍排名。训练回看只描述拟合能力；奢侈品窗口可能没有销量，此时 WAPE 无定义，应结合 MAE、RMSE 与 Bias。"""
        ),
        nbf.v4.new_markdown_cell("## 1. 数据、时间边界与防泄漏协议"),
        nbf.v4.new_code_cell(SETUP),
        nbf.v4.new_markdown_cell(
            """## 2. 特征工程

每个预测起点只使用当时已知的信息：日期趋势、周/年 Fourier 项、节假日、场景业务变量，以及截至起点冻结的最近销量、7 日位置滞后、7/28 日统计量、有效标签比例和预测步长。诊断真值 `conditional_mean`、`latent_demand`、未来销量、闭店和缺货结果均不会进入特征。"""
        ),
        nbf.v4.new_code_cell(
            """scenario = "grocery"
history = datasets[scenario].iloc[:TRAIN_END + 1]
future = datasets[scenario].iloc[TRAIN_END + 1:TRAIN_END + 1 + HORIZON]
X = origin_features(history, future, scenario)
display(X.head())
print("Feature columns:", list(X))
print("Truth excluded:", not {"y", "latent_demand", "conditional_mean", "stockout", "is_open"}.intersection(X))"""
        ),
        nbf.v4.new_markdown_cell(
            f"""## 3. {chinese_name} 的候选参数与训练方式

候选配置固定且可检查，选择指标为四个滚动窗口合并后的 MAE。最终测试集不参与选参。提升树还会在每个训练历史末端保留一个内部窗口用于早停，随后按选出的轮数在该起点的完整训练历史上重训。"""
        ),
        nbf.v4.new_code_cell(
            f"""MODEL_NAME = {model_name!r}
display(pd.DataFrame(PARAMETER_GRIDS[MODEL_NAME]))
display(Code(inspect.getsource(fit_model), language="python"))"""
        ),
        nbf.v4.new_markdown_cell("## 4. 调参、最终训练与测试"),
        nbf.v4.new_code_cell(
            """m, p, tuning, selected, importance, final_models, out = run_model_benchmark(MODEL_NAME)
display(selected)
display(tuning.sort_values(["scenario", "validation_MAE"]).groupby("scenario").head(4).round(4))
display(m.round(4))

baseline = seasonal_naive_diagnostics()
display(baseline.groupby(["scenario", "split"], as_index=False)[["MAE", "RMSE", "WAPE", "Bias"]].mean().round(4))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 5. 性能图与特征重要性

原生重要性用于解释模型内部的分裂选择；跨模型比较使用滚动验证上的 permutation MAE 增量。重要性描述关联而非因果效应。"""
        ),
        nbf.v4.new_code_cell(
            """plot_model_benchmark(MODEL_NAME, m, p, importance, out)
display(
    importance.loc[importance.importance_type.eq("permutation_MAE_increase")]
    .sort_values(["scenario", "importance"], ascending=[True, False])
    .groupby("scenario").head(12).round(4)
)"""
        ),
        nbf.v4.new_markdown_cell("## 6. 导出、复现与业务边界"),
        nbf.v4.new_code_cell(
            """required = {"metrics.csv", "tuning_results.csv", "selected_parameters.csv", "predictions.csv", "feature_importance.csv", "manifest.json"}
assert required.issubset({path.name for path in out.iterdir()})
assert set(m.scenario) == set(SCENARIOS)
assert set(m.split) == {"train_hindcast", "validation", "test"}
for name, digest in INPUT_HASHES.items():
    assert hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() == digest
print("Exports:", out.resolve())"""
        ),
    ]
    return nb


def comparison_notebook():
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# 四套零售数据的五种树模型比较

按 Decision Tree → Random Forest → LightGBM → XGBoost → CatBoost 汇总训练回看、训练期滚动验证和最终 28 天测试。该 Notebook 只读取输入哈希完全一致的最新模型运行；请先依次执行前五个 Notebook。

比较反映当前 CSV、共同特征、候选参数范围和随机种子，不代表任何模型在其他数据上的普遍优劣。"""
        ),
        nbf.v4.new_markdown_cell("## 1. 载入并验证五组实验"),
        nbf.v4.new_code_cell(SETUP),
        nbf.v4.new_code_cell(
            """tables, sources = load_comparison_results()
display(pd.DataFrame([{"model": model, "source": str(path)} for model, path in sources.items()]))
assert list(sources) == MODEL_ORDER"""
        ),
        nbf.v4.new_markdown_cell("## 2. 训练、验证与测试指标"),
        nbf.v4.new_code_cell(
            """comparison, out = export_comparison(tables)
display(comparison.sort_values(["scenario", "split", "model"]).round(4))
display(comparison.loc[comparison.split.eq("test")].pivot(index="scenario", columns="model", values=["MAE", "RMSE", "WAPE", "Bias"]).round(4))"""
        ),
        nbf.v4.new_markdown_cell("## 3. 五模型性能与最终测试预测图"),
        nbf.v4.new_code_cell("plot_comparison(tables, comparison, out)"),
        nbf.v4.new_markdown_cell(
            """## 4. 可比较的特征重要性

不同库的原生 gain、impurity 和 feature importance 定义不同，不能直接比较。这里统一查看滚动验证数据被打乱后引起的 MAE 增量；负值表示该特征在有限验证样本上可能增加噪声。"""
        ),
        nbf.v4.new_code_cell(
            """comparable = tables["importance"].loc[tables["importance"].importance_type.eq("permutation_MAE_increase")]
display(
    comparable.sort_values(["scenario", "model", "importance"], ascending=[True, True, False])
    .groupby(["scenario", "model"]).head(10).round(4)
)"""
        ),
        nbf.v4.new_markdown_cell("## 5. 选定参数与交付边界"),
        nbf.v4.new_code_cell(
            """display(tables["parameters"])
assert set(comparison.model) == set(MODEL_ORDER)
assert set(comparison.scenario) == set(SCENARIOS)
assert set(comparison.split) == {"train_hindcast", "validation", "test"}
print("Exports:", out.resolve())"""
        ),
    ]
    return nb


def main() -> None:
    for filename, model_name, chinese_name in MODELS:
        nbf.write(model_notebook(model_name, chinese_name), NOTEBOOKS / filename)
    nbf.write(comparison_notebook(), NOTEBOOKS / "32f-tree-model-comparison.ipynb")


if __name__ == "__main__":
    main()
