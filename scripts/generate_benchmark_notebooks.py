from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
TAG = "standard_benchmark_extension"

SETUP = """# 首次使用请在仓库根目录运行：uv sync --group dev
from pathlib import Path
import os, sys, hashlib

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
import matplotlib.pyplot as plt
from IPython.display import display
from retail_forecast.common import *"""


def tagged(cell):
    cell.metadata["tags"] = [TAG]
    return cell


def extend_prophet(filename, scenarios, family, luxury=False):
    path = NOTEBOOKS / filename
    nb = nbf.read(path, as_version=4)
    nb.cells = [
        cell for cell in nb.cells if TAG not in cell.get("metadata", {}).get("tags", [])
    ]
    nb.cells.extend(
        [
            tagged(
                nbf.v4.new_markdown_cell(
                    """## 标准化四模型族比较接口

以下实验与前面的教学诊断分开：它只使用测试期之前的四个滚动窗口选择参数，然后在全部测试前历史上重训并评估最终 28 天。输出格式与 ARIMA、树模型和神经网络一致，供总汇总 Notebook 使用。结果只说明当前数据、特征和有限候选范围的联合表现。"""
                )
            ),
            tagged(
                nbf.v4.new_code_cell(
                    SETUP + "\nfrom retail_forecast.prophet_benchmark import *"
                )
            ),
            tagged(
                nbf.v4.new_code_cell(
                    f"""benchmark_scenarios = {list(scenarios)!r}
benchmark = run_prophet_benchmark(benchmark_scenarios, {family!r})
benchmark_metrics, benchmark_predictions, benchmark_tuning, benchmark_parameters, benchmark_fits, benchmark_explanations, benchmark_out = benchmark
display(benchmark_parameters)
display(benchmark_tuning.sort_values(["scenario", "validation_MAE"]).round(4))
display(benchmark_metrics.round(4))"""
                )
            ),
            tagged(
                nbf.v4.new_code_cell(
                    """fig, axes = plt.subplots(1, len(benchmark_scenarios), figsize=(6 * len(benchmark_scenarios), 4), constrained_layout=True)
axes = [axes] if len(benchmark_scenarios) == 1 else axes
for ax, scenario_name in zip(axes, benchmark_scenarios):
    group = benchmark_predictions.loc[(benchmark_predictions.scenario == scenario_name) & benchmark_predictions.split.eq("test")]
    ax.plot(group.ds, group.y, "ko-", ms=3, label="Observed")
    ax.plot(group.ds, group.prediction, label="Prophet")
    ax.set(title=f"{scenario_name} final 28-day test", ylabel="Units")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
fig.savefig(benchmark_out / "final_test_predictions.png", dpi=150)
plt.show(); plt.close(fig)

required = {"metrics.csv", "predictions.csv", "tuning_results.csv", "selected_parameters.csv", "fit_details.csv", "explanations.csv", "manifest.json"}
assert required.issubset({path.name for path in benchmark_out.iterdir()})
for name, digest in INPUT_HASHES.items():
    assert hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() == digest
print("Standard benchmark exports:", benchmark_out.resolve())"""
                )
            ),
        ]
    )
    if luxury:
        nb.cells.append(
            tagged(
                nbf.v4.new_code_cell(
                    """# Croston-SBA、TSB 与 Two-stage 只保留为奢侈品专项参考，不进入四场景总榜。
specialist_rows = []
if "final_table" in globals():
    table = final_table.reset_index().rename(columns={final_table.index.name or "index": "model"})
    table["split"] = "test"
    specialist_rows.append(table)
if "rolling_table" in globals():
    table = rolling_table.reset_index().rename(columns={rolling_table.index.name or "index": "model"})
    table["split"] = "validation"
    specialist_rows.append(table)
if specialist_rows:
    luxury_specialists = pd.concat(specialist_rows, ignore_index=True)
    luxury_specialists.to_csv(benchmark_out / "luxury_specialists.csv", index=False)
    display(luxury_specialists.round(4))"""
                )
            )
        )
    nbf.write(nb, path)


def neural_notebook(model_name):
    complexity_note = (
        " 当前候选约有 3.5 万至 6.5 万个参数，而每套数据按 7 天步长只有约 130 个训练窗口；TFT 在这里是复杂度上限基准，不是默认首选。"
        if model_name == "TFT"
        else ""
    )
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    nb.cells = [
        nbf.v4.new_markdown_cell(
            f"""# 用 {model_name} 做零售多步需求预测

本 Notebook 对商超、红酒、庭院和奢侈品手袋分别选择 {model_name} 配置。参数选择只使用最终测试期之前的四个滚动窗口；最终 28 天测试集仅评估一次。未来销量和模拟诊断真值不进入模型输入。{complexity_note}"""
        ),
        nbf.v4.new_markdown_cell("## 1. 数据、窗口与候选参数"),
        nbf.v4.new_code_cell(SETUP + "\nfrom retail_forecast.neural import *"),
        nbf.v4.new_code_cell(
            f"""MODEL_NAME = {model_name!r}
display(pd.DataFrame(NEURAL_CANDIDATES[MODEL_NAME]))
display(pd.DataFrame([{{"scenario": name, "days": len(frame), "zero_rate": float(frame.loc[frame.is_open, "y"].eq(0).mean())}} for name, frame in datasets.items()]))"""
        ),
        nbf.v4.new_markdown_cell("## 2. 滚动调参、最终训练与测试"),
        nbf.v4.new_code_cell(
            """result = run_neural_benchmark(MODEL_NAME)
m, p, tuning, selected, fits, explanations, out = result
display(selected)
display(tuning.sort_values(["scenario", "validation_MAE"]).round(4))
display(m.round(4))
display(fits.round(4))"""
        ),
        nbf.v4.new_markdown_cell("## 3. 最终预测与输入敏感度"),
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(22, 4), constrained_layout=True)
for ax, scenario in zip(axes, SCENARIOS):
    group = p.loc[(p.scenario == scenario) & p.split.eq("test")]
    ax.plot(group.ds, group.y, "ko-", ms=3, label="Observed")
    ax.plot(group.ds, group.prediction, label=MODEL_NAME)
    ax.set(title=scenario, ylabel="Units")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
fig.savefig(out / "final_test_predictions.png", dpi=150)
plt.show(); plt.close(fig)
display(explanations.sort_values(["scenario", "importance"], ascending=[True, False]).round(4))"""
        ),
        nbf.v4.new_markdown_cell("## 4. 导出与复现检查"),
        nbf.v4.new_code_cell(
            """required = {"metrics.csv", "predictions.csv", "tuning_results.csv", "selected_parameters.csv", "fit_details.csv", "explanations.csv", "manifest.json"}
assert required.issubset({path.name for path in out.iterdir()})
assert set(m.scenario) == set(SCENARIOS)
assert set(m.split) == {"train_hindcast", "validation", "test"}
for name, digest in INPUT_HASHES.items():
    assert hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() == digest
print("Exports:", out.resolve())"""
        ),
    ]
    return nb


def neural_comparison_notebook():
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
            "# MLP、LSTM 与 TFT 统一比较\n\n只读取当前输入哈希匹配的最新三组独立运行结果。"
        ),
        nbf.v4.new_code_cell(
            SETUP + "\nfrom retail_forecast.model_comparison import latest_compatible"
        ),
        nbf.v4.new_code_cell(
            """models = {"MLP": "neural_mlp", "LSTM": "neural_lstm", "TFT": "neural_tft"}
sources = {model: latest_compatible(directory) for model, directory in models.items()}
metrics_tables = [pd.read_csv(path / "metrics.csv") for path in sources.values()]
prediction_tables = [pd.read_csv(path / "predictions.csv", parse_dates=["ds", "forecast_origin"]) for path in sources.values()]
explanation_tables = [pd.read_csv(path / "explanations.csv") for path in sources.values()]
neural_metrics = pd.concat(metrics_tables, ignore_index=True)
neural_predictions = pd.concat(prediction_tables, ignore_index=True)
neural_explanations = pd.concat(explanation_tables, ignore_index=True)
display(neural_metrics.groupby(["scenario", "model", "split"], as_index=False)[["MAE", "RMSE", "WAPE", "Bias"]].mean().round(4))"""
        ),
        nbf.v4.new_code_cell(
            """out = RUNS_DIR / "neural_comparison"
out.mkdir(parents=True, exist_ok=False)
summary = neural_metrics.groupby(["scenario", "model", "split"], as_index=False)[["MAE", "RMSE", "WAPE", "Bias"]].mean()
summary.to_csv(out / "model_comparison.csv", index=False)
neural_predictions.to_csv(out / "predictions.csv", index=False)
neural_explanations.to_csv(out / "explanations.csv", index=False)
import json
(out / "manifest.json").write_text(json.dumps({"family": "neural_comparison", "models": list(models), "protocol_version": "retail-benchmark-v2", "hashes": INPUT_HASHES}, indent=2))
fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(22, 4), constrained_layout=True)
for ax, scenario in zip(axes, SCENARIOS):
    group = neural_predictions.loc[(neural_predictions.scenario == scenario) & neural_predictions.split.eq("test")]
    first = group.loc[group.model.eq("MLP")]
    ax.plot(first.ds, first.y, "ko-", ms=3, label="Observed")
    for model in models:
        values = group.loc[group.model.eq(model)]
        ax.plot(values.ds, values.prediction, label=model)
    ax.set(title=scenario); ax.legend()
fig.savefig(out / "neural_test_predictions.png", dpi=150)
plt.show(); plt.close(fig)"""
        ),
    ]
    return nb


def all_comparison_notebook():
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
            """# 四种需求风格 × 十个核心预测模型

比较 Prophet、ARIMA、五种树模型、MLP、LSTM 与 TFT；Seasonal naive 仅作共同参考。所有输入必须通过当前 CSV 哈希验证。商超、红酒、庭院和奢侈品分别解读，不把单一总平均作为模型选择依据。"""
        ),
        nbf.v4.new_code_cell(
            SETUP + "\nfrom retail_forecast.model_comparison import *"
        ),
        nbf.v4.new_markdown_cell("## 1. 载入并验证所有模型结果"),
        nbf.v4.new_code_cell(
            """tables, sources = load_all_results()
display(pd.DataFrame([{"model": model, "sources": "; ".join(map(str, paths))} for model, paths in sources.items()]))
summary, out = export_all_comparison(tables, sources)
assert set(CORE_MODELS).issubset(set(summary.model))
assert set(summary.scenario) == set(SCENARIOS)"""
        ),
        nbf.v4.new_markdown_cell("## 2. 按需求风格比较训练、验证与测试"),
        nbf.v4.new_code_cell(
            """display(summary.sort_values(["scenario", "split", "WAPE"])[["scenario", "model", "split", "MAE", "RMSE", "WAPE", "Bias", "n", "improvement_vs_naive", "test_minus_train_WAPE"]].round(4))
display(summary.loc[summary.split.eq("test")].pivot(index="scenario", columns="model", values=["MAE", "RMSE", "WAPE", "Bias"]).round(4))
plot_all_comparison(tables, summary, out)"""
        ),
        nbf.v4.new_markdown_cell("## 3. 训练成本与模型解释"),
        nbf.v4.new_code_cell(
            """display(tables["timing"].round(4))
display(tables["tuning"].groupby(["scenario", "model"], as_index=False).tuning_seconds.sum().round(4))
for family in ["Prophet", "MLP", "LSTM", "TFT"]:
    section = tables["explanations"].loc[tables["explanations"].model.eq(family)]
    if len(section):
        print(family)
        display(section.sort_values(["scenario", "importance"], ascending=[True, False]).groupby("scenario").head(10).round(4))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 4. 奢侈品专项与解释边界

奢侈品约 94% 的营业日销量为零，应优先查看 MAE、RMSE 与 Bias；WAPE 只在实际销量分母非零时解释。Croston-SBA、TSB 和 Two-stage 留在奢侈品 Prophet Notebook 的专项表中，不进入四场景总榜。模型排名只对当前数据、共同切分、特征和有限调参预算成立。"""
        ),
        nbf.v4.new_code_cell("print('Exports:', out.resolve())"),
    ]
    return nb


def main():
    extend_prophet(
        "10-prophet-retail-tutorial.ipynb",
        ["grocery", "wine", "garden"],
        "prophet_retail",
    )
    extend_prophet(
        "11-prophet-luxury-tutorial.ipynb", ["luxury"], "prophet_luxury", luxury=True
    )
    for filename, model in [
        ("33a-mlp-tutorial.ipynb", "MLP"),
        ("33b-lstm-tutorial.ipynb", "LSTM"),
        ("33c-tft-tutorial.ipynb", "TFT"),
    ]:
        nbf.write(neural_notebook(model), NOTEBOOKS / filename)
    nbf.write(
        neural_comparison_notebook(), NOTEBOOKS / "33d-neural-model-comparison.ipynb"
    )
    nbf.write(all_comparison_notebook(), NOTEBOOKS / "34-all-model-comparison.ipynb")


if __name__ == "__main__":
    main()
