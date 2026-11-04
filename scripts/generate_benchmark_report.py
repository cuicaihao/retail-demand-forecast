"""
scripts/generate_benchmark_report.py
------------------------------------
Synthesizes annual 52-week rolling walk-forward benchmark results from
`data/benchmarks/summary/` into:
1. Internal Technical Validation Report:
   `docs/reports/2026-annual-retail-forecast-benchmark.md`
2. Public Technical Blog Post:
   `docs/articles/2026-retail-demand-forecast-benchmark.md`

Guarantees 0 retraining cost and reproducible markdown reporting.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
SUMMARY_DIR = ROOT_DIR / "data" / "benchmarks" / "summary"
REPORTS_DIR = ROOT_DIR / "docs" / "reports"
ARTICLES_DIR = ROOT_DIR / "docs" / "articles"


def load_summaries():
    annual_df = pd.read_csv(SUMMARY_DIR / "annual_leaderboard.csv")
    scenario_df = pd.read_csv(SUMMARY_DIR / "scenario_leaderboard.csv")
    trajectories_df = pd.read_csv(SUMMARY_DIR / "period_trajectories.csv")
    return annual_df, scenario_df, trajectories_df


def format_table(
    df: pd.DataFrame,
    columns: list[str],
    rename_cols: dict[str, str],
    float_formats: dict[str, str],
) -> str:
    sub = df[columns].copy()
    for col, fmt in float_formats.items():
        if col in sub.columns:
            sub[col] = sub[col].apply(
                lambda x, f=fmt: f.format(x) if pd.notnull(x) else "-"
            )
    sub = sub.rename(columns=rename_cols)
    headers = list(sub.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in sub.iterrows():
        vals = [str(row[h]) for h in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def generate_technical_report(
    annual_df: pd.DataFrame, scenario_df: pd.DataFrame, trajectories_df: pd.DataFrame
):
    report_path = REPORTS_DIR / "2026-annual-retail-forecast-benchmark.md"

    # Leaderboard table
    annual_table = format_table(
        annual_df,
        columns=[
            "model_name",
            "Mean_WAPE",
            "Mean_MAE",
            "Mean_RMSE",
            "Mean_Bias",
            "Mean_ZeroAccuracy",
            "Stability_Std",
        ],
        rename_cols={
            "model_name": "模型与配置",
            "Mean_WAPE": "全场景平均 WAPE",
            "Mean_MAE": "平均 MAE",
            "Mean_RMSE": "平均 RMSE",
            "Mean_Bias": "平均 Bias",
            "Mean_ZeroAccuracy": "零值准确率",
            "Stability_Std": "周期波动性 Std",
        },
        float_formats={
            "Mean_WAPE": "{:.2%}",
            "Mean_MAE": "{:.3f}",
            "Mean_RMSE": "{:.3f}",
            "Mean_Bias": "{:+.3f}",
            "Mean_ZeroAccuracy": "{:.1%}",
            "Stability_Std": "{:.3f}",
        },
    )

    # Scenarios breakdown
    scenarios = ["grocery", "wine", "garden", "luxury"]
    scenario_tables = {}
    for sc in scenarios:
        sc_data = scenario_df[scenario_df["scenario"] == sc].sort_values("WAPE")
        scenario_tables[sc] = format_table(
            sc_data,
            columns=[
                "model_name",
                "WAPE",
                "MAE",
                "RMSE",
                "Bias",
                "ZeroAccuracy",
                "Period_WAPE_Mean",
                "Period_WAPE_Std",
            ],
            rename_cols={
                "model_name": "模型",
                "WAPE": "年度累计 WAPE",
                "MAE": "MAE",
                "RMSE": "RMSE",
                "Bias": "Bias",
                "ZeroAccuracy": "零值准确率",
                "Period_WAPE_Mean": "13期均值 WAPE",
                "Period_WAPE_Std": "跨期波动 Std",
            },
            float_formats={
                "WAPE": "{:.2%}",
                "MAE": "{:.3f}",
                "RMSE": "{:.3f}",
                "Bias": "{:+.3f}",
                "ZeroAccuracy": "{:.1%}",
                "Period_WAPE_Mean": "{:.2%}",
                "Period_WAPE_Std": "{:.3f}",
            },
        )

    content = f"""# 2026 全年 52 周前向滑动窗口零售需求预测基准评测报告

> **基准协议**: `retail-benchmark-v3 (Walk-Forward Rolling Evaluation)`  
> **评测年份**: 2026 年完整 52 周（13 期无重叠 28 天块步进，`2025-12-29` 至 `2026-12-27`，共 364 天）  
> **数据范围**: 2022-07-04 至 2027-06-27（5 年共 260 完整周，1,820 天）  
> **回看训练长度**: 每期前 24 完整周（168 天）严格历史滑动窗口  
> **参评模型家族**: 朴素基线（Seasonal Naive, Moving Average）、统计时间序列（Prophet）、经典树模型（Decision Tree, Random Forest）、现代梯度提升树（LightGBM, XGBoost, CatBoost）及其损失函数变体（MSE vs MAE vs Tweedie $p \\in [1.1, 1.2, 1.5, 1.8]$）

---

## 一、 执行摘要 (Executive Summary)

本报告是对澳大利亚墨尔本（VIC）四类经典零售场景在 2026 完整业务年度开展的**无数据泄露、高频滑动窗口时序交叉验证**评测总结。

相比于传统的单一切片（Single 28-day Cutoff），**13 期跨度一整年的前向滚动评测（Walk-Forward Rolling Evaluation）**彻底消除了“单一测试窗口巧合偏误”，完整穿透了秋季清仓、复活节、黑五大促、圣诞以及极端淡旺季。

### 核心发现与业务洞察：

1. **CatBoost (MAE) 与 XGBoost Tweedie (p=1.8) 荣登年度全场景综合前两强**：
   - **CatBoost Default (MAE)** 取得了 **40.99%** 的全场景平均 WAPE，在奢侈品场景采取极度保守策略将 MAE 压至 0.084；
   - **XGBoost Tweedie (p=1.8)** 取得 **41.92%** 全场景平均 WAPE，同时在周期稳定性标准差上表现卓越（$\\sigma=0.097$），在商超（6.87%）、红酒（27.90%）、庭院（16.68%）、奢侈品（116.22%）均保持第一梯队竞争力。
2. **Tweedie 目标函数在大促脉冲与零膨胀间歇性需求上形成质的突破**：
   - 在 **奢侈品手袋场景**（93.6% 零销量），标准 MSE 的 XGBoost 将 WAPE 推高至 **187.54%**，生成了大量 0.05~0.15 的浮点噪声预测（零值准确率仅 46.3%）。
   - **XGBoost Tweedie (p=1.8)** 将 WAPE 骤降至 **116.22%**（相对改善 **71.3 个百分点**），零值分类准确率提升至 **88.0%**，跨期波动标准差从 0.803 压缩至 **0.225**。
   - 在 **商超快消品连续需求** 下，**XGBoost Tweedie (p=1.2)**（6.59%）甚至超越了默认 MSE（6.83%），打破了“Tweedie 只适用于间歇数据”的传统认知。
3. **不同零售形态的绝对主导模型**：
   - **商超 (Grocery, FMCG)**: **XGBoost Tweedie ($p=1.2$)** 夺冠（WAPE 6.59%），紧随其后的是 LightGBM 与 Prophet。
   - **红酒 (Wine, 泊松脉冲大促)**: **XGBoost Tweedie ($p=1.8$)** 夺冠（WAPE 27.90%），Random Forest 与其他 XGBoost 变体紧随其后。
   - **庭院 (Garden, 正弦天气季节性)**: **Prophet** 夺冠（WAPE 14.02%），充分体现了广义可加模型在显式傅里叶分解长周期年度波动的强大能力。
   - **奢侈品 (Luxury, 零膨胀长尾)**: **CatBoost Default (MAE)** 与 **XGBoost Tweedie ($p=1.8$)** 形成绝对断层式优势。

![2026 全年 52 周各场景排行榜](../../images/benchmarks/fig-01-annual-leaderboard-all-scenarios.png)

---

## 二、 2026 全年总榜单 (Global Annual Leaderboard)

以下为 15 种模型及配置在 2026 年（13 个 28 天周期，四场景共 52 次独立滚动前向评测）的加权平均汇总指标：

{annual_table}

> **指标说明**：
> - **Mean_WAPE**: 4 个场景 2026 全年累计 WAPE 的简单算术平均。
> - **Mean_ZeroAccuracy**: 对实际为 0 的标签，模型预测值落在 $[0, 0.05)$ 内的准确率。
> - **Stability_Std**: 该模型在 13 个独立评价周期（P01-P13）中 WAPE 的标准差，越小代表模型抗季节性剧烈波动的鲁棒性越强。

---

## 三、 四类零售形态深度剖析 (Scenario Breakdown)

### 3.1 商超高频快消品 (Grocery)

商超品类日均销量约 109.5 件，需求高度连续，受每周工作日/周末循环及折扣强度驱动。

{scenario_tables["grocery"]}

- **分析**: 在高信噪比连续需求场景下，所有现代 GBDT 模型均将 WAPE 控制在 7.5% 以内。**XGBoost Tweedie ($p=1.2$)** 获得最低误差（6.59%），证明微调 Tweedie 幂方指数使损失函数偏向复合泊松-伽马特征对轻微右偏的日常零售数据同样具有微观正则化增益。

### 3.2 单 SKU 红酒泊松脉冲 (Wine)

单 SKU 红酒日均销量约 10.4 件，具有典型的“成箱购买（Case Buy）”非线性大促突刺、派对事件与 12 月节日高峰。

{scenario_tables["wine"]}

- **分析**: **XGBoost Tweedie ($p=1.8$)** 实现了 **27.90%** 的最优 WAPE。树模型在成箱促销等阶跃信号上明显优于 Prophet（29.91%）和 Seasonal Naive（49.26%）。

### 3.3 庭院用品强季节性 (Garden)

庭院用品日均销量约 64.9 件，呈现极为显著的跨年正弦波天气季节性（春秋旺、冬季低谷）与长周末 DIY 脉冲。

{scenario_tables["garden"]}

- **分析**: **Prophet** 以 **14.02%** 的 WAPE 拔得头筹！Prophet 的先验傅里叶级数对长达一年的平滑气候波动拟合得非常自然，而树模型受限于仅有 24 周（约半年）的历史窗口，对跨年季节性周期的外推相对吃力（XGBoost MSE 为 14.94%）。

### 3.4 奢侈品手袋极端零膨胀 (Luxury)

93.6% 的营业日销量为 0，年均仅有十余次偶发购买，属于典型的极端间歇性（Intermittent & Zero-Inflated）需求。

{scenario_tables["luxury"]}

- **分析**: 这是检验模型损失函数成色最严酷的试金石。
  - 传统基于高斯正态假设的 MSE 模型（如 XGBoost Default MSE 和 LightGBM Default MSE）彻底失效，WAPE 高达 **182%~187%**，它们因试图规避大残差惩罚而倾向于对每天均预测一个微小的正浮点数（~0.1），导致 50% 以上的无销量日产生虚假库存预测。
  - **XGBoost Tweedie ($p=1.8$)** 凭借其对零点质量集中的似然建模，将 WAPE 压低至 **116.22%**，并将零值预测准确率拉升至 **88.0%**！

![零销售日混淆与浮点噪声预测对比](../../images/benchmarks/fig-04-intermittent-zero-confusion.png)

---

## 四、 GBDT Tweedie 目标函数消融实验深度分析

为了深入探索 Tweedie 方差幂指数 $p$（Variance Power Parameter, $Var(Y) = \\phi \\mu^p$）在不同需求分布中的连续响应，我们在 XGBoost、LightGBM 与 CatBoost 上实施了系统的超参退火分析：

![GBDT Tweedie 目标函数消融深度剖析](../../images/benchmarks/fig-03-gbdt-tweedie-ablation-deepdive.png)

### 4.1 数学原理映射与梯度行为

在 $1 < p < 2$ 时，Tweedie 对应 Compound Poisson–Gamma 分布。损失函数（负对数似然）对预测对数对数值 $s = \\log \\mu$ 的一阶梯度（Gradient）与二阶导（Hessian）为：
$$
g_i = -y_i \\exp\\left((1-p) s_i\\right) + \\exp\\left((2-p) s_i\\right)
$$
$$
h_i = -y_i (1-p) \\exp\\left((1-p) s_i\\right) + (2-p) \\exp\\left((2-p) s_i\\right)
$$

- 当 $y_i = 0$ 时，梯度简化为 $g_i = \\exp\\left((2-p) s_i\\right) > 0$，强力推动模型将预测值推向负无穷大（即 $\\mu \\to 0$）；
- 当 $p \\to 2$ 时，$2-p \\to 0$，残差对正样本大值的相对尺度保持尺度不变性（Scale-invariant Gamma 效应），从而抑制了离群极值对树分裂的过度牵引。

### 4.2 消融实验核心结论

1. **随着 $p$ 从 1.1 递增至 1.8，间歇性场景指标单调显著改善**：在 Luxury 场景中，XGBoost 的 WAPE 从 $p=1.1$ 的 175.13% 单调降至 $p=1.5$ 的 140.92%，再降至 $p=1.8$ 的 116.22%。
2. **平滑连续数据对 $p \\in [1.1, 1.2]$ 表现极佳**：Grocery 场景中，$p=1.2$ 获得 6.59% 最低 WAPE。
3. **CatBoost Tweedie 的实现特异性**：CatBoost 内置的 Tweedie 目标函数在小样本零膨胀时可能遇到数值下溢限制，生产环境中更推荐使用其默认 MAE 或切换为经过充分调优的 XGBoost/LightGBM Tweedie。

---

## 五、 13 期前向滑动轨迹与季节波动稳定性

单一切片评测往往掩盖了模型在不同淡旺季的剧烈性能波动。通过跟踪 13 期（P01 至 P13）的连续 WAPE 轨迹，我们揭示了各模型的动态鲁棒性：

![13期滑动轨迹跨季节波动](../../images/benchmarks/fig-02-13-period-wape-trajectories.png)

- **波动性分析 ($\\sigma$)**：
  - XGBoost Tweedie (p=1.8) 不仅总误差极低，其跨期波动率 $\\sigma = 0.097$，远低于 Default MSE 的 $\\sigma = 0.242$；
  - 传统基线 Seasonal Naive 的波动率高达 $\\sigma = 0.345$，在非稳态季节切换时极易发生步长脱节。

---

## 六、 生产落地与架构推荐

根据本 2026 全年基准测试结果，我们为零售企业时序补货与预测系统提出以下分层落地建议：

1. **分级路由架构 (Hierarchical Scenario Routing)**：
   - 对日均销量 $\\ge 50$ 且连续的品类（如商超），采用 **XGBoost (Tweedie $p=1.2$ 或 MSE)**；
   - 对具有强烈气候物理背景和年周期的品类（如庭院、泳装、滑雪），保留 **Prophet** 或加入显式傅里叶特征的 GBDT；
   - 对低销量脉冲与间歇长尾品类（如酒类、奢侈品、长尾配件），**强制禁用标准 MSE/RMSE 目标函数**，全面切换为 **XGBoost Tweedie ($p=1.8$)** 或 **CatBoost (MAE)**。
2. **持续评测保障 (Continuous Rolling Evaluation)**：
   - 弃用静态测试集保留法，全面采用与本仓库一致的 **周度对齐 28 天无重叠滑动窗口（Walk-Forward 4-Week Block）** 持续回测。
"""

    report_path.write_text(content, encoding="utf-8")
    print(f"Technical Validation Report written to: {report_path}")


def generate_blog_article(
    annual_df: pd.DataFrame, scenario_df: pd.DataFrame, trajectories_df: pd.DataFrame
):
    article_path = ARTICLES_DIR / "2026-retail-demand-forecast-benchmark.md"

    annual_top5 = format_table(
        annual_df.head(6),
        columns=[
            "model_name",
            "Mean_WAPE",
            "Mean_MAE",
            "Mean_ZeroAccuracy",
            "Stability_Std",
        ],
        rename_cols={
            "model_name": "模型与配置",
            "Mean_WAPE": "平均 WAPE",
            "Mean_MAE": "平均 MAE",
            "Mean_ZeroAccuracy": "零值准确率",
            "Stability_Std": "跨期波动 Std",
        },
        float_formats={
            "Mean_WAPE": "{:.2%}",
            "Mean_MAE": "{:.3f}",
            "Mean_ZeroAccuracy": "{:.1%}",
            "Stability_Std": "{:.3f}",
        },
    )

    lux_comp = format_table(
        scenario_df[scenario_df["scenario"] == "luxury"].sort_values("WAPE").head(7),
        columns=["model_name", "WAPE", "MAE", "ZeroAccuracy", "Period_WAPE_Std"],
        rename_cols={
            "model_name": "模型与配置",
            "WAPE": "年度 WAPE",
            "MAE": "MAE",
            "ZeroAccuracy": "零值准确率",
            "Period_WAPE_Std": "波动性 Std",
        },
        float_formats={
            "WAPE": "{:.2%}",
            "MAE": "{:.3f}",
            "ZeroAccuracy": "{:.1%}",
            "Period_WAPE_Std": "{:.3f}",
        },
    )

    content = f"""---
title: "零售需求预测 2026 年度基准：52 周滑动窗口评测与 Tweedie 分布实践"
date: 2026-11-05T09:00:00+11:00
draft: false
summary: "基于墨尔本 5 年合成零售数据集，全面实施 2026 全年 52 周（13 期 × 28 天）前向滑动窗口基准评测。涵盖 Baseline、统计时间序列与三代主流 GBDT，系统验证了 Tweedie 复合泊松-伽马损失函数在零膨胀与长尾零售需求中的理论与实证突破。"
tags: ["Demand Forecasting", "Tweedie", "LightGBM", "XGBoost", "CatBoost", "Prophet", "Benchmark", "MLOps"]
categories: ["时间序列预测", "机器学习工程"]
math: true
---

## 一、 为什么传统的“留出 28 天测试集”不可靠？

在零售需求预测和供应链补货的算法实践中，最常见的实验设计是：拿过去两年或三年的历史数据作为训练集，留出最后四周（28 天）作为测试集。

然而，时间序列数据存在强烈的非平稳性（Non-stationarity）与季节异质性：
- 如果最后四周恰逢**双十一或圣诞大促销**，高估模型的惩罚会被瞬间放大；
- 如果最后四周是**节后极寒淡季**，原本适应平稳需求的高斯模型会被大量零销量击穿。

为了建立具备真正生产落地参考价值的权威标准，我们将墨尔本（VIC）四类零售场景的教学数据集扩展至 **完整 5 年（2022-07-04 至 2027-06-27，共 260 完整周）**，并在 **2026 年完整 52 周** 上实施了 **13 期无重叠 28 天块步进前向滑动窗口评测（Walk-Forward 4-Week Block Evaluation）**。

![2026 全年 52 周各场景排行榜](../../images/benchmarks/fig-01-annual-leaderboard-all-scenarios.png)

---

## 二、 2026 全年榜单：谁统治了零售时序？

在整整 13 个测试期、涵盖商超快消、脉冲红酒、强季节庭院与极端零膨胀奢侈品的 52 次严格盲测中，前五名模型脱颖而出：

{annual_top5}

### 核心结论一览：
1. **CatBoost (MAE) 凭借极度保守性拿下全场景加权第一**（40.99% WAPE），在极度零膨胀的奢侈品场景几乎全押零值，从而取得了极低的名义绝对误差；
2. **XGBoost Tweedie (p=1.8) 成为工业界最具实战价值的通用王者**（41.92% WAPE），且稳定性极高（$\\sigma=0.097$），在连续需求（商超 6.87%）、脉冲需求（红酒 27.90%）和零膨胀长尾（奢侈品 116.22%）中展现了全面碾压传统 MSE 的统治力；
3. **Prophet 在年度长周期正弦拟合上依旧不可替代**，在庭院场景以 14.02% 斩获第一。

---

## 三、 零膨胀长尾难题：Tweedie 分布的数学降维打击

零售中最令人头疼的是**低频高值长尾品类**（如奢侈品高价包包，93.6% 的日期销量为 0）。

### 3.1 MSE 的理论破产：浮点噪声与虚假库存
当我们使用基于均方误差（MSE）的经典回归时，模型背后隐含的假设是误差服从高斯分布：
$$
\\epsilon_i \\sim \\mathcal{{N}}(0, \\\\sigma^2)
$$
为了避免预测 0 时偶尔出现的正销量带来巨大的平方惩罚，树模型被迫在每一个叶节点给出一个微小的正数期望值（例如 $0.08$ 件）。
- **后果**：每天都有虚假库存预测，零值预测准确率暴跌至 **46.3%**，WAPE 高达 **187.54%**！

### 3.2 Tweedie 复合泊松-伽马分布解法
Tweedie 分布将“是否有购买发生”（泊松过程 $N \\sim \\text{{Pois}}(\\lambda)$）与“单次购买金额/数量”（伽马分布 $Z \\sim \\text{{Gamma}}(\\alpha, \\gamma)$）端到端融合为单分布。

在 $1 < p < 2$ 时，其负对数似然对对数打分 $s = \\log \\mu$ 的梯度为：
$$
g_i = -y_i \\exp\\left((1-p) s_i\\right) + \\exp\\left((2-p) s_i\\right)
$$

当样本真实销量 $y_i = 0$ 时，梯度仅保留正惩罚项 $\\exp\\left((2-p) s_i\\right)$，以极大的确定性驱动叶节点预测逼近 0！

![零销售日混淆与浮点噪声预测对比](../../images/benchmarks/fig-04-intermittent-zero-confusion.png)

在奢侈品场景的实际对抗中，指标变化惊人：

{lux_comp}

- **XGBoost Tweedie ($p=1.8$)** 相较于默认 MSE，**WAPE 暴降 71.3 个百分点**，零值识别准确率从 46.3% 飙升至 **88.0%**！
- 跨越 13 个周期的性能波动标准差从 0.803 收敛至 **0.225**。

---

## 四、 GBDT Tweedie 参数消融：$p$ 应该怎么选？

在 XGBoost 与 LightGBM 中，Tweedie 方差幂指数 $p$ 是一个核心超参数。通过退火实验，我们绘制出了清晰的工程选择路线：

![GBDT Tweedie 目标函数消融深度剖析](../../images/benchmarks/fig-03-gbdt-tweedie-ablation-deepdive.png)

1. **高频连续型快消（Grocery）**：推荐选择 $p \\in [1.1, 1.2]$。此时稍微引入泊松收缩，不仅保留了连续分布的弹性，还能微调高斯偏态，实测 WAPE（6.59%）优于纯 MSE（6.83%）。
2. **大促脉冲与间歇零膨胀（Wine, Luxury）**：坚决推向上限 $p \\in [1.7, 1.9]$（推荐 $p=1.8$）。此时分布充分吸收伽马长尾和零点质量集中特性，模型在非促销日能坚定输出接近 0 的预测，在大促日释放长尾弹性。

---

## 五、 穿越四季：13 期滑动窗口稳定性全景

![13期滑动轨迹跨季节波动](../../images/benchmarks/fig-02-13-period-wape-trajectories.png)

从 P01（年初夏季）到 P06（复活节及冬季淡季），再到 P12-P13（黑五与圣诞旺季），时序误差轨迹清晰表明：
- 依靠单一切片评价模型是极其危险的。传统季节朴素法（Seasonal Naive）在平稳期表现尚可，但在大促切换月份误差直冲 100% 以上（$\\sigma=0.345$）；
- 具备 Tweedie 似然对齐的现代 GBDT 模型，展示出了极强的全天候鲁棒性。

---

## 六、 总结与复现指引

本研究确立了零售时序基准评测的三大标准原则：
1. **时间单位与业务周期对齐**：统一采用以周一为起点的 28 天完整周期，彻底消除月份天数不均的统计扭曲；
2. **分层分治模型路由**：连续场景用轻度 Tweedie 提升树，气候正弦场景用 Prophet，零膨胀长尾场景坚决换用高阶 Tweedie 或 MAE 树；
3. **完全无数据泄露与可复现性**：所有实验代码、数据与评测脚本均已开源并实现独立模块化解耦。

### 一键复现代码
```bash
# 运行单个模型（例如独立重算 XGBoost Tweedie）
uv run python scripts/run_benchmark.py --models xgboost_tweedie_p18

# 零开销重新生成全局排行榜与可视化
uv run python scripts/run_benchmark.py --compare-only
uv run python scripts/plot_benchmark_summary.py
uv run python scripts/generate_benchmark_report.py
```
"""

    article_path.write_text(content, encoding="utf-8")
    print(f"Blog Article written to: {article_path}")


def main():
    annual_df, scenario_df, trajectories_df = load_summaries()
    generate_technical_report(annual_df, scenario_df, trajectories_df)
    generate_blog_article(annual_df, scenario_df, trajectories_df)
    print("\nAll benchmark documentation and articles successfully generated!")


if __name__ == "__main__":
    main()
