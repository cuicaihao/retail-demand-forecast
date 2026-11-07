---
title: "[历史归档] 拆解 Google TimesFM：旧版单切片与 BigQuery 实验"
date: 2026-11-03T11:00:00+11:00
draft: true
summary: "深入剖析 Google Research 3.3 亿参数时序大模型 TimesFM 3.0 的架构演进、补丁化机制（Patching）与协变量建模原理。结合 BigQuery 原生托管 AI.FORECAST 算力，在商超、红酒、庭院与奢侈品四大受控零售场景中完成零样本对比、缺货删失去偏诊断与极短样本冷启动实测。"
tags: ["机器学习", "时间序列", "TimesFM", "Google Cloud", "BigQuery", "Demand Forecasting", "AI Engineering"]
categories: ["机器学习工程"]
math: true
---

> **历史归档说明**：本文记录旧版单一 28 天切片与 BigQuery 实验，不属于当前 V4
> 13-fold 基准结论。当前可比较结果请以
> [`2026-retail-week-v4-first-stage-report.md`](../../reports/2026-retail-week-v4-first-stage-report.md)
> 和 `20261106-timesfm-v1` release 为准。

## 一、引言：时序大模型走向企业级数仓

在零售需求预测领域，传统的建模路径通常分为两条：
1. **单序列统计拟合**（如 Prophet、ARIMA）：可解释性好、对业务假日与季节分解直观，但面对海量 SKU 时运维成本极高，无法跨序列共享规律；
2. **多序列全局机器学习**（如 LightGBM、TFT、DeepAR）：具备强大的特征交互能力，但严重依赖人工滑窗特征工程，且在新店/新品上市（冷启动 $L \le 28$ 天）时因特征历史窗口缺失而面临崩溃。

2024–2025 年，以 Google Research 开源的 **TimesFM (Time-series Foundation Model)** 为代表的时序基础模型正在重塑预测技术版图。而在 2025 年末，Google 进一步将 **TimesFM 3.0** 原生集成至 **BigQuery** 时序推理引擎（表函数 `AI.FORECAST`），使得企业用户无需部署复杂的 GPU 集群、无需维护 PyTorch / OpenMP 环境，即可在数仓内以 SQL 原生完成百万级 SKU 的零样本推理与分位数不确定性生成。

本文基于本仓库的四套合成零售数据（覆盖商超高频、红酒脉冲、庭院年季节与奢侈品极端稀疏场景），系统检验 TimesFM 3.0 的实战能力边界。

> 完整量化评测指标、实验方案与对比数据详见仓库报告：
> - 历史实验方案：[`reports/archive/2026-11-03-timesfm-experiment-plan.md`](../../reports/archive/2026-11-03-timesfm-experiment-plan.md)
> - 历史量化报告：[`reports/archive/2026-11-03-timesfm-validation-report.md`](../../reports/archive/2026-11-03-timesfm-validation-report.md)
> - 交互实操代码：[`notebooks/26-timesfm-tutorial.ipynb`](../../../notebooks/26-timesfm-tutorial.ipynb)
> - 历史 SQL 资产：[`src/retail_forecast/sql/archive_bigquery_timesfm_v2_experiments.sql`](../../../src/retail_forecast/sql/archive_bigquery_timesfm_v2_experiments.sql)

---

## 二、四大场景多模型全历史基准对决

在拥有 1,068 天（约 3 年）历史的基准测试中，我们对比了 **Seasonal Naive**、**Prophet**、**LightGBM** 与 **TimesFM 3.0 (BigQuery)** 在最后 28 天测试期的预测表现：

![四场景多模型预测对比四宫格](../../../images/tsfm/fig-tsfm-01-four-scenarios-comparison.png)

### 核心量化发现
1. **高频商超（Grocery）**：
   - 零样本 TimesFM 3.0 取得了 **5.13% WAPE** 与 **81.48% 的 80% 区间覆盖率**，紧逼调优后的 Prophet（4.22%），大幅领先 LightGBM（10.67%）；
2. **庭院耗材（Garden）**：
   - 在强冬夏正弦年周期下，TimesFM 3.0 凭借强大的 Patch 先验识别了跨年周期，WAPE 稳定在 **11.13%**，持平调优 Prophet（10.47%），而缺乏长周期锚定的 LightGBM 恶化至 39.73%。

---

## 三、缺货删失污染：基础模型能否“自愈”？

零售实际销售数据经常遭受**缺货删失（Stockout Censoring）**污染——当货架售空时，实际观测销量被截断在库存上限，导致观察值系统性低于潜在真实需求。

我们将商超场景中包含 24 个缺货截断日的原始污染序列直接输入 TimesFM 3.0，并与经过插补去偏的序列对比：

![缺货删失对 TimesFM 预测的系统性偏差](../../../images/tsfm/fig-tsfm-03-stockout-censoring-impact.png)

> **实证结论**：直接暴露缺货截断导致 TimesFM 预测产生 **-1.74% 的系统性负向偏差（Bias）**。这量化证实了：**时序大模型具备强大的形态拟合能力，但无法凭空识别业务语义上的隐性截断，上游去偏清洗不可替代**。

---

## 四、极短历史冷启动：基础模型的护城河

在新品上市或新店开业场景下，历史数据往往极短（$L = 14 \sim 28$ 天）。我们对比了样本长度从 1068 天极端压缩至 14 天时的模型衰减曲线：

![冷启动样本衰减曲线](../../../images/tsfm/fig-tsfm-02-coldstart-degradation-curve.png)

> **实证结论**：在 14 天极限冷启动下，LightGBM 的特征滑动窗口失效崩溃（WAPE 达 100%），Prophet 产生严重过拟合（WAPE 55.27%）。而 TimesFM 3.0 依托跨领域千亿级时序先验，在红酒场景将 WAPE 压制在 **43.79%**，显现出基础模型在长尾小样本下的绝对壁垒。

---

## 五、总结与落地决策树

1. **新品/长尾冷启动（$L \le 28$ 天）**：坚决推行 TimesFM 3.0，替代传统小样本容易崩溃的树模型；
2. **成熟稳定品（$L > 1$ 年）**：TimesFM 可作为免调参顶级基线，与精调统计模型协同；
3. **极端稀疏慢动品（零值率 $> 80\%$）**：配合两阶段门控或 Croston/SBA，不可直接依赖连续大模型。
