# 2026 零售周 V4 数据科学分析报告

> 分析对象：`20261106-timesfm-v1`<br>
> 主榜：`2026-retail-week-expanding-v4`<br>
> 生产模拟：`2026-retail-week-104w-rolling-v4`<br>
> 评估范围：2025-12-29 至 2026-12-27，共 13 个非重叠 28 日 fold<br>
> 报告性质：基于固定合成数据和单一随机种子的描述性实验分析，不宣称统计显著性

## 1. 结论摘要

V4 的核心结论不是某个模型在所有零售问题上获胜，而是不同需求机制对应不同的有效建模
策略，并且“拥有更多历史”与“采用更复杂工具”都不保证自动改善预测。

1. **扩展历史主榜反映当前信息条件下的能力上限。** Grocery、Wine、Garden 的最佳普通
   代表分别达到 `4.16%`、`22.10%`、`7.80%` pooled WAPE。前三名主要来自 Prophet、
   Regression + ARIMA 和 LSTM/TFT，而不是同一个工具包垄断所有场景。
2. **104 周生产模拟检验的是有限训练窗口的稳健性。** Grocery 和 Wine 的 41 个配置中，
   分别有 30 和 32 个配置在扩展历史下更好；但最佳模型的绝对退化较小。Garden 对历史
   长度基本中性，说明两个完整零售年度已经覆盖其主要年度周期。
3. **Luxury 尚未被任何当前模型真正解决。** 主榜最低 WAPE 是 CatBoost 的 `100.25%`，
   Demand Event F1 只有 `9.09%`；TimesFM 和 AutoGluon 的约 `100%` WAPE 对应 `0%`
   Event F1。低 WAPE 在这里主要表示模型很少产生假阳性，不表示识别了购买事件。
4. **固定 240 秒 AutoML 没有超过最佳普通代表。** 在 Grocery、Wine、Garden，扩展历史下
   最佳 AutoGluon 分别落后 `1.36`、`2.93`、`2.39` 个 WAPE 百分点；104 周下分别落后
   `1.47`、`4.27`、`2.47` 个百分点。Luxury 的名义优势来自全零式预测，没有事件价值。
5. **TimesFM 的价值主要来自零样本基线与协变量吸收能力。** 已知协变量在扩展历史下将
   Grocery、Wine、Garden WAPE 分别降低 `0.37`、`1.60`、`0.69` 个百分点；104 周下
   分别降低 `0.26`、`1.21`、`0.31` 个百分点。它仍未解决 Luxury。
6. **Tweedie 是场景和工具包相关的配置，不是统一默认。** 它能改善部分 GBDT，特别是
   Wine 和 Luxury 的名义 WAPE，但 Luxury 的 Tweedie 模型 Event F1 仍为 0，不能将大幅
   WAPE 降幅解释成事件预测能力。

## 2. 证据、协议与指标口径

两套实验使用完全相同的四个场景、13 个测试 fold、28 日预测长度、冻结模型配置、有效标签
规则和预测起点。唯一实验因素是训练历史：主榜使用预测起点以前的全部历史，生产模拟只用
最近 104 个完整周。详细协议见 [V4 实验设计](experiment-design.md)。

| 证据 | SHA-256 |
| :--- | :--- |
| 扩展历史 release manifest | `4a086433f728cfa48ea301b4876a011ef1a6b4ca3b42eb32f55053224c34c51e` |
| 104 周 release manifest | `510d3994fb62dd5b5fa661d591702d4bc4fc1d2268201ba88d516cc6297a57ac` |
| 配对历史比较 release manifest | `3711fa3d1cbc6f3a8874d3a7f534cb374cf3168d7b23941a0aa056508508c385` |

本报告以 pooled WAPE 排名：先把全部有效逐日绝对误差与实际需求重新汇总，再计算 WAPE。
`13-fold WAPE 均值 ± 标准差`只描述各月波动，因为简单平均会让低需求 fold 与高需求 fold
拥有相同权重。MASE 以季节朴素误差为尺度；小于 1 表示优于该尺度基准。Bias 为带方向的
相对误差。Luxury 还必须联合 Demand Recall、Event F1 和正需求 MAE 阅读。

数据源：

- [扩展历史广度榜](../../data/benchmarks/published/2026-retail-week-expanding-v4/releases/20261106-timesfm-v1/summary/breadth_scorecard.csv)
- [104 周广度榜](../../data/benchmarks/published/2026-retail-week-104w-rolling-v4/releases/20261106-timesfm-v1/summary/breadth_scorecard.csv)
- [配对历史策略比较](../../data/benchmarks/published/comparisons/2026-retail-week-expanding-v4--vs--2026-retail-week-104w-rolling-v4/releases/20261106-timesfm-v1/training_history_comparison.csv)

## 3. 扩展历史主榜说明了什么

主榜回答：“在每个预测起点使用此前全部可用历史时，各工具包预先声明的代表配置能达到
什么水平？”它适合比较模型表达能力和完整历史利用能力，不等同于固定成本的生产部署。

### 3.1 四场景前三名

| 场景 | 排名 | 普通广度代表 | Pooled WAPE | Fold WAPE 均值 ± 标准差 | Bias | MASE | Event F1 |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| Grocery | 1 | Prophet | 4.16% | 4.18% ± 0.35% | +0.19% | 0.186 | 100.00% |
| Grocery | 2 | Regression + ARIMA (1,0,1) | 4.17% | 4.19% ± 0.35% | +0.19% | 0.187 | 100.00% |
| Grocery | 3 | LSTM | 4.44% | 4.45% ± 0.53% | -0.19% | 0.199 | 100.00% |
| Wine | 1 | Regression + ARIMA (1,0,1) | 22.10% | 22.27% ± 2.94% | +2.14% | 0.430 | 100.00% |
| Wine | 2 | Prophet | 22.78% | 23.09% ± 2.92% | -1.03% | 0.443 | 100.00% |
| Wine | 3 | LSTM | 23.20% | 23.41% ± 3.48% | +2.54% | 0.451 | 100.00% |
| Garden | 1 | Regression + ARIMA (1,0,1) | 7.80% | 8.94% ± 3.41% | -0.21% | 0.590 | 100.00% |
| Garden | 2 | Prophet | 7.81% | 8.94% ± 3.39% | -0.23% | 0.591 | 100.00% |
| Garden | 3 | TFT | 8.76% | 10.01% ± 3.69% | -1.11% | 0.662 | 100.00% |
| Luxury | 1 | CatBoost Default (MAE) | 100.25% | 100.55% ± 2.52% | -93.16% | 0.418 | 9.09% |
| Luxury | 2 | TimesFM 3.0 | 100.35% | 100.42% ± 0.75% | -99.61% | 0.418 | 0.00% |
| Luxury | 3 | Moving Average (14d) | 161.82% | 176.39% ± 77.90% | -24.88% | 0.674 | 0.00% |

### 3.2 场景分析

**Grocery：结构化统计模型已经接近误差下限。** Prophet 与 ARIMA 只相差 `0.006` 个
WAPE 百分点，远小于约 `0.35` 个百分点的 fold 标准差。配对结果中 ARIMA 赢 7 个 fold、
Prophet 赢 6 个 fold，因此不能把 pooled 排名第一解释为 Prophet 稳定支配 ARIMA。LSTM
达到 `4.44%`，证明神经模型可竞争，但在当前单序列和强可解释周期结构下没有形成明确优势。

**Wine：ARIMA 的领先更一致，但促销脉冲仍主导误差。** ARIMA 在 13 个 fold 中对 Prophet
取得 9 胜 4 负，pooled WAPE 低 `0.68` 个百分点。前三名 fold 标准差约 `2.9`–`3.5`
个百分点，明显高于 Grocery，符合低量泊松噪声和促销脉冲带来的月度不稳定性。ARIMA 的
`+2.14%` Bias 也说明最低绝对误差并不等于完全无偏。

**Garden：ARIMA 与 Prophet 实质上处于同一水平。** 两者 pooled WAPE 只差 `0.0075`
个百分点，配对 fold 为 7 比 6，fold 均值甚至由 Prophet 略低。TFT 排第三，但与前两名有
约 `0.95` 个百分点差距。结果支持显式年度结构对平滑季节场景有效，但不足以宣称 ARIMA
在统计上优于 Prophet。

**Luxury：当前主榜没有业务上令人满意的赢家。** CatBoost 的 `-93.16%` Bias 表示严重
低估，Demand Recall 只有 `4.76%`，Event F1 为 `9.09%`。TimesFM 的 Bias 接近 `-100%`
且 Event F1 为 0。两者约 `100%` 的 WAPE 本质上对应保守地预测“几乎没有需求”。因此
Luxury 的模型选择不能使用 WAPE 单指标；当前结果最多说明 CatBoost 偶尔识别到事件。

## 4. 104 周生产模拟说明了什么

生产模拟回答：“如果每四周重新运行一次，只保留预测月以前最近 104 个完整周，模型是否
仍能维持表现？”它控制训练数据规模并贴近固定成本 MLOps，但仍是单序列合成回测，不包含
真实数据延迟、计划修订、概念漂移、并发吞吐或服务故障。

### 4.1 四场景前三名

| 场景 | 排名 | 普通广度代表 | Pooled WAPE | Fold WAPE 均值 ± 标准差 | Bias | MASE | Event F1 |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| Grocery | 1 | Regression + ARIMA (1,0,1) | 4.26% | 4.28% ± 0.43% | +0.11% | 0.184 | 100.00% |
| Grocery | 2 | Prophet | 4.28% | 4.30% ± 0.41% | +0.08% | 0.185 | 100.00% |
| Grocery | 3 | LSTM | 4.62% | 4.61% ± 0.52% | -0.02% | 0.200 | 100.00% |
| Wine | 1 | Regression + ARIMA (1,0,1) | 22.46% | 22.67% ± 3.14% | +3.17% | 0.424 | 100.00% |
| Wine | 2 | Prophet | 22.49% | 22.71% ± 3.01% | -1.94% | 0.425 | 100.00% |
| Wine | 3 | LSTM | 23.74% | 24.01% ± 3.45% | +1.70% | 0.448 | 100.00% |
| Garden | 1 | Prophet | 7.84% | 9.01% ± 3.55% | +0.04% | 0.603 | 100.00% |
| Garden | 2 | Regression + ARIMA (1,0,1) | 7.85% | 9.01% ± 3.49% | +0.10% | 0.605 | 100.00% |
| Garden | 3 | TimesFM 3.0 | 8.86% | 9.84% ± 3.23% | -1.61% | 0.682 | 100.00% |
| Luxury | 1 | TimesFM 3.0 | 100.87% | 101.22% ± 1.59% | -99.06% | 0.431 | 0.00% |
| Luxury | 2 | CatBoost Default (MAE) | 111.81% | 105.35% ± 9.58% | -78.17% | 0.477 | 16.00% |
| Luxury | 3 | Prophet | 142.28% | 140.49% ± 35.27% | -49.33% | 0.607 | 0.00% |

### 4.2 训练历史的配对影响

| 场景 | 配置数 | 扩展历史胜 | 104 周胜 | 平局 | 104w−扩展历史 WAPE 中位数 | 广度配置中位数 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Grocery | 41 | 30 | 4 | 7 | +0.19 pp | +0.18 pp |
| Wine | 41 | 32 | 2 | 7 | +0.75 pp | +0.54 pp |
| Garden | 41 | 23 | 13 | 5 | +0.03 pp | +0.05 pp |
| Luxury | 41 | 17 | 19 | 5 | +0.00 pp | -3.58 pp |

**Grocery：更长历史有小幅但一致的价值。** ARIMA 从 `4.17%` 变为 `4.26%`，Prophet
从 `4.16%` 变为 `4.28%`，LSTM 从 `4.44%` 变为 `4.62%`。前三名没有结构性崩溃，
所以 104 周是可行的成本约束；但 30/41 配置支持扩展历史，不能声称截断历史会改善常规品。

**Wine：扩展历史的总体价值最明确。** 32/41 配置由扩展历史获胜，中位差 `+0.75 pp`。
不过最佳 ARIMA 只退化 `0.36 pp`，Prophet 反而改善约 `0.29 pp`。这说明总体方向不能替代
逐模型检查；固定窗口可能通过遗忘较早数据减少某些模型的结构偏差。

**Garden：104 周已经覆盖两个完整年度周期。** 41 个配置的中位差仅 `+0.03 pp`，Prophet
和 ARIMA 都只变化约 `0.03`–`0.05 pp`。在当前稳定年度生成机制下，更早历史的边际信息很
有限。真实业务若存在长期趋势、气候变化或品类生命周期，这个结论可能不成立。

**Luxury：WAPE 胜负不能解释历史价值。** 104 周下 CatBoost WAPE 从 `100.25%` 升至
`111.81%`，但 Event F1 从 `9.09%` 升至 `16.00%`；TimesFM WAPE 仅升至 `100.87%`，
Event F1 仍为 0。较高 WAPE 与较好的事件召回可以同时出现，说明生产决策必须先定义漏报与
误报成本，再选择阈值和目标函数。

## 5. AutoGluon：固定预算 AutoML 是否带来增益

AutoGluon 使用每个场景、每个 fold 固定 240 秒的共享训练套件。以下前三名只在 15 个
`ag-*` 输出内部排名，不进入普通工具包广度榜。

| 场景 | 扩展历史 AutoML 前三（pooled WAPE） | 104 周 AutoML 前三（pooled WAPE） |
| :--- | :--- | :--- |
| Grocery | Weighted Ensemble 5.52%；DirectTabular 5.53%；DeepAR 5.75% | DirectTabular 5.73%；Weighted Ensemble 6.06%；RecursiveTabular 6.20% |
| Wine | DeepAR 25.03%；DirectTabular 25.24%；Weighted Ensemble 26.20% | Weighted Ensemble 26.72%；DeepAR 26.97%；DirectTabular 26.98% |
| Garden | Weighted Ensemble 10.19%；DirectTabular 10.23%；DeepAR 10.24% | DirectTabular 10.31%；ETS 10.43%；DeepAR 10.43% |
| Luxury | NPTS 100.00%；Weighted Ensemble 100.00%；Zero 100.00% | NPTS 100.00%；Zero 100.00%；DeepAR 100.02% |

### 5.1 与最佳普通代表比较

| 协议 | 场景 | 最佳普通代表 | 最佳 AutoML | AutoML−普通 |
| :--- | :--- | :--- | :--- | ---: |
| 扩展历史 | Grocery | Prophet 4.16% | Weighted Ensemble 5.52% | +1.36 pp |
| 扩展历史 | Wine | ARIMA 22.10% | DeepAR 25.03% | +2.93 pp |
| 扩展历史 | Garden | ARIMA 7.80% | Weighted Ensemble 10.19% | +2.39 pp |
| 扩展历史 | Luxury | CatBoost 100.25%、F1 9.09% | NPTS 100.00%、F1 0% | -0.25 pp，事件更差 |
| 104 周 | Grocery | ARIMA 4.26% | DirectTabular 5.73% | +1.47 pp |
| 104 周 | Wine | ARIMA 22.46% | Weighted Ensemble 26.72% | +4.27 pp |
| 104 周 | Garden | Prophet 7.84% | DirectTabular 10.31% | +2.47 pp |
| 104 周 | Luxury | TimesFM 100.87%、F1 0% | NPTS 100.00%、F1 0% | -0.87 pp，无事件增益 |

AutoML 的主要观察是**模型选择不稳定且 ensemble 不保证最优**：扩展历史 Wine 的最佳组件
是 DeepAR，104 周变为 Weighted Ensemble；Garden 则从 Weighted Ensemble 变为
DirectTabular。固定预算让比较具有工程意义，但不能证明更长搜索预算仍不会改善结果。
当前证据只支持：在这四条合成单序列和每次 240 秒预算下，领域结构明确的普通模型更有效。

数据源：[扩展历史 AutoML](../../data/benchmarks/published/2026-retail-week-expanding-v4/releases/20261106-timesfm-v1/summary/automl_scorecard.csv)、
[104 周 AutoML](../../data/benchmarks/published/2026-retail-week-104w-rolling-v4/releases/20261106-timesfm-v1/summary/automl_scorecard.csv)。

## 6. TimesFM 3.0：零样本基础模型与已知协变量

`univariate` 是 TimesFM 的普通广度代表；`known_covariates` 是独立配置消融。两者均为
冻结 checkpoint 的零样本推理，不使用 2026 标签选择配置。

### 6.1 扩展历史

| 场景 | 配置 | Pooled WAPE | Fold 均值 ± 标准差 | Bias | Coverage80 | 区间宽度 | Pinball80 | Event F1 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Grocery | known covariates | 4.54% | 4.55% ± 0.50% | -1.06% | 81.45% | 18.15 | 1.278 | 100% |
| Grocery | univariate | 4.91% | 4.91% ± 0.69% | -0.65% | 83.19% | 19.92 | 1.491 | 100% |
| Wine | known covariates | 22.69% | 22.67% ± 2.90% | -3.11% | 77.07% | 8.05 | 0.590 | 100% |
| Wine | univariate | 24.29% | 24.78% ± 5.24% | -2.51% | 75.69% | 8.26 | 0.625 | 100% |
| Garden | known covariates | 8.17% | 9.32% ± 3.32% | -0.23% | 79.56% | 18.75 | 1.261 | 100% |
| Garden | univariate | 8.86% | 9.79% ± 3.07% | -0.58% | 74.03% | 18.38 | 1.451 | 100% |
| Luxury | known covariates | 100.41% | 100.32% ± 0.32% | -99.45% | 91.44% | 0.02 | 0.040 | 0% |
| Luxury | univariate | 100.35% | 100.42% ± 0.75% | -99.61% | 93.65% | 0.08 | 0.041 | 0% |

### 6.2 固定 104 周

| 场景 | 配置 | Pooled WAPE | Fold 均值 ± 标准差 | Bias | Coverage80 | 区间宽度 | Pinball80 | Event F1 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Grocery | known covariates | 4.93% | 4.93% ± 0.59% | -1.70% | 83.48% | 20.21 | 1.417 | 100% |
| Grocery | univariate | 5.19% | 5.19% ± 0.83% | -1.39% | 85.22% | 21.48 | 1.556 | 100% |
| Wine | known covariates | 23.98% | 24.31% ± 4.93% | -4.97% | 77.90% | 8.31 | 0.624 | 100% |
| Wine | univariate | 25.19% | 25.80% ± 5.76% | -2.52% | 76.24% | 8.42 | 0.666 | 100% |
| Garden | known covariates | 8.55% | 9.68% ± 3.31% | -1.24% | 78.45% | 18.77 | 1.294 | 100% |
| Garden | univariate | 8.86% | 9.84% ± 3.23% | -1.61% | 78.73% | 19.24 | 1.454 | 100% |
| Luxury | known covariates | 100.47% | 100.56% ± 0.59% | -99.44% | 88.67% | 0.004 | 0.040 | 0% |
| Luxury | univariate | 100.87% | 101.22% ± 1.59% | -99.06% | 87.57% | 0.33 | 0.044 | 0% |

已知协变量在三个非极端间歇场景同时降低 pooled WAPE 和 pinball loss，说明模型确实利用了
预测起点已知的促销、营业和日历信息。改善在 Wine 最大，符合促销与事件变量对脉冲需求的
直接作用。104 周下相同方向仍然存在，说明该收益不是仅由更长历史造成。

区间指标必须谨慎解释。Luxury 的 Coverage80 高达 `87.57%`–`93.65%`，但 Event F1 为
0，原因是绝大多数日期真实值为零，极窄的近零区间可以覆盖大量零销量日。这不是库存服务
水平保证，也不能证明对真正购买事件进行了概率校准。

## 7. GBDT Tweedie 配置消融

下表给出每个 GBDT 工具包在各场景的最佳预声明配置，以及相对该工具包 default 的 pooled
WAPE 变化。负值表示改善。

| 场景 | 工具包 | 扩展历史最佳配置（相对 default） | 104 周最佳配置（相对 default） |
| :--- | :--- | :--- | :--- |
| Grocery | LightGBM | default，0.00 pp | default，0.00 pp |
| Grocery | XGBoost | Tweedie p=1.1，-0.15 pp | Tweedie p=1.5，-0.28 pp |
| Grocery | CatBoost | Tweedie p=1.2，-0.11 pp | default，0.00 pp |
| Wine | LightGBM | Tweedie p=1.5，-1.16 pp | Tweedie p=1.5，-0.69 pp |
| Wine | XGBoost | default，0.00 pp | Tweedie p=1.1，-0.39 pp |
| Wine | CatBoost | Tweedie p=1.8，-1.15 pp | Tweedie p=1.8，-0.68 pp |
| Garden | LightGBM | Tweedie p=1.2，-0.03 pp | Tweedie p=1.2，-0.47 pp |
| Garden | XGBoost | Tweedie p=1.8，-0.15 pp | Tweedie p=1.1，-0.38 pp |
| Garden | CatBoost | Tweedie p=1.5，-0.13 pp | Tweedie p=1.5，-0.48 pp |
| Luxury | LightGBM | Tweedie p=1.8，-88.35 pp | Tweedie p=1.8，-66.38 pp |
| Luxury | XGBoost | Tweedie p=1.8，-114.82 pp | Tweedie p=1.8，-120.49 pp |
| Luxury | CatBoost | default，0.00 pp | default，0.00 pp |

三个结论可以成立：第一，最佳 power 随工具包、场景和历史窗口变化；第二，Wine 的计数脉冲
更经常从 Tweedie 获益；第三，Luxury 的巨大 WAPE 改善不能单独作为成功证据，因为这些
LightGBM/XGBoost Tweedie 配置的 Event F1 仍为 0。Tweedie 在这里主要抑制持续的小幅正
预测，没有学会购买事件发生机制。

数据源：[扩展历史配置消融](../../data/benchmarks/published/2026-retail-week-expanding-v4/releases/20261106-timesfm-v1/summary/configuration_ablations.csv)、
[104 周配置消融](../../data/benchmarks/published/2026-retail-week-104w-rolling-v4/releases/20261106-timesfm-v1/summary/configuration_ablations.csv)。

## 8. 面向模型选择与生产设计的结论

### 8.1 场景级建议

| 场景 | 当前首选候选 | 生产含义 | 主要风险 |
| :--- | :--- | :--- | :--- |
| Grocery | ARIMA / Prophet，LSTM 作为 challenger | 104 周窗口损失很小；优先选择可维护性、区间和运行成本合适者 | 当前合成促销与季节机制较规则，真实缺货去偏仍需验证 |
| Wine | ARIMA / Prophet；TimesFM + covariates 为强 challenger | 保留促销、派对和 12 月计划变量；监控 fold 波动与 Bias | 低量脉冲导致月度误差方差较大 |
| Garden | Prophet / ARIMA；TimesFM + covariates 可比较 | 104 周已覆盖主要年度周期，截断更早历史几乎无损 | 真实长期趋势和气候变化未进入生成协议 |
| Luxury | 当前没有可直接推荐的单阶段模型 | 应转向 occurrence + positive size 的两阶段模型、阈值成本和事件指标 | WAPE、MAE、Coverage 都可能奖励全零式预测 |

### 8.2 工程决策

- 不应为了 `0.01` 个百分点以内的描述性差异承担显著更高维护成本；Grocery 和 Garden 的
  ARIMA/Prophet 应视为同一候选层级，进一步选择需要运行时、区间校准和真实数据证据。
- 104 周固定窗口对 Grocery、Wine、Garden 的最佳模型没有造成结构性失效，适合作为生产
  起点；但扩展历史总体仍占优，应保留按场景验证更长历史的能力。
- AutoML 适合作为候选发现和基线生成工具，而不是默认生产赢家。当前固定预算下，它没有
  抵消领域结构明确模型的优势。
- TimesFM 适合作为低运维零样本 challenger，已知未来变量值得保留；默认权重的非商业
  许可证和 Luxury 事件失效阻止其直接成为统一生产方案。
- 所有区间都需要面向目标服务水平重新校准；Coverage80 只是经验覆盖率。

## 9. 当前证据不能回答什么

本报告不支持以下外推：

- 不支持“ARIMA、Prophet 或 TimesFM 在真实零售中普遍优于其他模型”；
- 不支持基于单个合成随机种子宣称统计显著性；
- 不支持把四种不同需求尺度平均成一个总冠军；
- 不支持把 AutoGluon 的 240 秒预算结果外推到无限搜索预算；
- 不支持把 TimesFM 的高 Coverage80 转换为库存服务水平；
- 不支持根据 Luxury 的 WAPE 选择生产模型。

下一轮实验的优先级应是：多随机种子复现；Luxury 两阶段 occurrence/size 模型；按预测周
拆分的 horizon profile；促销、缺货与事件分层误差；区间校准和业务成本函数。继续增加模型
数量的科学收益低于验证这些结论是否跨数据生成重复成立。
