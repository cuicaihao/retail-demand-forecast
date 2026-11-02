# TimesFM 与时序基础模型在四个合成零售场景上的受控实验计划（Experiment Plan & Spec）

> **用途说明**：本文档面向 `retail-demand-forecast` 仓库的开发者与 AI Agent。请按照本文档定义的环境配置、数据契约、四组受控实验步骤、可视化图表规格以及报告模板，在本地执行实验并最终生成数据分析报告 `articles/timesfm-experiment-report.md` 及配套图片 `images/tsfm/*.png`，供主博客《拆解 Google TimesFM》第五章选择性引用。

---

## 一、 实验背景与总体目标

在本仓库现有的教程中，我们已经使用 **4 套固定随机种子生成的墨尔本超市教学合成数据集**（位于 `data/teaching/`，由 `notebooks/00-generate-teaching-data.ipynb` 生成）系统检验了 `Seasonal Naive`、`Prophet` 与 `LightGBM` 的表现。

本实验模块的目标是：**在完全相同的时间切分、信息可用边界与评价指标口径下，引入 Google 的时序原生基础模型 TimesFM（以及可选的 Chronos-2 作为横向对照），通过四组受控实验回答以下四个核心工程问题**：

1. **全历史基线对决（Full-History Benchmark）**：在拥有完整 3 年历史（1,068 天训练/上下文，预测最后 28 天）的条件下，纯单变量零样本 `TimesFM` 与带外生协变量的 `TimesFM + XReg` 分别在**高频杂货**、**脉冲促销红酒**、**强季节庭院耗材**、**高零值奢侈品手袋**四个场景上表现如何？与调优后的 `Prophet` 和 `LightGBM` 差距在哪里？
2. **缺货删失污染测试（Stockout Censoring Sensitivity）**：在包含 24 个缺货日（上限被截断为 20 件）的杂货场景中，如果直接将未去偏的原始观测销量 `y` 喂给 `TimesFM`，相比输入经过缺货去偏插补后的序列，预测偏差（Bias）和 WAPE 会恶化多少？
3. **冷启动历史截断测试（Cold-Start Context Truncation）**：当输入历史从 `1,068` 天依次缩短为紧邻测试集前的 `90` 天、`28` 天、`14` 天（模拟新上架商品或新开门店）时，`Prophet`、`LightGBM` 与 `TimesFM (+ XReg)` 的预测误差如何变化？是否存在一个“历史长度拐点”使得基础模型反超传统模型？
4. **概率预测区间校准（Probabilistic Quantile Calibration）**：`TimesFM` 原生输出的 P10–P90（80% 预测区间）在四个场景下能否有效包络真实需求波动？

---

## 二、 仓库目录结构与环境准备

### 2.1 新增目录与模块结构（对齐现有仓库架构）

为与仓库现有的 `src/retail_forecast/` 模块体系保持一致，建议采用如下目录组织（在 `develop` 分支中开发与运行）：

```text
retail-demand-forecast/
├── articles/
│   ├── figures/                        # 博客文章配图归档
│   ├── timesfm-experiment-plan.md      # 本实验计划文档
│   └── timesfm-experiment-report.md    # [待生成] 实验完成后输出的数据分析报告
├── data/
│   └── teaching/                       # 00-generate-teaching-data.ipynb 生成的四套标准 CSV
├── images/
│   └── tsfm/                           # [待生成] 存放本实验生成的三张核心对比图表 (.png & .webp)
├── notebooks/
│   └── 00-generate-teaching-data.ipynb # 现有数据生成 Notebook
├── src/
│   └── retail_forecast/
│       ├── common.py                   # 统一数据契约、切分窗口与指标计算基础
│       └── timesfm_benchmark.py        # [新增] TimesFM 与基线受控实验核心实现
└── tsfm_benchmark/                     # [新增] 时序基础模型基准测试入口
    ├── README.md                       # 模块运行说明
    ├── run_timesfm_experiments.py      # 主实验脚本（一键运行四组实验并保存 CSV/JSON 与图表）
    └── evaluate_metrics.py             # 统一指标计算工具（WAPE, MAE, RMSE, Bias, Coverage）
```

### 2.2 依赖安装与模型权重选择

1. **确保教学数据已生成**：
   若 `data/teaching/` 下尚无四个场景的 CSV 文件，先运行 `notebooks/00-generate-teaching-data.ipynb`（或将其中的数据生成函数提取调用）。
2. **安装 TimesFM 依赖**：
   推荐使用 Apache-2.0 协议开源、支持 PyTorch 后端且原生支持 `forecast_with_covariates`（XReg）与连续分位数头的 **TimesFM 2.5（200M）**（Hugging Face ID：`google/timesfm-2.5-200m-pytorch`）：
   ```bash
   # 在仓库根目录下通过 uv 安装依赖（或创建可选 dependency group）
   uv add --group tsfm torch timesfm lightgbm prophet matplotlib pandas numpy scikit-learn
   ```
   *(注：若本地环境安装 `timesfm` 遇到 JAX/XReg 依赖问题，请确保同时安装用于 XReg 求解的线性代数后端，或提供一个纯 PyTorch/Scikit-learn `Ridge` 等价实现的 `xreg + timesfm` 对照函数以保证 100% 可复现。若环境允许，也可顺带安装 `chronos-forecasting` 测试 `amazon/chronos-2` 作为横向参考。)*

---

## 三、 数据契约与四个合成场景口径核对

四个场景的数据均覆盖 `2023-01-01` 至 `2025-12-31`（共 `1,096` 天）：
- **训练/上下文截止日（Forecast Origin）**：`2025-12-03`（前 `1,068` 天为历史可见期）；
- **测试评价窗口（Evaluation Horizon）**：`2025-12-04` 至 `2025-12-31`（共 `H = 28` 天）；
- **闭店日处理**：耶稣受难日（Good Friday）和圣诞节（`2025-12-25`）门店闭店（`is_open == False`），真实观测销量为 `0`。所有模型输出的营业日需求预测在最终转换为实际销量预测 `forecast_units` 时，均应乘以已知的未来营业状态指示变量 `is_open`（即闭店日预测强制归零后再计算测试集指标，或按现有 Prophet 教程口径统一在有效营业日上评估，请在报告中明确注明并与原有 Prophet 4.15% / LightGBM 4.36% 口径严格对齐）。

| 场景名称 | 对应数据文件（`data/teaching/`） | 目标列与真值列 | 未来已知协变量（Known Future Covariates） | 序列核心特征 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 杂货（Grocery）** | `grocery_scenario_data.csv` | 观测 `y`；无噪声均值 `conditional_mean`；潜在需求 `latent_demand` | `is_open`, `is_closed_holiday`, `discount`（每 14 天前 7 天为 0.20）, VIC 公共假日（`vic_holidays.csv`） | 高频连续（均值 109.45），含 **24 个随机缺货日**（`stockout == True` 时 `y` 被截断为 20） |
| **2. 红酒（Wine）** | `wine_scenario_data.csv` | 观测 `y`；`conditional_mean`；`latent_demand` | `is_open`, `is_closed_holiday`, `promo_case_buy`（每 28 天前 3 天成箱促销）, `is_party_event`（AFL/墨尔本杯/跨年）, `is_december_peak` | 泊松计数序列（均值 10.38，$\widehat{\mathrm{ADI}}=1.013$），带强烈短时脉冲 |
| **3. 庭院耗材（Garden）** | `garden_scenario_data.csv` | 观测 `y`；`conditional_mean`；`latent_demand` | `is_open`, `is_closed_holiday`, `is_long_weekend_diy`, `is_end_of_autumn_clearance` | 强年正弦周期 + 周末效应（均值 64.92，$\widehat{\mathrm{ADI}}=1.000$），无缺货 |
| **4. 奢侈品手袋（Luxury Handbag）** | `luxury_scenario_data.csv` | 观测 `y`；发生率 `occurrence_prob`；`conditional_mean`；`latent_demand` | `is_open`, `is_closed_holiday`, `is_saturday`, `is_sunday`, `is_december_peak`, `is_vip_event`, `discount` | **极端稀疏间歇性需求**（营业日零值率 **93.58%**，$\widehat{\mathrm{ADI}}=15.571$，$\widehat{\mathrm{CV}}_+^2=0.949$） |

---

## 四、 四组受控实验的具体步骤

### 实验 1：四场景全历史基准对决（Full-History Benchmark, Context = 1,068d）

**目的**：检验在历史充足时，TimesFM 单变量零样本与带协变量模式在四种不同需求形态下的表现。

**执行步骤**：
1. 对每个场景（杂货场景先使用将缺货日 `stockout == True` 视为缺失并完成历史插补去偏后的清洁目标序列，以对齐 Prophet 与 LightGBM 的公平训练输入），切分前 1,068 天为历史输入，最后 28 天为测试集。
2. 依次运行以下 6 个模型配置，预测未来 28 天：
   - `M1: Seasonal Naive (7d)`：$\hat{y}_{T+h} = y_{T+h-7}$（若遇历史闭店/缺货则向前顺延同星期有效值）；
   - `M2: Prophet (With Regressors)`：复现本仓库现有教程配置（含 VIC 公共假日与场景对应的已知促销回归项）；
   - `M3: LightGBM (Tabular Baseline)`：构造 `lag_1, lag_7, lag_14, lag_21, lag_28`、`rolling_mean_7, rolling_mean_28`、`dayofweek, month` 及已知促销/假日特征；
   - `M4: TimesFM (Univariate Zero-shot)`：仅传入长度为 1,068 的一维历史序列，不传入任何协变量（最后乘回已知的 `is_open`）；
   - `M5: TimesFM + XReg (xreg + timesfm)`：传入历史序列及长度为 $1068 + 28 = 1096$ 的动态协变量（如 `discount`、假日独热特征、成箱促销等），采用默认模式先由 Ridge Regression 剥离促销线性效应，再由 TimesFM 预测残差；
   - `M6: TimesFM + XReg (timesfm + xreg)`：先由 TimesFM 预测基线，再对残差拟合协变量回归。
3. 对于场景 4（奢侈品手袋），额外增加 `M7: Croston / SBA`（间歇性需求经典方法）与 `M8: LightGBM (Tweedie Loss)`，对比 TimesFM 在零膨胀序列上的表现。
4. 汇总所有模型在四个场景上的 **28 天 WAPE (%)、MAE、RMSE、Bias (%)、80% 区间覆盖率（P10–P90 Coverage）与单次推理耗时（ms）**。

---

### 实验 2：缺货删失污染实验（Stockout Censoring Sensitivity，仅限场景 1 杂货）

**目的**：验证“垃圾进、垃圾出”在时序基础模型上同样成立——如果不修复缺货截断，零样本基础模型的实例归一化与局部切片会被污染。

**执行步骤**：
1. 在杂货场景（含 24 个 `stockout == True`、观测销量被硬截断为 `20` 件的日期）中，保持测试集真值为最后 28 天营业日的无约束真实需求（或非缺货实际销量，请同时报告对观测有效日和 `latent_demand` 的指标）。
2. 对比以下两组输入喂给 `TimesFM (Univariate)` 与 `TimesFM + XReg`：
   - **Group A（Raw Censored Input）**：不对历史期做任何缺货处理，直接将 `y`（含 `20` 件缺货记录和 `0` 件闭店记录）作为历史上下文传入；
   - **Group B（Unbiased / Imputed Input）**：将历史期中 `stockout == True` 或 `is_open == False` 的点标记为缺失，先用简单的同星期滑动均值或上下文 Ridge 回归插补还原后，再传入 TimesFM。
3. 记录并对比 Group A 与 Group B 的 **预测偏差 Bias (%)、WAPE (%) 以及 P10/P50/P90 分位数带的整体下移幅度**。

---

### 实验 3：冷启动上下文截断压力测试（Cold-Start Truncation Curve）

**目的**：模拟新品上架或新店开业，寻找 `TimesFM + XReg` 反超 `LightGBM` 与 `Prophet` 的历史长度临界点。

**执行步骤**：
1. 选取场景 1（杂货）、场景 2（红酒）、场景 3（庭院耗材）。固定测试评价窗口仍为最后 28 天（`2025-12-04` 至 `2025-12-31`）。
2. 将训练/上下文窗口长度 $L$ 依次设置为紧邻 `2025-12-03` 之前的 **6 个梯度**：
   \[
   L \in \{1068,\, 365,\, 90,\, 56,\, 28,\, 14\} \text{ 天}
   \]
3. 在每个历史长度 $L$ 下，分别运行：
   - `Prophet (With Regressors)`（观察当 $L < 365$ 时开启/关闭 `yearly_seasonality` 的表现）；
   - `LightGBM`（仅使用当前长度 $L$ 内可计算的滞后特征，观察样本行数骤减与长滞后列缺失导致的退化）；
   - `TimesFM (Univariate Zero-shot)`（利用随机前缀掩码机制直接输入长度为 $L$ 的短序列）；
   - `TimesFM + XReg (xreg + timesfm)`（在长度为 $L$ 的短窗口上结合 L2 正则岭回归与 TimesFM 预训练先验）。
4. 记录 6 个长度梯度下各模型的 28 天 WAPE (%)，绘制“**WAPE 随历史长度 $L$ 缩短的退化曲线**”。

---

### 实验 4：概率分位数区间评估（Quantile Calibration & Inventory Implication）

**目的**：评估 TimesFM 输出的 P10–P90 分位数在补货安全库存设定中的实用性。

**执行步骤**：
1. 提取 `TimesFM`（及 `TimesFM + XReg`）在四个场景下输出的分位数预测（`P10, P20, ..., P90`）。
2. 计算 **80% 预测区间（`[P10, P90]`）的实际覆盖率（Empirical Coverage）** 与 **平均区间宽度（Mean Interval Width）**，并与 `Prophet` 的 80% 预测区间进行对比。
3. 检查在场景 4（奢侈品手袋）中，TimesFM 的低分位数（如 P10、P20）是否出现负值，以及截断为 $\max(0, \hat{y}^{(q)})$ 后的分布形态。

---

## 五、 需要生成的三张核心图表规格（保存在 `images/tsfm/`）

请使用清晰、出版级的 Matplotlib 样式（300 DPI，留白得当，字体清晰，同时保存 `.png` 并在可能时转换一份 `.webp`）生成以下三张图表：

1. **`images/tsfm/fig-tsfm-01-four-scenarios-comparison.png`（四场景 28 天预测曲线与区间对比四宫格图）**：
   - **布局**：$2 \times 2$ 子图，分别对应 **杂货（Grocery）**、**红酒（Wine）**、**庭院耗材（Garden）**、**奢侈品手袋（Luxury Handbag）**。
   - **内容**：横轴为最后 28 天日期（或前推 14 天历史 + 后 28 天预测以展示衔接），纵轴为销量。绘制真实值（黑色实线/圆点）、`Prophet`（蓝色虚线）、`LightGBM`（绿色点划线）、`TimesFM Univariate`（橙色细线）、`TimesFM + XReg`（红色粗实线及淡红色 P10–P90 阴影带）。在杂货和红酒子图中用浅灰色竖条标出促销窗口。
2. **`images/tsfm/fig-tsfm-02-coldstart-degradation-curve.png`（冷启动历史长度截断退化曲线）**：
   - **布局**：$1 \times 3$ 子图（杂货、红酒、庭院耗材）。
   - **内容**：横轴为历史上下文长度 $L \in \{14, 28, 56, 90, 365, 1068\}$（对数刻度或等距分类刻度），纵轴为 28 天测试集 WAPE (%)。对比 `Prophet`、`LightGBM`、`TimesFM Univariate` 与 `TimesFM + XReg` 四条折线，直观标出在 $L=14$ 和 $L=28$ 天时基础模型相比传统树模型的冷启动优势区间。
3. **`images/tsfm/fig-tsfm-03-stockout-censoring-impact.png`（缺货删失对 TimesFM 预测基线的污染诊断图）**：
   - **布局**：左右双图或上下双层图（针对杂货场景）。
   - **内容**：展示历史期中 `stockout == True`（销量跌至 20 件）的删失点如何拉低 `TimesFM (Raw Input)` 的预测均值与分位数区间，以及 `TimesFM (Unbiased Imputed Input)` 修复缺货后如何恢复到真实潜在需求水平。

---

## 六、 最终交付物：数据分析报告模板（`articles/timesfm-experiment-report.md`）

实验脚本运行完毕后，请将所有量化结果、图表路径与分析结论汇总写入 `articles/timesfm-experiment-report.md`，报告结构请包含：

1. **实验环境与模型版本记录**：Python 版本、PyTorch 版本、所加载的 TimesFM 具体权重名称（如 `google/timesfm-2.5-200m-pytorch`）、运行硬件（CPU/GPU 型号）及单序列平均推理耗时。
2. **实验 1 汇总表（四场景全历史 1,068 天基准对比表）**：完整列出 4 个场景 $\times$ 各模型的 `WAPE (%)`、`MAE`、`RMSE`、`Bias (%)`、`80% Coverage (%)` 与 `Inference Time (ms)`。
3. **实验 2 汇总表（杂货场景缺货未去偏 vs. 去偏对比表）**：列出 Raw vs. Imputed 在 `TimesFM` 上的各项指标差异及具体日期（如促销期与非促销期）的误差拆解。
4. **实验 3 汇总表（冷启动 6 个历史长度梯度 $14\text{d} \sim 1068\text{d}$ 的 WAPE 矩阵）**：精确列出每个长度下各模型的 WAPE，指出交叉拐点发生在多少天。
5. **实验 4 汇总与关键异常诊断**：总结分位数覆盖率表现，以及 TimesFM 在红酒 3 天脉冲促销和奢侈品手袋 93.6% 零值场景下的具体行为特征。
6. **核心结论提炼（供主博客直接引用）**：用 3–5 条精炼的定量结论总结本次实测发现。

