# Retail Demand Forecasting (零售需求预测教学与基准评测)

基于澳大利亚维多利亚州（VIC / 墨尔本）合成零售数据的可复现需求预测教程与多模型基准评测仓库。所有数据均为固定随机种子生成的教学合成数据，覆盖完整 5 年（2022-07-04 至 2027-06-27，共 260 完整周 / 1,820 天），用于受控检验不同预测模型在典型零售业务场景下的优势与局限，不代表任何真实企业经营数据。

## 📘 系列技术博客与报告 (Featured Blog Posts & Reports)

本仓库配套的技术博客与量化实证报告深度拆解了零售需求预测的建模机制、调参诊断、缺货去偏、Tweedie 损失函数与 52 周前向滑动窗口基准评测：

- **[2026 零售周 V4 完整基准报告](docs/reports/2026-retail-week-v4-first-stage-report.md)**
  使用审核后的扩展历史主榜和生产式 104 周滚动实验，分开呈现工具包广度、配置深度与训练历史影响。
- **[2026 零售周 V4 发布摘要](docs/reports/2026-retail-week-v4-release-summary.md)**
  一页汇总当前场景能力矩阵、AutoML、TimesFM 配置消融、104 周训练历史实验与发布边界。
- **[2026 零售周 V4 数据科学分析](docs/reports/2026-retail-week-v4-data-science-analysis.md)**
  深入解释扩展历史主榜、104 周生产模拟、AutoML、TimesFM、Tweedie 消融及其生产含义。
- **[TimesFM 3.0：13-fold 双协议与已知协变量消融](docs/articles/timesfm-retail-demand-forecast.md)**
  只引用 `20261106-timesfm-v1`，分析零样本预测、分位数区间、计算成本与 Luxury 事件识别边界。
- **[Retail Demand Forecast V4 实验设计](docs/reports/experiment-design.md)**
  固化当前日粒度、13 × 4 周滚动回测、双训练历史、指标、发布契约与新模型接入边界。
- **[零售需求预测实战：基于 Prophet 的建模、调参、诊断与局限性分析](https://cuicaihao.github.io/zh/posts/2026-10-01-prophet-retail-demand-forecast/)**  
  结合本仓库生成的四套合成零售数据，系统剖析 Prophet 的可加/可乘分解、VIC 州公共假日与促销回归项、缺货删失（Stockout Censoring）去偏、参数敏感性诊断，以及在脉冲促销与稀疏间歇需求下的结构性边界。

---

## 🛒 四大合成零售场景：本仓库的核心资产 (Synthetic Retail Scenarios)

本仓库最核心的学术与工程贡献，在于**抽象并严密实现了四个具有行业典型挑战的合成零售需求场景**（位于 `data/teaching/`，由 [scripts/generate_teaching_data.py](scripts/generate_teaching_data.py) 或 [notebooks/00-generate-teaching-data.ipynb](notebooks/00-generate-teaching-data.ipynb) 确定性生成）。每个场景代表现实供应链中不可回避的特定预测难题：

| 场景 | 数据文件 (`data/teaching/`) | 核心需求形态与业务机制 | 已知业务协变量 (Known Future Covariates) | 核心评测挑战与不可自愈陷阱 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 商超日用杂货 (Grocery)** | `grocery_scenario_data.csv` | 高频连续消费 + 强周/年周期 | `is_open`, `is_closed_holiday`, `discount`, VIC 公共假日 | 随机缺货截断与闭店，检验**缺货删失（Censoring）去偏**能力 |
| **2. 红酒促销 (Wine)** | `wine_scenario_data.csv` | 泊松计数脉冲 | `promo_case_buy`, `is_party_event`, `is_december_peak` | 短时促销脉冲与年末峰值，检验非线性特征吸收能力 |
| **3. 庭院耗材 (Garden)** | `garden_scenario_data.csv` | 强年度正弦周期 + 周末效应 | `is_long_weekend_diy`, `is_end_of_autumn_clearance` | 冬夏跨季振幅与长周末叠加，检验跨年长周期先验 |
| **4. 奢侈品手袋 (Luxury)** | `luxury_scenario_data.csv` | 极端稀疏间歇性需求 | `is_saturday`, `is_sunday`, `is_december_peak`, `is_vip_event`, `discount` | 零膨胀序列，检验全零陷阱、需求事件识别与分位数校准 |

精确均值、零值率、ADI、CV² 和缺货天数由 `scenario_diagnostics.csv` 生成，并由 `generation_metadata.json` 记录哈希；README 不再复制一份可能漂移的统计值。

![四场景需求形态与 ADI-CV 分布](images/retail-demand-profile-adi-cv.png)

---

## 🏆 2026 零售周 52 周基准库 (2026 Retail-Week Benchmark Store)

当前使用声明式 **V4 周对齐前向滚动协议**。主榜与生产模拟实验使用相同测试 fold，但禁止交叉聚合：

- **主榜 `2026-retail-week-expanding-v4`**：每个预测起点使用此前全部可用历史；
- **生产模拟 `2026-retail-week-104w-rolling-v4`**：每个月使用此前最近 104 个完整周训练，预测随后 4 个完整周；
- **严格兼容性**：协议、数据哈希、场景、fold、forecast origin、有效标签集合和冻结参数必须全部一致；
- **发布分层**：本地结果写入 `runs/`；审核后的不可变快照写入 `data/benchmarks/published/<benchmark-id>/releases/<release-id>/`，允许后续 release 增加昂贵模型而不改写旧证据。
- **实验层比较**：跨历史策略结果单独发布到 `data/benchmarks/published/comparisons/<comparison-id>/releases/<release-id>/`，并通过哈希固定两套源 release。

<!-- BENCHMARK_BLOCK:current_results:START -->
当前审核 release 为 `20261106-timesfm-v1`，包含 26 个普通配置和 15 个独立 AutoML 配置。
前三名统一按 pooled WAPE 排名；13-fold 均值与标准差只描述月度稳定性。
Luxury 必须联合 Event F1 阅读，约 100% WAPE 可能只是接近全零预测。

### 普通工具包广度榜

每个工具包只有一个预先声明的代表配置参与排名；AutoGluon 不进入本表。
TimesFM 单变量配置是其广度代表，因此达到前三名时会在本表出现。

#### 主榜：扩展历史

| 场景 | 排名 | 模型 | Pooled WAPE | 13-fold WAPE 均值 ± 标准差 | Event F1@0.5 |
| :--- | ---: | :--- | ---: | ---: | ---: |
| Grocery | 1 | Prophet | 4.16% | 4.18% ± 0.35% | 100.00% |
| Grocery | 2 | Regression + ARIMA (1,0,1) | 4.17% | 4.19% ± 0.35% | 100.00% |
| Grocery | 3 | LSTM | 4.44% | 4.45% ± 0.53% | 100.00% |
| Wine | 1 | Regression + ARIMA (1,0,1) | 22.10% | 22.27% ± 2.94% | 100.00% |
| Wine | 2 | Prophet | 22.78% | 23.09% ± 2.92% | 100.00% |
| Wine | 3 | LSTM | 23.20% | 23.41% ± 3.48% | 100.00% |
| Garden | 1 | Regression + ARIMA (1,0,1) | 7.80% | 8.94% ± 3.41% | 100.00% |
| Garden | 2 | Prophet | 7.81% | 8.94% ± 3.39% | 100.00% |
| Garden | 3 | TFT | 8.76% | 10.01% ± 3.69% | 100.00% |
| Luxury | 1 | CatBoost Default (MAE) | 100.25% | 100.55% ± 2.52% | 9.09% |
| Luxury | 2 | TimesFM 3.0 | 100.35% | 100.42% ± 0.75% | 0.00% |
| Luxury | 3 | Moving Average (14d) | 161.82% | 176.39% ± 77.90% | 0.00% |

#### 生产模拟：固定 104 周训练、预测 4 周

| 场景 | 排名 | 模型 | Pooled WAPE | 13-fold WAPE 均值 ± 标准差 | Event F1@0.5 |
| :--- | ---: | :--- | ---: | ---: | ---: |
| Grocery | 1 | Regression + ARIMA (1,0,1) | 4.26% | 4.28% ± 0.43% | 100.00% |
| Grocery | 2 | Prophet | 4.28% | 4.30% ± 0.41% | 100.00% |
| Grocery | 3 | LSTM | 4.62% | 4.61% ± 0.52% | 100.00% |
| Wine | 1 | Regression + ARIMA (1,0,1) | 22.46% | 22.67% ± 3.14% | 100.00% |
| Wine | 2 | Prophet | 22.49% | 22.71% ± 3.01% | 100.00% |
| Wine | 3 | LSTM | 23.74% | 24.01% ± 3.45% | 100.00% |
| Garden | 1 | Prophet | 7.84% | 9.01% ± 3.55% | 100.00% |
| Garden | 2 | Regression + ARIMA (1,0,1) | 7.85% | 9.01% ± 3.49% | 100.00% |
| Garden | 3 | TimesFM 3.0 | 8.86% | 9.84% ± 3.23% | 100.00% |
| Luxury | 1 | TimesFM 3.0 | 100.87% | 101.22% ± 1.59% | 0.00% |
| Luxury | 2 | CatBoost Default (MAE) | 111.81% | 105.35% ± 9.58% | 16.00% |
| Luxury | 3 | Prophet | 142.28% | 140.49% ± 35.27% | 0.00% |

### AutoGluon 独立 AutoML 对比

AutoGluon 的 15 个组件和 ensemble 只在 AutoML 层内部排名，不占据普通广度榜名次。

#### 主榜：扩展历史

| 场景 | 排名 | 模型 | Pooled WAPE | 13-fold WAPE 均值 ± 标准差 | Event F1@0.5 |
| :--- | ---: | :--- | ---: | ---: | ---: |
| Grocery | 1 | AG Weighted Ensemble | 5.52% | 5.56% ± 0.96% | 100.00% |
| Grocery | 2 | AG DirectTabular | 5.53% | 5.55% ± 0.99% | 100.00% |
| Grocery | 3 | AG DeepAR | 5.75% | 5.80% ± 1.17% | 100.00% |
| Wine | 1 | AG DeepAR | 25.03% | 25.74% ± 5.64% | 100.00% |
| Wine | 2 | AG DirectTabular | 25.24% | 25.61% ± 4.57% | 100.00% |
| Wine | 3 | AG Weighted Ensemble | 26.20% | 27.11% ± 7.19% | 100.00% |
| Garden | 1 | AG Weighted Ensemble | 10.19% | 11.61% ± 5.06% | 100.00% |
| Garden | 2 | AG DirectTabular | 10.23% | 11.49% ± 4.23% | 100.00% |
| Garden | 3 | AG DeepAR | 10.24% | 11.37% ± 3.90% | 100.00% |
| Luxury | 1 | AG NPTS | 100.00% | 100.00% ± 0.00% | 0.00% |
| Luxury | 2 | AG Weighted Ensemble | 100.00% | 100.00% ± 0.00% | 0.00% |
| Luxury | 3 | AG Zero | 100.00% | 100.00% ± 0.00% | 0.00% |

#### 生产模拟：固定 104 周训练、预测 4 周

| 场景 | 排名 | 模型 | Pooled WAPE | 13-fold WAPE 均值 ± 标准差 | Event F1@0.5 |
| :--- | ---: | :--- | ---: | ---: | ---: |
| Grocery | 1 | AG DirectTabular | 5.73% | 5.74% ± 0.88% | 100.00% |
| Grocery | 2 | AG Weighted Ensemble | 6.06% | 6.08% ± 0.84% | 100.00% |
| Grocery | 3 | AG RecursiveTabular | 6.20% | 6.20% ± 0.87% | 100.00% |
| Wine | 1 | AG Weighted Ensemble | 26.72% | 27.67% ± 7.78% | 100.00% |
| Wine | 2 | AG DeepAR | 26.97% | 27.78% ± 7.48% | 100.00% |
| Wine | 3 | AG DirectTabular | 26.98% | 27.62% ± 5.12% | 100.00% |
| Garden | 1 | AG DirectTabular | 10.31% | 11.61% ± 4.63% | 100.00% |
| Garden | 2 | AG ETS | 10.43% | 11.96% ± 5.31% | 100.00% |
| Garden | 3 | AG DeepAR | 10.43% | 11.30% ± 3.36% | 100.00% |
| Luxury | 1 | AG NPTS | 100.00% | 100.00% ± 0.00% | 0.00% |
| Luxury | 2 | AG Zero | 100.00% | 100.00% ± 0.00% | 0.00% |
| Luxury | 3 | AG DeepAR | 100.02% | 100.01% ± 0.03% | 0.00% |

### TimesFM 3.0 配置结果

`univariate` 是广度代表；`known_covariates` 是已知未来业务变量消融。
Coverage80 是经验区间覆盖率，不是库存服务水平保证。

#### 主榜：扩展历史

| 场景 | 配置 | Pooled WAPE | 13-fold WAPE 均值 ± 标准差 | Coverage80 | Event F1@0.5 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Grocery | known_covariates | 4.54% | 4.55% ± 0.50% | 81.45% | 100.00% |
| Grocery | univariate | 4.91% | 4.91% ± 0.69% | 83.19% | 100.00% |
| Wine | known_covariates | 22.69% | 22.67% ± 2.90% | 77.07% | 100.00% |
| Wine | univariate | 24.29% | 24.78% ± 5.24% | 75.69% | 100.00% |
| Garden | known_covariates | 8.17% | 9.32% ± 3.32% | 79.56% | 100.00% |
| Garden | univariate | 8.86% | 9.79% ± 3.07% | 74.03% | 100.00% |
| Luxury | known_covariates | 100.41% | 100.32% ± 0.32% | 91.44% | 0.00% |
| Luxury | univariate | 100.35% | 100.42% ± 0.75% | 93.65% | 0.00% |

#### 生产模拟：固定 104 周训练、预测 4 周

| 场景 | 配置 | Pooled WAPE | 13-fold WAPE 均值 ± 标准差 | Coverage80 | Event F1@0.5 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Grocery | known_covariates | 4.93% | 4.93% ± 0.59% | 83.48% | 100.00% |
| Grocery | univariate | 5.19% | 5.19% ± 0.83% | 85.22% | 100.00% |
| Wine | known_covariates | 23.98% | 24.31% ± 4.93% | 77.90% | 100.00% |
| Wine | univariate | 25.19% | 25.80% ± 5.76% | 76.24% | 100.00% |
| Garden | known_covariates | 8.55% | 9.68% ± 3.31% | 78.45% | 100.00% |
| Garden | univariate | 8.86% | 9.84% ± 3.23% | 78.73% | 100.00% |
| Luxury | known_covariates | 100.47% | 100.56% ± 0.59% | 88.67% | 0.00% |
| Luxury | univariate | 100.87% | 101.22% ± 1.59% | 87.57% | 0.00% |

完整能力矩阵、配置消融和历史策略比较见
[V4 发布摘要](docs/reports/2026-retail-week-v4-release-summary.md)；完整证据见
[V4 完整报告](docs/reports/2026-retail-week-v4-first-stage-report.md)。
<!-- BENCHMARK_BLOCK:current_results:END -->

历史 V2/V3 结果不再显示在当前 README；其分析、报告和数据分别保存在
[文章归档](docs/articles/archive/)、[报告归档](docs/reports/archive/)和
`data/benchmarks/archive_single_cutoff_28d/` / `data/benchmarks/published/2026-retail-week-24w-v3/`。

---

## 🧩 如何独立重跑与接入新模型 (How to Run Benchmarks & Onboard Models)

### 1. 独立运行与按需实验 CLI

使用统一的基准调度器 `scripts/run_benchmark.py`，支持缓存幂等与独立重算：

```bash
# 查看所有已注册的模型与当前已持久化的模型库
uv run python scripts/run_benchmark.py --list

# 独立运行指定模型（支持单个或逗号分隔列表）
uv run python scripts/run_benchmark.py --benchmark 2026-retail-week-expanding-v4 --models xgboost_tweedie_p18

# 按模型家族批量运行（如仅重跑 GBDT 系列）
uv run python scripts/run_benchmark.py --benchmark 2026-retail-week-expanding-v4 --family tree_gbdt

# 仅聚合一个明确的结果根目录；禁止隐式扫描与跨协议混合
uv run python scripts/run_benchmark.py --benchmark 2026-retail-week-expanding-v4 --compare-only --source-root runs/20261105-all-models-benchmark

# 两套协议都完成后，生成训练历史策略比较表
uv run python scripts/compare_training_histories.py \
  --source-root runs/20261105-all-models-benchmark \
  --main-benchmark 2026-retail-week-expanding-v4 \
  --fixed-window-benchmark 2026-retail-week-104w-rolling-v4

# 审核两套模型 release 后，发布可复算的训练历史比较
uv run python scripts/publish_training_history_comparison.py \
  --main-release data/benchmarks/published/2026-retail-week-expanding-v4/releases/20261106-timesfm-v1 \
  --fixed-window-release data/benchmarks/published/2026-retail-week-104w-rolling-v4/releases/20261106-timesfm-v1 \
  --release 20261106-timesfm-v1
```

### 2. 零重训图表与报告生成工具

无需重新拟合模型，直接从标准持久化存储中聚合指标并生成出版级配图与技术报告：

```bash
# 渲染 5 组出版级基准全景图表 (生成至 images/benchmarks/)
uv run python scripts/plot_benchmark_summary.py \
  --benchmark 2026-retail-week-expanding-v4 \
  --release 20261106-timesfm-v1

# 从已发布快照生成受控 Markdown 数据块，不覆盖人工叙事
uv run python scripts/generate_benchmark_report.py \
  --benchmark 2026-retail-week-expanding-v4 \
  --release 20261106-timesfm-v1

# 从两套审核 release 刷新 README 的普通模型、AutoML 与 TimesFM 结果
uv run python scripts/generate_readme_benchmark_results.py \
  --main-release data/benchmarks/published/2026-retail-week-expanding-v4/releases/20261106-timesfm-v1 \
  --production-release data/benchmarks/published/2026-retail-week-104w-rolling-v4/releases/20261106-timesfm-v1

# 重新渲染 4 个场景的时序全貌图与 ADI-CV 分布图 (生成至 images/)
uv run python scripts/plot_teaching_scenarios.py
```

### 3. 接入新模型 (XXX) 的标准化流程

未来若要接入新的预测模型（如 **Chronos、MOIRAI、TiDE、PatchTST、DLinear** 等）：
1. **查阅开发规程**：阅读 [`docs/reports/model-onboarding-protocol.md`](docs/reports/model-onboarding-protocol.md)；
2. **实现接口契约**：在 `src/retail_forecast/` 中继承 `BaseRetailForecaster`；
3. **注册运行器**：在 `scripts/run_benchmark.py` 的 `MODEL_REGISTRY` 注册配置与参数；
4. **一键执行并网**：运行 `uv run python scripts/run_benchmark.py --benchmark 2026-retail-week-expanding-v4 --models <model_key>`。

> 🤖 **面向 AI Coding Agents**：参与本项目的 AI 助手请严格遵循根目录顶级指引文档 **[`AGENTS.md`](AGENTS.md)** 中的架构红线与开发规程。

---

## 🗂️ 仓库目录结构 (Repository Structure)

```text
retail-demand-forecast/
├── data/                       # 📊 数据与模型资产 (严格维持 8 个顶级目录之一)
│   ├── teaching/               # 5 年合成场景 CSV (260 周 / 1,820 天) 与哈希清单
│   ├── artifacts/              # 经审核的参考快照 (published/) 与中间产物
│   ├── benchmarks/             # 🏆 核心基准结果持久化库
│   │   ├── protocols/          # 可哈希的 V4 评估协议声明
│   │   ├── published/          # 经审核的不可变轻量发布快照
│   │   └── archive_single_cutoff_28d/ # V2 单一切片历史基准归档
│   └── models/                 # 教学参考模型权重快照 (data/models/reference/)
├── docs/                       # 📚 统一文档中心 (系列博客、技术报告与学术文献)
│   ├── articles/               # 当前系列长文；旧协议文章进入 archive/
│   ├── reports/                # 当前 V4 报告；旧协议报告进入 archive/
│   └── references/             # 学术文献与经典论文 PDF 存档
├── images/                     # 🖼️ 教学数据概览与出版级实验对比图表 (.png / .webp)
│   ├── benchmarks/             # 按 benchmark ID / release ID 分层的结果图表
│   └── tweedie/                # 历史 Tweedie 报告配图
├── notebooks/                  # 交互式教学与基准评测 Notebook (00 ~ 36)
├── runs/                       # 本地实验独立输出目录 (命名规范: runs/YYYYMMDD-[model]-[type]/)
├── scripts/                    # 自动化与运维脚本 (基准评测、数据绘图、报告生成与发布)
├── src/
│   └── retail_forecast/        # 核心 Python 库与生产 SQL 资产
│       ├── common.py           # 数据加载、SHA256 校验与统一评价指标
│       ├── baselines.py        # Seasonal Naive 与 Moving Average 基准实现
│       ├── benchmark_protocol.py # V4 声明式协议与 fold 生成
│       ├── benchmark_registry.py # 严格结果契约、指标与汇聚引擎
│       ├── benchmark_adapters.py # 模型适配器与正式评估前参数冻结
│       ├── trees.py            # Decision Tree, Random Forest, LightGBM, XGBoost, CatBoost
│       ├── prophet_benchmark.py# Prophet 建模与评测模块
│       ├── arima.py            # SARIMAX / 动态回归实现
│       ├── neural.py           # MLP 与 LSTM 时序神经网络实现
│       ├── tft.py              # Temporal Fusion Transformer (TFT) 实现
│       ├── autogluon_v4.py     # V4 AutoML 独立对比实验
│       ├── timesfm_v4.py       # V4 TimesFM 3.0 适配器
│       ├── archive/            # 仅供历史教程追溯的 V2/V3 实现
│       └── sql/                # SQL 资产与归档的 BigQuery TimesFM 实验
└── tests/                      # 单元测试 (仓库布局规范、数据哈希一致性、模型基准测试)
```

---

## 🧭 教程 Notebook 路线图 (Tutorial Roadmap)

请先运行 `00` 生成本地合成数据集，随后可按模型专题或全模型对比顺序阅读与执行：

| 阶段 | Notebook 文件 | 主题与涵盖模型 |
| :--- | :--- | :--- |
| **00 数据生成** | [00-generate-teaching-data.ipynb](notebooks/00-generate-teaching-data.ipynb) | 生成 5 年四套墨尔本零售教学合成数据、VIC 公共假日表与数据概览图 |
| **10–11 Prophet 专项** | [10-prophet-retail-tutorial.ipynb](notebooks/10-prophet-retail-tutorial.ipynb)<br>[11-prophet-luxury-tutorial.ipynb](notebooks/11-prophet-luxury-tutorial.ipynb) | Prophet 可加分解、假日与促销回归项、缺货去偏、参数消融及在稀疏奢侈品场景下的局限性诊断 |
| **20 五模型初探** | [20-five-model-comparison.ipynb](notebooks/20-five-model-comparison.ipynb) | Seasonal Naive、Prophet、SARIMAX、LightGBM 与 TFT 五类代表性模型的初步横向对照 |
| **31 经典统计模型** | [31-arima-tutorial.ipynb](notebooks/31-arima-tutorial.ipynb) | ARIMA / SARIMAX 动态回归、傅里叶季节项与信息准则选参 |
| **32 树模型与 GBDT** | [32a-decision-tree-tutorial.ipynb](notebooks/32a-decision-tree-tutorial.ipynb)<br>[32b-random-forest-tutorial.ipynb](notebooks/32b-random-forest-tutorial.ipynb)<br>[32c-lightgbm-tutorial.ipynb](notebooks/32c-lightgbm-tutorial.ipynb)<br>[32d-xgboost-tutorial.ipynb](notebooks/32d-xgboost-tutorial.ipynb)<br>[32e-catboost-tutorial.ipynb](notebooks/32e-catboost-tutorial.ipynb)<br>[32f-tree-model-comparison.ipynb](notebooks/32f-tree-model-comparison.ipynb) | 时序转表格化特征工程（Lags、Rolling Stats、日历与促销特征），从单棵决策树到主流 GBDT 家族的系统对比 |
| **33 深度学习模型** | [33a-mlp-tutorial.ipynb](notebooks/33a-mlp-tutorial.ipynb)<br>[33b-lstm-tutorial.ipynb](notebooks/33b-lstm-tutorial.ipynb)<br>[33c-tft-tutorial.ipynb](notebooks/33c-tft-tutorial.ipynb)<br>[33d-neural-model-comparison.ipynb](notebooks/33d-neural-model-comparison.ipynb) | MLP、序列编码 LSTM 与 Temporal Fusion Transformer (TFT) 的训练、早停与样本复杂度边界 |
| **34–35 全模型与 AutoML** | [34-all-model-comparison.ipynb](notebooks/34-all-model-comparison.ipynb)<br>[35-autogluon-timeseries-tutorial.ipynb](notebooks/35-autogluon-timeseries-tutorial.ipynb) | 10 种核心模型在统一滚动验证与测试口径下的全量对比，以及 AutoGluon-TimeSeries 多层集成实验 |
| **36 时序基础模型（历史 V2）** | [36-timesfm-bigquery-tutorial.ipynb](notebooks/36-timesfm-bigquery-tutorial.ipynb) | 历史 BigQuery 单切片教程；当前 TimesFM 3.0 双协议结论以 [V4 文章](docs/articles/timesfm-retail-demand-forecast.md)为准 |

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备

本项目基于 **Python 3.12** 并使用 [uv](https://docs.astral.sh/uv/) 进行依赖与虚拟环境管理。在仓库根目录下执行：

```bash
uv sync --group dev
```

### 2. 生成 5 年教学数据并启动交互式环境

```bash
# 生成 5 年完整教学数据与配图
uv run python scripts/generate_teaching_data.py
uv run python scripts/plot_teaching_scenarios.py

# 启动交互式 Notebook 环境
uv run jupyter lab
```

### 3. 执行测试与基准校验

```bash
# 运行仓库布局规范、数据哈希契约与模型协议单元测试
uv run python -m unittest discover -s tests -v
```
