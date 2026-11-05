# Retail Demand Forecasting (零售需求预测教学与基准评测)

基于澳大利亚维多利亚州（VIC / 墨尔本）合成零售数据的可复现需求预测教程与多模型基准评测仓库。所有数据均为固定随机种子生成的教学合成数据，覆盖完整 5 年（2022-07-04 至 2027-06-27，共 260 完整周 / 1,820 天），用于受控检验不同预测模型在典型零售业务场景下的优势与局限，不代表任何真实企业经营数据。

## 📘 系列技术博客与报告 (Featured Blog Posts & Reports)

本仓库配套的技术博客与量化实证报告深度拆解了零售需求预测的建模机制、调参诊断、缺货去偏、Tweedie 损失函数与 52 周前向滑动窗口基准评测：

- **[零售需求预测 2026 年度基准：52 周滑动窗口评测与 Tweedie 分布实践](docs/articles/2026-retail-demand-forecast-benchmark.md)**  
  全面实施 2026 全年 52 周（13 期 $\times$ 28 天无重叠块步进）前向滑动窗口基准评测。涵盖 Baseline、统计时间序列与三代主流 GBDT，系统验证了 Tweedie 复合泊松-伽马损失函数在零膨胀与长尾零售需求中的理论与实证突破。
- **[2026 全年 52 周前向滑动窗口零售需求预测技术验证报告](docs/reports/2026-annual-retail-forecast-benchmark.md)**  
  详细记录 15 种模型及配置在四大场景 52 个测试切片下的综合榜单、跨四季动态轨迹、残差稳定性分析与生产系统分级路由落地建议。
- **[GBDT Tweedie 目标函数数学原理推导与实验报告](docs/reports/2026-11-03-gbdt-tweedie-validation-report.md)**  
  深入探讨指数离散族（Exponential Dispersion Family）、Compound Poisson-Gamma 分布的极大似然估计、一阶梯度与二阶 Hessian 推导及其在 LightGBM / XGBoost C++ 源码中的工程映射。
- **[零售需求预测实战：基于 Prophet 的建模、调参、诊断与局限性分析](https://cuicaihao.github.io/zh/posts/2026-10-01-prophet-retail-demand-forecast/)**  
  结合本仓库生成的四套合成零售数据，系统剖析 Prophet 的可加/可乘分解、VIC 州公共假日与促销回归项、缺货删失（Stockout Censoring）去偏、参数敏感性诊断，以及在脉冲促销与稀疏间歇需求下的结构性边界。

---

## 🛒 四大合成零售场景：本仓库的核心资产 (Synthetic Retail Scenarios)

本仓库最核心的学术与工程贡献，在于**抽象并严密实现了四个具有行业典型挑战的合成零售需求场景**（位于 `data/teaching/`，由 [scripts/generate_teaching_data.py](scripts/generate_teaching_data.py) 或 [notebooks/00-generate-teaching-data.ipynb](notebooks/00-generate-teaching-data.ipynb) 确定性生成）。每个场景代表现实供应链中不可回避的特定预测难题：

| 场景 | 数据文件 (`data/teaching/`) | 核心需求形态与业务机制 | 已知业务协变量 (Known Future Covariates) | 核心评测挑战与不可自愈陷阱 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 商超日用杂货 (Grocery)** | `grocery_scenario_data.csv` | 高频连续消费（均值 ~109.5）+ 强周/年周期 | `is_open`, `is_closed_holiday`, `discount`（每 14 天前 7 天 8 折）, VIC 公共假日 | 含 **24 个随机缺货截断日**（`stockout=True` 时销量被硬截断为 20 件），考验模型对**缺货删失（Censoring）去偏**能力 |
| **2. 红酒促销 (Wine)** | `wine_scenario_data.csv` | 泊松计数脉冲（均值 ~10.4，$\widehat{\mathrm{ADI}}=1.013$） | `promo_case_buy`（每 28 天前 3 天成箱促销）, `is_party_event`, `is_december_peak` | 强烈短时促销脉冲与年末峰值，检验平滑先验与非线性特征吸收能力 |
| **3. 庭院耗材 (Garden)** | `garden_scenario_data.csv` | 强年度正弦周期 + 周末效应（均值 ~64.9） | `is_long_weekend_diy`（长周末 DIY 旺季）, `is_end_of_autumn_clearance` | 极长的冬夏跨季振幅与长周末叠加效应，检验跨年长周期先验与季节项（加法 vs. 乘法） |
| **4. 奢侈品手袋 (Luxury)** | `luxury_scenario_data.csv` | **极端稀疏间歇性需求**（营业日零值率 **93.58%**，$\widehat{\mathrm{ADI}}=15.571$） | `is_saturday`, `is_sunday`, `is_december_peak`, `is_vip_event`, `discount` | 零膨胀序列（Zero-Inflated），检验点预测保守全零陷阱、概率发生率与分位数校准 |

![四场景需求形态与 ADI-CV 分布](images/retail-demand-profile-adi-cv.png)

---

## 🏆 2026 全年 52 周滑动窗口基准库 (2026 Annual Benchmark Store)

为彻底解决单一切片评估（Single 28-day Cutoff）因大促或淡季偶发性带来的测试偏差，本仓库全面升级至 **V3 周对齐前向滑动窗口基准评测体系（Walk-Forward Rolling Evaluation）**：

- **52 周完整覆盖**：将 2026 年（`2025-12-29` 至 `2026-12-27`，共 364 天）划分为 **13 个无重叠 28 天分块（P01 至 P13）**，历史训练固定采用预测起点前 **24 完整周（168 天）** 动态滚动；
- **标准五契约持久化**：每个模型在 `data/benchmarks/results/<model_key>/` 下固化保存 `predictions.csv`、`metrics_period.csv`、`metrics_annual.csv`、`manifest.json` 与独立报告 `model_report.md`；
- **多模式物理隔离**：支持 13 期无重叠分块步进（`--mode 4week`）与 50 期每周重叠步进（`--mode weekly`），存储路径完全隔离，互不覆盖；
- **零重训报告与出图**：毫秒级并入全景基准排行榜，支持随时重新合成图表与分析报告。

### 2026 全年总排行榜 (Global Annual Leaderboard - 52 Weeks / 13 Periods)

以下为 15 种模型及配置在 2026 年完整 52 周（13 期 $\times$ 4 场景 = 52 次独立滚动前向回测）的综合排行榜：

| 排名 | 模型与目标函数配置 | 模型家族 | 全场景平均 WAPE | 平均 MAE | 平均 RMSE | 平均 Bias | 零值准确率 | 跨期波动 Std |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **CatBoost Default (MAE)** | 梯度提升树 | **40.99%** | 6.924 | 8.824 | -0.245 | **97.9%** | **0.067** |
| 🥈 | **XGBoost Tweedie (p=1.8)** | 梯度提升树 | **41.92%** | **5.936** | **7.709** | -0.222 | **88.0%** | **0.097** |
| 3 | XGBoost Tweedie (p=1.5) | 梯度提升树 | 47.97% | 5.733 | 7.547 | -0.139 | 70.4% | 0.194 |
| 4 | LightGBM Tweedie (p=1.8) | 梯度提升树 | 53.35% | 6.820 | 8.916 | -0.115 | 56.3% | 0.230 |
| 5 | XGBoost Tweedie (p=1.2) | 梯度提升树 | 54.32% | 5.768 | 7.559 | -0.050 | 66.0% | 0.228 |
| 6 | XGBoost Tweedie (p=1.1) | 梯度提升树 | 56.58% | 5.794 | 7.586 | -0.030 | 51.6% | 0.235 |
| 7 | LightGBM Tweedie (p=1.1) | 梯度提升树 | 57.27% | 6.468 | 8.432 | -0.050 | 44.9% | 0.243 |
| 8 | Moving Average (14d) | 平滑基线 | 58.57% | 8.605 | 10.924 | -0.075 | 47.8% | 0.250 |
| 9 | Prophet | 广义可加模型 | 58.63% | 5.730 | 7.503 | +0.063 | 65.4% | 0.245 |
| 10 | XGBoost Default (MSE) | 梯度提升树 | 59.45% | 5.642 | 7.364 | +0.030 | 46.3% | 0.242 |
| 11 | LightGBM Default (MSE) | 梯度提升树 | 60.57% | 6.349 | 8.241 | -0.007 | 46.3% | 0.264 |
| 12 | Seasonal Naive (7d) | 周期基线 | 65.66% | 8.909 | 11.761 | -0.027 | 93.5% | 0.345 |
| 13 | Decision Tree | 基础树模型 | 67.21% | 7.588 | 9.961 | +0.085 | 68.3% | 0.367 |
| 14 | Random Forest | Bagging 集成树 | 67.42% | 6.874 | 8.700 | +0.084 | 25.8% | 0.337 |
| 15 | CatBoost Tweedie (p=1.8) | 梯度提升树 | 72.90% | 20.372 | 23.301 | -0.322 | 59.8% | 0.259 |

![2026 全年总榜单全景图](images/benchmarks/fig-01-annual-leaderboard-all-scenarios.png)

> 💡 **历史单一切片基准存档**：旧版基于 2025 年 12 月单一 28 天切片的基准测试结果（含 AutoGluon、TFT、TimesFM、ARIMA 等），已完整归档保存在 [`data/benchmarks/archive_single_cutoff_28d/`](data/benchmarks/archive_single_cutoff_28d/)，其历史代码状态已永久标记为 Git Tag `v0.2.0-single-cutoff-benchmark`。

---

## 🧩 如何独立重跑与接入新模型 (How to Run Benchmarks & Onboard Models)

### 1. 独立运行与按需实验 CLI

使用统一的基准调度器 `scripts/run_benchmark.py`，支持缓存幂等与独立重算：

```bash
# 查看所有已注册的模型与当前已持久化的模型库
uv run python scripts/run_benchmark.py --list

# 独立运行指定模型（支持单个或逗号分隔列表）
uv run python scripts/run_benchmark.py --models xgboost_tweedie_p18

# 按模型家族批量运行（如仅重跑 GBDT 系列）
uv run python scripts/run_benchmark.py --family tree_gbdt

# 启动 50 步周度重叠高频回测（结果自动隔离存入 results_weekly/，不覆盖主基准）
uv run python scripts/run_benchmark.py --models seasonal_naive --mode weekly

# 仅刷新全景排行榜（0 重训，毫秒级更新 summary/）
uv run python scripts/run_benchmark.py --compare-only
```

### 2. 零重训图表与报告生成工具

无需重新拟合模型，直接从标准持久化存储中聚合指标并生成出版级配图与技术报告：

```bash
# 渲染 4 组出版级基准全景图表 (生成至 images/benchmarks/)
uv run python scripts/plot_benchmark_summary.py

# 自动生成 2026 年度技术验证报告与 Hugo 博客文章 (Markdown)
uv run python scripts/generate_benchmark_report.py

# 重新渲染 4 个场景的时序全貌图与 ADI-CV 分布图 (生成至 images/)
uv run python scripts/plot_teaching_scenarios.py
```

### 3. 接入新模型 (XXX) 的标准化流程

未来若要接入新的预测模型（如 **Chronos、MOIRAI、TiDE、PatchTST、DLinear** 等）：
1. **查阅开发规程**：阅读 [`docs/reports/model-onboarding-protocol.md`](docs/reports/model-onboarding-protocol.md)；
2. **实现接口契约**：在 `src/retail_forecast/` 中继承 `BaseRetailForecaster`；
3. **注册运行器**：在 `scripts/run_benchmark.py` 的 `MODEL_REGISTRY` 注册配置与参数；
4. **一键执行并网**：运行 `uv run python scripts/run_benchmark.py --models <model_key>`。

> 🤖 **面向 AI Coding Agents**：参与本项目的 AI 助手请严格遵循根目录顶级指引文档 **[`AGENTS.md`](AGENTS.md)** 中的架构红线与开发规程。

---

## 🗂️ 仓库目录结构 (Repository Structure)

```text
retail-demand-forecast/
├── data/                       # 📊 数据与模型资产 (严格维持 8 个顶级目录之一)
│   ├── teaching/               # 5 年合成场景 CSV (260 周 / 1,820 天) 与哈希清单
│   ├── artifacts/              # 经审核的参考快照 (published/) 与中间产物
│   ├── benchmarks/             # 🏆 核心基准结果持久化库
│   │   ├── results/            # 13 期无重叠 4-week 年度基准标准五契约存储
│   │   ├── summary/            # 全局排行榜与周期演进大表 (annual_leaderboard.csv)
│   │   └── archive_single_cutoff_28d/ # v0.2.0 单一切片历史基准归档
│   └── models/                 # 教学参考模型权重快照 (data/models/reference/)
├── docs/                       # 📚 统一文档中心 (系列博客、技术报告与学术文献)
│   ├── articles/               # 对外发布的系列长文 (*-retail-demand-forecast.md)
│   ├── reports/                # 内部量化实证报告与系统设计 (YYYY-MM-DD-*.md)
│   └── references/             # 学术文献与经典论文 PDF 存档
├── images/                     # 🖼️ 教学数据概览与出版级实验对比图表 (.png / .webp)
│   ├── benchmarks/             # 2026 全年基准系列图表 (Fig 1~4)
│   └── tweedie/                # Tweedie 目标函数消融图表
├── notebooks/                  # 交互式教学与基准评测 Notebook (00 ~ 36)
├── runs/                       # 本地实验独立输出目录 (命名规范: runs/YYYYMMDD-[model]-[type]/)
├── scripts/                    # 自动化与运维脚本 (基准评测、数据绘图、报告生成与发布)
├── src/
│   └── retail_forecast/        # 核心 Python 库与生产 SQL 资产
│       ├── common.py           # 数据加载、SHA256 校验与统一评价指标
│       ├── baselines.py        # Seasonal Naive 与 Moving Average 基准实现
│       ├── benchmark_registry.py # V3 周对齐滚动评测切片器、存储契约与汇聚引擎
│       ├── trees.py            # Decision Tree, Random Forest, LightGBM, XGBoost, CatBoost
│       ├── prophet_benchmark.py# Prophet 建模与评测模块
│       ├── arima.py            # SARIMAX / 动态回归实现
│       ├── neural.py           # MLP 与 LSTM 时序神经网络实现
│       ├── tft.py              # Temporal Fusion Transformer (TFT) 实现
│       ├── autogluon_benchmark.py # AutoGluon-TimeSeries 自动集成评测
│       ├── timesfm_benchmark.py# Google TimesFM 时序大模型评测
│       └── sql/                # 生产级 BigQuery SQL (含 TimesFM AI.FORECAST 原生实验)
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
| **36 时序基础模型 (TSFM)** | [36-timesfm-bigquery-tutorial.ipynb](notebooks/36-timesfm-bigquery-tutorial.ipynb) | Google TimesFM 3.0 与 BigQuery 原生托管时序大模型零样本预测、缺货去偏与冷启动极限评测 |

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
