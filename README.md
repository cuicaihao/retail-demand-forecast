# Retail Demand Forecasting (零售需求预测教学与基准评测)

基于澳大利亚维多利亚州（VIC / 墨尔本）合成零售数据的可复现需求预测教程与多模型基准评测仓库。所有数据均为固定随机种子生成的教学合成数据，用于受控检验不同预测模型在典型零售业务场景下的优势与局限，不代表任何真实企业经营数据。

## 📘 系列技术博客 (Featured Blog Posts)

本仓库配套的技术博客深度拆解了零售需求预测的建模机制、调参诊断、缺货去偏与多模型横向评测：

- **[零售需求预测实战：基于 Prophet 的建模、调参、诊断与局限性分析](https://cuicaihao.github.io/zh/posts/2026-10-01-prophet-retail-demand-forecast/)**  
  结合本仓库生成的四套合成零售数据，系统剖析 Prophet 的可加/可乘分解、VIC 州公共假日与促销回归项、缺货删失（Stockout Censoring）去偏、参数敏感性诊断，以及在脉冲促销与稀疏间歇需求下的结构性边界。

---

## 🛒 四大合成零售场景 (Synthetic Retail Scenarios)

由 [notebooks/00-generate-teaching-data.ipynb](notebooks/00-generate-teaching-data.ipynb) 生成的四套日度数据集（位于 `data/teaching/`）覆盖 `2023-01-01` 至 `2025-12-31`（共 1,096 天；默认前 1,068 天为历史训练期，最后 28 天 `2025-12-04` 至 `2025-12-31` 为测试集）：

| 场景 | 数据文件 (`data/teaching/`) | 核心需求形态 | 已知业务协变量 (Known Future Covariates) | 核心评测挑战 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 商超日用杂货 (Grocery)** | `grocery_scenario_data.csv` | 高频连续需求（均值 ~109.5）+ 周/年周期 | `is_open`, `is_closed_holiday`, `discount`（每 14 天前 7 天 8 折）, VIC 假日 | 含 **24 个随机缺货日**（`stockout=True` 时销量被硬截断为 20 件），考验缺货去偏能力 |
| **2. 红酒促销 (Wine)** | `wine_scenario_data.csv` | 泊松计数需求（均值 ~10.4，$\widehat{\mathrm{ADI}}=1.013$） | `promo_case_buy`（每 28 天前 3 天成箱促销）, `is_party_event`, `is_december_peak` | 强烈短时促销脉冲与赛事/年末峰值，检验平滑先验与非线性特征交互 |
| **3. 庭院耗材 (Garden)** | `garden_scenario_data.csv` | 强年度正弦周期 + 周末效应（均值 ~64.9） | `is_long_weekend_diy`（长周末 DIY 旺季）, `is_end_of_autumn_clearance` | 显著的冬夏跨季振幅与长周末叠加效应，检验季节模式（加法 vs. 乘法）与滞后锚定 |
| **4. 奢侈品手袋 (Luxury)** | `luxury_scenario_data.csv` | **极端稀疏间歇性需求**（营业日零值率 **93.58%**，$\widehat{\mathrm{ADI}}=15.571$） | `is_saturday`, `is_sunday`, `is_december_peak`, `is_vip_event`, `discount` | 零膨胀序列（Zero-Inflated），检验点预测保守全零陷阱、概率发生率与分位数校准 |

---

## 🗂️ 仓库目录结构 (Repository Structure)

```text
retail-demand-forecast/
├── notebooks/                  # 17 个循序渐进的交互式教学与基准评测 Notebook (00 ~ 36)
├── src/
│   └── retail_forecast/        # 核心 Python 包：统一数据契约、特征工程与各模型家族实现
│       ├── common.py           # 数据加载、SHA256 校验、滚动验证窗口切分与统一评价指标
│       ├── prophet_benchmark.py# Prophet 建模与调参评测模块
│       ├── arima.py            # SARIMAX / 动态回归实现
│       ├── trees.py            # 决策树、随机森林、LightGBM、XGBoost、CatBoost 统一管线
│       ├── neural.py           # MLP 与 LSTM 时序神经网络实现
│       ├── tft.py              # Temporal Fusion Transformer (TFT) 训练与推理封装
│       ├── autogluon_benchmark.py # AutoGluon-TimeSeries 自动集成评测模块
│       ├── timesfm_benchmark.py# TimesFM 时序大模型评测与基准对比模块
│       └── model_comparison.py # 跨模型家族统一汇总、切片分析与图表生成
├── data/
│   ├── README.md               # 数据生成与防覆盖策略说明
│   └── teaching/               # [本地生成] 四套合成场景 CSV、BigQuery Schema JSON 与哈希清单
├── sql/                        # 生产级 BigQuery SQL（含 TimesFM 原生 AI.FORECAST 全套受控实验）
├── scripts/                    # 自动化脚本（基准评测、数据绘图、运行校验、Notebook 生成与发布打包）
├── tests/                      # 单元测试（仓库布局规范、数据哈希一致性、模型基准测试）
├── references/                 # 参考文献与论文存档（如 Prophet 2017 原始论文）
├── artifacts/                  # [本地生成] 经审核的参考快照 (published/) 与中间产物，详见 artifacts/README.md
├── models/                     # [本地生成] 教学参考模型权重快照 (models/reference/)
├── runs/                       # [本地生成] 每次实验运行的独立输出目录 (runs/<run-id>/)
├── images/                     # [本地生成] 教学数据总览与实验对比图表 (.png / .webp)
├── articles/                   # [本地/develop] 系列技术博客文稿与配图 (figures/)
└── reports/                    # [本地/develop] 受控实验方案协议、量化实证报告与系统架构设计文档
```

---

## 🧭 教程 Notebook 路线图 (Tutorial Roadmap)

请先运行 `00` 生成本地合成数据集，随后可按模型专题或全模型对比顺序阅读与执行：

| 阶段 | Notebook 文件 | 主题与涵盖模型 |
| :--- | :--- | :--- |
| **00 数据生成** | [00-generate-teaching-data.ipynb](notebooks/00-generate-teaching-data.ipynb) | 生成四套墨尔本超市教学合成数据、VIC 公共假日表与数据概览图 |
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

### 2. 生成教学数据并启动 JupyterLab

```bash
# 启动交互式 Notebook 环境，首先运行 notebooks/00-generate-teaching-data.ipynb
uv run jupyter lab
```

### 3. 运行测试与校验

```bash
# 运行仓库布局、数据哈希契约与模型基准单元测试
uv run python -m unittest discover -s tests -v

# 校验指定实验运行目录的输入数据哈希清单
uv run python scripts/validate_artifacts.py runs/<run-id>
```

---

## 🌿 分支规范与本地产物策略 (Branch & Artifact Policy)

- **`main` 分支（远端同步分支）**：保持精简、干净且 100% 可复现，仅追踪核心源码 (`src/`)、无输出状态的教学 Notebook (`notebooks/`)、工具脚本 (`scripts/`)、单元测试 (`tests/`) 与项目文档。生成的数据 (`data/teaching/`)、运行输出 (`runs/`)、模型权重 (`models/`)、配图 (`images/`) 与未定稿报告默认通过 [.gitignore](.gitignore) 忽略，详情参见 [data/README.md](data/README.md) 与 [artifacts/README.md](artifacts/README.md)。
- **`develop` 分支（本地实验与报告分支）**：用于在本地环境（如 Mac mini）开展前沿模型受控实验（例如 Google TimesFM 零样本与协变量基准测试）、保存实验计划 (`articles/`) 与深度评测报告 (`reports/`)。
