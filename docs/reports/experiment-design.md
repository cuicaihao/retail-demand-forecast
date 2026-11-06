# Retail Demand Forecast V4 实验设计

> 文档状态：当前有效实验协议  
> 协议版本：`retail-benchmark-v4`  
> 主榜：`2026-retail-week-expanding-v4`  
> 生产模拟：`2026-retail-week-104w-rolling-v4`

本文定义仓库当前已经实现并可复算的实验边界。它不是理想化的未来路线图，也不把尚未
实现的分析写成既成事实。模型接入细节见
[模型接入协议](model-onboarding-protocol.md)，当前结果见
[V4 分阶段基准报告](2026-retail-week-v4-first-stage-report.md)。

## 1. 实验目标与证据边界

本实验使用确定性合成日销量，在相同数据、预测日期和已知未来信息下回答三个当前可验证
的问题：

1. 不同模型工具包在四类需求机制中的表现有何差异？
2. 同一 GBDT 工具包采用默认目标或不同 Tweedie power 时，结果如何变化？
3. 每个预测起点使用全部历史或最近 104 个完整周时，预测表现如何变化？

当前发布不声称已经完成下列研究：

- Recursive、Direct 与 Multi-output 的受控比较；
- 第 1 至第 4 个预测周的正式 Horizon Profile；
- Grocery 缺货、Wine 促销、Garden 季节和 Luxury 事件的完整业务分层消融；
- Oracle 特征实验、多数据随机种子复现或正式显著性检验；
- 跨四种需求尺度的单一总冠军。

这些内容可以在未来协议中扩展，但不得反向改变已经发布的 V4 release。

## 2. 合成数据协议

### 2.1 数据范围与粒度

- 日期：2022-07-04 至 2027-06-27；
- 粒度：每个场景一条日序列，共 1,820 天，即 260 个完整周；
- 场景：`grocery`、`wine`、`garden`、`luxury`；
- 主目标：CSV 中的 `y`，表示合成的日观测销量；
- 主键：当前单序列场景中的 `scenario × ds`；
- 生成规范：`data/teaching/generation_spec.json`；
- 生成元数据与文件哈希：`data/teaching/generation_metadata.json`。

数据由 `scripts/generate_teaching_data.py` 确定性生成。Notebook 是教学展示入口；正式数据
是否一致以已提交 CSV、生成规范、元数据和仓库测试为准。

### 2.2 四种需求机制

| 场景 | 主要机制 | 当前允许的未来业务特征 | 主要失败风险 |
| :--- | :--- | :--- | :--- |
| Grocery | 高频连续需求、周/年周期、促销、闭店和缺货删失 | `discount`、日历、假日、营业安排 | 把缺货销量下降误解为需求下降 |
| Wine | 泊松计数、周期促销、派对事件和 12 月高峰 | `promo_case_buy`、`is_party_event`、`is_december_peak` | 系统性低估促销脉冲 |
| Garden | 强年度周期、周末效应、长周末和秋末清仓 | `is_long_weekend_diy`、`is_end_of_autumn_clearance` | 年度振幅或季节转折估计错误 |
| Luxury | 极端零膨胀、VIP 事件、周末和折扣效应 | `is_saturday`、`is_sunday`、`is_december_peak`、`is_vip_event`、`discount` | 以接近全零预测取得表面低误差 |

日历、假日、营业安排和上述业务变量在当前合成协议中被声明为预测起点已知。当前数据没有
模拟计划发布时间、临时促销变更或未来营业安排修订，因此结论只适用于这一信息可用性假设。

### 2.3 诊断真值与防泄漏

`conditional_mean`、`latent_demand`、`occurrence_probability` 和
`positive_demand` 是生成器诊断真值。它们可以用于生成质量检查，但不得进入正式模型特征。
未来 `y` 和未来 `stockout` 同样不得作为推理特征。

正式特征由以下信息组成：

- 预测起点以前的有效历史目标；
- 仅从历史计算的 origin-frozen lag 和 rolling 统计量；
- 日期、傅里叶周期、假日和已声明的已知未来业务变量；
- 预测日相对当前 forecast origin 的位置。

测试通过扰动未来目标和诊断真值，验证这些变化不会改变模型输入或预测。

## 3. 正式滚动回测

### 3.1 评估年度

正式评估覆盖 2025-12-29 至 2026-12-27，共 364 天、52 个完整周。它是一个以周一开始、
周日结束的“2026 零售周年度”，不是完整自然年。

每个 fold 在一个预测起点完成训练，然后一次性产生未来 28 个日预测。fold 内第 2 至第 4
周不得读取该 fold 前面日期已经发生的真实销量。测试窗口连续且互不重叠：

| Fold | 测试开始 | 测试结束 | 天数 |
| :--- | :--- | :--- | ---: |
| P01 | 2025-12-29 | 2026-01-25 | 28 |
| P02 | 2026-01-26 | 2026-02-22 | 28 |
| P03 | 2026-02-23 | 2026-03-22 | 28 |
| P04 | 2026-03-23 | 2026-04-19 | 28 |
| P05 | 2026-04-20 | 2026-05-17 | 28 |
| P06 | 2026-05-18 | 2026-06-14 | 28 |
| P07 | 2026-06-15 | 2026-07-12 | 28 |
| P08 | 2026-07-13 | 2026-08-09 | 28 |
| P09 | 2026-08-10 | 2026-09-06 | 28 |
| P10 | 2026-09-07 | 2026-10-04 | 28 |
| P11 | 2026-10-05 | 2026-11-01 | 28 |
| P12 | 2026-11-02 | 2026-11-29 | 28 |
| P13 | 2026-11-30 | 2026-12-27 | 28 |

结果中的 `forecast_origin` 当前表示最后一个可用历史日，即测试开始前的周日。模型拟合历史
必须满足 `train_end < test_start`。

### 3.2 两种训练历史策略

| Benchmark ID | 训练历史 | 用途 |
| :--- | :--- | :--- |
| `2026-retail-week-expanding-v4` | 从 2022-07-04 到当前测试开始前的全部可用历史 | 正式工具包广度榜和配置消融主榜 |
| `2026-retail-week-104w-rolling-v4` | 当前测试开始前最近 104 个完整周，即 728 天 | 模拟固定训练成本的四周滚动生产流程 |

两套协议使用完全相同的场景、测试日期、预测长度、步长、有效标签规则、模型配置和选择
截止日期。不同 benchmark ID 的模型结果不得直接混入同一排行榜；训练历史影响通过独立的
comparison release 配对比较。

104 周是两个完整的 52 周零售周期，不等于两个完整日历年。对依赖至少 730 天历史的模型，
报告必须保留相应告警及其解释。

### 3.3 参数选择与正式评估分离

- 所有参数选择窗口结束于 2025-12-28 或更早；
- 正式 13 个 fold 的标签不得参与参数或配置选择；
- 每个模型和场景的完整解析参数写入 `selected_config.json` 和 manifest；
- 选择缓存键包含选择策略、截止日期、工具包和输入数据哈希；
- GBDT Tweedie power 是预先声明的配置矩阵，不根据正式测试成绩临时扩展。

当前协议属于“冻结结构配置后完整回测”。MLP、LSTM 与 TFT 仅允许在每个正式 fold 的
历史内部验证集上选择训练轮数；候选结构、学习率和容量参数仍在 2026 正式评估前冻结，
不得读取该 fold 的测试标签。

## 4. 当前模型与配置矩阵

当前 V4 release `20261106-autogluon-v1` 包含 24 个普通配置和 15 个独立 AutoML 配置。

### 4.1 工具包广度配置

每个工具包只有一个预先声明的代表配置进入广度榜：

- Seasonal Naive（最近 7 日模式重复）；
- Moving Average（最近 14 个营业日）；
- Decision Tree；
- Random Forest；
- LightGBM default；
- XGBoost default；
- CatBoost default；
- Prophet；
- Regression + ARIMA(1,0,1)；
- MLP；
- LSTM；
- TFT。

### 4.2 配置深度

LightGBM、XGBoost 和 CatBoost 分别包含：

- default；
- Tweedie `p=1.1`；
- Tweedie `p=1.2`；
- Tweedie `p=1.5`；
- Tweedie `p=1.8`。

default 是工具包代表配置；Tweedie 配置只进入独立消融表，不重复占据广度榜名次。

### 4.3 当前预测方式

所有模型遵守同一个 28 日 origin-frozen 批量预测边界，但其内部多步机制不同：

- 基线直接重复历史模式或输出固定历史均值；
- 树模型用一个 pooled direct 模型，根据未来日期、已知协变量和 `horizon_day` 输出 28 日预测；
- Prophet 和 ARIMA 使用各自的原生多步预测接口；
- MLP 与 LSTM 使用冻结的直接 28 日输出网络；TFT 使用原生多步预测接口；
- ARIMA 输出名义 80% 预测区间；其余当前模型只输出点预测。

V4 当前不把预测方式作为独立消融因素，因此不得根据这些结果声称某种多步策略普遍更优。

### 4.4 AutoGluon 独立 AutoML 对比层

AutoGluon 不进入普通工具包广度榜，也不与 GBDT 配置消融混排。它使用独立的
`comparison_role="automl"`：

- 每个场景、每个正式 fold 共享一次固定 240 秒的 AutoML fit；
- 候选库、计算预算、28 日预测长度和随机种子在正式评估前固定；
- AutoGluon 只能在该 fold 的训练历史内部进行验证、选择和集成，不得读取正式测试标签；
- 选择后使用 `refit_full=True` 在该 fold 的全部可用历史上重拟合；
- 内部组件和加权集成统一使用 `ag-` 前缀，例如 `ag-auto-arima`、
  `ag-direct-tabular` 和 `ag-weighted-ensemble`；
- 只有覆盖完整 4 场景 × 13 folds 的组件才能进入 AutoML 对比表；两种训练历史协议发布时
  只保留共同完整的 `ag-*` 集合，避免用不同模型集合比较历史策略。

因此，AutoML 表回答的是“固定计算预算下使用 AutoML 管线与不使用 AutoML 的差异”，
不是为 AutoGluon 的每个内部组件增加普通广度榜投票权。

## 5. 目标、有效标签与指标

### 5.1 有效标签

正式排名使用：

```text
is_open and not stockout and y is not null
```

因此主榜衡量的是营业且未发生缺货删失日期上的预测质量，而不是所有日历日 observed-sales
误差，也不是 latent-demand Oracle 误差。闭店和缺货日期仍保留在逐日预测文件中，但不进入
当前主榜分母。报告必须明确这一边界。

Grocery 历史中的闭店与缺货目标在模型训练前被遮蔽并插值，以避免直接学习人为截断值。
当前 V4 没有发布“原始删失销量训练 vs 去偏训练”消融。

### 5.2 共用指标

对全部有效逐日预测重新汇总：

- MAE；
- RMSE；
- WAPE；
- Bias；
- 基于 7 日季节朴素尺度的 MASE；
- WAPE 分母和有效样本数。

WAPE 和 Bias 的销量分母为零时返回缺失值，不以零误差替代。年度 WAPE 使用全部有效日的
绝对误差和销量重新计算，不使用 13 个 fold WAPE 的简单平均。

### 5.3 间歇需求与概率指标

所有模型均计算 `ZeroRecall@0.5`、`DemandRecall@0.5`、事件 F1 和正需求 MAE；这些指标
主要用于解释 Luxury。只按 WAPE 排名不足以证明模型能识别稀疏需求事件。

提供 q10/q90 时，额外报告：

- Coverage80；
- MeanIntervalWidth80；
- q10/q90 平均 pinball loss。

没有分位数输出的模型将这些字段标记为不适用。

### 5.4 时间稳定性与比较边界

- 保存每个模型、场景和 fold 的完整指标序列；
- 报告 fold WAPE 的均值、标准差、中位数和 IQR；
- 模型间胜负只在相同场景和相同 fold 上配对；
- bootstrap 区间是描述性摘要，不构成统计显著性声明；
- 不跨四种不同需求尺度计算单一总冠军。

当前 `latency_ms` 包含一次 fold 的拟合和预测总时间，应理解为 fold execution latency，
不是独立的在线推理延迟。manifest 的 `elapsed_seconds` 是一个模型完成全部场景和 fold 的
墙钟时间。峰值内存、模型大小与纯推理耗时尚未进入当前正式比较。

## 6. 结果与发布分层

### 6.1 本地运行层

本地结果写入：

```text
runs/YYYYMMDD-[model]-[benchmark|validation|experiment]/<benchmark-id>/
```

每个模型目录包含：

- `predictions.csv`：逐日真实值、预测、区间、fold 和 forecast origin；
- `metrics_period.csv`：逐场景、逐 fold 指标；
- `metrics_annual.csv`：逐场景 pooled 指标；
- `selected_config.json`：冻结参数选择记录；
- `manifest.json`：协议、参数、数据、环境和源码 provenance。

部分场景运行只允许用于本地诊断，不能发布为完整 benchmark release。

### 6.2 不可变发布层

审核后的轻量结果写入：

```text
data/benchmarks/published/<benchmark-id>/releases/<release-id>/
```

release manifest 固定模型清单、协议哈希、每个文件 SHA-256 和逐模型源码 provenance。
release ID 不可覆盖；增加新模型时创建新 release，并可从已审核 release 继承旧模型结果。

训练历史比较写入：

```text
data/benchmarks/published/comparisons/
  <expanding-id>--vs--<fixed-window-id>/releases/<release-id>/
```

比较 release 固定两套源 release manifest 的哈希，并重新生成逐模型差值和场景摘要。

### 6.3 报告层

- 模型层：模型预测、fold 指标、pooled 指标、配置和 manifest；
- 实验层：场景榜、工具包广度、配置消融和训练历史比较；
- 发布层：不可变 release、图表、受控 Markdown 数据块和 release manifest；
- 人工正文：维护解释、限制与业务含义；生成器只替换 `BENCHMARK_BLOCK` 标记区域。

## 7. 可复现性与失败关闭

缓存和聚合至少同时检查：

- benchmark ID、协议版本和协议哈希；
- 场景集合与全部 fold；
- 输入 CSV 哈希；
- 完整解析参数和冻结配置哈希；
- 软件版本、随机种子、Git commit 和源码树哈希；
- forecast origin、逐日日期完整性、重复预测和有效标签日期集合。

协议、数据、代码、软件版本、参数、场景或有效标签集合任一变化都会使当前运行缓存失效。
历史发布验证允许在新代码环境中复算文件和旧协议完整性，但不会把旧 release 冒充为当前
运行缓存。

## 8. 新模型进入 V4 的门槛

MLP、LSTM、TFT、AutoGluon 和 TimesFM 只有满足以下条件后才能进入当前 V4 榜单：

1. 通过统一 `BaseRetailForecaster` 适配器注册，不在运行器增加模型专属分支；
2. 明确 `toolkit`、`family`、`variant`、`comparison_role` 和 `config_group`；
3. 使用同一组冻结参数、13 个 fold、有效标签和两种历史协议；
4. 预测过程不能读取未来目标、未来诊断真值或未来实际 stockout；
5. 完整记录依赖版本、随机种子、解析参数、训练耗时和源码 provenance；
6. 模型权重和临时 checkpoint 留在 `runs/` 或忽略目录，不进入 Git；
7. 先完成单模型本地审核，再分别发布 expanding 和 104 周 release；
8. 只有两套 release 都通过复算后，才更新训练历史 comparison release 和人工报告。

新模型默认作为工具包代表配置进入广度榜。若同一工具包需要多个配置，必须提前声明一个
代表配置，其余配置进入单独消融表。不得根据正式 13-fold 成绩事后选择代表配置。

## 9. 当前限制与后续扩展

以下限制不会使现有 V4 比较失效，但约束结论范围：

1. 数据是单场景、单序列的教学合成数据，不代表真实多门店、多 SKU 企业分布。
2. 业务计划变量被假定为预测起点已知，没有模拟计划版本和发布时间延迟。
3. 主榜排除闭店与缺货删失标签，尚未提供 observed-sales 与 latent-demand 的并列诊断。
4. 当前没有正式 Horizon Profile、业务 segment 表、Oracle 消融和独立数据种子复现。
5. 当前成本记录不能分离训练、纯推理、内存和模型存储成本。
6. 104 周等于 728 天；对以 730 天作为两年阈值的模型属于边界条件。

这些扩展必须通过新增版本字段、结果表或新 benchmark ID 实现。已经发布的预测、指标和
manifest 不得原地改写。

## 10. 验收命令

```bash
# 协议、缓存、指标、发布和仓库结构测试
.venv/bin/python -m unittest discover -s tests -v

# 静态检查
.venv/bin/ruff check src scripts tests

# 补丁空白检查
git diff --check

# 发布快照复算
.venv/bin/python scripts/validate_benchmark_release.py \
  data/benchmarks/published/<benchmark-id>/releases/<release-id>
```

同时必须确认 Notebook 无保存输出、Markdown 相对链接有效、顶级目录仍为既定八个，并且
发布目录中不包含模型权重或临时训练产物。
