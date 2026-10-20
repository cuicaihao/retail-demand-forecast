# AutoGluon TimeSeries 与既有预测模型比较报告

> 实验日期：2026-10-20
>
> AutoGluon：1.5.0（与项目现有 Torch 2.8 兼容）
>
> 协议：`retail-benchmark-v2-autogluon-v1`
>
> 最终测试期：2025-12-04 至 2025-12-31

## 执行摘要

AutoGluon 的价值在本实验中得到了一半验证：它确实能在统一接口下训练统计、表格和轻量神经模型，并通过 Weighted Ensemble 改善单个组件；但它没有在商超、红酒或庭院用品上超过已有的最佳专用模型。

| 数据 | AutoGluon 验证 WAPE | AutoGluon 测试 WAPE | 测试排名 | 原报告最佳测试模型 | 最佳测试 WAPE |
|---|---:|---:|---:|---|---:|
| 商超 | 6.03% | 4.73% | 5 / 11 | ARIMA | 4.21% |
| 红酒 | 29.09% | 25.58% | 9 / 11 | LSTM | 21.26% |
| 庭院 | 9.74% | 11.54% | 6 / 11 | MLP | 9.76% |
| 奢侈品 | 100.00% | 100.00% | 并列 1 | CatBoost / Seasonal naive | 100.00% |

这里的 11 个正式比较对象是原来的 10 个核心模型加 AutoGluon Ensemble；Seasonal naive 只作基线，不计入排名。奢侈品的“并列第一”不能理解为识别了购买事件：AutoGluon 最终选择 Zero，预测总量为 0，而测试期实际销售 5 件。

总体结论如下：

- **AutoGluon 适合快速建立强而完整的候选基线。**它用约 9.6 分钟完成四场景、20 次拟合，输出排行榜、预测区间和 ensemble。
- **自动集成并不保证超过人工设计的领域模型。**原实验中的 Prophet、ARIMA、LSTM、MLP 和 XGBoost 已使用相同业务特征和针对性验证，优势没有被通用 AutoML 自动消除。
- **对连续需求，ensemble 是有意义的。**商超和红酒最终测试中，Weighted Ensemble 都优于所有 AutoGluon 单组件。
- **对间歇需求，自动模型选择会忠实优化指标，也会暴露指标缺陷。**奢侈品最终选择全零模型，MAE 最低但没有业务事件召回能力。
- **解释性介于单模型与黑盒之间。**组件权重和 permutation importance 比单一深度网络更透明，但不如 Prophet 的趋势与季节分解直观，也不能提供单日因果解释。

![AutoGluon 与原十种模型的最终测试 WAPE](../runs/autogluon-validation-20261020/autogluon_timeseries/comparison_with_all_models.png)

## 1. 实验设计

### 1.1 为什么使用 AutoGluon TimeSeries

AutoGluon TimeSeries 的 `TimeSeriesPredictor` 能在一次接口调用中比较统计模型、间歇需求模型、表格回归和神经网络，并根据内部验证结果构造 Weighted Ensemble。官方接口还支持已知未来协变量、概率预测、排行榜和 permutation importance。参见 [TimeSeries 快速入门](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-quick-start.html)、[模型库](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html) 和 [特征重要性说明](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.feature_importance.html)。

本项目固定 Torch 2.8.0，而 AutoGluon 1.6.1 要求 Torch 2.10 以上。因此实验使用与现有环境兼容的 AutoGluon 1.5.0，并将 SciPy、scikit-learn 和 TorchMetrics 调整到其允许范围。该差异不会改变实验数据和评测协议，但本报告的结论应标注具体 AutoGluon 版本。

### 1.2 模型库与预算

每套数据分别请求 14 个组件：

- 基线和统计：SeasonalNaive、Zero、ETS、Theta、AutoARIMA、Croston、ADIDA、IMAPA、NPTS。
- 表格：DirectTabular、RecursiveTabular。
- 轻量神经网络：DeepAR、DLinear、SimpleFeedForward。
- 在可用组件之上自动训练 Weighted Ensemble。

Chronos、Chronos2、Toto、Toto2、PatchTST 和 AutoGluon TFT 被明确排除。这既避免下载预训练权重，也避免在仅有约三年日数据的单序列上投入与旧模型明显不对等的算力。

每个场景有四个外部滚动验证窗口，每次拟合上限 90 秒；最终测试前拟合上限 240 秒。所有组件实际都在上限前结束，因此每场景五次拟合的累计耗时为：红酒 124.8 秒、商超 138.2 秒、庭院 140.2 秒、奢侈品 171.7 秒。

### 1.3 与原报告保持一致的边界

- 四套输入 CSV、节假日数据及哈希与原报告完全一致。
- 四个滚动窗口和最终 28 天测试日期保持不变。
- 闭店和缺货目标在训练中设为缺失，评测时不计入有效标签。
- 预测执行非负约束，闭店日强制为零。
- 未来只提供营业状态、节假日和计划业务变量。
- `conditional_mean`、`latent_demand`、未来销量、缺货结果、发生概率和正需求量均未进入模型。
- 测试集不参与 AutoGluon 的内部模型选择或 ensemble 权重学习。

AutoGluon 没有与原协议严格同义的 `train_hindcast`。本报告不制造一个看似可比的训练成绩，而是用共同的 validation、test 和 `test WAPE - validation WAPE` 比较泛化表现。

## 2. 商超：集成有效，但仍落后于专用时间序列模型

### 2.1 结果

| 模型 | 验证 WAPE | 测试 WAPE | 测试 MAE | 测试 RMSE | 测试 Bias |
|---|---:|---:|---:|---:|---:|
| ARIMA | 3.79% | **4.21%** | **4.801** | 6.169 | -0.46% |
| Prophet | **3.79%** | 4.22% | 4.813 | 6.185 | -0.46% |
| LSTM | 4.05% | 4.30% | 4.898 | **6.161** | 0.04% |
| XGBoost | 4.70% | 4.61% | 5.257 | 6.713 | -2.49% |
| **AutoGluon Ensemble** | 6.03% | **4.73%** | **5.397** | 7.403 | -2.21% |
| Seasonal naive | 14.49% | 16.21% | 18.481 | 21.125 | -2.24% |

AutoGluon 的测试 WAPE 比 Seasonal naive 降低 70.8%，是一个强基线；但相对 ARIMA 仍高 0.52 个百分点，相对 Prophet 高 0.51 个百分点。验证期差距更大，AutoGluon 验证 MAE 为 7.294，而 Prophet 为 4.564。

### 2.2 Ensemble 做了什么

最终 ensemble 权重为：RecursiveTabular 58.9%、DLinear 21.1%、DeepAR 13.7%、AutoARIMA 6.3%。测试中 Weighted Ensemble 的 MAE 为 5.397，优于最佳单组件 RecursiveTabular 的 5.787，也明显优于 DirectTabular 的 7.868。这说明集成不是形式上的包装，而是产生了真实增益。

但是四个滚动窗口的 ensemble 组成变化较大：普通窗口以 DeepAR 为主，低需求窗口以 DirectTabular 为主，最终模型又以 RecursiveTabular 为主。这说明不同阶段偏好的结构不稳定，AutoGluon 通过加权缓冲变化，但没有学到 Prophet 和 ARIMA 那样稳定的显式周期结构。

### 2.3 可解释性

Permutation importance 显示，扰动折扣变量会令 MAE 增加约 10.80，远高于其他变量；营业状态和闭店节假日的重要性接近零且置信区间跨零。折扣的重要性与原报告中的 Prophet、树模型和神经网络解释一致。

这项解释回答“模型是否依赖折扣”，但不能回答某一天为何预测 118 件，也不能把 10.80 解释为折扣的因果销量提升。

### 2.4 评论

理论上，AutoGluon 应通过多模型集成降低模型选择风险；实际结果支持“降低单组件风险”，但不支持“自动超过最佳专用模型”。商超的规则较稳定，显式趋势、周期和促销结构足以让 Prophet/ARIMA 达到更低误差。AutoGluon 更适合作为自动 challenger，而不是替换当前首选。

## 3. 红酒：集成优于内部组件，但促销峰值仍被低估

### 3.1 结果

| 模型 | 验证 WAPE | 测试 WAPE | 测试 MAE | 测试 RMSE | 测试 Bias |
|---|---:|---:|---:|---:|---:|
| LSTM | 27.89% | **21.26%** | **2.441** | 3.247 | -7.95% |
| Prophet | **25.22%** | 21.26% | 2.441 | **2.967** | 0.74% |
| ARIMA | 25.42% | 21.27% | 2.442 | 2.994 | -3.67% |
| XGBoost | 27.20% | 21.36% | 2.452 | 3.544 | -9.20% |
| **AutoGluon Ensemble** | 29.09% | **25.58%** | **2.937** | 4.629 | -12.10% |
| Seasonal naive | 55.34% | 59.68% | 6.852 | 8.863 | 35.16% |

AutoGluon 测试排名第 9，只优于 CatBoost 和单棵决策树；不过它仍比 Seasonal naive 的 WAPE 低 57.1%。其主要问题是整体低估：测试实际总量 310，预测总量约 272.5，Bias 为 -12.1%。

### 3.2 Ensemble 与业务窗口

最终权重为 DirectTabular 55.8%、SimpleFeedForward 24.2%、DLinear 8.4%、NPTS 5.3%、RecursiveTabular 4.2%，其余权重很小。Weighted Ensemble 测试 MAE 2.937，优于最佳内部单组件 DirectTabular 的 2.991，证明集成仍有价值。

在促销验证窗口中，AutoGluon WAPE 为 25.36%，明显高于 Prophet 的 15.39%；普通窗口 WAPE 为 30.15%。促销峰值未被充分捕捉，是它落后于 Prophet、ARIMA 和 LSTM 的关键原因。

### 3.3 可解释性与评论

整箱促销的 permutation importance 为 1.215，远高于聚会事件、12 月旺季和营业状态。方向与原模型一致，但模型即使识别了促销重要，也不代表正确估计了促销增量。

结果与 AutoML 理论没有真正冲突：AutoGluon 在自己的候选空间和内部验证期内选择了较优组合，但自动选择不能补偿稀少促销事件、有限历史以及外部窗口分布变化。红酒仍应以 Prophet 为稳定首选，LSTM 和 ARIMA 为挑战者；当前 AutoGluon 适合作为额外基线。

## 4. 庭院用品：接近最佳树模型，但没有复制 MLP 的测试优势

### 4.1 结果

| 模型 | 验证 WAPE | 测试 WAPE | 测试 MAE | 测试 RMSE | 测试 Bias |
|---|---:|---:|---:|---:|---:|
| MLP | 12.43% | **9.76%** | **5.383** | **6.609** | 0.98% |
| LSTM | 11.27% | 10.16% | 5.604 | 6.856 | -0.30% |
| Prophet | **9.47%** | 10.47% | 5.775 | 6.781 | -3.68% |
| ARIMA | 9.48% | 10.50% | 5.792 | 6.796 | -3.58% |
| LightGBM | 9.69% | 11.37% | 6.272 | 7.552 | -5.60% |
| **AutoGluon Ensemble** | 9.74% | **11.54%** | **6.365** | 7.449 | -0.63% |
| Seasonal naive | 13.76% | 19.27% | 10.630 | 14.067 | -15.51% |

AutoGluon 的验证 WAPE 接近 LightGBM，但其验证 MAE 为 7.724，明显高于 Prophet 的 6.325。原因是 WAPE 按总销量加权，高销量窗口影响更大；MAE 对每个有效日期等权。既然模型选择协议以 MAE 为主，不能只用 9.74% 的 WAPE 宣称 AutoGluon 验证排名第三。

### 4.2 Ensemble 与业务窗口

最终 ensemble 由 SimpleFeedForward 51.7%、DeepAR 27.0%、RecursiveTabular 11.2%、NPTS 9.0% 和少量 DLinear 构成。它的内部验证成绩优于所有组件，但最终测试中 DeepAR 的 MAE 为 6.111，反而优于 ensemble 的 6.365。测试集不能用于反向选择 DeepAR，因此正式成绩仍必须使用事先选择的 ensemble。

普通验证窗口中 AutoGluon WAPE 为 8.72%，表现尚可；促销窗口升至 13.82%，节假日窗口达到 16.93%。相较之下，原报告中 MLP 的促销 WAPE 为 7.50%，Prophet 和 ARIMA 在节假日窗口更稳。

### 4.3 可解释性与评论

秋末清仓和长周末 DIY 的 permutation importance 均为正，但 95% 置信区间跨零；仅三次置换不足以断言二者稳定重要。最终 ensemble 更依赖神经组件，也使解释进一步停留在全局敏感性层面。

AutoGluon 没有复制 MLP 在最终测试期的第一名，但表现优于 XGBoost、CatBoost、随机森林、单树和 TFT。它是合格的中上游模型，却没有解决原报告已指出的阶段性差异：庭院需求的活动窗口仍需要专门校准。

## 5. 奢侈品：自动选择准确暴露了指标问题

### 5.1 结果

| 模型 | 验证 MAE | 测试 MAE | 测试 RMSE | 测试 WAPE | 测试 Bias |
|---|---:|---:|---:|---:|---:|
| **AutoGluon Ensemble** | **0.127** | **0.185** | 0.430 | 100.0% | -100.0% |
| CatBoost | 0.137 | **0.185** | 0.430 | 100.0% | -100.0% |
| Seasonal naive | 0.184 | **0.185** | 0.430 | 100.0% | -100.0% |
| TFT | 0.209 | 0.191 | 0.422 | 102.9% | -92.9% |
| Prophet | 0.147 | 0.264 | **0.379** | 142.4% | -18.4% |

AutoGluon 的验证 MAE 是所有方法中最低的，但四个外部窗口均选择 NPTS 或 SeasonalNaive，预测接近全零；最终内部选择为 Zero。测试实际总量为 5，预测总量为 0，因此 WAPE 和 Bias 分别为 100% 与 -100%。

### 5.2 Ensemble、区间和解释性

最终 Zero、SeasonalNaive、NPTS 和 WeightedEnsemble 的测试结果完全相同。所谓 ensemble 实际退化为 Zero，权重为 100%。全部业务协变量的 permutation importance 都为零，因为输出不随输入变化。

10%–90% 预测区间宽度也是 0。区间只覆盖 81.5% 的有效测试日，覆盖的都是零销量日期；所有正销量事件均落在区间之外。这说明概率区间在退化模型上不能自动提供可靠的不确定性保护。

### 5.3 评论

从纯 MAE 角度，AutoGluon 的选择完全合理：93.58% 的有效营业日销量为零，全零预测最小化了多数日期的绝对误差。从库存业务角度，它没有识别任何购买事件，不能称为更好的需求预测。

AutoGluon 的优势恰好在这里变成风险：它会高效地优化指定指标，但不会替业务方决定漏报一次高价值销售与多备一件库存哪个成本更高。奢侈品仍需两阶段发生模型、事件召回率、库存成本或分位数服务水平目标；不能仅用当前 MAE 排名上线。

## 6. 概率预测是否可靠

AutoGluon 输出 10%、50% 和 90% 分位数。最终测试的经验覆盖率如下：

| 数据 | 10%–90% 区间覆盖率 | 平均区间宽度 | 判断 |
|---|---:|---:|---|
| 商超 | 100.0% | 56.86 | 覆盖充分但明显偏宽 |
| 红酒 | 100.0% | 24.42 | 相对日均销量很宽 |
| 庭院 | 100.0% | 32.55 | 覆盖充分，信息量有限 |
| 奢侈品 | 81.5% | 0.00 | 退化为零区间，漏掉全部正需求事件 |

名义 80% 区间在三个连续需求场景中达到 100% 覆盖，并不代表校准优秀；过宽区间也能获得高覆盖。测试期只有 27 个有效日，尚不足以稳定评价尾部概率。生产使用前应跨更多滚动窗口计算覆盖率和区间得分，而不是只看一段测试期。

![AutoGluon 最终 28 天点预测与区间](../runs/autogluon-validation-20261020/autogluon_timeseries/final_test_predictions.png)

## 7. 可解释性比较

### 7.1 AutoGluon 提供的三层解释

1. **Leaderboard：**说明各组件在内部验证窗口中的精度、拟合时间和预测时间。
2. **Ensemble 权重：**说明最终预测由哪些组件组合，但权重会随预测起点变化。
3. **Permutation importance：**说明打乱业务变量后整体 MAE 增加多少，并提供采样不确定性。

![四个场景最终 AutoGluon ensemble 权重](../runs/autogluon-validation-20261020/autogluon_timeseries/final_ensemble_weights.png)

### 7.2 与原模型家族比较

| 家族 | 最容易解释的内容 | 主要解释缺口 |
|---|---|---|
| Prophet | 趋势、周/年季节、节假日和回归量分解 | 非线性交互表达有限，组件不是因果效应 |
| 树模型 | 分裂/增益和统一 permutation importance | 原生重要性尺度跨算法不可比，相关特征会互相替代 |
| 神经网络 | permutation sensitivity | 内部时序表示和单日决策较难解释 |
| AutoGluon | 组件权重、排行榜、协变量置换敏感性 | 多层组合增加黑盒程度，权重不稳定，无法直接解释某一天 |

因此，AutoGluon 比只给出单一预测的黑盒系统透明，但不比结构明确的 Prophet 更容易向业务读者解释。它适合回答“系统主要依赖哪个模型家族、哪个业务变量”，不适合回答“这一天为什么比昨天多预测 12 件”。

## 8. 优点、缺点与适用建议

### 优点

- 用统一接口覆盖多类预测模型，减少手工实现和调参代码。
- Weighted Ensemble 在商超和红酒测试中确实优于所有内部单组件。
- 自动记录内部排行榜、拟合时间和预测时间，便于工程诊断。
- 原生提供分位数预测和 permutation importance。
- 某些组件因预算未训练时仍能用其余组件完成预测，失败容错较好。
- 对奢侈品自动识别到零模型的 MAE 优势，准确反映了目标函数的数学结果。

### 缺点

- 依赖较重；为兼容 AutoGluon 1.5.0，需要调整 SciPy、scikit-learn 和 TorchMetrics 版本。
- 结果高度依赖时间预算、候选顺序、内部验证窗口和允许的模型库。
- 自动 ensemble 比单模型更难解释，且四个外部窗口的权重变化明显。
- 在单条、约三年的日序列上，深度组件无法发挥多商品全局学习优势。
- 概率区间可能过宽或退化，不能假定天然校准良好。
- AutoML 只优化提供的指标；奢侈品案例证明，指标与业务目标不一致时，自动化会放大而不是修复问题。

### 建议

- **商超：**保留 Prophet/ARIMA 为主模型，将 AutoGluon 作为自动 challenger 和故障备用。
- **红酒：**继续使用 Prophet、ARIMA、LSTM 的三模型比较；暂不采用 AutoGluon 替换主模型。
- **庭院：**AutoGluon 可与 LightGBM 一起作为稳定候选，但 MLP 的测试优势仍需更多窗口复现。
- **奢侈品：**不要直接上线当前 AutoGluon Zero；先把目标从单一 MAE 扩展为发生事件与库存成本。
- **数据扩展后重试：**当项目拥有多个 SKU、门店或更长历史时，可使用多序列全局训练重新评估 DeepAR、DLinear 和更复杂模型。当前“每场景单序列”的结论不代表 AutoGluon 在大规模面板数据上的能力上限。

## 9. 与原全模型报告的最终关系

原报告的核心结论保持不变：Prophet 是四类当前数据上最稳妥的统一默认模型，但没有单一模型在所有场景获胜。加入 AutoGluon 后，应补充一条：

> AutoGluon 是高质量的自动基线和模型组合工具；在当前四条短序列、固定 CPU 预算和相同业务特征下，它提高了实验覆盖面，却没有取代针对性训练的最佳模型。

如果必须在工程上选择统一入口，可以让 AutoGluon承担“候选生成与 challenger 管理”，同时保留 Prophet/ARIMA 等专用模型进入同一外部滚动验证。最终决策仍应由共同的样本外指标和业务成本决定，而不是由 AutoGluon 的内部 leaderboard 单独决定。

## 附录：证据与复现

- Notebook：`notebooks/35-autogluon-timeseries-tutorial.ipynb`
- AutoGluon 运行：`runs/autogluon-validation-20261020/autogluon_timeseries/`
- 原模型汇总：`runs/all-models-summary-validation-20261020/all_model_comparison/`
- 原报告：`reports/2026-10-20-all-models-validation-report.md`
- 输入数据：`data/teaching/`

主要结果文件包括 `aggregate_metrics.csv`、`predictions.csv`、`leaderboard.csv`、`ensemble_weights.csv`、`component_test_diagnostics.csv`、`feature_importance.csv`、`business_slices.csv`、`model_failures.csv` 和 `comparison_with_all_models.csv`。

保存后的四个最终 predictor 已重新加载，并逐场景复算预测；结果与首次导出的点预测在 `1e-8` 容差内一致。Manifest 中的全部输入哈希也与当前 CSV 一致。

本报告只解释当前数据、当前外部窗口、AutoGluon 1.5.0、固定模型库与 CPU 预算下的结果。特别是测试期只有 27 个有效日，细小排名差异不应推广为模型家族的普遍优劣。
