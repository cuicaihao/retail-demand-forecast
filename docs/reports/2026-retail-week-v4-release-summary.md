# 2026 零售周 V4 发布摘要

这是一页式发布入口，只汇总当前审核 release，不保留 V2/V3 排名。详细方法、fold 指标、
区间诊断和限制见 [V4 完整技术报告](2026-retail-week-v4-first-stage-report.md)。

<!-- BENCHMARK_BLOCK:provenance:START -->
> Benchmark ID: `2026-retail-week-expanding-v4`<br>
> Release ID: `20261106-timesfm-v1`<br>
> Protocol hash: `281307a393825938d595cf34f22be25c9a894f634ddbadcf073c2d82eea8612e`<br>
> Release manifest SHA-256: `4a086433f728cfa48ea301b4876a011ef1a6b4ca3b42eb32f55053224c34c51e`<br>
> Fold uncertainty intervals are descriptive; they are not claims of statistical significance.
<!-- BENCHMARK_BLOCK:provenance:END -->

## 1. 场景能力矩阵

下表只包含每个工具包预先声明的广度代表。四种需求尺度不同，不计算跨场景平均分或总冠军。

<!-- BENCHMARK_BLOCK:capability_matrix:START -->
| 模型 | 工具包 | Grocery WAPE | Wine WAPE | Garden WAPE | Luxury WAPE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Default (MAE) | catboost | 5.05% | 24.40% | 8.86% | 100.25% |
| Decision Tree | decision_tree | 6.95% | 27.05% | 10.75% | 238.73% |
| LSTM | pytorch_lstm | 4.44% | 23.20% | 11.43% | 204.49% |
| LightGBM Default (MSE) | lightgbm | 5.31% | 25.08% | 9.17% | 239.15% |
| MLP | pytorch_mlp | 4.66% | 23.23% | 8.82% | 202.14% |
| Moving Average (14d) | moving_average | 12.45% | 39.99% | 20.02% | 161.82% |
| Prophet | prophet | 4.16% | 22.78% | 7.81% | 172.55% |
| Random Forest | random_forest | 5.39% | 25.81% | 9.33% | 225.78% |
| Regression + ARIMA (1,0,1) | statsmodels | 4.17% | 22.10% | 7.80% | 209.29% |
| Seasonal Naive (7d) | seasonal_naive | 15.28% | 49.26% | 15.36% | 182.76% |
| TFT | pytorch_forecasting | 4.99% | 24.40% | 8.76% | 195.41% |
| TimesFM 3.0 | timesfm_3_0 | 4.91% | 24.29% | 8.86% | 100.35% |
| XGBoost Default (MSE) | xgboost | 5.01% | 23.94% | 9.03% | 224.11% |
<!-- BENCHMARK_BLOCK:capability_matrix:END -->

当前扩展历史广度榜中，Grocery 的最低 WAPE 是 Prophet 的 `4.16%`；Wine 和 Garden
分别是 Regression + ARIMA 的 `22.10%` 和 `7.80%`。Luxury 的 CatBoost WAPE 为
`100.25%`，但事件 F1 只有 `9.09%`，因此不能仅凭 WAPE 宣称有效预测了购买事件。

## 2. AutoML 与普通代表模型

AutoGluon 是独立 AutoML 层，不占据普通工具包广度榜名次。表中的普通模型是每个场景
观察到的最低广度 WAPE，仅用于描述，不是事前模型选择规则。

<!-- BENCHMARK_BLOCK:automl_vs_standard:START -->
| 场景 | 观察到的最佳普通广度模型 | 普通模型 WAPE | AutoML 输出 | AutoML WAPE | 差值 (AutoML-普通) | 普通模型 Event F1 | AutoML Event F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| garden | Regression + ARIMA (1,0,1) | 7.80% | AG Weighted Ensemble | 10.19% | +2.39 pp | 100.00% | 100.00% |
| grocery | Prophet | 4.16% | AG Weighted Ensemble | 5.52% | +1.36 pp | 100.00% | 100.00% |
| luxury | CatBoost Default (MAE) | 100.25% | AG Weighted Ensemble | 100.00% | -0.25 pp | 9.09% | 0.00% |
| wine | Regression + ARIMA (1,0,1) | 22.10% | AG Weighted Ensemble | 26.20% | +4.10 pp | 100.00% | 100.00% |
<!-- BENCHMARK_BLOCK:automl_vs_standard:END -->

扩展历史运行中，AutoGluon 共享训练套件记录约 `2,398` 秒；TimesFM 单变量和已知协变量
配置分别约 `10` 秒和 `63` 秒。AutoML 包含训练、验证和集成，TimesFM 是预训练权重的
零样本推理，因此这些耗时不能解释为同一种计算任务的直接效率排名。

## 3. TimesFM 3.0 配置消融

`univariate` 是工具包广度代表；`known_covariates` 是独立消融。Coverage80 是经验覆盖率，
没有经过库存服务水平校准。

<!-- BENCHMARK_BLOCK:timesfm_ablation:START -->
| 场景 | 模型 | 配置 | WAPE | Coverage80 | 80% interval width | 80% pinball loss | Event F1@0.5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| garden | TimesFM 3.0 + Known Covariates | known_covariates | 8.17% | 79.56% | 18.7483 | 1.2607 | 100.00% |
| garden | TimesFM 3.0 | univariate | 8.86% | 74.03% | 18.3776 | 1.4512 | 100.00% |
| grocery | TimesFM 3.0 + Known Covariates | known_covariates | 4.54% | 81.45% | 18.1489 | 1.2776 | 100.00% |
| grocery | TimesFM 3.0 | univariate | 4.91% | 83.19% | 19.9169 | 1.4906 | 100.00% |
| luxury | TimesFM 3.0 + Known Covariates | known_covariates | 100.41% | 91.44% | 0.0161 | 0.0401 | 0.00% |
| luxury | TimesFM 3.0 | univariate | 100.35% | 93.65% | 0.0787 | 0.0410 | 0.00% |
| wine | TimesFM 3.0 + Known Covariates | known_covariates | 22.69% | 77.07% | 8.0549 | 0.5904 | 100.00% |
| wine | TimesFM 3.0 | univariate | 24.29% | 75.69% | 8.2645 | 0.6255 | 100.00% |
<!-- BENCHMARK_BLOCK:timesfm_ablation:END -->

已知协变量在 Grocery、Wine 和 Garden 上降低 WAPE，但 Luxury 的两个配置 Event F1 都为
`0%`。在极端零膨胀场景中，接近 `100%` 的 WAPE 可能来自接近全零预测，而非事件识别。

## 4. 扩展历史与固定 104 周

<!-- BENCHMARK_BLOCK:history_provenance:START -->
> Comparison ID: `2026-retail-week-expanding-v4--vs--2026-retail-week-104w-rolling-v4`<br>
> Release ID: `20261106-timesfm-v1`<br>
> Main source: `2026-retail-week-expanding-v4` / `20261106-timesfm-v1`<br>
> Fixed-window source: `2026-retail-week-104w-rolling-v4` / `20261106-timesfm-v1`<br>
> Comparison release manifest SHA-256: `3711fa3d1cbc6f3a8874d3a7f534cb374cf3168d7b23941a0aa056508508c385`<br>
> WAPE differences are paired descriptive results, not statistical significance claims.
<!-- BENCHMARK_BLOCK:history_provenance:END -->

<!-- BENCHMARK_BLOCK:history_policy_summary:START -->
| 场景 | 配置数 | 扩展历史胜 | 104 周胜 | 平局 | WAPE 差值中位数 (104w-expanding) | 广度配置 WAPE 差值中位数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| garden | 41 | 23 | 13 | 5 | +0.03 pp | +0.05 pp |
| grocery | 41 | 30 | 4 | 7 | +0.19 pp | +0.18 pp |
| luxury | 41 | 17 | 19 | 5 | +0.00 pp | -3.58 pp |
| wine | 41 | 32 | 2 | 7 | +0.75 pp | +0.54 pp |
<!-- BENCHMARK_BLOCK:history_policy_summary:END -->

正差值表示固定 104 周的 WAPE 更高。Grocery 和 Wine 多数配置受益于更长历史；Garden
方向混合；Luxury 必须联合事件召回和 F1 阅读，不能把较低 WAPE 自动解释为改善。

## 5. 发布边界

- 当前结果来自一套固定生成协议和一个随机种子，不代表真实企业分布；
- 正式评估覆盖 52 个完整零售周，不等同于完整日历年；
- 模型只允许读取预测起点可见历史和已知未来计划变量；
- 所有 bootstrap 和 fold 区间均为描述性结果，不宣称统计显著；
- TimesFM 3.0 默认权重受非商业许可证约束；
- 下一项科学实验应是独立合成数据种子复现，而不是继续增加模型数量。
