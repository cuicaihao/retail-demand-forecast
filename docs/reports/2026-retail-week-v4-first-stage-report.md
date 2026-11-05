# 2026 零售周 V4 首阶段基准报告

本报告记录 `2026-retail-week-expanding-v4` 的首阶段审核结果，并使用
`2026-retail-week-24w-ablation-v4` 量化固定 24 周训练历史带来的影响。正式评估覆盖
2025-12-29 至 2026-12-27，共 13 个互不重叠的 28 天 fold，也就是 52 个完整零售周
和 364 个评估日；它不是完整日历年。

当前发布包含基线、Prophet、经典树，以及 LightGBM、XGBoost、CatBoost 的默认配置
与对称 Tweedie `p=1.1/1.2/1.5/1.8` 消融，共 20 个模型配置。ARIMA、神经网络、
AutoGluon 和 TimesFM 尚未迁移到 V4，因此不进入本报告的当前结论。

## 1. 证据边界与主要结论

- Grocery、Wine 和 Garden 中，扩展历史在 20 个配置里均取得 18 胜、2 平；两个平局
  来自不依赖拟合历史长度的 Seasonal Naive 与 Moving Average。
- Garden 对训练历史最敏感。代表配置的 WAPE 差值中位数为 `+6.79 pp`
  （24 周减扩展历史），符合年度季节性需要跨年证据的预期。
- Luxury 呈现 9 胜、9 负、2 平，且 WAPE 与需求事件 F1 并不总是同向。因此这里不能
  得出“历史越长一定越好”或基于 WAPE 的单一冠军结论。
- 广度榜每个工具包只保留一个预先声明的代表配置；Tweedie power 的比较单列为配置
  消融，不重复占据工具包广度排名。
- 所有区间和 fold 分布只用于描述稳定性。本报告没有进行预注册的显著性检验，也不把
  bootstrap 描述区间表述为统计显著。

## 2. 主榜发布来源

<!-- BENCHMARK_BLOCK:provenance:START -->
> Benchmark ID: `2026-retail-week-expanding-v4`  
> Release ID: `20261106-first-stage-v1`  
> Protocol hash: `281307a393825938d595cf34f22be25c9a894f634ddbadcf073c2d82eea8612e`  
> Release manifest SHA-256: `4435debb0c4ed09e699359c3ca8cd3a7137b806c4d17d40978af39afc9506b98`  
> Fold uncertainty intervals are descriptive; they are not claims of statistical significance.
<!-- BENCHMARK_BLOCK:provenance:END -->

模型执行使用冻结参数和正式评估开始前结束的选择窗口。发布快照只保存轻量预测、指标、
配置和 manifest；模型权重与临时训练产物不进入 Git。

## 3. 工具包广度榜

<!-- BENCHMARK_BLOCK:scenario_leaderboards:START -->
## grocery

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Prophet | prophet | default | 4.16% | 5.0931 | 6.4422 | 0.19% | 0.1864 | 0.0414 | 0.0058 |
| XGBoost Default (MSE) | xgboost | default | 5.01% | 6.1295 | 7.8354 | -1.17% | 0.2244 | 0.0485 | 0.0178 |
| CatBoost Default (MAE) | catboost | default | 5.05% | 6.1832 | 8.2256 | -1.33% | 0.2263 | 0.0512 | 0.0154 |
| LightGBM Default (MSE) | lightgbm | default | 5.31% | 6.5005 | 8.5586 | -1.48% | 0.2379 | 0.0533 | 0.0138 |
| Random Forest | random_forest | default | 5.39% | 6.5940 | 8.5277 | -1.67% | 0.2414 | 0.0546 | 0.0048 |
| Decision Tree | decision_tree | default | 6.95% | 8.5039 | 10.8132 | -1.51% | 0.3113 | 0.0672 | 0.0160 |
| Moving Average (14d) | moving_average | default | 12.45% | 15.2312 | 18.4582 | -0.38% | 0.5575 | 0.1240 | 0.0202 |
| Seasonal Naive (7d) | seasonal_naive | default | 15.28% | 18.6957 | 22.9534 | -8.89% | 0.6843 | 0.1484 | 0.0190 |

## wine

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Prophet | prophet | default | 22.78% | 2.6649 | 3.3746 | -1.03% | 0.4431 | 0.2223 | 0.0432 |
| XGBoost Default (MSE) | xgboost | default | 23.94% | 2.8004 | 3.5918 | -2.26% | 0.4656 | 0.2278 | 0.0671 |
| CatBoost Default (MAE) | catboost | default | 24.40% | 2.8551 | 3.7194 | -6.04% | 0.4747 | 0.2307 | 0.0692 |
| LightGBM Default (MSE) | lightgbm | default | 25.08% | 2.9346 | 3.7990 | -4.04% | 0.4879 | 0.2440 | 0.0786 |
| Random Forest | random_forest | default | 25.81% | 3.0190 | 3.8472 | 0.19% | 0.5020 | 0.2554 | 0.0344 |
| Decision Tree | decision_tree | default | 27.05% | 3.1644 | 4.0257 | -5.54% | 0.5261 | 0.2800 | 0.0726 |
| Moving Average (14d) | moving_average | default | 39.99% | 4.6780 | 6.1696 | -4.28% | 0.7778 | 0.4073 | 0.1260 |
| Seasonal Naive (7d) | seasonal_naive | default | 49.26% | 5.7624 | 8.0430 | 4.39% | 0.9581 | 0.4893 | 0.1176 |

## garden

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Prophet | prophet | default | 7.81% | 5.6108 | 6.9749 | -0.23% | 0.5906 | 0.0881 | 0.0628 |
| CatBoost Default (MAE) | catboost | default | 8.86% | 6.3611 | 7.8148 | -1.59% | 0.6696 | 0.0889 | 0.0848 |
| XGBoost Default (MSE) | xgboost | default | 9.03% | 6.4864 | 8.0532 | -1.53% | 0.6828 | 0.0880 | 0.0855 |
| LightGBM Default (MSE) | lightgbm | default | 9.17% | 6.5871 | 8.0916 | -1.21% | 0.6934 | 0.0880 | 0.0747 |
| Random Forest | random_forest | default | 9.33% | 6.7041 | 8.2132 | -4.16% | 0.7057 | 0.1032 | 0.0773 |
| Decision Tree | decision_tree | default | 10.75% | 7.7236 | 9.4988 | -3.46% | 0.8130 | 0.1214 | 0.0724 |
| Seasonal Naive (7d) | seasonal_naive | default | 15.36% | 11.0304 | 15.5556 | -2.92% | 1.1611 | 0.1695 | 0.0984 |
| Moving Average (14d) | moving_average | default | 20.02% | 14.3816 | 18.6762 | -0.62% | 1.5139 | 0.2301 | 0.1179 |

## luxury

| 模型 | 工具包 | 配置 | WAPE | MAE | ZeroRecall@0.5 | DemandRecall@0.5 | Event F1@0.5 | Positive-demand MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Default (MAE) | catboost | default | 100.25% | 0.0803 | 100.00% | 4.76% | 9.09% | 1.3355 |
| Moving Average (14d) | moving_average | default | 161.82% | 0.1296 | 100.00% | 0.00% | 0.00% | 1.2891 |
| Prophet | prophet | default | 172.55% | 0.1382 | 99.12% | 4.76% | 8.00% | 1.2954 |
| Seasonal Naive (7d) | seasonal_naive | default | 182.76% | 0.1464 | 93.55% | 9.52% | 8.89% | 1.2857 |
| XGBoost Default (MSE) | xgboost | default | 224.11% | 0.1795 | 98.24% | 0.00% | 0.00% | 1.2531 |
| Random Forest | random_forest | default | 225.78% | 0.1809 | 99.41% | 14.29% | 23.08% | 1.1967 |
| Decision Tree | decision_tree | default | 238.73% | 0.1912 | 96.77% | 9.52% | 11.76% | 1.2593 |
| LightGBM Default (MSE) | lightgbm | default | 239.15% | 0.1916 | 98.53% | 4.76% | 7.41% | 1.2444 |
<!-- BENCHMARK_BLOCK:scenario_leaderboards:END -->

Luxury 表必须联合阅读 WAPE、`ZeroRecall@0.5`、`DemandRecall@0.5`、事件 F1 和正需求
MAE。高零值召回可能仅表示模型倾向预测零，不能独立证明它识别了真实需求事件。

## 4. GBDT 配置消融

<!-- BENCHMARK_BLOCK:configuration_ablations:START -->
## grocery / lightgbm

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM Default (MSE) | lightgbm | default | 5.31% | 6.5005 | 8.5586 | -1.48% | 0.2379 | 0.0533 | 0.0138 |
| LightGBM Tweedie (p=1.5) | lightgbm | tweedie_p15 | 5.37% | 6.5632 | 8.7291 | -1.95% | 0.2402 | 0.0557 | 0.0137 |
| LightGBM Tweedie (p=1.1) | lightgbm | tweedie_p11 | 5.40% | 6.6071 | 8.7282 | -1.78% | 0.2418 | 0.0548 | 0.0142 |
| LightGBM Tweedie (p=1.8) | lightgbm | tweedie_p18 | 5.44% | 6.6556 | 8.7709 | -1.88% | 0.2436 | 0.0558 | 0.0094 |
| LightGBM Tweedie (p=1.2) | lightgbm | tweedie_p12 | 5.55% | 6.7829 | 8.9827 | -1.96% | 0.2483 | 0.0567 | 0.0161 |

## grocery / xgboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost Tweedie (p=1.1) | xgboost | tweedie_p11 | 4.86% | 5.9501 | 7.5671 | -1.22% | 0.2178 | 0.0513 | 0.0141 |
| XGBoost Tweedie (p=1.2) | xgboost | tweedie_p12 | 4.87% | 5.9590 | 7.5603 | -1.30% | 0.2181 | 0.0510 | 0.0118 |
| XGBoost Tweedie (p=1.5) | xgboost | tweedie_p15 | 4.92% | 6.0206 | 7.7636 | -1.48% | 0.2204 | 0.0499 | 0.0170 |
| XGBoost Default (MSE) | xgboost | default | 5.01% | 6.1295 | 7.8354 | -1.17% | 0.2244 | 0.0485 | 0.0178 |
| XGBoost Tweedie (p=1.8) | xgboost | tweedie_p18 | 5.03% | 6.1537 | 7.8465 | -1.32% | 0.2253 | 0.0514 | 0.0156 |

## grocery / catboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Tweedie (p=1.2) | catboost | tweedie_p12 | 4.94% | 6.0435 | 7.7332 | -1.24% | 0.2212 | 0.0461 | 0.0116 |
| CatBoost Tweedie (p=1.5) | catboost | tweedie_p15 | 4.97% | 6.0820 | 7.7716 | -1.61% | 0.2226 | 0.0487 | 0.0148 |
| CatBoost Tweedie (p=1.1) | catboost | tweedie_p11 | 5.00% | 6.1199 | 7.7418 | -1.44% | 0.2240 | 0.0486 | 0.0078 |
| CatBoost Default (MAE) | catboost | default | 5.05% | 6.1832 | 8.2256 | -1.33% | 0.2263 | 0.0512 | 0.0154 |
| CatBoost Tweedie (p=1.8) | catboost | tweedie_p18 | 5.15% | 6.2981 | 8.0769 | -2.21% | 0.2305 | 0.0519 | 0.0161 |

## wine / lightgbm

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM Tweedie (p=1.5) | lightgbm | tweedie_p15 | 23.92% | 2.7989 | 3.6101 | -4.02% | 0.4654 | 0.2329 | 0.0870 |
| LightGBM Tweedie (p=1.1) | lightgbm | tweedie_p11 | 23.96% | 2.8033 | 3.6476 | -3.58% | 0.4661 | 0.2366 | 0.0830 |
| LightGBM Tweedie (p=1.2) | lightgbm | tweedie_p12 | 24.21% | 2.8320 | 3.6899 | -4.43% | 0.4709 | 0.2364 | 0.0869 |
| LightGBM Tweedie (p=1.8) | lightgbm | tweedie_p18 | 24.48% | 2.8639 | 3.6991 | -5.54% | 0.4762 | 0.2332 | 0.0877 |
| LightGBM Default (MSE) | lightgbm | default | 25.08% | 2.9346 | 3.7990 | -4.04% | 0.4879 | 0.2440 | 0.0786 |

## wine / xgboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost Default (MSE) | xgboost | default | 23.94% | 2.8004 | 3.5918 | -2.26% | 0.4656 | 0.2278 | 0.0671 |
| XGBoost Tweedie (p=1.8) | xgboost | tweedie_p18 | 24.48% | 2.8639 | 3.7620 | -6.95% | 0.4762 | 0.2402 | 0.0863 |
| XGBoost Tweedie (p=1.1) | xgboost | tweedie_p11 | 24.63% | 2.8810 | 3.7107 | -4.92% | 0.4790 | 0.2476 | 0.0858 |
| XGBoost Tweedie (p=1.5) | xgboost | tweedie_p15 | 24.83% | 2.9049 | 3.8082 | -7.01% | 0.4830 | 0.2443 | 0.0934 |
| XGBoost Tweedie (p=1.2) | xgboost | tweedie_p12 | 24.83% | 2.9050 | 3.7344 | -5.26% | 0.4830 | 0.2451 | 0.0934 |

## wine / catboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Tweedie (p=1.8) | catboost | tweedie_p18 | 23.26% | 2.7208 | 3.4742 | -4.10% | 0.4524 | 0.2271 | 0.0538 |
| CatBoost Tweedie (p=1.1) | catboost | tweedie_p11 | 23.65% | 2.7671 | 3.5196 | -2.94% | 0.4601 | 0.2380 | 0.0288 |
| CatBoost Tweedie (p=1.5) | catboost | tweedie_p15 | 23.67% | 2.7693 | 3.5554 | -3.75% | 0.4605 | 0.2351 | 0.0373 |
| CatBoost Tweedie (p=1.2) | catboost | tweedie_p12 | 24.06% | 2.8150 | 3.5914 | -3.82% | 0.4680 | 0.2397 | 0.0504 |
| CatBoost Default (MAE) | catboost | default | 24.40% | 2.8551 | 3.7194 | -6.04% | 0.4747 | 0.2307 | 0.0692 |

## garden / lightgbm

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM Tweedie (p=1.2) | lightgbm | tweedie_p12 | 9.15% | 6.5691 | 8.2037 | -2.28% | 0.6915 | 0.0896 | 0.0892 |
| LightGBM Default (MSE) | lightgbm | default | 9.17% | 6.5871 | 8.0916 | -1.21% | 0.6934 | 0.0880 | 0.0747 |
| LightGBM Tweedie (p=1.1) | lightgbm | tweedie_p11 | 9.21% | 6.6189 | 8.1860 | -2.52% | 0.6967 | 0.0946 | 0.0638 |
| LightGBM Tweedie (p=1.8) | lightgbm | tweedie_p18 | 9.32% | 6.6937 | 8.3763 | -2.98% | 0.7046 | 0.0855 | 0.0901 |
| LightGBM Tweedie (p=1.5) | lightgbm | tweedie_p15 | 9.39% | 6.7460 | 8.3332 | -2.56% | 0.7101 | 0.0924 | 0.0839 |

## garden / xgboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost Tweedie (p=1.8) | xgboost | tweedie_p18 | 8.88% | 6.3793 | 7.9584 | -2.95% | 0.6715 | 0.0874 | 0.0807 |
| XGBoost Tweedie (p=1.5) | xgboost | tweedie_p15 | 8.94% | 6.4216 | 8.0111 | -2.17% | 0.6760 | 0.0814 | 0.0890 |
| XGBoost Tweedie (p=1.2) | xgboost | tweedie_p12 | 8.97% | 6.4425 | 7.9938 | -2.06% | 0.6782 | 0.0842 | 0.0840 |
| XGBoost Tweedie (p=1.1) | xgboost | tweedie_p11 | 9.02% | 6.4817 | 8.0127 | -2.07% | 0.6823 | 0.0888 | 0.0902 |
| XGBoost Default (MSE) | xgboost | default | 9.03% | 6.4864 | 8.0532 | -1.53% | 0.6828 | 0.0880 | 0.0855 |

## garden / catboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Tweedie (p=1.5) | catboost | tweedie_p15 | 8.73% | 6.2690 | 7.8811 | -3.02% | 0.6599 | 0.0878 | 0.0719 |
| CatBoost Tweedie (p=1.8) | catboost | tweedie_p18 | 8.85% | 6.3561 | 7.8965 | -3.21% | 0.6691 | 0.0853 | 0.0693 |
| CatBoost Default (MAE) | catboost | default | 8.86% | 6.3611 | 7.8148 | -1.59% | 0.6696 | 0.0889 | 0.0848 |
| CatBoost Tweedie (p=1.2) | catboost | tweedie_p12 | 8.96% | 6.4367 | 8.1183 | -3.32% | 0.6776 | 0.0873 | 0.0742 |
| CatBoost Tweedie (p=1.1) | catboost | tweedie_p11 | 9.07% | 6.5158 | 8.2189 | -3.24% | 0.6859 | 0.0934 | 0.0687 |

## luxury / lightgbm

| 模型 | 工具包 | 配置 | WAPE | MAE | ZeroRecall@0.5 | DemandRecall@0.5 | Event F1@0.5 | Positive-demand MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM Tweedie (p=1.8) | lightgbm | tweedie_p18 | 150.80% | 0.1208 | 100.00% | 0.00% | 0.00% | 1.2985 |
| LightGBM Tweedie (p=1.5) | lightgbm | tweedie_p15 | 161.70% | 0.1295 | 100.00% | 0.00% | 0.00% | 1.3018 |
| LightGBM Tweedie (p=1.2) | lightgbm | tweedie_p12 | 168.45% | 0.1349 | 100.00% | 0.00% | 0.00% | 1.3010 |
| LightGBM Tweedie (p=1.1) | lightgbm | tweedie_p11 | 172.96% | 0.1386 | 100.00% | 0.00% | 0.00% | 1.2950 |
| LightGBM Default (MSE) | lightgbm | default | 239.15% | 0.1916 | 98.53% | 4.76% | 7.41% | 1.2444 |

## luxury / xgboost

| 模型 | 工具包 | 配置 | WAPE | MAE | ZeroRecall@0.5 | DemandRecall@0.5 | Event F1@0.5 | Positive-demand MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost Tweedie (p=1.8) | xgboost | tweedie_p18 | 109.29% | 0.0876 | 99.71% | 0.00% | 0.00% | 1.3789 |
| XGBoost Tweedie (p=1.5) | xgboost | tweedie_p15 | 128.79% | 0.1032 | 99.12% | 0.00% | 0.00% | 1.3508 |
| XGBoost Tweedie (p=1.2) | xgboost | tweedie_p12 | 164.77% | 0.1320 | 98.83% | 0.00% | 0.00% | 1.3113 |
| XGBoost Tweedie (p=1.1) | xgboost | tweedie_p11 | 185.16% | 0.1483 | 98.53% | 0.00% | 0.00% | 1.3065 |
| XGBoost Default (MSE) | xgboost | default | 224.11% | 0.1795 | 98.24% | 0.00% | 0.00% | 1.2531 |

## luxury / catboost

| 模型 | 工具包 | 配置 | WAPE | MAE | ZeroRecall@0.5 | DemandRecall@0.5 | Event F1@0.5 | Positive-demand MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Default (MAE) | catboost | default | 100.25% | 0.0803 | 100.00% | 4.76% | 9.09% | 1.3355 |
| CatBoost Tweedie (p=1.8) | catboost | tweedie_p18 | 186.46% | 0.1494 | 100.00% | 4.76% | 9.09% | 1.2322 |
| CatBoost Tweedie (p=1.5) | catboost | tweedie_p15 | 190.03% | 0.1522 | 100.00% | 0.00% | 0.00% | 1.2252 |
| CatBoost Tweedie (p=1.2) | catboost | tweedie_p12 | 198.23% | 0.1588 | 100.00% | 0.00% | 0.00% | 1.2227 |
| CatBoost Tweedie (p=1.1) | catboost | tweedie_p11 | 205.06% | 0.1643 | 100.00% | 0.00% | 0.00% | 1.2142 |
<!-- BENCHMARK_BLOCK:configuration_ablations:END -->

配置表回答的是同一工具包内损失函数与 Tweedie power 的敏感性问题，不用于增加该工具包
在广度榜中的投票权。完整图表位于：

- [扩展历史场景图](../../images/benchmarks/2026-retail-week-expanding-v4/20261106-first-stage-v1/fig-01-retail-week-leaderboard-all-scenarios.png)
- [扩展历史 Tweedie 消融图](../../images/benchmarks/2026-retail-week-expanding-v4/20261106-first-stage-v1/fig-03-gbdt-tweedie-ablation-deepdive.png)

## 5. 训练历史策略消融

<!-- BENCHMARK_BLOCK:history_provenance:START -->
> Comparison ID: `2026-retail-week-expanding-v4--vs--2026-retail-week-24w-ablation-v4`  
> Release ID: `20261106-first-stage-v1`  
> Main source: `2026-retail-week-expanding-v4` / `20261106-first-stage-v1`  
> Ablation source: `2026-retail-week-24w-ablation-v4` / `20261106-first-stage-v1`  
> Comparison release manifest SHA-256: `1072b28aa7d4f29bcd196a427eeaced57a0500e7f11f3c0febbd620b273ffc03`  
> WAPE differences are paired descriptive results, not statistical significance claims.
<!-- BENCHMARK_BLOCK:history_provenance:END -->

<!-- BENCHMARK_BLOCK:history_policy_summary:START -->
| 场景 | 配置数 | 扩展历史胜 | 24 周胜 | 平局 | WAPE 差值中位数 (24w-expanding) | 广度配置 WAPE 差值中位数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| garden | 20 | 18 | 0 | 2 | +8.82 pp | +6.79 pp |
| grocery | 20 | 18 | 0 | 2 | +2.13 pp | +2.12 pp |
| luxury | 20 | 9 | 9 | 2 | +0.00 pp | -6.22 pp |
| wine | 20 | 18 | 0 | 2 | +6.96 pp | +5.03 pp |
<!-- BENCHMARK_BLOCK:history_policy_summary:END -->

正差值表示 24 周 WAPE 更高，因而扩展历史更好；负差值表示 24 周结果更低。该方向只描述
相同模型、场景和评估日期下的配对差异，不跨不同需求尺度计算总冠军。

<!-- BENCHMARK_BLOCK:history_policy_breadth:START -->
## grocery

| 代表模型 | 工具包 | 扩展历史 WAPE | 24 周 WAPE | 差值 | WAPE 胜者 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Prophet | prophet | 4.16% | 7.53% | +3.36 pp | expanding |
| CatBoost Default (MAE) | catboost | 5.05% | 7.45% | +2.39 pp | expanding |
| Random Forest | random_forest | 5.39% | 7.63% | +2.24 pp | expanding |
| LightGBM Default (MSE) | lightgbm | 5.31% | 7.54% | +2.23 pp | expanding |
| Decision Tree | decision_tree | 6.95% | 8.97% | +2.02 pp | expanding |
| XGBoost Default (MSE) | xgboost | 5.01% | 6.83% | +1.82 pp | expanding |
| Moving Average (14d) | moving_average | 12.45% | 12.45% | +0.00 pp | tie |
| Seasonal Naive (7d) | seasonal_naive | 15.28% | 15.28% | +0.00 pp | tie |

## wine

| 代表模型 | 工具包 | 扩展历史 WAPE | 24 周 WAPE | 差值 | WAPE 胜者 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM Default (MSE) | lightgbm | 25.08% | 35.33% | +10.24 pp | expanding |
| Prophet | prophet | 22.78% | 29.91% | +7.13 pp | expanding |
| CatBoost Default (MAE) | catboost | 24.40% | 31.20% | +6.79 pp | expanding |
| Decision Tree | decision_tree | 27.05% | 32.56% | +5.51 pp | expanding |
| XGBoost Default (MSE) | xgboost | 23.94% | 28.49% | +4.55 pp | expanding |
| Random Forest | random_forest | 25.81% | 28.25% | +2.45 pp | expanding |
| Moving Average (14d) | moving_average | 39.99% | 39.99% | +0.00 pp | tie |
| Seasonal Naive (7d) | seasonal_naive | 49.26% | 49.26% | +0.00 pp | tie |

## garden

| 代表模型 | 工具包 | 扩展历史 WAPE | 24 周 WAPE | 差值 | WAPE 胜者 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Default (MAE) | catboost | 8.86% | 20.68% | +11.82 pp | expanding |
| Random Forest | random_forest | 9.33% | 20.44% | +11.11 pp | expanding |
| Decision Tree | decision_tree | 10.75% | 21.45% | +10.70 pp | expanding |
| LightGBM Default (MSE) | lightgbm | 9.17% | 16.55% | +7.38 pp | expanding |
| Prophet | prophet | 7.81% | 14.02% | +6.21 pp | expanding |
| XGBoost Default (MSE) | xgboost | 9.03% | 14.94% | +5.91 pp | expanding |
| Moving Average (14d) | moving_average | 20.02% | 20.02% | +0.00 pp | tie |
| Seasonal Naive (7d) | seasonal_naive | 15.36% | 15.36% | +0.00 pp | tie |

## luxury

| 代表模型 | 工具包 | 扩展历史 WAPE | 24 周 WAPE | 差值 | WAPE 胜者 | 扩展历史 Event F1 | 24 周 Event F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Prophet | prophet | 172.55% | 183.06% | +10.51 pp | expanding | 8.00% | 7.41% |
| CatBoost Default (MAE) | catboost | 100.25% | 104.63% | +4.38 pp | expanding | 9.09% | 0.00% |
| Moving Average (14d) | moving_average | 161.82% | 161.82% | +0.00 pp | tie | 0.00% | 0.00% |
| Seasonal Naive (7d) | seasonal_naive | 182.76% | 182.76% | +0.00 pp | tie | 8.89% | 8.89% |
| Random Forest | random_forest | 225.78% | 213.34% | -12.44 pp | sliding_24w | 23.08% | 0.00% |
| Decision Tree | decision_tree | 238.73% | 205.84% | -32.89 pp | sliding_24w | 11.76% | 9.76% |
| XGBoost Default (MSE) | xgboost | 224.11% | 186.40% | -37.71 pp | sliding_24w | 0.00% | 7.41% |
| LightGBM Default (MSE) | lightgbm | 239.15% | 183.86% | -55.29 pp | sliding_24w | 7.41% | 0.00% |
<!-- BENCHMARK_BLOCK:history_policy_breadth:END -->

[训练历史 WAPE 配对差值图](../../images/benchmarks/comparisons/2026-retail-week-expanding-v4--vs--2026-retail-week-24w-ablation-v4/20261106-first-stage-v1/fig-01-training-history-wape-delta.png)
只展示代表配置。Luxury 面板仍需回到表中的事件 F1 一起解释。

## 6. 局限与后续工作

1. 数据为固定生成协议下的教学合成数据，支持受控机制分析，但不能直接外推到真实企业。
2. 24 周窗口刻意构成短历史消融。它不包含一个完整年度周期，不能作为年度季节模型的
   公平主榜；Prophet 在该设置下也会提示年度季节性所需的历史不足。
3. 第一阶段多数模型只输出点预测，因此 Coverage80、区间宽度和 pinball loss 为不适用；
   后续概率模型必须补齐这些指标。
4. 下一阶段按相同适配器和发布协议逐个迁移 ARIMA、MLP/LSTM/TFT、AutoGluon 与
   TimesFM，每个模型单独审核并通过新 release 继承现有结果，不重写本次快照。
