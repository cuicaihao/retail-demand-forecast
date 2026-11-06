## grocery / catboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Tweedie (p=1.2) | catboost | tweedie_p12 | 4.94% | 6.0435 | 7.7332 | -1.24% | 0.2212 | 0.0461 | 0.0116 |
| CatBoost Tweedie (p=1.5) | catboost | tweedie_p15 | 4.97% | 6.0820 | 7.7716 | -1.61% | 0.2226 | 0.0487 | 0.0148 |
| CatBoost Tweedie (p=1.1) | catboost | tweedie_p11 | 5.00% | 6.1199 | 7.7418 | -1.44% | 0.2240 | 0.0486 | 0.0078 |
| CatBoost Default (MAE) | catboost | default | 5.05% | 6.1832 | 8.2256 | -1.33% | 0.2263 | 0.0512 | 0.0154 |
| CatBoost Tweedie (p=1.8) | catboost | tweedie_p18 | 5.15% | 6.2981 | 8.0769 | -2.21% | 0.2305 | 0.0519 | 0.0161 |

## grocery / lightgbm

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM Default (MSE) | lightgbm | default | 5.31% | 6.5005 | 8.5586 | -1.48% | 0.2379 | 0.0533 | 0.0138 |
| LightGBM Tweedie (p=1.5) | lightgbm | tweedie_p15 | 5.37% | 6.5632 | 8.7291 | -1.95% | 0.2402 | 0.0557 | 0.0137 |
| LightGBM Tweedie (p=1.1) | lightgbm | tweedie_p11 | 5.40% | 6.6071 | 8.7282 | -1.78% | 0.2418 | 0.0548 | 0.0142 |
| LightGBM Tweedie (p=1.8) | lightgbm | tweedie_p18 | 5.44% | 6.6556 | 8.7709 | -1.88% | 0.2436 | 0.0558 | 0.0094 |
| LightGBM Tweedie (p=1.2) | lightgbm | tweedie_p12 | 5.55% | 6.7829 | 8.9827 | -1.96% | 0.2483 | 0.0567 | 0.0161 |

## grocery / timesfm_3_0

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TimesFM 3.0 + Known Covariates | timesfm_3_0 | known_covariates | 4.54% | 5.5501 | 7.1767 | -1.06% | 0.2032 | 0.0448 | 0.0058 |
| TimesFM 3.0 | timesfm_3_0 | univariate | 4.91% | 6.0039 | 8.2953 | -0.65% | 0.2198 | 0.0479 | 0.0106 |

## grocery / xgboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost Tweedie (p=1.1) | xgboost | tweedie_p11 | 4.86% | 5.9501 | 7.5671 | -1.22% | 0.2178 | 0.0513 | 0.0141 |
| XGBoost Tweedie (p=1.2) | xgboost | tweedie_p12 | 4.87% | 5.9590 | 7.5603 | -1.30% | 0.2181 | 0.0510 | 0.0118 |
| XGBoost Tweedie (p=1.5) | xgboost | tweedie_p15 | 4.92% | 6.0206 | 7.7636 | -1.48% | 0.2204 | 0.0499 | 0.0170 |
| XGBoost Default (MSE) | xgboost | default | 5.01% | 6.1295 | 7.8354 | -1.17% | 0.2244 | 0.0485 | 0.0178 |
| XGBoost Tweedie (p=1.8) | xgboost | tweedie_p18 | 5.03% | 6.1537 | 7.8465 | -1.32% | 0.2253 | 0.0514 | 0.0156 |

## wine / catboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Tweedie (p=1.8) | catboost | tweedie_p18 | 23.26% | 2.7208 | 3.4742 | -4.10% | 0.4524 | 0.2271 | 0.0538 |
| CatBoost Tweedie (p=1.1) | catboost | tweedie_p11 | 23.65% | 2.7671 | 3.5196 | -2.94% | 0.4601 | 0.2380 | 0.0288 |
| CatBoost Tweedie (p=1.5) | catboost | tweedie_p15 | 23.67% | 2.7693 | 3.5554 | -3.75% | 0.4605 | 0.2351 | 0.0373 |
| CatBoost Tweedie (p=1.2) | catboost | tweedie_p12 | 24.06% | 2.8150 | 3.5914 | -3.82% | 0.4680 | 0.2397 | 0.0504 |
| CatBoost Default (MAE) | catboost | default | 24.40% | 2.8551 | 3.7194 | -6.04% | 0.4747 | 0.2307 | 0.0692 |

## wine / lightgbm

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM Tweedie (p=1.5) | lightgbm | tweedie_p15 | 23.92% | 2.7989 | 3.6101 | -4.02% | 0.4654 | 0.2329 | 0.0870 |
| LightGBM Tweedie (p=1.1) | lightgbm | tweedie_p11 | 23.96% | 2.8033 | 3.6476 | -3.58% | 0.4661 | 0.2366 | 0.0830 |
| LightGBM Tweedie (p=1.2) | lightgbm | tweedie_p12 | 24.21% | 2.8320 | 3.6899 | -4.43% | 0.4709 | 0.2364 | 0.0869 |
| LightGBM Tweedie (p=1.8) | lightgbm | tweedie_p18 | 24.48% | 2.8639 | 3.6991 | -5.54% | 0.4762 | 0.2332 | 0.0877 |
| LightGBM Default (MSE) | lightgbm | default | 25.08% | 2.9346 | 3.7990 | -4.04% | 0.4879 | 0.2440 | 0.0786 |

## wine / timesfm_3_0

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TimesFM 3.0 + Known Covariates | timesfm_3_0 | known_covariates | 22.69% | 2.6543 | 3.4164 | -3.11% | 0.4413 | 0.2219 | 0.0314 |
| TimesFM 3.0 | timesfm_3_0 | univariate | 24.29% | 2.8419 | 3.6736 | -2.51% | 0.4725 | 0.2320 | 0.0654 |

## wine / xgboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost Default (MSE) | xgboost | default | 23.94% | 2.8004 | 3.5918 | -2.26% | 0.4656 | 0.2278 | 0.0671 |
| XGBoost Tweedie (p=1.8) | xgboost | tweedie_p18 | 24.48% | 2.8639 | 3.7620 | -6.95% | 0.4762 | 0.2402 | 0.0863 |
| XGBoost Tweedie (p=1.1) | xgboost | tweedie_p11 | 24.63% | 2.8810 | 3.7107 | -4.92% | 0.4790 | 0.2476 | 0.0858 |
| XGBoost Tweedie (p=1.5) | xgboost | tweedie_p15 | 24.83% | 2.9049 | 3.8082 | -7.01% | 0.4830 | 0.2443 | 0.0934 |
| XGBoost Tweedie (p=1.2) | xgboost | tweedie_p12 | 24.83% | 2.9050 | 3.7344 | -5.26% | 0.4830 | 0.2451 | 0.0934 |

## garden / catboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Tweedie (p=1.5) | catboost | tweedie_p15 | 8.73% | 6.2690 | 7.8811 | -3.02% | 0.6599 | 0.0878 | 0.0719 |
| CatBoost Tweedie (p=1.8) | catboost | tweedie_p18 | 8.85% | 6.3561 | 7.8965 | -3.21% | 0.6691 | 0.0853 | 0.0693 |
| CatBoost Default (MAE) | catboost | default | 8.86% | 6.3611 | 7.8148 | -1.59% | 0.6696 | 0.0889 | 0.0848 |
| CatBoost Tweedie (p=1.2) | catboost | tweedie_p12 | 8.96% | 6.4367 | 8.1183 | -3.32% | 0.6776 | 0.0873 | 0.0742 |
| CatBoost Tweedie (p=1.1) | catboost | tweedie_p11 | 9.07% | 6.5158 | 8.2189 | -3.24% | 0.6859 | 0.0934 | 0.0687 |

## garden / lightgbm

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM Tweedie (p=1.2) | lightgbm | tweedie_p12 | 9.15% | 6.5691 | 8.2037 | -2.28% | 0.6915 | 0.0896 | 0.0892 |
| LightGBM Default (MSE) | lightgbm | default | 9.17% | 6.5871 | 8.0916 | -1.21% | 0.6934 | 0.0880 | 0.0747 |
| LightGBM Tweedie (p=1.1) | lightgbm | tweedie_p11 | 9.21% | 6.6189 | 8.1860 | -2.52% | 0.6967 | 0.0946 | 0.0638 |
| LightGBM Tweedie (p=1.8) | lightgbm | tweedie_p18 | 9.32% | 6.6937 | 8.3763 | -2.98% | 0.7046 | 0.0855 | 0.0901 |
| LightGBM Tweedie (p=1.5) | lightgbm | tweedie_p15 | 9.39% | 6.7460 | 8.3332 | -2.56% | 0.7101 | 0.0924 | 0.0839 |

## garden / timesfm_3_0

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TimesFM 3.0 + Known Covariates | timesfm_3_0 | known_covariates | 8.17% | 5.8716 | 7.2434 | -0.22% | 0.6181 | 0.0874 | 0.0609 |
| TimesFM 3.0 | timesfm_3_0 | univariate | 8.86% | 6.3649 | 8.4272 | -0.58% | 0.6700 | 0.0869 | 0.0402 |

## garden / xgboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost Tweedie (p=1.8) | xgboost | tweedie_p18 | 8.88% | 6.3793 | 7.9584 | -2.95% | 0.6715 | 0.0874 | 0.0807 |
| XGBoost Tweedie (p=1.5) | xgboost | tweedie_p15 | 8.94% | 6.4216 | 8.0111 | -2.17% | 0.6760 | 0.0814 | 0.0890 |
| XGBoost Tweedie (p=1.2) | xgboost | tweedie_p12 | 8.97% | 6.4425 | 7.9938 | -2.06% | 0.6782 | 0.0842 | 0.0840 |
| XGBoost Tweedie (p=1.1) | xgboost | tweedie_p11 | 9.02% | 6.4817 | 8.0127 | -2.07% | 0.6823 | 0.0888 | 0.0902 |
| XGBoost Default (MSE) | xgboost | default | 9.03% | 6.4864 | 8.0532 | -1.53% | 0.6828 | 0.0880 | 0.0855 |

## luxury / catboost

| 模型 | 工具包 | 配置 | WAPE | MAE | ZeroRecall@0.5 | DemandRecall@0.5 | Event F1@0.5 | Positive-demand MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Default (MAE) | catboost | default | 100.25% | 0.0803 | 100.00% | 4.76% | 9.09% | 1.3355 |
| CatBoost Tweedie (p=1.8) | catboost | tweedie_p18 | 186.46% | 0.1494 | 100.00% | 4.76% | 9.09% | 1.2322 |
| CatBoost Tweedie (p=1.5) | catboost | tweedie_p15 | 190.03% | 0.1522 | 100.00% | 0.00% | 0.00% | 1.2252 |
| CatBoost Tweedie (p=1.2) | catboost | tweedie_p12 | 198.23% | 0.1588 | 100.00% | 0.00% | 0.00% | 1.2227 |
| CatBoost Tweedie (p=1.1) | catboost | tweedie_p11 | 205.06% | 0.1643 | 100.00% | 0.00% | 0.00% | 1.2142 |

## luxury / lightgbm

| 模型 | 工具包 | 配置 | WAPE | MAE | ZeroRecall@0.5 | DemandRecall@0.5 | Event F1@0.5 | Positive-demand MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM Tweedie (p=1.8) | lightgbm | tweedie_p18 | 150.80% | 0.1208 | 100.00% | 0.00% | 0.00% | 1.2985 |
| LightGBM Tweedie (p=1.5) | lightgbm | tweedie_p15 | 161.70% | 0.1295 | 100.00% | 0.00% | 0.00% | 1.3018 |
| LightGBM Tweedie (p=1.2) | lightgbm | tweedie_p12 | 168.45% | 0.1349 | 100.00% | 0.00% | 0.00% | 1.3010 |
| LightGBM Tweedie (p=1.1) | lightgbm | tweedie_p11 | 172.96% | 0.1386 | 100.00% | 0.00% | 0.00% | 1.2950 |
| LightGBM Default (MSE) | lightgbm | default | 239.15% | 0.1916 | 98.53% | 4.76% | 7.41% | 1.2444 |

## luxury / timesfm_3_0

| 模型 | 工具包 | 配置 | WAPE | MAE | ZeroRecall@0.5 | DemandRecall@0.5 | Event F1@0.5 | Positive-demand MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TimesFM 3.0 | timesfm_3_0 | univariate | 100.35% | 0.0804 | 100.00% | 0.00% | 0.00% | 1.3807 |
| TimesFM 3.0 + Known Covariates | timesfm_3_0 | known_covariates | 100.41% | 0.0804 | 100.00% | 0.00% | 0.00% | 1.3800 |

## luxury / xgboost

| 模型 | 工具包 | 配置 | WAPE | MAE | ZeroRecall@0.5 | DemandRecall@0.5 | Event F1@0.5 | Positive-demand MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost Tweedie (p=1.8) | xgboost | tweedie_p18 | 109.29% | 0.0876 | 99.71% | 0.00% | 0.00% | 1.3789 |
| XGBoost Tweedie (p=1.5) | xgboost | tweedie_p15 | 128.79% | 0.1032 | 99.12% | 0.00% | 0.00% | 1.3508 |
| XGBoost Tweedie (p=1.2) | xgboost | tweedie_p12 | 164.77% | 0.1320 | 98.83% | 0.00% | 0.00% | 1.3113 |
| XGBoost Tweedie (p=1.1) | xgboost | tweedie_p11 | 185.16% | 0.1483 | 98.53% | 0.00% | 0.00% | 1.3065 |
| XGBoost Default (MSE) | xgboost | default | 224.11% | 0.1795 | 98.24% | 0.00% | 0.00% | 1.2531 |
