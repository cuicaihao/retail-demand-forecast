## grocery / lightgbm

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM Default (MSE) | lightgbm | default | 5.61% | 6.8615 | 9.2880 | -1.17% | 0.2427 | 0.0573 | 0.0096 |
| LightGBM Tweedie (p=1.8) | lightgbm | tweedie_p18 | 5.61% | 6.8636 | 9.2924 | -1.53% | 0.2427 | 0.0589 | 0.0061 |
| LightGBM Tweedie (p=1.5) | lightgbm | tweedie_p15 | 5.64% | 6.9043 | 9.2744 | -1.48% | 0.2442 | 0.0585 | 0.0127 |
| LightGBM Tweedie (p=1.1) | lightgbm | tweedie_p11 | 5.70% | 6.9785 | 9.4253 | -1.84% | 0.2468 | 0.0586 | 0.0110 |
| LightGBM Tweedie (p=1.2) | lightgbm | tweedie_p12 | 5.75% | 7.0376 | 9.4910 | -1.64% | 0.2489 | 0.0596 | 0.0117 |

## grocery / xgboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost Tweedie (p=1.5) | xgboost | tweedie_p15 | 5.04% | 6.1674 | 8.1657 | -1.24% | 0.2181 | 0.0520 | 0.0101 |
| XGBoost Tweedie (p=1.8) | xgboost | tweedie_p18 | 5.19% | 6.3485 | 8.3331 | -1.31% | 0.2245 | 0.0547 | 0.0079 |
| XGBoost Tweedie (p=1.1) | xgboost | tweedie_p11 | 5.20% | 6.3666 | 8.4265 | -1.33% | 0.2252 | 0.0503 | 0.0097 |
| XGBoost Tweedie (p=1.2) | xgboost | tweedie_p12 | 5.28% | 6.4606 | 8.5179 | -1.45% | 0.2285 | 0.0517 | 0.0104 |
| XGBoost Default (MSE) | xgboost | default | 5.33% | 6.5141 | 8.7372 | -1.39% | 0.2304 | 0.0540 | 0.0077 |

## grocery / catboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Default (MAE) | catboost | default | 5.37% | 6.5745 | 8.9311 | -1.45% | 0.2325 | 0.0529 | 0.0142 |
| CatBoost Tweedie (p=1.8) | catboost | tweedie_p18 | 7.48% | 9.1507 | 11.7992 | -5.49% | 0.3236 | 0.0600 | 0.0371 |
| CatBoost Tweedie (p=1.5) | catboost | tweedie_p15 | 8.78% | 10.7354 | 13.8492 | -6.80% | 0.3797 | 0.0697 | 0.0540 |
| CatBoost Tweedie (p=1.2) | catboost | tweedie_p12 | 12.19% | 14.9139 | 20.2017 | -10.30% | 0.5274 | 0.0959 | 0.1003 |
| CatBoost Tweedie (p=1.1) | catboost | tweedie_p11 | 17.03% | 20.8263 | 28.4902 | -15.13% | 0.7365 | 0.1540 | 0.1695 |

## wine / lightgbm

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM Tweedie (p=1.5) | lightgbm | tweedie_p15 | 24.94% | 2.9179 | 3.8097 | -3.13% | 0.4710 | 0.2407 | 0.0751 |
| LightGBM Tweedie (p=1.8) | lightgbm | tweedie_p18 | 25.16% | 2.9437 | 3.9066 | -4.01% | 0.4752 | 0.2502 | 0.0861 |
| LightGBM Tweedie (p=1.2) | lightgbm | tweedie_p12 | 25.55% | 2.9893 | 3.8695 | -1.65% | 0.4825 | 0.2473 | 0.1168 |
| LightGBM Default (MSE) | lightgbm | default | 25.63% | 2.9985 | 3.8530 | 0.12% | 0.4840 | 0.2394 | 0.0937 |
| LightGBM Tweedie (p=1.1) | lightgbm | tweedie_p11 | 25.68% | 3.0039 | 3.8615 | -1.05% | 0.4849 | 0.2481 | 0.1050 |

## wine / xgboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost Tweedie (p=1.1) | xgboost | tweedie_p11 | 24.98% | 2.9230 | 3.7872 | -2.18% | 0.4718 | 0.2408 | 0.0913 |
| XGBoost Tweedie (p=1.2) | xgboost | tweedie_p12 | 24.99% | 2.9238 | 3.7826 | -2.93% | 0.4719 | 0.2410 | 0.0979 |
| XGBoost Default (MSE) | xgboost | default | 25.38% | 2.9688 | 3.8300 | -1.40% | 0.4792 | 0.2426 | 0.0849 |
| XGBoost Tweedie (p=1.5) | xgboost | tweedie_p15 | 25.46% | 2.9788 | 3.8732 | -2.32% | 0.4808 | 0.2440 | 0.1073 |
| XGBoost Tweedie (p=1.8) | xgboost | tweedie_p18 | 25.84% | 3.0227 | 3.9818 | -3.39% | 0.4879 | 0.2430 | 0.1069 |

## wine / catboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Tweedie (p=1.8) | catboost | tweedie_p18 | 24.93% | 2.9170 | 3.8210 | -7.35% | 0.4708 | 0.2325 | 0.0859 |
| CatBoost Tweedie (p=1.5) | catboost | tweedie_p15 | 24.97% | 2.9208 | 3.7833 | -6.47% | 0.4715 | 0.2339 | 0.0705 |
| CatBoost Tweedie (p=1.2) | catboost | tweedie_p12 | 25.27% | 2.9562 | 3.8121 | -5.46% | 0.4772 | 0.2478 | 0.0455 |
| CatBoost Tweedie (p=1.1) | catboost | tweedie_p11 | 25.32% | 2.9622 | 3.8368 | -5.90% | 0.4781 | 0.2394 | 0.0579 |
| CatBoost Default (MAE) | catboost | default | 25.61% | 2.9966 | 3.8597 | -3.66% | 0.4837 | 0.2365 | 0.0515 |

## garden / lightgbm

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM Tweedie (p=1.2) | lightgbm | tweedie_p12 | 8.87% | 6.3728 | 7.9101 | -1.50% | 0.6832 | 0.0832 | 0.0583 |
| LightGBM Tweedie (p=1.8) | lightgbm | tweedie_p18 | 9.09% | 6.5288 | 8.0792 | -2.00% | 0.6999 | 0.0842 | 0.0570 |
| LightGBM Tweedie (p=1.1) | lightgbm | tweedie_p11 | 9.10% | 6.5345 | 8.1179 | -1.57% | 0.7005 | 0.0841 | 0.0558 |
| LightGBM Tweedie (p=1.5) | lightgbm | tweedie_p15 | 9.12% | 6.5473 | 8.0725 | -1.25% | 0.7019 | 0.0858 | 0.0573 |
| LightGBM Default (MSE) | lightgbm | default | 9.34% | 6.7071 | 8.4192 | -1.07% | 0.7191 | 0.0870 | 0.0690 |

## garden / xgboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost Tweedie (p=1.1) | xgboost | tweedie_p11 | 8.80% | 6.3182 | 7.8450 | -1.49% | 0.6774 | 0.0846 | 0.0683 |
| XGBoost Tweedie (p=1.2) | xgboost | tweedie_p12 | 9.10% | 6.5375 | 8.1057 | -1.34% | 0.7009 | 0.0828 | 0.0638 |
| XGBoost Default (MSE) | xgboost | default | 9.17% | 6.5900 | 8.1539 | -0.94% | 0.7065 | 0.0858 | 0.0677 |
| XGBoost Tweedie (p=1.5) | xgboost | tweedie_p15 | 9.35% | 6.7181 | 8.3618 | -1.75% | 0.7202 | 0.0845 | 0.0817 |
| XGBoost Tweedie (p=1.8) | xgboost | tweedie_p18 | 9.44% | 6.7792 | 8.4516 | -1.85% | 0.7268 | 0.0857 | 0.0774 |

## garden / catboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Tweedie (p=1.5) | catboost | tweedie_p15 | 8.76% | 6.2936 | 7.8699 | -3.11% | 0.6747 | 0.0969 | 0.0659 |
| CatBoost Tweedie (p=1.2) | catboost | tweedie_p12 | 8.80% | 6.3225 | 7.9720 | -3.15% | 0.6778 | 0.0887 | 0.0636 |
| CatBoost Tweedie (p=1.1) | catboost | tweedie_p11 | 8.80% | 6.3231 | 8.0288 | -2.60% | 0.6779 | 0.0889 | 0.0700 |
| CatBoost Default (MAE) | catboost | default | 9.25% | 6.6410 | 8.2881 | -1.15% | 0.7120 | 0.0924 | 0.0671 |
| CatBoost Tweedie (p=1.8) | catboost | tweedie_p18 | 9.55% | 6.8585 | 8.5296 | -4.68% | 0.7353 | 0.1080 | 0.0641 |

## luxury / lightgbm

| 模型 | 工具包 | 配置 | WAPE | MAE | ZeroRecall@0.5 | DemandRecall@0.5 | Event F1@0.5 | Positive-demand MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM Tweedie (p=1.8) | lightgbm | tweedie_p18 | 154.47% | 0.1237 | 100.00% | 0.00% | 0.00% | 1.3130 |
| LightGBM Tweedie (p=1.5) | lightgbm | tweedie_p15 | 160.63% | 0.1287 | 100.00% | 0.00% | 0.00% | 1.3072 |
| LightGBM Tweedie (p=1.1) | lightgbm | tweedie_p11 | 161.84% | 0.1297 | 99.71% | 0.00% | 0.00% | 1.3077 |
| LightGBM Tweedie (p=1.2) | lightgbm | tweedie_p12 | 164.43% | 0.1317 | 100.00% | 0.00% | 0.00% | 1.2861 |
| LightGBM Default (MSE) | lightgbm | default | 220.86% | 0.1769 | 98.24% | 4.76% | 7.14% | 1.2667 |

## luxury / xgboost

| 模型 | 工具包 | 配置 | WAPE | MAE | ZeroRecall@0.5 | DemandRecall@0.5 | Event F1@0.5 | Positive-demand MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost Tweedie (p=1.8) | xgboost | tweedie_p18 | 108.42% | 0.0869 | 100.00% | 0.00% | 0.00% | 1.3573 |
| XGBoost Tweedie (p=1.5) | xgboost | tweedie_p15 | 146.28% | 0.1172 | 98.83% | 4.76% | 7.69% | 1.3098 |
| XGBoost Tweedie (p=1.1) | xgboost | tweedie_p11 | 170.89% | 0.1369 | 98.53% | 4.76% | 7.41% | 1.3036 |
| XGBoost Tweedie (p=1.2) | xgboost | tweedie_p12 | 182.84% | 0.1465 | 98.53% | 9.52% | 14.29% | 1.2768 |
| XGBoost Default (MSE) | xgboost | default | 228.91% | 0.1834 | 99.41% | 4.76% | 8.33% | 1.2040 |

## luxury / catboost

| 模型 | 工具包 | 配置 | WAPE | MAE | ZeroRecall@0.5 | DemandRecall@0.5 | Event F1@0.5 | Positive-demand MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Default (MAE) | catboost | default | 111.81% | 0.0896 | 99.41% | 9.52% | 16.00% | 1.3118 |
| CatBoost Tweedie (p=1.5) | catboost | tweedie_p15 | 191.72% | 0.1536 | 100.00% | 9.52% | 17.39% | 1.2154 |
| CatBoost Tweedie (p=1.8) | catboost | tweedie_p18 | 196.71% | 0.1576 | 99.71% | 9.52% | 16.67% | 1.2097 |
| CatBoost Tweedie (p=1.2) | catboost | tweedie_p12 | 207.72% | 0.1664 | 99.71% | 4.76% | 8.70% | 1.2013 |
| CatBoost Tweedie (p=1.1) | catboost | tweedie_p11 | 209.56% | 0.1679 | 99.41% | 9.52% | 16.00% | 1.1947 |
