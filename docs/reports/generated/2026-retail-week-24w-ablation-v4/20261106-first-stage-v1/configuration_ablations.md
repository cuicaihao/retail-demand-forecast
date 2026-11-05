## grocery / lightgbm

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM Tweedie (p=1.2) | lightgbm | tweedie_p12 | 7.50% | 9.1744 | 11.9230 | -0.70% | 0.3259 | 0.0725 | 0.0224 |
| LightGBM Tweedie (p=1.1) | lightgbm | tweedie_p11 | 7.51% | 9.1886 | 11.9277 | -0.85% | 0.3264 | 0.0719 | 0.0255 |
| LightGBM Tweedie (p=1.5) | lightgbm | tweedie_p15 | 7.53% | 9.2061 | 12.0361 | -0.89% | 0.3270 | 0.0720 | 0.0251 |
| LightGBM Default (MSE) | lightgbm | default | 7.54% | 9.2288 | 11.8255 | -0.31% | 0.3278 | 0.0700 | 0.0158 |
| LightGBM Tweedie (p=1.8) | lightgbm | tweedie_p18 | 7.59% | 9.2807 | 12.0458 | -0.56% | 0.3296 | 0.0764 | 0.0209 |

## grocery / xgboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost Tweedie (p=1.2) | xgboost | tweedie_p12 | 6.59% | 8.0641 | 10.4421 | -0.51% | 0.2864 | 0.0624 | 0.0253 |
| XGBoost Tweedie (p=1.5) | xgboost | tweedie_p15 | 6.61% | 8.0801 | 10.4662 | -0.50% | 0.2870 | 0.0634 | 0.0223 |
| XGBoost Tweedie (p=1.1) | xgboost | tweedie_p11 | 6.72% | 8.2181 | 10.5769 | -0.33% | 0.2919 | 0.0653 | 0.0247 |
| XGBoost Default (MSE) | xgboost | default | 6.83% | 8.3569 | 10.7267 | -0.06% | 0.2968 | 0.0673 | 0.0114 |
| XGBoost Tweedie (p=1.8) | xgboost | tweedie_p18 | 6.87% | 8.4066 | 10.7559 | -0.13% | 0.2986 | 0.0681 | 0.0269 |

## grocery / catboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Default (MAE) | catboost | default | 7.45% | 9.1102 | 11.8385 | -0.60% | 0.3236 | 0.0720 | 0.0341 |
| CatBoost Tweedie (p=1.1) | catboost | tweedie_p11 | 36.78% | 44.9872 | 52.3323 | -36.52% | 1.5979 | 0.3353 | 0.1959 |
| CatBoost Tweedie (p=1.2) | catboost | tweedie_p12 | 38.25% | 46.7930 | 52.5348 | -38.14% | 1.6620 | 0.3416 | 0.2300 |
| CatBoost Tweedie (p=1.5) | catboost | tweedie_p15 | 39.34% | 48.1250 | 51.8125 | -39.34% | 1.7093 | 0.3715 | 0.1371 |
| CatBoost Tweedie (p=1.8) | catboost | tweedie_p18 | 40.52% | 49.5705 | 52.7971 | -40.52% | 1.7606 | 0.4066 | 0.1206 |

## wine / lightgbm

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM Tweedie (p=1.1) | lightgbm | tweedie_p11 | 33.77% | 3.9509 | 4.9849 | -0.48% | 0.6505 | 0.3143 | 0.1593 |
| LightGBM Tweedie (p=1.8) | lightgbm | tweedie_p18 | 34.63% | 4.0514 | 5.1876 | -3.14% | 0.6671 | 0.3150 | 0.1207 |
| LightGBM Tweedie (p=1.2) | lightgbm | tweedie_p12 | 34.80% | 4.0715 | 5.2041 | -0.68% | 0.6704 | 0.3148 | 0.1584 |
| LightGBM Tweedie (p=1.5) | lightgbm | tweedie_p15 | 34.90% | 4.0824 | 5.2128 | 0.02% | 0.6722 | 0.3241 | 0.1603 |
| LightGBM Default (MSE) | lightgbm | default | 35.33% | 4.1326 | 5.2552 | -0.11% | 0.6805 | 0.3206 | 0.1312 |

## wine / xgboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost Tweedie (p=1.8) | xgboost | tweedie_p18 | 27.90% | 3.2635 | 4.2026 | -2.24% | 0.5374 | 0.2551 | 0.0774 |
| XGBoost Tweedie (p=1.2) | xgboost | tweedie_p12 | 28.29% | 3.3100 | 4.2199 | 1.69% | 0.5450 | 0.2551 | 0.1075 |
| XGBoost Tweedie (p=1.1) | xgboost | tweedie_p11 | 28.46% | 3.3296 | 4.2484 | 1.07% | 0.5482 | 0.2722 | 0.1164 |
| XGBoost Tweedie (p=1.5) | xgboost | tweedie_p15 | 28.48% | 3.3319 | 4.2427 | -0.03% | 0.5486 | 0.2654 | 0.1018 |
| XGBoost Default (MSE) | xgboost | default | 28.49% | 3.3333 | 4.2421 | 4.53% | 0.5488 | 0.2783 | 0.0920 |

## wine / catboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Default (MAE) | catboost | default | 31.20% | 3.6496 | 4.6751 | -1.64% | 0.6009 | 0.2787 | 0.1562 |
| CatBoost Tweedie (p=1.2) | catboost | tweedie_p12 | 42.99% | 5.0291 | 6.7988 | -35.83% | 0.8281 | 0.4243 | 0.0573 |
| CatBoost Tweedie (p=1.5) | catboost | tweedie_p15 | 43.19% | 5.0522 | 6.8543 | -36.12% | 0.8319 | 0.4391 | 0.0701 |
| CatBoost Tweedie (p=1.1) | catboost | tweedie_p11 | 43.40% | 5.0767 | 6.8994 | -36.95% | 0.8359 | 0.4317 | 0.0585 |
| CatBoost Tweedie (p=1.8) | catboost | tweedie_p18 | 43.94% | 5.1400 | 6.8687 | -37.60% | 0.8463 | 0.4380 | 0.0706 |

## garden / lightgbm

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM Default (MSE) | lightgbm | default | 16.55% | 11.8865 | 15.4879 | 0.01% | 1.2888 | 0.1879 | 0.1403 |
| LightGBM Tweedie (p=1.1) | lightgbm | tweedie_p11 | 17.54% | 12.5977 | 16.4222 | -2.27% | 1.3660 | 0.1947 | 0.1092 |
| LightGBM Tweedie (p=1.2) | lightgbm | tweedie_p12 | 18.60% | 13.3578 | 17.4727 | -3.30% | 1.4484 | 0.2157 | 0.1096 |
| LightGBM Tweedie (p=1.5) | lightgbm | tweedie_p15 | 18.72% | 13.4437 | 17.4878 | -3.27% | 1.4577 | 0.2069 | 0.1415 |
| LightGBM Tweedie (p=1.8) | lightgbm | tweedie_p18 | 19.25% | 13.8247 | 18.0373 | -3.70% | 1.4990 | 0.2030 | 0.1559 |

## garden / xgboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost Default (MSE) | xgboost | default | 14.94% | 10.7292 | 14.0884 | -0.06% | 1.1634 | 0.1690 | 0.1365 |
| XGBoost Tweedie (p=1.5) | xgboost | tweedie_p15 | 15.88% | 11.4071 | 15.0821 | -2.99% | 1.2369 | 0.1825 | 0.0724 |
| XGBoost Tweedie (p=1.1) | xgboost | tweedie_p11 | 15.99% | 11.4874 | 15.1214 | -3.05% | 1.2456 | 0.1826 | 0.0704 |
| XGBoost Tweedie (p=1.2) | xgboost | tweedie_p12 | 16.10% | 11.5661 | 15.1775 | -2.88% | 1.2541 | 0.1812 | 0.0809 |
| XGBoost Tweedie (p=1.8) | xgboost | tweedie_p18 | 16.68% | 11.9803 | 15.4816 | -3.84% | 1.2990 | 0.1870 | 0.0697 |

## garden / catboost

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Default (MAE) | catboost | default | 20.68% | 14.8530 | 18.3877 | -0.85% | 1.6105 | 0.2365 | 0.0960 |
| CatBoost Tweedie (p=1.8) | catboost | tweedie_p18 | 37.09% | 26.6421 | 33.1446 | -35.93% | 2.8888 | 0.3207 | 0.2208 |
| CatBoost Tweedie (p=1.5) | catboost | tweedie_p15 | 37.41% | 26.8739 | 33.4243 | -36.38% | 2.9139 | 0.3521 | 0.2414 |
| CatBoost Tweedie (p=1.1) | catboost | tweedie_p11 | 37.44% | 26.8914 | 34.3279 | -35.22% | 2.9158 | 0.3445 | 0.2610 |
| CatBoost Tweedie (p=1.2) | catboost | tweedie_p12 | 37.73% | 27.0998 | 34.1375 | -36.52% | 2.9384 | 0.3349 | 0.2620 |

## luxury / lightgbm

| 模型 | 工具包 | 配置 | WAPE | MAE | ZeroRecall@0.5 | DemandRecall@0.5 | Event F1@0.5 | Positive-demand MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM Tweedie (p=1.8) | lightgbm | tweedie_p18 | 151.93% | 0.1217 | 100.00% | 0.00% | 0.00% | 1.3151 |
| LightGBM Tweedie (p=1.5) | lightgbm | tweedie_p15 | 157.51% | 0.1262 | 99.71% | 0.00% | 0.00% | 1.2857 |
| LightGBM Tweedie (p=1.2) | lightgbm | tweedie_p12 | 165.79% | 0.1328 | 100.00% | 0.00% | 0.00% | 1.2845 |
| LightGBM Tweedie (p=1.1) | lightgbm | tweedie_p11 | 170.28% | 0.1364 | 99.12% | 0.00% | 0.00% | 1.2902 |
| LightGBM Default (MSE) | lightgbm | default | 183.86% | 0.1473 | 99.41% | 0.00% | 0.00% | 1.2801 |

## luxury / xgboost

| 模型 | 工具包 | 配置 | WAPE | MAE | ZeroRecall@0.5 | DemandRecall@0.5 | Event F1@0.5 | Positive-demand MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost Tweedie (p=1.8) | xgboost | tweedie_p18 | 115.19% | 0.0923 | 99.71% | 0.00% | 0.00% | 1.3729 |
| XGBoost Tweedie (p=1.5) | xgboost | tweedie_p15 | 139.87% | 0.1121 | 99.41% | 0.00% | 0.00% | 1.3340 |
| XGBoost Tweedie (p=1.2) | xgboost | tweedie_p12 | 166.30% | 0.1332 | 99.41% | 4.76% | 8.33% | 1.2735 |
| XGBoost Tweedie (p=1.1) | xgboost | tweedie_p11 | 175.13% | 0.1403 | 98.83% | 4.76% | 7.69% | 1.2767 |
| XGBoost Default (MSE) | xgboost | default | 186.40% | 0.1493 | 98.53% | 4.76% | 7.41% | 1.2427 |

## luxury / catboost

| 模型 | 工具包 | 配置 | WAPE | MAE | ZeroRecall@0.5 | DemandRecall@0.5 | Event F1@0.5 | Positive-demand MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Default (MAE) | catboost | default | 104.63% | 0.0838 | 100.00% | 0.00% | 0.00% | 1.3781 |
| CatBoost Tweedie (p=1.8) | catboost | tweedie_p18 | 170.06% | 0.1362 | 100.00% | 0.00% | 0.00% | 1.2769 |
| CatBoost Tweedie (p=1.5) | catboost | tweedie_p15 | 216.28% | 0.1733 | 99.41% | 0.00% | 0.00% | 1.2267 |
| CatBoost Tweedie (p=1.2) | catboost | tweedie_p12 | 264.41% | 0.2118 | 98.24% | 4.76% | 7.14% | 1.1853 |
| CatBoost Tweedie (p=1.1) | catboost | tweedie_p11 | 278.35% | 0.2230 | 98.24% | 9.52% | 13.79% | 1.1659 |
