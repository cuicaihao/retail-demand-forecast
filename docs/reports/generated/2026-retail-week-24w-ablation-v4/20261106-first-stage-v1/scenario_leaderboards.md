## grocery

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost Default (MSE) | xgboost | default | 6.83% | 8.3569 | 10.7267 | -0.06% | 0.2968 | 0.0673 | 0.0114 |
| CatBoost Default (MAE) | catboost | default | 7.45% | 9.1102 | 11.8385 | -0.60% | 0.3236 | 0.0720 | 0.0341 |
| Prophet | prophet | default | 7.53% | 9.2055 | 11.7958 | 0.74% | 0.3270 | 0.0652 | 0.0433 |
| LightGBM Default (MSE) | lightgbm | default | 7.54% | 9.2288 | 11.8255 | -0.31% | 0.3278 | 0.0700 | 0.0158 |
| Random Forest | random_forest | default | 7.63% | 9.3383 | 11.8461 | -0.33% | 0.3317 | 0.0759 | 0.0367 |
| Decision Tree | decision_tree | default | 8.97% | 10.9708 | 13.8640 | 0.02% | 0.3897 | 0.0950 | 0.0384 |
| Moving Average (14d) | moving_average | default | 12.45% | 15.2312 | 18.4582 | -0.38% | 0.5410 | 0.1240 | 0.0202 |
| Seasonal Naive (7d) | seasonal_naive | default | 15.28% | 18.6957 | 22.9534 | -8.89% | 0.6640 | 0.1484 | 0.0190 |

## wine

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random Forest | random_forest | default | 28.25% | 3.3053 | 4.2807 | 1.73% | 0.5442 | 0.2438 | 0.1007 |
| XGBoost Default (MSE) | xgboost | default | 28.49% | 3.3333 | 4.2421 | 4.53% | 0.5488 | 0.2783 | 0.0920 |
| Prophet | prophet | default | 29.91% | 3.4990 | 4.5396 | 13.03% | 0.5761 | 0.2648 | 0.0934 |
| CatBoost Default (MAE) | catboost | default | 31.20% | 3.6496 | 4.6751 | -1.64% | 0.6009 | 0.2787 | 0.1562 |
| Decision Tree | decision_tree | default | 32.56% | 3.8087 | 4.8873 | 5.07% | 0.6271 | 0.3174 | 0.0806 |
| LightGBM Default (MSE) | lightgbm | default | 35.33% | 4.1326 | 5.2552 | -0.11% | 0.6805 | 0.3206 | 0.1312 |
| Moving Average (14d) | moving_average | default | 39.99% | 4.6780 | 6.1696 | -4.28% | 0.7703 | 0.4073 | 0.1260 |
| Seasonal Naive (7d) | seasonal_naive | default | 49.26% | 5.7624 | 8.0430 | 4.39% | 0.9488 | 0.4893 | 0.1176 |

## garden

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Prophet | prophet | default | 14.02% | 10.0692 | 13.2535 | 8.29% | 1.0918 | 0.1391 | 0.0604 |
| XGBoost Default (MSE) | xgboost | default | 14.94% | 10.7292 | 14.0884 | -0.06% | 1.1634 | 0.1690 | 0.1365 |
| Seasonal Naive (7d) | seasonal_naive | default | 15.36% | 11.0304 | 15.5556 | -2.92% | 1.1960 | 0.1695 | 0.0984 |
| LightGBM Default (MSE) | lightgbm | default | 16.55% | 11.8865 | 15.4879 | 0.01% | 1.2888 | 0.1879 | 0.1403 |
| Moving Average (14d) | moving_average | default | 20.02% | 14.3816 | 18.6762 | -0.62% | 1.5594 | 0.2301 | 0.1179 |
| Random Forest | random_forest | default | 20.44% | 14.6817 | 18.2764 | -0.50% | 1.5919 | 0.2214 | 0.1032 |
| CatBoost Default (MAE) | catboost | default | 20.68% | 14.8530 | 18.3877 | -0.85% | 1.6105 | 0.2365 | 0.0960 |
| Decision Tree | decision_tree | default | 21.45% | 15.4095 | 20.6762 | 1.71% | 1.6708 | 0.2446 | 0.1656 |

## luxury

| 模型 | 工具包 | 配置 | WAPE | MAE | ZeroRecall@0.5 | DemandRecall@0.5 | Event F1@0.5 | Positive-demand MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Default (MAE) | catboost | default | 104.63% | 0.0838 | 100.00% | 0.00% | 0.00% | 1.3781 |
| Moving Average (14d) | moving_average | default | 161.82% | 0.1296 | 100.00% | 0.00% | 0.00% | 1.2891 |
| Seasonal Naive (7d) | seasonal_naive | default | 182.76% | 0.1464 | 93.55% | 9.52% | 8.89% | 1.2857 |
| Prophet | prophet | default | 183.06% | 0.1466 | 98.53% | 4.76% | 7.41% | 1.3106 |
| LightGBM Default (MSE) | lightgbm | default | 183.86% | 0.1473 | 99.41% | 0.00% | 0.00% | 1.2801 |
| XGBoost Default (MSE) | xgboost | default | 186.40% | 0.1493 | 98.53% | 4.76% | 7.41% | 1.2427 |
| Decision Tree | decision_tree | default | 205.84% | 0.1649 | 94.72% | 9.52% | 9.76% | 1.2331 |
| Random Forest | random_forest | default | 213.34% | 0.1709 | 100.00% | 0.00% | 0.00% | 1.2464 |
