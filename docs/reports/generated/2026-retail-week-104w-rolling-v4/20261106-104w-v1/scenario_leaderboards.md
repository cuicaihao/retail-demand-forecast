## grocery

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Regression + ARIMA (1,0,1) | statsmodels | default | 4.26% | 5.2060 | 6.5713 | 0.11% | 0.1841 | 0.0411 | 0.0062 |
| Prophet | prophet | default | 4.28% | 5.2380 | 6.6059 | 0.08% | 0.1852 | 0.0410 | 0.0059 |
| XGBoost Default (MSE) | xgboost | default | 5.33% | 6.5141 | 8.7372 | -1.39% | 0.2304 | 0.0540 | 0.0077 |
| CatBoost Default (MAE) | catboost | default | 5.37% | 6.5745 | 8.9311 | -1.45% | 0.2325 | 0.0529 | 0.0142 |
| Random Forest | random_forest | default | 5.45% | 6.6609 | 8.6556 | -1.71% | 0.2356 | 0.0553 | 0.0076 |
| LightGBM Default (MSE) | lightgbm | default | 5.61% | 6.8615 | 9.2880 | -1.17% | 0.2427 | 0.0573 | 0.0096 |
| Decision Tree | decision_tree | default | 6.42% | 7.8565 | 10.0520 | -1.41% | 0.2779 | 0.0651 | 0.0132 |
| Moving Average (14d) | moving_average | default | 12.45% | 15.2312 | 18.4582 | -0.38% | 0.5387 | 0.1240 | 0.0202 |
| Seasonal Naive (7d) | seasonal_naive | default | 15.28% | 18.6957 | 22.9534 | -8.89% | 0.6612 | 0.1484 | 0.0190 |

## wine

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Regression + ARIMA (1,0,1) | statsmodels | default | 22.46% | 2.6272 | 3.3201 | 3.17% | 0.4241 | 0.2180 | 0.0464 |
| Prophet | prophet | default | 22.49% | 2.6309 | 3.3555 | -1.94% | 0.4247 | 0.2187 | 0.0368 |
| XGBoost Default (MSE) | xgboost | default | 25.38% | 2.9688 | 3.8300 | -1.40% | 0.4792 | 0.2426 | 0.0849 |
| CatBoost Default (MAE) | catboost | default | 25.61% | 2.9966 | 3.8597 | -3.66% | 0.4837 | 0.2365 | 0.0515 |
| LightGBM Default (MSE) | lightgbm | default | 25.63% | 2.9985 | 3.8530 | 0.12% | 0.4840 | 0.2394 | 0.0937 |
| Random Forest | random_forest | default | 26.18% | 3.0623 | 3.9229 | 0.13% | 0.4943 | 0.2686 | 0.0370 |
| Decision Tree | decision_tree | default | 27.44% | 3.2107 | 4.2478 | -1.14% | 0.5183 | 0.2860 | 0.0903 |
| Moving Average (14d) | moving_average | default | 39.99% | 4.6780 | 6.1696 | -4.28% | 0.7551 | 0.4073 | 0.1260 |
| Seasonal Naive (7d) | seasonal_naive | default | 49.26% | 5.7624 | 8.0430 | 4.39% | 0.9301 | 0.4893 | 0.1176 |

## garden

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Prophet | prophet | default | 7.84% | 5.6291 | 7.0439 | 0.04% | 0.6035 | 0.0879 | 0.0619 |
| Regression + ARIMA (1,0,1) | statsmodels | default | 7.85% | 5.6412 | 7.0540 | 0.10% | 0.6048 | 0.0878 | 0.0606 |
| XGBoost Default (MSE) | xgboost | default | 9.17% | 6.5900 | 8.1539 | -0.94% | 0.7065 | 0.0858 | 0.0677 |
| CatBoost Default (MAE) | catboost | default | 9.25% | 6.6410 | 8.2881 | -1.15% | 0.7120 | 0.0924 | 0.0671 |
| LightGBM Default (MSE) | lightgbm | default | 9.34% | 6.7071 | 8.4192 | -1.07% | 0.7191 | 0.0870 | 0.0690 |
| Random Forest | random_forest | default | 9.48% | 6.8084 | 8.3550 | -2.88% | 0.7299 | 0.1035 | 0.0786 |
| Decision Tree | decision_tree | default | 10.70% | 7.6821 | 9.3112 | -2.36% | 0.8236 | 0.1171 | 0.0665 |
| Seasonal Naive (7d) | seasonal_naive | default | 15.36% | 11.0304 | 15.5556 | -2.92% | 1.1825 | 0.1695 | 0.0984 |
| Moving Average (14d) | moving_average | default | 20.02% | 14.3816 | 18.6762 | -0.62% | 1.5418 | 0.2301 | 0.1179 |

## luxury

| 模型 | 工具包 | 配置 | WAPE | MAE | ZeroRecall@0.5 | DemandRecall@0.5 | Event F1@0.5 | Positive-demand MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Default (MAE) | catboost | default | 111.81% | 0.0896 | 99.41% | 9.52% | 16.00% | 1.3118 |
| Prophet | prophet | default | 142.28% | 0.1140 | 100.00% | 0.00% | 0.00% | 1.3230 |
| Moving Average (14d) | moving_average | default | 161.82% | 0.1296 | 100.00% | 0.00% | 0.00% | 1.2891 |
| Regression + ARIMA (1,0,1) | statsmodels | default | 174.82% | 0.1401 | 100.00% | 4.76% | 9.09% | 1.2547 |
| Seasonal Naive (7d) | seasonal_naive | default | 182.76% | 0.1464 | 93.55% | 9.52% | 8.89% | 1.2857 |
| Random Forest | random_forest | default | 218.24% | 0.1748 | 99.41% | 14.29% | 23.08% | 1.1977 |
| LightGBM Default (MSE) | lightgbm | default | 220.86% | 0.1769 | 98.24% | 4.76% | 7.14% | 1.2667 |
| XGBoost Default (MSE) | xgboost | default | 228.91% | 0.1834 | 99.41% | 4.76% | 8.33% | 1.2040 |
| Decision Tree | decision_tree | default | 275.31% | 0.2206 | 90.03% | 19.05% | 13.56% | 1.1394 |
