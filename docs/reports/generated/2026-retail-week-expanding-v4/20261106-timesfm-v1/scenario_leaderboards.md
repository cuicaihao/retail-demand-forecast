## grocery

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Prophet | prophet | default | 4.16% | 5.0931 | 6.4422 | 0.19% | 0.1864 | 0.0414 | 0.0058 |
| Regression + ARIMA (1,0,1) | statsmodels | default | 4.17% | 5.1005 | 6.4488 | 0.19% | 0.1867 | 0.0413 | 0.0062 |
| LSTM | pytorch_lstm | default | 4.44% | 5.4350 | 6.8335 | -0.19% | 0.1989 | 0.0424 | 0.0079 |
| MLP | pytorch_mlp | default | 4.66% | 5.6944 | 7.1760 | 0.81% | 0.2084 | 0.0478 | 0.0088 |
| TimesFM 3.0 | timesfm_3_0 | univariate | 4.91% | 6.0039 | 8.2953 | -0.65% | 0.2198 | 0.0479 | 0.0106 |
| TFT | pytorch_forecasting | default | 4.99% | 6.1063 | 7.8900 | -1.96% | 0.2235 | 0.0506 | 0.0076 |
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
| Regression + ARIMA (1,0,1) | statsmodels | default | 22.10% | 2.5856 | 3.2650 | 2.14% | 0.4299 | 0.2127 | 0.0319 |
| Prophet | prophet | default | 22.78% | 2.6649 | 3.3746 | -1.03% | 0.4431 | 0.2223 | 0.0432 |
| LSTM | pytorch_lstm | default | 23.20% | 2.7146 | 3.4496 | 2.54% | 0.4514 | 0.2244 | 0.0489 |
| MLP | pytorch_mlp | default | 23.23% | 2.7173 | 3.4086 | 3.17% | 0.4518 | 0.2221 | 0.0473 |
| XGBoost Default (MSE) | xgboost | default | 23.94% | 2.8004 | 3.5918 | -2.26% | 0.4656 | 0.2278 | 0.0671 |
| TimesFM 3.0 | timesfm_3_0 | univariate | 24.29% | 2.8419 | 3.6736 | -2.51% | 0.4725 | 0.2320 | 0.0654 |
| TFT | pytorch_forecasting | default | 24.40% | 2.8544 | 3.5873 | 0.49% | 0.4746 | 0.2307 | 0.0588 |
| CatBoost Default (MAE) | catboost | default | 24.40% | 2.8551 | 3.7194 | -6.04% | 0.4747 | 0.2307 | 0.0692 |
| LightGBM Default (MSE) | lightgbm | default | 25.08% | 2.9346 | 3.7990 | -4.04% | 0.4879 | 0.2440 | 0.0786 |
| Random Forest | random_forest | default | 25.81% | 3.0190 | 3.8472 | 0.19% | 0.5020 | 0.2554 | 0.0344 |
| Decision Tree | decision_tree | default | 27.05% | 3.1644 | 4.0257 | -5.54% | 0.5261 | 0.2800 | 0.0726 |
| Moving Average (14d) | moving_average | default | 39.99% | 4.6780 | 6.1696 | -4.28% | 0.7778 | 0.4073 | 0.1260 |
| Seasonal Naive (7d) | seasonal_naive | default | 49.26% | 5.7624 | 8.0430 | 4.39% | 0.9581 | 0.4893 | 0.1176 |

## garden

| 模型 | 工具包 | 配置 | WAPE | MAE | RMSE | Bias | MASE | Fold WAPE median | Fold WAPE IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Regression + ARIMA (1,0,1) | statsmodels | default | 7.80% | 5.6053 | 6.9706 | -0.21% | 0.5901 | 0.0880 | 0.0630 |
| Prophet | prophet | default | 7.81% | 5.6108 | 6.9749 | -0.23% | 0.5906 | 0.0881 | 0.0628 |
| TFT | pytorch_forecasting | default | 8.76% | 6.2915 | 7.9981 | -1.11% | 0.6623 | 0.0975 | 0.0658 |
| MLP | pytorch_mlp | default | 8.82% | 6.3334 | 7.8508 | -0.62% | 0.6667 | 0.0942 | 0.0539 |
| CatBoost Default (MAE) | catboost | default | 8.86% | 6.3611 | 7.8148 | -1.59% | 0.6696 | 0.0889 | 0.0848 |
| TimesFM 3.0 | timesfm_3_0 | univariate | 8.86% | 6.3649 | 8.4272 | -0.58% | 0.6700 | 0.0869 | 0.0402 |
| XGBoost Default (MSE) | xgboost | default | 9.03% | 6.4864 | 8.0532 | -1.53% | 0.6828 | 0.0880 | 0.0855 |
| LightGBM Default (MSE) | lightgbm | default | 9.17% | 6.5871 | 8.0916 | -1.21% | 0.6934 | 0.0880 | 0.0747 |
| Random Forest | random_forest | default | 9.33% | 6.7041 | 8.2132 | -4.16% | 0.7057 | 0.1032 | 0.0773 |
| Decision Tree | decision_tree | default | 10.75% | 7.7236 | 9.4988 | -3.46% | 0.8130 | 0.1214 | 0.0724 |
| LSTM | pytorch_lstm | default | 11.43% | 8.2119 | 11.0806 | 3.44% | 0.8644 | 0.0994 | 0.0663 |
| Seasonal Naive (7d) | seasonal_naive | default | 15.36% | 11.0304 | 15.5556 | -2.92% | 1.1611 | 0.1695 | 0.0984 |
| Moving Average (14d) | moving_average | default | 20.02% | 14.3816 | 18.6762 | -0.62% | 1.5139 | 0.2301 | 0.1179 |

## luxury

| 模型 | 工具包 | 配置 | WAPE | MAE | ZeroRecall@0.5 | DemandRecall@0.5 | Event F1@0.5 | Positive-demand MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Default (MAE) | catboost | default | 100.25% | 0.0803 | 100.00% | 4.76% | 9.09% | 1.3355 |
| TimesFM 3.0 | timesfm_3_0 | univariate | 100.35% | 0.0804 | 100.00% | 0.00% | 0.00% | 1.3807 |
| Moving Average (14d) | moving_average | default | 161.82% | 0.1296 | 100.00% | 0.00% | 0.00% | 1.2891 |
| Prophet | prophet | default | 172.55% | 0.1382 | 99.12% | 4.76% | 8.00% | 1.2954 |
| Seasonal Naive (7d) | seasonal_naive | default | 182.76% | 0.1464 | 93.55% | 9.52% | 8.89% | 1.2857 |
| TFT | pytorch_forecasting | default | 195.41% | 0.1565 | 99.71% | 0.00% | 0.00% | 1.2744 |
| MLP | pytorch_mlp | default | 202.14% | 0.1619 | 99.71% | 4.76% | 8.70% | 1.2361 |
| LSTM | pytorch_lstm | default | 204.49% | 0.1638 | 99.71% | 4.76% | 8.70% | 1.2357 |
| Regression + ARIMA (1,0,1) | statsmodels | default | 209.29% | 0.1677 | 99.12% | 9.52% | 15.38% | 1.2297 |
| XGBoost Default (MSE) | xgboost | default | 224.11% | 0.1795 | 98.24% | 0.00% | 0.00% | 1.2531 |
| Random Forest | random_forest | default | 225.78% | 0.1809 | 99.41% | 14.29% | 23.08% | 1.1967 |
| Decision Tree | decision_tree | default | 238.73% | 0.1912 | 96.77% | 9.52% | 11.76% | 1.2593 |
| LightGBM Default (MSE) | lightgbm | default | 239.15% | 0.1916 | 98.53% | 4.76% | 7.41% | 1.2444 |
