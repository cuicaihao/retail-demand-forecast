## grocery

| 代表模型 | 工具包 | 扩展历史 WAPE | 104 周 WAPE | 差值 | WAPE 胜者 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TFT | pytorch_forecasting | 4.99% | 5.75% | +0.75 pp | expanding |
| CatBoost Default (MAE) | catboost | 5.05% | 5.37% | +0.32 pp | expanding |
| XGBoost Default (MSE) | xgboost | 5.01% | 5.33% | +0.31 pp | expanding |
| LightGBM Default (MSE) | lightgbm | 5.31% | 5.61% | +0.30 pp | expanding |
| MLP | pytorch_mlp | 4.66% | 4.91% | +0.26 pp | expanding |
| LSTM | pytorch_lstm | 4.44% | 4.62% | +0.18 pp | expanding |
| Prophet | prophet | 4.16% | 4.28% | +0.12 pp | expanding |
| Regression + ARIMA (1,0,1) | statsmodels | 4.17% | 4.26% | +0.09 pp | expanding |
| Random Forest | random_forest | 5.39% | 5.45% | +0.05 pp | expanding |
| Moving Average (14d) | moving_average | 12.45% | 12.45% | +0.00 pp | tie |
| Seasonal Naive (7d) | seasonal_naive | 15.28% | 15.28% | +0.00 pp | tie |
| Decision Tree | decision_tree | 6.95% | 6.42% | -0.53 pp | sliding_104w |

## wine

| 代表模型 | 工具包 | 扩展历史 WAPE | 104 周 WAPE | 差值 | WAPE 胜者 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TFT | pytorch_forecasting | 24.40% | 26.64% | +2.24 pp | expanding |
| XGBoost Default (MSE) | xgboost | 23.94% | 25.38% | +1.44 pp | expanding |
| MLP | pytorch_mlp | 23.23% | 24.66% | +1.43 pp | expanding |
| CatBoost Default (MAE) | catboost | 24.40% | 25.61% | +1.21 pp | expanding |
| LightGBM Default (MSE) | lightgbm | 25.08% | 25.63% | +0.55 pp | expanding |
| LSTM | pytorch_lstm | 23.20% | 23.74% | +0.54 pp | expanding |
| Decision Tree | decision_tree | 27.05% | 27.44% | +0.40 pp | expanding |
| Random Forest | random_forest | 25.81% | 26.18% | +0.37 pp | expanding |
| Regression + ARIMA (1,0,1) | statsmodels | 22.10% | 22.46% | +0.36 pp | expanding |
| Moving Average (14d) | moving_average | 39.99% | 39.99% | +0.00 pp | tie |
| Seasonal Naive (7d) | seasonal_naive | 49.26% | 49.26% | +0.00 pp | tie |
| Prophet | prophet | 22.78% | 22.49% | -0.29 pp | sliding_104w |

## garden

| 代表模型 | 工具包 | 扩展历史 WAPE | 104 周 WAPE | 差值 | WAPE 胜者 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LSTM | pytorch_lstm | 11.43% | 19.34% | +7.91 pp | expanding |
| TFT | pytorch_forecasting | 8.76% | 10.84% | +2.08 pp | expanding |
| CatBoost Default (MAE) | catboost | 8.86% | 9.25% | +0.39 pp | expanding |
| LightGBM Default (MSE) | lightgbm | 9.17% | 9.34% | +0.17 pp | expanding |
| Random Forest | random_forest | 9.33% | 9.48% | +0.15 pp | expanding |
| XGBoost Default (MSE) | xgboost | 9.03% | 9.17% | +0.14 pp | expanding |
| Regression + ARIMA (1,0,1) | statsmodels | 7.80% | 7.85% | +0.05 pp | expanding |
| MLP | pytorch_mlp | 8.82% | 8.86% | +0.04 pp | expanding |
| Prophet | prophet | 7.81% | 7.84% | +0.03 pp | expanding |
| Moving Average (14d) | moving_average | 20.02% | 20.02% | +0.00 pp | tie |
| Seasonal Naive (7d) | seasonal_naive | 15.36% | 15.36% | +0.00 pp | tie |
| Decision Tree | decision_tree | 10.75% | 10.70% | -0.06 pp | sliding_104w |

## luxury

| 代表模型 | 工具包 | 扩展历史 WAPE | 104 周 WAPE | 差值 | WAPE 胜者 | 扩展历史 Event F1 | 104 周 Event F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Decision Tree | decision_tree | 238.73% | 275.31% | +36.58 pp | expanding | 11.76% | 13.56% |
| CatBoost Default (MAE) | catboost | 100.25% | 111.81% | +11.56 pp | expanding | 9.09% | 16.00% |
| XGBoost Default (MSE) | xgboost | 224.11% | 228.91% | +4.80 pp | expanding | 0.00% | 8.33% |
| Moving Average (14d) | moving_average | 161.82% | 161.82% | +0.00 pp | tie | 0.00% | 0.00% |
| Seasonal Naive (7d) | seasonal_naive | 182.76% | 182.76% | +0.00 pp | tie | 8.89% | 8.89% |
| MLP | pytorch_mlp | 202.14% | 198.56% | -3.58 pp | sliding_104w | 8.70% | 8.70% |
| Random Forest | random_forest | 225.78% | 218.24% | -7.54 pp | sliding_104w | 23.08% | 23.08% |
| LSTM | pytorch_lstm | 204.49% | 192.33% | -12.16 pp | sliding_104w | 8.70% | 0.00% |
| LightGBM Default (MSE) | lightgbm | 239.15% | 220.86% | -18.29 pp | sliding_104w | 7.41% | 7.14% |
| Prophet | prophet | 172.55% | 142.28% | -30.26 pp | sliding_104w | 8.00% | 0.00% |
| Regression + ARIMA (1,0,1) | statsmodels | 209.29% | 174.82% | -34.47 pp | sliding_104w | 15.38% | 9.09% |
| TFT | pytorch_forecasting | 195.41% | 150.09% | -45.32 pp | sliding_104w | 0.00% | 9.09% |
