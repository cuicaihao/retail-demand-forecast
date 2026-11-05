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
