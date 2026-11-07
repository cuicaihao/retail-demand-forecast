## grocery

| 消融配置 | 工具包 | 扩展历史 WAPE | 104 周 WAPE | 差值 | WAPE 胜者 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Tweedie (p=1.1) | catboost | 5.00% | 17.03% | +12.02 pp | expanding |
| CatBoost Tweedie (p=1.2) | catboost | 4.94% | 12.19% | +7.25 pp | expanding |
| CatBoost Tweedie (p=1.5) | catboost | 4.97% | 8.78% | +3.80 pp | expanding |
| CatBoost Tweedie (p=1.8) | catboost | 5.15% | 7.48% | +2.33 pp | expanding |
| LightGBM Tweedie (p=1.1) | lightgbm | 5.40% | 5.70% | +0.30 pp | expanding |
| LightGBM Tweedie (p=1.5) | lightgbm | 5.37% | 5.64% | +0.28 pp | expanding |
| LightGBM Tweedie (p=1.2) | lightgbm | 5.55% | 5.75% | +0.21 pp | expanding |
| LightGBM Tweedie (p=1.8) | lightgbm | 5.44% | 5.61% | +0.17 pp | expanding |
| TimesFM 3.0 + Known Covariates | timesfm_3_0 | 4.54% | 4.93% | +0.39 pp | expanding |
| XGBoost Tweedie (p=1.2) | xgboost | 4.87% | 5.28% | +0.41 pp | expanding |
| XGBoost Tweedie (p=1.1) | xgboost | 4.86% | 5.20% | +0.34 pp | expanding |
| XGBoost Tweedie (p=1.8) | xgboost | 5.03% | 5.19% | +0.16 pp | expanding |
| XGBoost Tweedie (p=1.5) | xgboost | 4.92% | 5.04% | +0.12 pp | expanding |

## wine

| 消融配置 | 工具包 | 扩展历史 WAPE | 104 周 WAPE | 差值 | WAPE 胜者 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Tweedie (p=1.8) | catboost | 23.26% | 24.93% | +1.68 pp | expanding |
| CatBoost Tweedie (p=1.1) | catboost | 23.65% | 25.32% | +1.67 pp | expanding |
| CatBoost Tweedie (p=1.5) | catboost | 23.67% | 24.97% | +1.29 pp | expanding |
| CatBoost Tweedie (p=1.2) | catboost | 24.06% | 25.27% | +1.21 pp | expanding |
| LightGBM Tweedie (p=1.1) | lightgbm | 23.96% | 25.68% | +1.71 pp | expanding |
| LightGBM Tweedie (p=1.2) | lightgbm | 24.21% | 25.55% | +1.34 pp | expanding |
| LightGBM Tweedie (p=1.5) | lightgbm | 23.92% | 24.94% | +1.02 pp | expanding |
| LightGBM Tweedie (p=1.8) | lightgbm | 24.48% | 25.16% | +0.68 pp | expanding |
| TimesFM 3.0 + Known Covariates | timesfm_3_0 | 22.69% | 23.98% | +1.30 pp | expanding |
| XGBoost Tweedie (p=1.8) | xgboost | 24.48% | 25.84% | +1.36 pp | expanding |
| XGBoost Tweedie (p=1.5) | xgboost | 24.83% | 25.46% | +0.63 pp | expanding |
| XGBoost Tweedie (p=1.1) | xgboost | 24.63% | 24.98% | +0.36 pp | expanding |
| XGBoost Tweedie (p=1.2) | xgboost | 24.83% | 24.99% | +0.16 pp | expanding |

## garden

| 消融配置 | 工具包 | 扩展历史 WAPE | 104 周 WAPE | 差值 | WAPE 胜者 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Tweedie (p=1.8) | catboost | 8.85% | 9.55% | +0.70 pp | expanding |
| CatBoost Tweedie (p=1.5) | catboost | 8.73% | 8.76% | +0.03 pp | expanding |
| CatBoost Tweedie (p=1.2) | catboost | 8.96% | 8.80% | -0.16 pp | sliding_104w |
| CatBoost Tweedie (p=1.1) | catboost | 9.07% | 8.80% | -0.27 pp | sliding_104w |
| LightGBM Tweedie (p=1.1) | lightgbm | 9.21% | 9.10% | -0.12 pp | sliding_104w |
| LightGBM Tweedie (p=1.8) | lightgbm | 9.32% | 9.09% | -0.23 pp | sliding_104w |
| LightGBM Tweedie (p=1.2) | lightgbm | 9.15% | 8.87% | -0.27 pp | sliding_104w |
| LightGBM Tweedie (p=1.5) | lightgbm | 9.39% | 9.12% | -0.28 pp | sliding_104w |
| TimesFM 3.0 + Known Covariates | timesfm_3_0 | 8.17% | 8.55% | +0.37 pp | expanding |
| XGBoost Tweedie (p=1.8) | xgboost | 8.88% | 9.44% | +0.56 pp | expanding |
| XGBoost Tweedie (p=1.5) | xgboost | 8.94% | 9.35% | +0.41 pp | expanding |
| XGBoost Tweedie (p=1.2) | xgboost | 8.97% | 9.10% | +0.13 pp | expanding |
| XGBoost Tweedie (p=1.1) | xgboost | 9.02% | 8.80% | -0.23 pp | sliding_104w |

## luxury

| 消融配置 | 工具包 | 扩展历史 WAPE | 104 周 WAPE | 差值 | WAPE 胜者 | 扩展历史 Event F1 | 104 周 Event F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CatBoost Tweedie (p=1.8) | catboost | 186.46% | 196.71% | +10.25 pp | expanding | 9.09% | 16.67% |
| CatBoost Tweedie (p=1.2) | catboost | 198.23% | 207.72% | +9.48 pp | expanding | 0.00% | 8.70% |
| CatBoost Tweedie (p=1.1) | catboost | 205.06% | 209.56% | +4.50 pp | expanding | 0.00% | 16.00% |
| CatBoost Tweedie (p=1.5) | catboost | 190.03% | 191.72% | +1.69 pp | expanding | 0.00% | 17.39% |
| LightGBM Tweedie (p=1.8) | lightgbm | 150.80% | 154.47% | +3.67 pp | expanding | 0.00% | 0.00% |
| LightGBM Tweedie (p=1.5) | lightgbm | 161.70% | 160.63% | -1.08 pp | sliding_104w | 0.00% | 0.00% |
| LightGBM Tweedie (p=1.2) | lightgbm | 168.45% | 164.43% | -4.02 pp | sliding_104w | 0.00% | 0.00% |
| LightGBM Tweedie (p=1.1) | lightgbm | 172.96% | 161.84% | -11.12 pp | sliding_104w | 0.00% | 0.00% |
| TimesFM 3.0 + Known Covariates | timesfm_3_0 | 100.41% | 100.47% | +0.06 pp | expanding | 0.00% | 0.00% |
| XGBoost Tweedie (p=1.2) | xgboost | 164.77% | 182.84% | +18.07 pp | expanding | 0.00% | 14.29% |
| XGBoost Tweedie (p=1.5) | xgboost | 128.79% | 146.28% | +17.49 pp | expanding | 0.00% | 7.69% |
| XGBoost Tweedie (p=1.8) | xgboost | 109.29% | 108.42% | -0.87 pp | sliding_104w | 0.00% | 0.00% |
| XGBoost Tweedie (p=1.1) | xgboost | 185.16% | 170.89% | -14.27 pp | sliding_104w | 0.00% | 7.41% |
