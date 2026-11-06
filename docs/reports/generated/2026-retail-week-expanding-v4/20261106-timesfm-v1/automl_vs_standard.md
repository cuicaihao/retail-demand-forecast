| 场景 | 观察到的最佳普通广度模型 | 普通模型 WAPE | AutoML 输出 | AutoML WAPE | 差值 (AutoML-普通) | 普通模型 Event F1 | AutoML Event F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| garden | Regression + ARIMA (1,0,1) | 7.80% | AG Weighted Ensemble | 10.19% | +2.39 pp | 100.00% | 100.00% |
| grocery | Prophet | 4.16% | AG Weighted Ensemble | 5.52% | +1.36 pp | 100.00% | 100.00% |
| luxury | CatBoost Default (MAE) | 100.25% | AG Weighted Ensemble | 100.00% | -0.25 pp | 9.09% | 0.00% |
| wine | Regression + ARIMA (1,0,1) | 22.10% | AG Weighted Ensemble | 26.20% | +4.10 pp | 100.00% | 100.00% |
