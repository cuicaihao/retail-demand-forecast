# 交互式实验与教学 Notebook 导航索引

> 本目录存放零售需求预测项目的 Jupyter Notebook（编号 `00` 至 `36`）。根据仓库三阶段生命周期设计，本目录承载**阶段一数据生成入口**、**阶段二单模型深度探索沙盒**，以及**阶段三统一评测辅助对比与历史存档**。

---

## 目录分层与生命周期映射

```mermaid
flowchart TD
    subgraph S1["阶段一：数据生成 (Stage 1)"]
        N00["00-generate-teaching-data.ipynb"]
        D1["data/teaching/ (4大零售挑战场景)"]
        N00 --> D1
    end

    subgraph S2["阶段二：单模型探索与教学实践 (Stage 2)"]
        direction TB
        N10["10/11 Prophet 专项"] <--> A10["docs/articles/prophet-*.md"]
        N31["31 ARIMA 经典统计"] <--> A31["docs/articles/arima-*.md"]
        N32["32a~e 树模型家族"] <--> A32["docs/articles/lightgbm-xgboost-*.md"]
        N33["33a~c 深度时序模型"] <--> A33["docs/articles/neural-*.md"]
        N35["35 AutoGluon AutoML"]
        N36["36 TimesFM 基础模型"] <--> A36["docs/articles/timesfm-*.md"]
    end

    subgraph S3["阶段三：统一评测辅助对比 (Stage 3)"]
        direction TB
        N20["20 五模型初探对照"]
        N32f["32f 树模型横向对照"]
        N33d["33d 神经网络对照"]
        N34["34 全模型滚动评测"]
        R1["docs/reports/ (正式 V4 评测报告)"]
        L1["scripts/run_benchmark.py (CLI 执行器)"]
        N34 -.-> R1
        L1 --> R1
    end
```

---

## 全量 Notebook 索引与成果映射表

| 编号 | Notebook 文件 | 生命周期归属 | 核心目标与技术内容 | 配套技术长文 (Stage 2 Article) | 统一基准报告 (Stage 3 Report) |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **00** | [00-generate-teaching-data.ipynb](00-generate-teaching-data.ipynb) | **阶段一**<br>数据生成 | 确定性生成 5 年墨尔本 4 场景数据、VIC 假日与 BigQuery Schema | — | — |
| **10** | [10-prophet-retail-tutorial.ipynb](10-prophet-retail-tutorial.ipynb) | **阶段二**<br>单模型探索 | Prophet 可加分解、假日与促销回归项、缺货去偏、参数消融实验 | [Prophet 教程](../docs/articles/prophet-retail-demand-forecast.md) | [V4 报告](../docs/reports/2026-retail-week-v4-first-stage-report.md) |
| **11** | [11-prophet-luxury-tutorial.ipynb](11-prophet-luxury-tutorial.ipynb) | **阶段二**<br>单模型探索 | 检验连续平滑的 Prophet 在 94% 极高零值奢侈品场景下的失效边界 | [Prophet 教程](../docs/articles/prophet-retail-demand-forecast.md) | [V4 报告](../docs/reports/2026-retail-week-v4-first-stage-report.md) |
| **20** | [20-five-model-comparison.ipynb](20-five-model-comparison.ipynb) | **阶段三**<br>辅助对比 (历史) | Seasonal Naive、Prophet、SARIMAX、LightGBM 与 TFT 五类模型初步对照 | — | [历史基准归档](../docs/reports/archive/) |
| **31** | [31-arima-tutorial.ipynb](31-arima-tutorial.ipynb) | **阶段二**<br>单模型探索 | auto_arima、动态回归、傅里叶年周期阶数敏感度与差分新息诊断 | [ARIMA 教程](../docs/articles/arima-retail-demand-forecast.md) | [V4 报告](../docs/reports/2026-retail-week-v4-first-stage-report.md) |
| **32a** | [32a-decision-tree-tutorial.ipynb](32a-decision-tree-tutorial.ipynb) | **阶段二**<br>单模型探索 | 决策树单树拟合、最大深度剪枝与时序特征重要性增益分析 | [树模型教程](../docs/articles/lightgbm-xgboost-retail-demand-forecast.md) | [V4 报告](../docs/reports/2026-retail-week-v4-first-stage-report.md) |
| **32b** | [32b-random-forest-tutorial.ipynb](32b-random-forest-tutorial.ipynb) | **阶段二**<br>单模型探索 | 随机森林 Bagging 方差缩减、袋外误差检验与特征扰动评估 | [树模型教程](../docs/articles/lightgbm-xgboost-retail-demand-forecast.md) | [V4 报告](../docs/reports/2026-retail-week-v4-first-stage-report.md) |
| **32c** | [32c-lightgbm-tutorial.ipynb](32c-lightgbm-tutorial.ipynb) | **阶段二**<br>单模型探索 | LightGBM 直方图加速、时序滞后特征、稀疏节假日分裂增益优化 | [树模型教程](../docs/articles/lightgbm-xgboost-retail-demand-forecast.md) | [V4 报告](../docs/reports/2026-retail-week-v4-first-stage-report.md) |
| **32d** | [32d-xgboost-tutorial.ipynb](32d-xgboost-tutorial.ipynb) | **阶段二**<br>单模型探索 | XGBoost 二阶泰勒展开目标、列采样与正则化收缩参数对比 | [树模型教程](../docs/articles/lightgbm-xgboost-retail-demand-forecast.md) | [V4 报告](../docs/reports/2026-retail-week-v4-first-stage-report.md) |
| **32e** | [32e-catboost-tutorial.ipynb](32e-catboost-tutorial.ipynb) | **阶段二**<br>单模型探索 | CatBoost 对称树结构、排序提升抗过拟合与类别特征编码 | [树模型教程](../docs/articles/lightgbm-xgboost-retail-demand-forecast.md) | [V4 报告](../docs/reports/2026-retail-week-v4-first-stage-report.md) |
| **32f** | [32f-tree-model-comparison.ipynb](32f-tree-model-comparison.ipynb) | **阶段三**<br>辅助对比 | Decision Tree、RF、LightGBM、XGBoost、CatBoost 树模型家族系统对照 | [树模型教程](../docs/articles/lightgbm-xgboost-retail-demand-forecast.md) | [V4 报告](../docs/reports/2026-retail-week-v4-first-stage-report.md) |
| **33a** | [33a-mlp-tutorial.ipynb](33a-mlp-tutorial.ipynb) | **阶段二**<br>单模型探索 | 多层感知机特征展平、Dropout、BatchNorm 与全连接层容量实验 | [神经网络教程](../docs/articles/neural-retail-demand-forecast.md) | [V4 报告](../docs/reports/2026-retail-week-v4-first-stage-report.md) |
| **33b** | [33b-lstm-tutorial.ipynb](33b-lstm-tutorial.ipynb) | **阶段二**<br>单模型探索 | 循环神经网络循环门控、长序列依赖捕捉与隐藏状态传递 | [神经网络教程](../docs/articles/neural-retail-demand-forecast.md) | [V4 报告](../docs/reports/2026-retail-week-v4-first-stage-report.md) |
| **33c** | [33c-tft-tutorial.ipynb](33c-tft-tutorial.ipynb) | **阶段二**<br>单模型探索 | Temporal Fusion Transformer 可变选择网络 (VSN) 与多头自注意力机制 | [神经网络教程](../docs/articles/neural-retail-demand-forecast.md) | [V4 报告](../docs/reports/2026-retail-week-v4-first-stage-report.md) |
| **33d** | [33d-neural-model-comparison.ipynb](33d-neural-model-comparison.ipynb) | **阶段三**<br>辅助对比 | MLP、LSTM、TFT 在 4 场景上的拟合收敛性与样本复杂度对比 | [神经网络教程](../docs/articles/neural-retail-demand-forecast.md) | [V4 报告](../docs/reports/2026-retail-week-v4-first-stage-report.md) |
| **34** | [34-all-model-comparison.ipynb](34-all-model-comparison.ipynb) | **阶段三**<br>辅助对比 | 10 种核心模型在统一滚动验证与测试口径下的全量横向对照与可视化分析 | — | [V4 报告](../docs/reports/2026-retail-week-v4-first-stage-report.md) |
| **35** | [35-autogluon-timeseries-tutorial.ipynb](35-autogluon-timeseries-tutorial.ipynb) | **阶段二**<br>单模型探索 | AutoGluon-TimeSeries 自动化模型库、加权集成与多层堆叠实验 | — | [AutoGluon 评测](../docs/reports/archive/2026-10-20-autogluon-comparison-report.md) |
| **36** | [36-timesfm-bigquery-tutorial.ipynb](36-timesfm-bigquery-tutorial.ipynb) | **阶段二**<br>单模型探索 (历史) | 历史 BigQuery 单切片教程；当前 TimesFM 3.0 评测结论以 V4 文章为准 | [TimesFM 3.0 文章](../docs/articles/timesfm-retail-demand-forecast.md) | [TimesFM 验证报告](../docs/reports/archive/2026-11-03-timesfm-validation-report.md) |

---

## 开发者与 Agent 协作原则

1. **Notebook 纯净性规则 (Rule 2)**：
   提交任何 `.ipynb` 文件前，**必须清空所有执行输出与单元格执行计数**（即 `"outputs": []` 且 `"execution_count": null`）。测试套件 `tests/test_repository_layout.py` 会严格执行断言校验。
2. **严防未来信息泄漏 (Rule 4)**：
   在单模型探索 Notebook 中，所有滚动验证的切片必须遵循时间前向因果律；测试期内的未来目标值 `y` 和诊断真实值列（`conditional_mean`, `latent_demand`）在推理时刻必须保持不可见。
3. **文章同步义务 (Rule 6)**：
   若对阶段二单模型 Notebook 进行了数据长度、特征工程或参数更新，请务必同步更新对应的 `docs/articles/` 教程长文中的时间定义、样本统计、指标表格与可视化图表。

