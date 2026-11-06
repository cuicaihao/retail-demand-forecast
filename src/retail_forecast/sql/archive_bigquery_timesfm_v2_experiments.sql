-- ==========================================================================================================
-- ARCHIVED V2: TimesFM 3.0 BigQuery single-cutoff controlled experiments
-- 架构核心：Google Cloud BigQuery AI.FORECAST 原生零样本时序大模型 (Zero-shot Foundation Model)
-- Project ID    : gcp-cc-ai-ml-llm
-- Input Dataset : retail_forecast (源数据层，6张基础表)
-- Output Dataset: retail_forecast_output (实验输出层，4张实验表 + 1张全局汇总表)
-- ==========================================================================================================
--
-- 【命名规律与字段元数据对照表】
--
-- +-----------+-----------------------+-----------------------------+-----------------------+------------------------------------------+--------+
-- | Scenario  | Experiment ID         | Model 标识                   | 输出表名               | 输入切分与特征说明                       | 预测行 |
-- +-----------+-----------------------+-----------------------------+-----------------------+------------------------------------------+--------+
-- | grocery   | exp1_full_history     | TimesFM (BigQuery)          | exp1_full_history     | 全历史(1068d), 剔除闭店与缺货(标准去偏基准) | 28 行  |
-- | wine      | exp1_full_history     | TimesFM (BigQuery)          | exp1_full_history     | 全历史(1068d), 营业日(is_open=TRUE)       | 28 行  |
-- | garden    | exp1_full_history     | TimesFM (BigQuery)          | exp1_full_history     | 全历史(1068d), 营业日(is_open=TRUE)       | 28 行  |
-- | luxury    | exp1_full_history     | TimesFM (BigQuery)          | exp1_full_history     | 全历史(1068d), 营业日(is_open=TRUE)       | 28 行  |
-- +-----------+-----------------------+-----------------------------+-----------------------+------------------------------------------+--------+
-- | grocery   | exp2_raw_censored     | TimesFM (BQ-RawCensored)    | exp2_stockout_censor  | 未清洗全量暴露(含24个缺货截断日+公休闭店0) | 28 行  |
-- | grocery   | exp2_unbiased_clean   | TimesFM (BigQuery)          | exp2_stockout_censor  | 去偏基准对照(引用exp1_full_history无偏行) | 28 行  |
-- +-----------+-----------------------+-----------------------------+-----------------------+------------------------------------------+--------+
-- | grocery   | exp3_L{L}             | TimesFM (BQ-L{L})           | exp3_cold_start       | L∈{365, 90, 56, 28, 14}天短历史窗口      | 140 行 |
-- | wine      | exp3_L{L}             | TimesFM (BQ-L{L})           | exp3_cold_start       | L∈{365, 90, 56, 28, 14}天短历史窗口      | 140 行 |
-- | garden    | exp3_L{L}             | TimesFM (BQ-L{L})           | exp3_cold_start       | L∈{365, 90, 56, 28, 14}天短历史窗口      | 140 行 |
-- +-----------+-----------------------+-----------------------------+-----------------------+------------------------------------------+--------+
-- | grocery   | exp4_covariates_tsfm3 | TimesFM 3.0 (+Discount)     | exp4_covariates_tsfm3 | 注入未来28天已知折扣特征(discount)       | 28 行  |
-- | wine      | exp4_covariates_tsfm3 | TimesFM 3.0 (+PromoEvents)  | exp4_covariates_tsfm3 | 注入成箱促销、年底旺季与派对事件协变量   | 28 行  |
-- | garden    | exp4_covariates_tsfm3 | TimesFM 3.0 (+ClearanceDIY) | exp4_covariates_tsfm3 | 注入长周末DIY与秋季清仓已知协变量        | 28 行  |
-- +-----------+-----------------------+-----------------------------+-----------------------+------------------------------------------+--------+
-- | [汇总表]  | (包含上述所有ID)      | (对应模型标识)              | all_timesfm_pred      | 全量 UNION ALL 汇聚表，供本地拉取评测    | 644 行 |
-- +-----------+-----------------------+-----------------------------+-----------------------+------------------------------------------+--------+
--
-- 【命名规律核心规则】
-- 1. scenario      : 严格限定为 'grocery' | 'wine' | 'garden' | 'luxury' 四种零售场景；
-- 2. experiment_id : 规范为 'exp1_full_history', 'exp2_raw_censored', 'exp3_L{14|28|56|90|365}', 'exp4_covariates_tsfm3'；
-- 3. model         : 反映模型类型与输入配置，如 'TimesFM (BigQuery)', 'TimesFM (BQ-L28)', 'TimesFM 3.0 (+Discount)'；
-- 4. 预测指标与分位数 : forecast_value -> prediction, lower_bound -> q10 (P10), upper_bound -> q90 (P90)；
-- 5. 依赖关系      : 脚本一次性顺序执行即可，所有表均可独立查询，亦可在汇总表中统一拉取。
-- ==========================================================================================================

-- 0. 确保输出数据集存在
-- 注意：BigQuery Web UI 在执行前会对全脚本进行静态 Dry Run 检查。
-- 若 dataset 尚未创建，请先选中本行点击『Run selected』，或在终端执行：
--   bq mk -d --project_id=gcp-cc-ai-ml-llm --location=US retail_forecast_output
CREATE SCHEMA IF NOT EXISTS `gcp-cc-ai-ml-llm.retail_forecast_output`
OPTIONS (location = 'US');

-- ==========================================================================================================
-- 实验 1: 四场景全历史基准对决 (Full-History Benchmark)
-- 输出表: `gcp-cc-ai-ml-llm.retail_forecast_output.exp1_full_history`
-- 目标  : 在拥有完整3年历史(1068天)条件下，评估 TimesFM 3.0 在四大场景的零样本预测与区间质量
-- ==========================================================================================================
CREATE OR REPLACE TABLE `gcp-cc-ai-ml-llm.retail_forecast_output.exp1_full_history` AS
-- 实验 1 [grocery]: 全历史基准
SELECT
  'grocery' AS scenario,
  'exp1_full_history' AS experiment_id,
  'TimesFM (BigQuery)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.grocery_scenario_data`
    WHERE ds <= '2025-12-03' AND is_open = TRUE AND stockout = FALSE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- 实验 1 [wine]: 全历史基准
SELECT
  'wine' AS scenario,
  'exp1_full_history' AS experiment_id,
  'TimesFM (BigQuery)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.wine_scenario_data`
    WHERE ds <= '2025-12-03' AND is_open = TRUE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- 实验 1 [garden]: 全历史基准
SELECT
  'garden' AS scenario,
  'exp1_full_history' AS experiment_id,
  'TimesFM (BigQuery)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.garden_scenario_data`
    WHERE ds <= '2025-12-03' AND is_open = TRUE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- 实验 1 [luxury]: 全历史基准
SELECT
  'luxury' AS scenario,
  'exp1_full_history' AS experiment_id,
  'TimesFM (BigQuery)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.luxury_scenario_data`
    WHERE ds <= '2025-12-03' AND is_open = TRUE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
);

-- ==========================================================================================================
-- 实验 2: 缺货删失污染对比 (Grocery Censoring Sensitivity)
-- 输出表: `gcp-cc-ai-ml-llm.retail_forecast_output.exp2_stockout_censoring`
-- 目标  : 对比未清洗原始暴露数据(含24个缺货截断日)与无偏清洗基准，量化基础模型对删失污染的抗性与偏差
-- ==========================================================================================================
CREATE OR REPLACE TABLE `gcp-cc-ai-ml-llm.retail_forecast_output.exp2_stockout_censoring` AS
-- Group A: 原始暴露 (Raw Censored，含缺货截断值20及闭店0值)
SELECT
  'grocery' AS scenario,
  'exp2_raw_censored' AS experiment_id,
  'TimesFM (BQ-RawCensored)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.grocery_scenario_data`
    WHERE ds <= '2025-12-03'
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- Group B: 无偏去偏基准 (引用实验1中剔除缺货与闭店日的预测结果作为标准对照)
SELECT
  'grocery' AS scenario,
  'exp2_unbiased_clean' AS experiment_id,
  'TimesFM (BigQuery)' AS model,
  ds,
  prediction,
  q10,
  q90
FROM `gcp-cc-ai-ml-llm.retail_forecast_output.exp1_full_history`
WHERE scenario = 'grocery';

-- ==========================================================================================================
-- 实验 3: 冷启动历史长度截断扫描 (Cold-Start Context Truncation)
-- 输出表: `gcp-cc-ai-ml-llm.retail_forecast_output.exp3_cold_start`
-- 目标  : 测量历史上下文长度从 365 天逐步缩短到 14 天时，TimesFM 3.0 的预测误差衰减曲线
-- ==========================================================================================================
CREATE OR REPLACE TABLE `gcp-cc-ai-ml-llm.retail_forecast_output.exp3_cold_start` AS
-- grocery 冷启动窗口 L=365 天
SELECT
  'grocery' AS scenario,
  'exp3_L365' AS experiment_id,
  'TimesFM (BQ-L365)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.grocery_scenario_data`
    WHERE ds BETWEEN DATE_SUB('2025-12-03', INTERVAL 365 DAY) AND '2025-12-03' AND is_open = TRUE AND stockout = FALSE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- grocery 冷启动窗口 L=90 天
SELECT
  'grocery' AS scenario,
  'exp3_L90' AS experiment_id,
  'TimesFM (BQ-L90)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.grocery_scenario_data`
    WHERE ds BETWEEN DATE_SUB('2025-12-03', INTERVAL 90 DAY) AND '2025-12-03' AND is_open = TRUE AND stockout = FALSE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- grocery 冷启动窗口 L=56 天
SELECT
  'grocery' AS scenario,
  'exp3_L56' AS experiment_id,
  'TimesFM (BQ-L56)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.grocery_scenario_data`
    WHERE ds BETWEEN DATE_SUB('2025-12-03', INTERVAL 56 DAY) AND '2025-12-03' AND is_open = TRUE AND stockout = FALSE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- grocery 冷启动窗口 L=28 天
SELECT
  'grocery' AS scenario,
  'exp3_L28' AS experiment_id,
  'TimesFM (BQ-L28)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.grocery_scenario_data`
    WHERE ds BETWEEN DATE_SUB('2025-12-03', INTERVAL 28 DAY) AND '2025-12-03' AND is_open = TRUE AND stockout = FALSE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- grocery 冷启动窗口 L=14 天
SELECT
  'grocery' AS scenario,
  'exp3_L14' AS experiment_id,
  'TimesFM (BQ-L14)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.grocery_scenario_data`
    WHERE ds BETWEEN DATE_SUB('2025-12-03', INTERVAL 14 DAY) AND '2025-12-03' AND is_open = TRUE AND stockout = FALSE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- wine 冷启动窗口 L=365 天
SELECT
  'wine' AS scenario,
  'exp3_L365' AS experiment_id,
  'TimesFM (BQ-L365)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.wine_scenario_data`
    WHERE ds BETWEEN DATE_SUB('2025-12-03', INTERVAL 365 DAY) AND '2025-12-03' AND is_open = TRUE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- wine 冷启动窗口 L=90 天
SELECT
  'wine' AS scenario,
  'exp3_L90' AS experiment_id,
  'TimesFM (BQ-L90)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.wine_scenario_data`
    WHERE ds BETWEEN DATE_SUB('2025-12-03', INTERVAL 90 DAY) AND '2025-12-03' AND is_open = TRUE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- wine 冷启动窗口 L=56 天
SELECT
  'wine' AS scenario,
  'exp3_L56' AS experiment_id,
  'TimesFM (BQ-L56)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.wine_scenario_data`
    WHERE ds BETWEEN DATE_SUB('2025-12-03', INTERVAL 56 DAY) AND '2025-12-03' AND is_open = TRUE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- wine 冷启动窗口 L=28 天
SELECT
  'wine' AS scenario,
  'exp3_L28' AS experiment_id,
  'TimesFM (BQ-L28)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.wine_scenario_data`
    WHERE ds BETWEEN DATE_SUB('2025-12-03', INTERVAL 28 DAY) AND '2025-12-03' AND is_open = TRUE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- wine 冷启动窗口 L=14 天
SELECT
  'wine' AS scenario,
  'exp3_L14' AS experiment_id,
  'TimesFM (BQ-L14)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.wine_scenario_data`
    WHERE ds BETWEEN DATE_SUB('2025-12-03', INTERVAL 14 DAY) AND '2025-12-03' AND is_open = TRUE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- garden 冷启动窗口 L=365 天
SELECT
  'garden' AS scenario,
  'exp3_L365' AS experiment_id,
  'TimesFM (BQ-L365)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.garden_scenario_data`
    WHERE ds BETWEEN DATE_SUB('2025-12-03', INTERVAL 365 DAY) AND '2025-12-03' AND is_open = TRUE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- garden 冷启动窗口 L=90 天
SELECT
  'garden' AS scenario,
  'exp3_L90' AS experiment_id,
  'TimesFM (BQ-L90)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.garden_scenario_data`
    WHERE ds BETWEEN DATE_SUB('2025-12-03', INTERVAL 90 DAY) AND '2025-12-03' AND is_open = TRUE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- garden 冷启动窗口 L=56 天
SELECT
  'garden' AS scenario,
  'exp3_L56' AS experiment_id,
  'TimesFM (BQ-L56)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.garden_scenario_data`
    WHERE ds BETWEEN DATE_SUB('2025-12-03', INTERVAL 56 DAY) AND '2025-12-03' AND is_open = TRUE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- garden 冷启动窗口 L=28 天
SELECT
  'garden' AS scenario,
  'exp3_L28' AS experiment_id,
  'TimesFM (BQ-L28)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.garden_scenario_data`
    WHERE ds BETWEEN DATE_SUB('2025-12-03', INTERVAL 28 DAY) AND '2025-12-03' AND is_open = TRUE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- garden 冷启动窗口 L=14 天
SELECT
  'garden' AS scenario,
  'exp3_L14' AS experiment_id,
  'TimesFM (BQ-L14)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM `gcp-cc-ai-ml-llm.retail_forecast.garden_scenario_data`
    WHERE ds BETWEEN DATE_SUB('2025-12-03', INTERVAL 14 DAY) AND '2025-12-03' AND is_open = TRUE
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
);

-- ==========================================================================================================
-- 实验 4: TimesFM 3.0 动态未来已知协变量增强对决 (Future Covariates)
-- 输出表: `gcp-cc-ai-ml-llm.retail_forecast_output.exp4_covariates_tsfm3`
-- 目标  : 全面测试 TimesFM 3.0 注入未来已知外部特征(促销折扣/成箱/节假日/清仓)相比纯单变量预测的增益
-- ==========================================================================================================
CREATE OR REPLACE TABLE `gcp-cc-ai-ml-llm.retail_forecast_output.exp4_covariates_tsfm3` AS
-- 4.1 Grocery: 注入未来 28 天已知 20% 双周促销计划 (discount)
SELECT
  'grocery' AS scenario,
  'exp4_covariates_tsfm3' AS experiment_id,
  'TimesFM 3.0 (+Discount)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y, discount
    FROM `gcp-cc-ai-ml-llm.retail_forecast.grocery_scenario_data`
    WHERE ds <= '2025-12-03' AND is_open = TRUE AND stockout = FALSE
    UNION ALL
    SELECT ds, CAST(NULL AS FLOAT64) AS y, discount
    FROM `gcp-cc-ai-ml-llm.retail_forecast.grocery_scenario_data`
    WHERE ds > '2025-12-03'
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- 4.2 Wine: 注入成箱促销(promo_case_buy)、年底旺季(is_december_peak)与派对事件(is_party_event)协变量
SELECT
  'wine' AS scenario,
  'exp4_covariates_tsfm3' AS experiment_id,
  'TimesFM 3.0 (+PromoEvents)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y, promo_case_buy, is_december_peak, is_party_event
    FROM `gcp-cc-ai-ml-llm.retail_forecast.wine_scenario_data`
    WHERE ds <= '2025-12-03' AND is_open = TRUE
    UNION ALL
    SELECT ds, CAST(NULL AS FLOAT64) AS y, promo_case_buy, is_december_peak, is_party_event
    FROM `gcp-cc-ai-ml-llm.retail_forecast.wine_scenario_data`
    WHERE ds > '2025-12-03'
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)

UNION ALL

-- 4.3 Garden: 注入长周末DIY(is_long_weekend_diy)与秋季清仓(is_end_of_autumn_clearance)协变量
SELECT
  'garden' AS scenario,
  'exp4_covariates_tsfm3' AS experiment_id,
  'TimesFM 3.0 (+ClearanceDIY)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y, is_long_weekend_diy, is_end_of_autumn_clearance
    FROM `gcp-cc-ai-ml-llm.retail_forecast.garden_scenario_data`
    WHERE ds <= '2025-12-03' AND is_open = TRUE
    UNION ALL
    SELECT ds, CAST(NULL AS FLOAT64) AS y, is_long_weekend_diy, is_end_of_autumn_clearance
    FROM `gcp-cc-ai-ml-llm.retail_forecast.garden_scenario_data`
    WHERE ds > '2025-12-03'
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
);

-- ==========================================================================================================
-- 5. 全局全量汇总表: all_timesfm_predictions
-- 输出表: `gcp-cc-ai-ml-llm.retail_forecast_output.all_timesfm_predictions`
-- 说明  : 将上述 4 张独立实验表的数据一次性合并，方便一键拉取到本地 runs/bq_timesfm_predictions.csv
-- ==========================================================================================================
CREATE OR REPLACE TABLE `gcp-cc-ai-ml-llm.retail_forecast_output.all_timesfm_predictions` AS
SELECT * FROM `gcp-cc-ai-ml-llm.retail_forecast_output.exp1_full_history`
UNION ALL
SELECT * FROM `gcp-cc-ai-ml-llm.retail_forecast_output.exp2_stockout_censoring` WHERE experiment_id = 'exp2_raw_censored'
UNION ALL
SELECT * FROM `gcp-cc-ai-ml-llm.retail_forecast_output.exp3_cold_start`
UNION ALL
SELECT * FROM `gcp-cc-ai-ml-llm.retail_forecast_output.exp4_covariates_tsfm3`;
