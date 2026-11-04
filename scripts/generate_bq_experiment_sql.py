from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "teaching"


def generate_sql(
    project_id: str = "gcp-cc-ai-ml-llm",
    input_dataset: str = "retail_forecast",
    output_dataset: str = "retail_forecast_output",
) -> str:
    in_prefix = f"`{project_id}.{input_dataset}" if project_id else f"`{input_dataset}"
    out_prefix = (
        f"`{project_id}.{output_dataset}" if project_id else f"`{output_dataset}"
    )

    sql = []
    sql.append(
        "-- =========================================================================================================="
    )
    sql.append("-- TimesFM 3.0 BigQuery Controlled Experiments SQL Suite")
    sql.append(
        "-- 架构核心：Google Cloud BigQuery AI.FORECAST 原生零样本时序大模型 (Zero-shot Foundation Model)"
    )
    sql.append(f"-- Project ID    : {project_id}")
    sql.append(f"-- Input Dataset : {input_dataset} (源数据层，6张基础表)")
    sql.append(
        f"-- Output Dataset: {output_dataset} (实验输出层，4张实验表 + 1张全局汇总表)"
    )
    sql.append(
        "-- =========================================================================================================="
    )
    sql.append("--")
    sql.append("-- 【命名规律与字段元数据对照表】")
    sql.append("--")
    sql.append(
        "-- +-----------+-----------------------+-----------------------------+-----------------------+------------------------------------------+--------+"
    )
    sql.append(
        "-- | Scenario  | Experiment ID         | Model 标识                   | 输出表名               | 输入切分与特征说明                       | 预测行 |"
    )
    sql.append(
        "-- +-----------+-----------------------+-----------------------------+-----------------------+------------------------------------------+--------+"
    )
    sql.append(
        "-- | grocery   | exp1_full_history     | TimesFM (BigQuery)          | exp1_full_history     | 全历史(1068d), 剔除闭店与缺货(标准去偏基准) | 28 行  |"
    )
    sql.append(
        "-- | wine      | exp1_full_history     | TimesFM (BigQuery)          | exp1_full_history     | 全历史(1068d), 营业日(is_open=TRUE)       | 28 行  |"
    )
    sql.append(
        "-- | garden    | exp1_full_history     | TimesFM (BigQuery)          | exp1_full_history     | 全历史(1068d), 营业日(is_open=TRUE)       | 28 行  |"
    )
    sql.append(
        "-- | luxury    | exp1_full_history     | TimesFM (BigQuery)          | exp1_full_history     | 全历史(1068d), 营业日(is_open=TRUE)       | 28 行  |"
    )
    sql.append(
        "-- +-----------+-----------------------+-----------------------------+-----------------------+------------------------------------------+--------+"
    )
    sql.append(
        "-- | grocery   | exp2_raw_censored     | TimesFM (BQ-RawCensored)    | exp2_stockout_censor  | 未清洗全量暴露(含24个缺货截断日+公休闭店0) | 28 行  |"
    )
    sql.append(
        "-- | grocery   | exp2_unbiased_clean   | TimesFM (BigQuery)          | exp2_stockout_censor  | 去偏基准对照(引用exp1_full_history无偏行) | 28 行  |"
    )
    sql.append(
        "-- +-----------+-----------------------+-----------------------------+-----------------------+------------------------------------------+--------+"
    )
    sql.append(
        "-- | grocery   | exp3_L{L}             | TimesFM (BQ-L{L})           | exp3_cold_start       | L∈{365, 90, 56, 28, 14}天短历史窗口      | 140 行 |"
    )
    sql.append(
        "-- | wine      | exp3_L{L}             | TimesFM (BQ-L{L})           | exp3_cold_start       | L∈{365, 90, 56, 28, 14}天短历史窗口      | 140 行 |"
    )
    sql.append(
        "-- | garden    | exp3_L{L}             | TimesFM (BQ-L{L})           | exp3_cold_start       | L∈{365, 90, 56, 28, 14}天短历史窗口      | 140 行 |"
    )
    sql.append(
        "-- +-----------+-----------------------+-----------------------------+-----------------------+------------------------------------------+--------+"
    )
    sql.append(
        "-- | grocery   | exp4_covariates_tsfm3 | TimesFM 3.0 (+Discount)     | exp4_covariates_tsfm3 | 注入未来28天已知折扣特征(discount)       | 28 行  |"
    )
    sql.append(
        "-- | wine      | exp4_covariates_tsfm3 | TimesFM 3.0 (+PromoEvents)  | exp4_covariates_tsfm3 | 注入成箱促销、年底旺季与派对事件协变量   | 28 行  |"
    )
    sql.append(
        "-- | garden    | exp4_covariates_tsfm3 | TimesFM 3.0 (+ClearanceDIY) | exp4_covariates_tsfm3 | 注入长周末DIY与秋季清仓已知协变量        | 28 行  |"
    )
    sql.append(
        "-- +-----------+-----------------------+-----------------------------+-----------------------+------------------------------------------+--------+"
    )
    sql.append(
        "-- | [汇总表]  | (包含上述所有ID)      | (对应模型标识)              | all_timesfm_pred      | 全量 UNION ALL 汇聚表，供本地拉取评测    | 644 行 |"
    )
    sql.append(
        "-- +-----------+-----------------------+-----------------------------+-----------------------+------------------------------------------+--------+"
    )
    sql.append("--")
    sql.append("-- 【命名规律核心规则】")
    sql.append(
        "-- 1. scenario      : 严格限定为 'grocery' | 'wine' | 'garden' | 'luxury' 四种零售场景；"
    )
    sql.append(
        "-- 2. experiment_id : 规范为 'exp1_full_history', 'exp2_raw_censored', 'exp3_L{14|28|56|90|365}', 'exp4_covariates_tsfm3'；"
    )
    sql.append(
        "-- 3. model         : 反映模型类型与输入配置，如 'TimesFM (BigQuery)', 'TimesFM (BQ-L28)', 'TimesFM 3.0 (+Discount)'；"
    )
    sql.append(
        "-- 4. 预测指标与分位数 : forecast_value -> prediction, lower_bound -> q10 (P10), upper_bound -> q90 (P90)；"
    )
    sql.append(
        "-- 5. 依赖关系      : 脚本一次性顺序执行即可，所有表均可独立查询，亦可在汇总表中统一拉取。"
    )
    sql.append(
        "-- ==========================================================================================================\n"
    )

    # 0. 自动创建输出数据集
    sql.append("-- 0. 确保输出数据集存在")
    sql.append("-- 注意：BigQuery Web UI 在执行前会对全脚本进行静态 Dry Run 检查。")
    sql.append(
        "-- 若 dataset 尚未创建，请先选中本行点击『Run selected』，或在终端执行："
    )
    sql.append(
        f"--   bq mk -d --project_id={project_id} --location=US {output_dataset}"
    )
    sql.append(f"CREATE SCHEMA IF NOT EXISTS {out_prefix}`")
    sql.append("OPTIONS (location = 'US');\n")

    # -----------------------------------------------------------------
    # 实验 1: 四场景全历史基准对决 (Full History, H=28, Zero-Shot)
    # -----------------------------------------------------------------
    sql.append(
        "-- =========================================================================================================="
    )
    sql.append("-- 实验 1: 四场景全历史基准对决 (Full-History Benchmark)")
    sql.append(f"-- 输出表: {out_prefix}.exp1_full_history`")
    sql.append(
        "-- 目标  : 在拥有完整3年历史(1068天)条件下，评估 TimesFM 3.0 在四大场景的零样本预测与区间质量"
    )
    sql.append(
        "-- =========================================================================================================="
    )

    exp1_queries = []
    for sc in ["grocery", "wine", "garden", "luxury"]:
        table = f"{in_prefix}.{sc}_scenario_data`"
        where_clause = "WHERE ds <= '2025-12-03' AND is_open = TRUE"
        if sc == "grocery":
            where_clause += " AND stockout = FALSE"

        exp1_queries.append(f"""-- 实验 1 [{sc}]: 全历史基准
SELECT
  '{sc}' AS scenario,
  'exp1_full_history' AS experiment_id,
  'TimesFM (BigQuery)' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM {table}
    {where_clause}
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)""")

    sql.append(
        f"""CREATE OR REPLACE TABLE {out_prefix}.exp1_full_history` AS
"""
        + "\n\nUNION ALL\n\n".join(exp1_queries)
        + ";\n"
    )

    # -----------------------------------------------------------------
    # 实验 2: 缺货删失污染对比 (Grocery Censoring Sensitivity)
    # -----------------------------------------------------------------
    sql.append(
        "-- =========================================================================================================="
    )
    sql.append("-- 实验 2: 缺货删失污染对比 (Grocery Censoring Sensitivity)")
    sql.append(f"-- 输出表: {out_prefix}.exp2_stockout_censoring`")
    sql.append(
        "-- 目标  : 对比未清洗原始暴露数据(含24个缺货截断日)与无偏清洗基准，量化基础模型对删失污染的抗性与偏差"
    )
    sql.append(
        "-- =========================================================================================================="
    )

    exp2_queries = [
        f"""-- Group A: 原始暴露 (Raw Censored，含缺货截断值20及闭店0值)
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
    FROM {in_prefix}.grocery_scenario_data`
    WHERE ds <= '2025-12-03'
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)""",
        f"""-- Group B: 无偏去偏基准 (引用实验1中剔除缺货与闭店日的预测结果作为标准对照)
SELECT
  'grocery' AS scenario,
  'exp2_unbiased_clean' AS experiment_id,
  'TimesFM (BigQuery)' AS model,
  ds,
  prediction,
  q10,
  q90
FROM {out_prefix}.exp1_full_history`
WHERE scenario = 'grocery'""",
    ]

    sql.append(
        f"""CREATE OR REPLACE TABLE {out_prefix}.exp2_stockout_censoring` AS
"""
        + "\n\nUNION ALL\n\n".join(exp2_queries)
        + ";\n"
    )

    # -----------------------------------------------------------------
    # 实验 3: 冷启动历史长度截断扫描 (Lengths: 365, 90, 56, 28, 14)
    # -----------------------------------------------------------------
    sql.append(
        "-- =========================================================================================================="
    )
    sql.append("-- 实验 3: 冷启动历史长度截断扫描 (Cold-Start Context Truncation)")
    sql.append(f"-- 输出表: {out_prefix}.exp3_cold_start`")
    sql.append(
        "-- 目标  : 测量历史上下文长度从 365 天逐步缩短到 14 天时，TimesFM 3.0 的预测误差衰减曲线"
    )
    sql.append(
        "-- =========================================================================================================="
    )

    exp3_queries = []
    for sc in ["grocery", "wine", "garden"]:
        table = f"{in_prefix}.{sc}_scenario_data`"
        for L in [365, 90, 56, 28, 14]:
            where_clause = f"WHERE ds BETWEEN DATE_SUB('2025-12-03', INTERVAL {L} DAY) AND '2025-12-03' AND is_open = TRUE"
            if sc == "grocery":
                where_clause += " AND stockout = FALSE"

            exp3_queries.append(f"""-- {sc} 冷启动窗口 L={L} 天
SELECT
  '{sc}' AS scenario,
  'exp3_L{L}' AS experiment_id,
  'TimesFM (BQ-L{L})' AS model,
  DATE(forecast_timestamp) AS ds,
  forecast_value AS prediction,
  prediction_interval_lower_bound AS q10,
  prediction_interval_upper_bound AS q90
FROM AI.FORECAST(
  (
    SELECT ds, y
    FROM {table}
    {where_clause}
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)""")

    sql.append(
        f"""CREATE OR REPLACE TABLE {out_prefix}.exp3_cold_start` AS
"""
        + "\n\nUNION ALL\n\n".join(exp3_queries)
        + ";\n"
    )

    # -----------------------------------------------------------------
    # 实验 4: TimesFM 3.0 动态未来已知协变量 (Future Covariates) 增强
    # -----------------------------------------------------------------
    sql.append(
        "-- =========================================================================================================="
    )
    sql.append("-- 实验 4: TimesFM 3.0 动态未来已知协变量增强对决 (Future Covariates)")
    sql.append(f"-- 输出表: {out_prefix}.exp4_covariates_tsfm3`")
    sql.append(
        "-- 目标  : 全面测试 TimesFM 3.0 注入未来已知外部特征(促销折扣/成箱/节假日/清仓)相比纯单变量预测的增益"
    )
    sql.append(
        "-- =========================================================================================================="
    )

    exp4_queries = [
        f"""-- 4.1 Grocery: 注入未来 28 天已知 20% 双周促销计划 (discount)
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
    FROM {in_prefix}.grocery_scenario_data`
    WHERE ds <= '2025-12-03' AND is_open = TRUE AND stockout = FALSE
    UNION ALL
    SELECT ds, CAST(NULL AS FLOAT64) AS y, discount
    FROM {in_prefix}.grocery_scenario_data`
    WHERE ds > '2025-12-03'
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)""",
        f"""-- 4.2 Wine: 注入成箱促销(promo_case_buy)、年底旺季(is_december_peak)与派对事件(is_party_event)协变量
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
    FROM {in_prefix}.wine_scenario_data`
    WHERE ds <= '2025-12-03' AND is_open = TRUE
    UNION ALL
    SELECT ds, CAST(NULL AS FLOAT64) AS y, promo_case_buy, is_december_peak, is_party_event
    FROM {in_prefix}.wine_scenario_data`
    WHERE ds > '2025-12-03'
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)""",
        f"""-- 4.3 Garden: 注入长周末DIY(is_long_weekend_diy)与秋季清仓(is_end_of_autumn_clearance)协变量
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
    FROM {in_prefix}.garden_scenario_data`
    WHERE ds <= '2025-12-03' AND is_open = TRUE
    UNION ALL
    SELECT ds, CAST(NULL AS FLOAT64) AS y, is_long_weekend_diy, is_end_of_autumn_clearance
    FROM {in_prefix}.garden_scenario_data`
    WHERE ds > '2025-12-03'
  ),
  data_col => 'y',
  timestamp_col => 'ds',
  model => 'TimesFM 3.0',
  horizon => 28,
  confidence_level => 0.80
)""",
    ]

    sql.append(
        f"""CREATE OR REPLACE TABLE {out_prefix}.exp4_covariates_tsfm3` AS
"""
        + "\n\nUNION ALL\n\n".join(exp4_queries)
        + ";\n"
    )

    # -----------------------------------------------------------------
    # 5. 统一全量聚合表: all_timesfm_predictions
    # -----------------------------------------------------------------
    sql.append(
        "-- =========================================================================================================="
    )
    sql.append("-- 5. 全局全量汇总表: all_timesfm_predictions")
    sql.append(f"-- 输出表: {out_prefix}.all_timesfm_predictions`")
    sql.append(
        "-- 说明  : 将上述 4 张独立实验表的数据一次性合并，方便一键拉取到本地 runs/bq_timesfm_predictions.csv"
    )
    sql.append(
        "-- =========================================================================================================="
    )

    sql.append(f"""CREATE OR REPLACE TABLE {out_prefix}.all_timesfm_predictions` AS
SELECT * FROM {out_prefix}.exp1_full_history`
UNION ALL
SELECT * FROM {out_prefix}.exp2_stockout_censoring` WHERE experiment_id = 'exp2_raw_censored'
UNION ALL
SELECT * FROM {out_prefix}.exp3_cold_start`
UNION ALL
SELECT * FROM {out_prefix}.exp4_covariates_tsfm3`;
""")

    return "\n".join(sql)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BigQuery TimesFM Experiment SQL Suite"
    )
    parser.add_argument(
        "--project-id", default="gcp-cc-ai-ml-llm", help="Google Cloud Project ID"
    )
    parser.add_argument(
        "--input-dataset", default="retail_forecast", help="Input BigQuery Dataset ID"
    )
    parser.add_argument(
        "--output-dataset",
        default="retail_forecast_output",
        help="Output BigQuery Dataset ID",
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "src"
            / "retail_forecast"
            / "sql"
            / "bigquery_timesfm_experiments.sql"
        ),
        help="Target SQL file path",
    )
    args = parser.parse_args()

    content = generate_sql(args.project_id, args.input_dataset, args.output_dataset)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(
        f"Generated BigQuery experiment suite (Project: {args.project_id}) -> {target}"
    )


if __name__ == "__main__":
    main()
