# 零售需求预测模型接入协议（V4）

> 当前主协议：`2026-retail-week-expanding-v4`<br>
> 历史策略消融：`2026-retail-week-24w-ablation-v4`<br>
> 测试范围：2025-12-29 至 2026-12-27，共 52 个完整零售周，不等同于完整日历年。

## 1. 公平比较边界

- 正式比较必须显式指定 `benchmark_id`；不同 benchmark ID 的结果不得聚合。
- 主榜在每个 forecast origin 使用此前全部历史；24 周滑窗只用于训练历史消融。
- 参数选择窗口必须结束于正式评估开始之前。选择完成后生成 `selected_config.json`，正式 fold 不重新选择配置。
- 预测期 `y`、`conditional_mean`、`latent_demand`、`occurrence_probability` 和 `positive_demand` 均不得作为推理特征。
- 日历、开店状态、计划促销及公共假日等已知未来协变量可以使用。

## 2. 模型接口与注册

模型实现 `BaseRetailForecaster.fit()` 和 `predict()`，并通过 `ModelConfig.adapter_factory` 注册。注册信息必须包含：

- `toolkit` 与 `family`；
- `variant` 与 `config_group`；
- `comparison_role="breadth"`（工具包代表配置）或 `"ablation"`（配置消融）；
- 完整、可 JSON 序列化的声明参数。

广度榜每个工具包只允许一个预先声明的代表配置。不同损失函数、Tweedie power 或容量参数进入独立消融表，不能重复占据广度榜名次。

## 3. 运行、聚合与发布

```bash
# 查看协议与 20 个第一阶段模型配置
uv run python scripts/run_benchmark.py --list

# 主榜：扩展历史
uv run python scripts/run_benchmark.py \
  --benchmark 2026-retail-week-expanding-v4 \
  --models seasonal_naive

# 24 周历史消融
uv run python scripts/run_benchmark.py \
  --benchmark 2026-retail-week-24w-ablation-v4 \
  --models seasonal_naive

# 只聚合某个明确的本地结果根目录
uv run python scripts/run_benchmark.py \
  --benchmark 2026-retail-week-expanding-v4 \
  --compare-only \
  --source-root runs/20261105-all-models-benchmark

# 严格比较完全相同的模型/场景/fold 下的历史策略
uv run python scripts/compare_training_histories.py \
  --source-root runs/20261105-all-models-benchmark \
  --main-benchmark 2026-retail-week-expanding-v4 \
  --ablation-benchmark 2026-retail-week-24w-ablation-v4

# 审核后发布不可变轻量快照
uv run python scripts/publish_benchmark.py \
  --benchmark 2026-retail-week-expanding-v4 \
  --release 20261106-first-stage-v1 \
  --source-root runs/20261106-first-stage-validation

# 后续仅审核新模型；旧模型从上一 release 原样继承
uv run python scripts/publish_benchmark.py \
  --benchmark 2026-retail-week-expanding-v4 \
  --release 20261120-expanded-models-v2 \
  --base-release 20261106-first-stage-v1 \
  --models arima \
  --source-root runs/20261120-arima-validation
```

本地拟合产物保存在 `runs/YYYYMMDD-[model]-[type]/`。发布前 Git 工作区必须干净，新增或替换模型 manifest 的 commit、源码哈希和软件版本必须与发布环境一致。发布操作只复制逐日预测、fold 指标、pooled 指标、冻结配置和 manifest；模型权重不进入 Git。每个 release ID 不可变，后续模型通过新的 release ID 发布，并从 `--base-release` 继承已审核模型。release manifest 分别记录每个模型的源码 provenance，因此不同模型可以来自不同的已审核提交。

## 4. V4 结果契约

每个发布模型包含：

- `predictions.csv`：逐 scenario、fold、date 的真实值、预测、forecast origin 和可选分位数；
- `metrics_period.csv`：fold 指标、WAPE 分母、样本数和耗时；
- `metrics_annual.csv`：从全部有效逐日预测重新计算的 pooled 指标；
- `selected_config.json`：正式评估前冻结的完整场景参数；
- `manifest.json`：协议、fold、场景、有效标签数、输入哈希、参数哈希、软件版本、Git 状态和源码树哈希。

聚合器会拒绝缺日期、重复日期、forecast origin 错位、部分场景、输入哈希变化或协议不一致的结果，不再兼容性回退到 V2 `metrics.csv`。

## 5. 指标与报告要求

- 常规场景：pooled MAE、RMSE、WAPE、Bias、MASE；fold 均值/标准差仅描述稳定性。
- Luxury：同时报告 `ZeroRecall@0.5`、`DemandRecall@0.5`、事件 F1 和正需求 MAE，不能只按 WAPE 下结论。
- 概率预测：报告 Coverage80、平均区间宽度和 pinball loss；未提供区间时写作不适用。
- bootstrap 区间是描述性不确定性，不得直接称为统计显著。
- 报告生成器只生成或替换 `BENCHMARK_BLOCK` 标记区域，不覆盖人工分析正文。

## 6. 合入检查清单

- [ ] 适配器和模型注册完整，未来目标扰动不会改变输入或预测；
- [ ] 参数选择截止日期早于正式评估；
- [ ] 四个场景、13 个 fold 和逐日日期全部通过严格验证；
- [ ] 单模型配置与工具包广度角色明确；
- [ ] Notebook 无保存输出；
- [ ] unittest、Ruff、`git diff --check` 和发布快照复算通过。

ARIMA、MLP/LSTM/TFT、AutoGluon 与 TimesFM 的 V2 数字只保留在历史归档；只有迁移到同一 V4 协议后才能进入当前场景榜。
