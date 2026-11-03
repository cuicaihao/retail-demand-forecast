# TimesFM 2.5 / 3.0 受控实验与基准评测模块

本目录包含面向 Google TimesFM 2.5（200M）与 TimesFM 3.0（330M）在四个合成零售场景上的完整受控实验代码与评估工具，严格遵循 `articles/timesfm-experiment-plan.md` 的规范。

## 模块结构

- `run_timesfm_experiments.py`: 主实验执行脚本，一键运行四大受控实验：
  1. **实验 1（全历史基准对决）**：Naive、Prophet、LightGBM、TimesFM 2.5 (Univariate / XReg)、TimesFM 3.0 (Native Covariates) 以及奢侈品 SBA / Tweedie。
  2. **实验 2（缺货删失污染敏感性）**：Raw Censored vs. Imputed Unbiased 输入对比。
  3. **实验 3（冷启动历史长度截断退化曲线）**：测试 14d ~ 1068d 6 个长度梯度。
  4. **实验 4（概率预测区间校准与负值截断分析）**。
- `evaluate_metrics.py`: 统一指标计算工具（WAPE、MAE、RMSE、Bias、80% 区间覆盖率与区间宽度）。

## 运行方式

在仓库根目录下通过本地虚拟环境执行：

```bash
./.venv/bin/python3 tsfm_benchmark/run_timesfm_experiments.py
```

实验生成的指标表将存入 `runs/`，高分辨率出版级图表（`.png` 与 `.webp`）将保存至 `images/tsfm/`。

