# AI Coding Agent Guidelines for Retail Demand Forecast

> This document defines operational instructions, architectural invariants, and domain rules for AI Assistants (Antigravity, Claude, Cursor, Copilot, etc.) working on this repository.

---

## 1. Project Purpose & Core Value

This repository provides **reproducible retail demand forecasting benchmarks and tutorials** built on synthetic daily sales series from Melbourne, Victoria (VIC), Australia (2023-01-01 to 2025-12-31, 1,096 days).

### The Primary Contribution: Four Retail Challenge Scenarios
The foundational asset of this repository is the **four distinct retail scenario datasets** (`data/teaching/`):
1. **`grocery` (High-frequency FMCG)**: Continuous daily demand (~109.5 units), weekday/weekend cycles, with **24 stockout censored days** (sales artificially capped at 20 units) and mandatory shop closure days (Good Friday, Christmas Day).
2. **`wine` (Poisson Pulse Promotions)**: Low-volume count demand (~10.4 units, $\mathrm{ADI} \approx 1.013$), sharp non-linear promotions (`promo_case_buy`), party events, and December seasonal peaks.
3. **`garden` (Strong Annual Seasonality)**: Multi-year sinusoidal weather seasonality (~64.9 units), weekend DIY peaks, long weekend holidays, and autumn clearances.
4. **`luxury` (Extreme Zero-Inflated Intermittent)**: **93.58% zero sales days** ($\mathrm{ADI} \approx 15.571$), high unit value, VIP events, and weekend skew.

---

## 2. Repository Layout & Structural Invariants

The repository strictly maintains **only 8 top-level directories**:

```text
retail-demand-forecast/
├── data/                       # Ground truth data, schemas, persistent benchmarks & model weights
│   ├── teaching/               # 4 Scenario CSVs, VIC holidays, and BigQuery JSON schemas
│   ├── artifacts/              # Reviewed golden snapshots (published/)
│   ├── benchmarks/             # Versioned protocols, reviewed releases, and historical archive
│   └── models/                 # Pretrained model checkpoints (reference/)
├── docs/                       # Single documentation hub
│   ├── articles/               # Public blog posts (*-retail-demand-forecast.md)
│   ├── reports/                # Internal validation reports & onboarding protocols
│   └── references/             # Academic literature and papers (PDFs)
├── images/                     # Generated charts and publication figures (.png / .webp)
├── notebooks/                  # Interactive tutorial notebooks (00 ~ 36)
├── runs/                       # Transient local experiment outputs (Git-ignored)
├── scripts/                    # Automation, CLI runners, and build tools
├── src/                        # Core Python package (retail_forecast) & SQL assets
└── tests/                      # Unit test suite (Repository layout, data hashes, model protocols)
```

### Critical Rules for AI Agents:
- **Rule 1 (Top-level Cleanliness)**: NEVER create new top-level directories. SQL belongs in `src/retail_forecast/sql/`, reports belong in `docs/reports/`, local benchmark runs belong in `runs/`, and reviewed releases belong in `data/benchmarks/published/<benchmark-id>/releases/<release-id>/`.
- **Rule 2 (Notebook Invariant)**: All Jupyter Notebooks (`notebooks/*.ipynb`) must NOT contain saved execution outputs. Cells must have `"outputs": []` and `"execution_count": null`. The test suite strictly asserts this.
- **Rule 3 (PEP 518/621 Layout)**: `tests/` must remain in the root directory and never be merged into `src/`.
- **Rule 4 (No Data Leakage)**: Evaluation dates are frozen by the selected file in `data/benchmarks/protocols/`. Parameter selection must end before formal evaluation. During inference, future `y` and diagnostic ground-truth columns must be invisible. Future known covariates (`is_open`, planned discounts, holidays) are permitted.
- **Rule 5 (`runs/` Directory Naming Invariant)**: Any experiment execution directory generated under `runs/` must strictly adopt the format `[时间]+[模型]+[type]` (i.e. `YYYYMMDD-[model]-[type]` in lowercase kebab-case, e.g., `20261020-prophet-experiment`, `20261020-all-models-summary`, `20261020-tree-models-validation`, `20261020-autogluon-validation`, `20261103-timesfm-validation`).

---

## 3. Benchmark Results Store & Avoidance of Recomputation

To avoid expensive repeated retraining of deep learning models (TFT, neural nets) and AutoML (AutoGluon):

1. **V4 Standard Contract**: Every local model run persists five lightweight files below `runs/<run-id>/<benchmark-id>/models/<model-id>/`:
   - `predictions.csv`: daily fold predictions and forecast origins;
   - `metrics_period.csv`: fold metrics, WAPE denominators, and sample counts;
   - `metrics_annual.csv`: metrics recomputed from pooled valid predictions;
   - `selected_config.json`: parameters frozen before formal evaluation;
   - `manifest.json`: protocol/fold/scenario/input/config provenance.
2. **Independent Re-computation**:
   Agents and developers can recompute or experiment with a single model or a subset of models without re-running everything:
   ```bash
   # List runnable models
   uv run python scripts/run_benchmark.py --list

   # Run or re-run a specific model independently
   uv run python scripts/run_benchmark.py --benchmark 2026-retail-week-expanding-v4 --models prophet
   uv run python scripts/run_benchmark.py --benchmark 2026-retail-week-expanding-v4 --models lightgbm,xgboost

   # Refresh leaderboard across all registered models (0 re-training, milliseconds)
   uv run python scripts/run_benchmark.py --benchmark 2026-retail-week-expanding-v4 --compare-only --source-root runs/<run-id>
   ```

---

## 4. Agent Workflow: How to Add a New Model (XXX)

When tasked with adding a new forecasting model (e.g. Chronos, MOIRAI, TiDE, PatchTST):

1. **Study Protocol**: Read [`docs/reports/model-onboarding-protocol.md`](docs/reports/model-onboarding-protocol.md).
2. **Implement Adapter**: In `src/retail_forecast/<model_key>.py`, subclass `BaseRetailForecaster` from `retail_forecast.benchmark_registry`.
3. **Add CLI Runner**: Register the runner in `AVAILABLE_RUNNERS` within `scripts/run_benchmark.py`.
4. **Execute Benchmark**: Run `uv run python scripts/run_benchmark.py --benchmark 2026-retail-week-expanding-v4 --models <model_key>` to evaluate on all 4 scenarios.
5. **Create Tutorial Notebook**: Generate `notebooks/3X-<model_key>-tutorial.ipynb` with stripped outputs (`"outputs": []`).
6. **Verify Tests**: Run `.venv/bin/python -m unittest discover -s tests -v` to ensure 100% test pass rate.

