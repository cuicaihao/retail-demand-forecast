# Retail Demand Forecasting

Tutorial notebooks for demand forecasting with synthetic VIC retail data. The data is illustrative and does not represent real businesses.

## Start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --group dev
uv run jupyter lab
```

Run `notebooks/00-generate-teaching-data.ipynb` first, then open the tutorials in `notebooks/`.

## Generated Files

The data-generation notebook writes datasets to `data/teaching/` and figures to `images/`. Generated data, figures, articles, models, and run artifacts are local-only; see [.gitignore](.gitignore).
