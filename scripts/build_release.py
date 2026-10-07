from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "prophet": [
        "README.md",
        "notebooks/00-generate-teaching-data.ipynb",
        "notebooks/10-prophet-retail-tutorial.ipynb",
        "articles/prophet-retail-demand-forecast.md",
        "data/teaching",
        "artifacts/published/prophet",
        "pyproject.toml",
    ],
    "five-models": [
        "README.md",
        "notebooks/00-generate-teaching-data.ipynb",
        "notebooks/20-five-model-comparison.ipynb",
        "data/teaching",
        "artifacts/published/five-model-comparison",
        "pyproject.toml",
    ],
    "model-series": [
        "README.md",
        "notebooks/31-arima-tutorial.ipynb",
        "notebooks/32-tree-models-tutorial.ipynb",
        "notebooks/33-neural-models-tutorial.ipynb",
        "articles/arima-retail-demand-forecast.md",
        "articles/lightgbm-xgboost-retail-demand-forecast.md",
        "articles/neural-retail-demand-forecast.md",
        "src/retail_forecast",
        "data/teaching",
        "artifacts/published/model-series",
        "models/reference/tft",
        "pyproject.toml",
    ],
}


def files_for(entries: list[str]):
    for entry in entries:
        path = ROOT / entry
        if path.is_dir():
            yield from (
                p for p in sorted(path.rglob("*")) if p.is_file() and p.name != ".gitkeep"
            )
        elif path.is_file():
            yield path
        elif entry.startswith("models/"):
            continue
        else:
            raise FileNotFoundError(path)


def validate_local_assets(package: str) -> None:
    data_dir = ROOT / "data" / "teaching"
    required_data = [
        data_dir / "grocery_scenario_data.csv",
        data_dir / "wine_scenario_data.csv",
        data_dir / "garden_scenario_data.csv",
        data_dir / "vic_holidays.csv",
    ]
    published = ROOT / "artifacts" / "published" / {
        "prophet": "prophet",
        "five-models": "five-model-comparison",
        "model-series": "model-series",
    }[package]
    missing = [str(path) for path in required_data if not path.is_file()]
    if not published.is_dir() or not any(
        path.is_file() and path.name != ".gitkeep" for path in published.rglob("*")
    ):
        missing.append(str(published) + " (no local published artifacts)")
    if missing:
        raise SystemExit(
            "Generate or restore the ignored local assets before building:\n- "
            + "\n- ".join(missing)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tutorial ZIP from canonical sources.")
    parser.add_argument("package", choices=sorted(PACKAGES))
    args = parser.parse_args()
    validate_local_assets(args.package)
    target = ROOT / "dist" / f"retail-demand-forecast-{args.package}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    seen: set[Path] = set()
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in files_for(PACKAGES[args.package]):
            relative = path.relative_to(ROOT)
            if relative not in seen:
                archive.write(path, relative)
                seen.add(relative)
    print(target)


if __name__ == "__main__":
    main()
