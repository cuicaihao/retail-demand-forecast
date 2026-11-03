# Data policy

`teaching/` is the local canonical copy of the synthetic teaching data. All
notebooks and Python modules read from this directory, but its generated CSV
and manifest files are ignored by Git. Only `.gitkeep` preserves the directory.

The teaching scenarios include grocery, wine, garden, and luxury handbag data;
the latter is exported as `luxury_scenario_data.csv`.

Run `notebooks/00-generate-teaching-data.ipynb` from the repository root to
reproduce the files. The notebook refuses to overwrite an existing file when
the generated content differs unless `ALLOW_DATA_OVERWRITE` is explicitly set
to `True` after review.

`generation_metadata.json` records seeds, package versions, the date range, and
CSV hashes. Do not copy these CSVs into tutorial or output directories.

Each run refreshes the four-scenario summary in `scenario_diagnostics.csv` and
its manifest in `generation_metadata.json`; scenario data CSVs retain the
overwrite protection described above.
