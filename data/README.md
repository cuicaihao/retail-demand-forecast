# Data policy

`teaching/` is the local canonical copy of the synthetic teaching data. The
tracked `generation_spec.json` freezes dates, seeds, event rules, and data
generating parameters. Generated CSV, schema, diagnostics, and metadata files
remain ignored by Git; reviewed benchmark releases record their SHA-256 hashes.

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

`benchmarks/protocols/` contains immutable, declarative comparison contracts.
Local benchmark work is written below `runs/`; only reviewed lightweight
snapshots are promoted to
`benchmarks/published/<benchmark-id>/releases/<release-id>/`. Release IDs are
immutable; later model additions create a new release rather than rewriting an
old one. A reviewed comparison of two compatible releases is stored separately
under `benchmarks/published/comparisons/<comparison-id>/releases/<release-id>/`;
its manifest pins both source release manifests by SHA-256. Results from
different benchmark IDs must never be aggregated except through such an
explicit, validated comparison. The historical V2 single-cutoff results remain
isolated in `benchmarks/archive_single_cutoff_28d/`.
