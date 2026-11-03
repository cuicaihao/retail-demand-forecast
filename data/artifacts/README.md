# Artifact policy

`published/` contains local reviewed figures, compact metrics, and manifests
needed by the articles. They are not versioned; Git keeps only `.gitkeep`
placeholders for the directory structure.

`generated/` contains reproducible row-level predictions, diagnostics,
unreferenced plots, and model exports. It is also ignored by Git and can be
rebuilt from the notebooks. These files are distinct from ordinary executions:

- notebooks and model code write new results under ignored `runs/<run-id>/`;
- article links resolve only to `data/artifacts/published/`;
- replacing a reference snapshot is an explicit review step;
- reviewed tables and figures do not contain training checkpoints.

Required teaching checkpoints and exploratory models have one local canonical
copy under ignored `models/reference/`; they are not duplicated here.
