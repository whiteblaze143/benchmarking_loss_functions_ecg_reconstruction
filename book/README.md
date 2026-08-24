# Rendering the ECG reconstruction book

The rendered entry point is [`_book/index.html`](_book/index.html).

All executable chapters are run with the project virtual environment:

```bash
cd /home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
QUARTO_PYTHON=/home/mithunmanivannan/.venv/bin/python \
PATH=/home/mithunmanivannan/.venv/bin:$PATH \
.tools/quarto-1.10.18/bin/quarto render book
```

The explicit `QUARTO_PYTHON` and `PATH` settings matter. Without them, Quarto may start a different `python3` kernel and fail to find the venv's `jupyter-cache`, WFDB, or analysis packages.

The project uses shared site assets (`embed-resources: false`). Embedding every Plotly dependency in every chapter previously produced pages larger than 250 MB and caused full renders to be killed for memory use.

## Release checks

After a render:

1. confirm `_book/index.html` exists;
2. confirm the sidebar contains Chapters 1–11;
3. open Chapters 9–11 and verify their executed tables are present;
4. verify Chapter 8's manifest gate reports 160 jobs and 160 unique masks for each seed;
5. treat any failed Python cell, missing data artifact, or absent declared sidecar as a failed release gate.

The book reads study data from `../data/` and locked results from
`../results/comprehensive_latest_48_models/`. Synthetic tutorial chapters are
explicitly separated from the real-data evidence chapters.
