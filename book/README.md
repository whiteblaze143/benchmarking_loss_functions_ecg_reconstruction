# Rendering the ECG reconstruction book

The rendered entry point is [`_book/index.html`](_book/index.html).

The first section, **Live Experiment Observatory**, is rebuilt directly from
the current SQLite databases. It separates the three-lead ECG-AIM benchmark
from the one-lead spatial/wavelet/SSL program and provides interactive raw
leaderboards, matched deltas, uncertainty tables, boundary heatmaps, and
performance-profile UMAPs.

All executable chapters are run with the project virtual environment:

```bash
cd /home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
QUARTO_PYTHON=/home/mithunmanivannan/.venv/bin/python \
PATH=/home/mithunmanivannan/.venv/bin:$PATH \
.tools/quarto-1.10.18/bin/quarto render book
```

The explicit `QUARTO_PYTHON` and `PATH` settings matter. Without them, Quarto may start a different `python3` kernel and fail to find the venv's `jupyter-cache`, WFDB, or analysis packages.

The project uses shared site assets (`embed-resources: false`). Embedding every Plotly dependency in every chapter previously produced pages larger than 250 MB and caused full renders to be killed for memory use.

### Fast live refresh

To refresh only the pages backed by actively changing databases:

```bash
for chapter in \
  book/15_live_results_observatory.qmd \
  book/16_three_lead_ecgaim_live.qmd \
  book/17_one_lead_wavelet_ssl_live.qmd
do
  QUARTO_PYTHON=/home/mithunmanivannan/.venv/bin/python3 \
    PATH=/home/mithunmanivannan/.venv/bin:$PATH \
    .tools/quarto-1.10.18/bin/quarto render "$chapter"
done
```

The readers use SQLite URI `mode=ro`; rendering cannot modify an experiment
database. No CSV export is required.

The longer checkpoint-representation study runs independently in tmux and can
be monitored with:

```bash
tmux capture-pane -p -t checkpoint_embedding_rdb
```

Its compact live database is `results/checkpoint_embeddings/compact.sqlite`.
Create `results/checkpoint_embeddings/STOP` for a graceful batch-boundary stop;
remove the sentinel and rerun the identical command to resume verified rows.

### GitHub Pages

The Pages workflow deploys the checked-in `book/_book` snapshot. This is
intentional: authoritative experiment databases and checkpoints stay ignored
and are not uploaded to GitHub. Refresh locally, inspect the generated pages,
then commit the source QMD plus the reviewed `_book` snapshot. In repository
Settings → Pages, select **GitHub Actions** as the source once.

## Release checks

After a render:

1. confirm `_book/index.html` exists;
2. confirm the sidebar begins with the three Live Experiment Observatory pages;
3. open Chapters 9–11 and verify their executed tables are present;
4. verify Chapter 8's manifest gate reports 160 jobs and 160 unique masks for each seed;
5. treat any failed Python cell, missing data artifact, or absent declared sidecar as a failed release gate.

The book reads study data from `../data/` and locked results from
`../results/comprehensive_latest_48_models/`. Synthetic tutorial chapters are
explicitly separated from the real-data evidence chapters.
