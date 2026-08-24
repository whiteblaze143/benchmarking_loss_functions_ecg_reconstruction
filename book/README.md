# Rendering the ECG reconstruction book

The rendered entry point is [`_book/index.html`](_book/index.html).

The first section, **Live Experiment Observatory**, is rebuilt directly from
the current SQLite databases. It separates the three-lead ECG-AIM benchmark
from the one-lead spatial/wavelet/SSL program and provides interactive raw
leaderboards, matched deltas, uncertainty tables, boundary heatmaps, and
performance-profile UMAPs.

All executable chapters use an explicitly selected environment. Activate one
that satisfies [`requirements-book.txt`](requirements-book.txt), then run from
the repository root:

```bash
python3 scripts/render_quarto_chapters.py \
  --round-dir review-stage/render-release
```

The wrapper rejects the wrong Quarto version, snapshots mutable SQLite inputs,
records the Python executable/version, and binds each page to its source hash.
It sets `QUARTO_PYTHON` to the interpreter used to launch the wrapper.

The project uses shared site assets (`embed-resources: false`). Those resources
are still embedded in the deployable `book/_book` artifact under `site_libs/`,
and the Pages workflow uploads the whole artifact. Embedding every Plotly
dependency separately inside every HTML file previously produced pages larger
than 250 MB and caused full renders to be killed for memory use. A release audit
must therefore verify every local `src`/`href` resolves inside `_book`.

### Fast live refresh

To refresh only the pages backed by actively changing databases:

```bash
python3 scripts/render_quarto_chapters.py \
  --round-dir review-stage/render-live \
  --chapter 15_live_results_observatory.qmd \
  --chapter 16_three_lead_ecgaim_live.qmd \
  --chapter 17_one_lead_wavelet_ssl_live.qmd
```

The readers use SQLite URI `mode=ro`; rendering cannot modify an experiment
database. No CSV export is required.

The longer checkpoint-representation study runs independently in tmux and can
be monitored with:

```bash
tmux capture-pane -p -t checkpoint_embedding_rdb
```

Its compact live database is `results/checkpoint_embeddings/compact.sqlite`.
The actual 2,304-dimensional pooled checkpoint embeddings are stored there as
finite float16 arrays compressed with zlib and keyed by checkpoint SHA-256,
record manifest, hook layer, and pooling version; they are not replaced by only
the 2-D UMAP coordinates. Derived statistical tables populate after extraction.
Create `results/checkpoint_embeddings/STOP` for a graceful batch-boundary stop;
remove the sentinel and rerun the identical command to resume verified rows.
The rigorous one-core handoff can be monitored with:

```bash
tmux capture-pane -p -t checkpoint_embedding_postanalysis
```

### GitHub Pages

The Pages workflow deploys the checked-in `book/_book` snapshot; authoritative
databases and checkpoints remain ignored. HTML existence alone is not a release
gate. Vendor runtime assets, run the audit, and build a manifest. The manifest
builder refuses production while a scientific completion gate fails;
`--provisional` labels the candidate explicitly nondeployable.
GitHub Pages uploads a single immutable artifact and independently verifies the
production status plus every page hash before deployment, so a partially
updated local directory cannot be promoted.

```bash
python3 scripts/vendor_book_runtime.py
python3 scripts/audit_quarto_book.py
python3 scripts/build_book_release_manifest.py \
  --render-state review-stage/render-release/render_state.json \
  --provisional
```

## Release checks

After a render:

1. confirm `_book/index.html` exists;
2. confirm the sidebar begins with the three Live Experiment Observatory pages;
3. open Chapters 9–11 and verify their executed tables are present;
4. verify Chapter 8's manifest gate reports 160 jobs and 160 unique masks for each seed;
5. require zero stale/missing pages, unresolved resources, external runtime
   dependencies, or exposed record identifiers in the mechanical audit;
6. treat failed cells, absent declared sidecars, or a failed scientific
   completion gate as a failed production-release gate.

The book reads study data from `../data/`, core generation-bound results from
`../results/factorial_v4/`, clinical results from
`../results/factorial_v4_clinical/`, and compact live SQLite databases under
`../results/`. Synthetic tutorial chapters are explicitly separated from
real-data evidence chapters. The removed `comprehensive_latest_48_models`
bundle is not a current dependency.
