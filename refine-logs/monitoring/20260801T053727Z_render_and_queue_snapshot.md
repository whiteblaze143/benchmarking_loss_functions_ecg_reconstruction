# Unified Render and Queue Snapshot

Recorded at 2026-08-01T05:37:27Z.

## Unified book

- Full Quarto render exit code: 0.
- Entry point: `book/_book/index.html` (61,070 bytes).
- Rendered pages: 15 HTML files = index plus 14 chapters.
- Total rendered book size: 5.9 MiB.
- Local HTML/CSS/JavaScript/image references checked: 672.
- Missing local references: 0.
- Rendered runtime-error containers detected: 0.
- Sidebar order begins with real-cohort EDA, then the LUDB/ISP/Sunnybrook atlas, then preprocessing.

The render exercised all 17 real-data EDA cells, all 13 dataset-atlas cells, all 12 EchoNext analysis cells, and the current queue/checkpoint diagnostics.

## Queue and archive

- Queue at this snapshot: 7 completed, 1 running, 472 pending.
- Running model: `f_1000000_s42`.
- Compatibility audit: 7 compatible, 0 incompatible.
- Checkpoint blobs are remotely verified; one 41 MiB cache copy may exist transiently while CPU evaluation is active and is pruned after the model finishes.

The checkpoint archiver was corrected to ignore `cached` inference copies. Before this change it could redundantly re-upload a verified model while evaluation had it materialized. The canonical training-checkpoint path remains the only source for a new archive generation. The focused checkpoint-store suite passed 20/20 tests after the change.

## Generation-bound evaluation

The temporal-MMD watcher runs on CPU core 7 only, with one Torch thread, one feature-extraction worker, niceness 19, and idle-class I/O. It writes one atomic CSV and JSON sidecar per full model ID, keyed by checkpoint, source, test-data, and evaluator digests. No partial per-model artifact is published.
