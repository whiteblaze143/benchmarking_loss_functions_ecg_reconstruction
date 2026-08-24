# Checkpoint store and strict diagnostic snapshot

Timestamp: 2026-08-01T06:45:26Z

- Queue: 10 completed, 1 running (`f_1000003_s42`), 469 pending under
  `factorial-v4-content-pinned-20260731`.
- Archive: 10/10 completed checkpoints pass the current compatibility audit;
  9 are remote-only and one bounded cache copy is in active diagnostic use.
- Local catalog footprint: 8.9 MB SQLite plus an 8.5 MB portable JSONL recovery
  snapshot. The active checkpoint cache is one approximately 40 MB file.
- Inference: `infer_factorial_checkpoint.py --list-ready` returns the ten exact
  compatible model ids. Inference now rejects a stale training contract, wrong
  approved source bundle, absent compatibility row, or catalog/audit digest
  mismatch, and prunes its cache in `finally` by default.
- End-to-end inference proof: `f_1011013_s42` loaded exact digest
  `5c4a6f3116f0faaab1587ebbadd9066a921147aef84be9c91214369cbaeaf6ad`
  from the bounded cache and reconstructed the real PTB-XL record 100 from
  `[1, 3, 5024]` padded input to a finite `[1, 12, 5000]` output. The retained
  cache stayed below the requested 0.05 GiB bound.
- Strict temporal morphology gate: 2/10 accepted, 0 excluded, 24 feature rows,
  2,198-record denominator, and zero reconstruction-batch failures for both
  accepted artifacts.
- Descriptive limitation: `f_1000000_s42` and `f_1011012_s42` differ in several
  loss factors and share only seed 42; their feature summaries are diagnostic,
  not a causal factorial contrast.
- Tests: 26 focused checkpoint/inference/temporal tests passed in the project
  virtual environment; the only warning is NeuroKit's upstream `scipy.misc`
  deprecation.
- Unified render: 16 HTML pages, 735 local references checked, zero missing,
  zero runtime-error markers, and zero unresolved Quarto cross-references.
