# Corrected target cache and actual-data detector EDA

Timestamp: 2026-08-01T08:00:16Z

- The final R-peak-constrained PTB-XL target cache published successfully from
  all 2,198 actual test records: 287,256 event rows, zero duplicate
  `(record, lead, feature, sample_index)` identities, and cache SHA-256
  `26ebc1bd52fe409a985361f51d838210703b78a0a4b35aba8ffc8f2488bf318c`.
- The untrimmed QT audit now spans 72--956 ms; no QT event exceeds 1,000 ms.
  This is a structural extraction result, not a post-hoc clipping rule.
- A hash-bound target-only EDA artifact now reports all 12 lead-feature
  combinations and 84 subgroup coverage rows from the actual PTB-XL metadata.
  V3 coverage is 99.09% for P, 94.68% for QT, 98.59% for Q, 99.32% for R,
  99.27% for S, and 96.72% for T. V6 coverage is 99.91%, 95.86%, 98.59%,
  99.95%, 99.95%, and 98.45%, respectively.
- The lowest observed subgroup coverage is V3 QT in the isolated invalid-age
  group (29/34, 85.29%). Raw sex codes remain unmapped because the local table
  does not provide a trustworthy textual coding key.
- The first compatible checkpoint is currently being evaluated against the
  corrected cache. Acceptance remains deliberately 0/13 until its atomic
  result passes the evaluator, cache, checkpoint, source, and row-accounting
  gates; four earlier-generation artifacts remain explicitly excluded.
- Queue state remains healthy at 13 completed, 1 running
  (`f_1000011_s42`), and 466 pending. The model has progressed through seven
  complete epochs with validation MSE decreasing from 0.1233 to 0.0545.
- Archive/inference storage is bounded: 13 current checkpoints are
  remote-verified, the active evaluator holds one 41 MB cache entry, and all
  13 models passed strict inference on actual PTB-XL test record 100. The
  focused integrity suite passes 40 tests.
