# Compact per-record morphology ledger generation

Timestamp: 2026-08-01T08:21:40Z

- The explicit-distribution-MMD evaluator was paused before publishing because
  its aggregate-only output still could not support patient clustering or
  reconstruction-side detector-failure strata.
- Each model now atomically publishes a compressed Parquet ledger with exactly
  2,198 records × two leads × six features = 26,376 rows. It retains scalar
  detector states, paired/unmatched counts, feature-error summaries, timing
  errors, and sufficient statistics; it does not retain waveforms or another
  model checkpoint.
- The acceptance builder verifies the Parquet digest, complete denominator,
  unique record-lead-feature identities, detector-state logic, finite/null
  semantics, and event accounting. It then reconstructs every aggregate count,
  mean, and variance ratio from the ledger and rejects disagreement.
- The engineering chapter now uses the first accepted ledger to report all four
  detector states and fixed-seed patient-cluster percentile intervals for
  paired mean feature differences, with explicit detector-conditioning and
  multiplicity limitations.
- The combined checkpoint/inference/morphology integrity suite passes 44 tests.
  Evaluator SHA-256 is
  `98da42cfb10d869e90ce96471d2790ad1d18ce622f032f17bf5bb0a927f391d8`.
- The evaluator restarted on CPU core 7 and reused the verified target cache.
  Training and archiving continued uninterrupted.
