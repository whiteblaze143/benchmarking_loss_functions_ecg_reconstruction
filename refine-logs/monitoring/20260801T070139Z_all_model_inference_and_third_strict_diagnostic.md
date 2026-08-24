# All-model inference and third strict diagnostic

Timestamp: 2026-08-01T07:01:39Z

- Queue: 10 completed, 1 running (`f_1000003_s42`), 469 pending under the
  content-pinned contract. The active run reached Epoch 8 after monotonically
  reducing validation MSE from 0.0973 at Epoch 1 to 0.0365 at Epoch 7.
- Storage/inference audit: all 10 compatible archived identities strictly
  loaded and reconstructed actual PTB-XL test tensor `100.pt`; all outputs were
  finite and shaped `[1, 12, 5000]`. The audit traversed 410,109,856 logical
  checkpoint bytes and retained zero checkpoint-cache bytes.
- Compact case diagnostics: 90 model-by-missing-lead rows record MSE, MAE,
  Pearson correlation, means, standard deviations, and variance ratios without
  retaining prediction tensors. On this indexed case, V1 has the weakest median
  correlation while V3/V4 combine high correlation with inflated variance.
- Strict temporal morphology gate: 3/10 accepted, 36 feature rows, zero
  exclusions, and zero reconstruction-batch failures.
- Matched partial contrast: `f_1011012_s42` and `f_1011013_s42` differ only in
  MMD kernel level. Kernel 3 lowers variance ratio in 9/12 detector-feature rows,
  but V3/V6 QT move oppositely and V6 QT coverage falls. This is one-seed
  diagnostic evidence, not a confirmatory factorial effect.
- Integrity gates: benchmark JSON binds the input, compatibility audit,
  benchmark source, current contract, and both CSVs by SHA-256. The book and
  artifact test fail if the compatible cohort changes without a fresh audit.
- Independent-review limitation: the requested Claude bridge remains
  unavailable; no same-agent assessment is labeled independent.

