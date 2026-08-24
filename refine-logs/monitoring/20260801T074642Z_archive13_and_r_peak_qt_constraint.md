# Archive 13 and R-peak-constrained QT association

Timestamp: 2026-08-01T07:46:42Z

- A duplicate-free QRS-cycle cache audit still found 105/39,922 QT values above
  1,000 ms (maximum 4,042 ms) and values as short as 16 ms. This showed that
  uniqueness alone did not prove same-beat association.
- The evaluator was stopped after one reconstruction batch and before any
  current-generation model artifact published.
- QT association now requires an ordered within-beat chain: detected QRS onset,
  a detected R peak after that onset, then the first T offset after the R peak
  and before the next detected R peak or QRS onset. This structurally rejects
  pre-R T offsets and cross-beat T offsets without hiding them behind a numeric
  clipping rule.
- Thirteen focused matcher/cache/QT/summary/EDA tests pass. Direct checks on
  previously pathological records retain unique, plausible case values.
- The R-peak-constrained target cache is rebuilding on CPU core 7. Current
  acceptance remains deliberately 0/13 until it publishes and a model passes.
- `f_1000010_s42` completed exit-zero with final validation MSE 0.0410, archived
  and evicted successfully. Queue: 13 completed, 1 running
  (`f_1000011_s42`), 466 pending. Compatibility: 13/13.
- The cohort-aware inference watcher automatically refreshed to 13/13 with
  zero retained cache bytes.

