# QT event-reuse bug repaired

Timestamp: 2026-08-01T07:33:28Z

- The first schema-v2 target cache exposed 6,707 duplicate
  `(record, lead, QT, QRS-onset sample)` rows across 2,554 groups. No amplitude
  feature duplicated a same-sample event.
- Root cause: the previous QT loop iterated over T offsets and repeatedly reused
  the last earlier QRS onset when intervening onsets were absent. Examples
  included pseudo-QT values above 9 seconds and up to 17 T offsets assigned to
  one onset.
- The evaluator was stopped after five reconstruction batches and before any
  artifact from that generation published. Its bounded checkpoint cache was
  pruned.
- Corrected extraction iterates unique QRS onsets and selects at most the first
  T offset after that onset and before the next QRS onset. The final onset may
  use the first later offset. Each QRS onset can therefore anchor at most one QT
  interval and no interval crosses a detected next-beat boundary.
- The QT helper source and clinical extractor source are both included in the
  extractor SHA, forcing a fresh target cache.
- Two QT-specific tests cover one-use-per-onset and no-cross-next-onset
  behavior. Combined matcher, cache, summary, QT, and target-EDA tests: 12
  passed; only NeuroKit's upstream `scipy.misc` deprecation warning remains.
- Corrected target-cache rebuild began on isolated CPU core 7. Current
  generation remains fail-closed at 0/12 accepted until it and a model artifact
  publish.

