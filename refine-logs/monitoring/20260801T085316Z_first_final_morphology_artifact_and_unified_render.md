# First final morphology artifact and unified book render

Timestamp: 2026-08-01T08:53:16Z

- `f_1000000_s42` is the first artifact accepted under the final evaluator
  SHA-256. All 2,198 records contribute to each denominator; 12 aggregate rows,
  26,376 per-record rows, and 60 tolerance-sensitivity rows passed the strict
  identity, digest, accounting, reconstruction, and cross-artifact gates.
- The compact additions occupy 1,945,896 bytes for the Parquet ledger and
  21,568 bytes for the sensitivity CSV. No reconstructed waveform is retained.
- Eleven of 12 paired lead-feature rows have variance ratio below one, but V6
  QT is counterevidence at 1.653. Equal-patient mean QT differences are
  -12.6 ms for V3 (descriptive 95% interval -14.5 to -10.7) and -31.2 ms for
  V6 (-35.3 to -27.0).
- QT is reconstruction-only in 82 V3 and 79 V6 records. Coverage is pairing-
  window sensitive: V6 QT rises from 79.44% at 25 ms to 92.31% at 150 ms, and
  V6 S from 85.58% to 99.04%.
- `f_1000013_s42` completed with final validation MSE 0.0414, archived and
  evicted successfully. Queue: 16 completed, 1 running (`f_1000014_s42`),
  463 pending. Compatibility and actual-data inference readiness refreshed to
  16/16 with zero retained inference cache.
- The complete unified Quarto book rendered from all 16 sources into
  `book/_book/index.html`; every executable chapter, including actual PTB-XL,
  EchoNext, LUDB, ISP, Sunnybrook, Zhejiang, checkpoint, and final morphology
  diagnostics, completed successfully.
