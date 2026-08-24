# Five-window pairing sensitivity and archive 15

Timestamp: 2026-08-01T08:29:19Z

- The final morphology evaluator computes association windows of 25, 50, 75,
  100, and 150 ms during one detector pass and emits 60 sufficient-statistic
  rows per model. No additional waveforms or checkpoints are retained.
- The gate checks the sensitivity artifact digest and full grid, requires paired
  record/event counts to be monotone nondecreasing and unmatched event counts
  monotone nonincreasing, and requires the 100 ms slice to reproduce the primary
  counts, coverage, means, and variance ratio.
- The book now plots coverage trajectories and endpoint changes from 25 to
  150 ms, while explaining that wider windows add matches by construction and
  can change the selected feature distribution.
- The focused integrity suite passes 45 tests. Final evaluator SHA-256 is
  `1948732602e43ada10c75193dc194230e814e8ff68c4d0d0591901252c2a9eee`.
- `f_1000012_s42` completed exit-zero with final validation MSE 0.0406. Its
  exact 41,010,928-byte checkpoint passed remote round-trip and semantic
  validation, was evicted locally, and refreshed the compatible/inference-ready
  cohort to 15/15 with zero retained inference cache.
- Queue state is 15 completed, 1 running (`f_1000013_s42`), 464 pending.
