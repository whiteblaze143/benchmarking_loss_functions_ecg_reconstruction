# Explicit distribution-MMD evaluator generation

Timestamp: 2026-08-01T08:16:18Z

- The first time-matched artifact passed all existing gates, but a content audit
  found that the temporal-MMD evaluator emitted coverage, mean, variance, and
  robust Bland--Altman diagnostics without an explicit MMD value.
- The CPU evaluator was interrupted during the next model and before that model
  published. Training, queue management, and checkpoint archiving continued.
- Every lead-feature row now includes a deterministic biased squared RBF MMD,
  the selected bandwidth, both sketch sizes, estimator name, bandwidth rule,
  sketch rule, and sample cap.
- The bounded estimator sorts each finite paired distribution, takes at most
  512 evenly spaced quantile-index values, and uses the combined sketch's median
  nonzero pairwise absolute distance as the RBF bandwidth. This bounds the
  quadratic kernel calculation and contains no random subsampling.
- The acceptance builder rejects a negative or non-finite MMD, invalid sketch
  counts, or any estimator/rule/cap mismatch. The new evaluator SHA-256 is
  `c56484b05eb75cabb8b011fbd005b6255a9b49669aad1aaf8a68fecb96cf47e6`.
- The combined focused suite now passes 43 tests. The evaluator restarted on
  CPU core 7 and reused the verified 287,256-row target cache; acceptance is
  intentionally 0/14 until the first new-generation artifact publishes.
