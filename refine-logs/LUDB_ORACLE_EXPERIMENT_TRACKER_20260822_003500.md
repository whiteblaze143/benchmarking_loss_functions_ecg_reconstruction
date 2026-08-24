# LUDB Oracle Experiment Tracker

Updated: 2026-08-22 00:35 America/Toronto

The CPU-only oracle evaluator remains healthy in tmux session
`ecgaim_ludb_oracle_eval`. SQLite integrity is `ok`; two models are complete
and a third is running. The training queue currently contains 160 ECG-AIM
cells: 118 trained, 1 training, and 41 pending. Therefore any present model
ordering is explicitly provisional.

The read-only analysis layer is now implemented at
`scripts/analyze_ecgaim_ludb_oracle.py`. It writes:

- raw model and validation comparison;
- the nondominated Pareto front;
- a transparent minimax-rank sensitivity ordering;
- matched factorial main effects and binary interactions, including
  correlation × VCG;
- MMD-kernel contrasts against no MMD;
- diagnostic-subgroup record-level tail summaries.

Validation missing-lead Pearson is included in the Pareto panel. Composite
validation loss is retained for audit only because its objective and scale
change across masks.

Current raw comparison:

- `1000000` (MSE only): signal Pearson p05 0.4232; event-voltage error p95
  0.3574 mV; QRS-window RMSE p95 0.2341 mV; validation Pearson 0.8191.
- `1000001` (MSE + global-RBF MMD): signal Pearson p05 0.0848;
  event-voltage error p95 0.5864 mV; QRS-window RMSE p95 0.3084 mV;
  validation Pearson 0.5212.

Thus global-RBF MMD is dominated in the only completed matched contrast, but
this is not yet a final kernel main-effect estimate or winner decision.

Tests: 9 passed across the evaluator and oracle analysis modules.
