# Experiment Tracker: Paper 07 — Continuous Measurement-Operator ECG

## Milestone Checklist

- [x] Mathematical specification audit and prior art boundary definition.
- [x] Operator bank definition and disjoint separation verification (`seen`, `derived_limb`, `dense`, `i_to_v2`).
- [x] Response atom orientation law proof ($a_{-q} = -a_q$).
- [x] Context sampling holdout verification ($t \notin C$).
- [x] Paired capacity-matched model architectures (`OperatorSetModel`).
- [x] 8 Synthetic proof worlds designed and verified in `tests/test_paper07_synthetic_recovery.py`.
- [x] Cross-paper regression test suite passing.
- [x] 1-epoch GPU smoke test executed in detached `tmux` session.
- [x] Research logs updated and committed.
