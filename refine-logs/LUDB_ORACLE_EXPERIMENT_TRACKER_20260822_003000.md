# LUDB Oracle Experiment Tracker

Updated: 2026-08-22 00:30 America/Toronto

## Status

The NeuroKit/DWT blinded watchdog is stopped. It is not part of the primary
evaluation and has not been restarted.

The CPU-only LUDB oracle evaluator is live in tmux session
`ecgaim_ludb_oracle_eval`:

- launcher: `scripts/run_ecgaim_ludb_oracle_daemon.sh`
- evaluator: `scripts/evaluate_ecgaim_ludb_oracle_daemon.py`
- database: `results/ecgaim_ludb_oracle/ecgaim_ludb_oracle.sqlite`
- log: `results/ecgaim_ludb_oracle/logs/oracle_daemon.log`
- CPU affinity: cores 0-7
- Torch/OMP/MKL/OpenBLAS threads: 6
- CUDA: disabled (`CUDA_VISIBLE_DEVICES` is empty)
- free-disk stop gate: 5 GiB
- eligible models at launch: 118

## Frozen LUDB clock

- records: 200
- annotated lead streams: 2,400 (all 12 leads)
- strict mapped onset/peak/offset landmarks: 175,109
- complete P/QRS/T wave intervals: 58,289
- valid QRS-offset to next-T-onset ST intervals: 19,636
- primary reconstructed leads: V1, V3, V4, V5, V6
- observed controls: I, II, V2
- derived limb controls: III, aVR, aVL, aVF

The evaluator never predicts, shifts, or re-finds a fiducial. NeuroKit is not
imported or executed. Orphan or incomplete annotations are audited and never
silently guessed.

## Verification gates passed

- unit tests: 6 passed
- Python compile and Bash syntax checks: passed
- `git diff --check`: passed
- one-record schema smoke: passed
- final full-200-record evaluation: passed in 171.05 seconds
- SQLite integrity: `ok`
- foreign-key violations: 0
- event error algebra maximum discrepancy: 0.0
- peak/area error algebra maximum discrepancies: 0.0, 0.0, 0.0
- final one-model database size: 39,190,528 bytes

Expected production storage for 118 models is approximately 1.7 GiB. The
daemon truncates the WAL after every model and refuses to continue below the
5 GiB free-space reserve.

## Primary decision rule

Use the record-aggregated Pareto panel on V1/V3/V4/V5/V6. Do not collapse the
panel into an unvalidated weighted score. The panel contains lower-tail signal
correlation, upper-tail signal MSE, exact landmark voltage error, QRS and T
fixed-window RMSE and area errors, and J-point voltage error. A robust winner
must be competitive across the tail metrics, not merely good on the mean.

Status at this snapshot: first production model running; results interpretation
remains pending.
