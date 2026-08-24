# LUDB Oracle Experiment Tracker

Updated: 2026-08-22 00:42 America/Toronto

- Oracle daemon: healthy, CPU-only, five complete and one running.
- SQLite: integrity `ok`, zero foreign-key violations.
- Primary analysis: oracle-only Pareto and minimax ordering.
- Supporting analysis: validation Pearson and a separate augmented Pareto.
- Statistical analysis: paired 2,000-resample record bootstrap against MSE-only.
- Completed slice: MMD kernel 0–4 with other optional losses disabled.
- Current oracle Pareto: `1000000`, `1000002`, `1000003`.
- Current extreme-robustness leader: `1000000` (provisional).
- Rejected in this slice: `1000001` and `1000004`.
- Full grid status: 118 trained, 1 training, 41 pending; 5/160 oracle-evaluated.

See `LUDB_ORACLE_INTERIM_FINDINGS_20260822_004200.md` for raw values and the
bounded interpretation.
