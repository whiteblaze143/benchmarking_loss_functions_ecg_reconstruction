# RDB Oracle Experiment Tracker

Updated: 2026-08-22 America/Toronto

- Status: implementation complete; production deliberately not launched.
- RDB cohort corrected: 31 of 123 completed LUDB oracle checkpoints meet `signal_pearson_p05 >= 0.50`; the other 92 are excluded.
- Cohort manifest SHA-256: `e88f3b31bda574a73e09678f2f0aa33346a9c2c1baa31062e91a606ac9182aa2`; archive checkpoint SHA-256 verification: PASS for all 31.
- Literal cutoff excludes MSE-only `1000000` (0.4232); nearest pass is `1001004` (0.5014), nearest failure is `1001013` (0.4956).
- Full preflight: PASS, 2,398 unique-source records.
- Full dataset SHA-256: `81d69552522ae2b86116e33f578cbdf5965214f284c5092523d19660acefda83`.
- Unit/integration tests: 15 passed, including vectorized paired-bootstrap orientation.
- CPU model smoke: PASS, MSE-only `1000000`, one RDB record, temporary DB only.
- Corrected-cohort CPU smoke: PASS with eligible `1110000`, one RDB record, temporary DB only; stored threshold/manifest verified and SQLite integrity `ok`.
- Smoke wall time/RSS: 12.81 s / 1,483,152 KiB with one Torch thread.
- Smoke SQLite: integrity `ok`, zero foreign-key violations; evaluator and analyzer exports created.
- Worst-case compact-schema sizing: 5.55 MB/model, approximately 0.83 GiB for 160 models before modest indexes/summary exports; budget 1–1.5 GiB total.
- Safety guard: launcher without confirmation exits 64; production tmux and production DB absent.
- Current ECG-AIM training grid at smoke time: 119 complete, 1 running, 40 pending.
- Current LUDB oracle daemon: active and left untouched.
- Current handoff state: blinded LUDB has 39/123 model evaluations complete (plus original ceiling) and is stopped at the 5 GiB disk gate; RDB has not launched and its production DB is absent.
- Fresh agent-follows-doc gate: PASS twice, including after analysis optimization; verbatim smoke exit 0 with `WITNESS RDB_SMOKE_COMPLETE`, no runbook divergence, and production DB/session absent.
- Deferred: production run, final 31-model analysis, multi-seed confirmation, and independent postrun audit.
