# RDB Oracle Experiment Tracker

Updated: 2026-08-22 01:01 America/Toronto

- Status: implementation complete; production deliberately not launched.
- Full preflight: PASS, 2,398 unique-source records.
- Full dataset SHA-256: `81d69552522ae2b86116e33f578cbdf5965214f284c5092523d19660acefda83`.
- Unit/integration tests: 15 passed, including vectorized paired-bootstrap orientation.
- CPU model smoke: PASS, MSE-only `1000000`, one RDB record, temporary DB only.
- Smoke wall time/RSS: 12.81 s / 1,483,152 KiB with one Torch thread.
- Smoke SQLite: integrity `ok`, zero foreign-key violations; evaluator and analyzer exports created.
- Worst-case compact-schema sizing: 5.55 MB/model, approximately 0.83 GiB for 160 models before modest indexes/summary exports; budget 1–1.5 GiB total.
- Safety guard: launcher without confirmation exits 64; production tmux and production DB absent.
- Current ECG-AIM training grid at smoke time: 119 complete, 1 running, 40 pending.
- Current LUDB oracle daemon: active and left untouched.
- Fresh agent-follows-doc gate: PASS twice, including after analysis optimization; verbatim smoke exit 0 with `WITNESS RDB_SMOKE_COMPLETE`, no runbook divergence, and production DB/session absent.
- Deferred: production run, final 160-cell analysis, multi-seed confirmation, and independent postrun audit.
