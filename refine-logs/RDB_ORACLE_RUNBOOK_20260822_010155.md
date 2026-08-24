# RDB Oracle Runbook

## Read-only preflight

```bash
cd /home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
CUDA_VISIBLE_DEVICES='' /home/mithunmanivannan/.venv/bin/python3 scripts/evaluate_ecgaim_rdb_oracle_daemon.py --preflight --max-records 2 --torch-threads 1
```

Expected witness: one JSON object with `event=preflight_complete`, `records=2`, `device` implicitly CPU by the empty CUDA environment, and no results DB creation.

## Documented smoke invocation

This uses only a newly created temporary directory and must not create the production DB or tmux session.

```bash
cd /home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
smoke_root="$(mktemp -d /tmp/ecgaim-rdb-doc-smoke.XXXXXX)"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/mithunmanivannan/.venv/bin/python3 scripts/evaluate_ecgaim_rdb_oracle_daemon.py --results-db "${smoke_root}/smoke.sqlite" --output-dir "${smoke_root}/exports" --torch-threads 1 --batch-size 1 --max-records 1 --max-models 1 --model-id factorial_ecg_aim_1000000_s42 --once --min-free-gb 0
/home/mithunmanivannan/.venv/bin/python3 scripts/analyze_ecgaim_rdb_oracle.py --db "${smoke_root}/smoke.sqlite" --output-dir "${smoke_root}/analysis" --bootstrap-resamples 10
/home/mithunmanivannan/.venv/bin/python3 -c 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); assert c.execute("pragma integrity_check").fetchone()[0]=="ok"; assert not c.execute("pragma foreign_key_check").fetchall(); assert c.execute("select status from evaluations").fetchone()[0]=="complete"; print("WITNESS RDB_SMOKE_COMPLETE")' "${smoke_root}/smoke.sqlite"
```

Expected final witness: `WITNESS RDB_SMOKE_COMPLETE`.

## Production launch — do not run yet

Only after LUDB is complete/stopped, at least 5 GiB is free, and the user explicitly authorizes launch:

```bash
cd /home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
RDB_ORACLE_CONFIRM_PRODUCTION=I_UNDERSTAND_RDB_PRODUCTION scripts/run_ecgaim_rdb_oracle_daemon.sh
```

The launcher creates detached tmux session `ecgaim_rdb_oracle_eval`. Never run a second copy.
