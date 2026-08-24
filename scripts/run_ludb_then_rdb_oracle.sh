#!/usr/bin/env bash
set -euo pipefail

project_root="/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction"
python_bin="/home/mithunmanivannan/.venv/bin/python3"
ludb_session="ecgaim_ludb_oracle_eval"
rdb_session="ecgaim_rdb_oracle_eval"
handoff_log="${project_root}/results/ecgaim_ludb_oracle/logs/ludb_to_blinded_to_rdb_handoff.log"

preflight() {
  cd "${project_root}"
  bash -n scripts/run_ludb_then_rdb_oracle.sh scripts/run_ecgaim_ludb_blinded_daemon.sh scripts/run_ecgaim_rdb_oracle_daemon.sh
  "${python_bin}" - <<'PY'
import os
import sqlite3
from pathlib import Path

db_path = Path("results/ecgaim_ludb_oracle/ecgaim_ludb_oracle.sqlite")
if not db_path.is_file():
    raise SystemExit(f"missing LUDB results database: {db_path}")
with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
    assert connection.execute("pragma integrity_check").fetchone()[0] == "ok"
    assert not connection.execute("pragma foreign_key_check").fetchall()
    complete = connection.execute(
        "select count(*) from evaluations where status='complete'"
    ).fetchone()[0]
    running = connection.execute(
        "select count(*) from evaluations where status='running'"
    ).fetchone()[0]
cpus = sorted(os.sched_getaffinity(0))
assert len(cpus) >= 7, f"RDB needs 7 available CPUs, found {len(cpus)}"
print(f"WITNESS HANDOFF_PREFLIGHT LUDB_COMPLETE={complete} LUDB_RUNNING={running} RDB_CPUS={','.join(map(str, cpus[:7]))}")
PY
  scripts/run_ecgaim_rdb_oracle_daemon.sh --preflight
}

worker() {
  mkdir -p "$(dirname "${handoff_log}")"
  cd "${project_root}"
  export CUDA_VISIBLE_DEVICES=""
  export OMP_NUM_THREADS=6
  export MKL_NUM_THREADS=6
  export OPENBLAS_NUM_THREADS=6
  allowed_cpus="$(${python_bin} -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')"
  {
    echo "$(date --iso-8601=seconds) LUDB one-pass resume starting"
    taskset -c "${allowed_cpus}" "${python_bin}" scripts/evaluate_ecgaim_ludb_oracle_daemon.py \
      --results-db results/ecgaim_ludb_oracle/ecgaim_ludb_oracle.sqlite \
      --output-dir results/ecgaim_ludb_oracle \
      --torch-threads 6 --batch-size 4 --min-free-gb 5 --poll-seconds 300 --once
    echo "$(date --iso-8601=seconds) LUDB oracle completed successfully; starting blinded LUDB delineation"
    taskset -c "${allowed_cpus}" "${python_bin}" scripts/evaluate_ecgaim_ludb_blinded_daemon.py \
      --results-db results/ecgaim_ludb/ecgaim_ludb_blinded.sqlite \
      --output-dir results/ecgaim_ludb \
      --workers 6 --torch-threads 6 --batch-size 4 --min-free-gb 5 \
      --poll-seconds 300 --once
    "${python_bin}" - <<'PY'
import sqlite3
from pathlib import Path
from scripts.evaluate_ecgaim_ludb_blinded_daemon import completed_ecgaim_models

queue = Path("refine-logs/queue_3arch/queue_state.json")
checkpoint_db = Path("results/checkpoint_store/catalog.sqlite")
results_db = Path("results/ecgaim_ludb/ecgaim_ludb_blinded.sqlite")
eligible = len(completed_ecgaim_models(queue, checkpoint_db))
with sqlite3.connect(f"file:{results_db}?mode=ro", uri=True) as connection:
    complete = connection.execute(
        "SELECT count(*) FROM evaluations WHERE status='complete' AND model_id!='__original__'"
    ).fetchone()[0]
if complete != eligible:
    raise SystemExit(
        f"Blinded LUDB incomplete ({complete}/{eligible}); refusing premature RDB handoff"
    )
print(f"WITNESS BLINDED_LUDB_COMPLETE={complete} ELIGIBLE={eligible}")
PY
    echo "$(date --iso-8601=seconds) Blinded LUDB delineation completed successfully; launching RDB"
    RDB_ORACLE_CONFIRM_PRODUCTION=I_UNDERSTAND_RDB_PRODUCTION \
      scripts/run_ecgaim_rdb_oracle_daemon.sh
    echo "$(date --iso-8601=seconds) RDB detached tmux launch accepted"
  } 2>&1 | tee -a "${handoff_log}"
}

case "${1:-}" in
  --preflight)
    preflight
    ;;
  --worker)
    worker
    ;;
  "")
    preflight
    if tmux has-session -t "${ludb_session}" 2>/dev/null; then
      echo "Refusing duplicate launch: tmux session ${ludb_session} already exists." >&2
      exit 65
    fi
    if tmux has-session -t "${rdb_session}" 2>/dev/null; then
      echo "Refusing launch: RDB tmux session ${rdb_session} already exists." >&2
      exit 66
    fi
    tmux new-session -d -s "${ludb_session}" \
      "exec '${project_root}/scripts/run_ludb_then_rdb_oracle.sh' --worker"
    echo "Started ${ludb_session}: LUDB oracle, then blinded LUDB delineation, then RDB"
    ;;
  *)
    echo "usage: $0 [--preflight|--worker]" >&2
    exit 64
    ;;
esac
