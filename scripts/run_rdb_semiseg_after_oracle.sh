#!/usr/bin/env bash
set -euo pipefail

project_root="/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction"
python_bin="/home/mithunmanivannan/.venv/bin/python3"
session="ecgaim_rdb_semiseg_handoff"
log="${project_root}/results/ecgaim_rdb_semiseg_blinded/supervisor.log"

worker() {
  cd "${project_root}"
  mkdir -p "$(dirname "${log}")"
  while true; do
    witness="$(${python_bin} - <<'PY'
import sqlite3
from pathlib import Path
p=Path('results/ecgaim_rdb_oracle/ecgaim_rdb_oracle.sqlite')
if not p.is_file(): print('MISSING'); raise SystemExit
with sqlite3.connect(f'file:{p.resolve()}?mode=ro',uri=True) as c:
    integrity=c.execute('pragma integrity_check').fetchone()[0]
    counts=dict(c.execute('select status,count(*) from evaluations group by status'))
print(f"INTEGRITY={integrity} COMPLETE={counts.get('complete',0)} RUNNING={counts.get('running',0)} ERROR={counts.get('error',0)}")
PY
)"
    echo "$(date --iso-8601=seconds) ${witness}" | tee -a "${log}"
    if [[ "${witness}" == "INTEGRITY=ok COMPLETE=31 RUNNING=0 ERROR=0" ]]; then break; fi
    if ! tmux has-session -t ecgaim_rdb_oracle_eval 2>/dev/null; then
      echo "RDB oracle exited before 31 clean completions; refusing learned handoff" | tee -a "${log}"
      exit 70
    fi
    sleep 60
  done
  oracle_pid="$(pgrep -f '^/home/mithunmanivannan/.venv/bin/python3 scripts/evaluate_ecgaim_rdb_oracle_daemon.py' | head -1 || true)"
  if [[ -n "${oracle_pid}" ]]; then kill -TERM "${oracle_pid}"; fi
  for _ in $(seq 1 60); do
    if ! tmux has-session -t ecgaim_rdb_oracle_eval 2>/dev/null; then break; fi
    sleep 1
  done
  if tmux has-session -t ecgaim_rdb_oracle_eval 2>/dev/null; then
    echo "RDB oracle did not stop cleanly; refusing learned handoff" | tee -a "${log}"
    exit 71
  fi
  echo "$(date --iso-8601=seconds) RDB oracle complete; starting learned blinded RDB" | tee -a "${log}"
  export CUDA_VISIBLE_DEVICES=""
  export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
  exec flock -n results/ecgaim_rdb_semiseg_blinded/worker.lock \
    taskset -c 0-5 "${python_bin}" scripts/evaluate_ecgaim_rdb_semiseg_blinded.py \
    --torch-threads 6 --reconstruction-batch-size 8 --delineation-batch-size 64 \
    --min-free-gib 8 2>&1 | tee -a "${log}"
}

if [[ "${1:-}" == "--worker" ]]; then worker; exit; fi
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "Refusing duplicate launch: ${session}" >&2; exit 65
fi
mkdir -p "$(dirname "${log}")"
tmux new-session -d -s "${session}" "exec '${project_root}/scripts/run_rdb_semiseg_after_oracle.sh' --worker"
sleep 1
tmux has-session -t "${session}" 2>/dev/null || { echo "handoff failed to remain live" >&2; exit 70; }
echo "Started ${session}: waits for fixed-region RDB oracle, then runs learned blinded RDB"
