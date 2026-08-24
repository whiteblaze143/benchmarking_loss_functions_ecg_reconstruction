#!/usr/bin/env bash
set -euo pipefail

project_root="/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction"
python_bin="/home/mithunmanivannan/.venv/bin/python3"
session_name="ecgaim_rdb_oracle_eval"
log_dir="${project_root}/results/ecgaim_rdb_oracle/logs"

allowed_cpus="$(${python_bin} -c 'import os; cpus=sorted(os.sched_getaffinity(0)); assert len(cpus) >= 7, f"need 7 CPUs, found {len(cpus)}"; print(",".join(map(str, cpus[:7])))')"

if [[ "${1:-}" == "--preflight" ]]; then
  cd "${project_root}"
  export CUDA_VISIBLE_DEVICES=""
  echo "WITNESS RDB_CPU_SET=${allowed_cpus} THREADS=7"
  exec taskset -c "${allowed_cpus}" "${python_bin}" scripts/evaluate_ecgaim_rdb_oracle_daemon.py --preflight --max-records 2 --torch-threads 7
fi

if [[ "${RDB_ORACLE_CONFIRM_PRODUCTION:-}" != "I_UNDERSTAND_RDB_PRODUCTION" ]]; then
  echo "Refusing to launch RDB production evaluation. Run --preflight, then set RDB_ORACLE_CONFIRM_PRODUCTION=I_UNDERSTAND_RDB_PRODUCTION explicitly." >&2
  exit 64
fi
if tmux has-session -t "${session_name}" 2>/dev/null; then
  echo "Refusing duplicate launch: tmux session ${session_name} already exists." >&2
  exit 65
fi

mkdir -p "${log_dir}"
cd "${project_root}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=7
export MKL_NUM_THREADS=7
export OPENBLAS_NUM_THREADS=7

tmux new-session -d -s "${session_name}" \
  "exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=7 MKL_NUM_THREADS=7 OPENBLAS_NUM_THREADS=7 \
  taskset -c '${allowed_cpus}' '${python_bin}' scripts/evaluate_ecgaim_rdb_oracle_daemon.py \
  --results-db results/ecgaim_rdb_oracle/ecgaim_rdb_oracle.sqlite \
  --output-dir results/ecgaim_rdb_oracle \
  --torch-threads 7 --batch-size 8 --min-free-gb 5 --poll-seconds 300 \
  2>&1 | tee -a '${log_dir}/oracle_daemon.log'"
echo "Started ${session_name}"
