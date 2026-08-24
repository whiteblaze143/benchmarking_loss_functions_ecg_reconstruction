#!/usr/bin/env bash
set -euo pipefail

project_root="/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction"
python_bin="/home/mithunmanivannan/.venv/bin/python3"
session="ecgaim_ludb_semiseg_blinded"
log="${project_root}/results/ecgaim_ludb_semiseg_blinded/daemon.log"

if tmux has-session -t "${session}" 2>/dev/null; then
  echo "Refusing duplicate launch: ${session}" >&2
  exit 65
fi
mkdir -p "$(dirname "${log}")"
tmux new-session -d -s "${session}" \
  "bash -lc 'set -o pipefail; cd ${project_root}; export CUDA_VISIBLE_DEVICES=; export OMP_NUM_THREADS=6; export MKL_NUM_THREADS=6; export OPENBLAS_NUM_THREADS=6; flock -n results/ecgaim_ludb_semiseg_blinded/worker.lock taskset -c 0-5 ${python_bin} scripts/evaluate_ecgaim_ludb_semiseg_blinded.py --torch-threads 6 --reconstruction-batch-size 4 --delineation-batch-size 64 --min-free-gib 8 2>&1 | tee -a ${log}'"
sleep 1
if ! tmux has-session -t "${session}" 2>/dev/null; then
  echo "Launch failed before the evaluator remained live; inspect ${log}" >&2
  exit 70
fi
echo "Started ${session} on CPU cores 0-5"
