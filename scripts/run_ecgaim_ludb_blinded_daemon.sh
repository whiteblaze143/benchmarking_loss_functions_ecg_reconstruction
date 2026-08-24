#!/usr/bin/env bash
set -euo pipefail

project_root="/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction"
python_bin="/home/mithunmanivannan/.venv/bin/python3"
log_dir="${project_root}/results/ecgaim_ludb/logs"

mkdir -p "${log_dir}"
cd "${project_root}"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

exec taskset -c 0-7 "${python_bin}" \
  scripts/evaluate_ecgaim_ludb_blinded_daemon.py \
  --results-db results/ecgaim_ludb/ecgaim_ludb_blinded.sqlite \
  --output-dir results/ecgaim_ludb \
  --workers 6 \
  --torch-threads 6 \
  --batch-size 4 \
  --min-free-gb 5 \
  --poll-seconds 300 \
  2>&1 | tee -a "${log_dir}/blinded_daemon.log"
