#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
vendor_root="$repo_root/external/semiseg/semi-seg-ecg"
runtime_python=/home/mithunmanivannan/.venv/bin/python3
runtime_deps="$repo_root/external/semiseg/runtime_deps"
config_path="$repo_root/configs/semiseg_ludb_mt_full_cpu.yaml"
log_path="$repo_root/results/semiseg_ludb_training/vit_tiny_mean_teacher_full_s42.console.log"
checkpoint_path="$repo_root/results/semiseg_ludb_training/vit_tiny_mean_teacher_full_s42/best-MeanIoU.pth"

cpu_list="$($runtime_python - <<'PY'
import os
allowed = sorted(os.sched_getaffinity(0))
if len(allowed) < 7:
    raise SystemExit(f"need seven CPUs, found {allowed}")
print(",".join(map(str, allowed[:7])))
PY
)"

mkdir -p "$(dirname "$log_path")"
export SEMISEG_CPU_ONLY=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=7
export MKL_NUM_THREADS=7
export OPENBLAS_NUM_THREADS=7
export NUMEXPR_NUM_THREADS=7
export PYTHONPATH="$repo_root/scripts/semiseg_cpu_site:$runtime_deps:$vendor_root/src"

cd "$vendor_root/src"
resume_args=()
if [[ -f "$checkpoint_path" ]]; then
    resume_args=(--resume "$checkpoint_path")
fi
taskset -c "$cpu_list" "$runtime_python" train.py --config_path "$config_path" "${resume_args[@]}" 2>&1 | tee -a "$log_path"
