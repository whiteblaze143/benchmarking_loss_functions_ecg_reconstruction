#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
config_path="$repo_root/configs/semiseg_ludb_mt_full_cpu.yaml"
run_dir="$repo_root/results/semiseg_ludb_training/vit_tiny_mean_teacher_full_s42"
supervisor_log="$repo_root/results/semiseg_ludb_training/vit_tiny_mean_teacher_full_s42.supervisor.log"
runtime_python=/home/mithunmanivannan/.venv/bin/python3

completed_epochs() {
    "$runtime_python" - "$run_dir/log.txt" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
if not p.exists():
    print(0)
else:
    print(len({int(json.loads(line)["epoch"]) for line in p.read_text().splitlines() if line.strip()}))
PY
}

mkdir -p "$(dirname "$supervisor_log")"
exec >>"$supervisor_log" 2>&1
echo "supervisor_start $(date --iso-8601=seconds)"

attempt=0
while true; do
    while pgrep -f "[t]rain.py --config_path $config_path" >/dev/null; do
        sleep 30
    done
    epochs="$(completed_epochs)"
    echo "training_exit $(date --iso-8601=seconds) epochs=$epochs attempt=$attempt"
    if [[ "$epochs" -ge 100 ]]; then
        break
    fi
    attempt=$((attempt + 1))
    if [[ "$attempt" -gt 5 ]]; then
        echo "supervisor_failure too_many_restarts epochs=$epochs"
        exit 1
    fi
    echo "training_restart $(date --iso-8601=seconds) attempt=$attempt"
    bash "$repo_root/scripts/run_semiseg_ludb_mt_cpu.sh"
done

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=7
export MKL_NUM_THREADS=7
export OPENBLAS_NUM_THREADS=7
export NUMEXPR_NUM_THREADS=7
cpu_list="$($runtime_python - <<'PY'
import os
print(",".join(map(str, sorted(os.sched_getaffinity(0))[:7])))
PY
)"
echo "finalization_start $(date --iso-8601=seconds) cpus=$cpu_list"
taskset -c "$cpu_list" "$runtime_python" "$repo_root/scripts/finalize_semiseg_ludb_mt.py"
echo "supervisor_complete $(date --iso-8601=seconds)"
