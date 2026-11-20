#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run_dir=${1:?Usage: bash scripts/run_hrs2027_full_rerun.sh NEW_OUTPUT_DIRECTORY}
python_bin=/home/mithunmanivannan/.venv/bin/python
hrs_cuda_libs=$(find /home/mithunmanivannan/.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -type d -name lib -printf '%p:')
export LD_LIBRARY_PATH="/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/runtime_cudnn93/nvidia/cudnn/lib:$hrs_cuda_libs"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$run_dir"
"$python_bin" -u scripts/train_hrs2027_full_corpus.py --output-dir "$run_dir" 2>&1 | tee "$run_dir/training.log"
prepare_pids=()
for shard_index in 0 1 2; do
    "$python_bin" -u scripts/run_corrected_benchmark_evaluation.py --output-dir "$run_dir" --prepare-only --shard-index "$shard_index" --shard-count 3 > "$run_dir/prepare_shard$shard_index.log" 2>&1 &
    prepare_pids+=("$!")
done
for prepare_pid in "${prepare_pids[@]}"; do wait "$prepare_pid"; done
"$python_bin" scripts/audit_model_provenance_and_leakage.py --run-dir "$run_dir"
"$python_bin" -u scripts/run_corrected_benchmark_evaluation.py --output-dir "$run_dir" --checkpoint "$run_dir/final_setoperator.pt" 2>&1 | tee "$run_dir/evaluation.log"
"$python_bin" -u scripts/run_wearable_completeness_monte_carlo.py --output-dir "$run_dir" --iterations 1000 2>&1 | tee "$run_dir/wear_simulation.log"
"$python_bin" -u scripts/summarize_hrs2027_rerun.py --output-dir "$run_dir" 2>&1 | tee "$run_dir/reconciliation.log"
