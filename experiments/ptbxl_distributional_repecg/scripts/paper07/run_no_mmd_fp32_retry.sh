#!/usr/bin/env bash
# Exact no-MMD deletion control, retried in float32 after bfloat16 became non-finite.
set -euo pipefail

project=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/ptbxl_distributional_repecg
python=/home/mithunmanivannan/.venv/bin/python
root=/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg
strict_log="$root/braid_field/logs/strict_graphecg_comparison_queue_r2.log"
output="$root/paper07_full12_aux_mmd/continuous_full12_aux_no_mmd_fp32_r1"

until test -f "$strict_log" && grep -q '\[Finished\] strict Braid-vs-GraphECG' "$strict_log"; do
    sleep 30
done

if test -e "$output"; then
    echo "refusing to overwrite existing retry output: $output" >&2
    exit 1
fi

cd "$project"
PYTHONPATH=src "$python" -u scripts/paper07/train_paper07_full12_aux_mmd.py \
    --representations "$root/paper07_full12_aux_mmd/frozen_full12_contexts" \
    --targets "$root/paper07_full12_aux_mmd/full12_waveform_targets" \
    --output "$output" \
    --variant continuous_full12_aux_no_mmd \
    --batch 32 --max-epochs 100 --patience 10 --seed 42 --precision float32
