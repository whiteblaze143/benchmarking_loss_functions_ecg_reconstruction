#!/usr/bin/env bash
set -euo pipefail

repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
python=/home/mithunmanivannan/.venv/bin/python
representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper14_representations/development_representations
output=$repo/experiments/ptbxl_distributional_repecg/outputs/paper14_invariant_mechanism

export PYTHONPATH="$repo/experiments/ptbxl_distributional_repecg/src"

mkdir -p "$output/cells"

echo "=== [START] paper14_invariant_mechanism: Real-Data Training Grid ==="
for variant in "erm" "mmd_lambda_0.1" "mmd_lambda_0.3" "mmd_lambda_1.0" "mmd_lambda_3.0" "mmd_lambda_10.0"; do
    echo "Running variant: $variant"
    if [ -f "$output/cells/.done_${variant}" ]; then
        echo "Variant $variant already completed. Skipping."
        continue
    fi
    CUDA_VISIBLE_DEVICES=0 "$python" -u "$repo/experiments/ptbxl_distributional_repecg/scripts/paper14/train_paper14_shared_grid.py" \
        --representations "$representations" \
        --output "$output" \
        --batch 2048 \
        --max-epochs 100 \
        --patience 10 \
        --seed 42 \
        --variant "$variant"
    touch "$output/cells/.done_${variant}"
done

echo "=== [AGGREGATING] paper14_invariant_mechanism ==="
"$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper14/aggregate_paper14_grid.py" \
    --cells "$output/cells" \
    --output "$output" \
    --seed 42

echo "=== [EVALUATING REAL-DATA ENDPOINTS] paper14_invariant_mechanism ==="
CUDA_VISIBLE_DEVICES=0 "$python" -u "$repo/experiments/ptbxl_distributional_repecg/scripts/paper14/evaluate_paper14_real.py" \
    --training "$output" \
    --representations "$representations" \
    --output "$output/real_evaluation" \
    --seed 42

echo "=== [FINISHED] paper14_invariant_mechanism real-data pipeline complete ==="
