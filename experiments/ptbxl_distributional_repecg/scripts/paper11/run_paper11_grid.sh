#!/usr/bin/env bash
set -euo pipefail

repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
python=/home/mithunmanivannan/.venv/bin/python
representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/development_representations
ood_representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/ood_representations
output=$repo/experiments/ptbxl_distributional_repecg/outputs/paper11_causal_state_ecg

export PYTHONPATH="$repo/experiments/ptbxl_distributional_repecg/src"

echo "=== [START] paper11_causal_state_ecg: Training Grid ==="
for variant in "full" "linear_probe" "shuffled_futures"; do
    echo "Running variant: $variant"
    if [ -f "$output/cells/.done_${variant}" ]; then
        echo "Variant $variant already completed. Skipping."
        continue
    fi
    CUDA_VISIBLE_DEVICES=0 "$python" -u "$repo/experiments/ptbxl_distributional_repecg/scripts/paper11/train_paper11_shared_grid.py" \
    --representations "$representations" \
    --output "$output" \
    --batch 2048 \
    --max-epochs 100 \
    --patience 10 \
    --seed 42 \
        --variant "$variant"
    touch "$output/cells/.done_${variant}"
done

echo "=== [AGGREGATING] paper11_causal_state_ecg ==="
"$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper11/aggregate_paper11_grid.py" \
    --cells "$output/cells" \
    --output "$output" \
    --seed 42

echo "=== [EVALUATING OOD] paper11_causal_state_ecg across 9 datasets ==="
CUDA_VISIBLE_DEVICES=0 "$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper11/evaluate_paper11_ood.py" \
    --training "$output" \
    --representations "$ood_representations" \
    --output "$output/ood_evaluation"

echo "=== [FINISHED] paper11_causal_state_ecg ==="
