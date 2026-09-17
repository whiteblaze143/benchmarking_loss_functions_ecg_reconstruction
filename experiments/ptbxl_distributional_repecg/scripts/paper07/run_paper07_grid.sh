#!/usr/bin/env bash
set -euo pipefail

repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
python=/home/mithunmanivannan/.venv/bin/python
representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/development_representations
ood_representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/ood_representations
output=$repo/experiments/ptbxl_distributional_repecg/outputs/paper07_operator_reconstruction

export PYTHONPATH="$repo/experiments/ptbxl_distributional_repecg/src"

echo "=== [START] paper07_operator_reconstruction: Training Grid ==="
for variant in "full" "linear_probe" "learned_lead_id"; do
    echo "Running variant: $variant"
    if [ -f "$output/cells/variant_${variant}/done" ]; then
        echo "Variant $variant already completed. Skipping."
        continue
    fi
    CUDA_VISIBLE_DEVICES=0 "$python" -u "$repo/experiments/ptbxl_distributional_repecg/scripts/paper07/train_paper07_shared_grid.py" \
    --representations "$representations" \
    --output "$output" \
    --batch 2048 \
    --max-epochs 100 \
    --patience 10 \
    --seed 42 \
        --variant "$variant"
    touch "$output/cells/variant_${variant}/done"
done

echo "=== [AGGREGATING] paper07_operator_reconstruction ==="
"$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper07/aggregate_paper07_grid.py" \
    --cells "$output/cells" \
    --output "$output" \
    --seed 42

echo "=== [EVALUATING OOD] paper07_operator_reconstruction across 9 datasets ==="
CUDA_VISIBLE_DEVICES=0 "$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper07/evaluate_paper07_ood.py" \
    --training "$output" \
    --representations "$ood_representations" \
    --output "$output/ood_evaluation"

echo "=== [FINISHED] paper07_operator_reconstruction ==="
