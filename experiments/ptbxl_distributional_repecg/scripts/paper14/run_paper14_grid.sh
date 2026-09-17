#!/usr/bin/env bash
set -euo pipefail

repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
python=/home/mithunmanivannan/.venv/bin/python
representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/development_representations
ood_representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/ood_representations
output=$repo/experiments/ptbxl_distributional_repecg/outputs/paper14_invariant_mechanism

export PYTHONPATH="$repo/experiments/ptbxl_distributional_repecg/src"

echo "=== [START] paper14_invariant_mechanism: Training Grid ==="
CUDA_VISIBLE_DEVICES=0 "$python" -u "$repo/experiments/ptbxl_distributional_repecg/scripts/paper14/train_paper14_shared_grid.py" \
    --representations "$representations" \
    --output "$output" \
    --batch 2048 \
    --max-epochs 100 \
    --patience 10 \
    --seed 42

echo "=== [AGGREGATING] paper14_invariant_mechanism ==="
"$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper14/aggregate_paper14_grid.py" \
    --cells "$output/cells" \
    --output "$output" \
    --seed 42

echo "=== [EVALUATING OOD] paper14_invariant_mechanism across 9 datasets ==="
CUDA_VISIBLE_DEVICES=0 "$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper14/evaluate_paper14_ood.py" \
    --training "$output" \
    --representations "$ood_representations" \
    --output "$output/ood_evaluation"

echo "=== [FINISHED] paper14_invariant_mechanism ==="
