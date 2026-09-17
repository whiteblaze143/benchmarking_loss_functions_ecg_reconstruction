#!/usr/bin/env bash
set -euo pipefail

repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
python=/home/mithunmanivannan/.venv/bin/python
representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/development_representations
output=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/development_training
ood_representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/ood_representations
for variant in "full" "linear" "no_circular"; do
    echo "Running variant: $ablation"
    CUDA_VISIBLE_DEVICES=0 "$python" -u "$repo/experiments/ptbxl_distributional_repecg/scripts/paper02/train_paper02_shared_grid.py" \
    --representations "$representations" \
    --output "$output" \
    --batch 2048 \
    --max-epochs 100 \
    --patience 10 \
    --seed 42 \
        --variant "$variant"
done
"$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper02/aggregate_paper02_grid.py" \
    --cells "$output/cells" \
    --output "$output" \
    --seed 42

"$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper02/build_paper02_ood_representations.py" \
    --kernel-fit "$representations/kernel_fit.npz" \
    --scaler "$repo/experiments/ptbxl_distributional_repecg/outputs/prepare_smoke/scaler.json" \
    --output "$ood_representations" \
    --seed 42

CUDA_VISIBLE_DEVICES=0 "$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper02/evaluate_paper02_ood.py" \
    --training "$output" \
    --representations "$ood_representations" \
    --output "$output/ood_evaluation"
