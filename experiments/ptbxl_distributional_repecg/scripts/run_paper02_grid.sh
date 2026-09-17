#!/usr/bin/env bash
set -euo pipefail

repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
python=/home/mithunmanivannan/.venv/bin/python
representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/development_representations
output=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/development_training
CUDA_VISIBLE_DEVICES=0 "$python" -u "$repo/experiments/ptbxl_distributional_repecg/scripts/train_paper02_shared_grid.py" \
    --representations "$representations" \
    --output "$output" \
    --batch 2048 \
    --max-epochs 100 \
    --patience 10 \
    --seed 42

"$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/aggregate_paper02_grid.py" \
    --cells "$output/cells" \
    --output "$output" \
    --seed 42
