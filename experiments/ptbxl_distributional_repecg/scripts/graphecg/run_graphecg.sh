#!/usr/bin/env bash
set -euo pipefail

REPO=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
PYTHON=/home/mithunmanivannan/.venv/bin/python
SCRIPT=$REPO/experiments/ptbxl_distributional_repecg/scripts/graphecg/train_graphecg_ptbxl.py
OUTPUT=$REPO/experiments/ptbxl_distributional_repecg/outputs/graphecg
mkdir -p "$OUTPUT"

export PYTHONPATH="$REPO/experiments/ptbxl_distributional_repecg/src"
export CUDA_VISIBLE_DEVICES=0

echo "[$(date '+%H:%M:%S')] Starting GraphECG PTB-XL Full Training..."
$PYTHON -u "$SCRIPT" \
    --batch-size 64 \
    --epochs 50 \
    --patience 10 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --sampling-rate 100 \
    --output-dir "$OUTPUT" \
    --seed 42

echo "[$(date '+%H:%M:%S')] GraphECG PTB-XL Training Complete."
