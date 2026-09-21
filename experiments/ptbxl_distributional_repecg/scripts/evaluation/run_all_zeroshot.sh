#!/usr/bin/env bash
# run_all_zeroshot.sh — Evaluates all 15 papers + GraphECG + Baselines across external datasets
set -euo pipefail

REPO=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
PYTHON=/home/mithunmanivannan/.venv/bin/python
SCRIPT=$REPO/experiments/ptbxl_distributional_repecg/scripts/evaluation/evaluate_all_datasets_zeroshot.py
OUTDIR=$REPO/experiments/ptbxl_distributional_repecg/outputs/external_dataset_evaluations

export PYTHONPATH="$REPO/experiments/ptbxl_distributional_repecg/src:$REPO/tit_ecg/src"
export CUDA_VISIBLE_DEVICES=0

echo "[$(date '+%H:%M:%S')] Starting Multi-Dataset Zero-Shot Evaluation Suite..."
$PYTHON "$SCRIPT" --model all --dataset all --output-dir "$OUTDIR" 2>&1 | tee "$OUTDIR/zeroshot_full_sweep.log"
echo "[$(date '+%H:%M:%S')] Multi-Dataset Zero-Shot Evaluation Complete."
