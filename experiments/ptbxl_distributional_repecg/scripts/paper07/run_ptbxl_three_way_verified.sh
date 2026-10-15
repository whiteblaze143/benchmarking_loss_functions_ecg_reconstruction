#!/usr/bin/env bash
# ==============================================================================
# run_ptbxl_three_way_verified.sh
# Authoritative persistent launcher for the PTB-XL Three-Way Model Comparison
# (SetOperator, SetONet, GraphECG)
# ==============================================================================

set -euo pipefail

REPO_DIR="/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction"
EXPERIMENT_DIR="${REPO_DIR}/experiments/ptbxl_distributional_repecg"
PYTHON_BIN="/home/mithunmanivannan/.venv/bin/python"
LOG_DIR="/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/ptbxl_three_way_v1/logs"

mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

echo "=================================================================="
echo "Launching PTB-XL Three-Way Comparison Master Pipeline"
echo "Log file: ${LOG_FILE}"
echo "=================================================================="

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

cd "${REPO_DIR}"

exec "${PYTHON_BIN}" -u "${EXPERIMENT_DIR}/scripts/paper07/run_ptbxl_three_way_verified.py" "$@" 2>&1 | tee -a "${LOG_FILE}"
