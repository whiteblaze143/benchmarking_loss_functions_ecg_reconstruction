#!/bin/bash
# M0 Baseline Training Launch Script
# Run with: tmux new-session -d -s m0 './projects/benchmarking_loss_functions_ecg_reconstruction/scripts/launch_m0_training.sh'

set -e

PROJECT_DIR="/home/mithunmanivannan"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/m0_training_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

# Activate venv
if [ -f "${PROJECT_DIR}/ecg_fm_integration/venv/bin/activate" ]; then
    source "${PROJECT_DIR}/ecg_fm_integration/venv/bin/activate"
fi

echo "========================================" | tee -a "${LOG_FILE}"
echo "M0 Baseline Training - ${TIMESTAMP}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

nvidia-smi --query-gpu=name,memory.free --format=csv 2>/dev/null | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

cd "${PROJECT_DIR}"
echo "Starting M0 training at $(date)" | tee -a "${LOG_FILE}"

python -u "${SCRIPT_DIR}/train_m0_baseline.py" \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-3 \
    2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "M0 training finished at $(date)" | tee -a "${LOG_FILE}"
