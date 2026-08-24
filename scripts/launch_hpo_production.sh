#!/bin/bash
# Production HPO Launch Script
# Run with: tmux new-session -d -s hpo './scripts/launch_hpo_production.sh'

set -e

# Configuration
N_TRIALS=100
EPOCHS=50
WARM_START=""  # Add "--warm-start" to enable

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="/home/mithunmanivannan"
LOG_DIR="${PROJECT_DIR}/logs"
RESULTS_DIR="${PROJECT_DIR}/results"

# Create directories
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/hpo_production_${TIMESTAMP}.log"

# Activate virtual environment if exists
if [ -f "${PROJECT_DIR}/ecg_fm_integration/venv/bin/activate" ]; then
    source "${PROJECT_DIR}/ecg_fm_integration/venv/bin/activate"
fi

# Log system info
echo "========================================" | tee -a "${LOG_FILE}"
echo "HPO Production Run - ${TIMESTAMP}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "N_TRIALS: ${N_TRIALS}" | tee -a "${LOG_FILE}"
echo "EPOCHS: ${EPOCHS}" | tee -a "${LOG_FILE}"
echo "WARM_START: ${WARM_START:-disabled}" | tee -a "${LOG_FILE}"
echo "LOG_FILE: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null | tee -a "${LOG_FILE}" || echo "No GPU info available"
echo "" | tee -a "${LOG_FILE}"

# Run HPO
cd "${PROJECT_DIR}"
echo "Starting HPO at $(date)" | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"

python -u scripts/m1_multiobj_hpo_v2_rigorous.py \
    --n-trials ${N_TRIALS} \
    --epochs ${EPOCHS} \
    ${WARM_START} \
    2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=$?

echo "" | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"
echo "HPO finished at $(date) with exit code ${EXIT_CODE}" | tee -a "${LOG_FILE}"

# Summary
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS: Results saved to ${RESULTS_DIR}" | tee -a "${LOG_FILE}"
else
    echo "FAILED: Check ${LOG_FILE} for errors" | tee -a "${LOG_FILE}"
fi
