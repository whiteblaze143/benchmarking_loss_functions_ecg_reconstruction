#!/bin/bash
# Extend HPO by 100 more trials (run after first 100 complete)
# Optuna will automatically resume from existing database

set -e

PROJECT_DIR="/home/mithunmanivannan"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/hpo_extension_${TIMESTAMP}.log"

# Activate venv
source "${PROJECT_DIR}/ecg_fm_integration/venv/bin/activate"

echo "========================================" | tee -a "${LOG_FILE}"
echo "HPO Extension Run - ${TIMESTAMP}" | tee -a "${LOG_FILE}"
echo "Adding 100 more trials to existing study" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# Check current trial count
COMPLETED=$(sqlite3 "${PROJECT_DIR}/m1_hpo_v2_rigorous.db" "SELECT COUNT(*) FROM trials WHERE state='COMPLETE'" 2>/dev/null || echo "0")
echo "Completed trials before extension: ${COMPLETED}" | tee -a "${LOG_FILE}"

cd "${PROJECT_DIR}"
echo "Starting extension at $(date)" | tee -a "${LOG_FILE}"

python -u scripts/m1_multiobj_hpo_v2_rigorous.py \
    --n-trials 100 \
    --epochs 50 \
    2>&1 | tee -a "${LOG_FILE}"

echo "Extension finished at $(date)" | tee -a "${LOG_FILE}"

# Final count
FINAL=$(sqlite3 "${PROJECT_DIR}/m1_hpo_v2_rigorous.db" "SELECT COUNT(*) FROM trials WHERE state='COMPLETE'" 2>/dev/null || echo "?")
echo "Total completed trials: ${FINAL}" | tee -a "${LOG_FILE}"
