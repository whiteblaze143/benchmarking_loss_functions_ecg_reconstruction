#!/usr/bin/env bash
# ==============================================================================
# Master Overnight Runner for SetOperator CVPR 2027 Pipeline
# ==============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
PYTHON_BIN="/home/mithunmanivannan/.venv/bin/python"

LOG_DIR="/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/r028_overnight_${TIMESTAMP}.log"

echo "==================================================================" | tee -a "${LOG_FILE}"
echo "LAUNCHING R028 VERIFIED OVERNIGHT PIPELINE: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" | tee -a "${LOG_FILE}"
echo "PID: $$" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"

# Trap errors cleanly
trap 'echo "CRITICAL FAILURE: Pipeline exited with code $? at line $LINENO on $(date -u +'%Y-%m-%dT%H:%M:%SZ')" | tee -a "${LOG_FILE}"' ERR

cd "${REPO_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_r028_verified_overnight.py" \
    --device cuda \
    --poll-interval 60 \
    2>&1 | tee -a "${LOG_FILE}"

echo "==================================================================" | tee -a "${LOG_FILE}"
echo "PIPELINE TERMINATED SUCCESSFULLY: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"
