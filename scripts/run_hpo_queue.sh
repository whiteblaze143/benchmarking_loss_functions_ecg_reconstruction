#!/usr/bin/env bash
# HPO Queue: Runs ECGFounder HPO after ECG-FM HPO finishes,
# then launches all 3 SOTA champion training runs sequentially.
set -e

PYTHON=/home/mithunmanivannan/.venv/bin/python3
LOGDIR=/home/mithunmanivannan/logs
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "[$(date)] Waiting for ECG-FM HPO to finish..."
# Wait for the ecgfm HPO process to exit
while pgrep -f "run_optuna_components.*ecgfm" > /dev/null 2>&1; do
    sleep 60
done
echo "[$(date)] ECG-FM HPO done."

echo "[$(date)] Starting ECGFounder HPO..."
$PYTHON "${SCRIPT_DIR}/run_optuna_components.py" \
    --backbone ecgfounder --trials 50 \
    --study_name loss_sweep_v1_ecgfounder --epochs 50 \
    >> "$LOGDIR/hpo_phase9_ecgfounder.log" 2>&1
echo "[$(date)] ECGFounder HPO done."

echo "[$(date)] All HPOs complete. Ready for SOTA champion training."
