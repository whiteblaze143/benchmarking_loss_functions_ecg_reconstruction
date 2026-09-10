#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="download_heedb_afib"
PYTHON_BIN="/home/mithunmanivannan/.venv/bin/python"
SCRIPT_PATH="/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/scripts/download_heedb_afib_pipeline.py"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "[WARNING] Session '${SESSION_NAME}' already exists. Attaching view:"
  tmux capture-pane -pt "${SESSION_NAME}" | tail -n 20
  exit 0
fi

echo "[INFO] Launching HEEDB AFib download pipeline in detached tmux session '${SESSION_NAME}'..."
tmux new-session -d -s "${SESSION_NAME}" "${PYTHON_BIN} ${SCRIPT_PATH} --count 20000 --workers 12; read -p 'Finished. Press enter to exit.'"

echo "[SUCCESS] Launched session '${SESSION_NAME}'. Current status:"
tmux ls | grep "${SESSION_NAME}"
