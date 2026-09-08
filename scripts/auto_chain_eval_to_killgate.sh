#!/usr/bin/env bash
# auto_chain_eval_to_killgate.sh
# Monitors eval_1lead_clinical_queue and immediately launches the ECG-AIM Kill-Gate training queue when eval completes.

set -u

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$_ROOT" || exit 1

LOG_FILE="$_ROOT/refine-logs/killgate/auto_chain_wait.log"
mkdir -p "$_ROOT/refine-logs/killgate"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-Chainer started: Monitoring eval_1lead_clinical_queue..." | tee -a "$LOG_FILE"

# Loop while eval_1lead_clinical_queue session exists and is active
while tmux has-session -t eval_1lead_clinical_queue 2>/dev/null; do
    # Check if python process is still running inside that session or if it ended
    if ! pgrep -f "run_1lead_clinical_eval_queue.py" >/dev/null 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_1lead_clinical_eval_queue.py is no longer active in process table." | tee -a "$LOG_FILE"
        break
    fi
    sleep 20
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] eval_1lead_clinical_queue has completed! Waiting 10 seconds for CUDA cleanup..." | tee -a "$LOG_FILE"
sleep 10

# Kill old dead ecgaim_killgate session if lingering
tmux kill-session -t ecgaim_killgate 2>/dev/null || true

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching ecgaim_killgate queue on CUDA..." | tee -a "$LOG_FILE"
tmux new-session -d -s ecgaim_killgate "cd '$_ROOT' && /home/mithunmanivannan/.venv/bin/python3 scripts/run_ecgaim_killgate_queue.py 2>&1 | tee refine-logs/killgate/queue_master.log; read"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Successfully launched ecgaim_killgate session! Verifying status:" | tee -a "$LOG_FILE"
tmux ls | tee -a "$LOG_FILE"
