#!/usr/bin/env bash
# resume_queue.sh — Safely resumes the paused training process in the gpu_queue tmux session
set -euo pipefail

PIDS=$(pgrep -f "train_paper" || true)
if [ -z "$PIDS" ]; then
    echo "[resume_queue] No training process found matching 'train_paper'."
    exit 0
fi

for PID in $PIDS; do
    echo "[resume_queue] Sending SIGCONT to process $PID..."
    kill -CONT "$PID"
done
echo "[resume_queue] Active training queue resumed safely."
