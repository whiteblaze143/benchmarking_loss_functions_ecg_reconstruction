#!/usr/bin/env bash
# pause_queue.sh — Safely pauses the active training process in the gpu_queue tmux session
set -euo pipefail

PIDS=$(pgrep -f "train_paper" || true)
if [ -z "$PIDS" ]; then
    echo "[pause_queue] No active training process found matching 'train_paper'."
    exit 0
fi

for PID in $PIDS; do
    echo "[pause_queue] Sending SIGSTOP to process $PID..."
    kill -STOP "$PID"
done
echo "[pause_queue] Active training queue paused safely. GPU compute released."
