#!/bin/bash
set -e

PROJECT_DIR="/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction"
RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
LOCAL_RUN_DIR="$PROJECT_DIR/experiment_queue/$RUN_TS"
mkdir -p "$LOCAL_RUN_DIR"

ARIS_REPO="/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction"
QUEUE_TOOLS="$ARIS_REPO/.agents/skills/experiment-queue/scripts"
REMOTE_RUN_REL=".aris_queue/runs/$RUN_TS"
REMOTE_RUN_DIR="\$HOME/$REMOTE_RUN_REL"

echo "Creating remote directories..."
ssh carleton-vm "mkdir -p \"$REMOTE_RUN_DIR/logs\" \"\$HOME/.aris_queue\""

echo "Copying scripts and manifest..."
scp "$QUEUE_TOOLS/queue_manager.py" "$QUEUE_TOOLS/build_manifest.py" carleton-vm:.aris_queue/
scp "$PROJECT_DIR/refine-logs/vcg_manifest.json" carleton-vm:"$REMOTE_RUN_REL/manifest.json"

echo "Launching queue manager..."
ssh carleton-vm "nohup python3 \"\$HOME/.aris_queue/queue_manager.py\" \
  --manifest \"$REMOTE_RUN_DIR/manifest.json\" \
  --state    \"$REMOTE_RUN_DIR/queue_state.json\" \
  --log-dir  \"$REMOTE_RUN_DIR/logs\" \
  > \"$REMOTE_RUN_DIR/queue_mgr.log\" 2>&1 &"

{
  printf 'PROJECT_DIR=%q\n'    "$PROJECT_DIR"
  printf 'RUN_TS=%q\n'         "$RUN_TS"
  printf 'LOCAL_RUN_DIR=%q\n'  "$LOCAL_RUN_DIR"
  printf 'REMOTE_RUN_REL=%q\n' "$REMOTE_RUN_REL"
  printf 'REMOTE_RUN_DIR=%q\n' "$REMOTE_RUN_DIR"
} > "$LOCAL_RUN_DIR/run_meta.txt"

echo "Launched successfully! RUN_TS is $RUN_TS"
