#!/bin/bash
export WANDB_API_KEY="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"
PROJECT_DIR="/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction"
RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
LOCAL_RUN_DIR="$PROJECT_DIR/experiment_queue/$RUN_TS"
mkdir -p "$LOCAL_RUN_DIR"
cp "$PROJECT_DIR/refine-logs/ablation_manifest.json" "$LOCAL_RUN_DIR/manifest.json"

echo "Staged ablation grid in $LOCAL_RUN_DIR/manifest.json"

# Launch the experiment-queue scheduler inside a tmux session
tmux new-session -d -s ablation_queue "WANDB_API_KEY=${WANDB_API_KEY} python3 .agents/skills/experiment-queue/scripts/queue_manager.py --manifest $LOCAL_RUN_DIR/manifest.json --state $LOCAL_RUN_DIR/queue_state.json --log-dir $LOCAL_RUN_DIR/logs; read"

echo "Queue manager started in tmux session 'ablation_queue'. Attach with: tmux attach -t ablation_queue"
