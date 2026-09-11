#!/usr/bin/env bash
set -euo pipefail

project_root=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
queue_log="$project_root/refine-logs/predictable_residual_subspace/queued_after_grail.log"
grail_pid=${1:?usage: queue_predictable_after_grail.sh GRAIL_PID}

cd "$project_root"
echo "QUEUED $(date -Ins) waiting for GRAIL" | tee -a "$queue_log"

while true; do
  if ! kill -0 "$grail_pid" 2>/dev/null; then
    break
  fi
  sleep 30
done

echo "STARTING $(date -Ins) GRAIL exited; resuming predictable-residual queue" | tee -a "$queue_log"
exec bash "$project_root/scripts/run_predictable_residual_queue.sh" 2>&1 | tee -a "$queue_log"
