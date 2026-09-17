#!/usr/bin/env bash
set -euo pipefail

REPO="/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction"
SCRIPTS="$REPO/experiments/ptbxl_distributional_repecg/scripts"
LOGS_DIR="$REPO/experiments/ptbxl_distributional_repecg/outputs/queue_logs"
mkdir -p "$LOGS_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGS_DIR/smoke_master_queue.log"
}

log "================================================================="
log "repECG Master Suite Orchestrator Initialized (SMOKE TEST)"
log "================================================================="

# Create a temporary runner that only does 1 epoch of training and small eval
TMP_RUNNER="/tmp/smoke_run_paper01_grid.sh"
cat << 'INNER_EOF' > "$TMP_RUNNER"
#!/usr/bin/env bash
set -euo pipefail
repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
python=/home/mithunmanivannan/.venv/bin/python
representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/development_representations
ood_representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/ood_representations
output=$repo/experiments/ptbxl_distributional_repecg/outputs/smoke_paper01

CUDA_VISIBLE_DEVICES=0 "$python" -u "$repo/experiments/ptbxl_distributional_repecg/scripts/paper01/train_paper01_shared_grid.py" \
    --representations "$representations" \
    --output "$output" \
    --batch 2048 \
    --max-epochs 1 \
    --patience 1 \
    --seed 42

"$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper01/aggregate_paper01_grid.py" \
    --cells "$output/cells" \
    --output "$output" \
    --seed 42

CUDA_VISIBLE_DEVICES=0 "$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper01/evaluate_paper01_ood.py" \
    --training "$output" \
    --representations "$ood_representations" \
    --output "$output/ood_evaluation"
INNER_EOF
chmod +x "$TMP_RUNNER"

PAPERS=("paper01")
TOTAL=${#PAPERS[@]}
INDEX=1

for paper in "${PAPERS[@]}"; do
    RUNNER="$TMP_RUNNER"
    log "-----------------------------------------------------------------"
    log "[$INDEX/$TOTAL] Starting SMOKE TEST for $paper via $RUNNER"
    log "-----------------------------------------------------------------"
    
    START_TIME=$(date +%s)
    
    if "$RUNNER" 2>&1 | tee "$LOGS_DIR/smoke_${paper}.log"; then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        log "SUCCESS: $paper completed in ${ELAPSED}s"
    else
        log "FAILURE: $paper failed! Check log: $LOGS_DIR/smoke_${paper}.log"
    fi
    INDEX=$((INDEX + 1))
done
log "================================================================="
log "SMOKE TEST COMPLETED SUCCESSFULLY!"
log "================================================================="
