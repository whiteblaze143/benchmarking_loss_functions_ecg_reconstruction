#!/usr/bin/env bash
# ============================================================================
# repECG Master Queue Orchestrator
# Executes all remaining papers (01, 03-15) sequentially once Paper 02 finishes.
# ============================================================================
set -euo pipefail

REPO="/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction"
SCRIPTS="$REPO/experiments/ptbxl_distributional_repecg/scripts"
LOGS_DIR="$REPO/experiments/ptbxl_distributional_repecg/outputs/queue_logs"
mkdir -p "$LOGS_DIR"

if [[ ! -f "$REPO/experiments/ptbxl_distributional_repecg/outputs/PRODUCTION_READY" ]]; then
    echo "Production blocked: strict smoke, mechanism coverage, dataset-label, and evaluation gates are not complete." >&2
    exit 1
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGS_DIR/master_queue.log"
}

log "================================================================="
log "repECG Master Suite Orchestrator Initialized"
log "================================================================="

# 1. Wait for Paper 02 pipeline if it is actively running in tmux
if tmux has-session -t repecg_p2_pipeline 2>/dev/null; then
    log "Detected active 'repecg_p2_pipeline' session. Waiting for Paper 02 completion..."
    while tmux has-session -t repecg_p2_pipeline 2>/dev/null; do
        sleep 30
    done
    log "Paper 02 pipeline has finished!"
else
    log "No active 'repecg_p2_pipeline' session detected. Proceeding immediately."
fi

# 2. Sequential execution list
PAPERS=(
    "paper01"
    "paper03"
    "paper04"
    "paper05"
    "paper06"
    "paper07"
    "paper08"
    "paper09"
    "paper10"
    "paper11"
    "paper12"
    "paper13"
    "paper14"
    "paper15"
)

TOTAL=${#PAPERS[@]}
INDEX=1

for paper in "${PAPERS[@]}"; do
    RUNNER="$SCRIPTS/$paper/run_${paper}_grid.sh"
    log "-----------------------------------------------------------------"
    log "[$INDEX/$TOTAL] Starting: $paper via $RUNNER"
    log "-----------------------------------------------------------------"
    
    if [ ! -f "$RUNNER" ]; then
        log "ERROR: Runner script $RUNNER not found! Stopping."
        exit 1
    fi
    
    START_TIME=$(date +%s)
    
    # Run pipeline and log output
    "$RUNNER" 2>&1 | tee "$LOGS_DIR/${paper}.log"
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    log "SUCCESS: $paper completed in ${ELAPSED}s"
    
    INDEX=$((INDEX + 1))
done

log "================================================================="
log "ALL 15 PAPERS IN repECG SUITE HAVE COMPLETED EXECUTION!"
log "================================================================="
