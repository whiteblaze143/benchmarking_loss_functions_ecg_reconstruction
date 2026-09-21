#!/usr/bin/env bash
# run_lean_queue.sh — Lean, scientifically-motivated queue.
# Replaces run_all_papers_queue.sh. Only runs cells that answer research claims.
# Stage 0 (eval of P01-P07) is already done — all fold8 JSONs exist.
# Stage 1: P02 no_circular (only missing baseline variant, ~25 min)
# Stage 2: P08 targeted (8 key cells, ~2 hr)
# Stage 3: P10, P11, P13, P15 (claims papers, ~4-5 hr total)
set -euo pipefail

REPO=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
PYTHON=/home/mithunmanivannan/.venv/bin/python
SCRIPTS=$REPO/experiments/ptbxl_distributional_repecg/scripts
OUTPUTS=$REPO/experiments/ptbxl_distributional_repecg/outputs
LOGDIR=$OUTPUTS/queue_logs
mkdir -p "$LOGDIR"

export PYTHONPATH="$REPO/experiments/ptbxl_distributional_repecg/src"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=7

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGDIR/lean_queue.log"; }

run_paper() {
  local name=$1
  local script=$2
  log "=== START $name ==="
  set +e
  bash "$script" 2>&1 | tee "$LOGDIR/${name}.log"
  local code=${PIPESTATUS[0]}
  set -e
  if [ "$code" -eq 0 ]; then
    log "=== DONE  $name (exit 0) ==="
  else
    log "=== FAIL  $name (exit $code) — continuing ==="
  fi
}

log "Lean queue starting. GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
log "Stage 0: P01-P07 fold8 eval already complete — skipping."

# ── Stage 1: P02 final missing variant ────────────────────────────────────────
# .done_ guards in /data skip full/linear_kernel/linear_probe — only no_circular runs
log "Stage 1: P02 no_circular (final missing baseline variant)"
run_paper "paper02_nocircular" "$SCRIPTS/paper02/run_paper02_grid.sh"

# ── Stage 2: P08 targeted (8 cells, not 100+) ─────────────────────────────────
log "Stage 2: P08 targeted cells (routing × representation factorial)"
run_paper "paper08_full_capped" "$SCRIPTS/paper08/run_paper08_full_capped.sh"

# ── Stage 3: Claims papers ─────────────────────────────────────────────────────
log "Stage 3: Pending claims papers"
run_paper "paper10" "$SCRIPTS/paper10/run_paper10_grid.sh"
run_paper "paper11" "$SCRIPTS/paper11/run_paper11_grid.sh"
# P12 BLOCKED — V2 G3-real FAIL; legacy grid retired
# run_paper "paper12" "$SCRIPTS/paper12/run_paper12_grid.sh"
run_paper "paper13" "$SCRIPTS/paper13/run_paper13_grid.sh"
run_paper "paper15" "$SCRIPTS/paper15/run_paper15_grid.sh"

# ── Stage 4: Post-queue fold8 eval for newly trained papers ───────────────────
log "Stage 4: Fold 8 evaluation for P02, P08, P10, P11, P13, P15"
for paper_id in 2 8 10 11 13 15; do
  log "  Evaluating Paper $paper_id on Fold 8..."
  $PYTHON "$SCRIPTS/evaluate_all_papers_fold8.py" --paper "$paper_id" 2>&1 | tee -a "$LOGDIR/fold8_eval.log" || \
    log "  Paper $paper_id eval failed — check fold8_eval.log"
done

# ── Stage 5: GraphECG PTB-XL Training ──────────────────────────────────────────
log "Stage 5: GraphECG (Ansari et al., 2026) PTB-XL training"
run_paper "graphecg" "$SCRIPTS/graphecg/run_graphecg.sh"

# ── Stage 6: Tier 4 Acquisition-Configuration Shift Evaluation ────────────────
log "Stage 6: Tier 4 Acquisition-Configuration Shift Battery (Clinical + 255 Combinatorial Subsets)"
$PYTHON "$SCRIPTS/evaluation/evaluate_tier4_shift.py" \
    --model graphecg \
    --checkpoint "$OUTPUTS/graphecg/graphecg_ptbxl_best.pt" \
    --output-dir "$OUTPUTS/tier4_configuration_shift" 2>&1 | tee -a "$LOGDIR/tier4_eval.log" || \
    log "  Tier 4 GraphECG eval failed — check tier4_eval.log"

log "=== LEAN QUEUE COMPLETE ==="
log "Results: $OUTPUTS/cross_paper_evaluation/ and $OUTPUTS/tier4_configuration_shift/"
