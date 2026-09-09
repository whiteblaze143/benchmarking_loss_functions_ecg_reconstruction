#!/usr/bin/env bash
# ==============================================================================
# Round-2 Adaptive VCG & MMD Ablation Queue (11 Configurations)
# Evaluates whether Homoscedastic Uncertainty Weighting (Adaptive Composite Loss)
# resolves prior gradient competition between point-wise correlation and
# higher-order biophysical geometry (Kors VCG loops) & distribution alignment (MMD).
# Backbone: LCT-MTL Champion Lean Base (enc4, dec4, width384, heads16, patch10, boundary0.2, nodice, no_lead_dropout, zscore)
# Evaluator: scripts/evaluate_killgate_bootstrap.py
# ==============================================================================

set -euo pipefail

BASE="/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction"
PY="/home/mithunmanivannan/.venv/bin/python3"
TRAIN="$BASE/scripts/train_1lead_wavelet_ssl_mtl.py"
EVAL="$BASE/scripts/evaluate_killgate_bootstrap.py"
OUT="$BASE/refine-logs/lean_abl2/runs"
LEDGER="$BASE/refine-logs/lean_abl2/summary_ledger.json"
LOG="$BASE/refine-logs/lean_abl2/queue_master.log"
LEAN_ANCHOR="$OUT/lean2_L1_clean_lean_best_s42_l0"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ADAPTIVE-VCG-MMD] $*" | tee -a "$LOG"; }

update_ledger() {
  local cell_id="$1"
  local run_name="$2"
  local status="$3"
  local eval_file="$4"

  "$PY" -c "
import json, sys, datetime
ledger_file = '$LEDGER'
eval_file = '$eval_file'
try:
    with open(ledger_file) as f:
        ledger = json.load(f)
except Exception:
    ledger = {}

eval_data = None
try:
    with open(eval_file) as f:
        eval_data = json.load(f)
except Exception:
    pass

st = '$status'
if eval_data and 'gates' in eval_data:
    verdict = 'PASS' if eval_data['gates'].get('pass_overall') else 'FAIL'
    if '$status'.startswith('EARLY_STOPPED'):
        st = f'EARLY_STOPPED ({verdict})'
    else:
        st = f'COMPLETED ({verdict})'

ledger['$cell_id'] = {
    'run_name': '$run_name',
    'status': st,
    'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'eval': eval_data
}

with open(ledger_file, 'w') as f:
    json.dump(ledger, f, indent=2)
"
}

run_adaptive_cell() {
  local cell_id="$1"
  local desc="$2"
  shift 2
  local run_name="lean2_${cell_id}_s42_l0"
  local rdir="$OUT/$run_name"

  # Skip if already successfully completed
  if [ -f "$rdir/_SUCCESS.json" ] && [ -f "$rdir/killgate_eval_val.json" ]; then
    log "Skipping $cell_id ($run_name): already completed."
    return 0
  fi

  log "======================================================================"
  log "STARTING ADAPTIVE VCG/MMD CELL $cell_id: $run_name"
  log "Description: $desc"
  log "Anchor: $LEAN_ANCHOR"
  mkdir -p "$rdir"

  # Base Lean Convolutional-Transformer Multi-Task Synthesizer (LCT-MTL) Base
  # Incorporates Round-2 Empirical Takeaways:
  # - Patch size 10 (20 ms at 500 Hz): +0.0083 gain, peak P05 tail robustness 0.4208
  # - 16 Attention Heads: +0.0023 gain, finer cross-lead spatial subspace projections
  # - Width 384: Optimal efficiency knee point, cuts parameters by 43.7% with only -0.0015 delta
  # - Auxiliary Delineation: Boundary transition penalty (weight 0.2) + CE; Soft Dice pruned (0.0)
  # - Loss Weighting: Homoscedastic Adaptive Uncertainty (LE2: adaptive_composite, +0.0054 gain)
  # - Spatial Input: Dedicated Lead I (no whole-lead dropout: +0.0097 gain)
  local cmd=(
    "$PY" "$TRAIN"
    --run-name "$run_name"
    --output-dir "$rdir"
    --observed-leads 0
    --epochs 15
    --batch-size 32
    --lr 0.0001
    --max-lr 0.0005
    --weight-decay 0.0001
    --pct-start 0.2
    --delineation-every 2
    --delineation-dir "$BASE/data/rdb_wavelet_delineation_cache"
    --no-use-wavelet-branch
    --ssl-mode none
    --width 384
    --heads 16
    --encoder-depth 4
    --decoder-depth 4
    --patch-size 10
    --artificial-mask-mode no_lead_dropout
    --zscore-norm
    --reconstruction-loss-type adaptive_composite
    --early-stop-epoch 10
    --early-stop-min-pearson 0.7471
    --seg-ce-weight 1.0
    --dice-weight 0.0
    --boundary-weight 0.2
    --seed 42
    --checkpoint-policy best
    --rolling-resume
    "$@"
  )

  log "Executing command (early-stop gate at epoch 10: min_r=0.7471)..."
  "${cmd[@]}" 2>&1 | tee "$rdir/train.log"
  local rc=${PIPESTATUS[0]}

  if [ $rc -ne 0 ] || [ ! -f "$rdir/_SUCCESS.json" ]; then
    log "ERROR: $run_name failed (rc=$rc). Check $rdir/train.log"
    update_ledger "$cell_id" "$run_name" "FAILED (code $rc)" ""
    return 1
  fi

  local completion_status="COMPLETED"
  if grep -q '"early_stopped": true' "$rdir/_SUCCESS.json"; then
    completion_status="EARLY_STOPPED"
    log "[EARLY-STOP-GATE] $run_name did not reach Clean Lean Core benchmark (0.7471) by epoch 10. Terminated early."
  fi

  log "Training step ended ($completion_status). Running paired bootstrap evaluation..."
  local eval_file="$rdir/killgate_eval_val.json"
  if [ ! -f "$eval_file" ]; then
    "$PY" "$EVAL" \
      --candidate-dir "$rdir" \
      --anchor-dir "$LEAN_ANCHOR" \
      --split val \
      --n-boot 5000 \
      --batch-size 32 2>&1 | tee -a "$LOG" || log "WARNING: Bootstrap evaluation failed for $run_name (continuing queue)."
  fi

  update_ledger "$cell_id" "$run_name" "$completion_status" "$eval_file"
  log "Done with $cell_id."
  log "======================================================================"
}

log "Starting Adaptive VCG & MMD Ablation Suite (10 Top Configurations)..."

# ==============================================================================
# Group 1: Pure 3D Vectorcardiographic (VCG) Loops with Adaptive Weighting
# ==============================================================================
# Cell 1: MSE + Pearson + VCG (Mask 1101000)
# Historical benchmark: r = 0.8026, AUROC = 0.8588 (Top-ranked VCG standalone)
run_adaptive_cell "AV1_vcg_adaptive" \
  "Adaptive VCG: MSE + Pearson + Kors 3D VCG loop alignment (mask 1101000)" \
  --factorial-mask 1101000

# Cell 2: MSE + Pearson + Deriv + VCG (Mask 1111000)
# Full biophysical quad combining temporal dynamics and 3D dipole orientation
run_adaptive_cell "AV2_triplet_vcg_adaptive" \
  "Adaptive Triplet+VCG: MSE + Pearson + 1st Deriv + Kors VCG (mask 1111000)" \
  --factorial-mask 1111000

# ==============================================================================
# Group 2: Top Maximum Mean Discrepancy (MMD) Distribution Kernels with Adaptive Weighting
# ==============================================================================
# Cell 3: MSE + Pearson + Anatomical IMQ MMD (Mask 1100003)
# Historical benchmark: r = 0.8598, AUROC = 0.8594 (#1 overall correlation winner in early sweep)
run_adaptive_cell "AM1_mmd_imq_adaptive" \
  "Adaptive MMD: MSE + Pearson + Anatomical Block Multiscale IMQ Kernel (mask 1100003)" \
  --factorial-mask 1100003

# Cell 4: MSE + Pearson + Temporal K-Means IMQ MMD (Mask 1100004)
# Historical benchmark: r = 0.8594, AUROC = 0.8593 (#2 overall correlation winner)
run_adaptive_cell "AM2_mmd_kmeans_adaptive" \
  "Adaptive MMD: MSE + Pearson + Temporal K-Means Dynamic Phase IMQ Kernel (mask 1100004)" \
  --factorial-mask 1100004

# Cell 5: MSE + Pearson + Anatomical Laplacian MMD (Mask 1100002)
# Historical benchmark: r = 0.8340, AUROC = 0.8483 (Heavy-tailed spatial regularizer)
run_adaptive_cell "AM3_mmd_laplace_adaptive" \
  "Adaptive MMD: MSE + Pearson + Anatomical Block Laplacian Kernel (mask 1100002)" \
  --factorial-mask 1100002

# ==============================================================================
# Group 3: Joint VCG + MMD Synergistic Couplings with Adaptive Weighting
# ==============================================================================
# Cell 6: MSE + Pearson + VCG + Anatomical IMQ MMD (Mask 1101003)
# Historical benchmark: r = 0.7990, AUROC = 0.8582
run_adaptive_cell "AVM1_vcg_mmd_imq_adaptive" \
  "Adaptive VCG+MMD: MSE + Pearson + VCG + Anatomical IMQ Kernel (mask 1101003)" \
  --factorial-mask 1101003

# Cell 7: MSE + Pearson + VCG + Temporal K-Means MMD (Mask 1101004)
# Historical benchmark: r = 0.7968, AUROC = 0.8573
run_adaptive_cell "AVM2_vcg_mmd_kmeans_adaptive" \
  "Adaptive VCG+MMD: MSE + Pearson + VCG + Temporal K-Means Kernel (mask 1101004)" \
  --factorial-mask 1101004

# Cell 8: MSE + Pearson + VCG + Anatomical Laplacian MMD (Mask 1101002)
# Historical benchmark: r = 0.7804, AUROC = 0.8488
run_adaptive_cell "AVM3_vcg_mmd_laplace_adaptive" \
  "Adaptive VCG+MMD: MSE + Pearson + VCG + Anatomical Laplacian Kernel (mask 1101002)" \
  --factorial-mask 1101002

# ==============================================================================
# Group 4: Full Biophysical Pentad & Physical Constraint Probes
# ==============================================================================
# Cell 9: MSE + Pearson + Deriv + VCG + Anatomical Laplacian MMD (Mask 1111002)
# The canonical 1111002 probe tested in earlier 8-job architecture search
run_adaptive_cell "AVM4_full_probe_laplace_adaptive" \
  "Adaptive Full Pentad: MSE + Pearson + Deriv + VCG + Anatomical Laplacian (mask 1111002)" \
  --factorial-mask 1111002

# Cell 10: MSE + Pearson + Deriv + VCG + Anatomical IMQ MMD (Mask 1111003)
# Full 5-objective composite under unified homoscedastic uncertainty optimization
run_adaptive_cell "AVM5_full_probe_imq_adaptive" \
  "Adaptive Full Pentad: MSE + Pearson + Deriv + VCG + Anatomical IMQ (mask 1111003)" \
  --factorial-mask 1111003

# Bonus Cell 11: MSE + Pearson + VCG + Goldberger Lead Consistency (Mask 1101010)
# Historical benchmark: r = 0.7803, AUROC = 0.8572
run_adaptive_cell "AVLead1_vcg_lead_adaptive" \
  "Adaptive VCG+Lead: MSE + Pearson + VCG + Goldberger Lead Consistency (mask 1101010)" \
  --factorial-mask 1101010

log "All Adaptive VCG & MMD Ablation cells have finished successfully!"
