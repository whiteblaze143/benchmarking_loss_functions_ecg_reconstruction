#!/usr/bin/env bash
# ==============================================================================
# Round-2 Lean Ablation Queue (Occam's Razor Exploration)
# Baseline: Clean Lean Core (enc4, width512, heads8, no_lead_dropout, zscore, mask 1110000)
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
ROUND1_ANCHOR="$BASE/refine-logs/convergence_10e/runs/conv15e_A0_raw_s42_l0"

mkdir -p "$OUT" "$(dirname $LEDGER)" "$(dirname $LOG)"
[ -f "$LEDGER" ] || echo '{}' > "$LEDGER"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

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

run_cell() {
  local cell_id="$1"
  local desc="$2"
  local anchor_dir="$3"
  shift 3
  local run_name="lean2_${cell_id}_s42_l0"
  local rdir="$OUT/$run_name"

  log "======================================================================"
  log "STARTING $cell_id: $run_name"
  log "Description: $desc"
  log "Anchor: $anchor_dir"
  mkdir -p "$rdir"

  # Base Lean Core Hyperparameters
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
    --width 512
    --heads 8
    --encoder-depth 4
    --decoder-depth 4
    --patch-size 25
    --artificial-mask-mode no_lead_dropout
    --zscore-norm
    --factorial-mask 1110000
    --seg-ce-weight 1.0
    --dice-weight 0.5
    --seed 42
    --checkpoint-policy best
    --rolling-resume
    "$@"
  )

  log "Executing command..."
  "${cmd[@]}" 2>&1 | tee "$rdir/train.log"
  local rc=${PIPESTATUS[0]}

  if [ $rc -ne 0 ] || [ ! -f "$rdir/_SUCCESS.json" ]; then
    log "ERROR: $run_name failed (rc=$rc). Check $rdir/train.log"
    update_ledger "$cell_id" "$run_name" "FAILED (code $rc)" ""
    return 1
  fi

  log "Training completed successfully. Running paired bootstrap evaluation..."
  "$PY" "$EVAL" \
    --candidate-dir "$rdir" \
    --anchor-dir "$anchor_dir" \
    --split val \
    --n-boot 10000 \
    --batch-size 64 2>&1 | tee -a "$LOG"

  local eval_file="$rdir/killgate_eval_val.json"
  update_ledger "$cell_id" "$run_name" "COMPLETED" "$eval_file"
  log "Done with $cell_id."
  log "======================================================================"
}

# ------------------------------------------------------------------------------
# Cell 1: Clean Lean Best (Anchor for subsequent Occam ablations)
# Combines: enc4 + width512 + heads8 + no_lead_dropout + zscore + clean mask 1110000
# Compared directly to Round-1 Raw Baseline A0
# ------------------------------------------------------------------------------
run_cell "L1_clean_lean_best" \
  "Clean Combined Lean Best (enc4, width512, no_lead_dropout, zscore, mask 1110000)" \
  "$ROUND1_ANCHOR"

LEAN_ANCHOR="$OUT/lean2_L1_clean_lean_best_s42_l0"

# ------------------------------------------------------------------------------
# Cell 2: Remove Delineation Head & Auxiliary Losses
# ------------------------------------------------------------------------------
run_cell "LA1_nodel" \
  "Occam 1: Remove Delineation MTL Head & Segmentation Losses" \
  "$LEAN_ANCHOR" \
  --no-delineation-head --seg-ce-weight 0.0 --dice-weight 0.0

# ------------------------------------------------------------------------------
# Cell 3: Decoder Depth D=3 (Between D=2 failure and D=4 standard)
# ------------------------------------------------------------------------------
run_cell "LB1_dec3" \
  "Occam 2: Decoder Depth D=3 (Prune 1 decoder layer)" \
  "$LEAN_ANCHOR" \
  --decoder-depth 3

# ------------------------------------------------------------------------------
# Cell 4: Width W=384, Heads H=6 (Further Capacity Reduction)
# ------------------------------------------------------------------------------
run_cell "LC1_width384" \
  "Occam 3: Prune Hidden Width to 384 and Heads to 6" \
  "$LEAN_ANCHOR" \
  --width 384 --heads 6

# ------------------------------------------------------------------------------
# Cell 5: Encoder Depth E=3 (Prune 1 more encoder layer)
# ------------------------------------------------------------------------------
run_cell "LD1_enc3" \
  "Occam 4: Encoder Depth E=3 (Prune below 4-layer sweet spot)" \
  "$LEAN_ANCHOR" \
  --encoder-depth 3

# ------------------------------------------------------------------------------
# Cell 6: Delineation CE-Only (Remove Dice Loss)
# ------------------------------------------------------------------------------
run_cell "LE1_nodice" \
  "Occam 5: Delineation with Cross-Entropy only (Dice weight 0.0)" \
  "$LEAN_ANCHOR" \
  --dice-weight 0.0

# ------------------------------------------------------------------------------
# Cell 7: Adaptive Homoscedastic Uncertainty Weighting
# ------------------------------------------------------------------------------
run_cell "LE2_adaptive" \
  "Occam 6: Adaptive Composite Loss Weighting" \
  "$LEAN_ANCHOR" \
  --reconstruction-loss-type adaptive_composite

log "All Round-2 Lean Ablation cells have finished successfully!"

# ==============================================================================
# Chained Extended Queue (26 Additional Systematic Occam Configurations)
# ==============================================================================
if [ -f "$BASE/refine-logs/lean_abl2_extended_queue.sh" ]; then
  log "Launching chained extended queue (26 configs): lean_abl2_extended_queue.sh"
  bash "$BASE/refine-logs/lean_abl2_extended_queue.sh"
fi
