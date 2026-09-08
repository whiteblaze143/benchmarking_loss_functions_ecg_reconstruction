#!/usr/bin/env bash
# ==============================================================================
# Round-2 Extended Lean Ablation Queue (26 Additional Systematic Configurations)
# Anchored on: L1_clean_lean_best (enc4, width512, heads8, no_lead_dropout, zscore, mask 1110000)
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

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [EXT-QUEUE] $*" | tee -a "$LOG"; }

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

run_ext_cell() {
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
  log "STARTING EXTENDED CELL $cell_id: $run_name"
  log "Description: $desc"
  log "Anchor: $LEAN_ANCHOR"
  mkdir -p "$rdir"

  # Default Lean Core Flags
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
    --anchor-dir "$LEAN_ANCHOR" \
    --split val \
    --n-boot 10000 \
    --batch-size 64 2>&1 | tee -a "$LOG"

  local eval_file="$rdir/killgate_eval_val.json"
  update_ledger "$cell_id" "$run_name" "COMPLETED" "$eval_file"
  log "Done with $cell_id."
  log "======================================================================"
}

log "Starting Extended Lean Ablation Queue (26 configurations)..."

# ==============================================================================
# Axis 1: Loss Triplet Decomposition (Occam on Reconstruction Objective)
# ==============================================================================
run_ext_cell "L_mse_only" \
  "Loss Decomposition: Pure MSE loss (mask 1000000, ablate corr & deriv)" \
  --factorial-mask 1000000

run_ext_cell "L_corr_only" \
  "Loss Decomposition: Pure Pearson Correlation loss (mask 0100000, ablate MSE & deriv)" \
  --factorial-mask 0100000

run_ext_cell "L_mse_corr" \
  "Loss Decomposition: MSE + Pearson (mask 1100000, ablate first derivative)" \
  --factorial-mask 1100000

run_ext_cell "L_mse_deriv" \
  "Loss Decomposition: MSE + Derivative (mask 1010000, ablate Pearson correlation)" \
  --factorial-mask 1010000

run_ext_cell "L_l1_direct" \
  "Loss Decomposition: Direct L1 Norm Reconstruction Objective" \
  --reconstruction-loss-type l1

# ==============================================================================
# Axis 2: Wavelet & Multi-Spectral Exploration on Lean Backbone
# ==============================================================================
run_ext_cell "W_conv_c128" \
  "Wavelet Lean: Lightweight 1D-CNN Morlet branch (hidden=128)" \
  --use-wavelet-branch --wavelet-encoder conv --wavelet-conv-hidden 128

run_ext_cell "W_conv_c256" \
  "Wavelet Lean: Standard 1D-CNN Morlet branch (hidden=256)" \
  --use-wavelet-branch --wavelet-encoder conv --wavelet-conv-hidden 256

run_ext_cell "W_timesformer_dim256" \
  "Wavelet Lean: Compact Time-Frequency Transformer (dim=256, depth=2)" \
  --use-wavelet-branch --wavelet-encoder timesformer --wavelet-dim 256 --wavelet-depth 2

run_ext_cell "W_fusion_gated" \
  "Wavelet Lean: Gated Residual Feature Addition Fusion" \
  --use-wavelet-branch --wavelet-fusion gated_add

run_ext_cell "W_fusion_crossattn" \
  "Wavelet Lean: Cross-Attention Wavelet Feature Conditioning (4 heads)" \
  --use-wavelet-branch --wavelet-fusion cross_attn --fusion-heads 4

# ==============================================================================
# Axis 3: Temporal Granularity & Patch Tokenization
# ==============================================================================
run_ext_cell "T_patch50" \
  "Patch Tokenization: Coarse Patch Size 50 (Half token count, ultra-fast)" \
  --patch-size 50

run_ext_cell "T_patch10" \
  "Patch Tokenization: Fine Patch Size 10 (High temporal resolution for steep QRS)" \
  --patch-size 10

run_ext_cell "T_heads4" \
  "Attention Granularity: 4 Attention Heads (Lower capacity)" \
  --heads 4

run_ext_cell "T_heads16" \
  "Attention Granularity: 16 Attention Heads (Higher multi-head subspace resolution)" \
  --heads 16

# ==============================================================================
# Axis 4: Multi-Task Clinical Delineation Supervision
# ==============================================================================
run_ext_cell "D_cadence1" \
  "Delineation Supervision: Every Epoch (Continuous anatomical alignment)" \
  --delineation-every 1

run_ext_cell "D_cadence4" \
  "Delineation Supervision: Sparse Cadence Every 4 Epochs" \
  --delineation-every 4

run_ext_cell "D_boundary" \
  "Delineation Loss: Add Morphological Transition Boundary Penalty (weight 0.2)" \
  --boundary-weight 0.2

run_ext_cell "D_fiducial" \
  "Delineation Loss: Add Peak Fiducial Point Alignment Penalty (weight 0.2)" \
  --fiducial-weight 0.2

run_ext_cell "D_heavy_seg" \
  "Delineation Loss: Stronger Segmentation Weight (CE=2.0, Dice=1.0)" \
  --seg-ce-weight 2.0 --dice-weight 1.0

run_ext_cell "D_light_ce" \
  "Delineation Loss: Soft Segmentation Weight (CE=0.5, Dice=0.25)" \
  --seg-ce-weight 0.5 --dice-weight 0.25

# ==============================================================================
# Axis 5: Spatial Geometry & Regularization
# ==============================================================================
run_ext_cell "S_film" \
  "Spatial Conditioning: Feature-wise Linear Modulation (FiLM) for Lead Projection" \
  --use-spatial-film

run_ext_cell "S_panorama" \
  "Spatial Conditioning: Panoramic Lead Geometry Embeddings" \
  --lead-conditioning-mode panorama

run_ext_cell "R_mask15" \
  "Regularization: 15% Random Point Masking Data Augmentation" \
  --random-mask-ratio 0.15

run_ext_cell "R_tempmask15" \
  "Regularization: 15% Contiguous Temporal Masking Data Augmentation" \
  --temporal-mask-ratio 0.15

run_ext_cell "R_wd_low" \
  "Regularization: Lower Weight Decay (1e-5)" \
  --weight-decay 0.00001

run_ext_cell "R_wd_high" \
  "Regularization: Higher Weight Decay (1e-3)" \
  --weight-decay 0.001

log "All 26 Extended Lean Ablation cells have finished successfully!"
