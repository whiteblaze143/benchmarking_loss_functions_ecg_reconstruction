#!/usr/bin/env bash
# Master GPU queue — runs all runnable papers sequentially on GPU 0.
# Idempotent: each grid script skips completed .done_ variants.
# Launched in a detached tmux session; captures per-paper logs.
set -euo pipefail

REPO=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
PYTHON=/home/mithunmanivannan/.venv/bin/python
SCRIPTS=$REPO/experiments/ptbxl_distributional_repecg/scripts
OUTPUTS=$REPO/experiments/ptbxl_distributional_repecg/outputs
DATA=/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg
LOGDIR=$OUTPUTS/queue_logs
mkdir -p "$LOGDIR"

export PYTHONPATH="$REPO/experiments/ptbxl_distributional_repecg/src"
export CUDA_VISIBLE_DEVICES=0
# Pin all workers to available CPUs; leave 1 for the main process
export OMP_NUM_THREADS=7

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGDIR/master.log"; }

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
    log "=== FAIL  $name (exit $code) — continuing queue ==="
  fi
}

log "GPU queue starting. GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

# ──────────────────────────────────────────────────────────────────────────────
# TIER 1a: Papers that use their own pre-built representations
# These can run immediately, no dependency on P02 reps.
# ──────────────────────────────────────────────────────────────────────────────

# P01 — Distributional Recurrence (reps already built)
run_paper "paper01" "$SCRIPTS/paper01/run_paper01_grid.sh"

# P03 — Path Signature (reps in /data)
run_paper "paper03" "$SCRIPTS/paper03/run_paper03_grid.sh"

# P05 — Koopman Operator (reps in /data)
run_paper "paper05" "$SCRIPTS/paper05/run_paper05_grid.sh"

# P06 — Conditional Distribution (reps in /data)
run_paper "paper06" "$SCRIPTS/paper06/run_paper06_grid.sh"

# P07 — Operator Theory (reps in /data)
run_paper "paper07" "$SCRIPTS/paper07/run_paper07_grid.sh"

# P08 — Equivalence Tokens (reps in /data, multiple versions — grid picks primary)
run_paper "paper08" "$SCRIPTS/paper08/run_paper08_grid.sh"

# ──────────────────────────────────────────────────────────────────────────────
# TIER 1b: P02 grid — fills missing variants (full, linear_probe, linear_kernel, no_circular)
# kernel + moments cells already exist; .done_ guards skip them.
# ──────────────────────────────────────────────────────────────────────────────
run_paper "paper02" "$SCRIPTS/paper02/run_paper02_grid.sh"

# ──────────────────────────────────────────────────────────────────────────────
# TIER 2: Papers that depend on P02 representations being present
# (P02 reps were already built; P02 training above just adds missing variants)
# ──────────────────────────────────────────────────────────────────────────────

# P10 — Interventional Rep-Stat (uses P02 dev reps: full, linear_probe, erm, irm, coral, causirl)
run_paper "paper10" "$SCRIPTS/paper10/run_paper10_grid.sh"

# P11 — Causal State ECG (uses P02 dev reps: full, linear_probe, shuffled_futures)
run_paper "paper11" "$SCRIPTS/paper11/run_paper11_grid.sh"

# P12 — BLOCKED: legacy BCE-only grid retired (final proposal: "RETHINK — must not run").
# V2 G3-real FAILED (conditional NLL loses to phase-only at all origins).
# V2 G4 (clinical utility) BLOCKED by failed G3.
# Uncomment only after V2 real-data gates are resolved.
# run_paper "paper12" "$SCRIPTS/paper12/run_paper12_grid.sh"

# P13 — Inductive Bias Surgery (gates passed, uses P02 dev reps)
run_paper "paper13" "$SCRIPTS/paper13/run_paper13_grid.sh"

# P15 — Phase-Specific Transition Modularity (gates passed, uses P02 dev reps)
run_paper "paper15" "$SCRIPTS/paper15/run_paper15_grid.sh"

# ──────────────────────────────────────────────────────────────────────────────
# TIER 3: P09 depends on P07 operator representations (built during P07 run above)
# ──────────────────────────────────────────────────────────────────────────────

# P09 — SKIPPED: all 8 inductive bias audits FAILED (status: failed)
# Counterfactual operator model cannot be scientifically validated.
# run_paper "paper09" "$SCRIPTS/paper09/run_paper09_grid.sh"

log "=== ALL QUEUED PAPERS COMPLETE ==="
log "Results in: $OUTPUTS"
log "Logs in:    $LOGDIR"
