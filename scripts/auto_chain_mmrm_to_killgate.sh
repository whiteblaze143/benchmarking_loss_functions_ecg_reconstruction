#!/usr/bin/env bash
# auto_chain_mmrm_to_killgate.sh
# Monitors empirical_patient_extraction, executes rigorous clinical MMRM & GEE,
# and automatically launches the ecgaim_killgate GPU training queue upon completion.

set -u

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$_ROOT" || exit 1

LOG_FILE="$_ROOT/refine-logs/killgate/auto_chain_mmrm_to_killgate.log"
mkdir -p "$_ROOT/refine-logs/killgate"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-Chainer started: Monitoring empirical_patient_extraction (55 models)..." | tee -a "$LOG_FILE"

# 1. Wait for extract_patient_level_clinical_predictions.py to finish
while tmux has-session -t empirical_patient_extraction 2>/dev/null; do
    if ! pgrep -f "extract_patient_level_clinical_predictions.py" >/dev/null 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] extract_patient_level_clinical_predictions.py has completed!" | tee -a "$LOG_FILE"
        break
    fi
    sleep 20
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Patient-level extraction finished. Fitting rigorous clinical MMRM & GEE..." | tee -a "$LOG_FILE"

# 2. Run MMRM and GEE pipeline
/home/mithunmanivannan/.venv/bin/python3 scripts/run_clinical_mmrm_and_gee_rigorous.py 2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] MMRM and GEE fitting complete! Generated artifacts:" | tee -a "$LOG_FILE"
ls -lh results/clinical_biomarkers_multids/EMPIRICAL_CLINICAL_MMRM_AND_GEE* | tee -a "$LOG_FILE"

# 3. Wait 10 seconds for clean GPU memory release
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting 10 seconds for CUDA memory cleanup..." | tee -a "$LOG_FILE"
sleep 10

# 4. Kill old dead ecgaim_killgate session if lingering
tmux kill-session -t ecgaim_killgate 2>/dev/null || true

# 5. Launch ecgaim_killgate queue on CUDA:0
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching ecgaim_killgate queue on CUDA:0..." | tee -a "$LOG_FILE"
tmux new-session -d -s ecgaim_killgate "cd '$_ROOT' && /home/mithunmanivannan/.venv/bin/python3 scripts/run_ecgaim_killgate_queue.py 2>&1 | tee refine-logs/killgate/queue_master.log; read"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Successfully queued and launched ecgaim_killgate session! Verifying status:" | tee -a "$LOG_FILE"
tmux ls | tee -a "$LOG_FILE"
