#!/bin/bash
PYTHON_EXEC="/home/mithunmanivannan/ecg_recon_env/bin/python"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT="${SCRIPT_DIR}/surgical_fix.py"

echo "Launching Surgical Fix Candidates..."

# Surgical 1: Pearson (0.05) + MMD (0.01)
$PYTHON_EXEC $SCRIPT --config_id surgical_1 --lambda_corr 0.05 --lambda_mmd 0.01 --corr_type pearson > logs/surgical_1.log 2>&1 &
PID1=$!
echo "Launched Surgical 1 (PID $PID1)"

# Surgical 2: Spearman (0.05) + MMD (0.01)
$PYTHON_EXEC $SCRIPT --config_id surgical_2 --lambda_corr 0.05 --lambda_mmd 0.01 --corr_type spearman > logs/surgical_2.log 2>&1 &
PID2=$!
echo "Launched Surgical 2 (PID $PID2)"

# Surgical 3: Pearson (0.05) + MMD (0.001)
$PYTHON_EXEC $SCRIPT --config_id surgical_3 --lambda_corr 0.05 --lambda_mmd 0.001 --corr_type pearson > logs/surgical_3.log 2>&1 &
PID3=$!
echo "Launched Surgical 3 (PID $PID3)"

echo "All surgical jobs launched. Monitor logs/surgical_*.log"
wait
echo "All jobs finished."
