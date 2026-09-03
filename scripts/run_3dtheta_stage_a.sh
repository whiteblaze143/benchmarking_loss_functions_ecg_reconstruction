#!/bin/bash
set -euo pipefail

# Stage A 3D Theta Spatial Reconstruction Benchmark Runner
# Sequentially trains D0-D5 on Lead I (PTB-XL, Seed 42, 10 Epochs)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="/home/mithunmanivannan/.venv/bin/python3"

echo "======================================================================"
echo "Starting Stage A: 3DRECON-QT 3D Theta Spatial Reconstruction Benchmark"
echo "Date: $(date -u)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "Disk Space:"
df -h /
echo "======================================================================"

# Cells to execute in priority order
# D3 and D5 are highest priority to test the core geometry hypothesis immediately
CELLS=(
    "D3_theta_mul_l1"
    "D5_permuted_theta_mul_l1"
    "D2_current_id_l1"
    "D4_learned12_mul_l1"
    "D1_theta_mul_currentloss"
    "D0_current_id_currentloss"
)

for cell in "${CELLS[@]}"; do
    echo ""
    echo "======================================================================"
    echo ">>> Starting Training for Cell: $cell (Seed 42, Lead I, 10 Epochs)"
    echo "Time: $(date -u)"
    echo "======================================================================"

    # Pre-run disk safety check (must have >= 5GB free)
    FREE_KB=$(df -k / | awk 'NR==2 {print $4}')
    if [ "$FREE_KB" -lt 5242880 ]; then
        echo "ERROR: Disk space low (< 5GB free). Aborting before $cell to ensure data safety."
        exit 1
    fi

    # Execute training
    $PYTHON scripts/train_3dtheta_ablation.py \
        --cell "$cell" \
        --seed 42 \
        --observed-lead 0 \
        --epochs 10 \
        --batch-size 32 \
        --lr 1e-4

    echo ">>> Completed Training for Cell: $cell"
    echo ">>> Updating Summary..."
    $PYTHON scripts/summarize_3dtheta_results.py

    echo "Current Disk Free:"
    df -h /
done

echo ""
echo "======================================================================"
echo "ALL STAGE A CELLS COMPLETED SUCCESSFULLY!"
echo "Date: $(date -u)"
echo "Final Summary in 3DTHETA_SUMMARY.md"
echo "======================================================================"
