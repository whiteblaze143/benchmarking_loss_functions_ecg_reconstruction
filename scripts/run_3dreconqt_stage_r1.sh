#!/usr/bin/env bash
# ==============================================================================
# Stage R1: 3DRECON-QT Sampled-Query Benchmark Runner (Seed 42)
# Primary Falsification Screen: RQ1Q (true theta) vs RQ2Q (permuted theta)
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="/home/mithunmanivannan/.venv/bin/python3"
OUT_ROOT="refine-logs/3dreconqt_reference/runs"
mkdir -p "$OUT_ROOT"

# Pre-flight Storage Safety Check
AVAIL_GB=$(df -BG "$ROOT" | awk 'NR==2 {gsub("G",""); print $4}')
echo ">>> Pre-flight storage check: ${AVAIL_GB} GB available on root disk."
if [ "$AVAIL_GB" -lt 5 ]; then
    echo "ERROR: Storage safety violation: Free space (${AVAIL_GB} GB) is below 5 GB minimum threshold."
    exit 1
fi

# Step 1: Run Primary Decisive Geometry Contrast (RQ1Q vs RQ2Q)
PRIMARY_CELLS=(
    "RQ1Q_theta_l1"
    "RQ2Q_permuted_theta_l1"
)

echo "=================================================================="
echo "Stage R1 Primary Screen: Sampled-Query Lead I (Seed 42)"
echo "Target cells: ${PRIMARY_CELLS[*]}"
echo "=================================================================="

for cell in "${PRIMARY_CELLS[@]}"; do
    run_dir="${OUT_ROOT}/${cell}_s42"
    echo ""
    echo ">>> Running Primary Reference Cell: ${cell} (Directory: ${run_dir}) <<<"
    
    if [ -f "${run_dir}/_SUCCESS.json" ]; then
        echo "Cell ${cell} already completed successfully. Skipping training."
    else
        CURRENT_AVAIL_GB=$(df -BG "$ROOT" | awk 'NR==2 {gsub("G",""); print $4}')
        if [ "$CURRENT_AVAIL_GB" -lt 4 ]; then
            echo "CRITICAL: Storage fell below 4 GB during queue. Halting queue."
            exit 1
        fi

        "$PY" scripts/train_3dreconqt_reference.py \
            --cell "$cell" \
            --seed 42 \
            --epochs 100 \
            --patience 15 \
            --physical-batch-size 32 \
            --effective-batch-size 128 \
            --lr 1e-3 \
            --momentum 0.9 \
            --weight-decay 1e-5 \
            --normalization "strict_deployable" \
            --runs-dir "$OUT_ROOT"

        echo ">>> Completed Training for: ${cell} <<<"
    fi

    # Run comprehensive evaluation on best checkpoint
    if [ -f "${run_dir}/best.pt" ] && [ ! -f "${run_dir}/eval_summary.json" ]; then
        echo ">>> Evaluating ${cell} on PTB-XL Validation Cohort... <<<"
        "$PY" scripts/evaluate_3dreconqt_reference.py \
            --checkpoint "${run_dir}/best.pt" \
            --out-dir "$run_dir"
    fi
done

# Perform Paired Bootstrap Analysis for Primary Screen (RQ1Q - RQ2Q)
echo ""
echo "=================================================================="
echo "Running Paired Patient-Level Bootstrap for Primary Contrast (10,000 resamples)..."
echo "=================================================================="

BOOT_DIR="refine-logs/3dreconqt_reference/bootstrap"
mkdir -p "$BOOT_DIR"

if [ -f "${OUT_ROOT}/RQ1Q_theta_l1_s42/patient_level_eval.npz" ] && [ -f "${OUT_ROOT}/RQ2Q_permuted_theta_l1_s42/patient_level_eval.npz" ]; then
    echo ">>> Primary Falsification Contrast: RQ1Q (True Theta) - RQ2Q (Permuted Theta) <<<"
    "$PY" scripts/compute_3dreconqt_bootstrap.py \
        --model-a-npz "${OUT_ROOT}/RQ1Q_theta_l1_s42/patient_level_eval.npz" \
        --model-b-npz "${OUT_ROOT}/RQ2Q_permuted_theta_l1_s42/patient_level_eval.npz" \
        --label-a "RQ1Q_theta" \
        --label-b "RQ2Q_permuted" \
        --n-boot 10000 \
        --out-json "${BOOT_DIR}/rq1q_minus_rq2q_bootstrap.json"
fi

echo "=================================================================="
echo "Primary Screen (RQ1Q vs RQ2Q) Complete!"
echo "Inspect delta_r. If |delta_r| < 0.003, proceed to RQ3Q/RQ4Q and V3-V2."
echo "=================================================================="
