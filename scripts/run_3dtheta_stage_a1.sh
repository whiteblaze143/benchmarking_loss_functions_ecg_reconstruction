#!/usr/bin/env bash
# ==============================================================================
# Stage A.1 Causal Factorial Cleanup Runner (Seed 42)
# Runs: D6 (theta + add + L1), D7 (learned + add + L1), D8 (random + mul + L1)
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="/home/mithunmanivannan/.venv/bin/python3"
OUT_ROOT="refine-logs/convergence_10e/runs"
mkdir -p "$OUT_ROOT"

# Pre-flight Disk Check
AVAIL_GB=$(df -BG "$ROOT" | awk 'NR==2 {gsub("G",""); print $4}')
echo ">>> Pre-flight storage check: ${AVAIL_GB} GB available on root disk."
if [ "$AVAIL_GB" -lt 5 ]; then
    echo "ERROR: Storage safety violation: Free space (${AVAIL_GB} GB) is below 5 GB minimum threshold."
    exit 1
fi

CELLS=(
    "D6_theta_add_l1"
    "D7_learned12_add_l1"
    "D8_random12_mul_l1"
)

echo "=================================================================="
echo "Starting Stage A.1 Factorial Decomposition on Lead I (Seed 42)"
echo "Target cells: ${CELLS[*]}"
echo "=================================================================="

for cell in "${CELLS[@]}"; do
    run_dir="${OUT_ROOT}/${cell}_s42_l0"
    echo ""
    echo ">>> Running Cell: ${cell} (Directory: ${run_dir}) <<<"
    
    if [ -f "${run_dir}/_SUCCESS.json" ]; then
        echo "Cell ${cell} already completed successfully. Skipping."
        continue
    fi

    # Check free disk space before each run
    CURRENT_AVAIL_GB=$(df -BG "$ROOT" | awk 'NR==2 {gsub("G",""); print $4}')
    if [ "$CURRENT_AVAIL_GB" -lt 4 ]; then
        echo "CRITICAL: Storage fell below 4 GB during queue. Halting queue."
        exit 1
    fi

    "$PY" scripts/train_3dtheta_ablation.py \
        --cell "$cell" \
        --seed 42 \
        --observed-lead 0 \
        --epochs 10 \
        --batch-size 32 \
        --runs-dir "$OUT_ROOT"

    echo ">>> Completed Cell: ${cell} <<<"
done

echo ""
echo "=================================================================="
echo "Stage A.1 training finished. Running evaluations & synthesis..."
echo "=================================================================="

echo ">>> 1. Computing per-lead and anatomical metrics in compact.sqlite..."
"$PY" scripts/evaluate_3dtheta_per_lead.py

echo ">>> 2. Computing 10,000-resample paired bootstrap uncertainty analysis..."
"$PY" scripts/compute_3dtheta_paired_bootstrap.py --n-boot 10000

echo ">>> 3. Updating 3DTHETA_SUMMARY.md..."
"$PY" scripts/summarize_3dtheta_results.py

echo ">>> 4. Updating Master CSV results/lead1_all_models_comprehensive_metrics.csv..."
"$PY" scripts/build_lead1_comprehensive_csv.py

echo ">>> 5. Regenerating book/LEAD_I_METRICS_AND_CHAMPIONS_COMPREHENSIVE.md..."
"$PY" scripts/generate_lead1_champions_markdown.py

echo ">>> ALL STAGE A.1 TASKS COMPLETED SUCCESSFULLY! <<<"
