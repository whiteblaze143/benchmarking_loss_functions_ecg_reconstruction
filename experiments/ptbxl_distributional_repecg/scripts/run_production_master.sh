#!/usr/bin/env bash
set -euo pipefail

repo="/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction"
experiment="$repo/experiments/ptbxl_distributional_repecg"

if [[ ! -f "$experiment/outputs/PRODUCTION_READY" ]]; then
    echo "Production blocked: strict smoke, mechanism coverage, dataset-label, and evaluation gates are not complete." >&2
    exit 1
fi

echo "========================================"
echo " Starting Master Production Queue v2"
echo "========================================"

for i in {1..15}; do
    paper_id=$(printf "%02d" $i)
    echo ""
    echo "----------------------------------------"
    echo " Executing Paper $paper_id Grid Search"
    echo "----------------------------------------"
    
    script="$repo/experiments/ptbxl_distributional_repecg/scripts/paper$paper_id/run_paper${paper_id}_grid.sh"
    

    if [ -f "$repo/experiments/ptbxl_distributional_repecg/outputs/paper${paper_id}.done" ]; then
        echo "Paper $paper_id already completed. Skipping."
        continue
    fi
    if [ ! -f "$script" ]; then
        echo "Missing $script; stopping." >&2
        exit 1
    fi
    
    # Run the grid search script (which handles representations, training, and evaluation)
    bash "$script" || {
        echo "ERROR: Paper $paper_id failed! Stopping queue to preserve state."
        exit 1
    }
    touch "$repo/experiments/ptbxl_distributional_repecg/outputs/paper${paper_id}.done"
done

echo ""
echo "========================================"
echo " Master Production Queue Completed Successfully!"
echo "========================================"
