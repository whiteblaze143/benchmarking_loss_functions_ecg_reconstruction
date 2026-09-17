#!/usr/bin/env bash
set -euo pipefail

repo="/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction"
python="/home/mithunmanivannan/.venv/bin/python"

echo "========================================"
echo " Starting Master Smoke Test Queue v2"
echo "========================================"

for i in {1..15}; do
    paper_id=$(printf "%02d" $i)
    echo ""
    echo "----------------------------------------"
    echo " Smoke Testing Paper $paper_id"
    echo "----------------------------------------"
    
    script="$repo/experiments/ptbxl_distributional_repecg/scripts/paper$paper_id/train_paper${paper_id}_shared_grid.py"
    
    # We need to know where representations are
    # For smoke tests, we assume outputs/prepare_smoke exists or similar, but
    # some papers use specific dirs. We can pull it from the run_paperXX_grid.sh
    
    # Since different papers might use different representation paths, we parse it from the bash script.
    run_script="$repo/experiments/ptbxl_distributional_repecg/scripts/paper$paper_id/run_paper${paper_id}_grid.sh"
    
    if [ ! -f "$run_script" ]; then
        echo "Missing $run_script, skipping."
        continue
    fi
    
    rep_dir=$(grep "representations=" "$run_script" | head -n 1 | cut -d'=' -f2)
    # evaluate variable
    rep_dir=$(eval echo "$rep_dir")
    
    out_dir="$repo/experiments/ptbxl_distributional_repecg/outputs/smoke_test/paper$paper_id"
    mkdir -p "$out_dir"
    
    if [ ! -f "$rep_dir/manifest.json" ]; then
        echo "Representations not found at $rep_dir, skipping smoke test for Paper $paper_id."
        continue
    fi
    
    echo "Running 1 epoch dry-run for Paper $paper_id variant 'full'..."
    CUDA_VISIBLE_DEVICES=0 "$python" -u "$script" \
        --representations "$rep_dir" \
        --output "$out_dir" \
        --batch 2048 \
        --max-epochs 1 \
        --patience 1 \
        --seed 42 \
        --variant "full" || { echo "Paper $paper_id smoke test FAILED!"; exit 1; }
        
    echo "Paper $paper_id smoke test passed."
done

echo ""
echo "========================================"
echo " Master Smoke Test Queue Completed!"
echo "========================================"
