#!/usr/bin/env bash
set -euo pipefail

repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
python=/home/mithunmanivannan/.venv/bin/python
representations=/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/development_representations
output=$repo/experiments/ptbxl_distributional_repecg/outputs/paper09_counterfactual_measurement

export PYTHONPATH="$repo/experiments/ptbxl_distributional_repecg/src"

echo "=== [START] paper09_counterfactual_measurement: Training Grid ==="
for variant in "full" "diagnosis_only" "mismatched_q"; do
    echo "Running variant: $variant"
    if [ -f "$output/cells/.done_${variant}" ]; then
        echo "Variant $variant already completed. Skipping."
        continue
    fi
    CUDA_VISIBLE_DEVICES=0 "$python" -u "$repo/experiments/ptbxl_distributional_repecg/scripts/paper09/train_paper09_shared_grid.py" \
    --representations "$representations" \
    --output "$output" \
    --batch 64 \
    --max-epochs 100 \
    --patience 10 \
    --seed 42 \
        --variant "$variant"
    touch "$output/cells/.done_${variant}"
done

echo "=== [AGGREGATING] paper09_counterfactual_measurement ==="
"$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper09/aggregate_paper09_grid.py" \
    --cells "$output/cells" \
    --output "$output" \
    --seed 42

echo "=== [FINISHED DEVELOPMENT] paper09_counterfactual_measurement ==="
echo "OOD evaluation remains fail-closed until dataset-specific operator-response artifacts exist."
