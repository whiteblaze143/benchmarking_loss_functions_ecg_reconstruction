#!/usr/bin/env bash
set -euo pipefail

repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
python=/home/mithunmanivannan/.venv/bin/python
representations=/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/development_representations
output=$repo/experiments/ptbxl_distributional_repecg/outputs/paper07_operator_reconstruction

export PYTHONPATH="$repo/experiments/ptbxl_distributional_repecg/src"

echo "=== [START] paper07_operator_reconstruction: Training Grid ==="
for variant in "continuous_primary" "categorical_primary" "continuous_auxiliary" "categorical_auxiliary"; do
    echo "Running variant: $variant"
    if [ -f "$output/cells/.done_${variant}" ]; then
        echo "Variant $variant already completed. Skipping."
        continue
    fi
    CUDA_VISIBLE_DEVICES=0 "$python" -u "$repo/experiments/ptbxl_distributional_repecg/scripts/paper07/train_paper07_shared_grid.py" \
    --representations "$representations" \
    --output "$output" \
    --batch 64 \
    --max-epochs 100 \
    --patience 10 \
    --seed 42 \
        --variant "$variant"
    touch "$output/cells/.done_${variant}"
done

echo "=== [AGGREGATING] paper07_operator_reconstruction ==="
"$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper07/aggregate_paper07_grid.py" \
    --cells "$output/cells" \
    --output "$output" \
    --seed 42

echo "=== [FINISHED DEVELOPMENT] paper07_operator_reconstruction ==="
echo "OOD evaluation remains fail-closed until dataset-specific operator-response artifacts exist."
