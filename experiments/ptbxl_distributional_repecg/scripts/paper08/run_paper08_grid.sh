#!/usr/bin/env bash
set -euo pipefail

repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
python=/home/mithunmanivannan/.venv/bin/python
representations=/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper08_tokens/development_representations
output=$repo/experiments/ptbxl_distributional_repecg/outputs/paper08_token_attention

export PYTHONPATH="$repo/experiments/ptbxl_distributional_repecg/src"

echo "=== [START] paper08_token_attention: Training Grid ==="
for variant in "global_dynamic" "local_banded" "static_attention" "uniform_attention" "phase_agnostic_set" "scrambled_phases" "linear_probe" "kmeans_tokens" "cnn_matched_control"; do
    echo "Running variant: $variant"
    if [ -f "$output/cells/.done_${variant}" ]; then
        echo "Variant $variant already completed. Skipping."
        continue
    fi
    CUDA_VISIBLE_DEVICES=0 "$python" -u "$repo/experiments/ptbxl_distributional_repecg/scripts/paper08/train_paper08_shared_grid.py" \
    --representations "$representations" \
    --output "$output" \
    --batch 2048 \
    --max-epochs 100 \
    --patience 10 \
    --seed 42 \
        --variant "$variant"
    touch "$output/cells/.done_${variant}"
done

echo "=== [AGGREGATING] paper08_token_attention ==="
"$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper08/aggregate_paper08_grid.py" \
    --cells "$output/cells" \
    --output "$output" \
    --seed 42

echo "=== [FINISHED DEVELOPMENT] paper08_token_attention ==="
echo "OOD evaluation remains fail-closed until dataset-specific token artifacts exist."
