#!/usr/bin/env bash
set -euo pipefail

repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
python=/home/mithunmanivannan/.venv/bin/python
representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper01_distributional_recurrence/development_representations
ood_representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/ood_representations
output=$repo/experiments/ptbxl_distributional_repecg/outputs/paper01_distributional_recurrence
for variant in "full" "linear_probe" "mean_distance_recurrence" "phase_content_permuted" "cyclic_relabel_sham"; do
    echo "Running variant: $variant"
    if [ -f "$output/cells/.done_${variant}" ]; then
        echo "Variant $variant already completed. Skipping."
        continue
    fi
    CUDA_VISIBLE_DEVICES=0 "$python" -u "$repo/experiments/ptbxl_distributional_repecg/scripts/paper01/train_paper01_shared_grid.py" \
    --representations "$representations" \
    --output "$output" \
    --batch 2048 \
    --max-epochs 100 \
    --patience 10 \
    --seed 42 \
        --variant "$variant"
    touch "$output/cells/.done_${variant}"
done
"$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper01/aggregate_paper01_grid.py" \
    --cells "$output/cells" \
    --output "$output" \
    --seed 42

# RETIRED: CUDA_VISIBLE_DEVICES=0 "$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper01/evaluate_paper01_ood.py" \
    # --training "$output" \
    # --representations "$ood_representations" \
    # --output "$output/ood_evaluation"
