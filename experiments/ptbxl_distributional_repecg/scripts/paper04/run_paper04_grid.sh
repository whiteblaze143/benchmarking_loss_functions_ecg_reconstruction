#!/usr/bin/env bash
set -euo pipefail

repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
python=/home/mithunmanivannan/.venv/bin/python
representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/development_representations
ood_representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/ood_representations
output=$repo/experiments/ptbxl_distributional_repecg/outputs/paper04_hankel_dynamics

export PYTHONPATH="$repo/experiments/ptbxl_distributional_repecg/src"

# Historical scientific audit: The historical local spectral-descriptor representation
# failed its prespecified Nyström fidelity criterion (status: ineligible_nystrom_fidelity).
# Rather than discarding this result, it is preserved fail-closed in manifest.json and tested in World 5.
# Modern Paper 04 evaluates differentiable cyclic delay operators over validated Phase-KME distributions.
echo "=== [HISTORICAL AUDIT] Local spectral-descriptor representation ineligible; evaluating cyclic delay operators over Phase-KME ==="

echo "=== [START] paper04_hankel_dynamics: Training Grid ==="
for variant in "full" "paper02_phasecnn" "flat_phase_mlp" "phase_aware_linear_probe" "linear_probe" "lag1" "lag2" "open_chain" "time_shuffled" "time_reversed" "operator_summary_probe" "ridge_strength_sensitivity"; do
    echo "Running variant: $variant"
    if [ -f "$output/cells/.done_${variant}" ]; then
        echo "Variant $variant already completed. Skipping."
        continue
    fi
    CUDA_VISIBLE_DEVICES=0 "$python" -u "$repo/experiments/ptbxl_distributional_repecg/scripts/paper04/train_paper04_shared_grid.py" \
    --representations "$representations" \
    --output "$output" \
    --batch 2048 \
    --max-epochs 100 \
    --patience 10 \
    --seed 42 \
        --variant "$variant"
    touch "$output/cells/.done_${variant}"
done

echo "=== [AGGREGATING] paper04_hankel_dynamics ==="
"$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper04/aggregate_paper04_grid.py" \
    --cells "$output/cells" \
    --output "$output" \
    --seed 42

echo "=== [EVALUATING OOD] paper04_hankel_dynamics across 9 datasets ==="
CUDA_VISIBLE_DEVICES=0 "$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper04/evaluate_paper04_ood.py" \
    --training "$output" \
    --representations "$ood_representations" \
    --output "$output/ood_evaluation"

echo "=== [FINISHED] paper04_hankel_dynamics ==="
