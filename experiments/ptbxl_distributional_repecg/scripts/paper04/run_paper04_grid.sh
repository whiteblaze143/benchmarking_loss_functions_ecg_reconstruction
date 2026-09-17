#!/usr/bin/env bash
set -euo pipefail

repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
python=/home/mithunmanivannan/.venv/bin/python
representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/development_representations
ood_representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/ood_representations
output=$repo/experiments/ptbxl_distributional_repecg/outputs/paper04_hankel_dynamics

export PYTHONPATH="$repo/experiments/ptbxl_distributional_repecg/src"

manifest=/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper04_hankel_dynamics/development_representations/manifest.json
if [[ -f "$manifest" ]] && "$python" - "$manifest" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if payload.get("status") == "ineligible_nystrom_fidelity" else 1)
PY
then
    echo "Paper 4 is scientifically ineligible: the frozen 128/256-landmark Nyström fidelity gate failed." >&2
    exit 2
fi

echo "=== [START] paper04_hankel_dynamics: Training Grid ==="
for variant in "full" "linear_probe" "time_shuffled"; do
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
