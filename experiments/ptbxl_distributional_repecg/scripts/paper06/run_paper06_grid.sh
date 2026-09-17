#!/usr/bin/env bash
set -euo pipefail

repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
python=/home/mithunmanivannan/.venv/bin/python
representations=/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper06_conditional/development_representations
output=$repo/experiments/ptbxl_distributional_repecg/outputs/paper06_conditional_repstat

export PYTHONPATH="$repo/experiments/ptbxl_distributional_repecg/src"

echo "=== [START] paper06_conditional_repstat: Training Grid ==="
manifest=$representations/manifest.json
if [ ! -f "$manifest" ]; then
    echo "Paper 6 conditional-residual representation manifest is missing: $manifest" >&2
    exit 1
fi
"$python" - "$manifest" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
if payload.get("kind") != "paper06_conditional_residual_representations" or payload.get("status") != "complete" or payload.get("audit", {}).get("passed") is not True:
    raise SystemExit("Paper 6 representation manifest is not production-compatible")
PY
for variant in "full" "linear_probe" "full_signal" "macro_component" "residual_marginal" "macro_residual_marginals" "shuffled_residual" "joint_pair_sham"; do
    echo "Running variant: $variant"
    if [ -f "$output/cells/.done_${variant}" ]; then
        echo "Variant $variant already completed. Skipping."
        continue
    fi
    CUDA_VISIBLE_DEVICES=0 "$python" -u "$repo/experiments/ptbxl_distributional_repecg/scripts/paper06/train_paper06_shared_grid.py" \
    --representations "$representations" \
    --output "$output" \
    --batch 2048 \
    --max-epochs 100 \
    --patience 10 \
    --seed 42 \
        --variant "$variant"
    touch "$output/cells/.done_${variant}"
done

echo "=== [AGGREGATING] paper06_conditional_repstat ==="
"$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper06/aggregate_paper06_grid.py" \
    --cells "$output/cells" \
    --output "$output" \
    --seed 42

echo "=== [FINISHED DEVELOPMENT] paper06_conditional_repstat ==="
echo "Native-task OOD evaluation remains fail-closed until dataset adapters are reconciled."
