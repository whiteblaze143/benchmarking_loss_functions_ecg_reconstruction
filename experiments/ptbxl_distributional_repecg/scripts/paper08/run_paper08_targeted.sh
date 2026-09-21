#!/usr/bin/env bash
# run_paper08_targeted.sh
# 8 scientifically key cells only (not 100+).
# continuous__dynamic_global already done (0.9174); this runs 7 more.
set -euo pipefail

repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
python=/home/mithunmanivannan/.venv/bin/python
representations=/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper08_tokens/development_representations_strict_b2000_cpu8
output=/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper08_factorial_grid

export PYTHONPATH="$repo/experiments/ptbxl_distributional_repecg/src"

echo "=== [START] paper08_targeted: 8 key cells ==="
"$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper08/freeze_equivalence_artifact.py" \
    --artifact "$representations"

run_cell() {
    local representation=$1
    local routing=$2
    local control=${3:-0}
    local name="${representation}__${routing}"
    if [[ "$representation" == "random_merge" || "$representation" == "frequency_matched_random_merge" ]]; then
        name+="__control$(printf '%02d' "$control")"
    fi
    echo "Running: $name"
    if [ -f "$output/cells/.done_${name}" ]; then
        echo "Already complete: $name — skipping"
        return
    fi
    CUDA_VISIBLE_DEVICES=0 "$python" -u "$repo/experiments/ptbxl_distributional_repecg/scripts/paper08/train_paper08_shared_grid.py" \
    --representations "$representations" \
    --output "$output" \
    --batch 2048 \
    --max-epochs 100 \
    --patience 10 \
    --seed 42 \
        --representation "$representation" \
        --routing "$routing" \
        --random-control "$control"
    touch "$output/cells/.done_${name}"
}

# continuous__dynamic_global already done (0.9174) — .done_ guard skips it
run_cell continuous dynamic_global

# Routing type comparisons on continuous representation
run_cell continuous static
run_cell continuous local_cyclic

# Controls
run_cell continuous uniform_attention   # ablate routing entirely
run_cell continuous scrambled_phases    # does phase order matter?

# Discrete vs continuous tokens
run_cell fine_kmeans dynamic_global

# Simplest structural prior
run_cell equivalence dynamic_global

# Null distribution reference
run_cell random_merge dynamic_global 0

echo "=== [AGGREGATING] paper08_targeted ==="
"$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper08/aggregate_paper08_grid.py" \
    --cells "$output/cells" \
    --output "$output" \
    --seed 42

echo "=== [FINISHED] paper08_targeted ==="
