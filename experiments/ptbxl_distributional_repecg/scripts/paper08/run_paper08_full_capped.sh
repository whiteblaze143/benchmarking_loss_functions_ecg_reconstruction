#!/usr/bin/env bash
# run_paper08_full_capped.sh
# Full P08 primary grid + 7 null controls per routing (capped from 10).
# Estimated: ~60 cells × 12 min = ~12 hours on A100.
# continuous__dynamic_global already done (0.9174) — .done_ guard skips it.
set -euo pipefail

repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
python=/home/mithunmanivannan/.venv/bin/python
representations=/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper08_tokens/development_representations_strict_b2000_cpu8
output=/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper08_factorial_grid

export PYTHONPATH="$repo/experiments/ptbxl_distributional_repecg/src"

echo "=== [START] paper08_full_capped: ~60 cells (7 null controls/routing) ==="
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

# ── Primary factorial: 3 routing × 4 representations = 12 cells ──────────────
for routing in dynamic_global static local_cyclic; do
    for representation in continuous fine_kmeans size_matched_kmeans equivalence; do
        run_cell "$representation" "$routing"
    done
    # Null distribution: 7 controls per routing (capped from 10)
    # 7 is sufficient to characterise the null distribution shape
    for control in $(seq 0 6); do
        run_cell random_merge "$routing" "$control"
        run_cell frequency_matched_random_merge "$routing" "$control"
    done
done

# ── Explicit controls (outside factorial) ─────────────────────────────────────
run_cell unk_pattern_only dynamic_global
run_cell token_without_unk_signal dynamic_global

# ── Alternative routing controls on continuous representation ─────────────────
for routing in uniform_attention phase_agnostic_set scrambled_phases cnn_matched_control; do
    run_cell continuous "$routing"
done

echo "=== [AGGREGATING] paper08_full_capped ==="
"$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/paper08/aggregate_paper08_grid.py" \
    --cells "$output/cells" \
    --output "$output" \
    --seed 42

echo "=== [FINISHED] paper08_full_capped ==="
echo "Held-out Fold 8 eval: run evaluate_all_papers_fold8.py --paper 8"
