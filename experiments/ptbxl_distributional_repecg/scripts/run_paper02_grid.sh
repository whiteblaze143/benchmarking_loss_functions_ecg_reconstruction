#!/usr/bin/env bash
set -euo pipefail

repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
python=/home/mithunmanivannan/.venv/bin/python
representations=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/development_representations
output=$repo/experiments/ptbxl_distributional_repecg/outputs/paper02_kernel_mean/development_training
cells=$output/cells
mkdir -p "$cells"

for variant in kernel moments gaussian; do
    for learning_rate in 0.0001 0.0003 0.001; do
        for weight_decay in 0.00001 0.0001 0.001; do
            printf '%s %s %s\n' "$variant" "$learning_rate" "$weight_decay"
        done
    done
done | xargs -P 4 -n 3 bash -c '
    set -euo pipefail
    cell_root=$1
    python=$2
    repo=$3
    representations=$4
    output=$5
    variant=$6
    learning_rate=$7
    weight_decay=$8
    cell="$cell_root/${variant}_lr${learning_rate}_wd${weight_decay}"
    if [[ -f "$cell/summary.json" ]]; then
        exit 0
    fi
    mkdir -p "$cell"
    CUDA_VISIBLE_DEVICES=0 "$python" -u "$repo/experiments/ptbxl_distributional_repecg/scripts/train_paper02_development.py" \
        --representations "$representations" \
        --output "$output" \
        --variant "$variant" \
        --learning-rate "$learning_rate" \
        --weight-decay "$weight_decay" \
        --cell-output "$cell" \
        --batch 2048 \
        --max-epochs 100 \
        --patience 10 \
        --seed 42 >"$cell/run.log" 2>&1
' _ "$cells" "$python" "$repo" "$representations" "$output"

"$python" "$repo/experiments/ptbxl_distributional_repecg/scripts/aggregate_paper02_grid.py" \
    --cells "$cells" \
    --output "$output" \
    --seed 42
