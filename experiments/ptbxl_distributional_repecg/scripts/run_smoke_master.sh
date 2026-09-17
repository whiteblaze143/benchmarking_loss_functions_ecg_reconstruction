#!/usr/bin/env bash
set -euo pipefail

repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
experiment=$repo/experiments/ptbxl_distributional_repecg
python=/home/mithunmanivannan/.venv/bin/python
representations=$experiment/outputs/paper02_kernel_mean/development_representations
smoke_root=$experiment/outputs/smoke_test
results_db=$experiment/outputs/results.sqlite3
export PYTHONPATH="$experiment/src"

"$python" "$experiment/scripts/register_results_plan.py" \
    --database "$results_db" \
    --run-type smoke \
    --datasets ptbxl \
    --split development_val_fold8 \
    --seed 42

if [[ ! -f "$representations/representation_train.npz" || ! -f "$representations/representation_val.npz" ]]; then
    echo "Development representations are incomplete: $representations" >&2
    exit 1
fi

for number in $(seq 1 15); do
    paper_id=$(printf '%02d' "$number")
    train_script=$experiment/scripts/paper$paper_id/train_paper${paper_id}_shared_grid.py
    output=$smoke_root/paper$paper_id
    if [[ ! -f "$train_script" ]]; then
        echo "Missing trainer: $train_script" >&2
        exit 1
    fi
    mapfile -t variants < <(
        "$python" - "$number" <<'PY'
import sys
from repecg.common.variants import get_variants_for_paper
print(*get_variants_for_paper(int(sys.argv[1])), sep="\n")
PY
    )
    echo "=== Paper $paper_id smoke: ${variants[*]} ==="
    for variant in "${variants[@]}"; do
        CUDA_VISIBLE_DEVICES=0 "$python" -u "$train_script" \
            --representations "$representations" \
            --output "$output" \
            --batch 2048 \
            --max-epochs 1 \
            --patience 1 \
            --seed 42 \
            --variant "$variant"
    done
    "$python" "$experiment/scripts/aggregate_grid.py" \
        --paper-id "$number" \
        --cells "$output/cells" \
        --output "$output" \
        --seed 42
    for variant in "${variants[@]}"; do
        CUDA_VISIBLE_DEVICES=0 "$python" "$experiment/scripts/evaluate_grid.py" \
            --paper-id "$number" \
            --variant "$variant" \
            --representations "$representations" \
            --training "$output" \
            --output "$output/development_evaluation"
    done
    "$python" - "$number" "$output" <<'PY'
import json
import sys
from pathlib import Path
from repecg.common.variants import get_variants_for_paper

paper_id = int(sys.argv[1])
output = Path(sys.argv[2])
variants = set(get_variants_for_paper(paper_id))
manifest = json.loads((output / "manifest.json").read_text())
assert set(manifest["variants"]) == variants
assert manifest["cells"] == 9 * len(variants)
for variant in variants:
    assert (output / f"{variant}_best.pt").is_file()
    assert (output / "development_evaluation" / f"metrics_{variant}.json").is_file()
print(f"Paper {paper_id:02d}: verified {len(variants)} variants, {manifest['cells']} cells, all evaluations")
PY
    "$python" "$experiment/scripts/ingest_results_db.py" \
        --database "$results_db" \
        --paper-id "$number" \
        --output "$output" \
        --dataset ptbxl \
        --split development_val_fold8 \
        --run-type smoke
done

echo "All 15 paper smoke pipelines passed strict training, aggregation, and evaluation checks."
