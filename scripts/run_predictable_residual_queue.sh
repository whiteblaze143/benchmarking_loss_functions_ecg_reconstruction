#!/usr/bin/env bash
set -euo pipefail

project_root=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
python_bin=/home/mithunmanivannan/.venv/bin/python3
output_root="$project_root/refine-logs/predictable_residual_subspace/runs"
manifest="$project_root/refine-logs/predictable_residual_subspace/K_STAR_manifest.json"
mkdir -p "$output_root"
cd "$project_root"

common=(
  --data-dir data/ptb_xl/tensors
  --data-manifest refine-logs/ptbxl_tensor_content_manifest.json
  --metadata data/ptb_xl/ptbxl_database.csv
  --delineation-dir data/rdb_wavelet_delineation_cache
  --k-star-manifest "$manifest"
  --factorial-mask 1110000
  --observed-leads 0
  --epochs 15
  --seed 42
  --batch-size 32
  --delineation-batch-size 32
  --num-workers 2
  --checkpoint-policy best
  --rolling-resume
  --require-cuda
  --patch-size 10
  --width 512
  --encoder-depth 4
  --decoder-depth 4
  --heads 8
  --artificial-mask-mode no_lead_dropout
  --reconstruction-loss-type composite
  --lead-conditioning-mode learned
  --no-use-relative-geometry
  --no-use-spatial-film
  --no-use-wavelet-branch
  --ssl-mode none
  --ssl-weight 0
  --seg-ce-weight 1
  --dice-weight .5
  --boundary-weight 0
  --fiducial-weight 0
  --delineation-hidden 96
  --delineation-kernel 15
  --zscore-norm
)

for mode in direct frozen_pca learned; do
  run_name="prs_${mode}_k5_s42_l0"
  run_dir="$output_root/$run_name"
  if [[ -f "$run_dir/_SUCCESS.json" ]]; then
    echo "SKIP completed $run_name"
    continue
  fi
  echo "START $run_name $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$python_bin" scripts/train_1lead_wavelet_ssl_mtl.py \
    "${common[@]}" \
    --output-representation "$mode" \
    --run-name "$run_name" \
    --output-dir "$run_dir" \
    2>&1 | tee "$output_root/${run_name}.log"
  echo "COMPLETE $run_name $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

echo "QUEUE_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
