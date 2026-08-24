#!/bin/bash
# Smoke test: 3 architectures × Lead I, 2 batches each
set -e

echo "======================================"
echo " 1-LEAD SMOKE TEST (2 batches each)"
echo "======================================"

MASKS=("1000000")

for ARCH in unet msvae ecg_aim; do
    echo ""
    echo "--- Testing $ARCH ---"
    ~/.venv/bin/python3 scripts/train_1lead_factorial_multimodel.py \
        --architecture $ARCH \
        --factorial_mask 1000000 \
        --seed 201 \
        --observed_leads 0 \
        --run_name smoke_1lead_${ARCH}_l0 \
        --checkpoint_path /tmp/smoke_1lead_${ARCH}.pt \
        --max_batches 2 \
        --epochs 1 \
        && echo "✅  $ARCH PASSED" \
        || echo "❌  $ARCH FAILED"
done

echo ""
echo "======================================"
echo " SMOKE TEST COMPLETE"
echo "======================================"
