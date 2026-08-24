#!/bin/bash
set -e

# Run linear probe on best models from all three architectures
# Assuming models are in checkpoints/

TARGETS=("rv_systolic_dysfunction_moderate_or_greater_flag" "shd_moderate_or_greater_flag" "lvef_lte_45_flag")

for TARGET in "${TARGETS[@]}"; do
    echo "========================================"
    echo "Evaluating Target: $TARGET"
    echo "========================================"
    
    echo "Running MSVAE linear probe..."
    python scripts/evaluate_linear_probe.py --arch msvae --ckpt checkpoints/best_msvae.pt --target $TARGET || echo "Failed or checkpoint missing."

    echo "Running UNet linear probe..."
    python scripts/evaluate_linear_probe.py --arch unet --ckpt checkpoints/best_unet.pt --target $TARGET || echo "Failed or checkpoint missing."

    echo "Running ECG-AIM linear probe..."
    # Change to the best ecg_aim checkpoint you have
    python scripts/evaluate_linear_probe.py --arch ecg_aim --ckpt checkpoints/best_ecg_aim.pt --target $TARGET || echo "Failed or checkpoint missing."
done

