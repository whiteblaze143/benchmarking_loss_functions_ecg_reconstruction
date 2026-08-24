# Reproducibility Statement

## Compute Budget
- **Hardware**: All experiments were run on an NVIDIA A100 (80GB) GPU.
- **Estimated Total Cost**: 12 GPU-hours.
  - Generative Baseline (cNVAE): 4 hours
  - Ablation Grid (5 variants + sweeps): 6 hours
  - Uncertainty Evaluation & Analysis: 2 hours

## Seeds and Hyperparameters
- **Random Seed**: Fixed to `42` across all training and evaluation scripts.
- **Base Learning Rate**: $3 \times 10^{-4}$ (AdamW) for U-Net variants, $1 \times 10^{-3}$ (Adamax) for cNVAE.
- **Batch Size**: 32 for U-Net, 16 for cNVAE (due to memory constraints of generative decoding).
- **Epochs**: 50 for ablations, 200 for cNVAE.

## Code and Data
- **Dataset**: PTB-XL v1.0.3, scaled according to Mason et al. (2024).
- **Environment**: Requirements specified in `requirements.txt`.
- **Training Orchestrator**: Run `scripts/train_ablation_grid.py` to reproduce the full factorial suite.

## Uncertainty Thresholds
- The Monte Carlo dropout variance estimator is calibrated (see ECE plots) using 10 stochastic forward passes per sample during inference.
