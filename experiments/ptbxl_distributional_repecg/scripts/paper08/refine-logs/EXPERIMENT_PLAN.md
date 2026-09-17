# Experiment Plan: Paper 08

## 1. Overview
Evaluate the Phase-Token Transformer architecture on PTB-XL development representations (`/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper02_kernel_mean/development_representations`).

## 2. Hyperparameter Grid
- **Learning Rates**: `1e-4`, `3e-4`, `1e-3`
- **Weight Decays**: `1e-5`, `1e-4`, `1e-3`
- **Batch Size**: `2048`
- **Optimizer**: AdamW + Cosine Annealing
- **Loss**: Multi-label BCE with positive class prevalence weighting
- **Total Cells per Variant**: 9 cells

## 3. Variant Grid
1. `global_dynamic` (`full`)
2. `local_banded`
3. `static_attention`
4. `uniform_attention`
5. `scrambled_phases`
6. `phase_agnostic_set`
7. `kmeans_tokens`
8. `linear_probe`
9. `cnn_matched_control`
10. `random_feature_control`

## 4. Resource Allocation
- Hardware: Local NVIDIA A100-PCIE-40GB GPU.
- Execution: Detached `tmux` session with shared CUDA streams.
