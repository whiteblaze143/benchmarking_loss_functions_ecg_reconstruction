# Experiment Tracker: Paper 03
# Phase-Path-Signature: Distributional Path Signatures on Cardiac Phase Trajectories

## Dataset & Environment Specs
- **Benchmark**: PTB-XL v1.0.3 (12-lead ECG, 500 Hz, 10-second recordings)
- **Folds**: Fit folds 1-7 (15,244 records), Selection fold 8 (2,173 records), Test folds 9-10
- **Artifact Path**: `/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper03_path_signature/development_representations`
- **Output Directory**: `experiments/ptbxl_distributional_repecg/outputs/paper03_signature_path`

## Development Gate Verification Checklist
- [x] **Nyström Representation Audit**: Spearman $\rho = 0.9106 \ge 0.90$, Median relative error $= 0.0610 \le 0.15$ (Passed).
- [x] **Descriptor Structure Audited**: Depth 3, dimension 228 ($3 \times 8 + 204$), PCA to 64 dims, 128 landmarks.
- [x] **Synthetic Recovery Suite**: 6 worlds covering Chen identity, reparameterization monotonicity, multiset order destruction, Nyström fidelity gate, null-relative chirality separation, and tree-like excursion cancellation.
- [x] **Codebase Updated**: `variants.py`, `train_paper03_shared_grid.py`, and `run_paper03_grid.sh` synchronized with `phase_signature_linear_probe`.

## Development Runs on PTB-XL Fold 8 (Smoke & Grid)
| Variant | Batch | LR | WD | Epochs | Val Macro AUROC | Status | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `full` | 2048 | Grid | Grid | 100 | TBD | Pending | Full distributional path signature |
| `phase_signature_linear_probe`| 2048 | Grid | Grid | 100 | TBD | Pending | Unpooled linear readout ($2048 \to 5$) |
| `linear_probe` | 2048 | Grid | Grid | 100 | TBD | Pending | Mean-pooled linear readout ($128 \to 5$) |
| `phase_mean_signature` | 2048 | Grid | Grid | 100 | TBD | Pending | Mean signature vector baseline |
| `order_scrambled` | 2048 | Grid | Grid | 100 | TBD | Pending | Internal sample order destroyed |
| `time_reversed` | 2048 | Grid | Grid | 100 | TBD | Pending | Trajectory time arrow inverted |
| `monotone_warp_sham` | 2048 | Grid | Grid | 100 | TBD | Pending | Negative control for Chen invariance |
| `unordered_kme` | 2048 | Grid | Grid | 100 | TBD | Pending | Paper 02 voltage KME baseline |
