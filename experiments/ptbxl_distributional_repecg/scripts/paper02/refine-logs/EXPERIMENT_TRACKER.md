# Experiment Tracker: Paper 02
# Phase-KME: Phase-Conditioned Kernel Mean Embeddings for Distribution-Valued ECG Representation

| Variant ID | Variant Name | Status | Rep Key | Arch | Readout | Target Claim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| P02-V01 | `full` | **Ready** | `kernel` | Circular PhaseCNN | GAP | Proposed Phase-KME Model |
| P02-V02 | `global_kme` | **Ready** | `kernel` (mean) | MLP | Linear | Scientific Reports 2025 prior-art baseline |
| P02-V03 | `no_circular` | **Ready** | `kernel` | Standard PhaseCNN | GAP | Discrete $C_{16}$ boundary prior ablation |
| P02-V04 | `moments_circular` | **Ready** | `moments` | Circular PhaseCNN | GAP | Distributional content beyond 1st/2nd moments |
| P02-V05 | `moments_noncircular` | **Ready** | `moments` | Standard PhaseCNN | GAP | Interaction of moments and cyclic prior |
| P02-V06 | `gaussian_surrogate` | **Ready** | `gaussian` | Circular PhaseCNN | GAP | **Destroyer**: Higher-order non-Gaussian shape |
| P02-V07 | `linear_kernel` | **Ready** | `linear` | Circular PhaseCNN | GAP | Classical SAECG (1st moment) baseline |
| P02-V08 | `rbf_kme` | **Ready** | `rbf_kernel` | Circular PhaseCNN | GAP | Characteristic kernel sensitivity (IMQ vs RBF) |
| P02-V09 | `phase_kme_linear_probe` | **Ready** | `kernel` (vec) | Ridge Linear | Linear | Representation linear separability ($4096$-dim) |
| P02-V10 | `global_kme_linear_probe`| **Ready** | `kernel` (mean) | Ridge Linear | Linear | Global KME linear probe |
| P02-V11 | `phase_shuffle` | **Ready** | `kernel` (shuffled)| Circular PhaseCNN | GAP | Phase chronological mechanism check |
| P02-V12 | `deepsets_phase` | **Ready** | Raw $\to$ DeepSets | Circular PhaseCNN | GAP | Fixed RKHS vs learned set encoder |

## Synthetic Recovery Proof Status
- [x] **Test World A**: Non-Gaussian Sensitivity with Matched Mean and Covariance ($D_{\text{alt}} > Q_{0.99}(D_{\text{null}})$).
- [x] **Test World B**: Moment-Matched Gaussian Surrogate Precision (Mean & Cov error $< 10^{-8}$).
- [x] **Test World C**: Discrete $C_{16}$ Cyclic Shift Equivariance & Readout Invariance.
- [x] **Test World D**: Nyström Operating Point Fidelity (Spearman $\rho_S \ge 0.95$, Median RE $\le 0.15$).
- [x] **Test World E**: Sample-Size Identifiability Power Curve across Beat Count $B \in \{2, \dots, 32\}$.
