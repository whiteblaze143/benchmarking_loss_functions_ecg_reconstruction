# Experiment Plan: Paper 06
## Macrostate-Conditioned Residual Recurrence for ECG

### 1. Locked Scientific Variant Hierarchy

| Variant | Channel 0 | Channel 1 | Conditioning | Scientific Role |
| :--- | :---: | :---: | :---: | :--- |
| **`macro_plus_conditional_residual`** | $R_{\text{macro}}$ | $R_{\text{cond-res}}$ | Yes (AC-MMD) | **Primary Proposed Model** |
| **`macro_only`** | $R_{\text{macro}}$ | Zeros | — | **Macrostate Baseline**: 3D spatial macrostate alone |
| **`macro_plus_marginal_residual`** | $R_{\text{macro}}$ | $R_{\text{marg-res}}$ | No (Marginal) | **Additive Marginal Control**: Conditioning ablation |
| **`conditional_residual_only`** | Zeros | $R_{\text{cond-res}}$ | Yes (AC-MMD) | Residual sufficiency alone |
| **`residual_marginal_only`** | Zeros | $R_{\text{marg-res}}$ | No (Marginal) | Unconditioned residual alone |
| **`vcg_plus_conditional_residual`** | $R_{\text{VCG}}$ | $R_{\text{cond-res}}$ | VCG (Kors) | **Physical-Spatial Control**: Kors VCG vs PCA |
| **`phase_shuffled_within_macro`** | $R_{\text{macro}}$ | $R_{\text{shuf}\mid Z}$ | Within-macro | **Conditional Null**: $R \perp G \mid Z$ |
| **`macro_overlap_only`** | $R_{\text{macro}}$ | $O_{gh}$ | Overlap mass | **Gate/Confounding Control**: Overlap mass alone |
| **`all_pairs`** | $R_{\text{macro}}$ | $R_{\text{cond-res}}$ (unmasked) | Yes | **Mask Ablation**: Retains adjacent phases |
| **`raw_affinity`** | $R_{\text{macro}}$ | $A_{\text{cond-res}}$ (unnorm) | Yes | **Normalization Ablation**: Unnormalized $A$ vs Laplacian $R$ |
| **`linear_probe`** | Upper-triangular features | — | Yes | **Complexity Probe**: Linear classifier probe |

### 2. Hypotheses & Success Criteria
- **$H_1$ (Residual Relevance)**: $\operatorname{AUROC}([R_{\text{macro}}, R_{\text{cond-res}}]) > \operatorname{AUROC}(R_{\text{macro}})$ (Residual adds predictive value beyond dominant macrostate).
- **$H_2$ (Conditioning Benefit)**: $\operatorname{AUROC}([R_{\text{macro}}, R_{\text{cond-res}}]) > \operatorname{AUROC}([R_{\text{macro}}, R_{\text{marg-res}}])$ (Conditioning outperforms equally informative marginal residual).
- **$H_3$ (Conditional Coupling)**: $\operatorname{AUROC}([R_{\text{macro}}, R_{\text{cond-res}}]) > \operatorname{AUROC}([R_{\text{macro}}, R_{\text{phase-shuf}\mid Z}])$ (Residual structure is phase-specific beyond macrostate composition).
- **$H_4$ (Macrostate Robustness)**: Performance is robust to replacing PCA top-3 with a recognized Kors VCG transform.
- **$H_5$ (Overlap Non-Triviality)**: $[R_{\text{macro}}, R_{\text{cond-res}}] > [R_{\text{macro}}, O_{gh}]$ (Performance is not driven solely by effective sample size / overlap mass).

### 3. Hyperparameter Sweep Grid
- Learning rates: $\{1\times 10^{-4}, 3\times 10^{-4}, 1\times 10^{-3}\}$
- Weight decays: $\{1\times 10^{-5}, 1\times 10^{-4}, 1\times 10^{-3}\}$
- Batch size: 2048, Max epochs: 100, Patience: 10, Seed: 42
- Metric: Multilabel macro AUROC, macro AUPRC, macro F1 on PTB-XL validation set.\n