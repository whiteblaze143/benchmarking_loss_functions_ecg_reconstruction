# Refinement Report: Paper 06
## Systematic Modifications from Initial Blueprint to Production Specification

| Component | Initial Blueprint | Refined Specification | Mathematical Justification |
| :--- | :--- | :--- | :--- |
| **Spatial Basis** | "Physical cardiac dipole loop" | Dominant rank-3 spatial macrostate | PCA eigenvectors are empirical, data-dependent bases |
| **Residual Terminology** | "Localized multipolar potentials" | Non-macro residual | Residual contains electrode noise and body conduction in addition to multipoles |
| **Voltage Units** | Per-lead standardized coordinates | Physical mV coordinates | Prevents $\operatorname{diag}(1/\sigma_j)$ distortion of spatial covariance geometry |
| **Residual Coordinates** | Ambient $\mathbb{R}^8$ (rank-5) | Orthonormal complement $V_\perp \in \mathbb{R}^{8 \times 5} \implies u \in \mathbb{R}^5$ | Removes redundant coordinates from Nyström kernel embeddings |
| **Metric Name** | "Conditional MMD" | Anchor-Conditioned MMD (AC-MMD) | Precisely describes macrostate-stratified weighted mixture of local MMDs |
| **Ineligibility Framing** | "Proof of disjoint support" | Effective-overlap ineligibility cliff ($n_{\text{eff}} < n_{\min}$) | Gaussian weights are strictly positive everywhere; gate is estimator-level |
| **Clinical Audit** | None | Systematic selection bias audit across superclasses | Prevents conditioning away pathological patient populations |
| **Reporting Standard** | Eligible records only | Dual reporting: Eligible-only + $(\text{performance}, \text{coverage})$ | Clinical comparability across complete cohorts |
| **Primary Comparison** | `full` vs `macro_component` | $[R_{\text{macro}}, R_{\text{cond-res}}]$ vs $R_{\text{macro}}$ | Rigorously measures incremental gain beyond macro information |
| **Conditioning Control** | Marginal MMD alone | $[R_{\text{macro}}, R_{\text{cond-res}}]$ vs $[R_{\text{macro}}, R_{\text{marg-res}}]$ | Equalizes macro information; isolates conditioning mechanism |
| **Null Construction** | Global residual shuffle | Within-macrostate phase shuffle | Preserves $P(r \mid z)$ while destroying $R \perp G \mid Z$ |
| **VCG Control** | Absent | Kors regression matrix transform | Answers criticism that PCA is not a true physical dipole |
| **Overlap Control** | Absent | `macro_overlap_only` ($O_{gh}$) | Proves performance is not driven by overlap mass alone |\n