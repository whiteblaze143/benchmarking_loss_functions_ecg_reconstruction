# Peer Review Summary: Paper 06
## Critical Methodological Review & Author Rebuttals (30-Point Alignment)

### Round 1 Review: Spatial PCA vs Physical Cardiac Dipole
- **Reviewer Critique**: The text claims the top-3 PCA basis is the "3D cardiac dipole" and the residual is "multipolar activity". Top-3 PCA is an empirical statistical basis, not automatically a physical Frank XYZ vectorcardiogram.
- **Author Response**: Fully adopted. We renamed the spatial basis to the **dominant rank-3 spatial macrostate** and the residual to the **non-macro residual**. A secondary Kors-type VCG transformation baseline (`vcg_plus_conditional_residual`) is added to evaluate whether findings generalize to recognized physical lead transformations.

### Round 2 Review: Physical Coordinates vs Standardization
- **Reviewer Critique**: Standardizing leads before PCA reweights covariance by $\operatorname{diag}(1/\sigma_j)$, destroying physical-spatial voltage geometry.
- **Author Response**: Corrected. All spatial covariance fitting and orthogonal decompositions are performed on physical signals in **physical mV units**. Descriptors are standardized only downstream.

### Round 3 Review: Mathematical Terminology: Anchor-Conditioned MMD (AC-MMD)
- **Reviewer Critique**: The metric is not operator-theoretic conditional MMD ($C_{RZ} C_{ZZ}^{-1} \phi(z)$). It is a weighted mixture of local MMDs across soft anchors.
- **Author Response**: Adopted. We formalize the metric as **Anchor-Conditioned MMD (AC-MMD)** / **macrostate-stratified weighted MMD**, explicitly defining the weighting coefficients $\omega_{ghk}$ and local MMD terms.

### Round 4 Review: Ineligibility Gate & 35.8% Selection Bias
- **Reviewer Critique**: The 35.8% exclusion was described as a "mathematical proof of disjoint support". Soft Gaussian weights are positive everywhere; it is an estimator-level sample size cliff. Furthermore, excluding 35.8% of records may introduce severe clinical selection bias.
- **Author Response**: Refactored to **effective-overlap ineligibility cliff**. We implemented a clinical selection bias audit comparing eligible vs ineligible cohorts, dual-cohort reporting ($(\text{performance}, \text{coverage})$), and added overlap mass $O_{gh}$ as an explicit control channel (`macro_overlap_only`).

### Round 5 Review: Primary Matched Comparisons
- **Reviewer Critique**: Comparing `full` vs `macro_component` does not prove residuals add information beyond macrostates because the full model does not contain macro features.
- **Author Response**: Redesigned primary hypotheses to compare $[R_{\text{macro}}, R_{\text{cond-res}}]$ vs $R_{\text{macro}}$ ($H_1$) and vs $[R_{\text{macro}}, R_{\text{marg-res}}]$ ($H_2$).\n