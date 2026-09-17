# Final Research Proposal: Paper 06
## Macrostate-Conditioned Residual Recurrence: Disentangling Dominant Spatial Fields from Phase-Specific Residual ECG Distributions

### 1. Abstract & Defensible Novelty Boundary
In multi-lead electrocardiography, surface potentials are dominated by a low-dimensional cardiac electric field vector that traverses a continuous 3D loop during each heartbeat. While classical vectorcardiography (VCG, Frank, 1956; Kors et al., 1990) and non-dipolar residual analysis (Titomir et al.; Schimpf et al., 2003) have a 40-year history of estimating residual energy, and statistical learning has formalized conditional two-sample testing (Fukumizu et al., 2008; Song et al., 2013; Park & Muandet, 2020), Paper 06 establishes a distinct, defensible methodology:

$$\boxed{ x(t) - \bar{x} = \underbrace{V_{\text{macro}} z(t)}_{\text{dominant rank-3 spatial macrostate } \in \mathbb{R}^3} + \underbrace{V_\perp u(t)}_{\text{non-macro residual } \in \mathbb{R}^5} }$$

$$\boxed{ D_{\text{ACMMD}}^2(g, h) = \sum_{k \in \mathcal{K}_{gh}} \omega_{ghk} \operatorname{MMD}^2\Big(P(u \mid g, z \approx \mathbf{a}_k), P(u \mid h, z \approx \mathbf{a}_k)\Big) }$$

Rather than assuming generic "multipolar dipole subtraction", Paper 06 explicitly addresses the central scientific question:
**After controlling for the dominant global spatial ECG component, is there repeatable diagnostic structure in the remaining phase-conditioned residual distribution?**

### 2. Electrophysiological & Mathematical Ontology

#### 2.1 PCA Top-3 vs Physical Dipole
The top 3 spatial principal eigenvectors $V_{\text{macro}} \in \mathbb{R}^{8 \times 3}$ span the subspace of maximum variance in the population covariance. Mathematically, this is a rank-3 PCA subspace. While motivated by the lumped dipole model of cardiac electrophysiology, its axes are empirical, data-dependent, and defined up to orthogonal rotations and sign flips. We strictly designate $V_{\text{macro}}$ as the **dominant rank-3 spatial macrostate**, reserving physical dipole claims for secondary validation against recognized Frank/Kors VCG transformations.

#### 2.2 Non-Macro Residual vs Multipolar Activity
The projection $r(t) = (I - V_{\text{macro}} V_{\text{macro}}^\top)(x(t) - \bar{x})$ satisfies $V_{\text{macro}}^\top r(t) = \mathbf{0}$. We designate $r(t)$ mathematically as the **non-macro residual**. While one electrophysiological hypothesis is that $r(t)$ reflects localized conduction delays, fragmented activation, and regional repolarization dispersion, it also captures lead-specific noise, electrode placement variability, thoracic volume conductor geometry, and morphologies outside the population-average subspace.

#### 2.3 Physical mV Coordinates
Spatial covariance estimation and decomposition are performed on physical voltage signals in **physical mV units** (not per-lead z-scored signals), ensuring that principal spatial directions preserve the true anatomical dipole geometry rather than being distorted by diagonal scale normalization $\operatorname{diag}(1/\sigma_j)$.

#### 2.4 Explicit 5D Orthonormal Complement ($V_\perp$)
Rather than computing kernel embeddings on rank-deficient 8D ambient residuals, we construct an explicit orthonormal complement $V_\perp \in \mathbb{R}^{8 \times 5}$ satisfying:
$$V_{\text{macro}}^\top V_\perp = \mathbf{0}_{3 \times 5}, \quad V_\perp^\top V_\perp = I_5$$
The residual state is parameterized in genuine 5-dimensional coordinates $u(t) = V_\perp^\top (x(t) - \bar{x}) \in \mathbb{R}^5$, eliminating coordinate redundancy.

### 3. Estimator: Anchor-Conditioned MMD (AC-MMD)
The estimator is formally defined as **Anchor-Conditioned MMD (AC-MMD)** (a macrostate-stratified weighted mixture of local MMDs):
$$D_{\text{ACMMD}}^2(g, h) = \sum_{k \in \mathcal{K}_{gh}} \omega_{ghk} \operatorname{MMD}^2(P_g^k, P_h^k), \quad \omega_{ghk} = \frac{\sqrt{p_g(k) p_h(k)}}{\sum_{\ell \in \mathcal{K}_{gh}} \sqrt{p_g(\ell) p_h(\ell)}}$$
where:
- $w_k(z) = \frac{\exp(-\|z - \mathbf{a}_k\|^2 / \sigma^2)}{\sum_j \exp(-\|z - \mathbf{a}_j\|^2 / \sigma^2)} \in \Delta^{15}$ is the soft RBF membership over $K=16$ global macro anchors.
- $P_g^k$ denotes the residual distribution in phase $g$ softly restricted to macro anchor $k$.
- $\mathcal{K}_{gh} = \{k : n_{\text{eff}}(g, k) \ge n_{\min} \text{ and } n_{\text{eff}}(h, k) \ge n_{\min}\}$ is the active shared support set.
This construction directly compares residual distributions **only within approximately matched macrostates**, aggregating local differences over shared macrostates.

### 4. The Effective-Overlap Ineligibility Cliff & Selection Bias Audit
In PTB-XL, **5,464 out of 15,244 training records (35.8%)** failed the gate $n_{\text{eff}} < 2.0$.
1. **Estimator-Level Cliff**: Because Gaussian weights are strictly positive ($w_k > 0$), this exclusion does not prove disjoint true probabilistic support; it represents **insufficient effective sample overlap under the chosen estimator and anchor resolution**.
2. **Selection Bias Audit**: The excluded population is audited for systematic clinical selection bias: evaluating diagnosis prevalence across superclasses (MI, CD, HYP, STTC, NORM), heart rate, QRS amplitude, signal quality, and lead amplitudes.
3. **Dual-Cohort Reporting**:
   - *Population A (Eligible-only)*: Evaluates the conditional estimator on records satisfying the overlap gate.
   - *Population B (Full cohort)*: Reports joint $(\text{performance}, \text{coverage})$ where $\text{coverage} = N_{\text{eligible}} / N_{\text{total}}$.
4. **Overlap Mass Feature Channel**: Overlap mass $O_{gh} = \sum_k \sqrt{p_g(k) p_h(k)}$ is added as an explicit feature channel (`macro_overlap_only`) to test whether overlap alone carries diagnostic signal.

### 5. Primary Matched Comparisons
- **Residual Relevance ($H_1$)**:
  $$\boxed{ [R_{\text{macro}}, R_{\text{cond-res}}] \quad \text{vs} \quad R_{\text{macro}} }$$
  Measuring incremental predictive gain $\Delta_{\text{res}} = \operatorname{AUROC}([R_{\text{macro}}, R_{\text{cond-res}}]) - \operatorname{AUROC}(R_{\text{macro}})$.
- **Conditioning Benefit ($H_2$)**:
  $$\boxed{ [R_{\text{macro}}, R_{\text{cond-res}}] \quad \text{vs} \quad [R_{\text{macro}}, R_{\text{marg-res}}] }$$
  Both models possess identical macro information; only residual conditioning varies.
- **Conditional Coupling ($H_3$)**:
  $$\boxed{ [R_{\text{macro}}, R_{\text{cond-res}}] \quad \text{vs} \quad [R_{\text{macro}}, R_{\text{phase-shuf}\mid Z}] }$$
  Within-macrostate phase shuffling tests the null hypothesis $R \perp G \mid Z$.\n