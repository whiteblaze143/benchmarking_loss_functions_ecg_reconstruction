# Multi-Round Review Summary: Paper 01 (Distributional Recurrence Operator)

## Round 1: Literature Prior Art Audit & Theoretical Redirection

### Reviewer 1 Assessment
**Verdict**: Rejection / Mandatory Method Redirection

#### Critique 1.1: Direct Prior Art in ECG Recurrence Plots & CNNs
> "A search of PubMed, IEEE, and GitHub reveals that feeding 2D recurrence plots into CNNs is already established prior art in the ECG domain:
> - Mathunjwa et al. (2022) converted 12-lead ECGs to 2D recurrence plots and classified them with ResNet architectures.
> - Earlier works (e.g., Inception-ResNet-v2 on recurrence plots, recurrence quantification in cardiac non-linear dynamics dating back decades).
> - Multiple public GitHub repositories implement ECG-RP-CNN pipelines.
> The proposed method currently defines $R_{ij} = \|x_i - x_j\|_2^2$ over 16 phase-normalized bins and feeds this into a small 2D CNN. The paper as written is merely a compact Euclidean recurrence-plot CNN, lacking novel representation theory."

#### Critique 1.2: Misleading "Distributional" Nomenclature
> "The title claims a 'Distributional Recurrence Operator', but the mathematical formulation operates strictly on single mean vectors $x_g \in \mathbb{R}^D$ per phase bin. Computing $R_{gh} = \|x_g - x_h\|^2$ is ordinary geometric point recurrence. Where is the distribution? Where is the beat-to-beat variability?"

#### Critique 1.3: Unsubstantiated "Exact" Invariance Claims & Duration Erasure
> "Claiming 'exact translation invariance across variable heart rates' is mathematically invalid. Phase normalization relies on heuristic R-peak detection, linear or spline interpolation, and ignores unequal systolic/diastolic duration scaling. Furthermore, normalizing out time deletes fundamental clinical biomarkers ($PR$, $QRS$, $QT$ intervals). Claiming duration is 'implicitly encoded in morphology' is an unverified assertion."

---

### Author Defense & Architectural Adaptation (Round 1)

#### Response to 1.1 & 1.2: Pivot to True Distributional MMD Recurrence
> We concede that Euclidean point recurrence has direct prior art. We therefore elevate the central contribution to **Distributional Recurrence between Phase-Conditioned ECG State Distributions**:
> 1. Each phase bin $g$ is treated as an empirical probability measure $P_g = \mathcal{L}\{X_b(s) : s \in C_g, b=1,\dots,B\}$ over all valid beats in the recording.
> 2. We lift each distribution to an RKHS via Kernel Mean Embeddings $\mu_{P_g} = \mathbb{E}_{X \sim P_g}[\phi(X)]$.
> 3. We compute pairwise distances using the squared Maximum Mean Discrepancy (MMD):
>    $$D_{gh} = \operatorname{MMD}^2(P_g, P_h) = \|\mu_{P_g} - \mu_{P_h}\|_{\mathcal{H}}^2$$
> 4. The recurrence operator is defined as:
>    $$\boxed{ R_{gh} = \exp\left( -\frac{\operatorname{MMD}^2(P_g, P_h)}{\tau} \right) }$$
> Existing Euclidean point recurrence is explicitly demoted to a baseline (`euclidean_phase_recurrence`).

#### Response to 1.3: Factored Morphology and Duration
> We strike the word "exact" from all invariance claims. To preserve interval biomarkers rigorously, we decouple duration:
> - Primary representation: Duration-free distributional recurrence $R_{\text{distributional}}$.
> - Duration feature vector: $d = [d_{RR}, d_{PR}, d_{QRS}, d_{QT}]$.
> - Composite model: $Z_{\text{record}} = [R_{\text{distributional}}, d]$.
> - Hypothesis test: Directly benchmark **Recurrence only** vs **Duration only** vs **Recurrence + Duration**.

---

## Round 2: Adversarial Stress-Testing of the Distributional Formulation

### Reviewer 2 Assessment
**Verdict**: Major Revision (Experimental Rigor & Falsification Requirements)

#### Critique 2.1: Flawed Amplitude Normalization
> "Defaulting to intra-record variance normalization erases essential diagnostic amplitude criteria (low-voltage QRS in amyloidosis/effusion, left ventricular hypertrophy voltage criteria, focal infarction attenuation). Normalization cannot be a blanket default."

#### Critique 2.2: Phase Permutation is Too Coarse
> "The plan uses a single `phase_content_permuted` control. A fixed permutation applied globally can simply be memorized by a 2D CNN with zero performance loss. You need a 3-way permutation battery: fixed global permutation, record-specific permutation, and within-cell sample scrambling."

#### Critique 2.3: Equivariance Contradiction in Cyclic Sham
> "You claim cyclic boundary robustness, yet evaluate a standard 2D CNN with zero-padding that is not cyclic-equivariant. Either use 2D circular padding `F.pad(x, (1,1,1,1), mode='circular')` and prove equivariance via $\Delta_{\text{cyclic}}$ and CKA, or abandon the invariance claim."

#### Critique 2.4: Meaningless Benchmark Target (>0.75)
> "Modern PTB-XL benchmarking (Strodthoff et al., 2021) establishes that xResNet1d101 achieves 0.928 and ResNet1D-Wang achieves 0.930 on diagnostic superclasses. Claiming success at $>0.75$ is substandard. You must target noninferiority to raw waveform SOTA ($\Delta \operatorname{AUROC} > -0.01$) and demonstrate superiority in sparse-lead settings."

#### Critique 2.5: Missing Prior-Art Replications & Synthetic Grounding
> "To be publishable, you must:
> 1. Replicate classical thresholded RP + RQA features (XGBoost baseline).
> 2. Replicate RP + ResNet-18 as an architectural control.
> 3. Add representation stability analysis (odd-beat vs even-beat split-half stability NFS and spectral correlation $\rho_{\text{spectral}}$).
> 4. Test on 4 synthetic recovery worlds (Synthetic A: identical means, distinct distributions; Synthetic B: finite sample convergence; Synthetic C: cyclic shift; Synthetic D: scale variance).
> 5. Execute a GraphECG-style sparse-lead degradation battery (12, 6, 3, 2, 1, ICM)."

---

### Author Defense & Rebuttal Resolution (Round 2)

#### Response to 2.1: Scale Disentanglement Matrix
> We implement explicit scale separation: $x(t) = a \cdot \tilde{x}(t)$ where $a = \sqrt{\frac{1}{T}\sum_t \|x(t)\|^2}$. We build recurrence $R_{\text{shape}}$ on unit-scale $\tilde{x}$, and append scale $a$ separately as $[R_{\text{shape}}, a]$. We also run the factorial matrix ($X_{\text{physical}}$, $X_{\text{train standardized}}$, $X_{\text{record standardized}}$).

#### Response to 2.2: 3-Way Permutation Hierarchy
> We implement:
> 1. `fixed_phase_permutation`: Same permutation $\pi$ across all ECGs (tests coordinate relabeling).
> 2. `recordwise_phase_permutation`: Independent $\pi_i$ per ECG (destroys cross-record phase semantics).
> 3. `within_cell_scramble`: Shuffles beat assignments within cells while preserving first-order marginals (tests coherent multi-beat distribution structure).

#### Response to 2.3: Circular Convolution & Metric Equivariance
> We replace standard convolutions with 2D circular residual convolutions:
> `x = F.pad(x, (1,1,1,1), mode='circular'); x = self.conv(x)`
> We measure $\Delta_{\text{cyclic}} = \frac{1}{16}\sum_{k=0}^{15} |p(R) - p(P_k R P_k^\top)|$ and representation CKA across cyclic shifts.

#### Response to 2.4 & 2.5: Expanded 15-Variant Hierarchy, Benchmarks & Synthetics
> We adopt the full 15-variant suite, integrate RQA/XGBoost and ResNet-18 prior-art controls, implement split-half beat stability (NFS), add the test-retest perturbation battery (jitter, noise, gain, subsampling), and execute the 4 synthetic recovery worlds before final PTB-XL production.

---

## Final Review Verdict
$$\boxed{\textbf{MAJOR REVISION — NOT YET PRODUCTION READY}}$$

### Required Pre-Production Milestones
1. Refactor representation generator from Euclidean distance to MMD $U$-statistic across beats.
2. Verify Synthetic Recovery Worlds A–D in offline unit tests.
3. Update `PAPER01_VARIANTS` in `src/repecg/common/variants.py` to support the full 15-variant evaluation hierarchy.
4. Execute split-half stability and sparse-lead degradation batteries.
