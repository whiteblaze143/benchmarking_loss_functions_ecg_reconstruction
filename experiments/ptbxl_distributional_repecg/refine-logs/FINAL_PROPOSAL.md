# repECG Final Proposal: Distributional, Dynamical, and Causal Foundations for Electrocardiography

**Status**: READY FOR SYSTEMATIC EVALUATION  
**Date**: September 17, 2026  
**Target Venues**: NeurIPS / ICML / ICLR / Nature Medicine (Clinical Validation)

---

## 1. Executive Summary & Problem Anchor

Current deep learning architectures in cardiology treat the electrocardiogram (ECG) as a 1D deterministic voltage time series. Despite achieving >0.90 AUROC on closed, benchmark datasets such as PTB-XL, these architectures suffer from three fatal modes of failure when deployed in external clinical workflows:
1. **Shortcut Learning on Physical Hardware**: Models overfit to non-physiological high-frequency powerline signatures, machine-specific sampling rates (250Hz vs 500Hz), and analog filter characteristics rather than electrophysiological pathology.
2. **Phase Dispersion and Heart-Rate Non-Invariance**: Fixed-window architectures fail to separate heart rate (RR-interval variability) from morphological cardiac conduction delays, conflating physiological tachycardia with ischemic repolarization abnormalities.
3. **Causal Conflation of Observation with Mechanism**: Supervised models cannot distinguish between the underlying biological 3D dipole vector of the heart ($Z_S$) and the measurement operator ($q$, electrode placement and lead configuration), leading to catastrophic failure under lead displacement or reduced lead subsets.

### The Thesis of repECG
We replace the deterministic voltage paradigm with a **distributional, dynamical, and causal framework**:
$$\text{Raw Voltage Sequence} \longrightarrow \text{Phase-Aligned Standardized Grid} \longrightarrow \text{RKHS Kernel Mean Embeddings (KME)} \longrightarrow \text{Dynamical / Causal Operators}$$
By mapping localized cardiac phase cells into a Reproducing Kernel Hilbert Space (RKHS) via characteristic kernels (Inverse Multiquadric / Gaussian RBF) approximated via Nyström anchors, every cardiac beat is represented as a sequence of empirical probability distributions. Over this distributional representation, we formulate 15 distinct, mathematically grounded architectures spanning non-linear recurrence, rough path theory, Koopman operator theory, and Judea Pearl's structural do-calculus.

---

## 2. The 15 Core Inventions & Dominant Contributions

### Group A: Distributional & Geometric Representations (Papers 01–03)
*   **Paper 01: Distributional Recurrence Operator (`RecurrenceCNN`)**
    *   *Problem Anchor*: Long-range non-linear interactions across the cardiac cycle (e.g., P-wave atrial depolarization influencing ST-T ventricular repolarization) are obscured in raw 1D convolutions.
    *   *Dominant Contribution*: Constructs an explicit $16 \times 16$ Maximum Mean Discrepancy (MMD) recurrence matrix $R_{ij} = \|\mu_i - \mu_j\|_{\mathcal{H}}^2$ measuring distributional distance between all phase cell pairs in RKHS.
    *   *Rejected Complexity*: Graph neural networks with learned edge weights; closed-form MMD provides an unparameterized, mathematically exact metric.
*   **Paper 02: Local Phase Kernel Mean Embeddings (`PhaseCNN`)**
    *   *Problem Anchor*: Dense $N \times N$ recurrence matrices incur quadratic overhead and lose local temporal transition ordering.
    *   *Dominant Contribution*: Directly feeds the temporal sequence of KME vectors through circular residual 1D convolutions with exact periodicity matching the cardiac cycle.
    *   *Rejected Complexity*: Unidirectional LSTMs that violate the cyclic topology of heartbeats.
*   **Paper 03: Phase-Cell Signature Path (`PathSignatureClassifier`)**
    *   *Problem Anchor*: Extreme biological time-warping (arrhythmias, premature ventricular contractions) alters traversal velocity along the phase trajectory.
    *   *Dominant Contribution*: Lifts discrete KME trajectories into continuous paths and computes truncated iterated integrals (path signatures) providing complete invariance to monotonic time reparameterization.
    *   *Rejected Complexity*: Dynamic Time Warping (DTW) alignment which requires $O(T^2)$ pairwise dynamic programming.

### Group B: Dynamical Systems & Operators (Papers 04–05)
*   **Paper 04: Hankel Dynamics & Dynamic Mode Decomposition (`HankelDynamicsModel`)**
    *   *Problem Anchor*: Ischemic cardiac tissue delays repolarization, causing local exponential decay variations that deep nets fail to isolate.
    *   *Dominant Contribution*: Constructs multi-lag Hankel matrices across phase cells and applies Dynamic Mode Decomposition (DMD) to extract complex eigenvalues corresponding to biological decay and oscillation frequencies.
    *   *Rejected Complexity*: Non-linear neural ODEs that are sensitive to numerical stiffness and optimization instability.
*   **Paper 05: The Koopman Operator in Lifted RKHS (`KoopmanOperatorModel`)**
    *   *Problem Anchor*: The heart's electrical system is a chaotic, non-linear dynamical system.
    *   *Dominant Contribution*: Exploits Koopman operator theory: non-linear dynamics become strictly linear when lifted into infinite-dimensional RKHS. A linear operator $K$ predicts forward state transitions: $Z_{t+1} = K Z_t$.
    *   *Rejected Complexity*: Non-linear autoregressive Transformers for next-step prediction.

### Group C: Statistical Independence & Multi-Head Bottlenecks (Papers 06–08)
*   **Paper 06: Conditional RepStat (`ConditionalRepStatModel`)**
    *   *Problem Anchor*: Standard networks suffer from collinear redundancy, double-counting diagnostic information already present in earlier waves.
    *   *Dominant Contribution*: Employs Conditional MMD to project each phase state $Z_g$ onto the orthogonal complement of preceding states $Z_{<g}$ and patient demographics (Age, Sex).
    *   *Rejected Complexity*: Adversarial minimax decorrelation penalties which are notoriously unstable during training.
*   **Paper 07: Operator Reconstruction Auxiliary (`OperatorReconstructionAuxiliary`)**
    *   *Problem Anchor*: Purely discriminative models discard biophysical voltage realities to optimize cross-entropy loss.
    *   *Dominant Contribution*: Dual-head architecture pairing a classification head with an 8-lead physical-mV generative decoder forced to reconstruct the raw ECG from the KME latent bottleneck.
    *   *Rejected Complexity*: Pixel-level GANs; a simple L1/L2 physical reconstruction loss guarantees complete biophysical retention.
*   **Paper 08: Local Token Cross-Attention (`LocalTokenCrossAttention`)**
    *   *Problem Anchor*: Static convolutional kernels cannot dynamically route attention between early and late cardiac intervals on a patient-specific basis.
    *   *Dominant Contribution*: Formulates phase cells as distinct tokens and applies multi-head self- and cross-attention to allow dynamic, interpretable routing across cardiac phases.
    *   *Rejected Complexity*: Full-sequence self-attention over 5000 raw time samples; 16 phase tokens provide optimal temporal abstraction.

### Group D: Causal Inference & Invariant Mechanisms (Papers 09–15)
*   **Paper 09: Counterfactual Measurement-Operator (`CounterfactualMeasurementOperator`)**
    *   *Problem Anchor*: Physical electrode placement variations cause apparent morphology shifts that are purely artifactual.
    *   *Dominant Contribution*: Disentangles intrinsic 3D cardiac dipole state $Z_S$ from spatial measurement operator $q$. Predicts counterfactual leads: $g_\theta(Z_S, q_*)$.
*   **Paper 10: Interventional RepStat (`InterventionalRepStatModel`)**
    *   *Problem Anchor*: Hospital-specific hardware filters and sampling rates confound multi-center clinical trials.
    *   *Dominant Contribution*: Factorizes representation into intervention-predictive latents $Z_A$ and diagnostic latents $Z_S$, enforcing $MMD(Z_S \mid \text{Hospital}_A, Z_S \mid \text{Hospital}_B) \to 0$.
*   **Paper 11: Causal-State ECG via $\epsilon$-Machines (`CausalStateECGModel`)**
    *   *Problem Anchor*: Heuristic clinical definitions of cardiac "states" ignore statistical sufficiency.
    *   *Dominant Contribution*: Discovers minimal sufficient causal states using Computational Mechanics: histories producing identical future distributions $P(F \mid h_i) = P(F \mid h_j)$ are clustered into unique causal states $\epsilon_k$.
*   **Paper 12: Structural-Innovation repECG (`StructuralInnovationModel`)**
    *   *Problem Anchor*: Normal sinus rhythm is 95% predictable; clinical pathology resides in unpredictable structural innovations.
    *   *Dominant Contribution*: Uses an autoregressive conditional normalizing flow to isolate the structural innovation $U_g = f_\theta^{-1}(Z_g; Z_{<g})$.
*   **Paper 13: Counterfactual Distribution Surgery (`CounterfactualSurgeryModel`)**
    *   *Problem Anchor*: Saliency maps (Grad-CAM) indicate correlation, not counterfactual necessity or sufficiency.
    *   *Dominant Contribution*: Implements Pearl's $do()$ calculus by surgically splicing healthy reference distributions into pathological phase slots: $do(P_g = P_g^{ref})$.
*   **Paper 14: Invariant Mechanism Discovery with repStat (`InvariantMechanismDiscoveryModel`)**
    *   *Problem Anchor*: Cross-hospital generalization collapses due to environment-specific spurious features.
    *   *Dominant Contribution*: Invariant Risk Minimization (IRM) across 9 diverse multi-center datasets, enforcing that the optimal classifier $w$ is invariant across all environments $e \in \mathcal{E}$.
*   **Paper 15: Causal Mechanism Factorization (`CausalMechanismFactorizationModel`)**
    *   *Problem Anchor*: Disease alters states, but fundamental cardiac electrodynamics (mechanisms) should remain autonomous.
    *   *Dominant Contribution*: Independent Causal Mechanisms (ICM) factorizing the cardiac cycle into autonomous transition modules $M_g: Z_g \to Z_{g+1}$.

---

## 3. Key Claims & Falsification Protocols (The "Kill Tests")

Every paper in this suite is equipped with a formal, falsifiable hypothesis:

| Paper | Main Claim | The Kill Test (Adversarial Falsification) |
| :--- | :--- | :--- |
| **01** | RKHS MMD distance captures non-linear recurrence better than Euclidean moments. | Chronological phase shuffling destroys performance; naive moment control underperforms by $>0.05$ AUROC. |
| **02** | Circular 1D residual convolution over KMEs outperforms raw voltage 1D-ResNets. | Heavy temporal over-smoothing collapses diagnostic boundaries; failure on circular boundary destroys cyclic advantage. |
| **03** | Path signatures are strictly invariant to non-linear heart rate time-warping. | Synthetic time-warping in test set degrades raw CNN by $>0.15$ AUROC but degrades signature classifier by $<0.02$. |
| **04** | Hankel DMD eigenvalues isolate ischemic decay rates without parameter learning. | Chronological shuffling shatters Hankel matrix rank and destroys diagnostic prediction. |
| **05** | The lifted KME space linearizes cardiac dynamics via Koopman operators. | High forward projection MSE ($Z_{t+1} \approx K Z_t$) directly indicates falsification of linearity in RKHS. |
| **06** | Orthogonalized conditional KMEs eliminate diagnostic double-counting. | Synthetic collinear feature injection does not degrade conditional model, but inflates baseline variance. |
| **07** | Dual-head physical reconstruction bottleneck prevents shortcut learning. | Model maintains $>95\%$ diagnostic accuracy while achieving $<0.05$ normalized MSE on 8-lead physical reconstruction. |
| **08** | Transformer attention dynamically routes between disease-specific phase intervals. | Attention head randomization or uniform masking drops macro AUROC significantly on focal pathologies. |
| **09** | 3D dipole state is invariant to measurement operator $q$. | Mismatched lead projection operator collapses counterfactual reconstruction. |
| **10** | Hardware intervention latents $Z_A$ isolate recording artifacts from diagnosis. | Random label perturbation fails to trigger MMD invariance benefit; works specifically on known acquisition tags. |
| **11** | $\epsilon$-machine causal states achieve minimal statistical complexity $C_\mu$. | Future predictive entropy is minimized compared to non-causal clustering. |
| **12** | Structural innovations $U_g$ carry higher per-parameter diagnostic mutual information than raw phase states. | Time-reversal of autoregressive conditioning destroys innovation diagnostic utility. |
| **13** | Counterfactual $do(P_g = P_g^{ref})$ proves necessity and sufficiency of ST-T wave in MI. | Random distribution replacement fails to reverse disease prediction; targeted replacement achieves $>0.80$ causal flip rate. |
| **14** | IRM across 9 clinical environments learns representations invariant to hospital domain. | Shuffled environment labels collapse IRM gradient penalty advantage to baseline ERM. |
| **15** | Independent causal mechanisms $M_g$ transfer autonomously to unseen pathological regimes. | Mechanism parameter entanglement collapses zero-shot domain adaptation. |

---

## 4. Resource & Compute Allocation
- **Dataset Foundations**: Shared precomputed representations on PTB-XL (21,799 12-lead ECGs) and 9 OOD datasets (EchoNext, LUDB, RDB, ISP, Kingston, Emory, Sunnybrook, Zhejiang).
- **GPU Budget**: Each paper requires ~15–30 minutes on a single NVIDIA A100 GPU utilizing shared-tensor CUDA streams and BF16 mixed precision.
- **Total Compute**: 15 papers $\times$ 9 grid cells $\approx$ 4.5 hours total runtime across the entire suite.
