# Final Proposal: Measurement-Operator Invariant ECG Representations Under Acquisition-Configuration Shift

**Problem Anchor**: Can an ECG representation preserve diagnostic information under a change in measurement operator?  
**Dominant Contribution**: Formulating ECG observations as continuous dual operator-response pairs $(q, s_q(t))$ with projective gauge symmetry $\mathbb{RP}^7$, providing configuration-robust diagnosis across clinical subsets, combinatorial lead drops, and unseen dual measurement functionals.  
**Target Venues**: ICLR / NeurIPS / Nature Medicine (AI & Cardiology)  
**Date**: September 21, 2026  

---

## 1. Executive Summary & The Problem Anchor

Standard deep learning approaches treat the 12-lead electrocardiogram as a fixed-size tensor $X \in \mathbb{R}^{12 \times T}$. This tensor formulation fundamentally conflates two distinct physical quantities:
1. **The intrinsic cardiac source**: The underlying time-varying 3D cardiac electrical dipole vector $Z(t) \in \mathbb{R}^3$.
2. **The measurement operator configuration**: The set of electrodes and lead projections $O_S = \{q_j\}_{j \in S}$ through which the volume conductor is observed.

When the measurement configuration changes—such as when leads detach in an intensive care unit, when only 1 or 2 leads are captured by a smartwatch or handheld patch, or when an implantable cardiac monitor (ICM) records an oblique non-standard vector—fixed-tensor architectures collapse catastrophically.

### The Central Empirical Thesis
$$\boxed{
\text{Can an ECG representation preserve diagnostic information under a change in measurement operator?}
}$$

Instead of viewing topological, canonical-field, or GNN machinery as ends in themselves, we treat them as **candidate solutions to acquisition-configuration shift**.

We evaluate whether explicitly representing observations as operator-response pairs, reconstructing a canonical cardiac field, or extracting marginal moment submatrices preserves diagnostic classification without retraining.

---

## 2. Four Benchmark Architectural Paradigms

To rigorously dissect the source of configuration invariance, we formalize four competing baseline and proposed paradigms:

```
                          Observation: O_S(X) = {(q_j, x_j(t)) : j in S}
                                        │
         ┌───────────────────┬──────────┴──────────┬───────────────────┐
         ▼                   ▼                     ▼                   ▼
    FixedTensor      FixedTensor+Mask           GraphECG          SetOperator / Field
  (Zero-Imputation)   (Mask-Augmented)      (Electrode GNN)      (Continuous Dual RP^7)
   ResNet / ConvNet   [X_mod ; Mask_obs]    Message Passing     Transformer Set Encoder
  (Standard Clinical) (Augmented Baseline) (Ansari et al. 2026) (Paper 07 / Paper 02)
```

1. **`FixedTensor` (Standard Baseline)**:
   - Evaluates standard 8-channel / 12-channel convolutional networks.
   - When inputs disappear, unobserved channels are zero-filled ($Z_0$).
2. **`FixedTensor+Mask` (Augmented Baseline)**:
   - Explicitly provides an observation mask $M \in \{0, 1\}^{C}$ concatenated along the channel dimension ($Z_M$).
3. **`GraphECG` (Electrode-Centric GNN Baseline, Ansari et al., Stanford 2026)**:
   - Models electrodes as discrete nodes with fixed 3D torso coordinates $(x, y, z)$.
   - Models leads as directed graph edges carrying waveform embeddings with spherical harmonic positional encodings.
   - Handles reduced leads by constructing induced subgraphs.
   - *Limitation*: Strictly discrete; cannot process arbitrary linear combinations of leads or continuous dual functionals.
4. **`SetOperator` & `CanonicalField` (Our Continuous Functional Approach)**:
   - **Paper 07 (`OperatorSetModel`)**: Formulates lead measurements as continuous linear functionals $q \in \mathbb{RP}^7 = S^7 / \{q \sim -q\}$ on the 8D dipole measurement span. Invariant to lead permutation, lead count, and lead inversion polarity.
   - **Paper 02 (`Moments`)**: Extracts marginal mean dipole $\mu_S$ and covariance submatrix $\Sigma_{S \times S}$, providing closed-form submatrix marginalization under missing leads.
   - **Topology as an Ablation (`CanonicalField + Topology`)**: Tests whether persistent homology / braid invariants on the reconstructed canonical field $\hat{\Phi}(q, t)$ provide additional invariance beyond the field representation itself.

---

## 3. The Four-Tier Evaluation Framework

We institute a strict four-tier validation hierarchy to prevent over-optimistic or cherry-picked claims:

*   **Tier 1: Engineering Plumbing Smokes**: 1-epoch runs verifying computational graph connectivity and gradient flow. Stamped `engineering_only_not_a_result`.
*   **Tier 2: Inductive Bias Mathematical Gates**: Synthetic mathematical tests on toy ODEs / controlled manifolds verifying structural invariants before clinical exposure.
*   **Tier 3: Converged PTB-XL Benchmark Grid**: Standard 100-epoch hyperparameter grid on PTB-XL Folds 1–7 (train), Folds 9–10 (val), evaluated on held-out Fold 8 test set.
*   **Tier 4: Dedicated Acquisition-Configuration Shift Tier (First-Class Empirical Core)**:
    - **Tier 4A: Structured Clinical Subsets**: $S_{12} \to S_6 \to S_3 \to S_2 \to S_1 \to S_{\rm ICM}$ ($V_3 - V_2$).
    - **Tier 4B: Exhaustive $Q_8$ Combinatorial Battery**: All $\sum_{k=1}^8 \binom{8}{k} = 255$ non-empty subsets of independent leads $Q_8 = \{I, II, V_1, \dots, V_6\}$.
    - **Tier 4C: Continuous Unseen Lead-Span Operators**: Evaluating zero-shot generalization to 100 held-out dual functionals $q_{\rm new} = \sum_{j=1}^8 \alpha_j e_j$ ($\|q_{\rm new}\|_2 = 1$).

---

## 4. Methodological Separation: P0 vs. P1

To ensure complete fairness against GraphECG and tensor baselines:
*   **Protocol P0 (Strict Zero-Shot Generalization)**:
    - Models are trained **strictly on standard full-lead configurations** ($S_{12}$ or $Q_8$).
    - Reduced subsets, single leads, and novel operators are evaluated zero-shot without prior exposure to missingness.
    - Directly isolates **representation inductive bias**.
*   **Protocol P1 (Missing-Lead Robustness Training)**:
    - Models are trained with dynamic random lead omission / masking during optimization.
    - Directly isolates **augmentation-induced robustness**.

---

## 5. Explicitly Rejected Complexity & Scope Boundary

1. **Rejected Complex 3D Torso Meshes**: We do not require patient-specific CT-derived torso meshes or finite-element boundary element solvers; linear functionals $q \in \mathbb{RP}^7$ capture the effective dipole volume conductor span directly.
2. **Rejected GNN Node Heuristics for Continuous Duals**: GraphECG snaps arbitrary measurements to nearest discrete electrode pairs; continuous set cross-attention natively accommodates any $q \in S^7$.
3. **Scope Clarification**: We claim **ECG acquisition-configuration robustness** and **measurement-operator generalization**, *not* universal hardware-agnosticism. Hardware transfer functions (ADC gain, analog bandpass filters, electrode impedance) represent a separate domain-shift axis.
