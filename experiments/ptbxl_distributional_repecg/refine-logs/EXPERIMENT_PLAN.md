# Experiment Plan: Acquisition-Configuration Robustness & Measurement-Operator Generalization

**Document Version**: 2.0 (Reframed around Tier 4 Configuration Shift)  
**Date**: September 21, 2026  

---

## 1. Experimental Objectives

1. **Quantify Configuration Degradation**: Measure the diagnostic retention $R_k = \bar{S}_k / S_8$ and degradation $\Delta_k = S_8 - \bar{S}_k$ across lead loss cardinalities $k \in \{1, \dots, 8\}$.
2. **Benchmark Against GraphECG**: Conduct an exact head-to-head comparison against the original GraphECG architecture (Ansari et al., Stanford 2026) trained and evaluated on PTB-XL Fold 8.
3. **Validate Continuous Dual Functionals**: Test whether parameterizing measurements as continuous linear functionals $q \in \mathbb{RP}^7$ enables zero-shot generalization to novel, non-standard lead combinations where discrete graph nodes fail.
4. **Isolate Topological Value**: Determine whether dynamic persistent homology / braid invariants on reconstructed canonical fields yield measurable performance gain over the field alone.

---

## 2. Evaluation Tiers Specification

### Tier 1: Engineering Plumbing Smokes
*   **Purpose**: Rapid wiring verification (`max_epochs=1, patience=1`).
*   **Output**: Stamped `engineering_only_not_a_result`. Verifies tensor shapes, forward/backward loops, and checkpoint saving.

### Tier 2: Inductive Bias Pre-Clinical Gates
*   **Purpose**: Synthetic mathematical tests verifying theoretical invariants before clinical exposure.
*   **Protocol**:
    *   *Linearity / Rank Tests*: Hankel DMD rank retention (falsified P04).
    *   *Interventional Invariance*: Pearlian counterfactual surgery consistency (falsified P09).
    *   *Gauge Invariance*: Antipodal symmetry $q \sim -q$ validation (passed P07).

### Tier 3: Converged PTB-XL Clinical Trials
*   **Purpose**: Full clinical benchmark training on PTB-XL (`max_epochs=100, patience=10`, 9 hyperparameter cells per variant).
*   **Splits**: Folds 1–7 (Train, 15,258 records), Folds 9–10 (Validation, 4,368 records), Fold 8 (Held-out Test, 2,173 records).
*   **Task**: 5-class superclass multilabel diagnosis (NORM, MI, STTC, CD, HYP).

### Tier 4: Dedicated Acquisition-Configuration Shift Suite (The Empirical Core)
Evaluates frozen Fold 8 checkpoints without retraining:

#### Tier 4A: Structured Clinical Subsets
*   $S_{12}$: Standard 12-lead ($I, II, III, aVR, aVL, aVF, V_1, \dots, V_6$)
*   $S_6$: 6 limb leads ($I, II, III, aVR, aVL, aVF$)
*   $S_3$: Einthoven triangle ($I, II, III$)
*   $S_2$: Minimal limb basis ($I, II$)
*   $S_{1,\text{I}}$ & $S_{1,\text{II}}$: Single wearable leads
*   $S_{\rm ICM}$: Implantable cardiac monitor bipolar proxy ($V_3 - V_2$)

#### Tier 4B: Exhaustive $Q_8$ Combinatorial Battery
*   All $\sum_{k=1}^8 \binom{8}{k} = 255$ non-empty subsets of independent leads $Q_8 = \{I, II, V_1, V_2, V_3, V_4, V_5, V_6\}$.
*   For each cardinality $k \in \{1, \dots, 8\}$, compute:
    $$\bar{S}_k = \mathbb{E}_{|S|=k}[\operatorname{score}(S)], \quad S_k^{\min} = \min_{|S|=k}\operatorname{score}(S), \quad \operatorname{Var}_{|S|=k}[\operatorname{score}(S)], \quad R_k = \frac{\bar{S}_k}{S_8}$$

#### Tier 4C: Continuous Held-Out Lead-Span Operators
*   Sample 100 random unit vectors $q_{\rm new} \in S^7$ ($q_{\rm new} = \sum_{j=1}^8 \alpha_j e_j, \|q_{\rm new}\|_2 = 1$).
*   Generate synthetic response $x_{q_{\rm new}}(t) = q_{\rm new}^\top X_{Q_8}(t)$.
*   Evaluate zero-shot diagnostic inference for `SetOperator` vs nearest-known lead snapping for `GraphECG` and `FixedTensor`.

#### Tier 4D: Protocol Bifurcation (P0 vs P1)
*   **P0**: Strict zero-shot from 12-lead trained models.
*   **P1**: Robustness-trained models with random subset augmentation.

---

## 3. Model Architecture Battery

| Architecture | Paradigm | Mechanism Under Shift |
| :--- | :--- | :--- |
| **FixedTensor** | Standard 1D ResNet/ConvNet | Zero-imputation ($Z_0$) of unobserved channels |
| **FixedTensor+Mask** | Channel-augmented ConvNet | Observation mask concatenated along channels ($Z_M$) |
| **GraphECG** | Electrode GNN (Ansari et al., 2026) | Dynamic induced subgraph with message passing |
| **SetOperator (P07)** | Continuous Dual $\mathbb{RP}^7$ Set Encoder | Variable set cross-attention over observed $(q_j, x_j(t))$ |
| **Moments (P02)** | Spatial Dipole Statistics | Closed-form extraction of marginal covariance $\Sigma_{S \times S}$ |
| **CanonicalField (P01)** | Field Reconstruction + Recurrence | Reconstructs canonical 3D dipole before downstream head |

---

## 4. Execution Sequence on GPU

1. **Current Running Tasks**:
   - Stage 2: P08 final explicit controls concluding.
   - Stage 3: P10, P11, P13, P15 claims papers.
   - Stage 4: Post-queue Fold 8 standardized evaluation.
2. **Appended Queue Tasks**:
   - **Stage 5: GraphECG PTB-XL Full Training**:
     - `train_graphecg_ptbxl.py` on Folds 1–7 (train) / Folds 9–10 (val), evaluated on Fold 8.
     - Saves `outputs/graphecg/graphecg_ptbxl_best.pt` and `outputs/cross_paper_evaluation/graphecg_fold8.json`.
   - **Stage 6: Tier 4 Acquisition-Configuration Shift Battery**:
     - Executes `evaluate_tier4_shift.py` across all frozen models.
     - Generates degradation curves, combinatorial retention profiles, and Table I/II outputs in `outputs/tier4_configuration_shift/`.
