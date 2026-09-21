# Reviewer Synthesis: Dissecting Configuration Invariance in ECG Representation

**Review Focus**: Addressing Critical Weaknesses, Methodological Confounders, and Reviewer Objections for Top-Tier Conference Submission (ICLR/NeurIPS).

---

## 1. Primary Reviewer Critiques & Architectural Defenses

### Critique 1: "Numerical Comparison with Published GraphECG Numbers is Invalid"
*   **The Issue**: GraphECG (Ansari et al., 2026) reports AUROC numbers on the **EchoNext structural heart disease** cohort (5,442 test ECGs, predicting left ventricular systolic dysfunction / aortic stenosis / hypertrophic cardiomyopathy with 7 clinical tabular covariates in late fusion). Directly comparing PTB-XL Fold 8 diagnostic AUROCs (e.g. 0.9230) against EchoNext AUROCs (e.g. 0.836) is mathematically and clinically fraudulent.
*   **The Resolution**: We implement and train the author's exact `GraphECG` GNN architecture directly on **PTB-XL** using our standardized patient-stratified split (Folds 1–7 train, Folds 9–10 validation, Fold 8 held-out test). This enables an apple-to-apple head-to-head comparison on the exact same records, sampling rate, and 5-class superclass multilabel task.

---

### Critique 2: "Conflating Representation Inductive Bias with Missingness Data Augmentation"
*   **The Issue**: If Paper 07 was trained using random cardinality sets $m \in [1, 6]$, claiming it possesses superior "zero-shot" generalization over models trained only on full 12 leads is a false claim. Any model exposed to missingness during training naturally learns missing-data robustness.
*   **The Resolution**: We enforce a strict protocol bifurcation:
    *   **Protocol P0 (Strict Zero-Shot)**: All models are trained *only* on full standard leads ($S_{12}$ or $Q_8$). Reduced subsets and novel operators are tested without any missingness exposure during training.
    *   **Protocol P1 (Robustness-Trained)**: When models are allowed random subset training, the fixed-tensor baseline is given identical training augmentation (random lead dropout + observation mask).
    This cleanly decouples **representation inductive bias** from **augmentation-induced robustness**.

---

### Critique 3: "Six Clinical Configurations are Vulnerable to Cherry-Picking"
*   **The Issue**: Reporting only $12 \to 6 \to 3 \to 2 \to 1 \to \text{ICM}$ risks selecting configurations where a particular model happens to excel (e.g. Lead II carries strong rhythm information, while Lead I carries frontal axis information).
*   **The Resolution**: We introduce the **Exhaustive $Q_8$ Combinatorial Battery**. Across the 8 linearly independent leads $Q_8 = \{I, II, V_1, \dots, V_6\}$, there are exactly:
    $$\sum_{k=1}^8 \binom{8}{k} = 255 \text{ non-empty subsets}.$$
    For every cardinality $k \in \{1, \dots, 8\}$, we compute:
    1. Expected score: $\bar{S}_k = \mathbb{E}_{|S|=k}[\operatorname{score}(S)]$
    2. Worst-case score: $S_k^{\min} = \min_{|S|=k}\operatorname{score}(S)$
    3. Retention ratio: $R_k = \bar{S}_k / S_8$
    4. Degradation: $\Delta_k = S_8 - \bar{S}_k$
    5. Configuration variance: $\operatorname{Var}_{|S|=k}[\operatorname{score}(S)]$
    A truly configuration-robust model must demonstrate both high retention $R_k$ and low configuration dispersion across surviving subsets.

---

### Critique 4: "Why Prefer Continuous Operators Over Discrete 3D Electrode Graphs?"
*   **The Issue**: GraphECG already models electrodes as nodes and leads as edges. Why do we need continuous functionals $q \in \mathbb{RP}^7$?
*   **The Resolution**: GraphECG is bound to a fixed graph topology with discrete anatomical vertices. If presented with a novel dual measurement functional $q \in S^7$ (e.g. an oblique wearable vector, an interpolated bipolar lead, or an arbitrary linear combination of torso potentials), GraphECG has no corresponding node pair and must snap heuristically to existing vertices. Paper 07 parameterizes the measurement dual continuously, providing exact $\mathbb{Z}_2$ projective gauge invariance ($q \sim -q$) and continuous extrapolation over the entire 8D lead span.

---

### Critique 5: "Is Topology Necessary if Field Reconstruction Solves the Problem?"
*   **The Issue**: If a canonical field representation $\hat{\Phi}(q, t)$ already eliminates configuration shift, does topological feature extraction (braid invariants, persistence diagrams) add genuine value or merely redundant complexity?
*   **The Resolution**: We structure the paper around a hierarchical hypothesis test:
    $$\boxed{\text{Operator Awareness}} \longrightarrow \boxed{\text{Canonical Field Reconstruction}} \longrightarrow \boxed{\text{Dynamic Topology on Canonical Field}}$$
    If topology fails to add statistical lift over the reconstructed canonical field, this negative result is published honestly as a falsification of topological necessity, while the operator-invariant field representation remains fully validated.
