# Refinement Report: Pivot to Operator-Invariant ECG Representation

**Date**: September 21, 2026  
**Refinement Focus**: Elevating Acquisition-Configuration Robustness to the Primary Empirical Core

---

## 1. The Core Pivot: From Topological Speculation to Physical Operator Invariance

| Dimension | Initial Conception | Refined Paradigm |
| :--- | :--- | :--- |
| **Central Question** | "Does topological recurrence or braid theory improve ECG classification?" | **"Can an ECG representation preserve diagnostic information under a change in measurement operator?"** |
| **Role of Topology** | Starting premise and central claim | **Ablation test** on reconstructed canonical electrical fields |
| **Primary Prior Art** | Standard 1D ResNets on fixed 12-lead tensors | **GraphECG (Ansari et al., 2026)** and flexible-lead message passing |
| **Lead Evaluation** | Ad-hoc zero-imputation ($Z_0$) on a few subsets | **Dedicated Tier 4 Suite**: Structured clinical subsets + Exhaustive 255-subset combinatorial suite + Continuous held-out dual functionals |
| **Falsification Safety** | If topology fails, the entire paper collapses | If topology fails, **operator invariance and canonical field reconstruction stand independently** |

---

## 2. Key Refinement Decisions

### Decision 1: Establishing Tier 4 as a First-Class Empirical Result
Rather than burying reduced-lead tests in a supplementary appendix, Tier 4 becomes the central experimental narrative. Every eligible architecture is evaluated from its **frozen Fold 8 checkpoint** across systematically degraded measurement operators.

### Decision 2: Direct PTB-XL Implementation of GraphECG
To avoid the methodological flaw of comparing PTB-XL diagnostic AUROCs to EchoNext structural heart disease figures, we integrated the author's exact GraphECG PyTorch Geometric architecture into our codebase (`src/repecg/graphecg/`) and created a standardized training pipeline on PTB-XL.

### Decision 3: Decoupling Zero-Shot (P0) from Robustness-Trained (P1)
We recognized that Paper 07 was trained with dynamic context sampling ($m \in [1, 6]$). To ensure scientific integrity, we explicitly demarcated:
- **P0**: Strict zero-shot evaluation of models trained solely on full leads.
- **P1**: Evaluation of models explicitly trained with random subset augmentation.

### Decision 4: Exhaustive Combinatorial Battery ($Q_8$)
Rather than sampling 5 or 6 hand-picked configurations, evaluating all $255$ subsets of the 8 linearly independent leads provides a complete, unbiased degradation curve $k \mapsto R_k$ and configuration variance profile $\operatorname{Var}_{|S|=k}[\operatorname{score}(S)]$.

### Decision 5: Precise Scope Demarcation
We restrict our claims to **acquisition-configuration robustness** and **measurement-operator generalization**. We explicitly reject claiming "universal hardware agnosticism", which would require controlling for ADC bit-depth, analog filter phase distortion, and skin-electrode impedance.
