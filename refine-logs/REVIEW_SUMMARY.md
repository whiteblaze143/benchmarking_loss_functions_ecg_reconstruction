# Review Summary: GRAIL-ECG v2 Representation Refinement

**Date**: 2026-09-11  
**Target Focus**: Representation-Theoretic Qualification vs Reconstruction & Classification  

---

## 1. Prior Weaknesses & Reviewer Critiques Addressed

| Prior Conceptual Weakness | Critical Flaw Identified | Methodological Resolution in v2 |
|---|---|---|
| **Reconstruction as Objective** | Synthesizing 60,000 voltage samples wastes model capacity on electrode skin noise and baseline drift rather than cardiac state. | Replaced with direct latent learning $E: \mathcal{X} \to \mathcal{Z} \in \mathbb{R}^{96}$; waveform reconstruction relegated to an auxiliary decoder factor ($V$) in factorial design. |
| **B0 Baseline Contamination** | Running baseline `B0 Plain SSL` contained clinical anchor BCE, contaminating the unsupervised baseline. | Fixed: B0 is now trained strictly with VICReg loss ($L_{\text{sim}} + L_{\text{var}} + L_{\text{cov}}$) and zero clinical BCE. |
| **Arbitrary Lead Sequence Order** | Brute-forcing lead sequences yields $1.3 \times 10^9$ permutations, confounding sequence order with clinical subsets. | Mathematical contract: Set cross-attention carries the trivial representation of $S_k$, proving lead-order permutation invariance and collapsing the space into exactly 4,095 subsets. |
| **Treating 12 Leads as Independent** | Frontal leads are deterministically coupled ($III = II - I$, etc.), meaning displayed count $k \neq$ information rank. | Formalized independent rank $r(S) \in \{1..8\}$ relative to $\{I, II, V1-V6\}$ and $r_{\text{limb}} \le 2$. |
| **Random 70/30 Label Splitting** | High label co-occurrence in PTB-XL creates leakage between anchor and probe tasks. | Built formal label dependency graph ($P(p\mid a), J(a, p)$) and partitioned probe targets into Tiers P1, P2, and pure ontology-distinct P3 (`LVOLT`, `NORM`, `PACE`). |
| **Lack of Standard Downstream Probing** | Using only ad-hoc linear probes hinders comparison with literature benchmarks. | Integrated LVCG's multi-benchmark probing suite (PTB-XL Superclass, Subclass, Form, Rhythm, ICBEB, Chapman) with validation threshold optimization. |

---

## 2. Intentional Rejections & Tradeoffs

1. **Rejected: Full Waveform Invertibility**  
   An invertible representation of 60,000 voltage samples is neither necessary nor desirable; clinical ECG interpretation relies on invariant physiological morphology, not sample-level noise reproduction.
2. **Rejected: Neural Mutual Information Estimators as Primary Proof**  
   Variational neural estimators of $I(Z; Y)$ suffer from high variance, sample complexity bounds, and loose approximations in high dimensions. Replaced with verifiable operational proxies: linear sufficiency gaps, Fisher separability, low-shot sample efficiency, and retrieval metrics.
3. **Rejected: Statistical Disentanglement as Independence**  
   ECG clinical factors are biologically correlated (e.g. QRS duration and bundle branch block; ventricular hypertrophy and ST depression). Forcing statistical independence is biologically false. We seek **functional factorization** verified by slot selectivity and intervention specificity.
4. **Rejected: Training 4,095 Individual Specialist Models**  
   Training 4,095 specialists is computationally wasteful. Instead, we train a generalist subset model and compare against specialists on key Pareto-frontier subsets ($k \in \{1, 2, 4, 8, 12\}$).
