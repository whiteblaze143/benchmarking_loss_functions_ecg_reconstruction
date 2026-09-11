# Experiment Plan: GRAIL-ECG v2 — Full Representation Qualification & Lead Information Theory

**Date**: 2026-09-11  
**Status**: ACTIVE EXECUTION ROADMAP  

---

## 1. Experimental Hierarchy & Run Order

```
[Phase R0: Contracts & Gates] -> COMPLETED (18/18 Unit Tests PASS)
         |
         v
[Phase R1 & R2: Full Factorial Screen (Fold 8)] -> IN PROGRESS (Session: grail_v2_pipeline)
  - Train & qualify 10 models (UB, B0, B1, B3_geom, B2_slots, 001, 110, 101, 011, M)
  - Extract embeddings & compute full qualification battery (compactness, probes, separability, retrieval, slots)
         |
         v
[Phase R3: Factorial ANOVA & Representation Selection]
  - Estimate Delta_G, Delta_S, Delta_V and all interaction terms
  - Evaluate LVCG downstream probing benchmarks on candidate representations
         |
         v
[Phase R4 & R5: Confirmation & Stability (Fold 9 & Seeds)]
  - Confirm surviving architecture on Fold 9
  - Seed stability check across seeds {42, 43, 44} via CKA and Procrustes alignment
         |
         v
[Phase R7: Freeze Full-ECG Teacher Z*]
  - Select champion model and freeze as canonical reference Z* = T(X_full)
         |
         v
[Phase R8 & R9: Exhaustive 4,095 Subset Lattice Evaluation]
  - Evaluate all 4,095 unique lead subsets using fast token caching
  - Measure coordinate-preserving recovery vs reprobed recovery
         |
         v
[Phase R10: Lead Information Theory & Figures]
  - Exact Lead Shapley values (Representation & Disease)
  - Pairwise Harsanyi synergy/redundancy interaction graphs
  - Minimal sufficient lead sets by pathology
  - Best/median/worst Pareto information frontiers
         |
         v
[Phase R11: Sealed Final Evaluation (Fold 10)]
```

---

## 2. Decision Gates & Threshold Policies

| Milestone Gate | Success Criteria | Action if Failed |
|---|---|---|
| **P1 Memorization Gate** | Clinical loss $< 0.05$ on tiny batch; View decoder $>50\%$ L1 reduction over baseline; Non-collapse variance $>0.5$. | **PASSED**. If failed, halt and fix gradient paths. |
| **Stage A Qualification Gate** | Macro AUROC on Tier P3 probes $>0.75$; Effective rank $r_{\text{eff}} \ge 16$; Low-shot retention $>85\%$ at 10% labels; Mean retrieval $P@5 > 0.60$. | If failed, refine slot capacity or SSL loss coefficients ($\lambda_{\text{sim}}, \lambda_{\text{var}}, \lambda_{\text{cov}}$). |
| **Factorial Inductive Bias Gate** | $\Delta_G > 0$ or $\Delta_S > 0$ on Tier P3 probe AUROC and retrieval $P@5$. | If main effects are negative, revert to unstructured baseline B1. |
| **Permutation Invariance Gate** | Maximum latent perturbation under lead permutation $< 10^{-5}$. | **PASSED** ($\Delta < 10^{-6}$). |
| **Shapley Axiom Verification Gate** | Efficiency ($\sum \phi_i = v(N)$), symmetry, and dummy lead zero-contribution hold exactly. | **PASSED**. |
| **Minimal Lead Non-Inferiority Gate** | Minimal sufficient set $k_d^\star$ satisfies $P_d(S) \ge P_d(\text{full}) - 0.02$. | Classify pathology as requiring complete 12-lead acquisition ($k=12$). |

---

## 3. Detailed Phase Protocols

### Phase R1 & R2: Full Factorial Screen on Fold 8
- **Datasets**: PTB-XL Folds 1–7 Train ($N=15,245$), Fold 8 Dev Val ($N=2,173$).
- **Models**:
  1. `UB`: End-to-end Supervised Upper Bound (clinical BCE with training-fold $w_c$).
  2. `B0`: True Plain SSL (VICReg only, zero clinical loss, G=0, S=0, V=0).
  3. `B1`: Clinical SSL reference (VICReg + Clinical BCE, G=0, S=0, V=0).
  4. `B3_geom`: Geometry inductive bias alone (G=1, S=0, V=0).
  5. `B2_slots`: Structured clinical slots alone (G=0, S=1, V=0).
  6. `Model_001`: View auxiliary alone (G=0, S=0, V=1).
  7. `Model_110`: Geometry + Slots (G=1, S=1, V=0).
  8. `Model_101`: Geometry + View Aux (G=1, S=0, V=1).
  9. `Model_011`: Slots + View Aux (G=0, S=1, V=1).
  10. `Model_M`: Full Factorial Model M (G=1, S=1, V=1).
- **Training Parameters**: Batch size 64, AdamW lr=1e-3, Cosine annealing, 12 epochs per model.
- **Output**: `results/grail_v2/factorial_qualification_summary.json` and Parquet embeddings.

### Phase R8 & R9: Exhaustive 4,095 Subset Lattice
- **Execution Script**: `scripts/run_exhaustive_4095_subsets.py`
- **Method**: Token caching over Fold 8 validation records.
- **Metrics Computed for all 4,095 Subsets**:
  - `latent_cosine_to_full`, `latent_l2_to_full`
  - `fixed_head_auroc`, `reprobe_auroc`
  - `retention_relative_to_full`
  - Exact Shapley values $\phi_i^{(Z)}$ and $\phi_i^{(d)}$
  - Pairwise Harsanyi/Möbius synergy matrix $I_{ij}$
  - Minimal sufficient lead sets $k_d^\star$ ($\delta = 0.02$)
  - Pareto frontiers $P_{\min}, P_{\text{med}}, P_{\max}$ vs displayed count $k$ and independent rank $r$.
