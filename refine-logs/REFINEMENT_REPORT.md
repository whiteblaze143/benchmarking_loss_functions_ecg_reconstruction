# Refinement Report: Operationalization of GRAIL-ECG v2

**Date**: 2026-09-11  
**Refinement Phase**: V3 Implementation Transition  

---

## 1. Concrete Engineering Realizations

1. **Unit-Verified Contracts**:
   Implemented and passed 18/18 mandatory unit tests in `tests/test_grail_v2_contracts.py` covering:
   - P1 memorization (supervised pathway & view decoder)
   - VICReg non-collapse (variance and covariance bounds)
   - Lead-order permutation invariance ($\Delta < 10^{-6}$)
   - Subset uniqueness (4,095 unique non-empty bitmasks)
   - Exact rank calculations and limb algebra constraints
   - Cooperative game Shapley axioms (efficiency, symmetry, dummy lead, synergy).

2. **Concept Hierarchy and Imbalance Registry**:
   Generated `configs/ptbxl_concept_tiers.yaml` with positive class weights computed strictly from training folds 1–7:
   - 25 Anchor concepts partitioned across Rhythm (2), Conduction (6), Morphology (9), and ST-T (8).
   - 11 Tier P1 probes (directly coupled).
   - 15 Tier P2 probes (within-domain transfer).
   - 3 Tier P3 probes (pure ontology-distinct: `LVOLT`, `NORM`, `PACE`).

3. **Multi-Dimensional Representation Qualification Engine**:
   Implemented `grail_ecg/src/evaluation/representation_qualification.py` providing automated computation of:
   - Singular values spectrum, effective rank, participation ratio, TwoNN intrinsic dimension.
   - Linear probes on frozen $Z$ for Anchor, P1, P2, P3 tiers.
   - Low-shot sample efficiency sweeps (1% to 100%).
   - Fisher separation, centroid distance, within/between ratio.
   - Latent retrieval $P@k$, $Recall@k$, $nDCG@k$.
   - Slot $\times$ concept probe matrix and intervention specificity.
   - Residual discovery slot retention challenge.
   - Nuisance metadata accessibility (Age regression $R^2$, Sex classification AUROC).

4. **LVCG Multi-Benchmark Probing Integration**:
   - Implemented `external/LVCG/probing/encoders/grail_encoder.py` adapting GRAIL-ECG to LVCG's standard `BaseEncoder` interface.
   - Created `configs/eval_grail_lvcg_probing.yaml` enabling standardized evaluations on `ptbxl_super_class`, `ptbxl_sub_class`, `ptbxl_form`, `ptbxl_rhythm`, `icbeb`, and `chapman`.

5. **Exhaustive 4,095 Subset Lattice Engine**:
   Implemented `grail_ecg/src/evaluation/exhaustive_subset_eval.py` and `scripts/run_exhaustive_4095_subsets.py`:
   - Token caching precomputes all 12 lead representations in $O(N \times 12)$ steps.
   - Computes coordinate-preserving fixed-head AUROC vs reprobed AUROC.
   - Computes exact Lead Shapley values and pairwise Harsanyi/Möbius synergy/redundancy graphs.
   - Extracts disease-specific minimal sufficient lead sets and Pareto information frontiers.
