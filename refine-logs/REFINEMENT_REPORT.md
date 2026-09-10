# Refinement Report: Progression to Multi-Institutional Longitudinal Clinical Grounding

**Date**: 2026-09-10  
**Refinement Phase**: V3 Method Stabilization & Longitudinal Clinical Expansion  

---

## 1. Evolution of the Method

```
[Phase 1: Exploratory Benchmark]
  • Evaluated 25-cell ablation grid (lean_abl2).
  • Discovered:
    - 20 ms tokenization (patch 10) sets all-time record r = 0.7635 (Δr = +0.0083).
    - 16 attention heads provides finer lead subspace projection at zero parameter cost.
    - Width 384 saves 43.7% parameters with only -0.0015 r degradation.
    - Homoscedastic uncertainty weighting (LE2) stabilizes multi-task training (+0.0054 r).

[Phase 2: Reviewer Vulnerability Audit]
  • High cross-sectional correlation on PTB-XL (r = 0.7635) can still hide clinical hallucination.
  • Reviewers will challenge: "Does the model preserve genuine AF in precordial leads or does it smooth it into sinus rhythm? Does it hallucinate AF after cardioversion?"
  • Inter-patient anatomical variability creates severe confounding in cross-sectional sets.

[Phase 3: Multi-Institutional Paired Grounding (Current)]
  • Solution: Integrate the Harvard-Emory ECG Database (HEEDB) across MGH and EUH.
  • Executed 10-point census extracting:
    - MGH: 412,500 active-AF ECGs; 42,650 AF->AF pairs; 28,250 AF->SR pairs.
    - EUH: 286,000 active-AF ECGs; 29,450 AF->AF pairs; 19,800 AF->SR pairs.
    - Total: 120,150 paired longitudinal transitions across 7–45d, 60–120d, 150–210d windows.
  • Storage budgeting:
    - Target paired waveforms require only 20.59 GB (4.1% of the 500 GB NFS quota).
    - Safely bounded, zero disk bloat.
```

---

## 2. Quantitative Verification of the Refined Program

1. **Thesis Integrity**: The method remains strictly lean (Occam's razor). No bloated architectures or hard projection matrices added.
2. **Clinical Endpoint Decisiveness**: The longitudinal pairs directly test biological preservation vs. hallucination on real human patients.
3. **Infrastructure Feasibility**: The 500GB NFS mount at `/data/mithunmanivannan/` is partitioned with clean directory boundaries (`papers/`, `heedb_metadata/`, `manifests/`).
