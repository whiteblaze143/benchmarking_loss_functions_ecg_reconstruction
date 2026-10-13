# Pipeline Summary: Calibration-Efficient Patient-Specific Physiological Foundation Models

**Problem**: AI models diagnose disease across populations, but clinical medicine requires detecting deviations relative to an individual patient's baseline. Full personalization fails due to calibration data bottlenecks.  
**Final Method Thesis**: Universal ECG representations contain persistent subject-specific structure; the key question is whether that structure can be converted into clinically useful state-change information with very little calibration ($K \in \{0, 1, 2, 3, 5\}$).  
**First-Class Comparator**: Cardiac Sensing Foundation Model (CSFM Base, $d=768$, Gu et al., *Nature Machine Intelligence* 2026) and ECG-LFM.  
**Final Verdict**: READY  
**Date**: 2026-09-30  

---

## Final Deliverables
- Proposal: [`FINAL_PROPOSAL.md`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/refine-logs/FINAL_PROPOSAL.md)
- Review summary: [`REVIEW_SUMMARY.md`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/refine-logs/REVIEW_SUMMARY.md)
- Refinement report: [`REFINEMENT_REPORT.md`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/refine-logs/REFINEMENT_REPORT.md)
- Experiment plan: [`EXPERIMENT_PLAN.md`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/refine-logs/EXPERIMENT_PLAN.md)
- Experiment tracker: [`EXPERIMENT_TRACKER.md`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/refine-logs/EXPERIMENT_TRACKER.md)

---

## Contribution Snapshot
- **Dominant Contribution**: Mapping the calibration–utility frontier across universal representations, rigorously comparing SetOperator against CSFM under identical personalization machinery.
- **Dimensionality Control**: Native representations ($d=256$ vs $d=768$) and matched dimensionality (PCA $d=128, 256$ fit strictly on training splits).
- **Explicitly Rejected Overclaims**:
  - Replaced "patient coordinate manifold" with "longitudinally persistent subject-specific structure."
  - Replaced "5.35x representational priority" with descriptive variance partitioning across repeated records.
  - Replaced "zero temporal decay" with "no monotonic degradation was detected across tested time bins."
  - Replaced "mathematically exact decomposition / pure perturbation" with empirical decodability of stable substrate attributes from $s_p$ vs residual $\delta_t$.
  - Replaced "97.9% confidence" with "the same-patient reference outperformed the matched wrong-patient reference in 97.9% of donor permutations."
  - Clarified that MIMIC analysis was an event-level transition feasibility audit (752 conversions), not full waveform encoding.
  - Formally accounted for pretraining domain overlap (`CSFM_PRETRAIN_DOMAIN_OVERLAP = TRUE` for MIMIC).

---

## Must-Prove Claims & Benchmark Tracks
1. **M0-FM (Patient Geometry Benchmark)**: Does SetOperator preserve patient-specific structure more strongly than a state-of-the-art generic cardiac FM (CSFM)?
2. **M1-C (Calibration Efficiency)**: Which representation becomes useful fastest ($G(K) = \text{Perf}(K) - \text{Perf}(0)$ for $K \in \{0, 1, 2, 3, 5\}$)?
3. **M1-S (Set Aggregation Factorial)**: $2 \times 2$ decomposition separating representation quality from permutation-invariant history memory.
4. **Sparse-Lead Resilience**: Information retention under $12L, II, I+II, I+II+V1, I+II+V5$.
5. **Disease-Preservation Triad**: State accessibility, patient preservation, and relative change gain across longitudinal disease transitions.

---

## Immediate Next Run to Launch
- **Milestone M0-FM**: Run the completed M0 patient-coordinate audit using frozen CSFM Base on the exact same 1,693-patient / 4,087-ECG PTB-XL cohort, side-by-side with SetOperator, reporting both native and dimension-matched ($d=128, 256$) geometry metrics.
- Persistent `tmux` session: `prism_m0_fm`.
