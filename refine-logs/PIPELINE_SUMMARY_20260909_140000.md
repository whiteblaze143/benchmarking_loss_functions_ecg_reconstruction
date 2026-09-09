# Pipeline Summary

**Problem**: Inverse 12-lead electrocardiographic reconstruction from single-lead wearable Lead I.  
**Final Method Thesis**: Parsimonious multi-task convolutional-transformer synthesis (LCT-MTL) grounded in 20 ms conduction-matched temporal tokenization, 16-head cross-lead spatial projection, homoscedastic uncertainty weighting, and boundary-transition wave delineation outperforms bloated overparameterized models while slashing parameter count by 43.7%.  
**Final Verdict**: READY  
**Date**: 2026-09-09  

---

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

---

## Contribution Snapshot
- **Dominant contribution**: Discovery that $20\text{ ms}$ temporal tokenization (`patch_size = 10`) matches intrinsic cardiac conduction physics, delivering the all-time project peak correlation ($r = \mathbf{0.7635}$, $\Delta r = \mathbf{+0.0083}$, $p < 10^{-6}$) and peak worst-tail robustness ($P_{05} = \mathbf{0.4208}$).
- **Supporting contribution 1**: Kendall homoscedastic uncertainty loss weighting (`LE2_adaptive`) eliminates manual multi-task gradient conflict, unlocking $\Delta r = \mathbf{+0.0054}$.
- **Supporting contribution 2**: 16 cross-lead attention heads refine spatial dipole projections across 12 leads with zero parameter overhead ($\Delta r = \mathbf{+0.0023}$).
- **Supporting contribution 3**: Identification of the optimal Pareto efficiency knee point at Width 384, reducing parameters by $43.7\%$ ($17.4\text{M}$ vs $30.5\text{M}$) with negligible $-0.0015$ delta.
- **Explicitly rejected complexity**:
  - Pruned whole-lead dropout (removing it gives $+0.0097$ boost).
  - Pruned soft Dice loss (neutral $\Delta r = -0.0002$, cuts GPU batch union operations).
  - Pruned 256-channel wavelet expansion (overfits by $-0.0020$).
  - Pruned continuous Morlet phase transformations (induces synthetic baseline bias).
  - Pruned hard deterministic algebraic projection matrices (restricts non-linear volume conductor dynamics).

---

## Must-Prove Claims
- **Claim 1 (Temporal Granularity Matches Conduction Velocity)**: $20\text{ ms}$ tokenization (`patch_size = 10`) matches cardiac electrophysiological activation wavefronts, yielding statistically significant gains ($\Delta r = +0.0083$, $p < 10^{-6}$) and superior $P_{05}$ tail robustness ($0.4208$) over coarser tokenization.
- **Claim 2 (Efficiency Knee Point)**: Width 384 preserves $99.8\%$ of reconstruction accuracy while slashing parameter count by $43.7\%$, establishing the optimal Pareto frontier for real-time edge synthesis.
- **Claim 3 (Loss Triad Invariance)**: Reconstruction fidelity requires the simultaneous optimization of absolute voltage (MSE), waveform morphology (Pearson), and high-frequency deflections (1st Derivative); omitting any component induces clinical degradation.
- **Claim 4 (Homoscedastic Loss Harmony)**: Dynamic task uncertainty weighting eliminates manual gradient scaling conflicts, unlocking higher performance when integrating higher-order biophysical constraints (VCG/MMD).

---

## First Runs to Launch (Chained in `lean_abl2_adaptive_vcg_mmd_queue.sh`)
1. `AV1_vcg_adaptive` (`1101000`): Adaptive MSE + Pearson + Kors 3D VCG loop alignment.
2. `AV2_triplet_vcg_adaptive` (`1111000`): Full biophysical quad (MSE + Pearson + Deriv + Kors VCG).
3. `AM1_mmd_imq_adaptive` (`1100003`): Anatomical block multiscale IMQ kernel MMD.

---

## Main Risks
- **Risk**: Adding higher-order biophysical Kors VCG loops or MMD kernels might over-constrain the latent space even under adaptive weighting.
- **Mitigation**: The 11-job queue tests isolated pairwise additions (VCG-only, MMD-only) against joint combinations (Pentads), with automatic early-stop killgates ($r < 0.7471$ at epoch 10) and paired bootstrap non-inferiority evaluation against the Clean Lean Core.

---

## Next Action
- The chained queue `lean_abl2_adaptive_vcg_mmd_queue.sh` has been updated on disk with the Champion LCT-MTL Base.
- Allow the active tmux session `lean_abl2` to autonomously complete Axis 5 and transition directly into the updated 11-job Adaptive VCG & MMD suite.
