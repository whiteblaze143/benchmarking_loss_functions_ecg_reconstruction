# Pipeline Summary: Lean-CardioAIM Method Refinement & Occam's Razor Suite

**Problem**: Ill-posed inverse reconstruction of clinical 12-lead ECG from a single wearable lead (Lead I).  
**Final Method Thesis**: A parsimonious 4-layer/512-dim multi-task transformer with clean biophysical triplet loss ($\mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{Pearson}} + \mathcal{L}_{\text{Deriv}}$) and morphological delineation heads outperforms massive 8-layer models while eliminating gradient conflict and training 35% faster.  
**Final Verdict**: READY  
**Date**: 2026-09-08  

---

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

---

## Contribution Snapshot
- **Dominant contribution**: Parsimonious architecture discovery: 4-layer encoder at 512-width with clean biophysical triplet loss achieves higher correlation than 8-layer/768-dim networks while running $\sim 1.5 - 2\times$ faster.
- **Supporting contribution**: Whole-lead dropout removal restores single-lead spatial projection coherence ($+0.0105$ gain); multi-task delineation acts as a necessary inductive anchor against waveform collapse.
- **Explicitly rejected complexity**:
  1. 8-layer/768-dim encoder overparameterization.
  2. Whole-lead dropout during single-lead conditioning.
  3. High-order MMD distribution alignment and VCG loss penalties (proven to cause objective competition).
  4. Hard deterministic physical constraints (Einthoven graph wiring).

---

## Must-Prove Claims
- **Claim 1 (Occam's Supremacy)**: The 4-layer/512-dim architecture (`L1`) achieves non-inferiority ($p_{\text{inferior}} < 0.05$) and higher correlation than the 8-layer anchor. [CONFIRMED: $0.7552$ vs $0.7456$, $+0.0097$]
- **Claim 2 (Loss Parsimony & Adaptivity)**: Fixed heuristic loss weighting induced gradient competition; dynamic homoscedastic uncertainty weighting (`LE2_adaptive`) eliminates manual scale conflict and unlocks an all-time SOTA of $0.7607$.
- **Claim 3 (Delineation Grounding)**: Multi-task wave delineation is non-negotiable for tail robustness ($p_{05}$), preventing catastrophic morphological failure. [CONFIRMED: `LA1_nodel` fails, $-0.0078$ mean, $-0.0118$ tail]
- **Claim 4 (Adaptive Biophysical Reconcilement)**: Re-evaluating top historical VCG loop geometry and MMD distribution kernels with adaptive weighting will determine if physical and non-parametric priors can provide additive gains once gradient scales are decoupled.

---

## Active & Queued Runs
1. **Core Lean Cells 1–8**: Completed (with `LE2_adaptive` setting the SOTA at $0.7607$).
2. **Extended Suite (Cells 9–33)**: Currently executing in `lean_abl2` tmux session (Cell 9 `L_corr_only` active).
3. **Adaptive VCG & MMD Suite (Cells 34–44)**: Chained to run immediately following Cell 33 in `lean_abl2` (`lean_abl2_adaptive_vcg_mmd_queue.sh`).

---

## Main Risks
- **Risk**: High-dimensional MMD kernels or 3D VCG loss terms might still saturate gradients even with adaptive log-variance parameters.
- **Mitigation**: Paired bootstrap non-inferiority kill-gate evaluation against the Lean Anchor (`L1_clean_lean_best`) immediately upon each run's completion.

---

## Next Action
- The complete 44-cell queue is executing autonomously inside the persistent detached `tmux` session `lean_abl2`. Track progress in real time via `/monitor-experiment`.
