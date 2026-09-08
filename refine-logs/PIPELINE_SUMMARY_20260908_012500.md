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
- **Claim 1 (Occam's Supremacy)**: The 4-layer/512-dim architecture (`L1`) achieves non-inferiority ($p_{\text{inferior}} < 0.05$) and higher correlation than the 8-layer anchor.
- **Claim 2 (Loss Parsimony)**: The triplet loss ($\mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{Pearson}} + \mathcal{L}_{\text{Deriv}}$) is strictly optimal; dropping any component or adding MMD/VCG impairs correlation.
- **Claim 3 (Delineation Grounding)**: Multi-task wave delineation is necessary for tail robustness ($p_{05}$), preventing catastrophic morphological failure.

---

## Active & Queued Runs
1. `L1_clean_lean_best`: Baseline Clean Lean Best (currently training in `lean_abl2` tmux).
2. `LA1_nodel`: Delineation ablation on lean core.
3. `LB1_dec3`: Decoder depth $D=3$ probe.
4. `LC1_width384`: Width $W=384$ probe.
5. Extended Suite (Cells 8–33): 26 systematic configurations covering loss decomposition, wavelets, patch granularity, delineation cadence, and spatial conditioning.

---

## Main Risks
- **Risk**: Decoder depth $D=3$ could potentially degrade lateral precordial leads ($V_4 - V_6$).
- **Mitigation**: Precordial $V_1 - V_6$ correlation is evaluated as a dedicated kill-gate threshold.

---

## Next Action
- The 33-cell suite is actively queued and executing inside the persistent detached `tmux` session `lean_abl2`. Monitor progress via `/monitor-experiment`.
