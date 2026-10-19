# Pipeline Summary: Paper 07 — 168 GPU-Hour SetOperator Architectural Ablation Campaign

**Problem**: Determining what architectural mechanisms enable continuous RKHS Phase-KME SetOperator to generalize to unfamiliar, continuous ECG measurement operators without sacrificing canonical clinical diagnosis.
**Final Method Thesis**: Explicit second-order operator measurement geometry ($\gamma(q) \in \mathbb{R}^{44}$) combined with phase-position-aware cardiac readout and cross-view Transformer interaction provides the optimal inductive bias for continuous measurement operator generalization on EchoNext.
**Final Verdict**: **READY**
**Date**: 2026-10-18

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Implementation plan: `<Artifact Directory>/setoperator_168h_ablation_campaign_plan.md`
- Supervisor Script: `scripts/run_paper07_ablation_week.py` (in dev worktree)
- Manifest: `experiments/echonext_operator_benchmark/ablation_week_manifest.json`

## Contribution Snapshot
- **Dominant contribution**: Dissecting continuous operator representation, phase morphology, and set aggregation mechanisms on EchoNext SHD detection under authentic RKHS Phase-KME.
- **Supporting contribution**: Resolving PhaseCNN cyclic-shift invariance via learned phase-position readout ($\Delta < 6 \times 10^{-8}$ proved).
- **Explicitly rejected complexity**: Speculative multi-seed sweeps prior to single-seed screening; heavy diffusion, VAE, or flow decoders; five-seed sweeps before identifying the winning architecture.

## Must-Prove Claims
1. **Geometry vs Capacity**: G1's 2nd-order projective geometry ($\mathbb{R}^{44}$) outperforms the capacity-matched raw-$q$ MLP control (`8 -> 96 -> 64`), confirming explicit geometry provides useful inductive bias rather than mere parameter expansion.
2. **Phase Readout Utility**: Breaking PhaseCNN's exact cyclic-shift invariance via learned phase-position readout improves SHD classification and out-of-distribution operator generalization.
3. **Cross-View Interaction**: Multi-head self-attention across observed lead tokens yields superior generalization compared to independent per-view encoding and pooling.
4. **Modality Attribution**: True ECG waveform morphology provides genuine diagnostic value beyond the 7 clinical tabular covariates.

## First Runs to Launch
1. `A0_baseline`: Original SetOperator SO-Span seed 42 (active in `train_so_span_s42`, PID 3937990).
2. `A1_g1_from_scratch`: G1 SetOperator SO-Span seed 42 (queued in `g1_launch_controller`).
3. `A2_capacity_control`: Capacity-matched raw-$q$ SetOperator (`Linear(8,96)->GELU->Linear(96,64)`).
4. `D2_tabular_only`: 7-covariate clinical MLP baseline.
5. `D1_ecg_only`: Pure ECG morphology SetOperator (`patient_context_dim = 0`).
6. `B2_phase_position_aware`: SetOperator with learned softmax phase weighting.

## Main Risks & Mitigations
- **Risk**: Baseline or G1 job interrupted or fails.
  - **Mitigation**: Dedicated launch controllers check exit codes, verify file hashes, and quarantine failures with `BLOCKER.json`.
- **Risk**: Exceeding 168 GPU-hours.
  - **Mitigation**: Dynamic early-stopping (~12 epochs per job ≈ 3.0 h), per-job duration caps, and strict priority ordering.

## Next Action
- Launch week-long supervisor in detached tmux session `paper07_ablation_week` with preflight validation gate.
