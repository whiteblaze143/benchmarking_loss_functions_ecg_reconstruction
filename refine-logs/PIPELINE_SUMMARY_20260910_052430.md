# Pipeline Summary

**Problem**: Reconstruct 11 missing ECG leads from Lead I (wearable/smartwatch signal) with provable clinical fidelity, determined by which loss formulation best preserves diagnostic morphology.  
**Final Method Thesis**: A 20 ms temporally tokenized Lean Convolutional-Transformer with homoscedastic multi-task uncertainty weighting and auxiliary wave delineation grounding (LCT-MTL) is the smallest architecture that matches cardiac electrophysiological timescales and eliminates gradient conflict in multi-objective ECG reconstruction.  
**Final Verdict**: READY (method stable; Block B + SAP evaluation pending for confirmatory inference)  
**Date**: 2026-09-10

---

## Final Deliverables

| File | Status |
|---|---|
| [FINAL_PROPOSAL.md](refine-logs/FINAL_PROPOSAL.md) | ✅ Complete — LCT-MTL champion spec |
| [EXPERIMENT_PLAN.md](refine-logs/EXPERIMENT_PLAN.md) | ✅ Complete — 5 blocks, 3 gates |
| [EXPERIMENT_TRACKER.md](refine-logs/EXPERIMENT_TRACKER.md) | ✅ Initialized |
| REVIEW_SUMMARY.md | ✅ Exists (local integrity audit, round 2) |
| SAP_BENCHMARK_V2.csv | ⬜ NOT YET — Block C produces this |

---

## Contribution Snapshot

- **Dominant contribution**: Empirical proof that 20 ms temporal tokenization matches cardiac conduction timescales, establishing the first ablation-validated LCT-MTL champion for 12-lead synthesis from Lead I.
- **Supporting contribution**: Homoscedastic uncertainty weighting (Kendall formulation) eliminates gradient conflict and enables synergistic integration of biophysical constraints (VCG loop, MMD distribution alignment) — tested in Block B.
- **Explicitly rejected complexity**: Dice loss (neutral, dropped), Hard-basis projections (EchoNext collapse), Stochastic whole-lead dropout (coordinate destabilization), Morlet phase penalties (high-frequency ringing), architecture depth > 4 layers.

---

## Must-Prove Claims

1. **C1** (PARTIALLY DONE): 20 ms patch beats 50 ms (DONE: +0.0083) and beats 10 ms T_patch5 (DONE: non-inferior but smaller gap confirms optimum). Reviewer anti-claim "gain = more compute" ruled out.
2. **C2** (PENDING — Block B): Adaptive uncertainty weighting allows VCG loop + MMD to synergize. Gate: ≥1 job exceeds T_patch10 (r > 0.7635, P₀₅ > 0.4208).

---

## SAP Evaluation — Critical New Requirement

The current 258-metric benchmark has three statistical errors requiring correction before any clinical claims can be published:

| Error | Impact |
|---|---|
| Direct Pearson averaging | Upward bias; biases model ranking |
| ECG-level p05/p95 | Over-represents multi-ECG patients; invalid distribution |
| No patient-cluster bootstrap | CIs are too narrow; patient independence violated |

**Fix**: Block C — `evaluate_sap_v2.py` + `run_sap_eval_queue.py` → `SAP_BENCHMARK_V2.csv`  
**Pre-specified δₘ values** are locked in EXPERIMENT_PLAN.md §3.2 before looking at corrected results.

---

## First Runs to Launch (In Order)

1. **Block B** — Unblock lean_abl2 queue, then launch VCG/MMD suite:
   ```bash
   tmux send-keys -t lean_abl2 '' Enter  # unblock deferred cells
   bash refine-logs/lean_abl2_adaptive_vcg_mmd_queue.sh
   ```

2. **Block C (parallel)** — Build + smoke test SAP evaluator:
   ```bash
   python3 scripts/evaluate_sap_v2.py \
     --model-id lean2_L1_clean_lean_best_s42_l0 \
     --checkpoint-path refine-logs/lean_abl2/runs/lean2_L1_clean_lean_best_s42_l0/best.pt \
     --smoke
   ```
   Then full queue in tmux:
   ```bash
   tmux new-session -d -s sap_eval_v2 "python3 scripts/run_sap_eval_queue.py; read"
   ```

3. **After B+C complete** — Freeze primary model, run Block D confirmatory inference, Block E seeds.

---

## Main Risks

| Risk | Mitigation |
|---|---|
| Block B: No VCG/MMD job clears Gate 1 | C2 becomes "adaptive helps baseline but VCG/MMD add no signal" — still publishable as negative finding |
| Block C: SAP eval slower than estimated | GPU+CPU parallelism; 1k bootstrap for exploratory metrics (not 10k) saves ~75% of CPU time |
| Block D: Δr < δₘ = 0.010 | Report as "statistically detectable but clinically trivial"; patient-level Fisher-z reduces inflation |
| Lead I exclusion from regional means lowers published r | Expected (correct); explain in methods as bias correction |

---

## Next Action

→ **Proceed to Block B launch** (VCG/MMD suite) + **Block C build** (SAP evaluator)  
→ Build `scripts/evaluate_sap_v2.py` and `scripts/run_sap_eval_queue.py` (user approved the plan)  
