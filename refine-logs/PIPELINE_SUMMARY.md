# Pipeline Summary: GRAIL-ECG v2 Representation-Theoretic Qualification & LVCG Probing Suite

**Problem**: Determining whether a learned latent space $E: \mathcal{X} \longrightarrow \mathcal{Z} \in \mathbb{R}^{96}$ is a valid, compact, and sufficient representation of clinical electrocardiography across the 4,095 subset lattice of 12-lead ECGs, rather than evaluating superficial waveform reconstruction or single-task classification accuracy.  
**Final Method Thesis**: Decoupling full-ECG representation qualification from subset mapping via a 10-model factorial design $(G, S, V)$ on $(96\text{D})$ latent space, evaluated across a 7-domain representation qualification suite, LVCG multi-benchmark linear probing, and an exact token-cached 4,095-subset Shapley and interaction lattice.  
**Final Verdict**: **READY**  
**Date**: 2026-09-11  

---

## Final Deliverables

- **Proposal**: `refine-logs/FINAL_PROPOSAL.md`
- **Review Summary**: `refine-logs/REVIEW_SUMMARY.md`
- **Refinement Report**: `refine-logs/REFINEMENT_REPORT.md`
- **Experiment Plan**: `refine-logs/EXPERIMENT_PLAN.md`
- **Experiment Tracker**: `refine-logs/EXPERIMENT_TRACKER.md`
- **LVCG Probing Integration**:
  - Encoder Adapter: `external/LVCG/probing/encoders/grail_encoder.py`
  - Encoder Registry: `external/LVCG/probing/encoders/__init__.py`
  - Evaluation Config: `configs/eval_grail_lvcg_probing.yaml`
- **Representation Qualification Suite**:
  - Module: `grail_ecg/src/evaluation/representation_qualification.py`
  - Contracts & Concept Tiers: `configs/ptbxl_concept_tiers.yaml`
  - Unit Tests: `tests/test_grail_v2_contracts.py` (18/18 PASS)
- **Exhaustive Subset Engine**:
  - Module: `grail_ecg/src/evaluation/exhaustive_subset_eval.py`
  - Runner: `scripts/run_exhaustive_4095_subsets.py`
- **Active Execution Pipeline**:
  - Orchestrator: `scripts/run_grail_v2_pipeline.py` (running in tmux session `grail_v2_pipeline`)
  - Log: `refine-logs/grail_ecg/pipeline_v2.log`

---

## Contribution Snapshot

- **Dominant Contribution**:
  - First mathematically principled representation qualification framework for ECG state spaces, replacing 60,000-point waveform reconstruction metrics with coordinate stability, effective rank, Fisher separability, latent retrieval, and zero-shot out-of-ontology clinical transfer (Tier P3).
  - Exact token-cached computation of all 4,095 non-empty lead subsets, unlocking exact Lead Shapley values and pairwise Harsanyi synergy/redundancy interaction graphs without $8.9 \times 10^6$ forward-pass intractability.
- **Optional Supporting Contribution**:
  - Spherical geometric inductive bias $G$ based on physical 3D cardiac dipole projection.
  - Clinical domain factorization $S$ into 6 semantically disentangled 16D slots (Rhythm, Conduction, Morphology, ST-T, Residual 1, Residual 2).
  - Standardized integration into the multi-benchmark LVCG probing framework (PTB-XL Superclass/Subclass/Form/Rhythm, ICBEB, Chapman).
- **Explicitly Rejected Complexity**:
  - End-to-end waveform reconstruction loss (MSE / STFT over 60,000 voltage samples): Pruned as the primary optimization target due to capacity waste on baseline wander and acquisition noise.
  - Single-metric classification AUROC as model selector: Pruned in favor of multi-dimensional radar profiles.
  - Arbitrary permutation penalties: Replaced by analytical set cross-attention permutation invariance ($S_k$ trivial representation).

---

## Must-Prove Claims

1. **Claim 1 (Compactness & Intrinsic Dimension)**: Structured inductive bias ($G=1, S=1$) constrains effective rank $r_{\text{eff}}$ closer to the true clinical manifold ($d_{\text{TwoNN}} \approx 10\text{--}18$) than unconstrained SSL baselines without collapsing capacity ($PR > 0.15$).
2. **Claim 2 (Linear Sufficiency & Out-of-Ontology Transfer)**: $Z^\star$ achieves sufficiency gap $\Delta_{\text{suff}} = A_{\text{UB}} - A_Z \le 0.03$ on anchor concepts and transfers to unobserved Tier P3 concepts (`LVOLT`, `NORM`, `PACE`) with zero retraining.
3. **Claim 3 (Factorial Effect Isolation)**: Factorial ANOVA on factors $(G, S, V)$ proves that geometric bias $\Delta_G$ and slot factorization $\Delta_S$ account for significant positive variance on disentanglement selectivity and low-shot retrieval ($p < 0.01$).
4. **Claim 4 (Subspace Disentanglement)**: Zeroing domain slot $z_j$ causes a statistically significant degradation ($\Delta \text{AUC} > 0.10$) strictly on concepts mapped to slot $j$ with minimal interference ($\le 0.02$) on non-target slots.
5. **Claim 5 (LVCG Benchmark Parity & Superiority)**: Frozen linear probing on LVCG downstream tasks matches or exceeds baseline foundation models while using a lean 96D representation.
6. **Claim 6 (Lead Information Geometry across 4,095 Subsets)**: Lead Shapley values conform to electrical vectorcardiography (leads II and V1-V2 dominating rhythm and septal information), and minimal sufficient lead subsets recover $\ge 98\%$ of full 12-lead diagnostic performance with $\le 4$ displayed leads.

---

## First Runs to Launch

1. **Run 1 (`grail_v2_pipeline` active)**: Train the 10 factorial models (`UB`, `B0`, `B1`, `B3_geom`, `B2_slots`, `Model_001`, `Model_110`, `Model_101`, `Model_011`, `Model_M`) on PTB-XL Folds 1–7 with validation on Fold 8.
2. **Run 2 (`eval_representation_qualification`)**: Execute Phase R2 multi-dimensional representation qualification across all 10 trained checkpoints and compute ANOVA factorial main effects.
3. **Run 3 (`eval_grail_lvcg_probing`)**: Run the LVCG linear probing battery across PTB-XL, ICBEB, and Chapman benchmarks.
4. **Run 4 (`run_exhaustive_4095_subsets`)**: Execute token-cached Stage B exhaustive subset evaluation on Fold 8, computing exact Shapley vectors and interaction matrices.

---

## Main Risks & Mitigations

- **Risk 1: A100 PyTorch 2.6 cuDNN 1D Convolution Crash**:
  - *Mitigation*: Disabled cuDNN 1D convolution globally via `torch.backends.cudnn.enabled = False` in all pipeline scripts, using native CUDA 1D convolution kernel.
- **Risk 2: GPU 0 Concurrency & Memory Pressure**:
  - *Mitigation*: Background job occupies 30.8 GB; pipeline uses ~6.9 GB. Sequential execution enforced; CPU fallback enabled for concurrent probing verification.
- **Risk 3: Latent Collapse in Pure SSL Baseline B0**:
  - *Mitigation*: Strictly monitored VICReg variance ($S(Z) \ge 1.0$) and covariance decorrelation penalties to ensure full 96D non-degeneracy.

---

## Next Action

- Monitor `grail_v2_pipeline` completion in tmux session `grail_v2_pipeline`.
- Verify completed LVCG probing test run and trigger full multi-benchmark probe on completed checkpoints.
- Compute and render final Stage A qualification radar and Stage B Shapley graphs.
