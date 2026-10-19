# Experiment Plan: Paper 07 — 168 GPU-Hour SetOperator Architectural Ablation Campaign

## 1. Experimental Setup & Protocol
- **Dataset**: EchoNext ($N_{\text{train}} = 72,475$, $N_{\text{val}} = 4,626$, $N_{\text{test}} = 24,000$ sealed).
- **Task**: Binary Structural Heart Disease (`shd_moderate_or_greater_flag`).
- **Hardware**: Dedicated NVIDIA A100-PCIE-40GB (168 GPU-hour cap).
- **Primary Seed**: 42.
- **Training Protocol**:
  - Regime: `SO-Span` (50% basis subsets, 25% continuous Gaussian frames, 25% sparse linear frames, set size $k \in [1, 8]$).
  - Optimizer: AdamW ($\text{lr} = 10^{-4}$, $\text{weight\_decay} = 0.01$).
  - Scheduler: CosineAnnealingLR to $10^{-6}$ over 50 epochs.
  - Early-stopping patience: 10 epochs on official validation AUROC.
  - Batch size: 64 with gradient norm clipping at 1.0.

## 2. Evaluation Suite & Metrics
All models are evaluated on identical validation records:
1. **Canonical Clinical Configurations**:
   - R01: Full 12-lead (8 canonical basis leads)
   - R02: Six limb leads (2D Einthoven subspace)
   - R03: Leads I, II, III
   - R04: Leads I, II (basis of limb plane)
   - R05: Lead I
   - R06: Lead II
   - R07: Precordial differential ($V_3 - V_2$)
2. **Development Grassmann Bank**:
   - 21 continuous random orthonormal subspaces (ranks 1 through 7, 3 per rank).
   - Report mean macro AUROC and per-rank AUROC.
3. **Calibration & Reliability**: ECE and Brier score.
4. **Basis-Change Rotation Stability**: $\Delta p = \frac{1}{N} \sum |p(Q) - p(UQ)|$ for $U \in O(r)$.
5. **Redundant-Lead Sensitivity**: Prediction shift $|p_{R02} - p_{R04}|$.
6. **Inference Latency & Parameter Counts**: Milliseconds per patient, active classification params vs total instantiated.

## 3. Prioritized Execution Order & Run Budget

### Phase 1: High-Priority Core Hypotheses (Hours 0 – 30)
1. `A0_baseline`: Original SetOperator SO-Span (reused baseline checkpoint).
2. `A1_g1_scratch`: G1 SetOperator SO-Span (currently queued).
3. `A2_capacity_control`: Capacity-matched raw-$q$ MLP (`8 -> 96 -> 64`).
4. `D2_tabular_only`: 7-covariate clinical MLP baseline.
5. `D1_ecg_only`: Pure ECG morphology SetOperator.
6. `B2_phase_position_aware`: Softmax learned phase-cell weighting.

### Phase 2: Representation & Morphology Ablations (Hours 30 – 80)
7. `A3_no_q`: Constant/ablated operator token.
8. `B1_global_kme`: Global mean pooling over phase cells.
9. `C1_no_transformer`: Independent per-measurement MLP + attention pooling.
10. `C2_mean_pooling`: Masked mean pooling instead of learned-query pooling.
11. `B3_noncircular_conv`: Standard zero-padded Conv1d blocks.
12. `A4_projective_only`: Unsigned projective features ($\mathbb{R}^{36}$).
13. `A5_diagonal_quadratic`: Coordinate magnitude features ($\mathbb{R}^{16}$).

### Phase 3: Relational Attention & Set Structure (Hours 80 – 130)
14. `A6_g2_relational`: Squared-Gram relational attention bias $\beta (q_i^\top q_j)^2$.
15. `A7_diagonal_attention`: Diagonal attention bias control.
16. `B4_no_local_conv`: Parameter-matched per-phase MLP.
17. `B5_phase_disorder`: Controlled permutation of phase cells.
18. `C3_one_layer_transformer`: Depth = 1 Transformer.
19. `C4_single_head_attention`: Single-head attention ($n_{\text{head}} = 1$).
20. `D3_late_fusion`: Late concatenation of tabular covariates.

### Phase 4: Constructive Interactions & Synthesis (Hours 130 – 168)
21. `B6_raw_block_control`: SetOperator trained on raw waveform chunks.
22. `E1_g1_plus_phase_aware`: G1 operator MLP + learned phase weighting.
23. `E2_best_geometry_best_phase`: Best of Group A + Best of Group B.
24. `final_evaluation_and_synthesis`: Master leaderboard generation and decision memo.
