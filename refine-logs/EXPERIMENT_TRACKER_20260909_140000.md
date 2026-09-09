# Experiment Tracker: Round-2 Systematic Lean Suite & Chained Adaptive VCG/MMD Queue

**Tracking Ledger**: `refine-logs/lean_abl2/summary_ledger.json`  
**Master Log**: `refine-logs/lean_abl2/queue_master.log`  
**Clean Lean Anchor**: `lean2_L1_clean_lean_best_s42_l0` (Missing-11 Mean = **0.7552**, Tail $P_{05} = \mathbf{0.4100}$)  
**Peak Token SOTA**: `lean2_T_patch10_s42_l0` (Missing-11 Mean = **0.7635**, Tail $P_{05} = \mathbf{0.4208}$)  
**Peak Loss SOTA**: `lean2_LE2_adaptive_s42_l0` (Missing-11 Mean = **0.7607**, Tail $P_{05} = \mathbf{0.4184}$)

---

## 1. Active Execution Status

- **Status**: RUNNING (Autonomous Queue Execution in tmux session `lean_abl2`)
- **Active Job**: `D_heavy_seg` (Cell 25, Final Epochs)
- **Remaining Extended Cells**: `D_light_ce`, `S_film`, `S_panorama`, `R_mask15`, `R_tempmask15`, `R_wd_low`, `R_wd_high` (7 cells)
- **Chained Suite**: `lean_abl2_adaptive_vcg_mmd_queue.sh` (11 Jobs, updated with Champion LCT-MTL Base)

---

## 2. Systematic Round-2 Completed Cells Ledger ($N=25$)

| # | Cell ID | Status | Cand Mean $r$ | Delta vs Anchor | 95% Bootstrap CI | Worst-Tail $P_{05}$ | Decision Verdict |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `L1_clean_lean_best` | COMPLETED | 0.7552 | 0.0000 | Reference | 0.4100 | **PASS (Clean Anchor)** |
| 2 | `LA1_nodel` | COMPLETED | 0.7474 | -0.0078 | [-0.0095, -0.0061] | 0.3982 | **FAIL (Delineation is Essential)** |
| 3 | `LB1_dec3` | COMPLETED | 0.7518 | -0.0035 | [-0.0048, -0.0021] | 0.4017 | **FAIL (Dec-4 is Floor)** |
| 4 | `LC1_width384` | COMPLETED | 0.7538 | -0.0015 | [-0.0028, -0.0001] | 0.4075 | **PASS (Optimal Knee Point: -43.7% params)** |
| 5 | `LD1_enc3` | COMPLETED | 0.7547 | -0.0005 | [-0.0017, +0.0007] | 0.4041 | **FAIL (Lower Tail Degradation)** |
| 6 | `LE1_nodice` | COMPLETED | 0.7551 | -0.0002 | [-0.0014, +0.0010] | 0.4149 | **PASS (Dice Loss Pruned as Dead Weight)** |
| 7 | `LE2_adaptive` | COMPLETED | **0.7607** | **+0.0054** | [+0.0038, +0.0071] | **0.4184** | **PASS (Homoscedastic Loss SOTA)** |
| 8 | `L_mse_only` | COMPLETED | 0.7469 | -0.0083 | [-0.0099, -0.0067] | 0.4003 | **FAIL (R-peak Smoothing)** |
| 9 | `L_corr_only` | COMPLETED | 0.7450 | -0.0102 | [-0.0118, -0.0086] | 0.4263 | **FAIL (Voltage Scale Destroyed)** |
| 10 | `L_mse_corr` | COMPLETED | 0.7523 | -0.0029 | [-0.0041, -0.0017] | 0.4091 | **FAIL (Derivative Penalty Required)** |
| 11 | `L_mse_deriv` | COMPLETED | 0.7537 | -0.0015 | [-0.0028, -0.0002] | 0.4088 | **PASS (Triad Member)** |
| 12 | `L_l1_direct` | COMPLETED | 0.7456 | -0.0096 | [-0.0112, -0.0080] | 0.4025 | **FAIL (L1 Alone Collapses)** |
| 13 | `W_conv_c128` | COMPLETED | 0.7549 | -0.0003 | [-0.0016, +0.0010] | 0.4171 | **PASS (Compact Wavelet Front-End)** |
| 14 | `W_conv_c256` | COMPLETED | 0.7532 | -0.0020 | [-0.0033, -0.0007] | 0.4164 | **FAIL (Overparameterized Front-End)** |
| 15 | `W_timesformer_dim256` | COMPLETED | 0.7563 | +0.0011 | [-0.0002, +0.0024] | 0.4146 | **PASS (Temporal Attention Wavelet)** |
| 16 | `W_fusion_gated` | COMPLETED | 0.7562 | +0.0010 | [-0.0003, +0.0023] | 0.4200 | **PASS (Gated Additive Injection)** |
| 17 | `W_fusion_crossattn` | COMPLETED | 0.7560 | +0.0008 | [-0.0005, +0.0021] | 0.4125 | **PASS (Cross-Attention Conditioning)** |
| 18 | `T_patch50` | COMPLETED | 0.7462 | -0.0090 | [-0.0107, -0.0073] | 0.4064 | **FAIL (Token Striding Too Coarse)** |
| 19 | `T_patch10` | COMPLETED | **0.7635** | **+0.0083** | [+0.0065, +0.0101] | **0.4208** | **PASS (All-Time Project Peak SOTA)** |
| 20 | `T_heads4` | COMPLETED | 0.7528 | -0.0024 | [-0.0038, -0.0010] | 0.4139 | **FAIL (Subspace Under-Capacity)** |
| 21 | `T_heads16` | COMPLETED | **0.7576** | **+0.0023** | [+0.0009, +0.0037] | **0.4165** | **PASS (Zero Parameter Overhead)** |
| 22 | `D_cadence1` | COMPLETED | 0.7550 | -0.0002 | [-0.0015, +0.0011] | 0.4057 | **PASS (Neutral Cadence)** |
| 23 | `D_cadence4` | COMPLETED | 0.7525 | -0.0027 | [-0.0041, -0.0013] | 0.4057 | **FAIL (Delineation Drift)** |
| 24 | `D_boundary` | COMPLETED | **0.7566** | **+0.0013** | [+0.0001, +0.0025] | **0.4124** | **PASS (Transition Boundary Loss)** |
| 25 | `D_fiducial` | COMPLETED | 0.7559 | +0.0006 | [-0.0006, +0.0019] | 0.4107 | **PASS (Fiducial Landmark Loss)** |

---

## 3. Chained Adaptive VCG & MMD Suite Queue (11 Jobs)

*Updated with LCT-MTL Champion Base (Patch 10, Heads 16, Width 384, Boundary 0.2, No Dice, No Lead Dropout, Adaptive Loss)*

| # | Cell ID | Factorial Mask | Formulation | Status |
|:---:|:---|:---:|:---|:---:|
| 1 | `AV1_vcg_adaptive` | `1101000` | Adaptive MSE + Pearson + Kors VCG | QUEUED |
| 2 | `AV2_triplet_vcg_adaptive` | `1111000` | Adaptive Triplet + Kors VCG | QUEUED |
| 3 | `AM1_mmd_imq_adaptive` | `1100003` | Adaptive MSE + Pearson + Anatomical IMQ MMD | QUEUED |
| 4 | `AM2_mmd_kmeans_adaptive` | `1100004` | Adaptive MSE + Pearson + K-Means Dynamic MMD | QUEUED |
| 5 | `AM3_mmd_laplace_adaptive` | `1100002` | Adaptive MSE + Pearson + Anatomical Laplace MMD | QUEUED |
| 6 | `AVM1_vcg_mmd_imq_adaptive` | `1101003` | Adaptive VCG + Anatomical IMQ MMD | QUEUED |
| 7 | `AVM2_vcg_mmd_kmeans_adaptive` | `1101004` | Adaptive VCG + Temporal K-Means MMD | QUEUED |
| 8 | `AVM3_vcg_mmd_laplace_adaptive` | `1101002` | Adaptive VCG + Anatomical Laplace MMD | QUEUED |
| 9 | `AVM4_full_probe_laplace_adaptive` | `1111002` | Full Pentad: Triplet + VCG + Laplace MMD | QUEUED |
| 10 | `AVM5_full_probe_imq_adaptive` | `1111003` | Full Pentad: Triplet + VCG + IMQ MMD | QUEUED |
| 11 | `AVLead1_vcg_lead_adaptive` | `1101010` | Adaptive VCG + Goldberger Lead Consistency | QUEUED |
