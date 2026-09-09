# Experiment Tracker: Occam's Razor Lean Ablation Suite

**Tracking Ledger**: `refine-logs/lean_abl2/summary_ledger.json`  
**Master Log**: `refine-logs/lean_abl2/queue_master.log`  
**Current Anchor**: `conv15e_A0_raw_s42_l0` (Missing-11 Mean = 0.7456, Tail $p_{05} = 0.4048$)

---

## 1. Active Execution Status

- **Status**: RUNNING (Autonomous Queue Execution)
- **Active Job**: `L_corr_only` (Cell 9, Epoch 9/15)
- **Queued Jobs Remaining**: 35 jobs (Extended Queue Cells 10–33 + Phase 7 Adaptive VCG/MMD Cells 34–44)
- **Hardware**: NVIDIA A100-PCIE-40GB (Session: `lean_abl2`)
- **Current All-Time SOTA**: `LE2_adaptive` (Missing-11 Mean = **0.7607**, Tail $p_{05} = \mathbf{0.4185}$, $+0.0054$ over Clean Lean Core)

---

## 2. Master Progress Ledger

| # | Cell ID | Status | Cand Mean | Delta Mean | Delta p05 | Verdict |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | `L1_clean_lean_best` | COMPLETED | 0.7552 | +0.0097 | +0.0052 | **PASS** |
| 2 | `LA1_nodel` | COMPLETED | 0.7474 | -0.0078 | -0.0118 | **FAIL** (Delineation needed) |
| 3 | `LB1_dec3` | COMPLETED | 0.7518 | -0.0035 | -0.0082 | **FAIL** ($D=4$ is floor) |
| 4 | `LC1_width384` | COMPLETED | 0.7538 | -0.0015 | -0.0024 | **PASS** (Non-inferior) |
| 5 | `LD1_enc3` | COMPLETED | 0.7547 | -0.0005 | -0.0059 | **FAIL** (Tail degrades) |
| 6 | `LE1_nodice` | COMPLETED | 0.7551 | -0.0002 | +0.0050 | **PASS** (CE sufficient) |
| 7 | `LE2_adaptive` | COMPLETED | **0.7607** | **+0.0054** | **+0.0085** | **PASS — SOTA WINNER** |
| 8 | `L_mse_only` | COMPLETED | 0.7469 | -0.0083 | -0.0096 | **FAIL** (Pearson required) |
| 9 | `L_corr_only` | **RUNNING** (Epoch 9/15) | — | — | — | Active |
| 10 | `L_mse_corr` | QUEUED | — | — | — | Pending |
| 11 | `L_mse_deriv` | QUEUED | — | — | — | Pending |
| 12 | `L_l1_direct` | QUEUED | — | — | — | Pending |
| 13 | `W_conv_c128` | QUEUED | — | — | — | Pending |
| 14 | `W_conv_c256` | QUEUED | — | — | — | Pending |
| 15 | `W_timesformer_dim256` | QUEUED | — | — | — | Pending |
| 16 | `W_fusion_gated` | QUEUED | — | — | — | Pending |
| 17 | `W_fusion_crossattn` | QUEUED | — | — | — | Pending |
| 18 | `T_patch50` | QUEUED | — | — | — | Pending |
| 19 | `T_patch10` | QUEUED | — | — | — | Pending |
| 20 | `T_heads4` | QUEUED | — | — | — | Pending |
| 21 | `T_heads16` | QUEUED | — | — | — | Pending |
| 22 | `D_cadence1` | QUEUED | — | — | — | Pending |
| 23 | `D_cadence4` | QUEUED | — | — | — | Pending |
| 24 | `D_boundary` | QUEUED | — | — | — | Pending |
| 25 | `D_fiducial` | QUEUED | — | — | — | Pending |
| 26 | `D_heavy_seg` | QUEUED | — | — | — | Pending |
| 27 | `D_light_ce` | QUEUED | — | — | — | Pending |
| 28 | `S_film` | QUEUED | — | — | — | Pending |
| 29 | `S_panorama` | QUEUED | — | — | — | Pending |
| 30 | `R_mask15` | QUEUED | — | — | — | Pending |
| 31 | `R_tempmask15` | QUEUED | — | — | — | Pending |
| 32 | `R_wd_low` | QUEUED | — | — | — | Pending |
| 33 | `R_wd_high` | QUEUED | — | — | — | Pending |
| 34 | `AV1_vcg_adaptive` | QUEUED | — | — | — | Pending |
| 35 | `AV2_triplet_vcg_adaptive` | QUEUED | — | — | — | Pending |
| 36 | `AM1_mmd_imq_adaptive` | QUEUED | — | — | — | Pending |
| 37 | `AM2_mmd_kmeans_adaptive` | QUEUED | — | — | — | Pending |
| 38 | `AM3_mmd_laplace_adaptive` | QUEUED | — | — | — | Pending |
| 39 | `AVM1_vcg_mmd_imq_adaptive` | QUEUED | — | — | — | Pending |
| 40 | `AVM2_vcg_mmd_kmeans_adaptive` | QUEUED | — | — | — | Pending |
| 41 | `AVM3_vcg_mmd_laplace_adaptive` | QUEUED | — | — | — | Pending |
| 42 | `AVM4_full_probe_laplace_adaptive` | QUEUED | — | — | — | Pending |
| 43 | `AVM5_full_probe_imq_adaptive` | QUEUED | — | — | — | Pending |
| 44 | `AVLead1_vcg_lead_adaptive` | QUEUED | — | — | — | Pending |
