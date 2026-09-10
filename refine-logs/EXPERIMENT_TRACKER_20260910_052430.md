# Experiment Tracker: LCT-MTL SAP Validation Pipeline

**Plan**: `refine-logs/EXPERIMENT_PLAN.md`  
**Last Updated**: 2026-09-10

## Block A — Champion Specification (COMPLETE)

| Cell | Model ID | r (Missing-11) | P₀₅ | Gate | Status |
|---|---|---|---|---|---|
| Anchor | lean2_L1_clean_lean_best_s42_l0 | 0.7552 | 0.4100 | Baseline | ✅ |
| T_patch10 | lean2_T_patch10_s42_l0 | **0.7635** | **0.4208** | PASS +0.0083 | ✅ SOTA |
| T_patch50 | lean2_T_patch50_s42_l0 | 0.7462 | — | FAIL −0.0090 | ✅ |
| T_patch5 | lean2_T_patch5_s42_l0 | 0.7612 | 0.4212 | PASS +0.0060 | ✅ |
| T_heads16 | lean2_T_heads16_s42_l0 | 0.7576 | — | PASS +0.0023 | ✅ |
| T_heads4 | lean2_T_heads4_s42_l0 | — | — | — | ✅ |
| LC1_width384 | lean2_LC1_width384_s42_l0 | 0.7538 | — | PASS (efficiency) | ✅ |
| LE1_nodice | lean2_LE1_nodice_s42_l0 | 0.7551 | 0.4149 | PASS NoDice | ✅ |
| LE2_adaptive | lean2_LE2_adaptive_s42_l0 | 0.7607 | 0.4184 | PASS +0.0054 | ✅ |
| LA1_nodel | lean2_LA1_nodel_s42_l0 | — | — | FAIL −0.0078 | ✅ |
| D_boundary | lean2_D_boundary_s42_l0 | 0.7566 | — | PASS +0.0013 | ✅ |
| L_mse_only | lean2_L_mse_only_s42_l0 | — | — | FAIL −0.0083 | ✅ |
| L_corr_only | lean2_L_corr_only_s42_l0 | — | — | FAIL −0.0102 | ✅ |
| L_mse_corr | lean2_L_mse_corr_s42_l0 | — | — | FAIL −0.0029 | ✅ |
| (other 11) | ... | — | — | evaluated | ✅ |

## Block B — Adaptive VCG & MMD Suite (QUEUED)

| # | Cell | Mask | Status | Δr vs T_patch10 | Gate 1 |
|---|---|---|---|---|---|
| 1 | AV1_vcg_adaptive | 1101000 | ⬜ QUEUED | — | — |
| 2 | AV2_triplet_vcg_adaptive | 1111000 | ⬜ QUEUED | — | — |
| 3 | AM1_mmd_imq_adaptive | 1100003 | ⬜ QUEUED | — | — |
| 4 | AM2_mmd_kmeans_adaptive | 1100004 | ⬜ QUEUED | — | — |
| 5 | AM3_mmd_laplace_adaptive | 1100002 | ⬜ QUEUED | — | — |
| 6 | AVM1_vcg_mmd_imq_adaptive | 1101003 | ⬜ QUEUED | — | — |
| 7 | AVM2_vcg_mmd_kmeans_adaptive | 1101004 | ⬜ QUEUED | — | — |
| 8 | AVM3_vcg_mmd_laplace_adaptive | 1101002 | ⬜ QUEUED | — | — |
| 9 | AVM4_full_probe_laplace_adaptive | 1111002 | ⬜ QUEUED | — | — |
| 10 | AVM5_full_probe_imq_adaptive | 1111003 | ⬜ QUEUED | — | — |
| 11 | AVLead1_vcg_lead_adaptive | 1101010 | ⬜ QUEUED | — | — |

## Block C — SAP Re-Evaluation (NOT STARTED)

| Task | Target | Status | Notes |
|---|---|---|---|
| Build `evaluate_sap_v2.py` | New evaluator | ⬜ NOT STARTED | Fixes Fisher-z, patient bootstrap |
| Smoke test (2 models) | anchor + T_patch10 | ⬜ NOT STARTED | Verify patient count = 1,904 |
| Full queue (188 models) | All models w/ best.pt | ⬜ NOT STARTED | ~18h CPU+GPU parallel |
| Build `SAP_BENCHMARK_V2.csv` | 258-col SAP table | ⬜ NOT STARTED | Replaces current master CSV |

## Block D — Confirmatory Inference (BLOCKED)

| Comparison | Metric | δₘ | Status |
|---|---|---|---|
| Primary vs anchor | mean_all_missing_r (Fisher-z) | 0.010 | ⬜ BLOCKED (B+C) |
| Primary vs anchor | ecgfounder_macro_150_auroc | 0.005 | ⬜ BLOCKED (C) |
| Primary vs anchor | echonext_shd_macro_12_auroc | 0.010 | ⬜ BLOCKED (C) |
| Primary vs anchor | semiseg_miou | 0.005 | ⬜ BLOCKED (C) |
| Primary vs anchor | qrs_duration_mae_ms | 2.0 ms | ⬜ BLOCKED (C) |
| Primary vs anchor | lvh_sokolowlyon_mae_mv | 0.05 mV | ⬜ BLOCKED (C) |

## Block E — Seed Stability (BLOCKED)

| Seed | Model | Status |
|---|---|---|
| 42 | lean2_T_patch10_s42_l0 | ✅ EXISTS |
| 43 | lean2_T_patch10_s43_l0 | ⬜ BLOCKED (primary model freeze) |
| 44 | lean2_T_patch10_s44_l0 | ⬜ BLOCKED (primary model freeze) |
