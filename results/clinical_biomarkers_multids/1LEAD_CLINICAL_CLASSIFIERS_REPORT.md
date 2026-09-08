# 1-Lead Downstream Clinical Foundation Models & Biomarkers Evaluation Report

**Generated:** 2026-09-03 21:25:36  
**Evaluation Version:** `1lead_clinical_v1`  
**Total Models Evaluated:** 49 / 55  

---

## 1. Top Clinical Diagnostic Leaders (PTB-XL ECGFounder 150 Tasks & EchoNext SHD)

| Rank | Model ID | ECGFounder 150 AUROC | EchoNext SHD AUROC | QRS Dur MAE (ms) | SemiSeg mIoU | V3 R-Var Ret % | Spurious Coupling (I->V3) |
|---|---|---|---|---|---|---|---|
| 1 | `conv15e_A0_wave_noSSL_gated_add_s42_l0` | **0.8208** | 0.7374 | 10.05 | 0.6661 | 20.2% | 8.06 |
| 2 | `conv15e_ssl_log_magnitude_real_both_gated_add_s42_l0` | **0.8205** | 0.7364 | 10.07 | 0.6638 | 18.7% | 8.59 |
| 3 | `conv15e_ssl_log_magnitude_phase_sin_local_gated_add_s42_l0` | **0.8199** | 0.7435 | 10.18 | 0.6644 | 17.4% | 9.55 |
| 4 | `conv15e_R5_morlet_mag_ueg_phase_wyatt_s42_l0` | **0.8180** | 0.7441 | 9.96 | 0.6629 | 16.5% | 9.29 |
| 5 | `conv15e_R7_morlet_mag_ueg_real_s42_l0` | **0.8179** | 0.7462 | 10.13 | 0.6638 | 21.8% | 8.75 |
| 6 | `conv15e_del_wave_ce_s42_l0` | **0.8179** | 0.7373 | 9.76 | 0.6630 | 16.6% | 9.87 |
| 7 | `spatial_1lead_c1_panorama_hybrid_1110000_s42_l0` | **0.8141** | 0.7321 | 10.22 | 0.5890 | 16.0% | 13.50 |
| 8 | `spatial_1lead_t000_exact_theta_1110000_s42_l0` | **0.8131** | 0.7355 | 10.25 | 0.5923 | 15.3% | 13.25 |
| 9 | `spatial_1lead_pg1_permuted_geometry_1110000_s42_l0` | **0.8131** | 0.7367 | 10.19 | 0.5840 | 15.2% | 14.73 |
| 10 | `spatial_1lead_e1_panorama_film_1110000_s42_l0` | **0.8124** | 0.7215 | 10.20 | 0.6014 | 15.7% | 15.08 |
| 11 | `spatial_1lead_pa1_panorama_author_1110000_s42_l0` | **0.8124** | 0.7252 | 10.23 | 0.5936 | 16.5% | 13.97 |
| 12 | `spatial_1lead_a0_1110000_s42_l0` | **0.8124** | 0.7323 | 10.26 | 0.5943 | 15.5% | 14.42 |
| 13 | `spatial_1lead_cm1_capacity_matched_1110000_s42_l0` | **0.8119** | 0.7361 | 10.17 | 0.5930 | 16.4% | 14.10 |
| 14 | `spatial_1lead_d1_panorama_relative_1110000_s42_l0` | **0.8118** | 0.7222 | 10.09 | 0.5914 | 16.0% | 14.54 |
| 15 | `D8_random12_mul_l1_s42_l0` | **0.8116** | 0.6896 | 10.58 | 0.5934 | 10.1% | 5.41 |

---

## 2. Key Observations & Invariants

1. **Downstream Foundation Model Diagnostic Preservation**:
   - Single Lead I reconstruction maintains high macro AUROC across all 150 PTB-XL clinical diagnostic tasks.
2. **Author-Faithful Deep Wave Delineation (SemiSeg ViT-Tiny)**:
   - Evaluated using frozen Mean Teacher LUDB weights on CUDA, computing exact P, QRS, T wave segmentation masks and median beat QRS duration.
3. **PreSACAN Representation Invariance**:
   - Physical R-wave variance retention and interlead spurious coupling ($I \to V_3$) distinguish models that genuinely reconstruct independent precordial physics versus models that hallucinate correlated projections.

*Full results logged in `/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/results/clinical_biomarkers_multids/clinical_metrics.db`.*
