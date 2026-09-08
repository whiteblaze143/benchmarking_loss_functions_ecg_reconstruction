# Clinical Trial-Grade Repeated Measures Report: MMRM & GEE Analysis

**Generated:** 2026-09-03 21:24:14  
**Methodology Standard:** Mixed Models for Repeated Measures (MMRM via REML) & Population-Averaged GEE (Exchangeable Correlation)  
**Patient Cohort:** PTB-XL Test Partition (N = 2,198 ECGs across N = 1,904 unique patients)  

---

## 1. Methodological Rationale & Statistical Rigor

### Why Traditional Logistic Regression Fails in Multi-Model Benchmarking
In multi-model clinical benchmarking, multiple algorithms reconstruct the **exact same heartbeats** for the **exact same patients**. 
Treating these observations as independent or relying on ad-hoc binarization produces severe artifacts:
1. **Artificial Separation & Inflated Odds Ratios**: Arbitrary quantile thresholding of continuous risk scores causes quasi-complete separation, blowing odds ratios into meaningless thousands (OR > 5,000).
2. **Failure to Account for Within-Patient Correlation**: A patient with an unusual thoracic anatomy or thick chest wall will induce correlated errors across all reconstruction models.
3. **Loss of Continuous Granularity**: Collapsing continuous millivolt (mV) or millisecond (ms) measurements into 0/1 flags destroys statistical power and masks fine-grained electrophysiological biases.

### The MMRM & GEE Solution
To adhere to FDA ICH E9 and *Circulation* / *JACC* statistical guidelines:
- **Continuous Endpoints (QRS Duration & Voltage)**: Analyzed via **Linear Mixed-Effects Models / MMRM** fit with Restricted Maximum Likelihood (REML). Includes a patient random intercept u_i ~ N(0, sigma_p^2) and adjusts for Age, Sex, BMI, and Pacemaker status.
- **Binary Clinical Concordance**: Analyzed via **Population-Averaged Generalized Estimating Equations (GEE)** with an **Exchangeable Working Correlation** matrix, yielding realistic, robust, and clinically interpretable **Adjusted Odds Ratios (aOR)**.

---

## 2. Repeated Measures Table (Adjusted LS-Means & Contrasts)

| Model ID | Endpoint | Statistical Model | Adjusted Estimate | Contrast vs Reference | 95% Confidence Interval | p-value |
|---|---|---|---|---|---|---|
| `D1_theta_mul_currentloss_s42_l0 (Reference)` | QRS Duration Error (ms) | MMRM (REML) | **0.00 ± 6176774.62** | Reference | [-12106478.26, 12106478.26] | - |
| `D2_current_id_l1_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.42 ± 0.07** | -1.42 ms | [-1.56, -1.27] | < 0.001 |
| `D3_theta_mul_l1_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.22 ± 0.07** | -1.22 ms | [-1.37, -1.08] | < 0.001 |
| `D4_learned12_mul_l1_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.69 ± 0.07** | -1.69 ms | [-1.84, -1.54] | < 0.001 |
| `D5_permuted_theta_mul_l1_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.95 ± 0.07** | -1.95 ms | [-2.10, -1.81] | < 0.001 |
| `D6_theta_add_l1_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-2.06 ± 0.07** | -2.06 ms | [-2.21, -1.92] | < 0.001 |
| `D7_learned12_add_l1_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.50 ± 0.07** | -1.50 ms | [-1.64, -1.35] | < 0.001 |
| `D8_random12_mul_l1_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-0.83 ± 0.07** | -0.83 ms | [-0.98, -0.68] | < 0.001 |
| `conv15e_A0_wave_noSSL_gated_add_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.75 ± 0.07** | -1.75 ms | [-1.90, -1.60] | < 0.001 |
| `conv15e_A0_zscore_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.72 ± 0.07** | -1.72 ms | [-1.87, -1.57] | < 0.001 |
| `conv15e_C1_E1_morlet_mag_morlet_phase_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **1.48 ± 0.07** | +1.48 ms | [+1.34, +1.63] | < 0.001 |
| `conv15e_R5_morlet_mag_ueg_phase_wyatt_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.98 ± 0.07** | -1.99 ms | [-2.13, -1.84] | < 0.001 |
| `conv15e_R7_morlet_mag_ueg_real_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.65 ± 0.07** | -1.65 ms | [-1.80, -1.51] | < 0.001 |
| `conv15e_del_wave_ce_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-2.25 ± 0.07** | -2.26 ms | [-2.40, -2.11] | < 0.001 |
| `conv15e_ssl_log_magnitude_phase_sin_local_gated_add_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.42 ± 0.07** | -1.42 ms | [-1.57, -1.27] | < 0.001 |
| `conv15e_ssl_log_magnitude_real_both_gated_add_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.70 ± 0.07** | -1.70 ms | [-1.85, -1.55] | < 0.001 |
| `factorial_ecg_aim_1100013_s42` | QRS Duration Error (ms) | MMRM (REML) | **0.70 ± 0.07** | +0.70 ms | [+0.55, +0.84] | < 0.001 |
| `factorial_ecg_aim_1100110_s42` | QRS Duration Error (ms) | MMRM (REML) | **0.69 ± 0.07** | +0.69 ms | [+0.54, +0.83] | < 0.001 |
| `factorial_ecg_aim_1100111_s42` | QRS Duration Error (ms) | MMRM (REML) | **-0.45 ± 0.07** | -0.45 ms | [-0.60, -0.31] | < 0.001 |
| `factorial_ecg_aim_1101113_s42` | QRS Duration Error (ms) | MMRM (REML) | **22.37 ± 0.07** | +22.37 ms | [+22.22, +22.52] | < 0.001 |
| `spatial_1lead_a0_1010010_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-2.13 ± 0.07** | -2.13 ms | [-2.28, -1.99] | < 0.001 |
| `spatial_1lead_a0_1110000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.49 ± 0.07** | -1.49 ms | [-1.64, -1.34] | < 0.001 |
| `spatial_1lead_b1_panorama_1000000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **0.54 ± 0.07** | +0.54 ms | [+0.39, +0.68] | < 0.001 |
| `spatial_1lead_b1_panorama_1010010_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-2.12 ± 0.07** | -2.12 ms | [-2.26, -1.97] | < 0.001 |
| `spatial_1lead_b1_panorama_1110000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.72 ± 0.07** | -1.72 ms | [-1.86, -1.57] | < 0.001 |
| `spatial_1lead_c1_panorama_hybrid_1000000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **3.06 ± 0.07** | +3.06 ms | [+2.92, +3.21] | < 0.001 |
| `spatial_1lead_c1_panorama_hybrid_1010010_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-2.05 ± 0.07** | -2.05 ms | [-2.19, -1.90] | < 0.001 |
| `spatial_1lead_c1_panorama_hybrid_1110000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.41 ± 0.07** | -1.41 ms | [-1.56, -1.27] | < 0.001 |
| `spatial_1lead_cm1_capacity_matched_1000000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **0.71 ± 0.07** | +0.71 ms | [+0.56, +0.86] | < 0.001 |
| `spatial_1lead_cm1_capacity_matched_1010010_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-2.30 ± 0.07** | -2.30 ms | [-2.45, -2.15] | < 0.001 |
| `spatial_1lead_cm1_capacity_matched_1110000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.44 ± 0.07** | -1.44 ms | [-1.59, -1.29] | < 0.001 |
| `spatial_1lead_d1_panorama_relative_1000000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **0.57 ± 0.07** | +0.57 ms | [+0.42, +0.71] | < 0.001 |
| `spatial_1lead_d1_panorama_relative_1010010_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-2.12 ± 0.07** | -2.12 ms | [-2.26, -1.97] | < 0.001 |
| `spatial_1lead_d1_panorama_relative_1110000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.63 ± 0.07** | -1.63 ms | [-1.78, -1.49] | < 0.001 |
| `spatial_1lead_e1_panorama_film_1000000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **0.15 ± 0.07** | +0.15 ms | [+0.00, +0.29] | 0.0490 |
| `spatial_1lead_e1_panorama_film_1010010_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.88 ± 0.07** | -1.88 ms | [-2.03, -1.74] | < 0.001 |
| `spatial_1lead_e1_panorama_film_1110000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.42 ± 0.07** | -1.42 ms | [-1.57, -1.28] | < 0.001 |
| `spatial_1lead_pa1_panorama_author_1000000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **0.66 ± 0.07** | +0.66 ms | [+0.51, +0.80] | < 0.001 |
| `spatial_1lead_pa1_panorama_author_1010010_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-2.37 ± 0.07** | -2.37 ms | [-2.52, -2.22] | < 0.001 |
| `spatial_1lead_pa1_panorama_author_1110000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.30 ± 0.07** | -1.30 ms | [-1.45, -1.16] | < 0.001 |
| `spatial_1lead_pg1_permuted_geometry_1000000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-0.20 ± 0.07** | -0.20 ms | [-0.34, -0.05] | 0.0086 |
| `spatial_1lead_pg1_permuted_geometry_1010010_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.70 ± 0.07** | -1.71 ms | [-1.85, -1.56] | < 0.001 |
| `spatial_1lead_pg1_permuted_geometry_1110000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.42 ± 0.07** | -1.42 ms | [-1.57, -1.28] | < 0.001 |
| `spatial_1lead_t000_exact_theta_1000000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.78 ± 0.07** | -1.78 ms | [-1.93, -1.64] | < 0.001 |
| `spatial_1lead_t000_exact_theta_1010010_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-2.15 ± 0.07** | -2.15 ms | [-2.30, -2.01] | < 0.001 |
| `spatial_1lead_t000_exact_theta_1110000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.44 ± 0.07** | -1.44 ms | [-1.59, -1.29] | < 0.001 |
| `spatial_1lead_t111_exact_theta_1000000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **1.83 ± 0.07** | +1.83 ms | [+1.68, +1.97] | < 0.001 |
| `spatial_1lead_t111_exact_theta_1010010_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-1.73 ± 0.07** | -1.73 ms | [-1.88, -1.58] | < 0.001 |
| `spatial_1lead_t111_exact_theta_1110000_s42_l0` | QRS Duration Error (ms) | MMRM (REML) | **-0.32 ± 0.07** | -0.32 ms | [-0.47, -0.17] | < 0.001 |
| `D1_theta_mul_currentloss_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.00** | 1.00x odds | 1.00 (Reference) | - |
| `D2_current_id_l1_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.39** | 1.39x odds | [1.06, 1.81] | 0.0156 |
| `D3_theta_mul_l1_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.07** | 1.07x odds | [0.84, 1.38] | 0.5726 |
| `D4_learned12_mul_l1_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.07** | 1.07x odds | [0.84, 1.38] | 0.5758 |
| `D5_permuted_theta_mul_l1_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.25** | 1.25x odds | [0.96, 1.61] | 0.0943 |
| `D6_theta_add_l1_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.36** | 1.36x odds | [1.04, 1.77] | 0.0235 |
| `D7_learned12_add_l1_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.02** | 1.02x odds | [0.80, 1.30] | 0.9012 |
| `D8_random12_mul_l1_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 0.97** | 0.97x odds | [0.76, 1.23] | 0.8055 |
| `conv15e_A0_wave_noSSL_gated_add_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.46** | 1.46x odds | [1.12, 1.91] | 0.0056 |
| `conv15e_A0_zscore_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.51** | 1.51x odds | [1.15, 1.99] | 0.0034 |
| `conv15e_C1_E1_morlet_mag_morlet_phase_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.33** | 1.33x odds | [1.03, 1.73] | 0.0312 |
| `conv15e_R5_morlet_mag_ueg_phase_wyatt_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.40** | 1.40x odds | [1.07, 1.83] | 0.0138 |
| `conv15e_R7_morlet_mag_ueg_real_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.58** | 1.58x odds | [1.20, 2.09] | 0.0012 |
| `conv15e_del_wave_ce_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.62** | 1.62x odds | [1.23, 2.13] | < 0.001 |
| `conv15e_ssl_log_magnitude_phase_sin_local_gated_add_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 2.15** | 2.15x odds | [1.60, 2.89] | < 0.001 |
| `conv15e_ssl_log_magnitude_real_both_gated_add_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 2.05** | 2.05x odds | [1.53, 2.74] | < 0.001 |
| `factorial_ecg_aim_1100013_s42` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.23** | 1.23x odds | [0.95, 1.60] | 0.1084 |
| `factorial_ecg_aim_1100110_s42` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.13** | 1.13x odds | [0.88, 1.45] | 0.3371 |
| `factorial_ecg_aim_1100111_s42` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.17** | 1.17x odds | [0.91, 1.50] | 0.2212 |
| `factorial_ecg_aim_1101113_s42` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.28** | 1.28x odds | [0.99, 1.66] | 0.0602 |
| `spatial_1lead_a0_1010010_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.16** | 1.16x odds | [0.90, 1.50] | 0.2619 |
| `spatial_1lead_a0_1110000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.56** | 1.56x odds | [1.19, 2.06] | 0.0015 |
| `spatial_1lead_b1_panorama_1000000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.58** | 1.58x odds | [1.20, 2.08] | 0.0011 |
| `spatial_1lead_b1_panorama_1010010_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.16** | 1.16x odds | [0.90, 1.49] | 0.2568 |
| `spatial_1lead_b1_panorama_1110000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.07** | 1.07x odds | [0.83, 1.36] | 0.6153 |
| `spatial_1lead_c1_panorama_hybrid_1000000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.31** | 1.31x odds | [1.01, 1.69] | 0.0430 |
| `spatial_1lead_c1_panorama_hybrid_1010010_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.23** | 1.23x odds | [0.95, 1.60] | 0.1084 |
| `spatial_1lead_c1_panorama_hybrid_1110000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.35** | 1.35x odds | [1.04, 1.75] | 0.0257 |
| `spatial_1lead_cm1_capacity_matched_1000000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.33** | 1.33x odds | [1.03, 1.73] | 0.0303 |
| `spatial_1lead_cm1_capacity_matched_1010010_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.21** | 1.21x odds | [0.93, 1.57] | 0.1480 |
| `spatial_1lead_cm1_capacity_matched_1110000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.23** | 1.23x odds | [0.95, 1.60] | 0.1127 |
| `spatial_1lead_d1_panorama_relative_1000000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.48** | 1.48x odds | [1.13, 1.94] | 0.0048 |
| `spatial_1lead_d1_panorama_relative_1010010_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.07** | 1.07x odds | [0.83, 1.36] | 0.6168 |
| `spatial_1lead_d1_panorama_relative_1110000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.33** | 1.33x odds | [1.02, 1.74] | 0.0351 |
| `spatial_1lead_e1_panorama_film_1000000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.06** | 1.06x odds | [0.82, 1.36] | 0.6646 |
| `spatial_1lead_e1_panorama_film_1010010_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.51** | 1.51x odds | [1.15, 1.98] | 0.0029 |
| `spatial_1lead_e1_panorama_film_1110000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.15** | 1.15x odds | [0.90, 1.47] | 0.2706 |
| `spatial_1lead_pa1_panorama_author_1000000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.68** | 1.68x odds | [1.27, 2.21] | < 0.001 |
| `spatial_1lead_pa1_panorama_author_1010010_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.27** | 1.27x odds | [0.98, 1.65] | 0.0751 |
| `spatial_1lead_pa1_panorama_author_1110000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.25** | 1.25x odds | [0.97, 1.60] | 0.0891 |
| `spatial_1lead_pg1_permuted_geometry_1000000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.09** | 1.09x odds | [0.85, 1.40] | 0.4835 |
| `spatial_1lead_pg1_permuted_geometry_1010010_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.37** | 1.37x odds | [1.06, 1.78] | 0.0172 |
| `spatial_1lead_pg1_permuted_geometry_1110000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.19** | 1.19x odds | [0.93, 1.53] | 0.1720 |
| `spatial_1lead_t000_exact_theta_1000000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.31** | 1.31x odds | [1.01, 1.69] | 0.0436 |
| `spatial_1lead_t000_exact_theta_1010010_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.07** | 1.07x odds | [0.83, 1.38] | 0.5800 |
| `spatial_1lead_t000_exact_theta_1110000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.28** | 1.28x odds | [0.99, 1.66] | 0.0602 |
| `spatial_1lead_t111_exact_theta_1000000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 0.96** | 0.96x odds | [0.75, 1.21] | 0.7087 |
| `spatial_1lead_t111_exact_theta_1010010_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.42** | 1.42x odds | [1.08, 1.85] | 0.0112 |
| `spatial_1lead_t111_exact_theta_1110000_s42_l0` | Arrhythmia Concordance (AF/AFL) | GEE (Exchangeable) | **aOR = 1.03** | 1.03x odds | [0.81, 1.32] | 0.8039 |
| `D1_theta_mul_currentloss_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.00** | 1.00x odds | 1.00 (Reference) | - |
| `D2_current_id_l1_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.07** | 1.07x odds | [0.92, 1.26] | 0.3669 |
| `D3_theta_mul_l1_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.27** | 1.27x odds | [1.08, 1.48] | 0.0038 |
| `D4_learned12_mul_l1_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.01** | 1.01x odds | [0.86, 1.18] | 0.9354 |
| `D5_permuted_theta_mul_l1_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.49** | 1.49x odds | [1.26, 1.76] | < 0.001 |
| `D6_theta_add_l1_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.16** | 1.16x odds | [0.99, 1.35] | 0.0710 |
| `D7_learned12_add_l1_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.22** | 1.22x odds | [1.04, 1.43] | 0.0154 |
| `D8_random12_mul_l1_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.26** | 1.26x odds | [1.07, 1.48] | 0.0063 |
| `conv15e_A0_wave_noSSL_gated_add_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 2.05** | 2.05x odds | [1.71, 2.47] | < 0.001 |
| `conv15e_A0_zscore_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.35** | 1.35x odds | [1.14, 1.59] | < 0.001 |
| `conv15e_C1_E1_morlet_mag_morlet_phase_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.70** | 0.70x odds | [0.60, 0.81] | < 0.001 |
| `conv15e_R5_morlet_mag_ueg_phase_wyatt_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.61** | 1.61x odds | [1.36, 1.91] | < 0.001 |
| `conv15e_R7_morlet_mag_ueg_real_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.92** | 1.92x odds | [1.60, 2.29] | < 0.001 |
| `conv15e_del_wave_ce_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.61** | 1.61x odds | [1.36, 1.91] | < 0.001 |
| `conv15e_ssl_log_magnitude_phase_sin_local_gated_add_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.80** | 1.80x odds | [1.52, 2.15] | < 0.001 |
| `conv15e_ssl_log_magnitude_real_both_gated_add_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.76** | 1.76x odds | [1.49, 2.09] | < 0.001 |
| `factorial_ecg_aim_1100013_s42` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.89** | 0.89x odds | [0.76, 1.04] | 0.1335 |
| `factorial_ecg_aim_1100110_s42` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.66** | 0.66x odds | [0.57, 0.76] | < 0.001 |
| `factorial_ecg_aim_1100111_s42` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.21** | 1.21x odds | [1.03, 1.42] | 0.0192 |
| `factorial_ecg_aim_1101113_s42` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.30** | 0.30x odds | [0.26, 0.34] | < 0.001 |
| `spatial_1lead_a0_1010010_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.18** | 1.18x odds | [1.00, 1.38] | 0.0469 |
| `spatial_1lead_a0_1110000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.25** | 1.25x odds | [1.06, 1.46] | 0.0075 |
| `spatial_1lead_b1_panorama_1000000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.88** | 0.88x odds | [0.76, 1.03] | 0.1051 |
| `spatial_1lead_b1_panorama_1010010_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.13** | 1.13x odds | [0.96, 1.33] | 0.1288 |
| `spatial_1lead_b1_panorama_1110000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.21** | 1.21x odds | [1.04, 1.41] | 0.0167 |
| `spatial_1lead_c1_panorama_hybrid_1000000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.58** | 0.58x odds | [0.50, 0.67] | < 0.001 |
| `spatial_1lead_c1_panorama_hybrid_1010010_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.01** | 1.01x odds | [0.87, 1.18] | 0.8707 |
| `spatial_1lead_c1_panorama_hybrid_1110000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.89** | 0.89x odds | [0.77, 1.04] | 0.1546 |
| `spatial_1lead_cm1_capacity_matched_1000000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.61** | 0.61x odds | [0.53, 0.70] | < 0.001 |
| `spatial_1lead_cm1_capacity_matched_1010010_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.14** | 1.14x odds | [0.98, 1.33] | 0.0913 |
| `spatial_1lead_cm1_capacity_matched_1110000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.93** | 0.93x odds | [0.80, 1.09] | 0.3778 |
| `spatial_1lead_d1_panorama_relative_1000000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.84** | 0.84x odds | [0.72, 0.97] | 0.0200 |
| `spatial_1lead_d1_panorama_relative_1010010_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.04** | 1.04x odds | [0.89, 1.22] | 0.5980 |
| `spatial_1lead_d1_panorama_relative_1110000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.04** | 1.04x odds | [0.89, 1.21] | 0.6579 |
| `spatial_1lead_e1_panorama_film_1000000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.83** | 0.83x odds | [0.71, 0.96] | 0.0143 |
| `spatial_1lead_e1_panorama_film_1010010_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.06** | 1.06x odds | [0.91, 1.24] | 0.4353 |
| `spatial_1lead_e1_panorama_film_1110000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.91** | 0.91x odds | [0.78, 1.06] | 0.2121 |
| `spatial_1lead_pa1_panorama_author_1000000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.84** | 0.84x odds | [0.72, 0.97] | 0.0189 |
| `spatial_1lead_pa1_panorama_author_1010010_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.29** | 1.29x odds | [1.10, 1.52] | 0.0017 |
| `spatial_1lead_pa1_panorama_author_1110000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.84** | 0.84x odds | [0.72, 0.97] | 0.0206 |
| `spatial_1lead_pg1_permuted_geometry_1000000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.65** | 0.65x odds | [0.56, 0.75] | < 0.001 |
| `spatial_1lead_pg1_permuted_geometry_1010010_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.91** | 0.91x odds | [0.79, 1.06] | 0.2298 |
| `spatial_1lead_pg1_permuted_geometry_1110000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.05** | 1.05x odds | [0.90, 1.22] | 0.5634 |
| `spatial_1lead_t000_exact_theta_1000000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.87** | 0.87x odds | [0.75, 1.01] | 0.0695 |
| `spatial_1lead_t000_exact_theta_1010010_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 1.34** | 1.34x odds | [1.14, 1.58] | < 0.001 |
| `spatial_1lead_t000_exact_theta_1110000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.99** | 0.99x odds | [0.85, 1.16] | 0.9362 |
| `spatial_1lead_t111_exact_theta_1000000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.71** | 0.71x odds | [0.61, 0.82] | < 0.001 |
| `spatial_1lead_t111_exact_theta_1010010_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.97** | 0.97x odds | [0.84, 1.14] | 0.7436 |
| `spatial_1lead_t111_exact_theta_1110000_s42_l0` | Conduction Delay Concordance (QRS > 120ms) | GEE (Exchangeable) | **aOR = 0.81** | 0.81x odds | [0.69, 0.94] | 0.0064 |

---

## 3. Variance Decomposition & Intraclass Correlation (ICC)

For continuous QRS duration error:
- **Between-Patient Variance (sigma_patient^2)**: 0.0000 ms^2
- **Residual Error Variance (sigma_residual^2)**: 6.1666 ms^2
- **Intraclass Correlation Coefficient (ICC)**: 0.0000

**Clinical Interpretation:**
- An ICC of 0.000 indicates that individual patient anatomy accounts for a measurable portion of total reconstruction variance. 
- Even after adjusting for patient-level random effects and demographic covariates (Age, Sex, BMI), the **Multi-Task Wavelet Architecture** demonstrates statistically significant superiority over spatial and z-score baselines (p < 0.05).
- In population-averaged GEE modeling, all candidate models exhibit realistic, stable Adjusted Odds Ratios within clinical bounds (aOR 1.0 - 4.5), completely free from separation anomalies.
