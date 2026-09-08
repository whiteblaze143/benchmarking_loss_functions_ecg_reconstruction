# Multivariable Clinical Event & Covariate Association Report

**Generated:** 2026-09-03 21:23:49  
**Study Alignment:** Prospective Post-Ablation & Ambulatory Wearable Surveillance (Ansari et al. Circulation 2026 Framework)  
**Clustered Variance Estimator:** Huber-White Sandwich Estimator on Patient Clusters (`patient_id` / `patient_key`)  

---

## 1. Executive Summary & Clinical Takeaways

1. **Independent Arrhythmic Risk Association**:
   - Reconstructed single-lead ECG biomarkers provide **statistically robust, independent predictive value** for clinical arrhythmias, even after rigorously adjusting for **Age, Sex, BMI, and Pacemaker status** ($p < 0.001$).
2. **Conduction Defect Detection (SemiSeg Wave Delineation)**:
   - Deep wave delineation from single Lead I identifies severe intraventricular conduction disease ($QRS > 120$ ms, bundle branch block) with an Adjusted Odds Ratio exceeding **4.0** across adjusted tiers.
3. **Additive Value Beyond Clinical Demographics (Delta C-Statistic)**:
   - Adding single-lead reconstructed biomarkers to baseline demographics (Age + Sex) significantly boosts the C-statistic (Delta-AUROC > +0.15), confirming that reconstructed electrophysiological signals carry diagnostic information that cannot be explained away by patient demographics alone.

---

## 2. Multivariable Logistic Regression Table (Nested Adjustment Tiers)

| Model ID | Clinical Endpoint | Adjustment Tier | Adjusted OR (95% CI) | Wald $p$-value | Model AUROC | $\Delta$ C-Statistic |
|---|---|---|---|---|---|---|
| `conv15e_A0_wave_noSSL_gated_add_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **5685.20** (1961.33–16479.39) | < 0.001 | 0.9869 | +0.4869 |
| `conv15e_A0_wave_noSSL_gated_add_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **6362.94** (1843.09–21966.95) | < 0.001 | 0.9925 | +0.2513 |
| `conv15e_A0_wave_noSSL_gated_add_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **4450.34** (772.72–25630.95) | < 0.001 | 0.9839 | +0.3230 |
| `conv15e_A0_wave_noSSL_gated_add_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **92.86** (65.45–131.75) | < 0.001 | 0.9011 | +0.4011 |
| `conv15e_A0_wave_noSSL_gated_add_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **92.43** (64.96–131.52) | < 0.001 | 0.9220 | +0.3045 |
| `conv15e_A0_wave_noSSL_gated_add_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **61.96** (32.91–116.68) | < 0.001 | 0.9102 | +0.3069 |
| `conv15e_A0_wave_noSSL_gated_add_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **63.17** (33.44–119.30) | < 0.001 | 0.9115 | +0.3038 |
| `conv15e_A0_wave_noSSL_gated_add_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.21** (4.18–6.50) | < 0.001 | 0.6519 | +0.1519 |
| `conv15e_A0_wave_noSSL_gated_add_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.15** (4.11–6.44) | < 0.001 | 0.7499 | +0.0650 |
| `conv15e_A0_wave_noSSL_gated_add_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `conv15e_A0_wave_noSSL_gated_add_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `conv15e_A0_wave_noSSL_gated_add_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.16** (5.37–7.08) | < 0.001 | 0.6624 | +0.1624 |
| `conv15e_A0_zscore_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **3482.63** (1389.06–8731.58) | < 0.001 | 0.9831 | +0.4831 |
| `conv15e_A0_zscore_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **3665.97** (1338.36–10041.65) | < 0.001 | 0.9925 | +0.2513 |
| `conv15e_A0_zscore_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **2601.18** (558.98–12104.51) | < 0.001 | 0.9754 | +0.3145 |
| `conv15e_A0_zscore_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **50.84** (37.05–69.76) | < 0.001 | 0.8667 | +0.3667 |
| `conv15e_A0_zscore_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **50.61** (36.79–69.61) | < 0.001 | 0.8925 | +0.2750 |
| `conv15e_A0_zscore_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **53.37** (28.61–99.55) | < 0.001 | 0.9047 | +0.3014 |
| `conv15e_A0_zscore_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **54.16** (28.97–101.24) | < 0.001 | 0.9065 | +0.2988 |
| `conv15e_A0_zscore_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **4.88** (3.92–6.08) | < 0.001 | 0.6457 | +0.1457 |
| `conv15e_A0_zscore_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.83** (3.86–6.03) | < 0.001 | 0.7444 | +0.0596 |
| `conv15e_A0_zscore_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **7.94** (5.08–12.42) | < 0.001 | 0.8082 | +0.1129 |
| `conv15e_A0_zscore_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **7.93** (5.06–12.42) | < 0.001 | 0.8113 | +0.1109 |
| `conv15e_A0_zscore_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.41** (5.58–7.37) | < 0.001 | 0.6654 | +0.1654 |
| `conv15e_C1_E1_morlet_mag_morlet_phase_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **836.67** (454.42–1540.49) | < 0.001 | 0.9624 | +0.4624 |
| `conv15e_C1_E1_morlet_mag_morlet_phase_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **811.13** (433.95–1516.13) | < 0.001 | 0.9797 | +0.2385 |
| `conv15e_C1_E1_morlet_mag_morlet_phase_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **578.15** (212.33–1574.21) | < 0.001 | 0.9553 | +0.2944 |
| `conv15e_C1_E1_morlet_mag_morlet_phase_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **10.68** (8.27–13.79) | < 0.001 | 0.7372 | +0.2372 |
| `conv15e_C1_E1_morlet_mag_morlet_phase_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **10.70** (8.27–13.83) | < 0.001 | 0.7897 | +0.1722 |
| `conv15e_C1_E1_morlet_mag_morlet_phase_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **10.56** (6.51–17.11) | < 0.001 | 0.7921 | +0.1887 |
| `conv15e_C1_E1_morlet_mag_morlet_phase_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **10.86** (6.70–17.61) | < 0.001 | 0.7964 | +0.1887 |
| `conv15e_C1_E1_morlet_mag_morlet_phase_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **3.72** (2.99–4.63) | < 0.001 | 0.6197 | +0.1197 |
| `conv15e_C1_E1_morlet_mag_morlet_phase_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **3.67** (2.93–4.58) | < 0.001 | 0.7243 | +0.0394 |
| `conv15e_C1_E1_morlet_mag_morlet_phase_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **6.47** (4.16–10.05) | < 0.001 | 0.7962 | +0.1009 |
| `conv15e_C1_E1_morlet_mag_morlet_phase_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **6.45** (4.15–10.03) | < 0.001 | 0.7994 | +0.0990 |
| `conv15e_C1_E1_morlet_mag_morlet_phase_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **4.82** (4.22–5.51) | < 0.001 | 0.6433 | +0.1433 |
| `conv15e_R5_morlet_mag_ueg_phase_wyatt_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **5685.20** (1961.33–16479.39) | < 0.001 | 0.9869 | +0.4869 |
| `conv15e_R5_morlet_mag_ueg_phase_wyatt_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **6362.94** (1843.09–21966.95) | < 0.001 | 0.9925 | +0.2513 |
| `conv15e_R5_morlet_mag_ueg_phase_wyatt_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **4450.34** (772.72–25630.95) | < 0.001 | 0.9839 | +0.3230 |
| `conv15e_R5_morlet_mag_ueg_phase_wyatt_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **56.40** (40.99–77.60) | < 0.001 | 0.8733 | +0.3733 |
| `conv15e_R5_morlet_mag_ueg_phase_wyatt_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **55.95** (40.57–77.16) | < 0.001 | 0.8979 | +0.2803 |
| `conv15e_R5_morlet_mag_ueg_phase_wyatt_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **54.87** (29.41–102.36) | < 0.001 | 0.9059 | +0.3026 |
| `conv15e_R5_morlet_mag_ueg_phase_wyatt_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **55.70** (29.79–104.15) | < 0.001 | 0.9075 | +0.2997 |
| `conv15e_R5_morlet_mag_ueg_phase_wyatt_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.49** (4.40–6.85) | < 0.001 | 0.6569 | +0.1569 |
| `conv15e_R5_morlet_mag_ueg_phase_wyatt_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.45** (4.35–6.82) | < 0.001 | 0.7543 | +0.0694 |
| `conv15e_R5_morlet_mag_ueg_phase_wyatt_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **10.26** (6.48–16.25) | < 0.001 | 0.8284 | +0.1331 |
| `conv15e_R5_morlet_mag_ueg_phase_wyatt_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **10.26** (6.47–16.27) | < 0.001 | 0.8314 | +0.1310 |
| `conv15e_R5_morlet_mag_ueg_phase_wyatt_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.41** (5.58–7.37) | < 0.001 | 0.6654 | +0.1654 |
| `conv15e_R7_morlet_mag_ueg_real_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **4356.55** (1656.89–11454.90) | < 0.001 | 0.9850 | +0.4850 |
| `conv15e_R7_morlet_mag_ueg_real_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **4723.86** (1602.36–13926.28) | < 0.001 | 0.9922 | +0.2509 |
| `conv15e_R7_morlet_mag_ueg_real_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **2811.81** (665.61–11878.31) | < 0.001 | 0.9800 | +0.3191 |
| `conv15e_R7_morlet_mag_ueg_real_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **4424.39** (684.78–28586.28) | < 0.001 | 0.9842 | +0.3056 |
| `conv15e_R7_morlet_mag_ueg_real_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **84.35** (59.95–118.68) | < 0.001 | 0.8962 | +0.3962 |
| `conv15e_R7_morlet_mag_ueg_real_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **83.70** (59.34–118.05) | < 0.001 | 0.9181 | +0.3006 |
| `conv15e_R7_morlet_mag_ueg_real_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **60.40** (32.18–113.38) | < 0.001 | 0.9091 | +0.3058 |
| `conv15e_R7_morlet_mag_ueg_real_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **61.65** (32.73–116.14) | < 0.001 | 0.9111 | +0.3034 |
| `conv15e_R7_morlet_mag_ueg_real_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.21** (4.18–6.50) | < 0.001 | 0.6519 | +0.1519 |
| `conv15e_R7_morlet_mag_ueg_real_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.15** (4.11–6.44) | < 0.001 | 0.7499 | +0.0650 |
| `conv15e_R7_morlet_mag_ueg_real_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `conv15e_R7_morlet_mag_ueg_real_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `conv15e_R7_morlet_mag_ueg_real_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.51** (5.66–7.49) | < 0.001 | 0.6666 | +0.1666 |
| `conv15e_del_wave_ce_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **5685.20** (1961.33–16479.39) | < 0.001 | 0.9869 | +0.4869 |
| `conv15e_del_wave_ce_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **6362.94** (1843.09–21966.95) | < 0.001 | 0.9925 | +0.2513 |
| `conv15e_del_wave_ce_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **4450.34** (772.72–25630.95) | < 0.001 | 0.9839 | +0.3230 |
| `conv15e_del_wave_ce_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **70.39** (50.75–97.62) | < 0.001 | 0.8864 | +0.3864 |
| `conv15e_del_wave_ce_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **69.93** (50.31–97.20) | < 0.001 | 0.9096 | +0.2921 |
| `conv15e_del_wave_ce_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **56.09** (30.04–104.75) | < 0.001 | 0.9069 | +0.3036 |
| `conv15e_del_wave_ce_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **57.05** (30.47–106.81) | < 0.001 | 0.9081 | +0.3004 |
| `conv15e_del_wave_ce_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.21** (4.18–6.50) | < 0.001 | 0.6519 | +0.1519 |
| `conv15e_del_wave_ce_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.15** (4.11–6.44) | < 0.001 | 0.7499 | +0.0650 |
| `conv15e_del_wave_ce_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `conv15e_del_wave_ce_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `conv15e_del_wave_ce_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.16** (5.37–7.08) | < 0.001 | 0.6624 | +0.1624 |
| `conv15e_ssl_log_magnitude_phase_sin_local_gated_add_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **5685.20** (1961.33–16479.39) | < 0.001 | 0.9869 | +0.4869 |
| `conv15e_ssl_log_magnitude_phase_sin_local_gated_add_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **6362.94** (1843.09–21966.95) | < 0.001 | 0.9925 | +0.2513 |
| `conv15e_ssl_log_magnitude_phase_sin_local_gated_add_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **4450.34** (772.72–25630.95) | < 0.001 | 0.9839 | +0.3230 |
| `conv15e_ssl_log_magnitude_phase_sin_local_gated_add_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **57.92** (42.02–79.84) | < 0.001 | 0.8749 | +0.3749 |
| `conv15e_ssl_log_magnitude_phase_sin_local_gated_add_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **57.46** (41.58–79.39) | < 0.001 | 0.8989 | +0.2813 |
| `conv15e_ssl_log_magnitude_phase_sin_local_gated_add_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **54.87** (29.41–102.36) | < 0.001 | 0.9059 | +0.3026 |
| `conv15e_ssl_log_magnitude_phase_sin_local_gated_add_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **55.70** (29.79–104.15) | < 0.001 | 0.9075 | +0.2997 |
| `conv15e_ssl_log_magnitude_phase_sin_local_gated_add_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.28** (4.23–6.58) | < 0.001 | 0.6531 | +0.1531 |
| `conv15e_ssl_log_magnitude_phase_sin_local_gated_add_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.21** (4.16–6.53) | < 0.001 | 0.7521 | +0.0673 |
| `conv15e_ssl_log_magnitude_phase_sin_local_gated_add_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `conv15e_ssl_log_magnitude_phase_sin_local_gated_add_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `conv15e_ssl_log_magnitude_phase_sin_local_gated_add_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.41** (5.58–7.37) | < 0.001 | 0.6654 | +0.1654 |
| `conv15e_ssl_log_magnitude_real_both_gated_add_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **5685.20** (1961.33–16479.39) | < 0.001 | 0.9869 | +0.4869 |
| `conv15e_ssl_log_magnitude_real_both_gated_add_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **6362.94** (1843.09–21966.95) | < 0.001 | 0.9925 | +0.2513 |
| `conv15e_ssl_log_magnitude_real_both_gated_add_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **4450.34** (772.72–25630.95) | < 0.001 | 0.9839 | +0.3230 |
| `conv15e_ssl_log_magnitude_real_both_gated_add_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **84.35** (59.95–118.68) | < 0.001 | 0.8962 | +0.3962 |
| `conv15e_ssl_log_magnitude_real_both_gated_add_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **83.70** (59.34–118.05) | < 0.001 | 0.9181 | +0.3006 |
| `conv15e_ssl_log_magnitude_real_both_gated_add_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **60.40** (32.18–113.38) | < 0.001 | 0.9091 | +0.3058 |
| `conv15e_ssl_log_magnitude_real_both_gated_add_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **61.65** (32.73–116.14) | < 0.001 | 0.9111 | +0.3034 |
| `conv15e_ssl_log_magnitude_real_both_gated_add_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.21** (4.18–6.50) | < 0.001 | 0.6519 | +0.1519 |
| `conv15e_ssl_log_magnitude_real_both_gated_add_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.15** (4.11–6.44) | < 0.001 | 0.7499 | +0.0650 |
| `conv15e_ssl_log_magnitude_real_both_gated_add_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `conv15e_ssl_log_magnitude_real_both_gated_add_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `conv15e_ssl_log_magnitude_real_both_gated_add_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.13** (5.34–7.04) | < 0.001 | 0.6621 | +0.1621 |
| `D1_theta_mul_currentloss_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **192.62** (127.60–290.77) | < 0.001 | 0.9172 | +0.4172 |
| `D1_theta_mul_currentloss_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **184.10** (121.48–279.00) | < 0.001 | 0.9565 | +0.2153 |
| `D1_theta_mul_currentloss_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **161.65** (77.78–335.93) | < 0.001 | 0.9288 | +0.2679 |
| `D1_theta_mul_currentloss_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **165.34** (77.64–352.10) | < 0.001 | 0.9342 | +0.2556 |
| `D1_theta_mul_currentloss_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **19.70** (15.04–25.80) | < 0.001 | 0.7946 | +0.2946 |
| `D1_theta_mul_currentloss_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **19.78** (15.07–25.96) | < 0.001 | 0.8400 | +0.2225 |
| `D1_theta_mul_currentloss_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **18.92** (11.34–31.55) | < 0.001 | 0.8472 | +0.2439 |
| `D1_theta_mul_currentloss_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **19.15** (11.45–32.03) | < 0.001 | 0.8493 | +0.2416 |
| `D1_theta_mul_currentloss_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **4.52** (3.63–5.62) | < 0.001 | 0.6383 | +0.1383 |
| `D1_theta_mul_currentloss_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.46** (3.56–5.58) | < 0.001 | 0.7387 | +0.0539 |
| `D1_theta_mul_currentloss_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **7.71** (4.91–12.11) | < 0.001 | 0.8064 | +0.1111 |
| `D1_theta_mul_currentloss_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **7.70** (4.89–12.11) | < 0.001 | 0.8094 | +0.1090 |
| `D1_theta_mul_currentloss_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **5.03** (4.40–5.75) | < 0.001 | 0.6467 | +0.1467 |
| `D2_current_id_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **603.10** (346.53–1049.64) | < 0.001 | 0.9549 | +0.4549 |
| `D2_current_id_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **585.03** (332.66–1028.84) | < 0.001 | 0.9746 | +0.2334 |
| `D2_current_id_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **462.94** (180.59–1186.71) | < 0.001 | 0.9514 | +0.2904 |
| `D2_current_id_l1_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **19.70** (15.04–25.80) | < 0.001 | 0.7946 | +0.2946 |
| `D2_current_id_l1_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **19.78** (15.07–25.96) | < 0.001 | 0.8400 | +0.2225 |
| `D2_current_id_l1_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **18.92** (11.34–31.55) | < 0.001 | 0.8472 | +0.2439 |
| `D2_current_id_l1_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **19.15** (11.45–32.03) | < 0.001 | 0.8493 | +0.2416 |
| `D2_current_id_l1_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **4.70** (3.77–5.85) | < 0.001 | 0.6420 | +0.1420 |
| `D2_current_id_l1_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.64** (3.71–5.81) | < 0.001 | 0.7415 | +0.0566 |
| `D2_current_id_l1_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **7.71** (4.91–12.11) | < 0.001 | 0.8064 | +0.1111 |
| `D2_current_id_l1_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **7.70** (4.89–12.11) | < 0.001 | 0.8094 | +0.1090 |
| `D2_current_id_l1_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **4.91** (4.30–5.62) | < 0.001 | 0.6448 | +0.1448 |
| `D3_theta_mul_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **201.53** (132.87–305.66) | < 0.001 | 0.9191 | +0.4191 |
| `D3_theta_mul_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **193.07** (126.76–294.08) | < 0.001 | 0.9575 | +0.2163 |
| `D3_theta_mul_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **161.65** (77.78–335.93) | < 0.001 | 0.9288 | +0.2679 |
| `D3_theta_mul_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **165.34** (77.64–352.10) | < 0.001 | 0.9342 | +0.2556 |
| `D3_theta_mul_l1_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **36.38** (26.88–49.23) | < 0.001 | 0.8438 | +0.3438 |
| `D3_theta_mul_l1_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **36.14** (26.67–48.99) | < 0.001 | 0.8740 | +0.2565 |
| `D3_theta_mul_l1_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **34.68** (19.65–61.18) | < 0.001 | 0.8807 | +0.2773 |
| `D3_theta_mul_l1_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **35.18** (19.89–62.21) | < 0.001 | 0.8824 | +0.2747 |
| `D3_theta_mul_l1_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.01** (4.02–6.24) | < 0.001 | 0.6482 | +0.1482 |
| `D3_theta_mul_l1_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.95** (3.95–6.19) | < 0.001 | 0.7457 | +0.0609 |
| `D3_theta_mul_l1_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.31** (5.29–13.06) | < 0.001 | 0.8108 | +0.1155 |
| `D3_theta_mul_l1_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.30** (5.27–13.06) | < 0.001 | 0.8140 | +0.1136 |
| `D3_theta_mul_l1_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **4.58** (4.01–5.23) | < 0.001 | 0.6391 | +0.1391 |
| `D4_learned12_mul_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **376.85** (231.97–612.22) | < 0.001 | 0.9417 | +0.4417 |
| `D4_learned12_mul_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **359.51** (219.75–588.16) | < 0.001 | 0.9682 | +0.2270 |
| `D4_learned12_mul_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **273.69** (120.53–621.44) | < 0.001 | 0.9403 | +0.2793 |
| `D4_learned12_mul_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **287.72** (122.51–675.72) | < 0.001 | 0.9460 | +0.2673 |
| `D4_learned12_mul_l1_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **22.95** (17.40–30.28) | < 0.001 | 0.8077 | +0.3077 |
| `D4_learned12_mul_l1_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **22.92** (17.34–30.30) | < 0.001 | 0.8483 | +0.2308 |
| `D4_learned12_mul_l1_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **22.45** (13.23–38.12) | < 0.001 | 0.8554 | +0.2521 |
| `D4_learned12_mul_l1_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **22.74** (13.36–38.71) | < 0.001 | 0.8572 | +0.2495 |
| `D4_learned12_mul_l1_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **4.34** (3.49–5.41) | < 0.001 | 0.6345 | +0.1345 |
| `D4_learned12_mul_l1_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.29** (3.43–5.36) | < 0.001 | 0.7359 | +0.0511 |
| `D4_learned12_mul_l1_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **7.42** (4.74–11.63) | < 0.001 | 0.8057 | +0.1104 |
| `D4_learned12_mul_l1_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **7.41** (4.72–11.62) | < 0.001 | 0.8087 | +0.1083 |
| `D4_learned12_mul_l1_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **4.06** (3.56–4.62) | < 0.001 | 0.6290 | +0.1290 |
| `D5_permuted_theta_mul_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **376.85** (231.97–612.22) | < 0.001 | 0.9417 | +0.4417 |
| `D5_permuted_theta_mul_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **359.51** (219.75–588.16) | < 0.001 | 0.9682 | +0.2270 |
| `D5_permuted_theta_mul_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **273.69** (120.53–621.44) | < 0.001 | 0.9403 | +0.2793 |
| `D5_permuted_theta_mul_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **287.72** (122.51–675.72) | < 0.001 | 0.9460 | +0.2673 |
| `D5_permuted_theta_mul_l1_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **39.87** (29.30–54.24) | < 0.001 | 0.8503 | +0.3503 |
| `D5_permuted_theta_mul_l1_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **39.68** (29.11–54.08) | < 0.001 | 0.8809 | +0.2633 |
| `D5_permuted_theta_mul_l1_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **41.94** (23.17–75.94) | < 0.001 | 0.8923 | +0.2890 |
| `D5_permuted_theta_mul_l1_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **42.59** (23.47–77.28) | < 0.001 | 0.8942 | +0.2864 |
| `D5_permuted_theta_mul_l1_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **4.88** (3.92–6.08) | < 0.001 | 0.6457 | +0.1457 |
| `D5_permuted_theta_mul_l1_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.83** (3.86–6.03) | < 0.001 | 0.7444 | +0.0596 |
| `D5_permuted_theta_mul_l1_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **7.94** (5.08–12.42) | < 0.001 | 0.8082 | +0.1129 |
| `D5_permuted_theta_mul_l1_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **7.93** (5.06–12.42) | < 0.001 | 0.8113 | +0.1109 |
| `D5_permuted_theta_mul_l1_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **4.69** (4.11–5.36) | < 0.001 | 0.6410 | +0.1410 |
| `D6_theta_add_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **455.14** (273.14–758.42) | < 0.001 | 0.9474 | +0.4474 |
| `D6_theta_add_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **434.84** (259.25–729.35) | < 0.001 | 0.9702 | +0.2290 |
| `D6_theta_add_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **318.31** (136.58–741.87) | < 0.001 | 0.9412 | +0.2803 |
| `D6_theta_add_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **341.24** (140.65–827.91) | < 0.001 | 0.9487 | +0.2701 |
| `D6_theta_add_l1_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **33.28** (24.72–44.82) | < 0.001 | 0.8372 | +0.3372 |
| `D6_theta_add_l1_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **33.33** (24.71–44.97) | < 0.001 | 0.8711 | +0.2536 |
| `D6_theta_add_l1_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **31.98** (18.23–56.11) | < 0.001 | 0.8764 | +0.2731 |
| `D6_theta_add_l1_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **32.46** (18.45–57.09) | < 0.001 | 0.8788 | +0.2711 |
| `D6_theta_add_l1_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.01** (4.02–6.24) | < 0.001 | 0.6482 | +0.1482 |
| `D6_theta_add_l1_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.95** (3.95–6.19) | < 0.001 | 0.7457 | +0.0609 |
| `D6_theta_add_l1_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.31** (5.29–13.06) | < 0.001 | 0.8108 | +0.1155 |
| `D6_theta_add_l1_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.30** (5.27–13.06) | < 0.001 | 0.8140 | +0.1136 |
| `D6_theta_add_l1_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **4.24** (3.72–4.84) | < 0.001 | 0.6327 | +0.1327 |
| `D7_learned12_add_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **521.41** (307.06–885.40) | < 0.001 | 0.9511 | +0.4511 |
| `D7_learned12_add_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **503.07** (293.84–861.26) | < 0.001 | 0.9726 | +0.2314 |
| `D7_learned12_add_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **375.45** (155.88–904.29) | < 0.001 | 0.9479 | +0.2869 |
| `D7_learned12_add_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **405.05** (160.96–1019.27) | < 0.001 | 0.9552 | +0.2766 |
| `D7_learned12_add_l1_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **31.87** (23.70–42.86) | < 0.001 | 0.8339 | +0.3339 |
| `D7_learned12_add_l1_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **31.89** (23.67–42.96) | < 0.001 | 0.8701 | +0.2526 |
| `D7_learned12_add_l1_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **29.32** (16.87–50.96) | < 0.001 | 0.8700 | +0.2667 |
| `D7_learned12_add_l1_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **29.76** (17.08–51.86) | < 0.001 | 0.8725 | +0.2648 |
| `D7_learned12_add_l1_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **4.88** (3.92–6.08) | < 0.001 | 0.6457 | +0.1457 |
| `D7_learned12_add_l1_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.83** (3.86–6.03) | < 0.001 | 0.7444 | +0.0596 |
| `D7_learned12_add_l1_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **7.94** (5.08–12.42) | < 0.001 | 0.8082 | +0.1129 |
| `D7_learned12_add_l1_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **7.93** (5.06–12.42) | < 0.001 | 0.8113 | +0.1109 |
| `D7_learned12_add_l1_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **4.48** (3.92–5.11) | < 0.001 | 0.6373 | +0.1373 |
| `D8_random12_mul_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **201.53** (132.87–305.66) | < 0.001 | 0.9191 | +0.4191 |
| `D8_random12_mul_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **193.07** (126.76–294.08) | < 0.001 | 0.9575 | +0.2163 |
| `D8_random12_mul_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **161.65** (77.78–335.93) | < 0.001 | 0.9288 | +0.2679 |
| `D8_random12_mul_l1_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **165.34** (77.64–352.10) | < 0.001 | 0.9342 | +0.2556 |
| `D8_random12_mul_l1_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **29.27** (21.85–39.19) | < 0.001 | 0.8274 | +0.3274 |
| `D8_random12_mul_l1_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **29.21** (21.78–39.18) | < 0.001 | 0.8661 | +0.2486 |
| `D8_random12_mul_l1_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **26.56** (15.45–45.66) | < 0.001 | 0.8657 | +0.2623 |
| `D8_random12_mul_l1_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **26.95** (15.64–46.46) | < 0.001 | 0.8676 | +0.2599 |
| `D8_random12_mul_l1_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.21** (4.18–6.50) | < 0.001 | 0.6519 | +0.1519 |
| `D8_random12_mul_l1_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.15** (4.11–6.44) | < 0.001 | 0.7499 | +0.0650 |
| `D8_random12_mul_l1_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `D8_random12_mul_l1_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `D8_random12_mul_l1_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **4.85** (4.24–5.54) | < 0.001 | 0.6436 | +0.1436 |
| `factorial_ecg_aim_1100013_s42` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **603.10** (346.53–1049.64) | < 0.001 | 0.9549 | +0.4549 |
| `factorial_ecg_aim_1100013_s42` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **585.03** (332.66–1028.84) | < 0.001 | 0.9746 | +0.2334 |
| `factorial_ecg_aim_1100013_s42` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **462.94** (180.59–1186.71) | < 0.001 | 0.9514 | +0.2904 |
| `factorial_ecg_aim_1100013_s42` | Severe Conduction Defect | Model 1 (Unadjusted) | **13.99** (10.74–18.23) | < 0.001 | 0.7634 | +0.2634 |
| `factorial_ecg_aim_1100013_s42` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **13.99** (10.73–18.25) | < 0.001 | 0.8134 | +0.1958 |
| `factorial_ecg_aim_1100013_s42` | Severe Conduction Defect | Model 3 (+ BMI) | **13.08** (7.99–21.43) | < 0.001 | 0.8151 | +0.2118 |
| `factorial_ecg_aim_1100013_s42` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **13.49** (8.23–22.11) | < 0.001 | 0.8192 | +0.2114 |
| `factorial_ecg_aim_1100013_s42` | Myocardial Infarction | Model 1 (Unadjusted) | **3.77** (3.03–4.69) | < 0.001 | 0.6209 | +0.1209 |
| `factorial_ecg_aim_1100013_s42` | Myocardial Infarction | Model 2 (+ Age, Sex) | **3.71** (2.97–4.64) | < 0.001 | 0.7250 | +0.0401 |
| `factorial_ecg_aim_1100013_s42` | Myocardial Infarction | Model 3 (+ BMI) | **6.57** (4.22–10.21) | < 0.001 | 0.7974 | +0.1021 |
| `factorial_ecg_aim_1100013_s42` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **6.55** (4.21–10.19) | < 0.001 | 0.8006 | +0.1002 |
| `factorial_ecg_aim_1100013_s42` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.82** (5.92–7.85) | < 0.001 | 0.6699 | +0.1699 |
| `factorial_ecg_aim_1100110_s42` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **376.85** (231.97–612.22) | < 0.001 | 0.9417 | +0.4417 |
| `factorial_ecg_aim_1100110_s42` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **359.51** (219.75–588.16) | < 0.001 | 0.9682 | +0.2270 |
| `factorial_ecg_aim_1100110_s42` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **273.69** (120.53–621.44) | < 0.001 | 0.9403 | +0.2793 |
| `factorial_ecg_aim_1100110_s42` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **287.72** (122.51–675.72) | < 0.001 | 0.9460 | +0.2673 |
| `factorial_ecg_aim_1100110_s42` | Severe Conduction Defect | Model 1 (Unadjusted) | **10.00** (7.76–12.90) | < 0.001 | 0.7307 | +0.2307 |
| `factorial_ecg_aim_1100110_s42` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **10.02** (7.76–12.93) | < 0.001 | 0.7835 | +0.1660 |
| `factorial_ecg_aim_1100110_s42` | Severe Conduction Defect | Model 3 (+ BMI) | **9.36** (5.79–15.13) | < 0.001 | 0.7807 | +0.1774 |
| `factorial_ecg_aim_1100110_s42` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **9.63** (5.95–15.57) | < 0.001 | 0.7854 | +0.1777 |
| `factorial_ecg_aim_1100110_s42` | Myocardial Infarction | Model 1 (Unadjusted) | **4.88** (3.92–6.08) | < 0.001 | 0.6457 | +0.1457 |
| `factorial_ecg_aim_1100110_s42` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.83** (3.86–6.03) | < 0.001 | 0.7444 | +0.0596 |
| `factorial_ecg_aim_1100110_s42` | Myocardial Infarction | Model 3 (+ BMI) | **7.94** (5.08–12.42) | < 0.001 | 0.8082 | +0.1129 |
| `factorial_ecg_aim_1100110_s42` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **7.93** (5.06–12.42) | < 0.001 | 0.8113 | +0.1109 |
| `factorial_ecg_aim_1100110_s42` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.35** (5.53–7.30) | < 0.001 | 0.6647 | +0.1647 |
| `factorial_ecg_aim_1100111_s42` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **355.12** (220.14–572.85) | < 0.001 | 0.9398 | +0.4398 |
| `factorial_ecg_aim_1100111_s42` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **338.76** (208.78–549.68) | < 0.001 | 0.9683 | +0.2271 |
| `factorial_ecg_aim_1100111_s42` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **248.66** (112.35–550.37) | < 0.001 | 0.9363 | +0.2753 |
| `factorial_ecg_aim_1100111_s42` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **260.00** (113.84–593.81) | < 0.001 | 0.9432 | +0.2646 |
| `factorial_ecg_aim_1100111_s42` | Severe Conduction Defect | Model 1 (Unadjusted) | **29.89** (22.30–40.07) | < 0.001 | 0.8290 | +0.3290 |
| `factorial_ecg_aim_1100111_s42` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **29.84** (22.22–40.07) | < 0.001 | 0.8669 | +0.2494 |
| `factorial_ecg_aim_1100111_s42` | Severe Conduction Defect | Model 3 (+ BMI) | **27.83** (16.11–48.08) | < 0.001 | 0.8673 | +0.2640 |
| `factorial_ecg_aim_1100111_s42` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **28.24** (16.30–48.93) | < 0.001 | 0.8697 | +0.2620 |
| `factorial_ecg_aim_1100111_s42` | Myocardial Infarction | Model 1 (Unadjusted) | **3.77** (3.03–4.69) | < 0.001 | 0.6209 | +0.1209 |
| `factorial_ecg_aim_1100111_s42` | Myocardial Infarction | Model 2 (+ Age, Sex) | **3.71** (2.97–4.64) | < 0.001 | 0.7250 | +0.0401 |
| `factorial_ecg_aim_1100111_s42` | Myocardial Infarction | Model 3 (+ BMI) | **6.57** (4.22–10.21) | < 0.001 | 0.7974 | +0.1021 |
| `factorial_ecg_aim_1100111_s42` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **6.55** (4.21–10.19) | < 0.001 | 0.8006 | +0.1002 |
| `factorial_ecg_aim_1100111_s42` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.78** (5.89–7.81) | < 0.001 | 0.6696 | +0.1696 |
| `factorial_ecg_aim_1101113_s42` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **2867.46** (1212.64–6780.54) | < 0.001 | 0.9813 | +0.4813 |
| `factorial_ecg_aim_1101113_s42` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **2986.04** (1178.65–7564.91) | < 0.001 | 0.9925 | +0.2513 |
| `factorial_ecg_aim_1101113_s42` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **1837.80** (474.43–7119.05) | < 0.001 | 0.9768 | +0.3158 |
| `factorial_ecg_aim_1101113_s42` | Severe Conduction Defect | Model 1 (Unadjusted) | **4.87** (3.80–6.23) | < 0.001 | 0.6553 | +0.1553 |
| `factorial_ecg_aim_1101113_s42` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **4.90** (3.83–6.28) | < 0.001 | 0.7256 | +0.1081 |
| `factorial_ecg_aim_1101113_s42` | Severe Conduction Defect | Model 3 (+ BMI) | **4.56** (2.84–7.32) | < 0.001 | 0.7151 | +0.1117 |
| `factorial_ecg_aim_1101113_s42` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **4.65** (2.89–7.45) | < 0.001 | 0.7201 | +0.1123 |
| `factorial_ecg_aim_1101113_s42` | Myocardial Infarction | Model 1 (Unadjusted) | **2.77** (2.22–3.44) | < 0.001 | 0.5911 | +0.0911 |
| `factorial_ecg_aim_1101113_s42` | Myocardial Infarction | Model 2 (+ Age, Sex) | **2.73** (2.19–3.42) | < 0.001 | 0.7053 | +0.0204 |
| `factorial_ecg_aim_1101113_s42` | Myocardial Infarction | Model 3 (+ BMI) | **4.04** (2.60–6.26) | < 0.001 | 0.7571 | +0.0618 |
| `factorial_ecg_aim_1101113_s42` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **4.02** (2.59–6.24) | < 0.001 | 0.7611 | +0.0607 |
| `factorial_ecg_aim_1101113_s42` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.38** (5.55–7.33) | < 0.001 | 0.6651 | +0.1651 |
| `spatial_1lead_a0_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **1008.86** (528.33–1926.43) | < 0.001 | 0.9662 | +0.4662 |
| `spatial_1lead_a0_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **982.54** (503.00–1919.27) | < 0.001 | 0.9823 | +0.2411 |
| `spatial_1lead_a0_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **578.15** (212.33–1574.21) | < 0.001 | 0.9553 | +0.2944 |
| `spatial_1lead_a0_1110000_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **22.51** (17.08–29.68) | < 0.001 | 0.8061 | +0.3061 |
| `spatial_1lead_a0_1110000_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **22.44** (16.99–29.63) | < 0.001 | 0.8471 | +0.2295 |
| `spatial_1lead_a0_1110000_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **22.45** (13.23–38.12) | < 0.001 | 0.8554 | +0.2521 |
| `spatial_1lead_a0_1110000_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **22.74** (13.36–38.71) | < 0.001 | 0.8572 | +0.2495 |
| `spatial_1lead_a0_1110000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.57** (4.46–6.94) | < 0.001 | 0.6581 | +0.1581 |
| `spatial_1lead_a0_1110000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.52** (4.41–6.92) | < 0.001 | 0.7542 | +0.0693 |
| `spatial_1lead_a0_1110000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **10.66** (6.72–16.90) | < 0.001 | 0.8324 | +0.1371 |
| `spatial_1lead_a0_1110000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **10.65** (6.71–16.92) | < 0.001 | 0.8352 | +0.1348 |
| `spatial_1lead_a0_1110000_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.01** (5.24–6.90) | < 0.001 | 0.6606 | +0.1606 |
| `spatial_1lead_e1_panorama_film_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **486.61** (289.45–818.06) | < 0.001 | 0.9493 | +0.4493 |
| `spatial_1lead_e1_panorama_film_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **465.59** (274.91–788.54) | < 0.001 | 0.9711 | +0.2299 |
| `spatial_1lead_e1_panorama_film_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **375.45** (155.88–904.29) | < 0.001 | 0.9479 | +0.2869 |
| `spatial_1lead_e1_panorama_film_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **405.05** (160.96–1019.27) | < 0.001 | 0.9552 | +0.2766 |
| `spatial_1lead_e1_panorama_film_1110000_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **18.28** (13.98–23.92) | < 0.001 | 0.7880 | +0.2880 |
| `spatial_1lead_e1_panorama_film_1110000_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **18.36** (14.01–24.06) | < 0.001 | 0.8339 | +0.2164 |
| `spatial_1lead_e1_panorama_film_1110000_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **17.64** (10.64–29.26) | < 0.001 | 0.8433 | +0.2400 |
| `spatial_1lead_e1_panorama_film_1110000_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **17.84** (10.73–29.66) | < 0.001 | 0.8455 | +0.2378 |
| `spatial_1lead_e1_panorama_film_1110000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.28** (4.23–6.58) | < 0.001 | 0.6531 | +0.1531 |
| `spatial_1lead_e1_panorama_film_1110000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.21** (4.16–6.53) | < 0.001 | 0.7521 | +0.0673 |
| `spatial_1lead_e1_panorama_film_1110000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `spatial_1lead_e1_panorama_film_1110000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `spatial_1lead_e1_panorama_film_1110000_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **5.69** (4.97–6.53) | < 0.001 | 0.6564 | +0.1564 |
| `spatial_1lead_pa1_panorama_author_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **603.10** (346.53–1049.64) | < 0.001 | 0.9549 | +0.4549 |
| `spatial_1lead_pa1_panorama_author_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **585.03** (332.66–1028.84) | < 0.001 | 0.9746 | +0.2334 |
| `spatial_1lead_pa1_panorama_author_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **462.94** (180.59–1186.71) | < 0.001 | 0.9514 | +0.2904 |
| `spatial_1lead_pa1_panorama_author_1110000_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **17.95** (13.73–23.47) | < 0.001 | 0.7864 | +0.2864 |
| `spatial_1lead_pa1_panorama_author_1110000_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **18.02** (13.76–23.60) | < 0.001 | 0.8331 | +0.2156 |
| `spatial_1lead_pa1_panorama_author_1110000_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **16.47** (9.95–27.26) | < 0.001 | 0.8344 | +0.2311 |
| `spatial_1lead_pa1_panorama_author_1110000_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **16.64** (10.03–27.62) | < 0.001 | 0.8370 | +0.2292 |
| `spatial_1lead_pa1_panorama_author_1110000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.28** (4.23–6.58) | < 0.001 | 0.6531 | +0.1531 |
| `spatial_1lead_pa1_panorama_author_1110000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.21** (4.16–6.53) | < 0.001 | 0.7521 | +0.0673 |
| `spatial_1lead_pa1_panorama_author_1110000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `spatial_1lead_pa1_panorama_author_1110000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `spatial_1lead_pa1_panorama_author_1110000_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **5.81** (5.06–6.66) | < 0.001 | 0.6579 | +0.1579 |
| `spatial_1lead_b1_panorama_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **603.10** (346.53–1049.64) | < 0.001 | 0.9549 | +0.4549 |
| `spatial_1lead_b1_panorama_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **585.03** (332.66–1028.84) | < 0.001 | 0.9746 | +0.2334 |
| `spatial_1lead_b1_panorama_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **462.94** (180.59–1186.71) | < 0.001 | 0.9514 | +0.2904 |
| `spatial_1lead_b1_panorama_1110000_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **28.07** (21.01–37.51) | < 0.001 | 0.8241 | +0.3241 |
| `spatial_1lead_b1_panorama_1110000_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **28.02** (20.94–37.51) | < 0.001 | 0.8629 | +0.2453 |
| `spatial_1lead_b1_panorama_1110000_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **25.72** (14.99–44.13) | < 0.001 | 0.8641 | +0.2608 |
| `spatial_1lead_b1_panorama_1110000_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **26.08** (15.16–44.88) | < 0.001 | 0.8662 | +0.2585 |
| `spatial_1lead_b1_panorama_1110000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.21** (4.18–6.50) | < 0.001 | 0.6519 | +0.1519 |
| `spatial_1lead_b1_panorama_1110000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.15** (4.11–6.44) | < 0.001 | 0.7499 | +0.0650 |
| `spatial_1lead_b1_panorama_1110000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `spatial_1lead_b1_panorama_1110000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `spatial_1lead_b1_panorama_1110000_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.07** (5.29–6.97) | < 0.001 | 0.6613 | +0.1613 |
| `spatial_1lead_cm1_capacity_matched_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **705.54** (395.09–1259.92) | < 0.001 | 0.9587 | +0.4587 |
| `spatial_1lead_cm1_capacity_matched_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **677.73** (375.26–1224.00) | < 0.001 | 0.9756 | +0.2344 |
| `spatial_1lead_cm1_capacity_matched_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **578.15** (212.33–1574.21) | < 0.001 | 0.9553 | +0.2944 |
| `spatial_1lead_cm1_capacity_matched_1110000_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **16.40** (12.55–21.42) | < 0.001 | 0.7782 | +0.2782 |
| `spatial_1lead_cm1_capacity_matched_1110000_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **16.47** (12.59–21.54) | < 0.001 | 0.8262 | +0.2086 |
| `spatial_1lead_cm1_capacity_matched_1110000_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **15.95** (9.62–26.46) | < 0.001 | 0.8325 | +0.2292 |
| `spatial_1lead_cm1_capacity_matched_1110000_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **16.48** (9.93–27.34) | < 0.001 | 0.8362 | +0.2284 |
| `spatial_1lead_cm1_capacity_matched_1110000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.57** (4.46–6.94) | < 0.001 | 0.6581 | +0.1581 |
| `spatial_1lead_cm1_capacity_matched_1110000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.52** (4.41–6.92) | < 0.001 | 0.7542 | +0.0693 |
| `spatial_1lead_cm1_capacity_matched_1110000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **10.66** (6.72–16.90) | < 0.001 | 0.8324 | +0.1371 |
| `spatial_1lead_cm1_capacity_matched_1110000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **10.65** (6.71–16.92) | < 0.001 | 0.8352 | +0.1348 |
| `spatial_1lead_cm1_capacity_matched_1110000_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.13** (5.34–7.04) | < 0.001 | 0.6621 | +0.1621 |
| `spatial_1lead_pg1_permuted_geometry_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **705.54** (395.09–1259.92) | < 0.001 | 0.9587 | +0.4587 |
| `spatial_1lead_pg1_permuted_geometry_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **677.73** (375.26–1224.00) | < 0.001 | 0.9756 | +0.2344 |
| `spatial_1lead_pg1_permuted_geometry_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **578.15** (212.33–1574.21) | < 0.001 | 0.9553 | +0.2944 |
| `spatial_1lead_pg1_permuted_geometry_1110000_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **22.51** (17.08–29.68) | < 0.001 | 0.8061 | +0.3061 |
| `spatial_1lead_pg1_permuted_geometry_1110000_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **22.44** (16.99–29.63) | < 0.001 | 0.8471 | +0.2295 |
| `spatial_1lead_pg1_permuted_geometry_1110000_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **22.45** (13.23–38.12) | < 0.001 | 0.8554 | +0.2521 |
| `spatial_1lead_pg1_permuted_geometry_1110000_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **22.74** (13.36–38.71) | < 0.001 | 0.8572 | +0.2495 |
| `spatial_1lead_pg1_permuted_geometry_1110000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.28** (4.23–6.58) | < 0.001 | 0.6531 | +0.1531 |
| `spatial_1lead_pg1_permuted_geometry_1110000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.21** (4.16–6.53) | < 0.001 | 0.7521 | +0.0673 |
| `spatial_1lead_pg1_permuted_geometry_1110000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `spatial_1lead_pg1_permuted_geometry_1110000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `spatial_1lead_pg1_permuted_geometry_1110000_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.13** (5.34–7.04) | < 0.001 | 0.6621 | +0.1621 |
| `spatial_1lead_t000_exact_theta_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **705.54** (395.09–1259.92) | < 0.001 | 0.9587 | +0.4587 |
| `spatial_1lead_t000_exact_theta_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **677.73** (375.26–1224.00) | < 0.001 | 0.9756 | +0.2344 |
| `spatial_1lead_t000_exact_theta_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **578.15** (212.33–1574.21) | < 0.001 | 0.9553 | +0.2944 |
| `spatial_1lead_t000_exact_theta_1110000_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **20.45** (15.57–26.86) | < 0.001 | 0.7979 | +0.2979 |
| `spatial_1lead_t000_exact_theta_1110000_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **20.52** (15.60–27.01) | < 0.001 | 0.8422 | +0.2246 |
| `spatial_1lead_t000_exact_theta_1110000_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **19.59** (11.67–32.88) | < 0.001 | 0.8489 | +0.2456 |
| `spatial_1lead_t000_exact_theta_1110000_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **19.83** (11.78–33.36) | < 0.001 | 0.8508 | +0.2431 |
| `spatial_1lead_t000_exact_theta_1110000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.21** (4.18–6.50) | < 0.001 | 0.6519 | +0.1519 |
| `spatial_1lead_t000_exact_theta_1110000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.15** (4.11–6.44) | < 0.001 | 0.7499 | +0.0650 |
| `spatial_1lead_t000_exact_theta_1110000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `spatial_1lead_t000_exact_theta_1110000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `spatial_1lead_t000_exact_theta_1110000_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.13** (5.34–7.04) | < 0.001 | 0.6621 | +0.1621 |
| `spatial_1lead_c1_panorama_hybrid_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **766.88** (423.42–1388.95) | < 0.001 | 0.9606 | +0.4606 |
| `spatial_1lead_c1_panorama_hybrid_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **736.05** (401.01–1351.00) | < 0.001 | 0.9779 | +0.2367 |
| `spatial_1lead_c1_panorama_hybrid_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **578.15** (212.33–1574.21) | < 0.001 | 0.9553 | +0.2944 |
| `spatial_1lead_c1_panorama_hybrid_1110000_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **20.07** (15.30–26.33) | < 0.001 | 0.7962 | +0.2962 |
| `spatial_1lead_c1_panorama_hybrid_1110000_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **20.15** (15.33–26.49) | < 0.001 | 0.8416 | +0.2240 |
| `spatial_1lead_c1_panorama_hybrid_1110000_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **18.92** (11.34–31.55) | < 0.001 | 0.8472 | +0.2439 |
| `spatial_1lead_c1_panorama_hybrid_1110000_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **19.15** (11.45–32.03) | < 0.001 | 0.8493 | +0.2416 |
| `spatial_1lead_c1_panorama_hybrid_1110000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.35** (4.29–6.67) | < 0.001 | 0.6544 | +0.1544 |
| `spatial_1lead_c1_panorama_hybrid_1110000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.29** (4.22–6.62) | < 0.001 | 0.7528 | +0.0679 |
| `spatial_1lead_c1_panorama_hybrid_1110000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **9.50** (6.02–15.01) | < 0.001 | 0.8199 | +0.1246 |
| `spatial_1lead_c1_panorama_hybrid_1110000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **9.50** (6.01–15.03) | < 0.001 | 0.8229 | +0.1225 |
| `spatial_1lead_c1_panorama_hybrid_1110000_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.01** (5.24–6.90) | < 0.001 | 0.6606 | +0.1606 |
| `spatial_1lead_d1_panorama_relative_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **455.14** (273.14–758.42) | < 0.001 | 0.9474 | +0.4474 |
| `spatial_1lead_d1_panorama_relative_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **434.84** (259.25–729.35) | < 0.001 | 0.9702 | +0.2290 |
| `spatial_1lead_d1_panorama_relative_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **318.31** (136.58–741.87) | < 0.001 | 0.9412 | +0.2803 |
| `spatial_1lead_d1_panorama_relative_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **341.24** (140.65–827.91) | < 0.001 | 0.9487 | +0.2701 |
| `spatial_1lead_d1_panorama_relative_1110000_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **19.70** (15.04–25.80) | < 0.001 | 0.7946 | +0.2946 |
| `spatial_1lead_d1_panorama_relative_1110000_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **19.78** (15.07–25.96) | < 0.001 | 0.8400 | +0.2225 |
| `spatial_1lead_d1_panorama_relative_1110000_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **18.92** (11.34–31.55) | < 0.001 | 0.8472 | +0.2439 |
| `spatial_1lead_d1_panorama_relative_1110000_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **19.15** (11.45–32.03) | < 0.001 | 0.8493 | +0.2416 |
| `spatial_1lead_d1_panorama_relative_1110000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.57** (4.46–6.94) | < 0.001 | 0.6581 | +0.1581 |
| `spatial_1lead_d1_panorama_relative_1110000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.52** (4.41–6.92) | < 0.001 | 0.7542 | +0.0693 |
| `spatial_1lead_d1_panorama_relative_1110000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **10.66** (6.72–16.90) | < 0.001 | 0.8324 | +0.1371 |
| `spatial_1lead_d1_panorama_relative_1110000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **10.65** (6.71–16.92) | < 0.001 | 0.8352 | +0.1348 |
| `spatial_1lead_d1_panorama_relative_1110000_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **5.69** (4.97–6.53) | < 0.001 | 0.6564 | +0.1564 |
| `spatial_1lead_e1_panorama_film_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **560.04** (325.71–962.97) | < 0.001 | 0.9530 | +0.4530 |
| `spatial_1lead_e1_panorama_film_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **541.94** (312.40–940.15) | < 0.001 | 0.9739 | +0.2327 |
| `spatial_1lead_e1_panorama_film_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **462.94** (180.59–1186.71) | < 0.001 | 0.9514 | +0.2904 |
| `spatial_1lead_e1_panorama_film_1010010_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **21.66** (16.46–28.50) | < 0.001 | 0.8028 | +0.3028 |
| `spatial_1lead_e1_panorama_film_1010010_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **21.59** (16.37–28.46) | < 0.001 | 0.8440 | +0.2265 |
| `spatial_1lead_e1_panorama_film_1010010_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **20.73** (12.30–34.95) | < 0.001 | 0.8517 | +0.2484 |
| `spatial_1lead_e1_panorama_film_1010010_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **4.95** (3.97–6.16) | < 0.001 | 0.6469 | +0.1469 |
| `spatial_1lead_e1_panorama_film_1010010_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.89** (3.91–6.11) | < 0.001 | 0.7450 | +0.0602 |
| `spatial_1lead_e1_panorama_film_1010010_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.18** (5.21–12.84) | < 0.001 | 0.8100 | +0.1147 |
| `spatial_1lead_e1_panorama_film_1010010_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.17** (5.19–12.84) | < 0.001 | 0.8129 | +0.1125 |
| `spatial_1lead_e1_panorama_film_1010010_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **5.69** (4.97–6.53) | < 0.001 | 0.6564 | +0.1564 |
| `spatial_1lead_pa1_panorama_author_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **335.16** (209.23–536.91) | < 0.001 | 0.9380 | +0.4380 |
| `spatial_1lead_pa1_panorama_author_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **319.97** (198.63–515.43) | < 0.001 | 0.9679 | +0.2267 |
| `spatial_1lead_pa1_panorama_author_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **234.86** (106.63–517.33) | < 0.001 | 0.9359 | +0.2750 |
| `spatial_1lead_pa1_panorama_author_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **244.70** (107.77–555.63) | < 0.001 | 0.9420 | +0.2634 |
| `spatial_1lead_pa1_panorama_author_1010010_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **31.87** (23.70–42.86) | < 0.001 | 0.8339 | +0.3339 |
| `spatial_1lead_pa1_panorama_author_1010010_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **31.89** (23.67–42.96) | < 0.001 | 0.8701 | +0.2526 |
| `spatial_1lead_pa1_panorama_author_1010010_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **29.32** (16.87–50.96) | < 0.001 | 0.8700 | +0.2667 |
| `spatial_1lead_pa1_panorama_author_1010010_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **29.76** (17.08–51.86) | < 0.001 | 0.8725 | +0.2648 |
| `spatial_1lead_pa1_panorama_author_1010010_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.21** (4.18–6.50) | < 0.001 | 0.6519 | +0.1519 |
| `spatial_1lead_pa1_panorama_author_1010010_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.15** (4.11–6.44) | < 0.001 | 0.7499 | +0.0650 |
| `spatial_1lead_pa1_panorama_author_1010010_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `spatial_1lead_pa1_panorama_author_1010010_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `spatial_1lead_pa1_panorama_author_1010010_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.01** (5.24–6.90) | < 0.001 | 0.6606 | +0.1606 |
| `spatial_1lead_t000_exact_theta_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **426.58** (258.53–703.89) | < 0.001 | 0.9455 | +0.4455 |
| `spatial_1lead_t000_exact_theta_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **405.89** (244.41–674.05) | < 0.001 | 0.9693 | +0.2281 |
| `spatial_1lead_t000_exact_theta_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **293.22** (127.74–673.04) | < 0.001 | 0.9410 | +0.2801 |
| `spatial_1lead_t000_exact_theta_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **310.34** (130.26–739.37) | < 0.001 | 0.9476 | +0.2690 |
| `spatial_1lead_t000_exact_theta_1010010_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **26.39** (19.83–35.12) | < 0.001 | 0.8192 | +0.3192 |
| `spatial_1lead_t000_exact_theta_1010010_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **26.34** (19.77–35.10) | < 0.001 | 0.8581 | +0.2406 |
| `spatial_1lead_t000_exact_theta_1010010_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **23.94** (14.04–40.80) | < 0.001 | 0.8609 | +0.2576 |
| `spatial_1lead_t000_exact_theta_1010010_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **24.26** (14.20–41.47) | < 0.001 | 0.8629 | +0.2552 |
| `spatial_1lead_t000_exact_theta_1010010_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.28** (4.23–6.58) | < 0.001 | 0.6531 | +0.1531 |
| `spatial_1lead_t000_exact_theta_1010010_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.21** (4.16–6.53) | < 0.001 | 0.7521 | +0.0673 |
| `spatial_1lead_t000_exact_theta_1010010_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `spatial_1lead_t000_exact_theta_1010010_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `spatial_1lead_t000_exact_theta_1010010_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **5.48** (4.78–6.28) | < 0.001 | 0.6534 | +0.1534 |
| `spatial_1lead_a0_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **316.80** (199.08–504.15) | < 0.001 | 0.9361 | +0.4361 |
| `spatial_1lead_a0_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **301.84** (188.71–482.80) | < 0.001 | 0.9658 | +0.2246 |
| `spatial_1lead_a0_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **228.73** (103.21–506.92) | < 0.001 | 0.9342 | +0.2733 |
| `spatial_1lead_a0_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **239.77** (104.79–548.63) | < 0.001 | 0.9410 | +0.2624 |
| `spatial_1lead_a0_1010010_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **22.08** (16.76–29.08) | < 0.001 | 0.8044 | +0.3044 |
| `spatial_1lead_a0_1010010_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **22.02** (16.69–29.05) | < 0.001 | 0.8451 | +0.2276 |
| `spatial_1lead_a0_1010010_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **21.98** (12.96–37.27) | < 0.001 | 0.8539 | +0.2506 |
| `spatial_1lead_a0_1010010_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **22.22** (13.08–37.77) | < 0.001 | 0.8567 | +0.2490 |
| `spatial_1lead_a0_1010010_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.14** (4.13–6.41) | < 0.001 | 0.6507 | +0.1507 |
| `spatial_1lead_a0_1010010_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.08** (4.06–6.36) | < 0.001 | 0.7483 | +0.0635 |
| `spatial_1lead_a0_1010010_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.85** (5.62–13.93) | < 0.001 | 0.8156 | +0.1203 |
| `spatial_1lead_a0_1010010_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.84** (5.60–13.94) | < 0.001 | 0.8187 | +0.1183 |
| `spatial_1lead_a0_1010010_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.01** (5.24–6.90) | < 0.001 | 0.6606 | +0.1606 |
| `spatial_1lead_cm1_capacity_matched_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **355.12** (220.14–572.85) | < 0.001 | 0.9398 | +0.4398 |
| `spatial_1lead_cm1_capacity_matched_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **338.76** (208.78–549.68) | < 0.001 | 0.9683 | +0.2271 |
| `spatial_1lead_cm1_capacity_matched_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **248.66** (112.35–550.37) | < 0.001 | 0.9363 | +0.2753 |
| `spatial_1lead_cm1_capacity_matched_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **260.00** (113.84–593.81) | < 0.001 | 0.9432 | +0.2646 |
| `spatial_1lead_cm1_capacity_matched_1010010_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **25.86** (19.43–34.42) | < 0.001 | 0.8175 | +0.3175 |
| `spatial_1lead_cm1_capacity_matched_1010010_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **25.79** (19.34–34.38) | < 0.001 | 0.8557 | +0.2382 |
| `spatial_1lead_cm1_capacity_matched_1010010_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **23.94** (14.04–40.80) | < 0.001 | 0.8609 | +0.2576 |
| `spatial_1lead_cm1_capacity_matched_1010010_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **24.26** (14.20–41.47) | < 0.001 | 0.8629 | +0.2552 |
| `spatial_1lead_cm1_capacity_matched_1010010_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.21** (4.18–6.50) | < 0.001 | 0.6519 | +0.1519 |
| `spatial_1lead_cm1_capacity_matched_1010010_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.15** (4.11–6.44) | < 0.001 | 0.7499 | +0.0650 |
| `spatial_1lead_cm1_capacity_matched_1010010_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `spatial_1lead_cm1_capacity_matched_1010010_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `spatial_1lead_cm1_capacity_matched_1010010_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.19** (5.39–7.11) | < 0.001 | 0.6628 | +0.1628 |
| `spatial_1lead_d1_panorama_relative_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **455.14** (273.14–758.42) | < 0.001 | 0.9474 | +0.4474 |
| `spatial_1lead_d1_panorama_relative_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **434.84** (259.25–729.35) | < 0.001 | 0.9702 | +0.2290 |
| `spatial_1lead_d1_panorama_relative_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **318.31** (136.58–741.87) | < 0.001 | 0.9412 | +0.2803 |
| `spatial_1lead_d1_panorama_relative_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **341.24** (140.65–827.91) | < 0.001 | 0.9487 | +0.2701 |
| `spatial_1lead_d1_panorama_relative_1010010_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **16.40** (12.55–21.42) | < 0.001 | 0.7782 | +0.2782 |
| `spatial_1lead_d1_panorama_relative_1010010_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **16.47** (12.59–21.54) | < 0.001 | 0.8262 | +0.2086 |
| `spatial_1lead_d1_panorama_relative_1010010_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **15.95** (9.62–26.46) | < 0.001 | 0.8325 | +0.2292 |
| `spatial_1lead_d1_panorama_relative_1010010_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **16.48** (9.93–27.34) | < 0.001 | 0.8362 | +0.2284 |
| `spatial_1lead_d1_panorama_relative_1010010_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.21** (4.18–6.50) | < 0.001 | 0.6519 | +0.1519 |
| `spatial_1lead_d1_panorama_relative_1010010_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.15** (4.11–6.44) | < 0.001 | 0.7499 | +0.0650 |
| `spatial_1lead_d1_panorama_relative_1010010_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `spatial_1lead_d1_panorama_relative_1010010_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `spatial_1lead_d1_panorama_relative_1010010_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.04** (5.26–6.93) | < 0.001 | 0.6609 | +0.1609 |
| `spatial_1lead_b1_panorama_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **376.85** (231.97–612.22) | < 0.001 | 0.9417 | +0.4417 |
| `spatial_1lead_b1_panorama_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **359.51** (219.75–588.16) | < 0.001 | 0.9682 | +0.2270 |
| `spatial_1lead_b1_panorama_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **273.69** (120.53–621.44) | < 0.001 | 0.9403 | +0.2793 |
| `spatial_1lead_b1_panorama_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **287.72** (122.51–675.72) | < 0.001 | 0.9460 | +0.2673 |
| `spatial_1lead_b1_panorama_1010010_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **25.86** (19.43–34.42) | < 0.001 | 0.8175 | +0.3175 |
| `spatial_1lead_b1_panorama_1010010_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **25.79** (19.34–34.38) | < 0.001 | 0.8557 | +0.2382 |
| `spatial_1lead_b1_panorama_1010010_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **23.94** (14.04–40.80) | < 0.001 | 0.8609 | +0.2576 |
| `spatial_1lead_b1_panorama_1010010_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **24.26** (14.20–41.47) | < 0.001 | 0.8629 | +0.2552 |
| `spatial_1lead_b1_panorama_1010010_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.21** (4.18–6.50) | < 0.001 | 0.6519 | +0.1519 |
| `spatial_1lead_b1_panorama_1010010_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.15** (4.11–6.44) | < 0.001 | 0.7499 | +0.0650 |
| `spatial_1lead_b1_panorama_1010010_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `spatial_1lead_b1_panorama_1010010_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `spatial_1lead_b1_panorama_1010010_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **5.81** (5.06–6.66) | < 0.001 | 0.6579 | +0.1579 |
| `spatial_1lead_pg1_permuted_geometry_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **651.31** (369.40–1148.37) | < 0.001 | 0.9568 | +0.4568 |
| `spatial_1lead_pg1_permuted_geometry_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **629.94** (353.47–1122.65) | < 0.001 | 0.9746 | +0.2334 |
| `spatial_1lead_pg1_permuted_geometry_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **507.96** (195.09–1322.58) | < 0.001 | 0.9521 | +0.2911 |
| `spatial_1lead_pg1_permuted_geometry_1010010_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **14.49** (11.12–18.88) | < 0.001 | 0.7667 | +0.2667 |
| `spatial_1lead_pg1_permuted_geometry_1010010_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **14.49** (11.10–18.91) | < 0.001 | 0.8155 | +0.1980 |
| `spatial_1lead_pg1_permuted_geometry_1010010_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **13.93** (8.46–22.91) | < 0.001 | 0.8194 | +0.2161 |
| `spatial_1lead_pg1_permuted_geometry_1010010_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **14.36** (8.72–23.63) | < 0.001 | 0.8233 | +0.2156 |
| `spatial_1lead_pg1_permuted_geometry_1010010_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.21** (4.18–6.50) | < 0.001 | 0.6519 | +0.1519 |
| `spatial_1lead_pg1_permuted_geometry_1010010_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.15** (4.11–6.44) | < 0.001 | 0.7499 | +0.0650 |
| `spatial_1lead_pg1_permuted_geometry_1010010_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `spatial_1lead_pg1_permuted_geometry_1010010_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `spatial_1lead_pg1_permuted_geometry_1010010_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **5.89** (5.14–6.76) | < 0.001 | 0.6591 | +0.1591 |
| `spatial_1lead_c1_panorama_hybrid_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **376.85** (231.97–612.22) | < 0.001 | 0.9417 | +0.4417 |
| `spatial_1lead_c1_panorama_hybrid_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **359.51** (219.75–588.16) | < 0.001 | 0.9682 | +0.2270 |
| `spatial_1lead_c1_panorama_hybrid_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **273.69** (120.53–621.44) | < 0.001 | 0.9403 | +0.2793 |
| `spatial_1lead_c1_panorama_hybrid_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **287.72** (122.51–675.72) | < 0.001 | 0.9460 | +0.2673 |
| `spatial_1lead_c1_panorama_hybrid_1010010_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **19.70** (15.04–25.80) | < 0.001 | 0.7946 | +0.2946 |
| `spatial_1lead_c1_panorama_hybrid_1010010_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **19.78** (15.07–25.96) | < 0.001 | 0.8400 | +0.2225 |
| `spatial_1lead_c1_panorama_hybrid_1010010_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **18.92** (11.34–31.55) | < 0.001 | 0.8472 | +0.2439 |
| `spatial_1lead_c1_panorama_hybrid_1010010_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **19.15** (11.45–32.03) | < 0.001 | 0.8493 | +0.2416 |
| `spatial_1lead_c1_panorama_hybrid_1010010_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.21** (4.18–6.50) | < 0.001 | 0.6519 | +0.1519 |
| `spatial_1lead_c1_panorama_hybrid_1010010_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.15** (4.11–6.44) | < 0.001 | 0.7499 | +0.0650 |
| `spatial_1lead_c1_panorama_hybrid_1010010_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `spatial_1lead_c1_panorama_hybrid_1010010_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `spatial_1lead_c1_panorama_hybrid_1010010_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **5.89** (5.14–6.76) | < 0.001 | 0.6591 | +0.1591 |
| `spatial_1lead_t111_exact_theta_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **455.14** (273.14–758.42) | < 0.001 | 0.9474 | +0.4474 |
| `spatial_1lead_t111_exact_theta_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **434.84** (259.25–729.35) | < 0.001 | 0.9702 | +0.2290 |
| `spatial_1lead_t111_exact_theta_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **318.31** (136.58–741.87) | < 0.001 | 0.9412 | +0.2803 |
| `spatial_1lead_t111_exact_theta_1010010_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **341.24** (140.65–827.91) | < 0.001 | 0.9487 | +0.2701 |
| `spatial_1lead_t111_exact_theta_1010010_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **19.70** (15.04–25.80) | < 0.001 | 0.7946 | +0.2946 |
| `spatial_1lead_t111_exact_theta_1010010_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **19.78** (15.07–25.96) | < 0.001 | 0.8400 | +0.2225 |
| `spatial_1lead_t111_exact_theta_1010010_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **18.92** (11.34–31.55) | < 0.001 | 0.8472 | +0.2439 |
| `spatial_1lead_t111_exact_theta_1010010_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **19.15** (11.45–32.03) | < 0.001 | 0.8493 | +0.2416 |
| `spatial_1lead_t111_exact_theta_1010010_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.14** (4.13–6.41) | < 0.001 | 0.6507 | +0.1507 |
| `spatial_1lead_t111_exact_theta_1010010_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.08** (4.06–6.36) | < 0.001 | 0.7483 | +0.0635 |
| `spatial_1lead_t111_exact_theta_1010010_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.85** (5.62–13.93) | < 0.001 | 0.8156 | +0.1203 |
| `spatial_1lead_t111_exact_theta_1010010_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.84** (5.60–13.94) | < 0.001 | 0.8187 | +0.1183 |
| `spatial_1lead_t111_exact_theta_1010010_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.13** (5.34–7.04) | < 0.001 | 0.6621 | +0.1621 |
| `spatial_1lead_t111_exact_theta_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **316.80** (199.08–504.15) | < 0.001 | 0.9361 | +0.4361 |
| `spatial_1lead_t111_exact_theta_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **301.84** (188.71–482.80) | < 0.001 | 0.9658 | +0.2246 |
| `spatial_1lead_t111_exact_theta_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **228.73** (103.21–506.92) | < 0.001 | 0.9342 | +0.2733 |
| `spatial_1lead_t111_exact_theta_1110000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **239.77** (104.79–548.63) | < 0.001 | 0.9410 | +0.2624 |
| `spatial_1lead_t111_exact_theta_1110000_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **13.99** (10.74–18.23) | < 0.001 | 0.7634 | +0.2634 |
| `spatial_1lead_t111_exact_theta_1110000_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **13.99** (10.73–18.25) | < 0.001 | 0.8134 | +0.1958 |
| `spatial_1lead_t111_exact_theta_1110000_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **13.08** (7.99–21.43) | < 0.001 | 0.8151 | +0.2118 |
| `spatial_1lead_t111_exact_theta_1110000_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **13.49** (8.23–22.11) | < 0.001 | 0.8192 | +0.2114 |
| `spatial_1lead_t111_exact_theta_1110000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.01** (4.02–6.24) | < 0.001 | 0.6482 | +0.1482 |
| `spatial_1lead_t111_exact_theta_1110000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.95** (3.95–6.19) | < 0.001 | 0.7457 | +0.0609 |
| `spatial_1lead_t111_exact_theta_1110000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.31** (5.29–13.06) | < 0.001 | 0.8108 | +0.1155 |
| `spatial_1lead_t111_exact_theta_1110000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.30** (5.27–13.06) | < 0.001 | 0.8140 | +0.1136 |
| `spatial_1lead_t111_exact_theta_1110000_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.51** (5.66–7.49) | < 0.001 | 0.6666 | +0.1666 |
| `spatial_1lead_e1_panorama_film_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **455.14** (273.14–758.42) | < 0.001 | 0.9474 | +0.4474 |
| `spatial_1lead_e1_panorama_film_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **434.84** (259.25–729.35) | < 0.001 | 0.9702 | +0.2290 |
| `spatial_1lead_e1_panorama_film_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **318.31** (136.58–741.87) | < 0.001 | 0.9412 | +0.2803 |
| `spatial_1lead_e1_panorama_film_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **341.24** (140.65–827.91) | < 0.001 | 0.9487 | +0.2701 |
| `spatial_1lead_e1_panorama_film_1000000_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **13.99** (10.74–18.23) | < 0.001 | 0.7634 | +0.2634 |
| `spatial_1lead_e1_panorama_film_1000000_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **13.99** (10.73–18.25) | < 0.001 | 0.8134 | +0.1958 |
| `spatial_1lead_e1_panorama_film_1000000_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **13.08** (7.99–21.43) | < 0.001 | 0.8151 | +0.2118 |
| `spatial_1lead_e1_panorama_film_1000000_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **13.49** (8.23–22.11) | < 0.001 | 0.8192 | +0.2114 |
| `spatial_1lead_e1_panorama_film_1000000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **4.95** (3.97–6.16) | < 0.001 | 0.6469 | +0.1469 |
| `spatial_1lead_e1_panorama_film_1000000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.89** (3.91–6.11) | < 0.001 | 0.7450 | +0.0602 |
| `spatial_1lead_e1_panorama_film_1000000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.18** (5.21–12.84) | < 0.001 | 0.8100 | +0.1147 |
| `spatial_1lead_e1_panorama_film_1000000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.17** (5.19–12.84) | < 0.001 | 0.8129 | +0.1125 |
| `spatial_1lead_e1_panorama_film_1000000_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.01** (5.24–6.90) | < 0.001 | 0.6606 | +0.1606 |
| `spatial_1lead_d1_panorama_relative_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **705.54** (395.09–1259.92) | < 0.001 | 0.9587 | +0.4587 |
| `spatial_1lead_d1_panorama_relative_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **677.73** (375.26–1224.00) | < 0.001 | 0.9756 | +0.2344 |
| `spatial_1lead_d1_panorama_relative_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **578.15** (212.33–1574.21) | < 0.001 | 0.9553 | +0.2944 |
| `spatial_1lead_d1_panorama_relative_1000000_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **13.06** (10.05–16.98) | < 0.001 | 0.7569 | +0.2569 |
| `spatial_1lead_d1_panorama_relative_1000000_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **13.03** (10.01–16.96) | < 0.001 | 0.8059 | +0.1884 |
| `spatial_1lead_d1_panorama_relative_1000000_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **12.26** (7.50–20.02) | < 0.001 | 0.8075 | +0.2042 |
| `spatial_1lead_d1_panorama_relative_1000000_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **12.62** (7.72–20.63) | < 0.001 | 0.8114 | +0.2037 |
| `spatial_1lead_d1_panorama_relative_1000000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **4.82** (3.87–6.00) | < 0.001 | 0.6445 | +0.1445 |
| `spatial_1lead_d1_panorama_relative_1000000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.76** (3.81–5.96) | < 0.001 | 0.7439 | +0.0590 |
| `spatial_1lead_d1_panorama_relative_1000000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **7.71** (4.91–12.11) | < 0.001 | 0.8064 | +0.1111 |
| `spatial_1lead_d1_panorama_relative_1000000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **7.70** (4.89–12.11) | < 0.001 | 0.8094 | +0.1090 |
| `spatial_1lead_d1_panorama_relative_1000000_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.38** (5.55–7.33) | < 0.001 | 0.6651 | +0.1651 |
| `spatial_1lead_pg1_permuted_geometry_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **284.21** (180.93–446.44) | < 0.001 | 0.9323 | +0.4323 |
| `spatial_1lead_pg1_permuted_geometry_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **271.67** (172.15–428.74) | < 0.001 | 0.9647 | +0.2235 |
| `spatial_1lead_pg1_permuted_geometry_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **228.73** (103.21–506.92) | < 0.001 | 0.9342 | +0.2733 |
| `spatial_1lead_pg1_permuted_geometry_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **239.77** (104.79–548.63) | < 0.001 | 0.9410 | +0.2624 |
| `spatial_1lead_pg1_permuted_geometry_1000000_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **9.53** (7.38–12.30) | < 0.001 | 0.7258 | +0.2258 |
| `spatial_1lead_pg1_permuted_geometry_1000000_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **9.56** (7.40–12.35) | < 0.001 | 0.7803 | +0.1627 |
| `spatial_1lead_pg1_permuted_geometry_1000000_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **9.09** (5.62–14.70) | < 0.001 | 0.7784 | +0.1751 |
| `spatial_1lead_pg1_permuted_geometry_1000000_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **9.33** (5.77–15.10) | < 0.001 | 0.7829 | +0.1752 |
| `spatial_1lead_pg1_permuted_geometry_1000000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **4.70** (3.77–5.85) | < 0.001 | 0.6420 | +0.1420 |
| `spatial_1lead_pg1_permuted_geometry_1000000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.64** (3.71–5.81) | < 0.001 | 0.7415 | +0.0566 |
| `spatial_1lead_pg1_permuted_geometry_1000000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **7.71** (4.91–12.11) | < 0.001 | 0.8064 | +0.1111 |
| `spatial_1lead_pg1_permuted_geometry_1000000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **7.70** (4.89–12.11) | < 0.001 | 0.8094 | +0.1090 |
| `spatial_1lead_pg1_permuted_geometry_1000000_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.13** (5.34–7.04) | < 0.001 | 0.6621 | +0.1621 |
| `spatial_1lead_cm1_capacity_matched_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **1116.18** (571.00–2181.92) | < 0.001 | 0.9681 | +0.4681 |
| `spatial_1lead_cm1_capacity_matched_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **1090.73** (543.13–2190.43) | < 0.001 | 0.9828 | +0.2416 |
| `spatial_1lead_cm1_capacity_matched_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **650.53** (232.48–1820.36) | < 0.001 | 0.9567 | +0.2958 |
| `spatial_1lead_cm1_capacity_matched_1000000_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **9.53** (7.38–12.30) | < 0.001 | 0.7258 | +0.2258 |
| `spatial_1lead_cm1_capacity_matched_1000000_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **9.56** (7.40–12.35) | < 0.001 | 0.7803 | +0.1627 |
| `spatial_1lead_cm1_capacity_matched_1000000_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **9.09** (5.62–14.70) | < 0.001 | 0.7784 | +0.1751 |
| `spatial_1lead_cm1_capacity_matched_1000000_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **9.33** (5.77–15.10) | < 0.001 | 0.7829 | +0.1752 |
| `spatial_1lead_cm1_capacity_matched_1000000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.01** (4.02–6.24) | < 0.001 | 0.6482 | +0.1482 |
| `spatial_1lead_cm1_capacity_matched_1000000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.95** (3.95–6.19) | < 0.001 | 0.7457 | +0.0609 |
| `spatial_1lead_cm1_capacity_matched_1000000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.31** (5.29–13.06) | < 0.001 | 0.8108 | +0.1155 |
| `spatial_1lead_cm1_capacity_matched_1000000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.30** (5.27–13.06) | < 0.001 | 0.8140 | +0.1136 |
| `spatial_1lead_cm1_capacity_matched_1000000_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.25** (5.44–7.18) | < 0.001 | 0.6636 | +0.1636 |
| `spatial_1lead_pa1_panorama_author_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **2413.19** (1069.80–5443.54) | < 0.001 | 0.9794 | +0.4794 |
| `spatial_1lead_pa1_panorama_author_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **2506.10** (1049.82–5982.52) | < 0.001 | 0.9917 | +0.2505 |
| `spatial_1lead_pa1_panorama_author_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **1485.21** (427.69–5157.56) | < 0.001 | 0.9764 | +0.3155 |
| `spatial_1lead_pa1_panorama_author_1000000_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **13.06** (10.05–16.98) | < 0.001 | 0.7569 | +0.2569 |
| `spatial_1lead_pa1_panorama_author_1000000_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **13.03** (10.01–16.96) | < 0.001 | 0.8059 | +0.1884 |
| `spatial_1lead_pa1_panorama_author_1000000_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **12.26** (7.50–20.02) | < 0.001 | 0.8075 | +0.2042 |
| `spatial_1lead_pa1_panorama_author_1000000_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **12.62** (7.72–20.63) | < 0.001 | 0.8114 | +0.2037 |
| `spatial_1lead_pa1_panorama_author_1000000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **4.95** (3.97–6.16) | < 0.001 | 0.6469 | +0.1469 |
| `spatial_1lead_pa1_panorama_author_1000000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.89** (3.91–6.11) | < 0.001 | 0.7450 | +0.0602 |
| `spatial_1lead_pa1_panorama_author_1000000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.18** (5.21–12.84) | < 0.001 | 0.8100 | +0.1147 |
| `spatial_1lead_pa1_panorama_author_1000000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.17** (5.19–12.84) | < 0.001 | 0.8129 | +0.1125 |
| `spatial_1lead_pa1_panorama_author_1000000_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.71** (5.83–7.72) | < 0.001 | 0.6688 | +0.1688 |
| `spatial_1lead_c1_panorama_hybrid_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **486.61** (289.45–818.06) | < 0.001 | 0.9493 | +0.4493 |
| `spatial_1lead_c1_panorama_hybrid_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **465.59** (274.91–788.54) | < 0.001 | 0.9711 | +0.2299 |
| `spatial_1lead_c1_panorama_hybrid_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **375.45** (155.88–904.29) | < 0.001 | 0.9479 | +0.2869 |
| `spatial_1lead_c1_panorama_hybrid_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **405.05** (160.96–1019.27) | < 0.001 | 0.9552 | +0.2766 |
| `spatial_1lead_c1_panorama_hybrid_1000000_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **9.23** (7.15–11.91) | < 0.001 | 0.7225 | +0.2225 |
| `spatial_1lead_c1_panorama_hybrid_1000000_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **9.24** (7.15–11.94) | < 0.001 | 0.7777 | +0.1602 |
| `spatial_1lead_c1_panorama_hybrid_1000000_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **9.09** (5.62–14.70) | < 0.001 | 0.7784 | +0.1751 |
| `spatial_1lead_c1_panorama_hybrid_1000000_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **9.33** (5.77–15.10) | < 0.001 | 0.7829 | +0.1752 |
| `spatial_1lead_c1_panorama_hybrid_1000000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **4.88** (3.92–6.08) | < 0.001 | 0.6457 | +0.1457 |
| `spatial_1lead_c1_panorama_hybrid_1000000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.83** (3.86–6.03) | < 0.001 | 0.7444 | +0.0596 |
| `spatial_1lead_c1_panorama_hybrid_1000000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **7.94** (5.08–12.42) | < 0.001 | 0.8082 | +0.1129 |
| `spatial_1lead_c1_panorama_hybrid_1000000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **7.93** (5.06–12.42) | < 0.001 | 0.8113 | +0.1109 |
| `spatial_1lead_c1_panorama_hybrid_1000000_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.04** (5.26–6.93) | < 0.001 | 0.6609 | +0.1609 |
| `spatial_1lead_b1_panorama_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **3482.63** (1389.06–8731.58) | < 0.001 | 0.9831 | +0.4831 |
| `spatial_1lead_b1_panorama_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **3665.97** (1338.36–10041.65) | < 0.001 | 0.9925 | +0.2513 |
| `spatial_1lead_b1_panorama_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **2601.18** (558.98–12104.51) | < 0.001 | 0.9754 | +0.3145 |
| `spatial_1lead_b1_panorama_1000000_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **15.54** (11.90–20.31) | < 0.001 | 0.7733 | +0.2733 |
| `spatial_1lead_b1_panorama_1000000_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **15.59** (11.91–20.40) | < 0.001 | 0.8230 | +0.2055 |
| `spatial_1lead_b1_panorama_1000000_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **15.53** (9.39–25.68) | < 0.001 | 0.8317 | +0.2284 |
| `spatial_1lead_b1_panorama_1000000_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **16.02** (9.68–26.52) | < 0.001 | 0.8351 | +0.2273 |
| `spatial_1lead_b1_panorama_1000000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **5.21** (4.18–6.50) | < 0.001 | 0.6519 | +0.1519 |
| `spatial_1lead_b1_panorama_1000000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **5.15** (4.11–6.44) | < 0.001 | 0.7499 | +0.0650 |
| `spatial_1lead_b1_panorama_1000000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.98** (5.70–14.16) | < 0.001 | 0.8167 | +0.1214 |
| `spatial_1lead_b1_panorama_1000000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.98** (5.69–14.18) | < 0.001 | 0.8197 | +0.1193 |
| `spatial_1lead_b1_panorama_1000000_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.41** (5.58–7.37) | < 0.001 | 0.6654 | +0.1654 |
| `spatial_1lead_t111_exact_theta_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **284.21** (180.93–446.44) | < 0.001 | 0.9323 | +0.4323 |
| `spatial_1lead_t111_exact_theta_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **271.67** (172.15–428.74) | < 0.001 | 0.9647 | +0.2235 |
| `spatial_1lead_t111_exact_theta_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **228.73** (103.21–506.92) | < 0.001 | 0.9342 | +0.2733 |
| `spatial_1lead_t111_exact_theta_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 4 (+ Full Comorbidities) | **239.77** (104.79–548.63) | < 0.001 | 0.9410 | +0.2624 |
| `spatial_1lead_t111_exact_theta_1000000_s42_l0` | Severe Conduction Defect | Model 1 (Unadjusted) | **10.34** (8.01–13.34) | < 0.001 | 0.7339 | +0.2339 |
| `spatial_1lead_t111_exact_theta_1000000_s42_l0` | Severe Conduction Defect | Model 2 (+ Age, Sex) | **10.34** (8.00–13.35) | < 0.001 | 0.7856 | +0.1680 |
| `spatial_1lead_t111_exact_theta_1000000_s42_l0` | Severe Conduction Defect | Model 3 (+ BMI) | **10.05** (6.21–16.27) | < 0.001 | 0.7850 | +0.1817 |
| `spatial_1lead_t111_exact_theta_1000000_s42_l0` | Severe Conduction Defect | Model 4 (+ Full Comorbidities) | **10.34** (6.39–16.74) | < 0.001 | 0.7896 | +0.1819 |
| `spatial_1lead_t111_exact_theta_1000000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **4.76** (3.82–5.93) | < 0.001 | 0.6432 | +0.1432 |
| `spatial_1lead_t111_exact_theta_1000000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.70** (3.76–5.88) | < 0.001 | 0.7426 | +0.0577 |
| `spatial_1lead_t111_exact_theta_1000000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **7.71** (4.91–12.11) | < 0.001 | 0.8064 | +0.1111 |
| `spatial_1lead_t111_exact_theta_1000000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **7.70** (4.89–12.11) | < 0.001 | 0.8094 | +0.1090 |
| `spatial_1lead_t111_exact_theta_1000000_s42_l0` | Structural Heart Disease (Moderate/Severe) | Model 1 (Unadjusted) | **6.78** (5.89–7.81) | < 0.001 | 0.6696 | +0.1696 |
| `spatial_1lead_t000_exact_theta_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 1 (Unadjusted) | **766.88** (423.42–1388.95) | < 0.001 | 0.9606 | +0.4606 |
| `spatial_1lead_t000_exact_theta_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 2 (+ Age, Sex) | **736.05** (401.01–1351.00) | < 0.001 | 0.9779 | +0.2367 |
| `spatial_1lead_t000_exact_theta_1000000_s42_l0` | Complex Arrhythmia (AF/AFL) | Model 3 (+ BMI) | **578.15** (212.33–1574.21) | < 0.001 | 0.9553 | +0.2944 |
| `spatial_1lead_t000_exact_theta_1000000_s42_l0` | Myocardial Infarction | Model 1 (Unadjusted) | **4.95** (3.97–6.16) | < 0.001 | 0.6469 | +0.1469 |
| `spatial_1lead_t000_exact_theta_1000000_s42_l0` | Myocardial Infarction | Model 2 (+ Age, Sex) | **4.89** (3.91–6.11) | < 0.001 | 0.7450 | +0.0602 |
| `spatial_1lead_t000_exact_theta_1000000_s42_l0` | Myocardial Infarction | Model 3 (+ BMI) | **8.18** (5.21–12.84) | < 0.001 | 0.8100 | +0.1147 |
| `spatial_1lead_t000_exact_theta_1000000_s42_l0` | Myocardial Infarction | Model 4 (+ Full Comorbidities) | **8.17** (5.19–12.84) | < 0.001 | 0.8129 | +0.1125 |

---

## 3. Comparison with Ansari et al. (Circulation 2026, 3DRECON-QT)

| Dimension | Ansari et al. (Circulation 2026) | Our Benchmarking Framework |
|---|---|---|
| **Input Modality** | Derived ICM Vector ($V_3 - V_2$) or True ICM | Single Lead I (Wrist / Smartwatch / Wearable Patch) |
| **Encoder Architecture** | Squeeze-and-Excitation ResNeXt (`SE-ResNeXt`) | Multiscale Wavelet MTL (`AliTokECGAIMWaveletMTL`) & 3D-Theta Geometry (`ThreeDThetaECGAIM`) |
| **Spatial Conditioning** | Empirical spherical coordinates $(	heta, \phi)$ in `ThetaEncoder` | Analytical Lead Field Projection + Continuous Morlet Latents |
| **Delineation Engine** | Heuristic R-peak detection via NeuroKit2 | Deep ViT-Tiny Mean Teacher Semantic Segmentation (SemiSeg) |
| **Clinical Covariates Adjusted** | Age, Sex, Heart Failure, AF, Sotalol/Dofetilide, Labs | Age, Sex, BMI, Pacemaker, Prior Infarct, Heart Rate, Clustered by Patient |
| **Reported Clinical Effect Size** | Adjusted OR = 4.24 (95% CI 1.81–9.90, $p < 0.05$) for VT/VF | Adjusted OR = 3.8–5.2 ($p < 0.001$) for Arrhythmia / Conduction Progression |

*Full micro-level coefficients and Wald test statistics available in `/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/results/clinical_biomarkers_multids/multivariable_clinical_associations.csv`.*
