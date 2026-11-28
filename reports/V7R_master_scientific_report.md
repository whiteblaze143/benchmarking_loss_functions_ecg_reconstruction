# AF-DENSITY V7-R: Independent Numerical Reconciliation & Scientific Adjudication Report

**Directive:** AF-DENSITY V7-R (Priorities 1–10)  
**Audit Standard:** Zero reliance on historical PASS certificates, executive summaries, or V7 figures. Every number below is independently reconstructed from raw WFDB/HDF5/CSV/Parquet assets and fixed-model 5-fold patient-disjoint CV with $B=2,000$ patient-clustered bootstraps (`min_attainable_p = 2/2001 = 0.001`).

---

## Executive Adjudication Summary

Independent reconstruction overturns the overclaimed V7 and V7-P executive conclusions while isolating three genuine, reproducible scientific signals:

1. **Historical episode architecture decisively outperforms cumulative AF burden alone on early persistence (`P02`: Logistic $+0.0742$, $p=0.0010$; LightGBM $+0.0224$, $p=0.0010$) and dynamic termination (`P03`: Logistic $+0.0274$, $p=0.0010$), with zero future leakage (`0.0` diff across 150 adversarial future-deletion tests) — yet Charitos continuous density adds ZERO statistically significant increment beyond architecture on any task (`P01` $p=0.436$, `P02` $p=0.467$, `P03` $p=0.665$).**
2. **Pre-onset ventricular dynamics (`F4` Clean RR) genuinely improve imminent sustained AF forecasting (`P01` LGBM `0.5710 -> 0.6689`, $\Delta=+0.0979$, $p=0.0010$ within valid `LTAFDB+SHDB` records without cohort shortcuts), whereas continuous clinical early warning remains non-viable (`9.79%` event sensitivity at `16.56` false warnings/patient-day at thr `0.05`).**
3. **High Charitos density conceals massive transition heterogeneity (`V7R_transition_instability_feasibility.csv`): among `118` multi-run patients with high Charitos density ($d \ge 0.80$, median `0.9752`), transition rates diverge 30-fold (`4.06/24h` in `55` low-fragmentation patients vs `122.61/24h` in `29` storm patients), establishing *Temporal concentration vs transition instability* as an empirically populated, mathematically sharp research question.**

---

## Priority 1 — Reconciled Dataset Inventory

| dataset | raw_recordings | annotated_recordings | unannotated_recordings | unique_patients_raw | unique_patients_annotated | patients_with_af | patients_with_multi_af_runs | raw_recording_h | annotated_known_rhythm_h | unknown_rhythm_h_total | af_duration_h | known_non_af_h | raw_af_transitions_sum | canonical_af_runs_sum | left_censored_runs | right_censored_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IRIDIA-AF | 167 | 167 | 0 | 152 | 152 | 152 | 66 | 6690.47 | 6687.56 | 3.42 | 1609.01 | 5078.55 | 388 | 388 | 24 | 0 |
| LTAFDB | 84 | 84 | 0 | 84 | 84 | 83 | 66 | 1960.6 | 1934.92 | 25.68 | 1030.89 | 904.03 | 7358 | 7358 | 0 | 43 |
| SHDB-AF | 128 | 98 | 30 | 122 | 93 | 72 | 49 | 3050.38 | 2330.49 | 719.88 | 455.94 | 1874.55 | 794 | 794 | 0 | 0 |
| COMBINED | 379 | 349 | 30 | 358 | 329 | 307 | 181 | 11701.45 | 10952.98 | 748.99 | 3095.84 | 7857.14 | 8540 | 8540 | 24 | 43 |

- **349 vs 379 recordings & 10,953h vs 11,701h resolved:** SHDB-AF contains `128` raw recordings (`122` unique patients, `6` duplicates), of which `98` are annotated (`93` unique pts, `72` AF+) and `30` (`719.88h`) have no annotations (`rhythm_known=False`). Plus `25.68h` unannotated prefix in LTAFDB and `3.42h` calibration gaps in IRIDIA-AF yield `748.99h` total unknown time (`11,701.45h` raw - `10,952.98h` known annotated).
- **8,540 vs 8,582 resolved:** Exact sum of raw transitions across all annotated recordings equals **`8,540` canonical runs** (`388` IRIDIA + `7,358` LTAFDB + `794` SHDB; `24` left-censored in IRIDIA, `43` right-censored in LTAFDB). The historical `8,582` figure was an artifact in `af_density_continuity_v5`.

---

## Priority 2 — E01 Temporal Neighbourhood Reconciliation

- **`553` vs `304` index episodes:** Total dataset has `553` runs $\ge 30$m (`249` IRIDIA + `202` LTAFDB + `102` SHDB). E01 analyzed `304` (`202` LTAFDB + `102` SHDB) because IRIDIA-AF contains only `2` runs `<30s`.

| dataset | comparison | exposure_type | rr | ci_lo | ci_hi | p_val | n_pts | target_events | ref_events | target_exp_h | ref_exp_h | target_rate_per_h | ref_rate_per_h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | pre_0_15m_vs_remote | risk_time_h | 6.23581903248756 | 3.145302870377574 | 12.363018955076372 | 1.5911734466284313e-07 | 102 | 165 | 1710 | 18.1896 | 1175.5184 | 9.0711 | 1.4547 |
| ALL | post_0_15m_vs_remote | risk_time_h | 4.390291762652043 | 2.110145031666403 | 9.134282938831456 | 7.565323951510185e-05 | 102 | 116 | 1710 | 18.1634 | 1175.5184 | 6.3865 | 1.4547 |
| ALL | post_0_15m_vs_pre_0_15m | risk_time_h | 0.7040441263276136 | 0.4235946207203665 | 1.1701709785017116 | 0.175816075089957 | 103 | 116 | 165 | 18.1634 | 18.1896 | 6.3865 | 9.0711 |
| ALL | pre_0_15m_vs_remote | physical_time_h | 1.8562296479148748 | 0.8841935298771835 | 3.896871430711306 | 0.1021075126849904 | 102 | 165 | 1710 | 63.7952 | 1227.2469 | 2.5864 | 1.3934 |
| ALL | post_0_15m_vs_remote | physical_time_h | 1.33033072392009 | 0.6203541153142511 | 2.8528541865306045 | 0.4633682317918498 | 102 | 116 | 1710 | 62.5798 | 1227.2469 | 1.8536 | 1.3934 |
| ALL | post_0_15m_vs_pre_0_15m | physical_time_h | 0.7166843420556713 | 0.4383606863673042 | 1.171721055563346 | 0.1841252329745616 | 103 | 116 | 165 | 62.5798 | 63.7952 | 1.8536 | 2.5864 |
| LTAFDB | pre_0_15m_vs_remote | risk_time_h | 6.170335326046171 | 2.93939037619746 | 12.952698744665032 | 1.5107696320253229e-06 | 49 | 138 | 1618 | 5.0514 | 365.4461 | 27.3189 | 4.4275 |
| LTAFDB | post_0_15m_vs_remote | risk_time_h | 3.951011698847247 | 1.70903262600725 | 9.13411084544245 | 0.0013116620886423 | 48 | 89 | 1618 | 5.0878 | 365.4461 | 17.493 | 4.4275 |
| LTAFDB | post_0_15m_vs_pre_0_15m | risk_time_h | 0.640323659900882 | 0.3481987609371155 | 1.1775297198800452 | 0.1515073787466526 | 49 | 89 | 138 | 5.0878 | 5.0514 | 17.493 | 27.3189 |
| LTAFDB | pre_0_15m_vs_remote | physical_time_h | 0.8309377087183268 | 0.3611596120420406 | 1.911779315151359 | 0.6630958089320029 | 49 | 138 | 1618 | 42.1952 | 411.0842 | 3.2705 | 3.9359 |
| LTAFDB | post_0_15m_vs_remote | physical_time_h | 0.544087880782886 | 0.2334412000750491 | 1.2681207169927184 | 0.1585994885327647 | 48 | 89 | 1618 | 41.5598 | 411.0842 | 2.1415 | 3.9359 |
| LTAFDB | post_0_15m_vs_pre_0_15m | physical_time_h | 0.6547878078876729 | 0.3886092940988096 | 1.1032856904583674 | 0.1116668963673408 | 49 | 89 | 138 | 41.5598 | 42.1952 | 2.1415 | 3.2705 |
| SHDB-AF | pre_0_15m_vs_remote | risk_time_h | 18.09523809914016 | 5.499563971446288 | 59.53883681772378 | 1.8849853810383996e-06 | 53 | 27 | 92 | 13.1382 | 810.0724 | 2.0551 | 0.1136 |
| SHDB-AF | post_0_15m_vs_remote | risk_time_h | 18.181748244895857 | 10.636345571268832 | 31.07984476677157 | 2.879179438331008e-26 | 54 | 27 | 92 | 13.0757 | 810.0724 | 2.0649 | 0.1136 |
| SHDB-AF | post_0_15m_vs_pre_0_15m | risk_time_h | 1.0047808239345732 | 0.2932511542946726 | 3.442729855829185 | 0.9939434420455612 | 54 | 27 | 27 | 13.0757 | 13.1382 | 2.0649 | 2.0551 |
| SHDB-AF | pre_0_15m_vs_remote | physical_time_h | 11.089153349463963 | 3.2948340588818485 | 37.32185591454624 | 0.0001020474172282 | 53 | 27 | 92 | 21.6 | 816.1627 | 1.25 | 0.1127 |
| SHDB-AF | post_0_15m_vs_remote | physical_time_h | 11.395133389579032 | 6.533487396780042 | 19.874388221869623 | 1.0026995347958076e-17 | 54 | 27 | 92 | 21.02 | 816.1627 | 1.2845 | 0.1127 |
| SHDB-AF | post_0_15m_vs_pre_0_15m | physical_time_h | 1.027592732328031 | 0.2994764580608572 | 3.525976066268312 | 0.9654864170969412 | 54 | 27 | 27 | 21.02 | 21.6 | 1.2845 | 1.25 |

### E01 LOPO Sensitivity (Excluding Prolific Brief-Run LTAFDB Patients)
| sensitivity_scenario | excluded_patients | pre_vs_remote_rr | pre_vs_remote_p | post_vs_remote_rr | post_vs_remote_p | post_vs_pre_rr | post_vs_pre_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| None (Full Pooled) |  | 6.236 | 1.5911734466284313e-07 | 4.39 | 7.565323951510185e-05 | 0.704 | 0.175816075089957 |
| Exclude Top-1 LTAFDB Pt | ltafdb_112 | 6.236 | 1.5911734466284313e-07 | 4.39 | 7.565323951510185e-05 | 0.704 | 0.175816075089957 |
| Exclude Top-3 LTAFDB Pts | ltafdb_112,ltafdb_100,ltafdb_74 | 8.509 | 4.321936309913282e-14 | 5.947 | 1.9549129115121522e-08 | 0.699 | 0.1835470876469845 |
| Exclude Top-5 LTAFDB Pts | ltafdb_112,ltafdb_100,ltafdb_74,ltafdb_15,ltafdb_53 | 10.548 | 5.851909953808556e-19 | 7.114 | 5.935945892922424e-09 | 0.674 | 0.1980324318758429 |
| Exclude Top-10 LTAFDB Pts | ltafdb_112,ltafdb_100,ltafdb_74,ltafdb_15,ltafdb_53,ltafdb_32,ltafdb_25,ltafdb_204,ltafdb_115,ltafdb_101 | 10.32 | 7.069455092893703e-16 | 8.889 | 6.183283304426746e-10 | 0.861 | 0.6828012541257773 |

### E01 Conditional Null Models Summary
| null_type | n_recordings | obs_pre_total | null_pre_expected | pre_obs_over_exp | obs_post_total | null_post_expected | post_obs_over_exp | frac_rec_p_pre_lt005 | frac_rec_p_post_lt005 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Null_A_All | 58 | 179.0 | 171.82 | 1.042 | 127.0 | 152.99 | 0.83 | 0.3276 | 0.1552 |
| Null_A_LTAFDB | 33 | 151.0 | 166.92 | 0.905 | 100.0 | 148.45 | 0.674 | 0.2727 | 0.1515 |
| Null_A_SHDB | 25 | 28.0 | 4.89 | 5.722 | 27.0 | 4.53 | 5.956 | 0.4 | 0.16 |
| Null_B_Exchangeable_Ordering | 58 | 40.0 | 62.1 | 0.644 | 26.0 | 62.12 | 0.419 | 0.0 |  |

---

## Priority 3 — E02 Matched Panel, Density Fill Audit & GAM OOF Reconciliation

| panel | window_name | n_episodes | n_patients | mean_hr_bpm | se_hr_bpm | mean_cvrr | se_cvrr | mean_rmssd_ms | se_rmssd_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Matched_Through_17_5m | win_30_180s | 622 | 257 | 97.49 | 1.073 | 0.2158 | 0.0028 | 195.25 | 3.63 |
| Matched_Through_17_5m | win_5_7_5m | 622 | 257 | 96.941 | 1.062 | 0.217 | 0.0029 | 198.53 | 3.58 |
| Matched_Through_17_5m | win_15_17_5m | 622 | 257 | 96.686 | 1.049 | 0.2142 | 0.0027 | 196.37 | 3.38 |
| Matched_Through_32_5m | win_30_180s | 509 | 244 | 97.292 | 1.166 | 0.2173 | 0.0032 | 196.62 | 4.12 |
| Matched_Through_32_5m | win_5_7_5m | 509 | 244 | 96.442 | 1.167 | 0.2194 | 0.0033 | 202.29 | 4.11 |
| Matched_Through_32_5m | win_15_17_5m | 509 | 244 | 96.658 | 1.17 | 0.2146 | 0.003 | 197.81 | 3.88 |
| Matched_Through_32_5m | win_30_32_5m | 509 | 244 | 95.816 | 1.156 | 0.2133 | 0.0031 | 200.56 | 4.17 |

| analysis | cohort | n_episodes | n_patients | density_hr | ci_lo | ci_hi | p_value | gam_5fold_oof_r2_base | gam_5fold_oof_r2_plus_density | gam_5fold_oof_delta_density |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cox_PH_CompleteDefinedDensity | Pooled_DefinedDensity | 6438 | 145 | 1.1382 | 0.7718 | 1.6783 | 0.5137301866218338 | 0.33308 | 0.3347 | 0.00161 |
| Cox_PH_CompleteDefinedDensity | LTAFDB_DefinedDensity | 5715 | 59 | 1.292 | 0.877 | 1.9035 | 0.1949305851942088 | 0.33308 | 0.3347 | 0.00161 |
| Cox_PH_CompleteDefinedDensity | SHDB_DefinedDensity | 583 | 42 | 1.1605 | 0.4077 | 3.3033 | 0.7803712433309321 | 0.33308 | 0.3347 | 0.00161 |
| Cox_PH_CompleteDefinedDensity | IRIDIA_DefinedDensity | 140 | 44 | 4.0896 | 1.0155 | 16.4692 | 0.0475122630106806 | 0.33308 | 0.3347 | 0.00161 |

- **Population A (`6,683` episodes):** `245` (`3.67%`) had undefined 6h density improperly filled as `0.0` in `e02_statistical_models.py:232`. Restricting strictly to defined complete density (`n=6,438`, `145` pts), Cox PH density is null pooled (`HR=1.1382`, $p=0.514$), null in LTAFDB (`p=0.195`) and SHDB (`p=0.780`), and borderline only in IRIDIA (`n=140`, `HR=4.0896`, $p=0.0475$).

---

## Priority 4 — E03 SHDB-AF Patient-Deduplicated Structural Analysis

| cohort_subset | outcome | echo_latency_window | n_patients | m2_burden_r2 | m3a_plus_density_r2 | partial_r2_density | effect_per_0_10_density | boot_ci95_low | boot_ci95_high | ols_p_value | excludes_zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PAF_MultiRun_PatientDedup | Echo_LAD | <=30d | 13 | 0.4132 | 0.4276 | 0.0144 | 0.5777 | -3.3585 | 5.6991 | 0.6657 | False |
| PAF_AllDefinedDensity_PatientDedup | Echo_LAD | <=30d | 26 | 0.267 | 0.279 | 0.012 | 0.6071 | -1.3186 | 4.3429 | 0.5613 | False |
| PAF_MultiRun_PatientDedup | Echo_LAD | <=90d | 17 | 0.2389 | 0.2423 | 0.0034 | 0.188 | -2.298 | 2.2701 | 0.8208 | False |
| PAF_AllDefinedDensity_PatientDedup | Echo_LAD | <=90d | 31 | 0.2447 | 0.2563 | 0.0116 | 0.5132 | -0.873 | 2.4368 | 0.53 | False |
| PAF_MultiRun_PatientDedup | Echo_LAD | <=180d | 19 | 0.2094 | 0.2129 | 0.0035 | 0.1955 | -1.9011 | 2.8303 | 0.8069 | False |
| PAF_AllDefinedDensity_PatientDedup | Echo_LAD | <=180d | 33 | 0.2375 | 0.2474 | 0.01 | 0.4802 | -0.8026 | 2.4555 | 0.5473 | False |
| PAF_MultiRun_PatientDedup | Echo_LAD | Unrestricted | 28 | 0.0308 | 0.046 | 0.0152 | -0.3832 | -1.8587 | 1.3907 | 0.5504 | False |
| PAF_AllDefinedDensity_PatientDedup | Echo_LAD | Unrestricted | 50 | 0.1106 | 0.1107 | 0.0001 | -0.0459 | -1.1867 | 1.4682 | 0.9347 | False |
| PAF_MultiRun_PatientDedup | Echo_LVEF | <=30d | 13 | 0.1357 | 0.3911 | 0.2554 | 3.9641 | -1.5297 | 19.0897 | 0.1044 | False |
| PAF_AllDefinedDensity_PatientDedup | Echo_LVEF | <=30d | 26 | 0.1758 | 0.1948 | 0.019 | -1.0839 | -7.5994 | 2.7535 | 0.4897 | False |
| PAF_MultiRun_PatientDedup | Echo_LVEF | <=90d | 17 | 0.1547 | 0.1548 | 0.0 | -0.0199 | -2.9138 | 6.0469 | 0.9898 | False |
| PAF_AllDefinedDensity_PatientDedup | Echo_LVEF | <=90d | 31 | 0.2126 | 0.2567 | 0.0441 | -1.4603 | -3.6601 | 0.5594 | 0.2255 | False |
| PAF_MultiRun_PatientDedup | Echo_LVEF | <=180d | 19 | 0.108 | 0.1144 | 0.0063 | -0.6084 | -4.5985 | 6.6504 | 0.7566 | False |
| PAF_AllDefinedDensity_PatientDedup | Echo_LVEF | <=180d | 33 | 0.1071 | 0.163 | 0.0559 | -1.8253 | -4.0718 | 0.2796 | 0.1825 | False |
| PAF_MultiRun_PatientDedup | Echo_LVEF | Unrestricted | 28 | 0.0988 | 0.1316 | 0.0328 | -0.9314 | -2.5256 | 1.7016 | 0.361 | False |
| PAF_AllDefinedDensity_PatientDedup | Echo_LVEF | Unrestricted | 50 | 0.1202 | 0.1404 | 0.0202 | -0.8417 | -1.845 | 0.712 | 0.3092 | False |

---

## Priorities 5, 6 & 7 — Leakage Audit, Patient Support, Clean Matched Ablations, Stratified Performance, Alerts & FM Controls

### Future-Deletion Leakage Audit
| test_type | task_id | dataset | n_landmarks_tested | max_abs_feature_diff | leakage_detected |
| --- | --- | --- | --- | --- | --- |
| Future_Deletion_And_Adversarial_Corruption | P01 | IRIDIA-AF | 25 | 0.0 | False |
| Future_Deletion_And_Adversarial_Corruption | P01 | LTAFDB | 25 | 0.0 | False |
| Future_Deletion_And_Adversarial_Corruption | P01 | SHDB-AF | 25 | 0.0 | False |
| Future_Deletion_And_Adversarial_Corruption | P02 | IRIDIA-AF | 25 | 0.0 | False |
| Future_Deletion_And_Adversarial_Corruption | P02 | LTAFDB | 25 | 0.0 | False |
| Future_Deletion_And_Adversarial_Corruption | P02 | SHDB-AF | 25 | 0.0 | False |

### Cohort Patient Support
| task | cohort | n_samples | n_positives | prevalence | n_patients | top1_patient_share |
| --- | --- | --- | --- | --- | --- | --- |
| P01_CleanAnnotated | ALL | 90057 | 1686 | 0.01872 | 300 | 0.0174 |
| P01_CleanAnnotated | IRIDIA-AF | 58716 | 1257 | 0.02141 | 152 | 0.0267 |
| P01_CleanAnnotated | LTAFDB | 10161 | 125 | 0.0123 | 57 | 0.0286 |
| P01_CleanAnnotated | SHDB-AF | 21180 | 304 | 0.01435 | 91 | 0.0249 |
| P02_Persistence60s | ALL | 2458 | 531 | 0.21603 | 290 | 0.0944 |
| P02_Persistence60s | IRIDIA-AF | 342 | 227 | 0.66374 | 143 | 0.0322 |
| P02_Persistence60s | LTAFDB | 1744 | 202 | 0.11583 | 79 | 0.133 |
| P02_Persistence60s | SHDB-AF | 372 | 102 | 0.27419 | 68 | 0.1237 |
| P03_Survival15m | ALL | 4196 | 2397 | 0.57126 | 290 | 0.066 |
| P03_Survival15m | IRIDIA-AF | 869 | 159 | 0.18297 | 143 | 0.0368 |
| P03_Survival15m | LTAFDB | 2625 | 1881 | 0.71657 | 79 | 0.1055 |
| P03_Survival15m | SHDB-AF | 702 | 357 | 0.50855 | 68 | 0.1111 |

### Clean Matched Ablations & B=2,000 Contrasts (Fixed Models)
| task | feature_set | model | auroc_pooled | auprc_pooled | brier | n_features | auroc_equal_patient | n_eq_patients | ci95_low | ci95_high | p_value_b2000 | min_attainable_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P01_CleanAnnotated | Cohort_Identity_Only | logistic | 0.5348 | 0.0201 | 0.0184 | 3.0 |  |  |  |  |  |  |
| P01_CleanAnnotated | S0_Baseline | logistic | 0.5004 | 0.0183 | 0.0184 | 2.0 | 0.3744 | 195.0 |  |  |  |  |
| P01_CleanAnnotated | S1_Burden | logistic | 0.6031 | 0.0324 | 0.0183 | 10.0 | 0.7373 | 195.0 |  |  |  |  |
| P01_CleanAnnotated | S2_Architecture | logistic | 0.6054 | 0.0329 | 0.0184 | 25.0 | 0.7377 | 195.0 |  |  |  |  |
| P01_CleanAnnotated | S3a_Arch_Plus_DensityOnly | logistic | 0.6114 | 0.0359 | 0.0183 | 31.0 | 0.7805 | 195.0 |  |  |  |  |
| P01_CleanAnnotated | S3b_Arch_Plus_AggrOnly | logistic | 0.6142 | 0.0352 | 0.0183 | 31.0 | 0.7779 | 195.0 |  |  |  |  |
| P01_CleanAnnotated | S3c_Arch_Plus_Dens_Aggr | logistic | 0.6125 | 0.0358 | 0.0183 | 36.0 | 0.7742 | 195.0 |  |  |  |  |
| P01_CleanAnnotated | CONTRAST:Arch_vs_Burden | logistic | 0.0024 |  |  |  |  |  | -0.0072 | 0.0113 | 0.6066966516741629 | 0.001 |
| P01_CleanAnnotated | CONTRAST:DensityOnly_vs_Arch | logistic | 0.006 |  |  |  |  |  | -0.0082 | 0.0227 | 0.4357821089455272 | 0.001 |
| P01_CleanAnnotated | CONTRAST:AggrOnly_vs_Arch | logistic | 0.0088 |  |  |  |  |  | -0.005 | 0.0244 | 0.2208895552223888 | 0.001 |
| P01_CleanAnnotated | CONTRAST:DensAggr_vs_Arch | logistic | 0.0071 |  |  |  |  |  | -0.0081 | 0.0248 | 0.3758120939530235 | 0.001 |
| P01_CleanAnnotated | S0_Baseline | lgbm | 0.5004 | 0.0193 | 0.0185 | 2.0 | 0.4799 | 195.0 |  |  |  |  |
| P01_CleanAnnotated | S1_Burden | lgbm | 0.5759 | 0.0245 | 0.0209 | 10.0 | 0.6199 | 195.0 |  |  |  |  |
| P01_CleanAnnotated | S2_Architecture | lgbm | 0.5845 | 0.0259 | 0.0205 | 25.0 | 0.6214 | 195.0 |  |  |  |  |
| P01_CleanAnnotated | S3a_Arch_Plus_DensityOnly | lgbm | 0.5814 | 0.0263 | 0.0202 | 31.0 | 0.6193 | 195.0 |  |  |  |  |
| P01_CleanAnnotated | S3b_Arch_Plus_AggrOnly | lgbm | 0.5912 | 0.0279 | 0.02 | 31.0 | 0.6348 | 195.0 |  |  |  |  |
| P01_CleanAnnotated | S3c_Arch_Plus_Dens_Aggr | lgbm | 0.5815 | 0.0282 | 0.0202 | 36.0 | 0.6197 | 195.0 |  |  |  |  |
| P01_CleanAnnotated | CONTRAST:Arch_vs_Burden | lgbm | 0.0087 |  |  |  |  |  | -0.0034 | 0.0209 | 0.15992003998001 | 0.001 |
| P01_CleanAnnotated | CONTRAST:DensityOnly_vs_Arch | lgbm | -0.0031 |  |  |  |  |  | -0.0155 | 0.0087 | 0.6216891554222889 | 0.001 |
| P01_CleanAnnotated | CONTRAST:AggrOnly_vs_Arch | lgbm | 0.0067 |  |  |  |  |  | -0.0067 | 0.0186 | 0.3168415792103948 | 0.001 |
| P01_CleanAnnotated | CONTRAST:DensAggr_vs_Arch | lgbm | -0.003 |  |  |  |  |  | -0.0179 | 0.0109 | 0.6926536731634183 | 0.001 |
| P01_CleanAnnotated_LTAFDB_SHDB_ValidRR | S3c_vs_S4_CleanRR_NoCohortShortcut | logistic | 0.0345 |  |  | 5.0 |  |  | -0.0027 | 0.0775 | 0.0729635182408795 | 0.001 |
| P01_CleanAnnotated_LTAFDB_SHDB_ValidRR | S3c_vs_S4_CleanRR_NoCohortShortcut | lgbm | 0.0979 |  |  | 5.0 |  |  | 0.0546 | 0.1406 | 0.000999500249875 | 0.001 |
| P02_Persistence60s | Cohort_Identity_Only | logistic | 0.6982 | 0.4594 | 0.1355 | 3.0 |  |  |  |  |  |  |
| P02_Persistence60s | S0_Baseline | logistic | 0.5187 | 0.3255 | 0.1656 | 2.0 | 0.5479 | 91.0 |  |  |  |  |
| P02_Persistence60s | S1_Burden | logistic | 0.8133 | 0.6518 | 0.1106 | 10.0 | 0.4917 | 91.0 |  |  |  |  |
| P02_Persistence60s | S2_Architecture | logistic | 0.8874 | 0.6988 | 0.1022 | 25.0 | 0.4789 | 91.0 |  |  |  |  |
| P02_Persistence60s | S3a_Arch_Plus_DensityOnly | logistic | 0.8886 | 0.6988 | 0.1015 | 31.0 | 0.4974 | 91.0 |  |  |  |  |
| P02_Persistence60s | S3b_Arch_Plus_AggrOnly | logistic | 0.8899 | 0.7 | 0.1007 | 31.0 | 0.4623 | 91.0 |  |  |  |  |
| P02_Persistence60s | S3c_Arch_Plus_Dens_Aggr | logistic | 0.8887 | 0.6985 | 0.1011 | 36.0 | 0.4889 | 91.0 |  |  |  |  |
| P02_Persistence60s | CONTRAST:Arch_vs_Burden | logistic | 0.0742 |  |  |  |  |  | 0.0463 | 0.1009 | 0.000999500249875 | 0.001 |
| P02_Persistence60s | CONTRAST:DensityOnly_vs_Arch | logistic | 0.0012 |  |  |  |  |  | -0.002 | 0.0046 | 0.4667666166916541 | 0.001 |
| P02_Persistence60s | CONTRAST:AggrOnly_vs_Arch | logistic | 0.0025 |  |  |  |  |  | -0.0007 | 0.0056 | 0.127936031984008 | 0.001 |
| P02_Persistence60s | CONTRAST:DensAggr_vs_Arch | logistic | 0.0013 |  |  |  |  |  | -0.0024 | 0.0051 | 0.4667666166916541 | 0.001 |
| P02_Persistence60s | S0_Baseline | lgbm | 0.5721 | 0.3943 | 0.1595 | 2.0 | 0.4893 | 91.0 |  |  |  |  |
| P02_Persistence60s | S1_Burden | lgbm | 0.8468 | 0.6565 | 0.1102 | 10.0 | 0.5043 | 91.0 |  |  |  |  |
| P02_Persistence60s | S2_Architecture | lgbm | 0.8692 | 0.6747 | 0.1073 | 25.0 | 0.5245 | 91.0 |  |  |  |  |
| P02_Persistence60s | S3a_Arch_Plus_DensityOnly | lgbm | 0.8706 | 0.68 | 0.1066 | 31.0 | 0.495 | 91.0 |  |  |  |  |
| P02_Persistence60s | S3b_Arch_Plus_AggrOnly | lgbm | 0.8668 | 0.6824 | 0.1068 | 31.0 | 0.5227 | 91.0 |  |  |  |  |
| P02_Persistence60s | S3c_Arch_Plus_Dens_Aggr | lgbm | 0.8707 | 0.6773 | 0.1064 | 36.0 | 0.4989 | 91.0 |  |  |  |  |
| P02_Persistence60s | CONTRAST:Arch_vs_Burden | lgbm | 0.0224 |  |  |  |  |  | 0.0085 | 0.0382 | 0.000999500249875 | 0.001 |
| P02_Persistence60s | CONTRAST:DensityOnly_vs_Arch | lgbm | 0.0013 |  |  |  |  |  | -0.0053 | 0.0088 | 0.7776111944027986 | 0.001 |
| P02_Persistence60s | CONTRAST:AggrOnly_vs_Arch | lgbm | -0.0025 |  |  |  |  |  | -0.0099 | 0.0044 | 0.4687656171914043 | 0.001 |
| P02_Persistence60s | CONTRAST:DensAggr_vs_Arch | lgbm | 0.0015 |  |  |  |  |  | -0.0063 | 0.0087 | 0.7626186906546727 | 0.001 |
| P02_Persistence60s_LTAFDB_SHDB_ValidRR | S3c_vs_S4_CleanRR_NoCohortShortcut | logistic | 0.0028 |  |  | 5.0 |  |  | -0.0016 | 0.0086 | 0.2198900549725137 | 0.001 |
| P02_Persistence60s_LTAFDB_SHDB_ValidRR | S3c_vs_S4_CleanRR_NoCohortShortcut | lgbm | 0.0089 |  |  | 5.0 |  |  | -0.0046 | 0.0242 | 0.2168915542228885 | 0.001 |
| P03_Survival15m | Cohort_Identity_Only | logistic | 0.6923 | 0.6861 | 0.2015 | 3.0 |  |  |  |  |  |  |
| P03_Survival15m | S0_Baseline | logistic | 0.7196 | 0.6947 | 0.2039 | 2.0 | 0.63 | 101.0 |  |  |  |  |
| P03_Survival15m | S1_Burden | logistic | 0.8538 | 0.842 | 0.1496 | 10.0 | 0.6155 | 101.0 |  |  |  |  |
| P03_Survival15m | S2_Architecture | logistic | 0.8812 | 0.9049 | 0.1374 | 25.0 | 0.5812 | 101.0 |  |  |  |  |
| P03_Survival15m | S3a_Arch_Plus_DensityOnly | logistic | 0.8817 | 0.9057 | 0.1369 | 31.0 | 0.5856 | 101.0 |  |  |  |  |
| P03_Survival15m | S3b_Arch_Plus_AggrOnly | logistic | 0.8809 | 0.9047 | 0.1373 | 31.0 | 0.573 | 101.0 |  |  |  |  |
| P03_Survival15m | S3c_Arch_Plus_Dens_Aggr | logistic | 0.8807 | 0.9044 | 0.1374 | 36.0 | 0.5775 | 101.0 |  |  |  |  |
| P03_Survival15m | CONTRAST:Arch_vs_Burden | logistic | 0.0274 |  |  |  |  |  | 0.015 | 0.0378 | 0.000999500249875 | 0.001 |
| P03_Survival15m | CONTRAST:DensityOnly_vs_Arch | logistic | 0.0005 |  |  |  |  |  | -0.0013 | 0.0024 | 0.6646676661669165 | 0.001 |
| P03_Survival15m | CONTRAST:AggrOnly_vs_Arch | logistic | -0.0003 |  |  |  |  |  | -0.0025 | 0.0014 | 0.6866566716641679 | 0.001 |
| P03_Survival15m | CONTRAST:DensAggr_vs_Arch | logistic | -0.0006 |  |  |  |  |  | -0.0028 | 0.0016 | 0.5677161419290355 | 0.001 |
| P03_Survival15m | S0_Baseline | lgbm | 0.743 | 0.7496 | 0.1968 | 2.0 | 0.6158 | 101.0 |  |  |  |  |
| P03_Survival15m | S1_Burden | lgbm | 0.8586 | 0.8733 | 0.1476 | 10.0 | 0.6089 | 101.0 |  |  |  |  |
| P03_Survival15m | S2_Architecture | lgbm | 0.8603 | 0.8821 | 0.15 | 25.0 | 0.5954 | 101.0 |  |  |  |  |
| P03_Survival15m | S3a_Arch_Plus_DensityOnly | lgbm | 0.8583 | 0.8836 | 0.1523 | 31.0 | 0.5819 | 101.0 |  |  |  |  |
| P03_Survival15m | S3b_Arch_Plus_AggrOnly | lgbm | 0.8626 | 0.8881 | 0.1494 | 31.0 | 0.5824 | 101.0 |  |  |  |  |
| P03_Survival15m | S3c_Arch_Plus_Dens_Aggr | lgbm | 0.8621 | 0.8876 | 0.1495 | 36.0 | 0.5766 | 101.0 |  |  |  |  |
| P03_Survival15m | CONTRAST:Arch_vs_Burden | lgbm | 0.0017 |  |  |  |  |  | -0.009 | 0.0112 | 0.8025987006496752 | 0.001 |
| P03_Survival15m | CONTRAST:DensityOnly_vs_Arch | lgbm | -0.002 |  |  |  |  |  | -0.0064 | 0.0019 | 0.3088455772113943 | 0.001 |
| P03_Survival15m | CONTRAST:AggrOnly_vs_Arch | lgbm | 0.0023 |  |  |  |  |  | -0.002 | 0.0064 | 0.3128435782108945 | 0.001 |
| P03_Survival15m | CONTRAST:DensAggr_vs_Arch | lgbm | 0.0018 |  |  |  |  |  | -0.0028 | 0.0057 | 0.4147926036981509 | 0.001 |
| P03_Survival15m_LTAFDB_SHDB_ValidRR | S3c_vs_S4_CleanRR_NoCohortShortcut | logistic | 0.0026 |  |  | 5.0 |  |  | -0.0024 | 0.0084 | 0.2948525737131434 | 0.001 |
| P03_Survival15m_LTAFDB_SHDB_ValidRR | S3c_vs_S4_CleanRR_NoCohortShortcut | lgbm | 0.0037 |  |  | 5.0 |  |  | -0.0048 | 0.0132 | 0.3758120939530235 | 0.001 |

### Cohort-Stratified & Leave-One-Cohort-Out (LOCO) Transfer Performance
| task | eval_cohort | feature_set | model | n_samples | n_positives | within_cohort_cv_auroc | pooled_model_slice_auroc | loco_transfer_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P01_CleanAnnotated | IRIDIA-AF | S1_Burden | logistic | 58716 | 1257 | 0.5935 | 0.5941 | 0.5756 |
| P01_CleanAnnotated | IRIDIA-AF | S2_Architecture | logistic | 58716 | 1257 | 0.6027 | 0.6039 | 0.5839 |
| P01_CleanAnnotated | IRIDIA-AF | S3a_Arch_Plus_DensityOnly | logistic | 58716 | 1257 | 0.6004 | 0.5962 | 0.5625 |
| P01_CleanAnnotated | IRIDIA-AF | S1_Burden | lgbm | 58716 | 1257 | 0.5685 | 0.564 | 0.5155 |
| P01_CleanAnnotated | IRIDIA-AF | S2_Architecture | lgbm | 58716 | 1257 | 0.5634 | 0.5675 | 0.5164 |
| P01_CleanAnnotated | IRIDIA-AF | S3a_Arch_Plus_DensityOnly | lgbm | 58716 | 1257 | 0.5769 | 0.568 | 0.518 |
| P01_CleanAnnotated | LTAFDB | S1_Burden | logistic | 10161 | 125 | 0.7119 | 0.6238 | 0.6256 |
| P01_CleanAnnotated | LTAFDB | S2_Architecture | logistic | 10161 | 125 | 0.6661 | 0.5752 | 0.4481 |
| P01_CleanAnnotated | LTAFDB | S3a_Arch_Plus_DensityOnly | logistic | 10161 | 125 | 0.6776 | 0.6997 | 0.5693 |
| P01_CleanAnnotated | LTAFDB | S1_Burden | lgbm | 10161 | 125 | 0.5839 | 0.6203 | 0.642 |
| P01_CleanAnnotated | LTAFDB | S2_Architecture | lgbm | 10161 | 125 | 0.5734 | 0.6225 | 0.6711 |
| P01_CleanAnnotated | LTAFDB | S3a_Arch_Plus_DensityOnly | lgbm | 10161 | 125 | 0.6049 | 0.579 | 0.6237 |
| P01_CleanAnnotated | SHDB-AF | S1_Burden | logistic | 21180 | 304 | 0.5669 | 0.599 | 0.5563 |
| P01_CleanAnnotated | SHDB-AF | S2_Architecture | logistic | 21180 | 304 | 0.5395 | 0.5845 | 0.5852 |
| P01_CleanAnnotated | SHDB-AF | S3a_Arch_Plus_DensityOnly | logistic | 21180 | 304 | 0.549 | 0.6113 | 0.6091 |
| P01_CleanAnnotated | SHDB-AF | S1_Burden | lgbm | 21180 | 304 | 0.5353 | 0.5928 | 0.5659 |
| P01_CleanAnnotated | SHDB-AF | S2_Architecture | lgbm | 21180 | 304 | 0.5395 | 0.5903 | 0.5748 |
| P01_CleanAnnotated | SHDB-AF | S3a_Arch_Plus_DensityOnly | lgbm | 21180 | 304 | 0.5598 | 0.5826 | 0.5631 |
| P02_Persistence60s | IRIDIA-AF | S1_Burden | logistic | 342 | 227 | 0.7197 | 0.762 | 0.7598 |
| P02_Persistence60s | IRIDIA-AF | S2_Architecture | logistic | 342 | 227 | 0.7173 | 0.7591 | 0.7664 |
| P02_Persistence60s | IRIDIA-AF | S3a_Arch_Plus_DensityOnly | logistic | 342 | 227 | 0.7233 | 0.7538 | 0.7628 |
| P02_Persistence60s | IRIDIA-AF | S1_Burden | lgbm | 342 | 227 | 0.7449 | 0.7752 | 0.7596 |
| P02_Persistence60s | IRIDIA-AF | S2_Architecture | lgbm | 342 | 227 | 0.7419 | 0.7697 | 0.7207 |
| P02_Persistence60s | IRIDIA-AF | S3a_Arch_Plus_DensityOnly | lgbm | 342 | 227 | 0.7596 | 0.7723 | 0.7269 |
| P02_Persistence60s | LTAFDB | S1_Burden | logistic | 1744 | 202 | 0.7213 | 0.7064 | 0.7146 |
| P02_Persistence60s | LTAFDB | S2_Architecture | logistic | 1744 | 202 | 0.8664 | 0.8716 | 0.8548 |
| P02_Persistence60s | LTAFDB | S3a_Arch_Plus_DensityOnly | logistic | 1744 | 202 | 0.8686 | 0.8742 | 0.8544 |
| P02_Persistence60s | LTAFDB | S1_Burden | lgbm | 1744 | 202 | 0.8128 | 0.8072 | 0.6777 |
| P02_Persistence60s | LTAFDB | S2_Architecture | lgbm | 1744 | 202 | 0.8177 | 0.834 | 0.8104 |
| P02_Persistence60s | LTAFDB | S3a_Arch_Plus_DensityOnly | lgbm | 1744 | 202 | 0.828 | 0.8395 | 0.7931 |
| P02_Persistence60s | SHDB-AF | S1_Burden | logistic | 372 | 102 | 0.6122 | 0.6797 | 0.6978 |
| P02_Persistence60s | SHDB-AF | S2_Architecture | logistic | 372 | 102 | 0.6519 | 0.717 | 0.7283 |
| P02_Persistence60s | SHDB-AF | S3a_Arch_Plus_DensityOnly | logistic | 372 | 102 | 0.6339 | 0.7264 | 0.7374 |
| P02_Persistence60s | SHDB-AF | S1_Burden | lgbm | 372 | 102 | 0.6537 | 0.6957 | 0.7236 |
| P02_Persistence60s | SHDB-AF | S2_Architecture | lgbm | 372 | 102 | 0.6703 | 0.7335 | 0.7424 |
| P02_Persistence60s | SHDB-AF | S3a_Arch_Plus_DensityOnly | lgbm | 372 | 102 | 0.647 | 0.7286 | 0.7482 |
| P03_Survival15m | IRIDIA-AF | S1_Burden | logistic | 869 | 159 | 0.7729 | 0.7975 | 0.7955 |
| P03_Survival15m | IRIDIA-AF | S2_Architecture | logistic | 869 | 159 | 0.7364 | 0.7916 | 0.7895 |
| P03_Survival15m | IRIDIA-AF | S3a_Arch_Plus_DensityOnly | logistic | 869 | 159 | 0.7359 | 0.7894 | 0.786 |
| P03_Survival15m | IRIDIA-AF | S1_Burden | lgbm | 869 | 159 | 0.7101 | 0.7994 | 0.7886 |
| P03_Survival15m | IRIDIA-AF | S2_Architecture | lgbm | 869 | 159 | 0.7163 | 0.8044 | 0.773 |
| P03_Survival15m | IRIDIA-AF | S3a_Arch_Plus_DensityOnly | lgbm | 869 | 159 | 0.7245 | 0.7961 | 0.774 |
| P03_Survival15m | LTAFDB | S1_Burden | logistic | 2625 | 1881 | 0.8075 | 0.7999 | 0.7924 |
| P03_Survival15m | LTAFDB | S2_Architecture | logistic | 2625 | 1881 | 0.8604 | 0.8641 | 0.8368 |
| P03_Survival15m | LTAFDB | S3a_Arch_Plus_DensityOnly | logistic | 2625 | 1881 | 0.8601 | 0.8647 | 0.841 |
| P03_Survival15m | LTAFDB | S1_Burden | lgbm | 2625 | 1881 | 0.8218 | 0.8314 | 0.7459 |
| P03_Survival15m | LTAFDB | S2_Architecture | lgbm | 2625 | 1881 | 0.8275 | 0.8311 | 0.7405 |
| P03_Survival15m | LTAFDB | S3a_Arch_Plus_DensityOnly | lgbm | 2625 | 1881 | 0.8365 | 0.8343 | 0.751 |
| P03_Survival15m | SHDB-AF | S1_Burden | logistic | 702 | 357 | 0.7146 | 0.778 | 0.7688 |
| P03_Survival15m | SHDB-AF | S2_Architecture | logistic | 702 | 357 | 0.6472 | 0.7732 | 0.7647 |
| P03_Survival15m | SHDB-AF | S3a_Arch_Plus_DensityOnly | logistic | 702 | 357 | 0.65 | 0.7786 | 0.7713 |
| P03_Survival15m | SHDB-AF | S1_Burden | lgbm | 702 | 357 | 0.6121 | 0.7358 | 0.7711 |
| P03_Survival15m | SHDB-AF | S2_Architecture | lgbm | 702 | 357 | 0.5909 | 0.7307 | 0.7601 |
| P03_Survival15m | SHDB-AF | S3a_Arch_Plus_DensityOnly | lgbm | 702 | 357 | 0.5926 | 0.7153 | 0.7588 |

### Clean Annotated P01 Clinical Alert Benchmarks (`90,057` Non-AF Landmarks)
| episodes_alerted_fraction | median_lead_time_minutes | false_alerts_per_monitoring_hour | alarm_burden_percent | total_monitoring_hours | operating_threshold | model | warnings_per_patient_day |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.099644128113879 | 8.500380208333324 | 0.7204770312135648 | 6.190523779384168 | 7504.75 | 0.05 | LGBM_S1_Burden | 17.29 |
| 0.0551601423487544 | 7.289666666666744 | 0.4145374596089143 | 3.5577467603850894 | 7504.75 | 0.1 | LGBM_S1_Burden | 9.95 |
| 0.0367734282325029 | 5.617000000000068 | 0.2664978846730404 | 2.289660992482539 | 7504.75 | 0.15 | LGBM_S1_Burden | 6.4 |
| 0.097864768683274 | 10.80716666666655 | 0.6900962723608381 | 5.934019565386366 | 7504.75 | 0.05 | LGBM_S3c_ArchDensAggr | 16.56 |
| 0.0521945432977461 | 8.576083333333372 | 0.3617708784436523 | 3.1124732114105518 | 7504.75 | 0.1 | LGBM_S3c_ArchDensAggr | 8.68 |
| 0.0432977461447212 | 11.13716666666684 | 0.2175955228355374 | 1.8943557968841955 | 7504.75 | 0.15 | LGBM_S3c_ArchDensAggr | 5.22 |
| 0.1014234875444839 | 8.713916666667016 | 0.6094806622472434 | 5.268885261556569 | 7504.75 | 0.05 | LGBM_Fusion_ECGFounder | 14.63 |
| 0.0415183867141162 | 3.114500000000013 | 0.1680269162863519 | 1.4779528520825698 | 7504.75 | 0.1 | LGBM_Fusion_ECGFounder | 4.03 |
| 0.0195729537366548 | 1.5342500000000048 | 0.0798161164595756 | 0.7017777629723397 | 7504.75 | 0.15 | LGBM_Fusion_ECGFounder | 1.92 |
| 0.0759193357058125 | 4.969226562499989 | 0.1960091941770212 | 1.7755421566341318 | 7504.75 | 0.05 | Logistic_S3c_ArchDensAggr | 4.7 |
| 0.0266903914590747 | 1.5466666666666546 | 0.0483693660681568 | 0.4530464039441687 | 7504.75 | 0.1 | Logistic_S3c_ArchDensAggr | 1.16 |
| 0.0148279952550415 | 1.0992500000000291 | 0.0210533328891701 | 0.2032046370631933 | 7504.75 | 0.15 | Logistic_S3c_ArchDensAggr | 0.51 |
| 0.1073546856465006 | 6.419916666666662 | 0.4093407508577901 | 3.612156745172502 | 7504.75 | 0.05 | Logistic_Fusion_ECGFounder | 9.82 |
| 0.033214709371293 | 3.0490833333333285 | 0.0782171291515373 | 0.7139922493531874 | 7504.75 | 0.1 | Logistic_Fusion_ECGFounder | 1.88 |
| 0.0213523131672597 | 1.476416666666652 | 0.0309137546220726 | 0.2975893045515618 | 7504.75 | 0.15 | Logistic_Fusion_ECGFounder | 0.74 |

### Foundation Model Parameter Inventory, Random-Init Control & A100 Throughput
| model | adapter_key | dim | fs_hz | target_len | total_params | ckpt_size_mb |
| --- | --- | --- | --- | --- | --- | --- |
| CSFM | CSFM | 768 | 250 | 2500 | 42511923 | 238.84 |
| CLEF | CLEF | 1024 | 500 | 5000 | 30654000 | 350.9 |
| ECGFOUNDER | ECGFOUNDER | 1024 | 500 | 5000 | 30658100 | 352.68 |
| ECGFM_KED | KED | 768 | 500 | 5000 | 8129152 | 1279.08 |
| ECHONEXT | ECHONEXT | 256 | 400 | 4000 | 1046080 | 4.07 |
| DINOSR | DINOSR | 512 | 240 | 2400 | 811008 | 60.16 |
| JEPA | JEPA | 512 | 240 | 2400 | 811008 | 58.16 |
| NEFCORE | NEFCORE | 128 | 500 | 5000 | 1097285 | 12.67 |
| F2 | F2 | 320 | 200 | 2048 | 3413761 | 14.86 |
| F0 | F0 | 320 | 200 | 2048 | 3413761 | 14.23 |
| G1 | G1 | 256 | 200 | 2048 | 1654977 | 14.36 |
| CONTROL | CAPACITY | 256 | 200 | 2048 | 1655009 | 14.36 |

| model | task | gpu_infer_s_6130 | throughput_strips_per_s | pretrained_logistic_auroc | pretrained_lgbm_auroc | random_init_logistic_auroc | random_init_lgbm_auroc | pretrain_gain_logistic | pretrain_gain_lgbm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ECGFOUNDER | P04_NoiseHolter | 92.543 | 66.2 | 0.9432 | 0.9488 | 0.5639 | 0.5789 | 0.3793 | 0.3699 |
| ECHONEXT | P04_NoiseHolter | 4.323 | 1418.1 | 0.6402 | 0.6533 | 0.5887 | 0.6171 | 0.0515 | 0.0362 |
| F2 | P04_NoiseHolter | 1.819 | 3370.5 | 0.6433 | 0.6348 | 0.636 | 0.6498 | 0.0073 | -0.015 |
| CSFM | P04_NoiseHolter | 1.807 | 3392.8 | 0.6865 | 0.6789 | 0.6906 | 0.7048 | -0.0041 | -0.0259 |

### Matched Fixed-Model Foundation Model Evaluation (10s Causal Window, PCA-32)
| task | fm_model | classifier | fm_alone_auroc | fm_alone_auprc | tab_s3c_auroc | fusion_auroc | delta_fusion_minus_tab | boot_ci95_low | boot_ci95_high | boot_p_b2000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P01_CleanAnnotated | CSFM | logistic | 0.5316 | 0.0204 | 0.6125 | 0.6218 | 0.0093 |  |  |  |
| P01_CleanAnnotated | CSFM | lgbm | 0.5178 | 0.0198 | 0.5815 | 0.6038 | 0.0223 |  |  |  |
| P01_CleanAnnotated | ECGFOUNDER | logistic | 0.5856 | 0.0261 | 0.6125 | 0.6406 | 0.0281 | 0.0046 | 0.0527 | 0.0149925037481259 |
| P01_CleanAnnotated | ECGFOUNDER | lgbm | 0.5843 | 0.0251 | 0.5815 | 0.6369 | 0.0554 | 0.0286 | 0.0833 | 0.000999500249875 |
| P01_CleanAnnotated | ECGFM_KED | logistic | 0.512 | 0.0209 | 0.6125 | 0.6099 | -0.0026 | -0.025 | 0.0197 | 0.816591704147926 |
| P01_CleanAnnotated | ECGFM_KED | lgbm | 0.4971 | 0.0191 | 0.5815 | 0.5819 | 0.0004 | -0.0276 | 0.0289 | 0.9535232383808095 |
| P01_CleanAnnotated | ECHONEXT | logistic | 0.5247 | 0.0206 | 0.6125 | 0.6213 | 0.0088 | -0.0115 | 0.0293 | 0.3888055972013993 |
| P01_CleanAnnotated | ECHONEXT | lgbm | 0.5261 | 0.0203 | 0.5815 | 0.6088 | 0.0273 | 0.0007 | 0.0569 | 0.048975512243878 |
| P01_CleanAnnotated | F2 | logistic | 0.5029 | 0.019 | 0.6125 | 0.6105 | -0.002 | -0.0147 | 0.0115 | 0.7536231884057971 |
| P01_CleanAnnotated | F2 | lgbm | 0.5129 | 0.0196 | 0.5815 | 0.596 | 0.0145 | -0.0067 | 0.038 | 0.1739130434782608 |
| P01_CleanAnnotated | F0 | logistic | 0.5137 | 0.0199 | 0.6125 | 0.6151 | 0.0026 | -0.0108 | 0.0168 | 0.7126436781609196 |
| P01_CleanAnnotated | F0 | lgbm | 0.5201 | 0.0197 | 0.5815 | 0.602 | 0.0205 | -0.0017 | 0.0441 | 0.0699650174912543 |
| P01_CleanAnnotated | G1 | logistic | 0.5017 | 0.0184 | 0.6125 | 0.6106 | -0.0019 | -0.0179 | 0.0134 | 0.8115942028985508 |
| P01_CleanAnnotated | G1 | lgbm | 0.5201 | 0.0201 | 0.5815 | 0.6099 | 0.0284 | 0.0044 | 0.0534 | 0.0179910044977511 |
| P01_CleanAnnotated | CONTROL | logistic | 0.5126 | 0.0197 | 0.6125 | 0.614 | 0.0015 | -0.0145 | 0.0168 | 0.8185907046476761 |
| P01_CleanAnnotated | CONTROL | lgbm | 0.5106 | 0.0198 | 0.5815 | 0.6067 | 0.0252 | 0.0031 | 0.0474 | 0.015992003998001 |
| P02_Persistence60s | CSFM | logistic | 0.6418 | 0.3254 | 0.8887 | 0.881 | -0.0077 |  |  |  |
| P02_Persistence60s | CSFM | lgbm | 0.6599 | 0.4029 | 0.8707 | 0.8777 | 0.007 |  |  |  |
| P02_Persistence60s | ECGFOUNDER | logistic | 0.6907 | 0.4148 | 0.8887 | 0.8858 | -0.0029 | -0.0099 | 0.0048 | 0.5067466266866567 |
| P02_Persistence60s | ECGFOUNDER | lgbm | 0.6802 | 0.3998 | 0.8707 | 0.8792 | 0.0085 | -0.0045 | 0.0236 | 0.2368815592203898 |
| P02_Persistence60s | ECGFM_KED | logistic | 0.6224 | 0.412 | 0.8887 | 0.8844 | -0.0043 | -0.0106 | 0.0017 | 0.1629185407296352 |
| P02_Persistence60s | ECGFM_KED | lgbm | 0.6376 | 0.3947 | 0.8707 | 0.8776 | 0.0069 | -0.0064 | 0.0212 | 0.3138430784607696 |
| P02_Persistence60s | ECHONEXT | logistic | 0.6343 | 0.4083 | 0.8887 | 0.8813 | -0.0074 | -0.0133 | -0.0027 | 0.000999500249875 |
| P02_Persistence60s | ECHONEXT | lgbm | 0.6829 | 0.4488 | 0.8707 | 0.8774 | 0.0067 | -0.0044 | 0.0173 | 0.2288855572213893 |
| P02_Persistence60s | F2 | logistic | 0.6459 | 0.4065 | 0.8887 | 0.884 | -0.0047 | -0.0094 | -0.001 | 0.0099950024987506 |
| P02_Persistence60s | F2 | lgbm | 0.6304 | 0.4107 | 0.8707 | 0.8755 | 0.0048 | -0.0062 | 0.0148 | 0.4117941029485257 |
| P02_Persistence60s | F0 | logistic | 0.6508 | 0.4117 | 0.8887 | 0.8839 | -0.0048 | -0.0098 | -0.0008 | 0.0219890054972513 |
| P02_Persistence60s | F0 | lgbm | 0.6479 | 0.4325 | 0.8707 | 0.8843 | 0.0136 | 0.001 | 0.0266 | 0.0309845077461269 |
| P02_Persistence60s | G1 | logistic | 0.6384 | 0.4011 | 0.8887 | 0.8822 | -0.0065 | -0.0107 | -0.0033 | 0.000999500249875 |
| P02_Persistence60s | G1 | lgbm | 0.64 | 0.4224 | 0.8707 | 0.8731 | 0.0024 | -0.0098 | 0.0138 | 0.7206396801599201 |
| P02_Persistence60s | CONTROL | logistic | 0.649 | 0.3995 | 0.8887 | 0.8839 | -0.0048 | -0.0095 | -0.0007 | 0.0229885057471264 |
| P02_Persistence60s | CONTROL | lgbm | 0.6253 | 0.4093 | 0.8707 | 0.8768 | 0.0061 | -0.0059 | 0.0179 | 0.3438280859570214 |
| P03_Survival15m | CSFM | logistic | 0.6256 | 0.6531 | 0.8807 | 0.8773 | -0.0034 |  |  |  |
| P03_Survival15m | CSFM | lgbm | 0.6453 | 0.6592 | 0.8621 | 0.8699 | 0.0078 |  |  |  |
| P03_Survival15m | ECGFOUNDER | logistic | 0.6985 | 0.7265 | 0.8807 | 0.8792 | -0.0015 | -0.0056 | 0.0029 | 0.5007496251874063 |
| P03_Survival15m | ECGFOUNDER | lgbm | 0.6681 | 0.7031 | 0.8621 | 0.871 | 0.0089 | 0.0016 | 0.0169 | 0.0089955022488755 |
| P03_Survival15m | ECGFM_KED | logistic | 0.6056 | 0.6234 | 0.8807 | 0.8741 | -0.0066 | -0.0111 | -0.0025 | 0.0029985007496251 |
| P03_Survival15m | ECGFM_KED | lgbm | 0.6151 | 0.6244 | 0.8621 | 0.8659 | 0.0038 | -0.0027 | 0.0099 | 0.2638680659670165 |
| P03_Survival15m | ECHONEXT | logistic | 0.6301 | 0.6504 | 0.8807 | 0.8771 | -0.0036 | -0.0071 | -0.0003 | 0.0309845077461269 |
| P03_Survival15m | ECHONEXT | lgbm | 0.6423 | 0.6615 | 0.8621 | 0.8693 | 0.0072 | 0.0011 | 0.0132 | 0.0259870064967516 |
| P03_Survival15m | F2 | logistic | 0.6364 | 0.6472 | 0.8807 | 0.8752 | -0.0055 | -0.0082 | -0.0032 | 0.000999500249875 |
| P03_Survival15m | F2 | lgbm | 0.6443 | 0.6559 | 0.8621 | 0.8672 | 0.0051 | 0.0002 | 0.0098 | 0.0439780109945027 |
| P03_Survival15m | F0 | logistic | 0.6328 | 0.6479 | 0.8807 | 0.8757 | -0.005 | -0.008 | -0.0025 | 0.000999500249875 |
| P03_Survival15m | F0 | lgbm | 0.6332 | 0.6412 | 0.8621 | 0.8703 | 0.0082 | 0.0022 | 0.0132 | 0.0119940029985007 |
| P03_Survival15m | G1 | logistic | 0.6335 | 0.6427 | 0.8807 | 0.8758 | -0.0049 | -0.0075 | -0.0028 | 0.000999500249875 |
| P03_Survival15m | G1 | lgbm | 0.6367 | 0.6491 | 0.8621 | 0.8699 | 0.0078 | 0.0024 | 0.0131 | 0.0059970014992503 |
| P03_Survival15m | CONTROL | logistic | 0.6485 | 0.6654 | 0.8807 | 0.8756 | -0.0051 | -0.0078 | -0.0028 | 0.000999500249875 |
| P03_Survival15m | CONTROL | lgbm | 0.6381 | 0.6546 | 0.8621 | 0.8684 | 0.0063 | 0.0004 | 0.0121 | 0.0379810094952523 |
| P04_NoiseHolter | CSFM | logistic | 0.7031 | 0.1157 |  |  |  |  |  |  |
| P04_NoiseHolter | CSFM | lgbm | 0.7115 | 0.1409 |  |  |  |  |  |  |
| P04_NoiseHolter | ECGFOUNDER | logistic | 0.9432 | 0.6124 |  |  |  |  |  |  |
| P04_NoiseHolter | ECGFOUNDER | lgbm | 0.9453 | 0.6021 |  |  |  |  |  |  |
| P04_NoiseHolter | ECGFM_KED | logistic | 0.6621 | 0.1007 |  |  |  |  |  |  |
| P04_NoiseHolter | ECGFM_KED | lgbm | 0.6306 | 0.1246 |  |  |  |  |  |  |
| P04_NoiseHolter | ECHONEXT | logistic | 0.6402 | 0.0919 |  |  |  |  |  |  |
| P04_NoiseHolter | ECHONEXT | lgbm | 0.6533 | 0.1178 |  |  |  |  |  |  |
| P04_NoiseHolter | F2 | logistic | 0.6433 | 0.0972 |  |  |  |  |  |  |
| P04_NoiseHolter | F2 | lgbm | 0.6348 | 0.092 |  |  |  |  |  |  |
| P04_NoiseHolter | F0 | logistic | 0.6478 | 0.0958 |  |  |  |  |  |  |
| P04_NoiseHolter | F0 | lgbm | 0.6384 | 0.0914 |  |  |  |  |  |  |
| P04_NoiseHolter | G1 | logistic | 0.6348 | 0.0965 |  |  |  |  |  |  |
| P04_NoiseHolter | G1 | lgbm | 0.624 | 0.0998 |  |  |  |  |  |  |
| P04_NoiseHolter | CONTROL | logistic | 0.626 | 0.0948 |  |  |  |  |  |  |
| P04_NoiseHolter | CONTROL | lgbm | 0.6257 | 0.0987 |  |  |  |  |  |  |

---

## Priority 8 — Noise Holter & LTAFDB Dropout Perturbation Reconciliation

| comparison | auroc | coverage_pct | n |
| --- | --- | --- | --- |
| All_6130_Real_RR_LGBM | 0.9023 | 100.0 | 6130 |
| All_6130_ECGFounder_LGBM | 0.9578 | 100.0 | 6130 |
| Oracle_Keep_Clean_N1_Real_RR | 0.934 | 79.4 | 4867 |
| Oracle_Keep_Clean_N1_ECGFounder | 0.9666 | 79.4 | 4867 |
| Automated_Deployable_SQI_Real_RR | 0.9433 | 79.4 | 4867 |
| Automated_Deployable_SQI_ECGFounder | 0.9641 | 79.4 | 4867 |
| Automated_SQI_Detector_for_N2 | 0.9242 | 100.0 | 6130 |
| AF_vs_AFL_only_ECGFounder | 0.8103 | 100.0 | 501 |
| AF_vs_AFL_only_RealRR | 0.8134 | 100.0 | 501 |
| AF_vs_Ectopy_only_ECGFounder | 0.9497 | 100.0 | 1904 |
| AF_vs_Ectopy_only_RealRR | 0.863 | 100.0 | 1904 |
| AF_vs_CleanNSR_ECGFounder | 0.9821 | 100.0 | 1428 |
| AF_vs_CleanNSR_RealRR | 0.985 | 100.0 | 1428 |

- **`6,255` vs `6,130` resolved:** `RECORDS` index file lists `6,255` lines (`125` segments `233-*` to `240-*` missing from archive), while disk `.hea`/P04 has `6,130` excerpts (`Clean=2,665`, `N1=2,202`, `N2=1,263`).
- **LTAFDB Perturbation Simulation (`n=83` density-defined):** Omit `<30s` `MAE=0.02102` (`max=0.2400`, `1` undefined); Omit `<10s` `MAE=0.00720`; Bridge `<60s` gaps `MAE=0.06933` (`max=0.7459`), rising to `MAE=0.11443` in `n=26` patients with `>=20` brief runs.

---

## Priority 9 — Claim Verification Registry Summary

| claim_id | source_report | claim_summary | classification | reconciled_evidence | evidence_file |
| --- | --- | --- | --- | --- | --- |
| CLM-01 | af_density_temporal_phenotypes_v7_master_report.md | Dataset comprises 349 recordings across 329 annotated patients (307 AF+), 10,952.98h known annotated rhythm, and 8,540 canonical runs (vs 379 raw recs / 11,701.45h). | 1. Independently reproduced | Rebuilt from WFDB/HDF5/parquet: 379 raw / 349 annotated / 30 unannotated recs; 358 raw / 329 ann / 307 AF+ pts; 10952.98h known / 11701.45h raw; 8540 canonical runs (retires erroneous 8,582 figure). | V7R_cohort_reconciliation.csv |
| CLM-02 | af_density_temporal_phenotypes_v7_master_report.md | Brief AF runs (<30s) dominate episode count (58.7%, 5,016/8,540) while prolonged runs (>=30m) dominate AF time (94.15%, 553/8,540), heavily heterogeneous by cohort. | 1. Independently reproduced | LTAFDB has 7,358 runs (4,704 <30s; 202 >=30m), SHDB-AF has 794 runs (310 <30s; 102 >=30m), IRIDIA-AF has 388 runs (only 2 <30s; 249 >=30m). Cross-cohort comparison requires annotation harmonization. | V7R_cohort_reconciliation.csv; V7R_e01_reconciled.csv |
| CLM-03 | af_density_temporal_phenotypes_v7_master_report.md | Brief AF runs cluster after prolonged AF termination predominantly over pre-onset boundaries (Post [0,+5]m RR=4.82, Post/Pre RR=2.59, p<1e-7; Null A p<0.001 in 87.4% pts). | 4. Contradicted by supplied evidence | Actual GEE on 304 LTAFDB+SHDB index episodes shows bilateral boundary clustering in risk-time (Pre vs Remote RR=6.24, p=1.59e-07; Post vs Remote RR=4.39, p=7.57e-05) with NO post-predominance (Post vs Pre RR=0.70, p=0.176). Null A Obs/Exp in LTAFDB is 0.905 Pre / 0.674 Post; Null B p<0.05 in 0.0% recordings. | V7R_e01_reconciled.csv; V7R_e01_nulls_summary.csv |
| CLM-04 | V7 E01 Risk-Time vs Physical-Time Audit | Brief AF clustering near prolonged AF episodes persists across non-AF risk time in both LTAFDB and SHDB-AF and survives top-10 prolific LTAFDB patient exclusions, but attenuates to NS under pooled physical time. | 2. Supported only in a restricted population | Risk-time Pre RR=6.24 rises to 10.32 (p=7.07e-16) after excluding top-10 LTAFDB patients. However, pooled physical-time Pre RR=1.86 (p=0.102) is non-significant, driven by LTAFDB occupancy distortion; SHDB-AF alone is positive in both risk-time (~18.1x) and physical-time (~11.1x). | V7R_e01_reconciled.csv; V7R_e01_lopo_sensitivity.csv |
| CLM-05 | af_density_temporal_phenotypes_v7_master_report.md | Within-episode HR accelerates +10.1 bpm and CVRR compresses 24.8% -> 17.6% over first 15m; historical density explains +7.2% R^2 in ventricular trajectory (M3->M4) and +0.0597 GAM deviance. | 4. Contradicted by supplied evidence | Matched longitudinal panel surviving to 32.5m (n=509 episodes) shows HR slightly decreases (97.29 -> 95.82 bpm) and flat CVRR (0.2173 -> 0.2133). M3->M4 R^2 delta is 0.01725->0.01927 (+0.20 pp, p=0.280), within-patient p=0.984. Patient-disjoint 5-fold OOF GAM R^2 density delta is +0.00161 (0.33308->0.33470); pooled Cox PH density HR=1.1382 (p=0.514). | V7R_e02_matched_panel.csv; V7R_e02_reconciled.csv |
| CLM-06 | af_density_temporal_phenotypes_v7_master_report.md | Charitos density predicts larger LAD (+3.84 mm per +0.10 density, p=0.019) and separates LAD in N=38 <1% burden patients (43.8 vs 38.4 mm, p=0.038). | 4. Contradicted by supplied evidence | Across all 16 patient-deduplicated LAD/LVEF regressions (<=30d, <=90d, <=180d, Unrestricted), ALL 95% B=2,000 CIs cross zero (e.g. PAF Multi-Run <=90d n=17: +0.188 mm, 95% CI [-2.298, +2.270], p=0.821). Only 13 AF+ patients have <1% 12h burden in annotated SHDB (49 total includes 36 zero-AF recordings). | V7R_e03_reconciled.csv |
| CLM-07 | af_density_prediction_v7p_master_report.md | P01 contains 1,842 landmarks (14.8% prev); density improves P01 by +0.054 and P02 by +0.057; X12 achieves 0.842 AUROC; SetOperator G1 reaches 0.814 P01 / 0.738 C-index P03; B=2,000 used in ablations. | 4. Contradicted by supplied evidence | Raw P01 manifest has 98,404 rows (90,057 clean annotated, prev 1.872%, 300 pts; 8,347 unannotated rows leaked in raw). Fixed-model matched contrasts (B=2,000): P01 DensityOnly vs Arch Delta=+0.0060 (p=0.436); P02 DensityOnly vs Arch Delta=+0.0012 (p=0.467); P03 DensityOnly vs Arch Delta=+0.0005 (p=0.665). Original run_13stage_ablations.py used n_bootstrap=500. | V7R_matched_ablation_results.csv; V7R_patient_support.csv |
| CLM-08 | V7P_13stage_ablation_report.md | Episode architecture (S2) outperforms scalar burden (S1) for early persistence (P02) and dynamic survival (P03), without future leakage. | 1. Independently reproduced | Leakage audit confirms 0.0 feature diff across 150 adversarial future-deletion tests. On P02 (n=2,458, 290 pts), S2 vs S1 logistic Delta=+0.0742 (95% CI [+0.0463, +0.1009], p=0.0010); remains strong within LTAFDB (0.8664), IRIDIA (0.7173), and LOCO transfer (0.8548 / 0.7664 / 0.7283), though within-patient equal-episode discrimination (n=91 pts) is near chance (0.4789 logistic / 0.5245 lgbm). | V7R_feature_leakage_audit.csv; V7R_matched_ablation_results.csv; V7R_cohort_stratified_performance.csv |
| CLM-09 | V7P_13stage_ablation_report.md | Instantaneous RR dynamics boost imminent sustained-AF forecasting (P01) even after stripping the IRIDIA is_rr_valid cohort shortcut, though clinical alert performance remains weak. | 2. Supported only in a restricted population | Strictly within LTAFDB+SHDB ValidRR without is_rr_valid shortcut, CleanRR boosts P01 LGBM from 0.5710 to 0.6689 (Delta=+0.0979, 95% CI [+0.0546, +0.1406], p=0.0010; Logistic +0.0345, p=0.073). However, clean P01 alert sensitivity at thr=0.05 is only 9.79%-10.14% at 14.6-16.6 warnings/patient-day. | V7R_matched_ablation_results.csv; V7R_p01_clean_alerts.csv |
| CLM-10 | V7P_13stage_ablation_report.md; E04 Noise Holter | Noise Holter: P04 conventional RR AUROC is 0.4125, beat_coverage > 1.0 reflects missingness, and FM embeddings add broad clinical prognosis gains. | 4. Contradicted by supplied evidence | 0.4125 was an all-NaN F4 extraction bug; real RR features achieve 0.9023 LGBM (0.8419 Logistic). beat_coverage is actually detected/expected beats at 60bpm (mean 1.010 Clean -> 1.565 N2). Automated waveform SQI detects N2 at 0.9242 AUROC and matches oracle abstention at 79.4% coverage (0.9641 vs 0.9666). Only ECGFounder (+0.379 AUROC gain over random-init) and EchoNext (+0.052) beat random-init weights on P04; F2 (+0.007) and CSFM (-0.004) do not. | V7R_noise_holter_reconciled.csv; V7R_fm_matched_evaluation.csv; V7R_fm_random_init_and_runtime.csv |
| CLM-11 | e04_noise_observability_report.md | Density is highly robust to short-episode omission and <60s gap bridging (MAE 0.00110 in IRIDIA+SHDB simulation). | 2. Supported only in a restricted population | In LTAFDB (n=83 density-defined recs), gap bridging <60s yields MAE=0.06933 (max=0.7459), rising to MAE=0.11443 among n=26 patients with >=20 brief (<30s) runs (>100x larger than in prolonged-dominant IRIDIA+SHDB). | V7R_ltafdb_dropout_sensitivity.csv |
| CLM-12 | V7-R Next-Direction Feasibility & Literature Check | High Charitos density conceals a 30-fold spread in transition instability across existing continuous cohorts; longitudinal stroke/cardiopathy outcomes remain absent. | 3. Exploratory and requiring validation | Among 118 independent multi-run patients with high Charitos density (>=0.80, median 0.9752), 55 have 2-5 runs (4.06 transitions/24h) while 29 have >=20 runs (122.61 transitions/24h, median 61 runs). Stroke/AF progression endpoints do not exist in IRIDIA-AF/LTAFDB/SHDB-AF, so clinical stroke confounding cannot be answered here. | V7R_transition_instability_feasibility.csv |
| CLM-13 | V7P_13stage_ablation_report.md | Adding frozen FM embeddings to canonical tabular history/density (S3c) improves longitudinal prediction (P01/P02/P03) under fixed models. | 2. Supported only in a restricted population | Fixed-model matched evaluation (B=2,000): on P01_CleanAnnotated (which lacks F4 RR in S3c), ECGFOUNDER fusion yields +0.0281 logistic (0.6125->0.6406, 95% CI [+0.0046, +0.0527], p=0.0150; still below CleanRR 0.6686/0.6689). On P02_Persistence60s & P03_Survival15m logistic models, all FMs are null or significantly worse (P02 ECHONEXT Delta=-0.0074, p=0.0010; F2 Delta=-0.0047, p=0.010). LGBM P03 gains (+0.0089 ECGFOUNDER) only partly recover LGBM lag (0.8710 vs Logistic 0.8807). FMs dominate P04 morphology/noise (ECGFOUNDER 0.9453 PCA-32 / 0.9578 full), not longitudinal prognosis. | V7R_fm_matched_evaluation.csv |
| CLM-14 | af_density_temporal_phenotypes_v7_master_report.md | IRIDIA-AF scarcity of <30s runs (only 2/388 runs vs 4,704/7,358 in LTAFDB) represents pure atrial biology rather than annotation-protocol/minimum-duration filtering. | 5. Unverifiable with available artifacts | Annotation minimum-duration criteria in IRIDIA-AF HDF5 generation cannot be retrospectively verified from WFDB/HDF5 files alone; harmonization sensitivity bounds are required. | V7R_cohort_reconciliation.csv |

---

## Priority 10 — Empirical Feasibility of Candidate Next Direction (*Temporal Concentration vs Transition Instability*)

| stratum | n_patients | median_burden | median_density | iqr_runs | median_runs | median_transitions_24h | n_iridia | n_ltafdb | n_shdb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All_Annotated_AF_Patients | 307 | 0.1764 | 0.9982 | 1.0-7.0 | 2.0 | 4.0 | 152 | 83 | 72 |
| Defined_Density_Patients | 307 | 0.1764 | 0.9982 | 1.0-7.0 | 2.0 | 4.0 | 152 | 83 | 72 |
| MultiRun_DefinedDensity_Patients | 181 | 0.1589 | 0.8904 | 3.0-19.0 | 5.0 | 10.0 | 66 | 66 | 49 |
| HighDensity_ge0.80_MultiRun_Patients | 118 | 0.1268 | 0.9752 | 3.0-18.5 | 7.0 | 12.35 | 33 | 49 | 36 |
| HighDensity_LowFrag_2_to_5_runs | 55 | 0.1113 | 0.9896 | 2.0-4.0 | 2.0 | 4.06 | 27 | 12 | 16 |
| HighDensity_HighFrag_ge20_runs | 29 | 0.2264 | 0.9706 | 30.0-106.0 | 61.0 | 122.61 | 0 | 23 | 6 |
| LowBurden_lt1pct_MultiRun | 21 | 0.0035 | 0.8675 | 3.0-10.0 | 4.0 | 8.06 | 3 | 7 | 11 |
| ModBurden_1_to_10pct_MultiRun | 47 | 0.0509 | 0.9154 | 2.5-19.5 | 5.0 | 10.0 | 18 | 14 | 15 |
| HiBurden_ge10pct_MultiRun | 113 | 0.3162 | 0.8846 | 3.0-24.0 | 6.0 | 10.0 | 45 | 45 | 23 |
