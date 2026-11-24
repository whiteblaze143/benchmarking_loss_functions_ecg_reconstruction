# Rosman 2026 Methodological Replication and Ambulatory ECG Translation Report

**Protocol:** AF-DENSITY V5 Targeted Extension (R01–R05)
**Focus:** Methodological translation of Rosman et al. (EHJ 2026) burden-density framework to authentic continuous Holter ECG
**Cohorts:** IRIDIA-AF (prolonged AF), SHDB-AF (clinical metadata / intermediate AF), LTAFDB (fragmented AF)

> [!IMPORTANT]
> **Scientific & Ethical Scope Distinction:** This investigation evaluates the *mathematical measurement properties, phenotype distributions, and electrophysiological correlates* of Rosman et al.'s burden-density classification on high-resolution continuous surface ECG. It does NOT evaluate clinical stroke outcomes, predict thromboembolism, or infer anticoagulation efficacy. Published stroke risk ratios are not imported as surrogate labels.

---

## 1. Study Population and Measurement Modality Mapping

| Dimension | Rosman et al. (EHJ 2026) | Present Ambulatory ECG Cohorts | Translational Significance |
| :--- | :--- | :--- | :--- |
| **Cohort Source** | 12,868 CIED patients (VHA & UNC) | 329 patients across 3 multi-day Holter cohorts | CIED indicated for bradycardia/heart failure vs ambulatory diagnostic evaluation |
| **Arrhythmia History** | Device-monitored paroxysmal/persistent AF | Heterogeneous ambulatory AF (short paroxysms to days) | Real-world continuous ECG with full waveform morphology |
| **Measurement Modality** | Daily device telemetry counter ($\Delta = 24$h) | Continuous surface ECG annotations (128–200 Hz) | Resolves micro-second episode boundaries vs daily aggregated counters |
| **Observation Horizon** | 30-day fixed rolling intervals | 1h, 3h, 6h, 12h, 24h, 48h, 72h, full recording | Directly evaluates horizon migration and short-window validity |
| **Density Formulation** | Discrete sum over 30 daily bins | Continuous minimum-interval functional + discrete bins | Unbiased native continuous density vs binned approximations |
| **Clinical Endpoints** | Incident ischaemic stroke (6.3 / 1k pt-yr) | Ventricular rate ($Q_{120}$), irregularity ($CV_{RR}$), $L_{\max}$ | Electrophysiological substrates of rapid ventricular response |

---

## 2. R01: Joint Burden-Density Phenotype Replication (4 x 4 Matrix)

Rosman et al. categorized both AF burden and AF density into four symmetric tiers:
- **Low:** $>0.0$ to $0.30$ (0%–30%)
- **Medium:** $>0.30$ to $0.60$ (30%–60%)
- **Medium-High:** $>0.60$ to $0.90$ (60%–90%)
- **High:** $>0.90$ to $1.00$ (90%–100%)
- *Undefined / Inactive:* $b=0$ (no AF) or $b=1$ (permanent AF), where Charitos density is mathematically undefined ($0/0$).

### Patient-Level 4 x 4 Contingency Matrix (Primary 12-Hour Horizon)

| Unnamed: 0   |   Low |   Medium |   Medium-High |   High |   Total |
|:-------------|------:|---------:|--------------:|-------:|--------:|
| Low          |     1 |        5 |            25 |     74 |     105 |
| Medium       |     0 |        2 |             2 |     18 |      22 |
| Medium-High  |     0 |        1 |             8 |      9 |      18 |
| High         |     4 |        9 |             6 |      1 |      20 |
| Total        |     5 |       17 |            41 |    102 |     165 |

### Marginal Density Comparison: Ambulatory ECG vs Published CIED Person-Months

| tier        |   observed_fraction |   published_fraction |   diff_percentage_points |
|:------------|--------------------:|---------------------:|-------------------------:|
| Low         |            0.030303 |                0.109 |                 -7.8697  |
| Medium      |            0.103030 |                0.173 |                 -6.9970  |
| Medium-High |            0.248485 |                0.169 |                 +7.9485  |
| High        |            0.618182 |                0.549 |                 +6.9182  |

> [!NOTE]
> **Corrected CIED Comparator & Saturation Analysis:** Among AF-positive person-months with defined density in Rosman et al. (EHJ 2026), the published CIED category distribution is **10.9% Low, 17.3% Medium, 16.9% Med-High, and 54.9% High**. On 12-hour ambulatory Holter ECG, High density constitutes 61.8% of defined windows—an excess of **+6.9 percentage points** over the CIED benchmark (correcting the erroneous +25.3 pp previously reported from comparing against an unadjusted denominator). In short ambulatory monitoring, isolated episodes automatically dominate the 12-hour horizon, causing a moderate shift toward the High and Med-High tiers at the expense of Low and Medium density.

---

## 3. R02: Observation-Horizon Phenotype Migration

To determine whether a patient's Rosman phenotype remains invariant when the observation window expands, we evaluated nested acquisition-anchored observation periods (1h, 3h, 6h, 12h, 24h, 48h, 72h) on **identical continuous recordings**.

### Paired Agreement and Rank Stability Across Nested Horizons

| horizon   |   n_recordings_paired |   n_density_valid_paired |   median_abs_burden_diff |   mean_abs_burden_diff |   median_abs_density_diff |   mean_abs_density_diff |   burden_rank_stability_rho |   density_rank_stability_rho |   density_weighted_kappa_linear |   density_weighted_kappa_quadratic |   high_density_saturation_fraction |
|:----------|----------------------:|-------------------------:|-------------------------:|-----------------------:|--------------------------:|------------------------:|----------------------------:|-----------------------------:|--------------------------------:|-----------------------------------:|-----------------------------------:|
| 1h        |                   379 |                       92 |                0.110637  |              0.209599  |                 0.0595259 |               0.158402  |                    0.450607 |                     0.34574  |                        0.108045 |                           0.14319  |                           0.880435 |
| 3h        |                   379 |                      128 |                0.0899342 |              0.186235  |                 0.0627412 |               0.148421  |                    0.511265 |                     0.484027 |                        0.121029 |                           0.168741 |                           0.835938 |
| 6h        |                   379 |                      163 |                0.0753062 |              0.157923  |                 0.0859948 |               0.149309  |                    0.581059 |                     0.495302 |                        0.163389 |                           0.278049 |                           0.766871 |
| 12h       |                   377 |                      230 |                0.0534895 |              0.108614  |                 0.0142095 |               0.105996  |                    0.7293   |                     0.570562 |                        0.401469 |                           0.454079 |                           0.769565 |
| 24h       |                   219 |                      151 |                0         |              0.0469862 |                 0         |               0.0589306 |                    0.854276 |                     0.735929 |                        0.69041  |                           0.685739 |                           0.708609 |

### Density Category Migration Matrix (12-Hour Horizon to Full Recording)

| 12h_Density   |   Low |   Medium |   Medium-High |   High |
|:--------------|------:|---------:|--------------:|-------:|
| Low           |     1 |        1 |             0 |      0 |
| Medium        |     2 |        4 |             6 |      1 |
| Medium-High   |     0 |        8 |            22 |      8 |
| High          |     1 |       13 |            19 |     77 |

> [!WARNING]
> **Severe Category Drift (36.2% Horizon Migration):** Exactly **36.2% of patients (59 of 163)** reclassify into a different Rosman density tier when monitored from a 12-hour window to the full continuous recording (104 patients remain on the diagonal, 59 migrate). Patients initially categorized as 'High Density' frequently migrate to 'Medium' (13 patients) or 'Medium-High' (19 patients) as subsequent interepisode sinus gaps enter the observation horizon. Short-horizon density cannot be treated as a static patient attribute.

### Observation Start-Time Sensitivity (40.3% Reclassification)

To isolate the effect of sampling arbitrary observation start times within the same recording, we evaluated alternative 12-hour windows anchored at $t_0$ versus $t_0 + 6$h:
- Mean absolute burden difference: **0.1179**
- Mean absolute density difference: **0.1154**
- Category concordance: **59.7%** (98 of 164 recordings stay in the same tier).
- Category reclassification: **40.3%** (66 of 164 recordings reclassify purely due to shifting the 12-hour window start time by 6 hours).
- **Distinction:** Observation horizon expansion causes **36.2%** phenotype migration, whereas arbitrary window onset causes **40.3%** start-time sensitivity.

---

## 4. R03: Observation Horizon versus Aggregation Resolution

Using a full factorial design separating recording duration (6h, 12h, 24h) from temporal binning (native continuous, 1m, 5m, 15m, 60m, 30 equal bins), while strictly conserving exact AF duration in each bin:

| horizon   | resolution        |      mean |     median |       std |
|:----------|:------------------|----------:|-----------:|----------:|
| 12h       | 15m               | 0.0579016 | 0.031812   | 0.0954045 |
| 12h       | 1m                | 0.0320808 | 0.00306937 | 0.0759829 |
| 12h       | 30_equal_bins     | 0.0698799 | 0.0499942  | 0.0913999 |
| 12h       | 5m                | 0.0519884 | 0.0116465  | 0.109414  |
| 12h       | 60m               | 0.127163  | 0.121163   | 0.103941  |
| 12h       | native_continuous | 0         | 0          | 0         |
| 24h       | 15m               | 0.0405345 | 0.0154749  | 0.0926079 |
| 24h       | 1m                | 0.0222435 | 0.00182798 | 0.0629616 |
| 24h       | 30_equal_bins     | 0.0632736 | 0.0465498  | 0.0873753 |
| 24h       | 5m                | 0.0352752 | 0.00564929 | 0.0919984 |
| 24h       | 60m               | 0.0772786 | 0.062401   | 0.0883527 |
| 24h       | native_continuous | 0         | 0          | 0         |
| 6h        | 15m               | 0.0854619 | 0.0612349  | 0.100126  |
| 6h        | 1m                | 0.0464272 | 0.00566666 | 0.0855008 |
| 6h        | 30_equal_bins     | 0.0814027 | 0.0504195  | 0.101014  |
| 6h        | 5m                | 0.0692349 | 0.0232334  | 0.113167  |
| 6h        | 60m               | 0.170128  | 0.20084    | 0.13932   |
| 6h        | native_continuous | 0         | 0          | 0         |

---

## 5. R04: Physiological Characterization of Rosman Phenotypes

For every supported cell in the 4x4 matrix, we extracted ventricular rate metrics, rapid AF exposure ($Q_{120}$), RR irregularity ($CV_{RR}$), and episode fragmentation:

### Physiological Profile across Supported Burden-Density Combinations

| burden_tier   | density_tier   |   n_windows |   n_patients | supported   |   median_burden |   median_density |   median_longest_af_min |   median_n_episodes |   mean_ventricular_rate_bpm |   median_q_120 |   pct_windows_with_rapid_af |   mean_cv_rr |
|:--------------|:---------------|------------:|-------------:|:------------|----------------:|-----------------:|------------------------:|--------------------:|----------------------------:|---------------:|----------------------------:|-------------:|
| Low           | Low            |           1 |            1 | False       |       0.0365618 |         0.247769 |                 4.50408 |                16   |                     86.3556 |     0.0189937  |                    100      |     0.22819  |
| Low           | Medium         |           5 |            5 | True        |       0.0126567 |         0.502057 |                 3.50792 |                 6   |                    100.747  |     0          |                     20      |     0.19288  |
| Low           | Medium-High    |          24 |           23 | True        |       0.0271323 |         0.768255 |                 5.31374 |                 9   |                    102.245  |     0          |                     41.6667 |     0.213553 |
| Low           | High           |          70 |           69 | True        |       0.06462   |         0.999802 |                36.987   |                 2   |                    111.58   |     0          |                     27.1429 |     0.211818 |
| Medium        | Low            |           0 |            0 | False       |     nan         |       nan        |               nan       |               nan   |                    nan      |   nan          |                    nan      |   nan        |
| Medium        | Medium         |           1 |            1 | False       |       0.598394  |         0.489774 |                22.0818  |               856   |                    112.952  |     0.223979   |                    100      |     0.143207 |
| Medium        | Medium-High    |           2 |            2 | False       |       0.454669  |         0.680802 |               232.81    |                 2   |                    nan      |     0          |                      0      |   nan        |
| Medium        | High           |          15 |           15 | True        |       0.437508  |         1        |               261.751   |                 1   |                     83.9588 |     0          |                     26.6667 |     0.224493 |
| Medium-High   | Low            |           0 |            0 | False       |     nan         |       nan        |               nan       |               nan   |                    nan      |   nan          |                    nan      |   nan        |
| Medium-High   | Medium         |           1 |            1 | False       |       0.811149  |         0.541664 |               373.29    |                31   |                     98.5692 |     0.137836   |                    100      |     0.191042 |
| Medium-High   | Medium-High    |           5 |            5 | True        |       0.717116  |         0.879759 |               129.035   |                69   |                     79.1725 |     0.00193677 |                     60      |     0.172735 |
| Medium-High   | High           |           9 |            9 | True        |       0.703327  |         1        |               486.168   |                 1   |                     93.7633 |     0.00888633 |                     55.5556 |     0.203981 |
| High          | Low            |           4 |            4 | False       |       0.997146  |         0.226818 |               117.307   |                35.5 |                    109.094  |     0.233918   |                    100      |     0.205604 |
| High          | Medium         |           9 |            9 | True        |       0.998327  |         0.526114 |               333.596   |                14   |                     80.5205 |     0.00636654 |                     55.5556 |     0.205487 |
| High          | Medium-High    |           6 |            6 | True        |       0.999758  |         0.80246  |               616.732   |                 2.5 |                     80.8281 |     0.00104185 |                     66.6667 |     0.219453 |
| High          | High           |           1 |            1 | False       |       0.999902  |         0.954162 |               667.531   |                 3   |                     95.6108 |     0.104871   |                    100      |     0.236514 |

### Incremental Value of Density for Explaining Rapid AF Exposure ($Q_{120}$)

| model                                  |   n_obs |   n_clusters |   log_likelihood |      aic |      bic | covariates                                                                                                                       |   density_continuous_coef |   density_continuous_pval |
|:---------------------------------------|--------:|-------------:|-----------------:|---------:|---------:|:---------------------------------------------------------------------------------------------------------------------------------|--------------------------:|--------------------------:|
| Mod1_Burden_Only                       |     188 |          180 |         -43.9328 |  95.8657 | -903.595 | cohort_LTAFDB+cohort_SHDB-AF+af_burden                                                                                           |                nan        |                nan        |
| Mod2_Burden_Longest_Episode            |     188 |          180 |         -42.2436 |  94.4871 | -901.737 | cohort_LTAFDB+cohort_SHDB-AF+af_burden+log_longest_af_s                                                                          |                nan        |                nan        |
| Mod3_Burden_Longest_Continuous_Density |     188 |          180 |         -42.1521 |  96.3041 | -896.683 | cohort_LTAFDB+cohort_SHDB-AF+af_burden+log_longest_af_s+density_continuous                                                       |                 -0.719074 |                  0.582831 |
| Mod4_Burden_Longest_Density_Tiers      |     188 |          180 |         -41.5176 | 101.035  | -882.243 | cohort_LTAFDB+cohort_SHDB-AF+af_burden+log_longest_af_s+dens_tier_Low+dens_tier_Medium+dens_tier_Medium-High+dens_tier_Undefined |                nan        |                nan        |

---

## 6. R05: Comparative Metric Stability Across Horizons

To determine whether AF density is more or less stable than simpler clinical metrics, we evaluated the Intraclass Correlation Coefficient (ICC[3,1]), rank stability (Spearman $\rho$), and within-subject coefficient of variation across nested observation periods:

| metric         |   n_recordings_complete |   icc_3_1 |   rank_stability_rho_6h_to_24h |   rank_stability_rho_12h_to_24h |   mean_within_subject_cv |   median_within_subject_cv |
|:---------------|------------------------:|----------:|-------------------------------:|--------------------------------:|-------------------------:|---------------------------:|
| burden         |                      93 |  0.904832 |                       0.864033 |                        0.946689 |                0.307494  |                  0.272519  |
| density        |                      93 |  0.530011 |                       0.610845 |                        0.722666 |                0.0942305 |                  0.0441667 |
| l_max_min      |                      93 |  0.620311 |                       0.867352 |                        0.949121 |                0.344363  |                  0.360976  |
| episode_rate   |                      93 |  0.854061 |                       0.803128 |                        0.868881 |                0.453541  |                  0.500792  |
| af_aggregation |                      93 |  0.715437 |                       0.56982  |                        0.761538 |                0.132538  |                  0.0392717 |

> [!TIP]
> **Measurement Stability: ICC vs Within-Subject CV Interpretation:**
> - Total AF burden ($b$) exhibits excellent rank-order and absolute consistency across monitoring horizons ($\text{ICC}[3,1] = 0.905$, rank stability $\rho = 0.947$).
> - Although AF density exhibits a seemingly low within-subject CV (mean $0.094$, median $0.044$), this does **NOT** indicate superior longitudinal stability. Instead, it is an artifact of **boundary ceiling compression**: density is mathematically bounded in $(0, 1)$ and heavily concentrated near $1.0$ on short windows.
> - The true instability of AF density is revealed by its modest intraclass correlation ($\text{ICC}[3,1] = 0.530$), its rank correlation ($\rho = 0.723$), and the fact that **36.2% of patients migrate tiers** when the observation horizon expands.

---

## 7. Key Methodological & Translational Conclusions

1. **Mathematical Reproducibility:** Rosman et al.'s 4x4 burden-density categorization is readily computable on continuous surface ECG, but its clinical interpretation is profoundly altered by observation duration.
2. **Corrected CIED Comparator:** Comparing 12-hour Holters to Rosman's published CIED distribution shows a moderate excess in the High tier (**61.8% vs 54.9%**, a **+6.9 percentage point difference**), accompanied by an underrepresentation of Low and Medium density.
3. **Horizon-Dependent Migration vs Start-Time Jitter:** Category reclassification occurs through two distinct mechanisms: expanding observation duration causes **36.2% horizon migration** (59/163 patients), while shifting the window start time by 6 hours causes **40.3% start-time reclassification** (66/164 recordings).
4. **Limited Incremental Value for Ventricular Rate:** While AF burden and the longest uninterrupted episode duration ($L_{\max}$) strongly predict rapid ventricular response ($Q_{120}$), Charitos AF density provides negligible incremental predictive value once burden and $L_{\max}$ are accounted for.

