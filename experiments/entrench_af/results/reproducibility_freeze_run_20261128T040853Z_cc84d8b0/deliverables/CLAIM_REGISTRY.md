# ENTRENCH-AF Claim Registry

**Run Directory:** `/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/entrench_af/results/run_20261128T040853Z_cc84d8b0`  
**Generated:** `2026-11-28T04:16:24Z` | **Git:** `cc84d8b0` | **Definitions:** `6cc57e3a858b7b5f00277abb3ca0299bf8febcb84fc247192e9169ec5c8575a3`

## A. Established Findings (Verified Directly in Current Run)
1. **100% Run Reconciliation & No Synthetic Fallbacks:** All `8,540` canonical `v5` AF intervals reconcile to `4,425` harmonized `g = 0 s` runs (`388` IRIDIA, `751` SHDB, `3,286` LTAFDB; zero balance error), with synthetic beat imputation removed (`IRIDIA n_beats = NA`, `beat_source = unavailable_iridia`) and uncovered segments fixed to `unknown`. Post-hoc cohort hierarchy (`IRIDIA-AF` primary PAF, `SHDB-AF` stratified, `LTAFDB` supporting) is disclosed explicitly.
2. **Statistical Negative Duration Dependence After Accounting for Frailty & Onset Censoring:** Excluding `88` unknown-onset runs (`4,337` known-onset runs), shared-gamma-frailty Weibull models reject constant patient-specific hazard (`k = 1`) in `IRIDIA-AF` (`k = 0.661`, LRT `p = 4.30e-19`), `SHDB-AF` PAF (`k = 0.530`, LRT `p = 1.80e-182`), and `LTAFDB` (`k = 0.591` harmonized; `0.590` non-sinus gaps merged). Parametric frailty adjustment demonstrates statistical duration dependence within this model family rather than isolating cellular or electrical remodeling.
3. **Paired Aggregation Shift & Clinical Threshold Disagreement (`L_W <= A_W` Verified):** On matched `24-h` observation windows (2,195 known-onset native runs vs. 210 aggregated episodes; exact ledger balance + `rev_dis == 0`), daily summation shifts frailty Weibull shape by `Δk = +0.423` (95% CI `+0.284` to `+0.614`), and classifies `1.7%` of `IRIDIA-AF` windows (`1.6%` of patients), `9.5%` of `SHDB-AF` PAF windows (`8.5%` of patients), and `8.5%` of `LTAFDB` windows at `≥1 h` without any uninterrupted `≥1-h` run.
4. **Rhythm-Label Continuity Sensitivity (`continuity_nonsinus_gaps` / `continuity_except_normal`):** Merging non-sinus rhythm interruptions (`VT`, `B`, `IVR`, `T`, `AB`, `SVTA`) preserves the negative duration dependence (`LTAFDB k = 0.590`) and threshold disagreement findings.
5. **Duration–Reinitiation Association Is Primarily Between-Patient:** In patient-stratified Cox decomposition, the inverse `log_dur` hazard ratio attenuates toward null within individual patients, whereas between-patient phenotype drives the marginal hazard ratio.
6. **Negative Surface-ECG / Longitudinal Biomarker Results:** Landmark Cox (`120 s` and `720 s`) and matched-row LOCO validation show negligible out-of-cohort predictive discrimination from terminal f-wave or post-termination P-wave features.

## B. Exploratory / Withdrawn Observations (Not Claimed as Population Evidence)
1. **Population-Level Longitudinal ICC Claim Withdrawn:** Across `9` repeat-recording `IRIDIA-AF` patients (`>=2` event visits), `8/9` had `0%` `<360-s` reinitiation across all visits and `1` single-patient example (`patient_045`: `9/10` and `3/5` events 4 years apart) generated almost all variance (`n_boot_undefined_zero_var = 181/500`; bootstrap CI withheld as invalid/`NaN`). Rapid reinitiation cannot be claimed as a demonstrated population-level longitudinal phenotype.
2. **`PMV_A` Sensitivity Signal:** Nominally significant in one unlandmarked specification, null in primary complete-case and `120-s` / `720-s` landmarks; exploratory only.
3. **Prior Cumulative AF Burden (`cum_af_sec_before_t0`):** Sensitive to model specification and cohort stratification.

## C. Unresolved Limitations & Open Adjudication Items
1. **IRIDIA-AF Binary `NON_AF` Labels:** IRIDIA Zenodo metadata distinguishes only `AF` vs. `NON_AF` (without rhythm sub-classification; un-adjudicated prior to EP panel review via `reports/EP_Review_Packet_Cheung.md`).
2. **24-h Holter Horizon vs. Multi-Day Midnight Bridging:** Only `IRIDIA-AF` contains `>=36-h` / `>=48-h` multi-day recordings enabling wall-clock midnight bridging (`n = 68` calendar episodes, `3` multi-day consolidations); `SHDB-AF` and `LTAFDB` (`<=26.4 h`) support recording-aligned `24-h` windows.
3. **No Prospective Clinical Outcomes:** Analyses quantify episode-duration and hazard-shape reclassification rather than prospective stroke, ablation, or mortality outcomes.

## D. Consistent With Prior Literature
1. **Early Entrenchment Within Continuous Episodes:** Consistent with `Spitaler et al., Heart Rhythm 2026` (`10.1016/j.hrthm.2026.10.019`) over overlapping sub-day horizons, while quantifying how daily aggregation modifies estimated Weibull shape and threshold classification.
2. **Temporal AF Phenotypes (Staccato vs. Legato):** Consistent with `Wineinger et al., Heart Rhythm 2019` (`10.1016/j.hrthm.2018.08.012`), showing daily-burden aggregation merges rapid termination–reinitiation clusters.
3. **Duration Distributions & Circadian Nulls:** Reaffirms mixed lognormal/Weibull fit preference at Holter scales without claiming novel heavy-tail theory or circadian triggering.
