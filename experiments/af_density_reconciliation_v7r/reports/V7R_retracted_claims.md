# AF-DENSITY V7-R: Retracted Claims & Superseding Evidence Ledger

**Audit Date:** 2026-11-24  
**Scope:** Independent numerical reconciliation of [af_density_temporal_phenotypes_v7_master_report.md](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/reports/af_density_temporal_phenotypes_v7_master_report.md), [af_density_prediction_v7p_master_report.md](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/reports/af_density_prediction_v7p_master_report.md), and [V7P_13stage_ablation_report.md](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/reports/V7P_13stage_ablation_report.md). Historical reports are preserved unmodified per V7-R governance; this document formally retracts contradicted claims.

---

## 1. Fabricated / Contradicted Predictive Master Report Claims (`af_density_prediction_v7p_master_report.md`)

1. **Retracted Population & Prevalence Figures (Lines 92, 115, 200, 256–258):**
   - *Retracted claim:* P01 contains 1,842 landmarks (14.8% prevalence), P02 contains 1,204 episodes (45.9% persistence), P03 contains 649 episodes.
   - *Verified ground truth:* Raw `P01_landmark_manifest.parquet` has `98,404` rows, including `8,347` unannotated rows (`8,126` from 30 unannotated SHDB-AF records + `221` `UNKNOWN` LTAFDB intervals). Clean annotated P01 has **`90,057` rows (`1,686` positives = `1.872%` prevalence across `300` patients)**. `P02` has **`2,458` episodes (`21.60%` persistence, `290` pts)**; `P03` has **`4,196` landmarks (`57.13%` 15m termination, `290` pts)**.
2. **Retracted Density Incremental AUROCs (+0.054 P01 / +0.057 P02; Lines 21, 103, 125):**
   - *Retracted claim:* Adding Charitos density to temporal architecture improves AUROC by `+0.054` on P01 (`0.714 -> 0.768`) and `+0.057` on P02 (`0.682 -> 0.739`).
   - *Verified fixed-model B=2,000 contrasts (`V7R_matched_ablation_results.csv`):* Adding Charitos density (`S3a_Arch_Plus_DensityOnly`) over `S2_Architecture` yields **zero statistically significant improvement** across every task and classifier:
     - `P01_CleanAnnotated`: Logistic $\Delta = +0.0060$ (95% CI $[-0.0082, +0.0227]$, $p = 0.436$); LightGBM $\Delta = -0.0031$ (95% CI $[-0.0155, +0.0087]$, $p = 0.622$).
     - `P02_Persistence60s`: Logistic $\Delta = +0.0012$ (95% CI $[-0.0020, +0.0046]$, $p = 0.467$); LightGBM $\Delta = +0.0013$ (95% CI $[-0.0053, +0.0088]$, $p = 0.778$).
     - `P03_Survival15m`: Logistic $\Delta = +0.0005$ (95% CI $[-0.0013, +0.0024]$, $p = 0.665$); LightGBM $\Delta = -0.0020$ (95% CI $[-0.0064, +0.0019]$, $p = 0.309$).
   - Furthermore, the detailed ablation report ([V7P_13stage_ablation_report.md](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/reports/V7P_13stage_ablation_report.md)) swapped stage meanings (`X02 -> X03` added episode architecture, while `X03 -> X04` added Charitos density).
3. **Retracted Multimodal Peak AUROC (0.842) & SetOperator Table (Lines 22, 110, 180–189, 210):**
   - *Retracted claim:* X12 achieves `0.842` AUROC on P01, `0.828` on P02, and SetOperator G1 achieves `0.814` P01 / `0.794` P02 / `0.738` C-index P03.
   - *Verified ground truth:* None of those numbers exist in `ablation_matrix_results.csv` or `setoperator_downstream_heads.csv`; `af_density_prediction_v7p_master_report.md` was written prior to the actual experimental runs.
4. **Retracted Bootstrap Resolution & X06->X07 RR Shortcut / P04 All-NaN Artifact:**
   - Original `run_13stage_ablations.py` used `n_bootstrap=500` (so `$p < 0.0001$` was impossible under simple tail-counting where `min_p = 2/501 = 0.00399`) and relied on `is_rr_valid` (`0.0` for all IRIDIA-AF, `1.0` for LTAFDB/SHDB) as an accidental cohort indicator while `P04 X07` (`0.4125`) ran on **all-NaN RR features**. With real RR features, `P04` RR LGBM reaches **`0.9023`** (`Logistic 0.8419`).

---

## 2. Fabricated / Contradicted V7 Master Report Claims (`af_density_temporal_phenotypes_v7_master_report.md`)

1. **Retracted Post-Termination Predominance & Null A 87.4% Claim (Lines 132–155):**
   - *Retracted claim:* E01 analyzed `553` prolonged episodes in LTAFDB+SHDB, found `[0,+5m]` RR = `4.82`, Post vs Pre RR = `2.59` ($p < 10^-7$), and Null A `p < 0.001` in `87.4%` of patients.
   - *Verified truth (`V7R_e01_reconciled.csv`, `V7R_e01_nulls_summary.csv`):* Only `304` prolonged episodes (`202` LTAFDB + `102` SHDB; `249` IRIDIA excluded) entered E01. Pooled risk-time `Pre vs Remote RR = 6.236` ($p = 1.59\times 10^-7$), `Post vs Remote RR = 4.390` ($p = 7.57\times 10^-5$), and **direct `Post vs Pre RR = 0.704` ($p = 0.1758$, NON-SIGNIFICANT)**. In Null A LTAFDB (`n=33`), observed counts are *below* Null A expectations (`Obs/Exp = 0.905` Pre, `0.674` Post); in `Null B`, `p < 0.05` holds in `0.00%` of recordings.
2. **Retracted Within-Episode HR/CVRR Progression & +7.2% Density R² Gain (Lines 165–190):**
   - *Retracted claim:* HR escalates `+10.1 bpm` (`118.4 -> 128.5`), CVRR compresses `24.8% -> 17.6%`, M4 density lifts trajectory $R^2$ from `0.212` to `0.284` (`+7.2%`), and 6h GAM deviance gain from density is `+0.0597`.
   - *Verified truth (`V7R_e02_matched_panel.csv`, `V7R_e02_reconciled.csv`):* In matched episodes surviving through 32.5m (`n=509`, `244` pts), HR changes from `97.29` (30–180s) to `95.82 bpm` (30–32.5m) and CVRR from `0.2173` to `0.2133`. Actual M3->M4 $R^2$ is `0.01725 -> 0.01927` (`+0.20 pp`, $p = 0.280$), within-patient $p = 0.984$. Actual 5-fold patient-disjoint OOF GAM $R^2$ delta from 6h density is **`+0.00161`** (`0.33308 -> 0.33470`); Cox PH density HR is `1.1382` ($p = 0.514$). Also `245 / 6,683` (`3.67%`) episodes had undefined 6h density improperly filled as `0.0`.
3. **Retracted SHDB LAD +3.84 mm Association & 38-Patient `<1%` Burden Subgroup (Lines 208, 213–215):**
   - *Retracted claim:* Density increases LAD by `+3.84 mm` per `+0.10` ($p = 0.019$), preserved after removing `shdb_4339581` (`+3.21 mm`, $p = 0.031$), and separates LAD in `N=38` patients with `<1%` burden (`p = 0.038`).
   - *Verified truth (`V7R_e03_reconciled.csv`):* Patient-deduplicated LAD/LVEF regressions across all latency cutoffs (`<=30d`, `<=90d`, `<=180d`, `Unrestricted`) are non-significant with every 95% CI spanning zero (e.g., Multi-Run PAF `<=90d` `n=17`: `+0.188 mm`, 95% CI `[-2.298, +2.270]`, $p = 0.821$). Annotated 12h SHDB has `49` patients `<1%` burden, of whom **only `13` are AF+** (`36` had zero AF in 12h).
4. **Retracted Noise Holter `beat_coverage` Definition & Uncalibrated Stability Claims:**
   - *Retracted claim:* `beat_coverage` (`1.010` Clean, `1.256` N1, `1.565` N2) was presented alongside `0.618` valid beat fraction and oracle N2 abstention (`0.724 -> 0.984` specificity) as a deployable policy, and E04 dropout simulation (`MAE=0.00110`) was generalized without LTAFDB.
   - *Verified truth (`V7R_noise_holter_reconciled.csv`, `V7R_ltafdb_dropout_sensitivity.csv`):* `beat_coverage` is actually `detected_beats / expected_beats_at_60bpm` (spurious peak over-detection under N2 noise). In LTAFDB (`n=83`), gap bridging `<60s` produces `MAE = 0.06933` (`max = 0.7459`), rising to `MAE = 0.11443` in `n=26` prolific brief-run patients.
5. **Retracted Canonical Counts (`8,582` runs & `10,982.8h`):**
   - *Verified truth:* Exact raw annotation count is **`8,540` canonical runs** (`10,952.98h` known annotated rhythm across `349` annotated / `379` raw recordings totaling `11,701.45h` raw).
