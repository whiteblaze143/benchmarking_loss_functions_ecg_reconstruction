# Experiment E01 Report: Temporal Neighbourhoods of Prolonged AF in LTAFDB and SHDB

**Program:** T-CAIREM Physiological Dynamics & Continuous Ambulatory ECG  
**Directive:** AF-DENSITY V7 Experiment E01 Final Scientific Report  
**Author:** Model & Registry Architect (@architect)  
**Governing Standard:** Four-Category Evidence Segregation (Category 1: Source Reference; Category 2: Derived Reference; Category 3: Empirical Findings; Category 4: Scientific Interpretation). Zero synthetic proxies; strictly patient-disjoint validation; patient-clustered Poisson GLMs with exact $\ln(T_{\text{risk}})$ offsets.  
**Artifact Path:** [`experiments/af_density_temporal_phenotypes_v7/results/`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_temporal_phenotypes_v7/results/)  

---

## 1. Executive Summary & Core Empirical Findings

Experiment E01 maps the temporal organization of brief atrial fibrillation (AF) episodes relative to sustained, prolonged AF episodes ($\ge 30$ minutes) across continuous ambulatory Holter recordings from **LTAFDB** (84 patients, 1,960.6 monitoring hours) and **SHDB-AF** (93 patients, 2,330.7 monitoring hours).

```
                             Temporal Neighbourhood Rate Ratios (E01)
  ┌───────────────────────────┬───────────────┬─────────────────┬─────────────────┬──────────────────────┐
  │ Horizon / Comparison      │ Brief Onsets  │ Risk Time (h)   │ Rate (/h risk)  │ Rate Ratio [95% CI]  │
  ├───────────────────────────┼───────────────┼─────────────────┼─────────────────┼──────────────────────┤
  │ Remote Baseline (Ref)     │ 1,710         │ 1,175.52        │ 1.455           │ 1.00 (Reference)     │
  │ Pre-Onset [-5m, 0]        │ 67            │ 6.21            │ 10.794          │ 7.42 [3.79 – 14.54]  │
  │ Pre-Onset [-15m, 0]       │ 165           │ 18.19           │ 9.071           │ 6.24 [3.15 – 12.36]  │
  │ Post-Termination [0, +5m] │ 61            │ 6.84            │ 8.922           │ 6.13 [3.00 – 12.55]  │
  │ Post-Termination [0, +15m]│ 116           │ 18.16           │ 6.386           │ 4.39 [2.11 – 9.13]   │
  │ Post-Termination [+5–15m] │ 55            │ 11.33           │ 4.856           │ 3.34 [1.41 – 7.93]   │
  │ Post-Termination [+15–30m]│ 58            │ 16.69           │ 3.476           │ 2.39 [1.32 – 4.33]   │
  │ Post-Termination [+30–60m]│ 124           │ 34.50           │ 3.595           │ 2.47 [1.28 – 4.79]   │
  └───────────────────────────┴───────────────┴─────────────────┴─────────────────┴──────────────────────┘
```

### Key Statistical Conclusions:
1. **Immediate Post-Termination Clustering (IRA Phenomenon):**
   - The brief-run onset rate in the immediate 5 minutes post-termination ($[0, +5]$ min) is **6.13-fold higher** than remote non-AF baseline ($p = 6.92 \times 10^{-7}$, $95\%$ CI: $[3.00, 12.55]$).
   - In SHDB-AF telemetry, the post-termination rate ratio reaches **26.96-fold** ($p = 4.45 \times 10^{-15}$, $95\%$ CI: $[11.83, 61.42]$).
   - This provides ambulatory confirmation of the immediate reinitiation of atrial tachyarrhythmias (IRA) phenomenon described by Oral et al. (2003).
2. **Pronounced Pre-Onset Escalation:**
   - Pre-onset brief-run rates surge sharply in the $[-5, 0]$ minute window ($RR = 7.42$, $p = 5.25 \times 10^{-9}$, $95\%$ CI: $[3.79, 14.54]$) and $[-15, 0]$ minute window ($RR = 6.24$, $p = 1.59 \times 10^{-7}$).
   - Pre-onset brief runs do not represent background noise; they constitute a distinct, localized pre-onset escalation cascade preceding sustained rhythm consolidation.
3. **Crucial Necessity of Risk-Time ($T_{\text{risk}}$) Normalization:**
   - In LTAFDB post-termination 0–15m windows, out of 41.56 physical hours, **36.47 hours (87.8%) were occupied by recurring AF**, leaving only 5.09 hours of true non-AF risk time.
   - Evaluating physical time yields a severely distorted, paradoxical rate ratio of $0.57$ ($p = 0.193$), whereas true risk-time rate ratio is $3.95$ ($p = 0.0013$). Physical time produces a severe false negative because a patient who is actively fibrillating cannot re-initiate AF.
4. **Conditional Null Model Validation:**
   - Across $B=2,000$ permutations, **Null Model A** (uniform spatial repositioning of brief runs conditional on prolonged episodes) is rejected in 87.4% of eligible patients ($p_{\text{empirical}} < 0.001$).
   - **Null Model B** (episode-duration permutation preserving interepisode non-AF intervals) confirms that brief-run clustering is specific to prolonged sustained episodes rather than short runs.
5. **Connection to Continuous Charitos AF Density:**
   - Patients with continuous Charitos density $d > 0.40$ exhibit 74.2% of their brief runs within 15 minutes of a prolonged episode boundary, whereas patients with low density ($d < 0.15$) distribute brief runs across remote quiescent intervals.

---

## 2. Cohort Episode-Scale Duration Architecture

Episodes across LTAFDB, SHDB-AF, and IRIDIA-AF were classified into eight duration bins ($S_0$ through $L_3$):

| Class | Duration Definition | Count ($N$) | Fraction of Count (%) | Total Time (h) | Fraction of AF Time (%) | Complete Fraction (%) |
|---|---|---|---|---|---|---|
| **S0** | $<10$ seconds | 3,247 | 38.02% | 3.68 h | 0.12% | 100.0% |
| **S1** | $10 \le t < 30$ seconds | 1,769 | 20.71% | 9.01 h | 0.29% | 100.0% |
| **S2** | $30\text{s} \le t < 2$ minutes | 1,788 | 20.94% | 29.72 h | 0.96% | 99.9% |
| **S3** | $2 \le t < 6$ minutes | 733 | 8.58% | 41.52 h | 1.34% | 99.7% |
| **M1** | $6 \le t < 30$ minutes | 450 | 5.27% | 96.97 h | 3.13% | 98.4% |
| **L1** | $30\text{m} \le t < 1$ hour | 136 | 1.59% | 95.78 h | 3.09% | 97.8% |
| **L2** | $1 \le t < 6$ hours | 278 | 3.26% | 710.38 h | 22.95% | 90.3% |
| **L3** | $\ge 6$ hours | 139 | 1.63% | 2,108.77 h | 68.12% | 80.6% |
| **Total** | Full Inventory | **8,540** | **100.0%** | **3,095.84 h** | **100.0%** | **98.7%** |

### Insights on Duration Dissociation:
- **Count Dominance:** Brief episodes ($S_0 + S_1$, $<30$ seconds) constitute **58.73% of all AF episodes** (5,016 runs) but account for only **0.41% of total AF burden** (12.69 hours).
- **Burden Dominance:** Prolonged episodes ($L_2 + L_3$, $\ge 1$ hour) constitute only **4.88% of episodes** (417 runs) but account for **91.07% of all AF duration** (2,819.15 hours).
- Conventional scalar burden reflects only $L_2/L_3$ persistence, completely ignoring the trigger dynamic captured by $S_0/S_1$.

---

## 3. Prolonged Index Episodes & Temporal Neighbourhoods

Across LTAFDB and SHDB-AF, **304 index prolonged episodes** ($\ge 30$ min, comprising classes $L_1$, $L_2$, $L_3$) were extracted:
- **LTAFDB:** 202 index episodes across 49 independent recordings.
- **SHDB-AF:** 102 index episodes across 54 independent recordings.

### Overlap Deduplication & Window Definitions:
1. **Pre-Onset Windows:** $[-5, 0]$, $[-15, -5]$, $[-30, -15]$, $[-60, -30]$ minutes relative to index episode onset.
2. **Post-Termination Windows:** $[0, +5]$, $[+5, +15]$, $[+15, +30]$, $[+30, +60]$ minutes relative to index episode termination.
3. **Nearest-Boundary Assignment:** Where adjacent prolonged episodes produced overlapping pre/post windows, each sub-interval was assigned strictly to its closest boundary, eliminating double-counting.
4. **Remote Comparison Intervals:** 4,987 matched 15-minute non-AF comparison intervals situated $\ge 60$ minutes from any prolonged AF episode, providing 1,175.52 hours of quiescent baseline risk time.

---

## 4. Clustered Poisson Rate Models & Sensitivity Suite

All models were fit via generalized linear models (Poisson family with log link) using patient-clustered robust standard errors:

$$\ln(\mathbb{E}[Y_i]) = \beta_0 + \beta_1 \cdot \text{Condition}_i + \ln(T_{\text{risk}, i})$$

### Primary Rate Contrast Results:

1. **Post-0–15m vs Remote Non-AF:**
   - **Combined Cohort (ALL):** $RR = 4.39$ ($95\%$ CI: $[2.11, 9.13]$, $p = 7.57 \times 10^{-5}$).
   - **LTAFDB Alone:** $RR = 3.95$ ($95\%$ CI: $[1.71, 9.13]$, $p = 0.0013$).
   - **SHDB-AF Alone:** $RR = 18.18$ ($95\%$ CI: $[10.64, 31.08]$, $p = 2.88 \times 10^{-26}$).
2. **Post-0–5m vs Remote Non-AF:**
   - **Combined Cohort (ALL):** $RR = 6.13$ ($95\%$ CI: $[3.00, 12.55]$, $p = 6.92 \times 10^{-7}$).
   - **LTAFDB Alone:** $RR = 5.36$ ($95\%$ CI: $[2.47, 11.66]$, $p = 2.27 \times 10^{-5}$).
   - **SHDB-AF Alone:** $RR = 26.96$ ($95\%$ CI: $[11.83, 61.42]$, $p = 4.45 \times 10^{-15}$).
3. **Pre-0–5m vs Remote Non-AF:**
   - **Combined Cohort (ALL):** $RR = 7.42$ ($95\%$ CI: $[3.79, 14.54]$, $p = 5.25 \times 10^{-9}$).
   - **LTAFDB Alone:** $RR = 7.02$ ($95\%$ CI: $[3.47, 14.22]$, $p = 6.08 \times 10^{-8}$).
   - **SHDB-AF Alone:** $RR = 29.12$ ($95\%$ CI: $[9.10, 93.20]$, $p = 1.34 \times 10^{-8}$).
4. **Post-0–15m vs Pre-0–15m Symmetry:**
   - **Combined Cohort (ALL):** $RR = 0.70$ ($95\%$ CI: $[0.42, 1.17]$, $p = 0.176$).
   - Both pre-onset and post-termination margins are intensely hyper-arrhythmogenic compared to remote non-AF, but pre-onset shows a slightly higher event rate per unit risk time due to the dense pre-onset burst cascade.

### Sensitivity Analyses:
- **Exclusion of Top Prolific Patient (LTAFDB 115):** Removing the single patient with the highest run count leaves post-termination rate ratio statistically robust ($RR = 3.82$, $p < 0.001$), proving the phenomenon is not driven by outlier individuals.
- **Physical Time vs Risk Time Denominator Comparison:** Physical time attenuates post-termination rate ratios by an average factor of $3.25\times$ across all models, establishing that rhythm occupancy adjustment is mathematically mandatory in episodic arrhythmology.

---

## 5. Artifact Verification & Manifest Index

All 12 required E01 deliverables are generated and verified on disk:

| Deliverable File | Format | Records | Content Description |
|---|---|---|---|
| [`E01_duration_class_inventory.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_temporal_phenotypes_v7/results/E01_duration_class_inventory.csv) | CSV | 32 rows | Complete $S_0$–$L_3$ class distribution by cohort |
| [`E01_patient_feasibility.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_temporal_phenotypes_v7/results/E01_patient_feasibility.csv) | CSV | 307 rows | Patient-level episode counts, feasibility, and prolonged status |
| [`E01_index_episode_manifest.parquet`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_temporal_phenotypes_v7/results/E01_index_episode_manifest.parquet) | Parquet | 304 rows | Index prolonged episodes ($\ge 30$ min) with boundary timestamps |
| [`E01_neighbourhood_intervals.parquet`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_temporal_phenotypes_v7/results/E01_neighbourhood_intervals.parquet) | Parquet | 3,648 rows | Sliced neighbourhood intervals with nearest-boundary deduplication |
| [`E01_neighbourhood_event_assignments.parquet`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_temporal_phenotypes_v7/results/E01_neighbourhood_event_assignments.parquet) | Parquet | 1,619 rows | Individual brief runs assigned to pre/post windows |
| [`E01_remote_comparison_manifest.parquet`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_temporal_phenotypes_v7/results/E01_remote_comparison_manifest.parquet) | Parquet | 4,987 rows | Matched 15m remote non-AF comparison intervals |
| [`E01_risk_time_summary.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_temporal_phenotypes_v7/results/E01_risk_time_summary.csv) | CSV | 15 rows | Risk time, physical time, and rates across cohorts and windows |
| [`E01_neighbourhood_rate_ratios.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_temporal_phenotypes_v7/results/E01_neighbourhood_rate_ratios.csv) | CSV | 66 rows | Clustered Poisson GLM rate ratios, CIs, and p-values |
| [`E01_patient_level_effects.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_temporal_phenotypes_v7/results/E01_patient_level_effects.csv) | CSV | 115 rows | Patient-level rate ratios and within-subject contrasts |
| [`E01_conditional_null_results.parquet`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_temporal_phenotypes_v7/results/E01_conditional_null_results.parquet) | Parquet | 116 rows | Permutation distributions ($B=2,000$) for Null A and Null B |
| [`E01_density_architecture_results.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_temporal_phenotypes_v7/results/E01_density_architecture_results.csv) | CSV | 155 rows | Continuous Charitos density linked to brief run concentration |
| [`E01_sensitivity_analysis.csv`](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_temporal_phenotypes_v7/results/E01_sensitivity_analysis.csv) | CSV | 4 rows | Robustness checks across sub-cohorts and outlier exclusions |

Mirrored to `/data/mithunmanivannan/codex_artifacts/af_density_temporal_phenotypes_v7/results/`.
