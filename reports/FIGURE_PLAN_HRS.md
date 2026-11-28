# Heart Rhythm Society (HRS) Manuscript Figure & Visualization Plan

**Target Venue:** *Heart Rhythm* / *Heart Rhythm O2* (Clinician & Electrophysiologist Readership)  
**Output Directory:** `reports/figures_hrs/` (Editable Type-42 Vector PDF + 300 DPI PNG Verification Previews + LaTeX Snippets)  
**Governing Design Standard:** Zero synthetic arrays; 100% programmatic sourcing from certified CSV/Parquet/JSON outputs and authentic Holter ECG waveforms; zero interior `plt.title()` headers; concise clinical terminology; compact non-overlapping legends; Okabe-Ito colorblind-safe palette.

---

## 1. Visual Grammar & Clinical Ergonomics Rules

1. **Clinician-First Nomenclature:** Replace algorithmic or internal coding identifiers (e.g., `X04_f0_f1_f2_f3_density`, `CVRR delta`, `token symmetry`, `Null A`) with intuitive clinical terminology (`AF Density`, `AF Burden`, `Brief Runs (<30 s)`, `Sustained AF (>=30 min)`, `History + Density + HR/HRV`, `ECG Foundation Model`, `Spatially Shuffled Null`).
2. **Zero Chart Junk & Zero Over-Verbose Legends:**
   - Top and right axes spines removed on all analytical panels.
   - Panel identifiers (`A`, `B`, `C`, `D`) placed cleanly in bold sans-serif at upper-left outside data space; figure titles live exclusively in LaTeX `\caption{}`.
   - Direct end-of-curve text labels preferred over large boxed legends; where legends are needed, limit to $\le 4$ short entries ($\le 3$ words each) positioned strictly clear of data points and confidence intervals.
   - Shared colorbars placed outside figure margins (`constrained_layout=True` / `gridspec`), never overlapping scatter points.
3. **2026 Domain-Specific Visualization Tooling:**
   - **`ecg-plot` (v0.3.0):** Authentic clinical rhythm strips calibrated to **25 mm/s paper speed** and **10 mm/mV gain** on standard millimeter ECG grid paper.
   - **`vitabel` (v0.1.1) & `mediplot` (v0.2.0) Alignment Patterns:** Multi-track aligned physiological timelines pairing discrete rhythm-episode bars with continuous ventricular rate / irregularity and cumulative density tracks.
   - **`dcurves` (v1.1.7):** Clinical Decision Curve Analysis quantifying Net Benefit across clinically actionable threshold probabilities against *Treat All* and *Treat None* reference strategies.
   - **`statannotations` (v0.7.2) & `lifelines` (v0.30.3):** Clean hypothesis-testing brackets and censoring-aware Kaplan–Meier curves with patient-at-risk counts.
4. **HRS Print Dimensions:** Single-column (`3.5 in` width) and two-column full-width (`7.2 in` width) vector PDFs (`pdf.fonttype = 42` for editable Illustrator/typesetting fonts).

---

## 2. Planned Manuscript Figures (10 Figures)

| Figure | Clinical Story & Question | Panel Architecture | 2026 Package / Plot Type | Certified Data Source |
|---|---|---|---|---|
| **Fig 1** | **Equal AF Burden Can Reflect Fundamentally Different Temporal Phenotypes** | **A**: Paired 12 h patient rhythm strips (`mediplot`/`vitabel`-style tracks) comparing two real patients with ~15% burden (1 consolidated 3.6 h episode vs dispersed brief runs); **B**: Authentic 10 s Holter AF onset strip at **25 mm/s, 10 mm/mV** (`ecg-plot`, IRIDIA `record_095`); **C**: Patient-level Burden vs Charitos Density across cohorts colored by longest-episode share (external colorbar). | `ecg-plot` + `vitabel`/`mediplot` timeline + Scatter | `canonical_af_runs.parquet`, `E01_density_architecture_results.csv`, `iridia_af/ecg_hdf5/record_095` |
| **Fig 2** | **Brief Runs Dominate Episode Count Whereas Prolonged Episodes Dominate AF Burden** | **A**: Log-scale AF duration density across `IRIDIA-AF`, `LTAFDB`, and `SHDB-AF` (`n=307` AF-positive patients, 8,540 runs); **B**: Stacked horizontal bar contrasting episode count share (`58.7% <30 s`) vs cumulative AF time (`91.1% >=1 h`); **C**: Episode-weighted vs patient-averaged median duration gap. | Smooth KDE + 100% Stacked Bar + Paired Bar | `E01_duration_class_inventory.csv`, `canonical_af_runs.parquet` |
| **Fig 3** | **Brief AF Runs Surge Immediately Before and After Prolonged Episodes (IRA Phenomenon)** | **A**: Pooled risk-time Rate Ratio (`95% CI`) across pre-onset (`30–60m` to `0–5m`: `7.42x`) and post-termination (`0–5m` to `30–60m`: `6.13x`) vs remote baseline (`RR = 1.0`); **B**: Cohort breakdown (`LTAFDB` vs `SHDB-AF`, showing `26.96x` post-termination surge in hospital telemetry); **C**: Denominator distortion (`3.95x` true risk-time vs `0.57x` naive physical-time artifact). | Trajectory Forest Plot + Paired Bar | `E01_neighbourhood_rate_ratios.csv`, `E01_risk_time_summary.csv` |
| **Fig 4** | **Clustering Persists Beyond Chance Spatial Placement or Duration Ordering** | **A**: Patient-level Z-score distribution (`n=102` eligible patients) contrasting observed post-termination brief-run clustering vs spatially shuffled uniform null; **B**: Duration-ordering lag correlation vs duration-permuted null. | Violin + Strip Overlay + Null Line (`statannotations`) | `E01_conditional_null_results.parquet` |
| **Fig 5** | **Acute Ventricular Remodeling & Non-Linear Dependence of Subsequent Duration** | **A**: Within-episode HR acceleration (`+10.1 bpm`) and CVRR irregularity compression (`24.8% -> 17.6%`) across fixed elapsed landmarks (`0.5–3m` to `30–32.5m`); **B**: Non-linear GAM spline vs OLS R²/deviance gain explaining subsequent episode duration (`p = 2.0e-6`, `43.6%` dev expl); **C**: Cox PH Hazard Ratios (`Preceding Gap`, `Preceding Duration`, `6h Burden`). | Multi-landmark Mean±SE + Model Bar + Forest Plot | `E02_fixed_elapsed_rr_features.parquet`, `E02_gam_vs_ols_duration.csv`, `E02_subsequent_episode_models.csv` |
| **Fig 6** | **Low-Burden Divergence & Absence of Independent Structural Substrate Link (SHDB-AF)** | **A**: Low-burden strata (`<5%`, `5–10%`, `10–30%`) longest-episode dominance; **B**: Echocardiographic LAD across AF subtypes (`PAF`, `PerAF`, `Non-AF`) with pairwise p-values (`statannotations`); **C**: LAD vs AF Density (`<=90 d` PAF multi-run, `n=18`) marking the 3 high-leverage outliers (`Cook's D > 0.22`) showing non-significant OLS (`p=0.72`) and GAM (`p=0.93`). | Box+Strip (`statannotations`) + Leverage Scatter | `shdb_low_burden_subgroups.csv`, `shdb_12h_standardized_cohort.csv`, `shdb_lopo_cooks_distance.json`, `shdb_gam_vs_ols_lad.csv` |
| **Fig 7** | **Ambulatory Noise Blunts Conventional HRV Metrics & Distorts AF Density** | **A**: Authentic 10 s Holter ECG strips (`ecg-plot` **25 mm/s, 10 mm/mV**) across `Clean AF` (`002-020`), `N1 Mild Noise` (`001-002`), and `N2 Severe Noise` (`001-001`); **B**: CVRR separation (`Sinus vs AF`) collapsing from Clean (`AUROC 0.94`) to N1 (`0.83`) and N2 (`0.49`); **C**: Cumulative burden error (`<0.08%`) vs density distortion (`MAE 0.124`) when brief runs go undetected. | `ecg-plot` Strips + Grouped Box (`statannotations`) + Bar | `noise_holter_feature_degradation.csv`, `noise_holter_feature_matrix.csv`, `sensing_dropout_simulation_summary.json` |
| **Fig 8** | **Longitudinal Episode History vs Instantaneous ECG Foundation Models** | **A**: AUROC trajectory across progressive clinical information tiers (`Prior` -> `Burden` -> `Architecture` -> `Density` -> `+HR/HRV` vs `ECG FM Only` vs `Multimodal`) for **P01 Forecast**, **P02 Persistence (60s)**, **P03 Survival (15m)**, and **P04 Noise Rhythm**; **B**: Patient-clustered bootstrap contrasts (`Delta AUROC, 95% CI`) quantifying the value of episode architecture (`+0.074`), HR/HRV (`+0.071`), and FM on noisy strips (`+0.245`). | Clean Multi-line Progression + Contrast Forest Plot | `ablation_matrix_results.csv`, `mandatory_contrasts_audit.json` |
| **Fig 9** | **12 Pretrained ECG Models & SetOperators Across Static vs Prognostic Tasks** | **A**: Clean grouped dot-plot matrix across all 12 deep encoders (`ECGFounder`, `EchoNext`, `ECG-FM KED`, `CLEF`, `CSFM`, `DINO-SR`, `JEPA`, `NEFCore`, `SetOp F2/F0/G1/Ctrl`) comparing P04 static rhythm detection (`ECGFounder 0.958`) vs longitudinal P01–P03 (`<=0.70`); **B**: 5-fold Cox PH vs Weibull AFT survival discrimination on **P03** (`C-index 0.733`). | Grouped Cleveland Dot-Plot + Fold Boxplot | `foundation_model_probe_benchmark.csv`, `setoperator_downstream_heads.csv`, `V7P_survival_benchmarks.csv` |
| **Fig 10** | **Clinical Calibration, Decision Curve Net Benefit & Continuous Alert Frontier** | **A**: Out-of-fold Calibration reliability curves for **P01** (1.7% prev), **P02** (21.6% prev), and **P03** (57.1% prev) from 3.05 M OOF rows; **B**: Decision Curve Analysis (`dcurves`) comparing Net Benefit of **P02 Persistence (60s)** vs *Treat All* and *Treat None*; **C**: **P01 Imminent AF Alert** frontier (sensitivity vs false alerts/hour across thresholds). | Calibration Curve + `dcurves` DCA + Trade-off Curve | `V7P_out_of_fold_predictions.parquet`, `V7P_calibration_results.csv`, `V7P_alert_benchmarks.csv` |

---

## 3. Standalone Publication Tables (`TABLE_*.tex`)

1. **`TABLE_1_cohort_architecture.tex`:** Verified patient counts, recording hours, AF episode counts across duration classes ($S_0$–$L_3$), and burden share across IRIDIA-AF, LTAFDB, SHDB-AF, and Noise Holter.
2. **`TABLE_2_e01_e03_clinical_findings.tex`:** Risk-time rate ratios (E01), GAM vs OLS subsequent-duration deviance explained (E02), Cox termination hazard ratios (E02), and SHDB LAD/LVEF structural regression & Cook's $D$ sensitivity (E03).
3. **`TABLE_3_v7p_prediction_summary.tex`:** 5-fold patient-disjoint benchmark performance (AUROC, AUPRC, Brier, Calibration Slope) for P01–P04 across tabular, foundation-model, and SetOperator models.

---

## 4. Verification Protocol (Render-Then-Verify)

Every figure script (`gen_fig01_*.py` ... `gen_fig10_*.py`) will be executed cleanly, and **every single rendered `.png` output will be visually inspected** to confirm:
- Zero overlapping labels, zero clipped tick text, zero crowded legends covering data.
- Legible 8–10 pt typography at final *Heart Rhythm* print width.
- Complete fidelity to certified `.csv` / `.parquet` / `.json` values.
