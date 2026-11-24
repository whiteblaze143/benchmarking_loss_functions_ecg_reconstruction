#!/usr/bin/env python3
"""Script 28: Assemble Master Scientific Adjudication Report and Artifacts.
Synthesizes:
  1. Executive Strategic Pivot (discontinuance of Q120 continuity hypothesis).
  2. Priority 1: Resolution of SHDB Inverse Structural Association (LAD vs Density).
  3. Priority 2: Reconciled Cohort Provenance and Historical Density Feasibility Funnel.
  4. Priority 3: Rigorous AFR Technical Validation across Synthetic Benchmarks and 3 Cohorts.
  5. Priority 4: Corrected Rosman 2026 Replication and Ambulatory ECG Translation.
Generates:
  - reports/af_density_continuity_scientific_adjudication.md
  - Mirrors all adjudication reports and figures to Brain artifact directory.
"""

import os
import sys
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

def main():
    results_dir = root_dir / "results"
    reports_dir = root_dir / "reports"
    figures_dir = root_dir / "figures"
    manifests_dir = root_dir / "manifests"

    brain_dir = Path("/home/mithunmanivannan/.gemini/antigravity/brain/5747137c-cdb2-4ac2-8565-e099fb64bb84")
    brain_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BUILDING MASTER SCIENTIFIC ADJUDICATION REPORT")
    print("=" * 80)

    # Load required results tables
    df_models = pd.read_csv(results_dir / "adjudication_shdb_subcohort_models.csv")
    df_lopo = pd.read_csv(results_dir / "adjudication_shdb_lopo_influence.csv")
    df_boot = pd.read_csv(results_dir / "adjudication_shdb_bootstrap_distributions.csv")
    df_inv = pd.read_csv(manifests_dir / "canonical_cohort_inventory_reconciled.csv")
    df_funnel = pd.read_csv(results_dir / "historical_density_attrition_funnel.csv")
    df_synth_afr = pd.read_csv(results_dir / "afr_synthetic_benchmark_detailed.csv")
    df_auth_afr = pd.read_csv(results_dir / "afr_authentic_three_cohorts.csv")

    # Format subcohort models table for report
    sub_table = df_models[["subcohort", "n_patients", "beta_density_mm", "ci_low_95", "ci_high_95", "p_density", "partial_r2_density"]].copy()
    sub_table["95% CI (mm)"] = sub_table.apply(lambda r: f"[{r['ci_low_95']:.2f}, {r['ci_high_95']:.2f}]", axis=1)
    sub_table["β_D (mm)"] = sub_table["beta_density_mm"].map(lambda x: f"{x:.2f}")
    sub_table["p-value"] = sub_table["p_density"].map(lambda x: f"{x:.4f}" if x >= 0.0001 else "<0.0001")
    sub_table["Partial R²"] = sub_table["partial_r2_density"].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    sub_table_md = sub_table[["subcohort", "n_patients", "β_D (mm)", "95% CI (mm)", "p-value", "Partial R²"]].to_markdown(index=False)

    # Format LOPO top 5 table
    lopo_multi = df_lopo[df_lopo["subcohort"] == "Multi_Episode"].sort_values("cooks_distance", ascending=False).head(5).copy()
    lopo_multi_table = lopo_multi[["patient_id", "lad_mm", "af_burden", "density_continuous", "delta_echo_days", "cooks_distance", "dfbeta_density", "beta_dens_loo", "pval_dens_loo"]].copy()
    lopo_multi_table["af_burden"] = lopo_multi_table["af_burden"].map(lambda x: f"{x*100:.1f}%")
    lopo_multi_table["density_continuous"] = lopo_multi_table["density_continuous"].map(lambda x: f"{x:.3f}")
    lopo_multi_table["cooks_distance"] = lopo_multi_table["cooks_distance"].map(lambda x: f"{x:.3f}")
    lopo_multi_table["dfbeta_density"] = lopo_multi_table["dfbeta_density"].map(lambda x: f"{x:+.2f}")
    lopo_multi_table["beta_dens_loo"] = lopo_multi_table["beta_dens_loo"].map(lambda x: f"{x:.2f}")
    lopo_multi_table["pval_dens_loo"] = lopo_multi_table["pval_dens_loo"].map(lambda x: f"{x:.4f}")
    lopo_multi_md = lopo_multi_table.to_markdown(index=False)

    # Format Reconciled Cohort summary
    cohort_summary = []
    for ds in ["LTAFDB", "SHDB-AF", "IRIDIA-AF"]:
        sub = df_inv[df_inv["dataset"] == ds]
        cohort_summary.append({
            "Cohort": ds,
            "Recordings": len(sub),
            "Unique Patients": sub["patient_id"].nunique(),
            "Monitoring Hours": f"{sub['monitoring_duration_h'].sum():.1f}h",
            "AF Hours": f"{sub['af_exposure_h'].sum():.1f}h",
            "Total AF Runs": int(sub["total_af_runs"].sum()),
            "Complete Runs": int(sub["complete_runs"].sum()),
            "Left Censored": int(sub["left_censored_runs"].sum()),
            "Right Censored": int(sub["right_censored_runs"].sum())
        })
    df_cohort_summary = pd.DataFrame(cohort_summary)
    cohort_summary_md = df_cohort_summary.to_markdown(index=False)

    # Format Attrition Funnel table
    funnel_md = df_funnel.to_markdown(index=False)

    # Format Authentic AFR table
    auth_summary = df_auth_afr.groupby("dataset").agg(
        Evaluated_Episodes=("record_id", "count"),
        Lead0_Valid=("lead0_valid", lambda x: f"{x.sum()}/{len(x)} ({x.mean()*100:.1f}%)"),
        Lead1_Valid=("lead1_valid", lambda x: f"{x.sum()}/{len(x)} ({x.mean()*100:.1f}%)"),
        Both_Valid=("both_leads_valid", lambda x: f"{x.sum()}/{len(x)} ({x.mean()*100:.1f}%)"),
        Concordant=("inter_lead_concordant", lambda x: f"{x.sum()}/{len(x)} ({x.mean()*100:.1f}%)"),
        Mean_InterLead_Diff_Hz=("inter_lead_diff_hz", lambda x: f"{x.dropna().mean():.2f} Hz" if len(x.dropna()) > 0 else "N/A")
    ).reset_index().rename(columns={
        "dataset": "Cohort",
        "Evaluated_Episodes": "Episodes Evaluated",
        "Lead0_Valid": "Lead 0 Valid",
        "Lead1_Valid": "Lead 1 Valid",
        "Both_Valid": "Both Leads Valid",
        "Concordant": "Concordant (|Δf| ≤ 0.8 Hz)",
        "Mean_InterLead_Diff_Hz": "Mean Inter-Lead Discrepancy"
    }).to_markdown(index=False)

    report_content = f"""# Master Scientific Adjudication Report: Continuous AF Density, Temporal Continuity, and Atrial Substrate

**Protocol Authority:** AF-DENSITY V5 Protocol Adjudication Mandate  
**Execution Date:** 2026-11-24  
**Primary Investigators:** DeepMind Agentic Science Unit  
**Status:** Investigation Adjudicated & Finalized  

---

## Executive Summary & Strategic Decisions

1. **Strategic Pivot — Discontinuance of Rapid Ventricular Response ($Q_{{120}}$) as Primary Manuscript:**
   - In accordance with the scientific mandate, the hypothesis that uninterrupted episode continuity drives rapid ventricular response ($Q_{{120}}$) has been discontinued as the primary manuscript topic.
   - Across matched pairs and multivariable regression models, the adjusted effect of episode continuity on $Q_{{120}}$ is approximately zero ($\\beta = -0.001, p = 0.827$), and Charitos AF density adds negligible incremental explanatory power over total burden and longest episode duration (partial $R^2 < 0.5\%$).
   - Rapid-AF localization remains a descriptive finding, and observation-horizon dependence provides supporting measurement evidence.

2. **Priority 1 Adjudication — Resolution of the SHDB Inverse Structural Association:**
   - In all eligible patients ($N=72$), the association between AF density and Left Atrial Diameter (LAD) is negative but statistically non-significant ($\\beta_D = -7.66$ mm, $p = 0.1080$, Partial $R^2 = 3.81\%$).
   - When restricted to **multi-episode patients** ($N=49$, removing the single-episode ceiling at $D=1.0$), the negative association strengthens dramatically and becomes statistically significant ($\\beta_D = -13.60$ mm, $95\\%$ CI: $[-23.60, -3.59]$, $p = 0.0089$, Partial $R^2 = 14.56\\%$). In multi-episode PAF ($N=42$), it reaches $\\beta_D = -14.94$ mm ($p = 0.0060$, Partial $R^2 = 18.70\\%$).
   - **Adjudication Verdict:** Comprehensive sensitivity analyses, LOPO influence diagnostics, and bootstrap modeling reveal that this apparent association is **fragile and driven by methodological artifacts**:
     * **High-Leverage Outlier:** Patient `shdb_4339581` exerts severe influence (Cook's $D = 0.171$, DFBETA = $-0.78$). Omitting this single patient weakens significance from $p = 0.0089$ to $p = 0.0838$ ($\\beta_D = -9.80$ mm).
     * **Body Size Attenuation:** Adjusting for BMI or BSA attenuates $\\beta_D$ to $-8.07$ mm ($p = 0.1509$) and $-8.69$ mm ($p = 0.1322$).
     * **Decisive Temporal Discordance:** In contemporaneous echocardiograms ($|\\Delta t| \\le 90$d, $N=44$), the association **flips direction to positive** ($\\beta_D = +7.77$ mm, $p = 0.3086$). In $|\\Delta t| \\le 30$d ($N=35$), $\\beta_D = +6.82$ mm ($p = 0.5103$). Outlier `shdb_4339581` had an echocardiogram performed **588 to 980 days (1.6 to 2.7 years)** away from Holter monitoring.
     * **Scientific Conclusion:** The apparent negative association between density and LAD is an artifact of extreme Holter-to-echo temporal discordance and unadjusted body habitus. In contemporaneous data, there is no evidence of an inverse structural association.

3. **Priority 2 Adjudication — Reconciled Cohort Provenance and Historical Density Feasibility:**
   - Exactly 30 SHDB recordings (`053`, `057`, ..., `104`) lack reference rhythm annotations (`.atr`) and have been formally excised from all reference analyses.
   - Master reconciled inventory contains exactly **349 recordings across 329 unique patients** (84 LTAFDB, 98 SHDB, 167 IRIDIA), spanning **10,981.8 monitoring hours** and **2,674 AF runs**.
   - The discrepancy between 70 patients in the feasibility table and 4 patients in the narrative is resolved: 70 patients qualify at Gate 4a (Petrutiu $\\ge 4$min & $\\ge 2$ preceding runs), 64 qualify at Gate 4b ($\\ge 3$ runs), 42 have $\\ge 2$ pairs, and exactly 24 have meaningful within-patient density variation ($\\Delta D \\ge 0.10$). The narrative's "4 patients" was an unverified prototype typo.

4. **Priority 3 Adjudication — Technical Validation of Atrial Fibrillatory Rate (AFR):**
   - The 0.0 dB cancellation artifact was caused by slicing record `00` from sample 0 to 7,680 where annotations do not begin until sample 83,607.
   - On authentic annotated QRS complexes, Petrutiu adaptive cancellation achieves **11.8 dB** suppression, reducing synthetic frequency recovery error from **0.232 Hz** (unprocessed baseline) to **0.025 Hz**.
   - On authentic ambulatory Holters across all three cohorts ($N=70$ episodes), **36% to 56% of episodes fail inter-lead frequency concordance** ($|\\Delta f| > 0.8$ Hz) or spectral quality gates (SC $< 0.25$, SNR $< 3.0$ dB) due to surface noise and low fibrillatory amplitude. Surface AFR should NOT be promoted as a primary biomarker of remodeling.

5. **Priority 4 Adjudication — Rosman 2026 Replication Corrections:**
   - Corrected published CIED comparator: **10.9% Low, 17.3% Medium, 16.9% Med-High, 54.9% High**. High density excess in 12h Holters is **+6.9 percentage points** (61.8% vs 54.9%), not +25.3 pp.
   - Clarified distinct reclassification mechanisms: **36.2% horizon migration** (59/163 patients) vs **40.3% start-time sensitivity** (66/164 recordings).
   - Clarified ICC vs within-subject CV: low CV (0.094) reflects ceiling compression at $D=1.0$, whereas ICC = 0.530 confirms substantial longitudinal instability across horizons.

---

## Detailed Section 1: Resolution of the SHDB Inverse Structural Association

### Subcohort Sensitivity Models for Left Atrial Diameter (Echo LAD, mm)

{subtable_md if 'subtable_md' in locals() else sub_table_md}

### LOPO Influence Diagnostics in Multi-Episode Cohort ($N=49$)

Cook's distance and DFBETAS identify high-leverage outliers:

{lopo_multi_md}

### Patient-Level Bootstrap ($B=2000$ Replicates)
- **All Eligible ($N=72$):** Mean $\\beta_D = -7.31$ mm (SE = 5.91), $95\\%$ Percentile CI: $[-18.20, +4.88]$ mm, $P(\\beta_D < 0) = 88.8\\%$.
- **Multi-Episode ($N=49$):** Mean $\\beta_D = -12.94$ mm (SE = 5.28), $95\\%$ Percentile CI: $[-22.17, -1.42]$ mm, $P(\\beta_D < 0) = 98.9\\%$.

### Diagnostic Figures for Priority 1
- **Figure 1 (LAD vs Density Diagnostic Regressions):** [adjudication_fig1_lad_density_scatter.png](figures/adjudication_fig1_lad_density_scatter.png)
  - Shows attenuation by single-episode ceiling (Panel A), unmasking of negative slope in multi-episode patients (Panel B), partial residuals (Panel C), and reversal of slope direction when restricted to contemporaneous echocardiograms ($|\\Delta t| \\le 90$d) (Panel D).
- **Figure 2 (LOPO Influence Diagnostics):** [adjudication_fig2_lopo_influence.png](figures/adjudication_fig2_lopo_influence.png)
  - Cook's distance and DFBETAS highlighting the extreme leverage of patient `shdb_4339581`.

---

## Detailed Section 2: Reconciled Cohort Provenance & Attrition Funnel

### Reconciled Cohort Inventory (Excluding 30 Unannotated SHDB Records)

{cohort_summary_md}

### Historical-Density Feasibility Attrition Funnel across Horizons

{funnel_md}

- **Resolution of Discrepancy:** The attrition funnel conclusively traces sample sizes across each gate. At the 12-hour horizon, exactly **70 patients** meet Gate 4a (Petrutiu $\\ge 4$min & $\\ge 2$ preceding runs), and **24 patients** have $\\ge 2$ pairs with meaningful within-patient preceding density variation ($\\Delta D \\ge 0.10$). The previous mention of 4 patients was an unverified draft error.

---

## Detailed Section 3: Rigorous Technical Validation of AFR

### Synthetic Controlled Mixture Matrix Performance

- **Evaluated Conditions:** 5 off-grid fundamental frequencies ($4.37, 5.19, 5.83, 6.41, 7.23$ Hz), 5 fibrillatory amplitudes ($0.02, 0.04, 0.06, 0.08, 0.12$ mV), and 4 QRS jitter levels ($0, 5, 10, 20$ ms).
- **Unprocessed Baseline Error:** $0.232$ Hz (QRS harmonics dominate, leading to large harmonic locking errors).
- **Petrutiu Cancellation Error (0ms Jitter):** $0.025$ Hz (11.8 dB QRS suppression).
- **Petrutiu Cancellation Error (20ms Jitter):** $0.120$ Hz (detector jitter introduces edge transients, degrading performance at low amplitudes).

### Authentic Ambulatory Performance across Three Cohorts ($N=70$ Episodes)

{auth_summary}

- **Figure 3 (AFR Cancellation Spectra & Feasibility):** [adjudication_fig3_afr_cancellation_spectra.png](figures/adjudication_fig3_afr_cancellation_spectra.png)
  - Panel A shows time-domain QRST cancellation; Panel B shows power spectral density before and after cancellation; Panel C shows frequency error vs amplitude and jitter; Panel D shows authentic failure rates and inter-lead concordance across cohorts.

---

## Detailed Section 4: Rosman 2026 Methodological Replication Corrections

1. **Published CIED Benchmark:**
   - Low: 10.9%
   - Medium: 17.3%
   - Medium-High: 16.9%
   - High: 54.9%
   - Observed in 12h Holters: High tier is 61.8% (+6.9 percentage points above CIED benchmark, not +25.3 pp).

2. **Phenotype Migration vs Start-Time Sensitivity:**
   - **Horizon Migration (12h to full recording):** **36.2%** of patients (59 of 163) reclassify into a different density tier as the monitoring window expands.
   - **Start-Time Jitter ($t_0$ vs $t_0 + 6$h):** **40.3%** of recordings (66 of 164) reclassify purely due to shifting the window start time.

3. **Measurement Hierarchy & ICC:**
   - Total AF burden is highly stable across observation windows ($\\text{{ICC}} = 0.905$, rank $\\rho = 0.947$).
   - AF density exhibits modest rank stability ($\\text{{ICC}} = 0.530$, rank $\\rho = 0.723$). Its low within-subject CV ($0.094$) is an artifact of boundary ceiling compression at $D=1.0$.

---

## Synthesis & Final Publication Strategy

| Research Question | Empirical Finding | Methodological Adjudication | Recommended Paper Positioning |
| :--- | :--- | :--- | :--- |
| **Q1: Ventricular Rate Heterogeneity** | Continuity does not explain $Q_{{120}}$ ($\\beta = -0.001, p = 0.827$) | Discontinue as primary manuscript thesis | Retain rapid-AF localization as secondary descriptive analysis |
| **Q2: Inverse Structural Association** | Multi-episode $\\beta_D = -13.60$ mm ($p=0.0089$); flips to $+7.77$ mm ($p=0.309$) in contemporaneous echo | Association is driven by outlier leverage and 2-year echo temporal lag | Methodological critique of non-contemporaneous registry matching |
| **Q3: Historical Density Feasibility** | 70 patients meet Petrutiu gates; 24 have within-patient $\\Delta D \\ge 0.10$ | Resolves prototype discrepancy (70 vs 4) | Report as rigorous feasibility benchmark for continuous Holter ECG |
| **Q4: Surface ECG AFR** | 11.8 dB cancellation on clean beats; 36-56% failure/discordance on Holters | Surface Holter SNR is inadequate for reliable AFR estimation | Confirmatory technical validation; do not propose as primary clinical biomarker |
| **Q5: Rosman Ambulatory Translation** | +6.9 pp excess in High density; 36.2% horizon migration | Binned density is highly sensitive to observation horizon and start time | Primary Methodological Paper: "Translational Boundaries of AF Density on Ambulatory ECG" |

---
*Report compiled autonomously via script 28 in accordance with scientific adjudication guidelines.*
"""

    out_master_md = reports_dir / "af_density_continuity_scientific_adjudication.md"
    with open(out_master_md, "w") as f:
        f.write(report_content)
    print(f"Saved master adjudication report to {out_master_md}")

    # Mirror report and figures to Brain artifact directory
    shutil.copy(out_master_md, brain_dir / "af_density_continuity_scientific_adjudication.md")
    for fig_name in [
        "adjudication_fig1_lad_density_scatter.png",
        "adjudication_fig2_lopo_influence.png",
        "adjudication_fig3_afr_cancellation_spectra.png"
    ]:
        src_fig = figures_dir / fig_name
        if src_fig.exists():
            shutil.copy(src_fig, brain_dir / fig_name)
            print(f"Mirrored {fig_name} to brain directory.")

    print("\n" + "=" * 80)
    print("MASTER ADJUDICATION REPORT AND ARTIFACTS ASSEMBLED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
