"""Priority 9: Build V7R Claim Registry and Audited Scientific Reports from Reconciled CSVs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path("experiments/af_density_reconciliation_v7r/results")
REP = Path("experiments/af_density_reconciliation_v7r/reports")
ROOT_REP = Path("reports")
REP.mkdir(parents=True, exist_ok=True)


def build_claim_registry():
    # Load verified empirical tables for dynamic cross-validation
    coh = pd.read_csv(RES / "V7R_cohort_reconciliation.csv").set_index("dataset")
    e01 = pd.read_csv(RES / "V7R_e01_reconciled.csv")
    e01_lopo = pd.read_csv(RES / "V7R_e01_lopo_sensitivity.csv")
    e01_nulls = pd.read_csv(RES / "V7R_e01_nulls_summary.csv").set_index("null_type")
    e02_panel = pd.read_csv(RES / "V7R_e02_matched_panel.csv")
    e02_rec = pd.read_csv(RES / "V7R_e02_reconciled.csv").set_index("cohort")
    e03_rec = pd.read_csv(RES / "V7R_e03_reconciled.csv")
    abl = pd.read_csv(RES / "V7R_matched_ablation_results.csv")
    nh = pd.read_csv(RES / "V7R_noise_holter_reconciled.csv").set_index("comparison")
    ltaf_drop = pd.read_csv(RES / "V7R_ltafdb_dropout_sensitivity.csv")

    def get_abl(task, fset, model="logistic"):
        r = abl[(abl["task"] == task) & (abl["feature_set"] == fset) & (abl["model"] == model)]
        return r.iloc[0]

    p1_dens = get_abl("P01_CleanAnnotated", "CONTRAST:DensityOnly_vs_Arch", "logistic")
    p2_dens = get_abl("P02_Persistence60s", "CONTRAST:DensityOnly_vs_Arch", "logistic")
    p2_arch = get_abl("P02_Persistence60s", "CONTRAST:Arch_vs_Burden", "logistic")
    p3_dens = get_abl("P03_Survival15m", "CONTRAST:DensityOnly_vs_Arch", "logistic")
    p1_rr_lgb = get_abl("P01_CleanAnnotated_LTAFDB_SHDB_ValidRR", "S3c_vs_S4_CleanRR_NoCohortShortcut", "lgbm")
    e01_all_pre = e01[(e01["dataset"] == "ALL") & (e01["comparison"] == "pre_0_15m_vs_remote") & (e01["exposure_type"] == "risk_time_h")].iloc[0]
    e01_all_post = e01[(e01["dataset"] == "ALL") & (e01["comparison"] == "post_0_15m_vs_remote") & (e01["exposure_type"] == "risk_time_h")].iloc[0]
    e01_all_pvp = e01[(e01["dataset"] == "ALL") & (e01["comparison"] == "post_0_15m_vs_pre_0_15m") & (e01["exposure_type"] == "risk_time_h")].iloc[0]
    e01_phys_pre = e01[(e01["dataset"] == "ALL") & (e01["comparison"] == "pre_0_15m_vs_remote") & (e01["exposure_type"] == "physical_time_h")].iloc[0]

    claims = [
        {
            "claim_id": "CLM-01",
            "source_report": "af_density_temporal_phenotypes_v7_master_report.md",
            "claim_summary": "Dataset comprises 349 recordings across 329 annotated patients (307 AF+), 10,952.98h known annotated rhythm, and 8,540 canonical runs (vs 379 raw recs / 11,701.45h).",
            "classification": "1. Independently reproduced",
            "reconciled_evidence": (
                f"Rebuilt from WFDB/HDF5/parquet: {int(coh.loc['COMBINED','raw_recordings'])} raw / "
                f"{int(coh.loc['COMBINED','annotated_recordings'])} annotated / {int(coh.loc['COMBINED','unannotated_recordings'])} unannotated recs; "
                f"{int(coh.loc['COMBINED','unique_patients_raw'])} raw / {int(coh.loc['COMBINED','unique_patients_annotated'])} ann / "
                f"{int(coh.loc['COMBINED','patients_with_af'])} AF+ pts; {coh.loc['COMBINED','annotated_known_rhythm_h']:.2f}h known / "
                f"{coh.loc['COMBINED','raw_recording_h']:.2f}h raw; {int(coh.loc['COMBINED','canonical_af_runs_sum'])} canonical runs (retires erroneous 8,582 figure)."
            ),
            "evidence_file": "V7R_cohort_reconciliation.csv",
        },
        {
            "claim_id": "CLM-02",
            "source_report": "af_density_temporal_phenotypes_v7_master_report.md",
            "claim_summary": "Brief AF runs (<30s) dominate episode count (58.7%, 5,016/8,540) while prolonged runs (>=30m) dominate AF time (94.15%, 553/8,540), heavily heterogeneous by cohort.",
            "classification": "1. Independently reproduced",
            "reconciled_evidence": "LTAFDB has 7,358 runs (4,704 <30s; 202 >=30m), SHDB-AF has 794 runs (310 <30s; 102 >=30m), IRIDIA-AF has 388 runs (only 2 <30s; 249 >=30m). Cross-cohort comparison requires annotation harmonization.",
            "evidence_file": "V7R_cohort_reconciliation.csv; V7R_e01_reconciled.csv",
        },
        {
            "claim_id": "CLM-03",
            "source_report": "af_density_temporal_phenotypes_v7_master_report.md",
            "claim_summary": "Brief AF runs cluster after prolonged AF termination predominantly over pre-onset boundaries (Post [0,+5]m RR=4.82, Post/Pre RR=2.59, p<1e-7; Null A p<0.001 in 87.4% pts).",
            "classification": "4. Contradicted by supplied evidence",
            "reconciled_evidence": (
                f"Actual GEE on 304 LTAFDB+SHDB index episodes shows bilateral boundary clustering in risk-time (Pre vs Remote RR={e01_all_pre['rr']:.2f}, p={e01_all_pre['p_val']:.2e}; "
                f"Post vs Remote RR={e01_all_post['rr']:.2f}, p={e01_all_post['p_val']:.2e}) with NO post-predominance (Post vs Pre RR={e01_all_pvp['rr']:.2f}, p={e01_all_pvp['p_val']:.3f}). "
                f"Null A Obs/Exp in LTAFDB is {e01_nulls.loc['Null_A_LTAFDB','pre_obs_over_exp']:.3f} Pre / {e01_nulls.loc['Null_A_LTAFDB','post_obs_over_exp']:.3f} Post; Null B p<0.05 in 0.0% recordings."
            ),
            "evidence_file": "V7R_e01_reconciled.csv; V7R_e01_nulls_summary.csv",
        },
        {
            "claim_id": "CLM-04",
            "source_report": "V7 E01 Risk-Time vs Physical-Time Audit",
            "claim_summary": "Brief AF clustering near prolonged AF episodes persists across non-AF risk time in both LTAFDB and SHDB-AF and survives top-10 prolific LTAFDB patient exclusions, but attenuates to NS under pooled physical time.",
            "classification": "2. Supported only in a restricted population",
            "reconciled_evidence": (
                f"Risk-time Pre RR={e01_all_pre['rr']:.2f} rises to {e01_lopo.iloc[-1]['pre_vs_remote_rr']:.2f} (p={e01_lopo.iloc[-1]['pre_vs_remote_p']:.2e}) after excluding top-10 LTAFDB patients. "
                f"However, pooled physical-time Pre RR={e01_phys_pre['rr']:.2f} (p={e01_phys_pre['p_val']:.3f}) is non-significant, driven by LTAFDB occupancy distortion; SHDB-AF alone is positive in both risk-time (~18.1x) and physical-time (~11.1x)."
            ),
            "evidence_file": "V7R_e01_reconciled.csv; V7R_e01_lopo_sensitivity.csv",
        },
        {
            "claim_id": "CLM-05",
            "source_report": "af_density_temporal_phenotypes_v7_master_report.md",
            "claim_summary": "Within-episode HR accelerates +10.1 bpm and CVRR compresses 24.8% -> 17.6% over first 15m; historical density explains +7.2% R^2 in ventricular trajectory (M3->M4) and +0.0597 GAM deviance.",
            "classification": "4. Contradicted by supplied evidence",
            "reconciled_evidence": (
                f"Matched longitudinal panel surviving to 32.5m (n={int(e02_panel.iloc[-1]['n_episodes'])} episodes) shows HR slightly decreases ({e02_panel.iloc[3]['mean_hr_bpm']:.2f} -> {e02_panel.iloc[-1]['mean_hr_bpm']:.2f} bpm) "
                f"and flat CVRR ({e02_panel.iloc[3]['mean_cvrr']:.4f} -> {e02_panel.iloc[-1]['mean_cvrr']:.4f}). M3->M4 R^2 delta is 0.01725->0.01927 (+0.20 pp, p=0.280), within-patient p=0.984. "
                f"Patient-disjoint 5-fold OOF GAM R^2 density delta is +{e02_rec.loc['Pooled_DefinedDensity','gam_5fold_oof_delta_density']:.5f} (0.33308->0.33470); pooled Cox PH density HR={e02_rec.loc['Pooled_DefinedDensity','density_hr']:.4f} (p={e02_rec.loc['Pooled_DefinedDensity','p_value']:.3f})."
            ),
            "evidence_file": "V7R_e02_matched_panel.csv; V7R_e02_reconciled.csv",
        },
        {
            "claim_id": "CLM-06",
            "source_report": "af_density_temporal_phenotypes_v7_master_report.md",
            "claim_summary": "Charitos density predicts larger LAD (+3.84 mm per +0.10 density, p=0.019) and separates LAD in N=38 <1% burden patients (43.8 vs 38.4 mm, p=0.038).",
            "classification": "4. Contradicted by supplied evidence",
            "reconciled_evidence": (
                f"Across all 16 patient-deduplicated LAD/LVEF regressions (<=30d, <=90d, <=180d, Unrestricted), ALL 95% B=2,000 CIs cross zero (e.g. PAF Multi-Run <=90d n=17: +{e03_rec[(e03_rec['cohort_subset']=='PAF_MultiRun_PatientDedup')&(e03_rec['echo_latency_window']=='<=90d')&(e03_rec['outcome']=='Echo_LAD')].iloc[0]['effect_per_0_10_density']:.3f} mm, "
                f"95% CI [-2.298, +2.270], p=0.821). Only 13 AF+ patients have <1% 12h burden in annotated SHDB (49 total includes 36 zero-AF recordings)."
            ),
            "evidence_file": "V7R_e03_reconciled.csv",
        },
        {
            "claim_id": "CLM-07",
            "source_report": "af_density_prediction_v7p_master_report.md",
            "claim_summary": "P01 contains 1,842 landmarks (14.8% prev); density improves P01 by +0.054 and P02 by +0.057; X12 achieves 0.842 AUROC; SetOperator G1 reaches 0.814 P01 / 0.738 C-index P03; B=2,000 used in ablations.",
            "classification": "4. Contradicted by supplied evidence",
            "reconciled_evidence": (
                "Raw P01 manifest has 98,404 rows (90,057 clean annotated, prev 1.872%, 300 pts; 8,347 unannotated rows leaked in raw). "
                f"Fixed-model matched contrasts (B=2,000): P01 DensityOnly vs Arch Delta={p1_dens['auroc_pooled']:+.4f} (p={p1_dens['p_value_b2000']:.3f}); "
                f"P02 DensityOnly vs Arch Delta={p2_dens['auroc_pooled']:+.4f} (p={p2_dens['p_value_b2000']:.3f}); "
                f"P03 DensityOnly vs Arch Delta={p3_dens['auroc_pooled']:+.4f} (p={p3_dens['p_value_b2000']:.3f}). Original run_13stage_ablations.py used n_bootstrap=500."
            ),
            "evidence_file": "V7R_matched_ablation_results.csv; V7R_patient_support.csv",
        },
        {
            "claim_id": "CLM-08",
            "source_report": "V7P_13stage_ablation_report.md",
            "claim_summary": "Episode architecture (S2) outperforms scalar burden (S1) for early persistence (P02) and dynamic survival (P03), without future leakage.",
            "classification": "1. Independently reproduced",
            "reconciled_evidence": (
                f"Leakage audit confirms 0.0 feature diff across 150 adversarial future-deletion tests. On P02 (n=2,458, 290 pts), S2 vs S1 logistic Delta={p2_arch['auroc_pooled']:+.4f} (95% CI [{p2_arch['ci95_low']:+.4f}, {p2_arch['ci95_high']:+.4f}], p={p2_arch['p_value_b2000']:.4f}); "
                "remains strong within LTAFDB (0.8664), IRIDIA (0.7173), and LOCO transfer (0.8548 / 0.7664 / 0.7283), though within-patient equal-episode discrimination (n=91 pts) is near chance (0.4789 logistic / 0.5245 lgbm)."
            ),
            "evidence_file": "V7R_feature_leakage_audit.csv; V7R_matched_ablation_results.csv; V7R_cohort_stratified_performance.csv",
        },
        {
            "claim_id": "CLM-09",
            "source_report": "V7P_13stage_ablation_report.md",
            "claim_summary": "Instantaneous RR dynamics boost imminent sustained-AF forecasting (P01) even after stripping the IRIDIA is_rr_valid cohort shortcut, though clinical alert performance remains weak.",
            "classification": "2. Supported only in a restricted population",
            "reconciled_evidence": (
                f"Strictly within LTAFDB+SHDB ValidRR without is_rr_valid shortcut, CleanRR boosts P01 LGBM from 0.5710 to 0.6689 (Delta={p1_rr_lgb['auroc_pooled']:+.4f}, 95% CI [{p1_rr_lgb['ci95_low']:+.4f}, {p1_rr_lgb['ci95_high']:+.4f}], p={p1_rr_lgb['p_value_b2000']:.4f}; Logistic +0.0345, p=0.073). "
                "However, clean P01 alert sensitivity at thr=0.05 is only 9.79%-10.14% at 14.6-16.6 warnings/patient-day."
            ),
            "evidence_file": "V7R_matched_ablation_results.csv; V7R_p01_clean_alerts.csv",
        },
        {
            "claim_id": "CLM-10",
            "source_report": "V7P_13stage_ablation_report.md; E04 Noise Holter",
            "claim_summary": "Noise Holter: P04 conventional RR AUROC is 0.4125, beat_coverage > 1.0 reflects missingness, and FM embeddings add broad clinical prognosis gains.",
            "classification": "4. Contradicted by supplied evidence",
            "reconciled_evidence": (
                f"0.4125 was an all-NaN F4 extraction bug; real RR features achieve {nh.loc['All_6130_Real_RR_LGBM','auroc']:.4f} LGBM (0.8419 Logistic). "
                f"beat_coverage is actually detected/expected beats at 60bpm (mean 1.010 Clean -> 1.565 N2). Automated waveform SQI detects N2 at {nh.loc['Automated_SQI_Detector_for_N2','auroc']:.4f} AUROC "
                f"and matches oracle abstention at 79.4% coverage ({nh.loc['Automated_Deployable_SQI_ECGFounder','auroc']:.4f} vs {nh.loc['Oracle_Keep_Clean_N1_ECGFounder','auroc']:.4f}). "
                "Only ECGFounder (+0.379 AUROC gain over random-init) and EchoNext (+0.052) beat random-init weights on P04; F2 (+0.007) and CSFM (-0.004) do not."
            ),
            "evidence_file": "V7R_noise_holter_reconciled.csv; V7R_fm_matched_evaluation.csv; V7R_fm_random_init_and_runtime.csv",
        },
        {
            "claim_id": "CLM-11",
            "source_report": "e04_noise_observability_report.md",
            "claim_summary": "Density is highly robust to short-episode omission and <60s gap bridging (MAE 0.00110 in IRIDIA+SHDB simulation).",
            "classification": "2. Supported only in a restricted population",
            "reconciled_evidence": (
                f"In LTAFDB (n={len(ltaf_drop)} density-defined recs), gap bridging <60s yields MAE={ltaf_drop['s2_bridge60s_dens_err'].abs().mean():.5f} "
                f"(max={ltaf_drop['s2_bridge60s_dens_err'].abs().max():.4f}), rising to MAE={ltaf_drop[ltaf_drop['n_brief_30s']>=20]['s2_bridge60s_dens_err'].abs().mean():.5f} "
                "among n=26 patients with >=20 brief (<30s) runs (>100x larger than in prolonged-dominant IRIDIA+SHDB)."
            ),
            "evidence_file": "V7R_ltafdb_dropout_sensitivity.csv",
        },
        {
            "claim_id": "CLM-12",
            "source_report": "V7-R Next-Direction Feasibility & Literature Check",
            "claim_summary": "High Charitos density conceals a 30-fold spread in transition instability across existing continuous cohorts; longitudinal stroke/cardiopathy outcomes remain absent.",
            "classification": "3. Exploratory and requiring validation",
            "reconciled_evidence": (
                "Among 118 independent multi-run patients with high Charitos density (>=0.80, median 0.9752), 55 have 2-5 runs (4.06 transitions/24h) while 29 have >=20 runs (122.61 transitions/24h, median 61 runs). "
                "Stroke/AF progression endpoints do not exist in IRIDIA-AF/LTAFDB/SHDB-AF, so clinical stroke confounding cannot be answered here."
            ),
            "evidence_file": "V7R_transition_instability_feasibility.csv",
        },
    ]

    # Check if V7R_fm_matched_evaluation.csv exists to include top FM summary claim
    fm_path = RES / "V7R_fm_matched_evaluation.csv"
    if fm_path.exists():
        fm = pd.read_csv(fm_path)
        p1_ef = fm[(fm["task"] == "P01_CleanAnnotated") & (fm["fm_model"] == "ECGFOUNDER") & (fm["classifier"] == "logistic")].iloc[0]
        p2_en = fm[(fm["task"] == "P02_Persistence60s") & (fm["fm_model"] == "ECHONEXT") & (fm["classifier"] == "logistic")].iloc[0]
        claims.append({
            "claim_id": "CLM-13",
            "source_report": "V7P_13stage_ablation_report.md",
            "claim_summary": "Adding frozen FM embeddings to canonical tabular history/density (S3c) improves longitudinal prediction (P01/P02/P03) under fixed models.",
            "classification": "2. Supported only in a restricted population",
            "reconciled_evidence": (
                f"Fixed-model matched evaluation (B=2,000): on P01_CleanAnnotated (which lacks F4 RR in S3c), ECGFOUNDER fusion yields +{p1_ef['delta_fusion_minus_tab']:.4f} logistic "
                f"(0.6125->0.6406, 95% CI [{p1_ef['boot_ci95_low']:+.4f}, {p1_ef['boot_ci95_high']:+.4f}], p={p1_ef['boot_p_b2000']:.4f}; still below CleanRR 0.6686/0.6689). "
                f"On P02_Persistence60s & P03_Survival15m logistic models, all FMs are null or significantly worse (P02 ECHONEXT Delta={p2_en['delta_fusion_minus_tab']:+.4f}, p={p2_en['boot_p_b2000']:.4f}; F2 Delta=-0.0047, p=0.010). "
                "LGBM P03 gains (+0.0089 ECGFOUNDER) only partly recover LGBM lag (0.8710 vs Logistic 0.8807). FMs dominate P04 morphology/noise (ECGFOUNDER 0.9453 PCA-32 / 0.9578 full), not longitudinal prognosis."
            ),
            "evidence_file": "V7R_fm_matched_evaluation.csv",
        })

    claims.append({
        "claim_id": "CLM-14",
        "source_report": "af_density_temporal_phenotypes_v7_master_report.md",
        "claim_summary": "IRIDIA-AF scarcity of <30s runs (only 2/388 runs vs 4,704/7,358 in LTAFDB) represents pure atrial biology rather than annotation-protocol/minimum-duration filtering.",
        "classification": "5. Unverifiable with available artifacts",
        "reconciled_evidence": "Annotation minimum-duration criteria in IRIDIA-AF HDF5 generation cannot be retrospectively verified from WFDB/HDF5 files alone; harmonization sensitivity bounds are required.",
        "evidence_file": "V7R_cohort_reconciliation.csv",
    })

    reg_df = pd.DataFrame(claims)
    reg_df.to_csv(RES / "V7R_claim_verification_registry.csv", index=False)
    print(f"Saved V7R_claim_verification_registry.csv ({len(reg_df)} claims)")
    return reg_df


def write_reports(reg_df):
    coh = pd.read_csv(RES / "V7R_cohort_reconciliation.csv")
    e01 = pd.read_csv(RES / "V7R_e01_reconciled.csv")
    e01_lopo = pd.read_csv(RES / "V7R_e01_lopo_sensitivity.csv")
    e01_nulls = pd.read_csv(RES / "V7R_e01_nulls_summary.csv")
    e02_panel = pd.read_csv(RES / "V7R_e02_matched_panel.csv")
    e02_rec = pd.read_csv(RES / "V7R_e02_reconciled.csv")
    e03_rec = pd.read_csv(RES / "V7R_e03_reconciled.csv")
    leak = pd.read_csv(RES / "V7R_feature_leakage_audit.csv")
    psup = pd.read_csv(RES / "V7R_patient_support.csv")
    abl = pd.read_csv(RES / "V7R_matched_ablation_results.csv")
    cstrat = pd.read_csv(RES / "V7R_cohort_stratified_performance.csv")
    alerts = pd.read_csv(RES / "V7R_p01_clean_alerts.csv")
    fm_params = pd.read_csv(RES / "V7R_fm_inventory_params.csv")
    fm_rnd = pd.read_csv(RES / "V7R_fm_random_init_and_runtime.csv")
    nh = pd.read_csv(RES / "V7R_noise_holter_reconciled.csv")
    ltaf_drop = pd.read_csv(RES / "V7R_ltafdb_dropout_sensitivity.csv")
    feas = pd.read_csv(RES / "V7R_transition_instability_feasibility.csv")
    fm_eval_path = RES / "V7R_fm_matched_evaluation.csv"
    fm_eval = pd.read_csv(fm_eval_path) if fm_eval_path.exists() else pd.DataFrame()

    def md_table(df):
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            vals = ["" if pd.isna(row[c]) else str(row[c]) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    # 1. V7R_retracted_claims.md
    retracted_md = f"""# AF-DENSITY V7-R: Retracted Claims & Superseding Evidence Ledger

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
     - `P01_CleanAnnotated`: Logistic $\\Delta = +0.0060$ (95% CI $[-0.0082, +0.0227]$, $p = 0.436$); LightGBM $\\Delta = -0.0031$ (95% CI $[-0.0155, +0.0087]$, $p = 0.622$).
     - `P02_Persistence60s`: Logistic $\\Delta = +0.0012$ (95% CI $[-0.0020, +0.0046]$, $p = 0.467$); LightGBM $\\Delta = +0.0013$ (95% CI $[-0.0053, +0.0088]$, $p = 0.778$).
     - `P03_Survival15m`: Logistic $\\Delta = +0.0005$ (95% CI $[-0.0013, +0.0024]$, $p = 0.665$); LightGBM $\\Delta = -0.0020$ (95% CI $[-0.0064, +0.0019]$, $p = 0.309$).
   - Furthermore, the detailed ablation report ([V7P_13stage_ablation_report.md](file:///home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/reports/V7P_13stage_ablation_report.md)) swapped stage meanings (`X02 -> X03` added episode architecture, while `X03 -> X04` added Charitos density).
3. **Retracted Multimodal Peak AUROC (0.842) & SetOperator Table (Lines 22, 110, 180–189, 210):**
   - *Retracted claim:* X12 achieves `0.842` AUROC on P01, `0.828` on P02, and SetOperator G1 achieves `0.814` P01 / `0.794` P02 / `0.738` C-index P03.
   - *Verified ground truth:* None of those numbers exist in `ablation_matrix_results.csv` or `setoperator_downstream_heads.csv`; `af_density_prediction_v7p_master_report.md` was written prior to the actual experimental runs.
4. **Retracted Bootstrap Resolution & X06->X07 RR Shortcut / P04 All-NaN Artifact:**
   - Original `run_13stage_ablations.py` used `n_bootstrap=500` (so `$p < 0.0001$` was impossible under simple tail-counting where `min_p = 2/501 = 0.00399`) and relied on `is_rr_valid` (`0.0` for all IRIDIA-AF, `1.0` for LTAFDB/SHDB) as an accidental cohort indicator while `P04 X07` (`0.4125`) ran on **all-NaN RR features**. With real RR features, `P04` RR LGBM reaches **`0.9023`** (`Logistic 0.8419`).

---

## 2. Fabricated / Contradicted V7 Master Report Claims (`af_density_temporal_phenotypes_v7_master_report.md`)

1. **Retracted Post-Termination Predominance & Null A 87.4% Claim (Lines 132–155):**
   - *Retracted claim:* E01 analyzed `553` prolonged episodes in LTAFDB+SHDB, found `[0,+5m]` RR = `4.82`, Post vs Pre RR = `2.59` ($p < 10^{-7}$), and Null A `p < 0.001` in `87.4%` of patients.
   - *Verified truth (`V7R_e01_reconciled.csv`, `V7R_e01_nulls_summary.csv`):* Only `304` prolonged episodes (`202` LTAFDB + `102` SHDB; `249` IRIDIA excluded) entered E01. Pooled risk-time `Pre vs Remote RR = 6.236` ($p = 1.59\\times 10^{-7}$), `Post vs Remote RR = 4.390` ($p = 7.57\\times 10^{-5}$), and **direct `Post vs Pre RR = 0.704` ($p = 0.1758$, NON-SIGNIFICANT)**. In Null A LTAFDB (`n=33`), observed counts are *below* Null A expectations (`Obs/Exp = 0.905` Pre, `0.674` Post); in `Null B`, `p < 0.05` holds in `0.00%` of recordings.
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
"""
    (REP / "V7R_retracted_claims.md").write_text(retracted_md)
    (ROOT_REP / "V7R_retracted_claims.md").write_text(retracted_md)

    # 2. V7R_remaining_uncertainties.md
    unc_md = """# AF-DENSITY V7-R: Remaining Uncertainties & Methodological Constraints

1. **Annotation Harmonization Gap Across Cohorts:**
   - `IRIDIA-AF` contains only `2` runs `<30s` out of `388` runs (`0.5%`), vs `310 / 794` (`39.0%`) in `SHDB-AF` and `4,704 / 7,358` (`63.9%`) in `LTAFDB`. Retrospective files do not document whether IRIDIA-AF annotation rules filtered sub-30s events; therefore raw cross-cohort comparisons of brief-run incidence remain confounded by annotation convention.
2. **Physical-Time vs Non-AF Risk-Time Distortion (E01):**
   - Under pooled `risk_time_h`, `Pre vs Remote RR = 6.236` ($p=1.59\\times 10^{-7}$), but under pooled `physical_time_h`, `Pre vs Remote RR = 1.856` ($p=0.102$, NS). In LTAFDB alone, physical-time RRs are $<1.0$ (`0.831` Pre, `0.544` Post) because peri-prolonged windows contain dense adjacent AF, compressing the non-AF denominator and inflating conditional rates (`27.32/h`) while conditional Null A (`Obs/Exp = 0.905` Pre, `0.674` Post) shows no excess above boundary-constrained random placement. Only SHDB-AF exhibits genuine clustering in both physical time (`~11.1x`) and Null A (`5.7x–6.0x`).
3. **Between-Patient Stratification vs Within-Patient Episode Prediction (P02):**
   - While P02 `S2_Architecture` achieves strong within-cohort (`0.8664` LTAFDB, `0.7173` IRIDIA) and LOCO transfer (`0.8548` LTAFDB, `0.7664` IRIDIA, `0.7283` SHDB) AUROCs, **patient-level equal-weight AUROC across the `n=91` patients who exhibit both brief and prolonged runs is `0.4789` (Logistic) / `0.5245` (LightGBM)**. Thus 1–6h episode architecture predominantly captures patient/regime state rather than episode-to-episode triggers within mixed-phenotype patients.
4. **Underpowered Structural Remodeling Cohort (E03):**
   - Only `13–28` patient-deduplicated multi-run PAF patients in SHDB-AF have defined 12h density and matched echocardiography within `30d–Unrestricted` windows. Null findings (`+0.188 mm`, 95% CI `[-2.298, +2.270]`) reflect severe underpowering and absence of LA volume index, strain, or LGE-MRI fibrosis data — not proof of zero biological relationship.
5. **Lack of Longitudinal Clinical Outcomes:**
   - Neither IRIDIA-AF, SHDB-AF, nor LTAFDB includes prospective ischemic stroke, heart failure hospitalization, or AF progression follow-up endpoints; questions on clinical thromboembolic risk remain unanswerable with these cohorts.
6. **SHDB-AF Repeated Subjects & 30 Unannotated Recordings:**
   - Raw `AdditionalData.csv` has `128` recordings for `122` unique subjects (`6` repeat Holters) and `30` completely unannotated recordings (`719.88h` of unknown rhythm). Any analysis must enforce both annotation filtering and patient deduplication / clustering.
7. **Foundation Model Input/Domain Mismatches & Lack of Pretraining Advantage (Excluding ECGFounder/EchoNext):**
   - Zero-padding 1-lead Holters into 12-lead architectures (`ECGFM_KED`, `EchoNext`, `DINO-SR`, `JEPA`) or assuming Lead I/II alignment for uncalibrated Holter leads (`CSFM`, `F0/F2`, `G1`) introduces channel mismatch; `F2` (`+0.0073` / `-0.0150`) and `CSFM` (`-0.0041` / `-0.0259`) fail to outperform untrained random-init weights on P04.
"""
    (REP / "V7R_remaining_uncertainties.md").write_text(unc_md)
    (ROOT_REP / "V7R_remaining_uncertainties.md").write_text(unc_md)

    # 3. V7R_master_scientific_report.md
    master_md = f"""# AF-DENSITY V7-R: Independent Numerical Reconciliation & Scientific Adjudication Report

**Directive:** AF-DENSITY V7-R (Priorities 1–10)  
**Audit Standard:** Zero reliance on historical PASS certificates, executive summaries, or V7 figures. Every number below is independently reconstructed from raw WFDB/HDF5/CSV/Parquet assets and fixed-model 5-fold patient-disjoint CV with $B=2,000$ patient-clustered bootstraps (`min_attainable_p = 2/2001 = 0.001`).

---

## Executive Adjudication Summary

Independent reconstruction overturns the overclaimed V7 and V7-P executive conclusions while isolating three genuine, reproducible scientific signals:

1. **Historical episode architecture decisively outperforms cumulative AF burden alone on early persistence (`P02`: Logistic $+0.0742$, $p=0.0010$; LightGBM $+0.0224$, $p=0.0010$) and dynamic termination (`P03`: Logistic $+0.0274$, $p=0.0010$), with zero future leakage (`0.0` diff across 150 adversarial future-deletion tests) — yet Charitos continuous density adds ZERO statistically significant increment beyond architecture on any task (`P01` $p=0.436$, `P02` $p=0.467$, `P03` $p=0.665$).**
2. **Pre-onset ventricular dynamics (`F4` Clean RR) genuinely improve imminent sustained AF forecasting (`P01` LGBM `0.5710 -> 0.6689`, $\\Delta=+0.0979$, $p=0.0010$ within valid `LTAFDB+SHDB` records without cohort shortcuts), whereas continuous clinical early warning remains non-viable (`9.79%` event sensitivity at `16.56` false warnings/patient-day at thr `0.05`).**
3. **High Charitos density conceals massive transition heterogeneity (`V7R_transition_instability_feasibility.csv`): among `118` multi-run patients with high Charitos density ($d \\ge 0.80$, median `0.9752`), transition rates diverge 30-fold (`4.06/24h` in `55` low-fragmentation patients vs `122.61/24h` in `29` storm patients), establishing *Temporal concentration vs transition instability* as an empirically populated, mathematically sharp research question.**

---

## Priority 1 — Reconciled Dataset Inventory

{md_table(coh)}

- **349 vs 379 recordings & 10,953h vs 11,701h resolved:** SHDB-AF contains `128` raw recordings (`122` unique patients, `6` duplicates), of which `98` are annotated (`93` unique pts, `72` AF+) and `30` (`719.88h`) have no annotations (`rhythm_known=False`). Plus `25.68h` unannotated prefix in LTAFDB and `3.42h` calibration gaps in IRIDIA-AF yield `748.99h` total unknown time (`11,701.45h` raw - `10,952.98h` known annotated).
- **8,540 vs 8,582 resolved:** Exact sum of raw transitions across all annotated recordings equals **`8,540` canonical runs** (`388` IRIDIA + `7,358` LTAFDB + `794` SHDB; `24` left-censored in IRIDIA, `43` right-censored in LTAFDB). The historical `8,582` figure was an artifact in `af_density_continuity_v5`.

---

## Priority 2 — E01 Temporal Neighbourhood Reconciliation

- **`553` vs `304` index episodes:** Total dataset has `553` runs $\\ge 30$m (`249` IRIDIA + `202` LTAFDB + `102` SHDB). E01 analyzed `304` (`202` LTAFDB + `102` SHDB) because IRIDIA-AF contains only `2` runs `<30s`.

{md_table(e01)}

### E01 LOPO Sensitivity (Excluding Prolific Brief-Run LTAFDB Patients)
{md_table(e01_lopo)}

### E01 Conditional Null Models Summary
{md_table(e01_nulls)}

---

## Priority 3 — E02 Matched Panel, Density Fill Audit & GAM OOF Reconciliation

{md_table(e02_panel)}

{md_table(e02_rec)}

- **Population A (`6,683` episodes):** `245` (`3.67%`) had undefined 6h density improperly filled as `0.0` in `e02_statistical_models.py:232`. Restricting strictly to defined complete density (`n=6,438`, `145` pts), Cox PH density is null pooled (`HR=1.1382`, $p=0.514$), null in LTAFDB (`p=0.195`) and SHDB (`p=0.780`), and borderline only in IRIDIA (`n=140`, `HR=4.0896`, $p=0.0475$).

---

## Priority 4 — E03 SHDB-AF Patient-Deduplicated Structural Analysis

{md_table(e03_rec)}

---

## Priorities 5, 6 & 7 — Leakage Audit, Patient Support, Clean Matched Ablations, Stratified Performance, Alerts & FM Controls

### Future-Deletion Leakage Audit
{md_table(leak)}

### Cohort Patient Support
{md_table(psup)}

### Clean Matched Ablations & B=2,000 Contrasts (Fixed Models)
{md_table(abl)}

### Cohort-Stratified & Leave-One-Cohort-Out (LOCO) Transfer Performance
{md_table(cstrat)}

### Clean Annotated P01 Clinical Alert Benchmarks (`90,057` Non-AF Landmarks)
{md_table(alerts)}

### Foundation Model Parameter Inventory, Random-Init Control & A100 Throughput
{md_table(fm_params)}

{md_table(fm_rnd)}
{"" if fm_eval.empty else chr(10) + "### Matched Fixed-Model Foundation Model Evaluation (10s Causal Window, PCA-32)" + chr(10) + md_table(fm_eval[fm_eval["fm_model"].isin(["ECGFOUNDER", "ECHONEXT", "ECGFM_KED", "F2", "F0", "G1", "CONTROL", "CSFM"])])}

---

## Priority 8 — Noise Holter & LTAFDB Dropout Perturbation Reconciliation

{md_table(nh)}

- **`6,255` vs `6,130` resolved:** `RECORDS` index file lists `6,255` lines (`125` segments `233-*` to `240-*` missing from archive), while disk `.hea`/P04 has `6,130` excerpts (`Clean=2,665`, `N1=2,202`, `N2=1,263`).
- **LTAFDB Perturbation Simulation (`n={len(ltaf_drop)}` density-defined):** Omit `<30s` `MAE={ltaf_drop['s1a_omit30s_dens_err'].abs().mean():.5f}` (`max={ltaf_drop['s1a_omit30s_dens_err'].abs().max():.4f}`, `1` undefined); Omit `<10s` `MAE={ltaf_drop['s1b_omit10s_dens_err'].abs().mean():.5f}`; Bridge `<60s` gaps `MAE={ltaf_drop['s2_bridge60s_dens_err'].abs().mean():.5f}` (`max={ltaf_drop['s2_bridge60s_dens_err'].abs().max():.4f}`), rising to `MAE={ltaf_drop[ltaf_drop['n_brief_30s']>=20]['s2_bridge60s_dens_err'].abs().mean():.5f}` in `n=26` patients with `>=20` brief runs.

---

## Priority 9 — Claim Verification Registry Summary

{md_table(reg_df)}

---

## Priority 10 — Empirical Feasibility of Candidate Next Direction (*Temporal Concentration vs Transition Instability*)

{md_table(feas)}
"""
    (REP / "V7R_master_scientific_report.md").write_text(master_md)
    (ROOT_REP / "V7R_master_scientific_report.md").write_text(master_md)
    # Mirror key CSVs to root reports/ with V7R_ prefix for discoverability without touching historical files
    for fname in [
        "V7R_cohort_reconciliation.csv",
        "V7R_claim_verification_registry.csv",
        "V7R_feature_leakage_audit.csv",
        "V7R_patient_support.csv",
        "V7R_matched_ablation_results.csv",
        "V7R_cohort_stratified_performance.csv",
    ]:
        (ROOT_REP / fname).write_text((RES / fname).read_text())
    print("Saved V7R_retracted_claims.md, V7R_remaining_uncertainties.md, V7R_master_scientific_report.md, and mirrored Priority 9 CSVs to reports/")


if __name__ == "__main__":
    reg_df = build_claim_registry()
    write_reports(reg_df)
    write_reports(reg_df)
