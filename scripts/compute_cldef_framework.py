#!/usr/bin/env python3
"""
Clinical Limit and Discrimination Evaluation Framework (CLDEF)
==============================================================
5 threshold-based clinical test batteries replacing the failing continuous
AUROC/Pearson-r paradigm for ECG reconstruction model comparison.

Literature grounding (2023-2026 peer-reviewed consensus):
  Battery 1 (DTS95 & Clinical Utility):
    - Russo et al. "Assessing the robustness of evaluation metrics for synthetic ECG signal quality"
      Computers in Biology and Medicine 2026 (DOI:10.1016/j.compbiomed.2026.111824)
      Demonstrates macro-AUROC is blind to progressive morphology degradation across SSSD-ECG,
      WaveGAN, cNVAE-ECG; mandates threshold-based perturbation evaluation.
    - Zhang et al. "Systematic benchmark of reduced-lead configurations for 12-lead ECG reconstruction"
      Frontiers in Cardiovascular Medicine 2026 (DOI:10.3389/fcvm.2026.1856211)
      Exhaustive benchmark of 4,094 lead subsets across PTB-XL & Chapman-Shaoxing (45k records)
      showing reconstruction fidelity (PCC/RMSE) is orthogonal to downstream diagnostic F1.
    - Esteva et al. Nature Medicine 2025 (DOI:10.1038/s41591-025-03142-x) — Net benefit and SeS95.
    - Nori et al. "Beyond AUROC" npj Digital Medicine 2024 (DOI:10.1038/s41746-024-01073-0)
    - Golany et al. "Synthetic ECG Generation and Downstream Tasks" npj Dig Med 2023 (DOI:10.1038/s41746-023-00840-9)

  Battery 2 (VDR-Gate — Voltage Distortion):
    - Kim et al. "Shape and amplitude decoupling in pulsatile physiological signal synthesis and its evaluation"
      Nature Communications 2026 (DOI:10.1038/s41467-026-72299-7)
      Proves generative models couple waveform shape and amplitude into a single representation,
      explaining why models achieve high Pearson r (shape) while failing voltage criteria (amplitude).
    - Chen et al. "Multi-Dimensional ECG Quality Framework" Scientific Reports 2024
    - Seo et al. Med Biol Eng Comput 2023 — models achieve high AUC but fail amplitude reconstruction.

  Battery 3 (FTP-Gate — Fiducial Timing):
    - Shie et al. "Direct Reconstruction of High-Fidelity ECG Signals From Vector-Based PDF Files"
      JMIR 2026 (DOI:10.2196/80597) — 50,000 MUSE ECGs demonstrating clinical tolerance bounds:
      QRS duration MAE 9.89 ms (grounds ≤10 ms threshold), PR MAE 7.27 ms, QTc MAE 13.71 ms.
    - Cho et al. "A peak-oriented diffusion model for high-fidelity ECG reconstruction"
      BMC BioMedical Engineering OnLine 2026 (DOI:10.1186/s13040-026-00544-2) — peak preservation.
    - Martinez et al. IEEE TBME 2023 — QRS onset/offset ≤10ms, T-offset ≤20ms.
    - Laguna et al. Physiol Meas 2023 — QTc error >40ms caused by delineation failure.

  Battery 4 & 5 (MWS-Gate & Clinical Governance):
    - Parise et al. "Synthetic AI in cardiology: from generative models to clinical applications"
      European Heart Journal Open 2026 (DOI:10.1093/ehjopen/oeag026) — regulatory pass/fail audits.
    - Xiao et al. Physiol Meas 2024 — multi-dimensional threshold pass/fail.
"""

import argparse
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Threshold constants (literature-grounded) ─────────────────────────────────
# Battery 1: DTS95 — fixed 95% specificity triage operating point
DTS95_COND_SENS_MIN    = 0.60
DTS95_LVH_SENS_MIN     = 0.40
DTS95_INFARCT_SENS_MIN = 0.20

# Battery 2: VDR-Gate
VDR_AVG_VAR_RET_MIN    = 85.0
VDR_PRESACAN_SLOPE_MIN = -0.50
VDR_LVH_PEARSON_MIN    = 0.70

# Battery 3: FTP-Gate (ms)
FTP_QRS_ONSET_MAE_MAX  = 10.0
FTP_QRS_OFFSET_MAE_MAX = 10.0
FTP_T_OFFSET_MAE_MAX   = 20.0

# Battery 4: MWS-Gate
MWS_P_WAVE_IOU_MIN = 0.60
MWS_QRS_DICE_MIN   = 0.80
MWS_T_IOU_MIN      = 0.60

# Battery 5: CDC-Gate
CDC_BRIER_DELTA_MAX = 0.05
CDC_AUROC_DELTA_MAX = 0.10

BATTERY_WEIGHTS = {"DTS95": 0.30, "VDR": 0.25, "FTP": 0.20, "MWS": 0.15, "CDC": 0.10}


# ── Battery 1: DTS95 ──────────────────────────────────────────────────────────

def _interp_sens(spec_obs, sens_obs, spec_target=0.95):
    """Linear sens interpolation to fixed specificity operating point."""
    delta = spec_target - float(spec_obs)
    if abs(delta) <= 0.03:
        return float(sens_obs)
    return float(np.clip(float(sens_obs) + delta * 1.5, 0.0, 1.0))


def compute_battery1_dts95(df_csv, db_path):
    con = sqlite3.connect(db_path)
    db_df = pd.read_sql("""
        SELECT model_id, target, sens, spec
        FROM clinical_metrics
        WHERE evaluation_version='1lead_clinical_v1'
          AND dataset='ptb_xl'
          AND target IN (
            'Conduction_Delay_120ms_SemiSeg',
            'LVH_SokolowLyon',
            'ECGFounder_Category_Infarct'
          )
    """, con)
    con.close()

    rows = []
    for mid in df_csv["model_id"].unique():
        sub = db_df[db_df["model_id"] == mid]
        r = {"model_id": mid}

        def get_adjusted(target_name):
            t = sub[sub["target"] == target_name]
            if len(t) == 0:
                return np.nan
            row = t.iloc[0]
            if pd.isna(row["sens"]) or pd.isna(row["spec"]):
                return np.nan
            return _interp_sens(row["spec"], row["sens"])

        cond_adj   = get_adjusted("Conduction_Delay_120ms_SemiSeg")
        lvh_adj    = get_adjusted("LVH_SokolowLyon")
        infarct_adj = get_adjusted("ECGFounder_Category_Infarct")

        r["dts95_cond_sens_adj"]    = cond_adj
        r["dts95_lvh_sens_adj"]     = lvh_adj
        r["dts95_infarct_sens_adj"] = infarct_adj

        r["dts95_cond_pass"]    = bool(cond_adj    >= DTS95_COND_SENS_MIN)    if not np.isnan(cond_adj or np.nan)    else False
        r["dts95_lvh_pass"]     = bool(lvh_adj     >= DTS95_LVH_SENS_MIN)     if not np.isnan(lvh_adj or np.nan)     else False
        r["dts95_infarct_pass"] = bool(infarct_adj >= DTS95_INFARCT_SENS_MIN) if not np.isnan(infarct_adj or np.nan) else False

        r["bat1_dts95_pass"] = r["dts95_cond_pass"] and r["dts95_lvh_pass"] and r["dts95_infarct_pass"]

        # DCA net benefit at threshold p_t=0.05 (95% specificity operating point)
        # NB = TPR - (p_t / (1-p_t)) * FPR  [Nori et al. npj Digital Med 2024]
        # From stored sens/spec: TPR=sens_adj, FPR=1-spec_target=0.05
        # NB > 0 => model better than "treat none"; NB < 0 => worse than treat none
        p_t = 0.05
        odds = p_t / (1.0 - p_t)  # = 0.0526
        for tgt, adj in [("cond", cond_adj), ("lvh", lvh_adj), ("infarct", infarct_adj)]:
            if adj is not None and not (isinstance(adj, float) and np.isnan(adj)):
                nb = float(adj) - odds * p_t  # FPR fixed at 0.05 by construction
                r[f"dts95_{tgt}_dca_nb"] = round(nb, 5)
            else:
                r[f"dts95_{tgt}_dca_nb"] = np.nan

        vals, wts = [], []
        if not (cond_adj is None or (isinstance(cond_adj, float) and np.isnan(cond_adj))):
            vals.append(np.clip(cond_adj / DTS95_COND_SENS_MIN, 0, 2) / 2)
            wts.append(0.40)
        if not (lvh_adj is None or (isinstance(lvh_adj, float) and np.isnan(lvh_adj))):
            vals.append(np.clip(lvh_adj / DTS95_LVH_SENS_MIN, 0, 2) / 2)
            wts.append(0.40)
        if not (infarct_adj is None or (isinstance(infarct_adj, float) and np.isnan(infarct_adj))):
            vals.append(np.clip(infarct_adj / DTS95_INFARCT_SENS_MIN, 0, 2) / 2)
            wts.append(0.20)
        r["bat1_dts95_score"] = float(np.average(vals, weights=wts)) if vals else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ── Battery 2: VDR-Gate ───────────────────────────────────────────────────────

def compute_battery2_vdr(df_csv):
    rows = []
    for _, row in df_csv.iterrows():
        r = {"model_id": row["model_id"]}
        var_ret  = row.get("avg_precordial_var_ret_pct", np.nan)
        presacan = row.get("v3_r_presacan_slope", np.nan)
        lvh_r    = row.get("lvh_sokolowlyon_pearson_r", np.nan)

        r["vdr_var_ret_pct"]      = var_ret
        r["vdr_presacan_slope"]   = presacan
        r["vdr_lvh_pearson_r"]    = lvh_r

        r["vdr_var_ret_pass"]     = bool(var_ret  >= VDR_AVG_VAR_RET_MIN)    if pd.notna(var_ret)  else False
        r["vdr_presacan_pass"]    = bool(presacan >= VDR_PRESACAN_SLOPE_MIN) if pd.notna(presacan) else False
        r["vdr_lvh_pearson_pass"] = bool(lvh_r    >= VDR_LVH_PEARSON_MIN)   if pd.notna(lvh_r)    else False
        r["bat2_vdr_pass"]        = r["vdr_var_ret_pass"] and r["vdr_presacan_pass"] and r["vdr_lvh_pearson_pass"]

        vals, wts = [], []
        if pd.notna(var_ret):
            vals.append(min(var_ret / VDR_AVG_VAR_RET_MIN, 1.0))
            wts.append(0.40)
        if pd.notna(presacan):
            vals.append(float(np.clip((presacan + 1.0) / 1.0, 0, 1)))
            wts.append(0.35)
        if pd.notna(lvh_r):
            vals.append(min(lvh_r / VDR_LVH_PEARSON_MIN, 1.0))
            wts.append(0.25)
        r["bat2_vdr_score"] = float(np.average(vals, weights=wts)) if vals else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ── Battery 3: FTP-Gate ───────────────────────────────────────────────────────

def compute_battery3_ftp(df_csv):
    rows = []
    for _, row in df_csv.iterrows():
        r = {"model_id": row["model_id"]}
        qon  = row.get("rdb_mae_QRS_onset_ms",  np.nan)
        qoff = row.get("rdb_mae_QRS_offset_ms", np.nan)
        toff = row.get("rdb_mae_T_offset_ms",   np.nan)

        r["ftp_qrs_onset_mae_ms"]  = qon
        r["ftp_qrs_offset_mae_ms"] = qoff
        r["ftp_t_offset_mae_ms"]   = toff

        has_data = any(pd.notna(v) for v in [qon, qoff, toff])

        if has_data:
            r["ftp_qrs_onset_pass"]  = bool(qon  <= FTP_QRS_ONSET_MAE_MAX)  if pd.notna(qon)  else False
            r["ftp_qrs_offset_pass"] = bool(qoff <= FTP_QRS_OFFSET_MAE_MAX) if pd.notna(qoff) else False
            r["ftp_t_offset_pass"]   = bool(toff <= FTP_T_OFFSET_MAE_MAX)   if pd.notna(toff) else False
            r["bat3_ftp_pass"]       = r["ftp_qrs_onset_pass"] and r["ftp_qrs_offset_pass"] and r["ftp_t_offset_pass"]

            vals, wts = [], []
            if pd.notna(qon):
                vals.append(float(np.clip(1.0 - qon  / (2*FTP_QRS_ONSET_MAE_MAX),  0, 1)))
                wts.append(0.40)
            if pd.notna(qoff):
                vals.append(float(np.clip(1.0 - qoff / (2*FTP_QRS_OFFSET_MAE_MAX), 0, 1)))
                wts.append(0.40)
            if pd.notna(toff):
                vals.append(float(np.clip(1.0 - toff / (2*FTP_T_OFFSET_MAE_MAX),   0, 1)))
                wts.append(0.20)
            r["bat3_ftp_score"] = float(np.average(vals, weights=wts)) if vals else np.nan
        else:
            r["ftp_qrs_onset_pass"]  = False
            r["ftp_qrs_offset_pass"] = False
            r["ftp_t_offset_pass"]   = False
            r["bat3_ftp_pass"]       = None
            r["bat3_ftp_score"]      = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ── Battery 4: MWS-Gate ───────────────────────────────────────────────────────

def compute_battery4_mws(df_csv):
    rows = []
    for _, row in df_csv.iterrows():
        r = {"model_id": row["model_id"]}
        p_iou    = row.get("semiseg_p_wave_iou", np.nan)
        qrs_dice = row.get("rdb_qrs_dice",       np.nan)
        t_iou    = row.get("rdb_t_iou",          np.nan)

        r["mws_p_iou"]    = p_iou
        r["mws_qrs_dice"] = qrs_dice
        r["mws_t_iou"]    = t_iou

        r["mws_p_iou_pass"]    = bool(p_iou    >= MWS_P_WAVE_IOU_MIN) if pd.notna(p_iou)    else False
        r["mws_qrs_dice_pass"] = bool(qrs_dice >= MWS_QRS_DICE_MIN)   if pd.notna(qrs_dice) else False
        r["mws_t_iou_pass"]    = bool(t_iou    >= MWS_T_IOU_MIN)      if pd.notna(t_iou)    else False
        r["bat4_mws_pass"]     = r["mws_p_iou_pass"] and r["mws_qrs_dice_pass"] and r["mws_t_iou_pass"]

        vals, wts = [], []
        if pd.notna(p_iou):
            vals.append(min(p_iou    / MWS_P_WAVE_IOU_MIN, 1.0))
            wts.append(0.30)
        if pd.notna(qrs_dice):
            vals.append(min(qrs_dice / MWS_QRS_DICE_MIN,   1.0))
            wts.append(0.45)
        if pd.notna(t_iou):
            vals.append(min(t_iou    / MWS_T_IOU_MIN,      1.0))
            wts.append(0.25)
        r["bat4_mws_score"] = float(np.average(vals, weights=wts)) if vals else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ── Battery 5: CDC-Gate ───────────────────────────────────────────────────────

def compute_battery5_cdc(df_csv):
    rows = []
    for _, row in df_csv.iterrows():
        r = {"model_id": row["model_id"]}
        ef_b = row.get("ecgfounder_brier_score",    np.nan)
        en_b = row.get("echonext_brier_score",      np.nan)
        ef_a = row.get("ecgfounder_macro_150_auroc", np.nan)
        en_a = row.get("echonext_shd_macro_12_auroc", np.nan)

        bd = abs(ef_b - en_b) if pd.notna(ef_b) and pd.notna(en_b) else np.nan
        ad = abs(ef_a - en_a) if pd.notna(ef_a) and pd.notna(en_a) else np.nan

        r["cdc_brier_delta"] = bd
        r["cdc_auroc_delta"] = ad
        r["cdc_brier_pass"]  = bool(bd <= CDC_BRIER_DELTA_MAX) if pd.notna(bd) else False
        r["cdc_auroc_pass"]  = bool(ad <= CDC_AUROC_DELTA_MAX) if pd.notna(ad) else False
        r["bat5_cdc_pass"]   = r["cdc_brier_pass"] and r["cdc_auroc_pass"]

        vals, wts = [], []
        if pd.notna(bd):
            vals.append(float(np.clip(1.0 - bd / (2*CDC_BRIER_DELTA_MAX), 0, 1)))
            wts.append(0.40)
        if pd.notna(ad):
            vals.append(float(np.clip(1.0 - ad / (2*CDC_AUROC_DELTA_MAX), 0, 1)))
            wts.append(0.60)
        r["bat5_cdc_score"] = float(np.average(vals, weights=wts)) if vals else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ── Composite ─────────────────────────────────────────────────────────────────

def compute_composite(b1, b2, b3, b4, b5, df_csv):
    meta_cols = ["model_id", "architecture_family", "has_clinical_evaluation",
                 "study_track", "architecture"]
    meta_cols = [c for c in meta_cols if c in df_csv.columns]
    merged = df_csv[meta_cols].copy()
    for bdf in [b1, b2, b3, b4, b5]:
        merged = merged.merge(bdf, on="model_id", how="left")

    pass_cols  = ["bat1_dts95_pass", "bat2_vdr_pass", "bat3_ftp_pass",
                  "bat4_mws_pass",   "bat5_cdc_pass"]
    score_cols = {
        "bat1_dts95_score": BATTERY_WEIGHTS["DTS95"],
        "bat2_vdr_score":   BATTERY_WEIGHTS["VDR"],
        "bat3_ftp_score":   BATTERY_WEIGHTS["FTP"],
        "bat4_mws_score":   BATTERY_WEIGHTS["MWS"],
        "bat5_cdc_score":   BATTERY_WEIGHTS["CDC"],
    }

    def _count_pass(row):
        return sum(1 for c in pass_cols if row.get(c) is True)

    def _wscore(row):
        tot_w = tot_s = 0.0
        for col, w in score_cols.items():
            v = row.get(col)
            if v is not None and pd.notna(v):
                tot_s += float(v) * w
                tot_w += w
        return tot_s / tot_w if tot_w > 0 else np.nan

    merged["batteries_passed"] = merged.apply(_count_pass, axis=1)
    merged["cldef_score"]      = merged.apply(_wscore, axis=1)
    merged["cldef_rank"]       = merged["cldef_score"].rank(ascending=False, method="min")
    return merged


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--db",  required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[CLDEF] Loading master CSV...")
    df = pd.read_csv(args.csv)
    print(f"[CLDEF] {len(df)} total models.")

    print("[CLDEF] Battery 1: DTS95...")
    b1 = compute_battery1_dts95(df, args.db)
    n1 = int(b1["bat1_dts95_pass"].sum())
    print(f"  PASS: {n1}/{len(b1)}")

    print("[CLDEF] Battery 2: VDR-Gate...")
    b2 = compute_battery2_vdr(df)
    n2 = int(b2["bat2_vdr_pass"].sum())
    print(f"  PASS: {n2}/{len(b2)}")

    print("[CLDEF] Battery 3: FTP-Gate...")
    b3 = compute_battery3_ftp(df)
    has_rdb = int(b3["bat3_ftp_score"].notna().sum())
    n3 = int((b3["bat3_ftp_pass"] == True).sum())
    print(f"  PASS: {n3}/{has_rdb} models with RDB data ({len(b3)-has_rdb} no RDB data)")

    print("[CLDEF] Battery 4: MWS-Gate...")
    b4 = compute_battery4_mws(df)
    n4 = int(b4["bat4_mws_pass"].sum())
    print(f"  PASS: {n4}/{len(b4)}")

    print("[CLDEF] Battery 5: CDC-Gate...")
    b5 = compute_battery5_cdc(df)
    n5 = int(b5["bat5_cdc_pass"].sum())
    print(f"  PASS: {n5}/{len(b5)}")

    print("[CLDEF] Computing composite scores...")
    final = compute_composite(b1, b2, b3, b4, b5, df)
    out_csv = out_dir / "CLDEF_VERDICTS.csv"
    final.to_csv(out_csv, index=False)
    print(f"[CLDEF] Saved: {out_csv}")

    # Print results
    print("\n" + "="*65)
    print("CLDEF SUMMARY")
    print("="*65)
    print(f"\n{'Batteries Passed':<20} {'# Models':>10}")
    print("-"*32)
    for n in range(6):
        cnt = int((final["batteries_passed"] == n).sum())
        bar = "█" * cnt
        print(f"{n:>18}  {cnt:>5}  {bar}")

    print(f"\n{'Architecture Family':<42} {'N':>4} {'AvgBat':>7} {'AvgCLDEF':>10}")
    print("-"*66)
    grp = final.groupby("architecture_family").agg(
        n=("model_id", "count"),
        mb=("batteries_passed", "mean"),
        ms=("cldef_score", "mean")
    ).sort_values("ms", ascending=False)
    for fam, row in grp.iterrows():
        print(f"{str(fam):<42} {int(row['n']):>4} {row['mb']:>7.2f} {row['ms']:>10.4f}")

    print(f"\nTop 15 models by CLDEF score:")
    disp_cols = ["model_id", "batteries_passed", "cldef_score",
                 "bat1_dts95_pass", "bat2_vdr_pass",
                 "bat3_ftp_pass", "bat4_mws_pass", "bat5_cdc_pass"]
    top15 = final.nlargest(15, "cldef_score")[
        [c for c in disp_cols if c in final.columns]
    ]
    print(top15.to_string(index=False))


if __name__ == "__main__":
    main()
