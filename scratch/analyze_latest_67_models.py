import sqlite3, re
from collections import defaultdict
import numpy as np
import pandas as pd

conn = sqlite3.connect("results/clinical_biomarkers_multids/clinical_metrics.db")
cur = conn.cursor()

cur.execute("""
    SELECT model_id, dataset, target, auroc, auprc, f1, mae, pearson_r, r2, bland_bias
    FROM clinical_metrics
    WHERE evaluation_version = 'missing_leads_v2';
""")
rows = cur.fetchall()

data = defaultdict(lambda: defaultdict(dict))
for row in rows:
    mid, ds, target, auroc, auprc, f1, mae, pr, r2, bias = row
    data[mid][ds][target] = {
        "auroc": auroc, "auprc": auprc, "f1": f1, "mae": mae, "pearson_r": pr, "r2": r2, "bias": bias
    }

mids = sorted(data.keys())

parsed = []
for mid in mids:
    if mid == "reference":
        continue
    ptb = data[mid].get("ptb_xl", {})
    if not ptb:
        continue
    
    macro_auroc = ptb.get("ECGFounder_Macro_150", {}).get("auroc", 0.0)
    macro_auprc = ptb.get("ECGFounder_Macro_150", {}).get("auprc", 0.0)
    delin_cov = ptb.get("Delineation_Missing_Lead_Coverage", {}).get("mae", 0.0)
    
    # QRS & LVH metrics
    qrs_r = ptb.get("QRS_Overall", {}).get("pearson_r", 0.0)
    qrs_mae = ptb.get("QRS_Overall", {}).get("mae", 0.0)
    qrs_bias = ptb.get("QRS_Overall", {}).get("bias", 0.0)
    
    lvh_r = ptb.get("LVH_SokolowLyon", {}).get("pearson_r", 0.0)
    lvh_mae = ptb.get("LVH_SokolowLyon", {}).get("mae", 0.0)
    
    # Signal fidelity across 9 missing leads (III, aVR, aVL, aVF, V1, V3, V4, V5, V6)
    missing_leads = ['Signal_Lead_III', 'Signal_Lead_aVR', 'Signal_Lead_aVL', 'Signal_Lead_aVF', 
                     'Signal_Lead_V1', 'Signal_Lead_V3', 'Signal_Lead_V4', 'Signal_Lead_V5', 'Signal_Lead_V6']
    sig_r_list = [ptb.get(l, {}).get("pearson_r", 0.0) for l in missing_leads if l in ptb]
    mean_signal_r = float(np.mean(sig_r_list)) if sig_r_list else 0.0
    
    sig_mae_list = [ptb.get(l, {}).get("mae", 0.0) for l in missing_leads if l in ptb]
    mean_signal_mae = float(np.mean(sig_mae_list)) if sig_mae_list else 0.0

    # Specific Critical Pathologies
    afib_auroc = ptb.get("ECGFounder_ATRIAL_FIBRILLATION", {}).get("auroc", 0.0)
    lbbb_auroc = ptb.get("ECGFounder_LEFT_BUNDLE_BRANCH_BLOCK", {}).get("auroc", 0.0)
    rbbb_auroc = ptb.get("ECGFounder_RIGHT_BUNDLE_BRANCH_BLOCK", {}).get("auroc", 0.0)
    st_widening = ptb.get("ECGFounder_WITH_QRS_WIDENING", {}).get("auroc", 0.0)
    lvh_diag = ptb.get("ECGFounder_LEFT_VENTRICULAR_HYPERTROPHY", {}).get("auroc", 0.0)
    
    # EchoNext SHD Macro
    echo = data[mid].get("echonext", {})
    echo_macro = echo.get("EchoNext_SHD_Macro", {}).get("auroc", 0.0)
    
    # External Datasets (LUDB, ISP, Zhejiang)
    ludb_r = data[mid].get("ludb", {}).get("QRS_Overall", {}).get("pearson_r", 0.0)
    isp_r = data[mid].get("isp", {}).get("QRS_Overall", {}).get("pearson_r", 0.0)
    zhejiang_r = data[mid].get("zhejiang", {}).get("QRS_Overall", {}).get("pearson_r", 0.0)
    
    parsed.append({
        "model_id": mid,
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "mean_signal_r": mean_signal_r,
        "mean_signal_mae": mean_signal_mae,
        "qrs_r": qrs_r,
        "qrs_mae": qrs_mae,
        "lvh_r": lvh_r,
        "lvh_mae": lvh_mae,
        "delin_cov": delin_cov,
        "afib_auroc": afib_auroc,
        "lbbb_auroc": lbbb_auroc,
        "rbbb_auroc": rbbb_auroc,
        "lvh_diag": lvh_diag,
        "echo_macro": echo_macro,
        "ludb_r": ludb_r,
        "isp_r": isp_r,
        "zhejiang_r": zhejiang_r,
    })

df = pd.DataFrame(parsed)

# Rank columns
df["rank_auroc"] = df["macro_auroc"].rank(ascending=False)
df["rank_sig_r"] = df["mean_signal_r"].rank(ascending=False)
df["rank_qrs_r"] = df["qrs_r"].rank(ascending=False)
df["rank_lvh_r"] = df["lvh_r"].rank(ascending=False)
df["rank_afib"] = df["afib_auroc"].rank(ascending=False)
df["rank_lbbb"] = df["lbbb_auroc"].rank(ascending=False)

df["composite_score"] = (
    df["rank_auroc"] * 1.5 + 
    df["rank_sig_r"] * 1.0 + 
    df["rank_qrs_r"] * 1.0 + 
    df["rank_lvh_r"] * 1.0 +
    df["rank_afib"] * 1.0 +
    df["rank_lbbb"] * 1.0
) / 6.5

print("================================================================================")
print(f"EVALUATED FACTORIAL UNET MODELS (Total: {len(df)} models)")
print("================================================================================")

print("\n--- TOP 10 MODELS BY COMPOSITE CLINICAL RANK ---")
top_comp = df.sort_values(by="composite_score").head(10)
print(top_comp[["model_id", "composite_score", "macro_auroc", "mean_signal_r", "qrs_r", "lvh_r", "afib_auroc", "lbbb_auroc", "delin_cov"]].to_string(index=False))

print("\n--- TOP 10 MODELS BY ECGFounder 150-Task MACRO AUROC ---")
top_auroc = df.sort_values(by="macro_auroc", ascending=False).head(10)
print(top_auroc[["model_id", "macro_auroc", "macro_auprc", "mean_signal_r", "qrs_r", "lvh_r", "afib_auroc"]].to_string(index=False))

print("\n--- TOP 10 MODELS BY SIGNAL RECONSTRUCTION PEARSON R ---")
top_sig = df.sort_values(by="mean_signal_r", ascending=False).head(10)
print(top_sig[["model_id", "mean_signal_r", "mean_signal_mae", "macro_auroc", "qrs_r", "lvh_r"]].to_string(index=False))

print("\n--- TOP 10 MODELS BY QRS MORPHOLOGY DELINEATION PEARSON R ---")
top_qrs = df.sort_values(by="qrs_r", ascending=False).head(10)
print(top_qrs[["model_id", "qrs_r", "qrs_mae", "macro_auroc", "mean_signal_r", "lvh_r"]].to_string(index=False))

print("\n--- TOP 10 MODELS BY AFIB DETECTION AUROC ---")
top_afib = df.sort_values(by="afib_auroc", ascending=False).head(10)
print(top_afib[["model_id", "afib_auroc", "macro_auroc", "mean_signal_r", "qrs_r"]].to_string(index=False))

print("\n--- NEWEST EVALUATED BATCH (MODELS 55 to 67) ---")
print(df.tail(13)[["model_id", "composite_score", "macro_auroc", "mean_signal_r", "qrs_r", "lvh_r", "afib_auroc"]].to_string(index=False))
