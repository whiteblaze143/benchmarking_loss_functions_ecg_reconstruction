import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("results/clinical_biomarkers_multids/clinical_metrics.db")

# 1. Presacan Nature Benchmark split by architecture
df_presacan = pd.read_sql_query("""
    SELECT model_id, evaluation_version,
           v3_r_presacan_r2, v3_r_presacan_slope, v3_r_var_ret_pct, v3_r_direct_r2, v3_r_direct_slope,
           v6_r_presacan_r2, v6_r_presacan_slope, v6_r_var_ret_pct,
           interlead_r2_real_I_V3, interlead_r2_recon_I_V3,
           spurious_coupling_ratio_v3, avg_precordial_var_ret_pct
    FROM presacan_model_summary
    WHERE evaluation_version = 'missing_leads_v2'
""", conn)

# 2. Clinical Metrics split by architecture
df_clin = pd.read_sql_query("""
    SELECT dataset, model_id, target, auroc, auprc, f1, mae, pearson_r
    FROM clinical_metrics
    WHERE evaluation_version = 'missing_leads_v2'
""", conn)

conn.close()

def classify_arch(mid):
    if 'msvae' in mid:
        return 'MS-VAE'
    elif 'aim' in mid or 'ecg_aim' in mid:
        return 'ECG-AIM'
    else:
        return 'U-Net'

print(f"=== SPLIT BY ARCHITECTURE: OVERVIEW ===")
df_presacan['arch'] = df_presacan['model_id'].apply(classify_arch)
print("Presacan Benchmark Architecture Counts:")
print(df_presacan['arch'].value_counts())

df_clin['arch'] = df_clin['model_id'].apply(classify_arch)
print("\nClinical Metrics Architecture Unique Model Counts:")
print(df_clin.groupby('arch')['model_id'].nunique())

# Architecture Breakdown for Presacan Metrics
print("\n" + "="*95)
print("=== ARCHITECTURE BREAKDOWN: PRESACAN BENCHMARK (NATURE PROTOCOL) ===")
print("="*95)
pres_agg = df_presacan.groupby('arch')[[
    'avg_precordial_var_ret_pct', 'v3_r_var_ret_pct', 'v6_r_var_ret_pct',
    'v3_r_presacan_slope', 'v3_r_presacan_r2', 'interlead_r2_recon_I_V3', 'spurious_coupling_ratio_v3'
]].agg(['mean', 'median', 'count'])
print(pres_agg.to_string())

for arch_name in ['U-Net', 'MS-VAE', 'ECG-AIM']:
    sub = df_presacan[df_presacan['arch'] == arch_name]
    if len(sub) > 0:
        print(f"\n--- Top 5 {arch_name} Models by V3 Variance Retention (%) ---")
        top_arch = sub.sort_values(by='v3_r_var_ret_pct', ascending=False).head(5)
        print(top_arch[['model_id', 'v3_r_var_ret_pct', 'avg_precordial_var_ret_pct', 'v3_r_presacan_slope', 'v3_r_presacan_r2', 'interlead_r2_recon_I_V3']].to_string(index=False))

# Architecture Breakdown for Clinical Metrics (PTB-XL & EchoNext)
print("\n" + "="*95)
print("=== ARCHITECTURE BREAKDOWN: CLINICAL DOWNSTREAM (AUROC & SIGNAL) ===")
print("="*95)

# Signal Missing Leads
sig_df = df_clin[(df_clin['dataset'] == 'ptb_xl') & (df_clin['target'].str.startswith('Signal_Lead')) & (~df_clin['target'].isin(['Signal_Lead_I', 'Signal_Lead_II', 'Signal_Lead_V2']))]
if len(sig_df) > 0:
    sig_piv = sig_df.pivot(index='model_id', columns='target', values='pearson_r')
    sig_piv['arch'] = [classify_arch(m) for m in sig_piv.index]
    sig_piv['missing_lead_mean_r'] = sig_piv.drop(columns=['arch']).mean(axis=1)
    
    print("\n--- Missing Lead Signal Pearson Correlation by Architecture ---")
    print(sig_piv.groupby('arch')['missing_lead_mean_r'].agg(['mean', 'max', 'min', 'count']))

# Diagnostic AUROC
diag_df = df_clin[(df_clin['dataset'] == 'ptb_xl') & (~df_clin['target'].str.startswith('Signal_'))]
if len(diag_df) > 0:
    diag_piv = diag_df.pivot(index='model_id', columns='target', values='auroc')
    diag_piv['arch'] = [classify_arch(m) for m in diag_piv.index]
    diag_piv['mean_auroc'] = diag_piv.drop(columns=['arch']).mean(axis=1)
    
    print("\n--- Diagnostic Classification Mean AUROC by Architecture ---")
    print(diag_piv.groupby('arch')['mean_auroc'].agg(['mean', 'max', 'min', 'count']))
