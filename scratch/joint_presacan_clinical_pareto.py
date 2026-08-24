import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("results/clinical_biomarkers_multids/clinical_metrics.db")

# 1. Load Presacan Metrics
df_pres = pd.read_sql_query("""
    SELECT model_id,
           v3_r_presacan_r2, v3_r_presacan_slope, v3_r_var_ret_pct, v3_r_direct_r2, v3_r_direct_slope,
           v6_r_presacan_r2, v6_r_presacan_slope, v6_r_var_ret_pct,
           interlead_r2_real_I_V3, interlead_r2_recon_I_V3,
           spurious_coupling_ratio_v3, avg_precordial_var_ret_pct
    FROM presacan_model_summary
    WHERE evaluation_version = 'missing_leads_v2'
""", conn)

# 2. Load Clinical Downstream Metrics
df_clin = pd.read_sql_query("""
    SELECT dataset, model_id, target, auroc, auprc, f1, mae, pearson_r
    FROM clinical_metrics
    WHERE evaluation_version = 'missing_leads_v2'
""", conn)

conn.close()

# Compute PTB-XL Mean Diagnostic AUROC per model
diag_df = df_clin[(df_clin['dataset'] == 'ptb_xl') & (~df_clin['target'].str.startswith('Signal_'))]
ptb_auroc = diag_df.groupby('model_id')['auroc'].mean().reset_index(name='mean_ptbxl_auroc')

# Compute Missing-Lead Signal Pearson r and MAE
sig_df = df_clin[(df_clin['dataset'] == 'ptb_xl') & (df_clin['target'].str.startswith('Signal_Lead')) & (~df_clin['target'].isin(['Signal_Lead_I', 'Signal_Lead_II', 'Signal_Lead_V2']))]
sig_summary = sig_df.groupby('model_id').agg({
    'pearson_r': 'mean',
    'mae': 'mean'
}).reset_index().rename(columns={'pearson_r': 'missing_lead_pearson_r', 'mae': 'missing_lead_mae'})

# Merge all three views
merged = pd.merge(df_pres, ptb_auroc, on='model_id', how='inner')
merged = pd.merge(merged, sig_summary, on='model_id', how='inner')

def get_mask(mid):
    parts = mid.split('_')
    for p in parts:
        if len(p) == 7 and p.isdigit():
            return p
    return mid

merged['mask'] = merged['model_id'].apply(get_mask)

# Factor breakdown
merged['corr_loss'] = merged['mask'].apply(lambda m: int(m[1]) if len(m)==7 else 0)
merged['deriv_loss'] = merged['mask'].apply(lambda m: int(m[2]) if len(m)==7 else 0)
merged['vcg_loss'] = merged['mask'].apply(lambda m: int(m[3]) if len(m)==7 else 0)
merged['energy_loss'] = merged['mask'].apply(lambda m: int(m[4]) if len(m)==7 else 0)
merged['lead_cons_loss'] = merged['mask'].apply(lambda m: int(m[5]) if len(m)==7 else 0)
merged['mmd_kernel'] = merged['mask'].apply(lambda m: int(m[6]) if len(m)==7 else 0)

# Compute Normalized Z-scores for Multi-Objective Ranking
# Objectives to maximize:
# 1. mean_ptbxl_auroc
# 2. avg_precordial_var_ret_pct
# 3. v3_r_var_ret_pct
# 4. missing_lead_pearson_r
# Objectives to minimize (penalize):
# 5. spurious_coupling_ratio_v3 (log scale)
# 6. presacan slope deviation from 0 (|v3_r_presacan_slope - 0|)

z_auroc = (merged['mean_ptbxl_auroc'] - merged['mean_ptbxl_auroc'].mean()) / merged['mean_ptbxl_auroc'].std()
z_precord_var = (merged['avg_precordial_var_ret_pct'] - merged['avg_precordial_var_ret_pct'].mean()) / merged['avg_precordial_var_ret_pct'].std()
z_v3_var = (merged['v3_r_var_ret_pct'] - merged['v3_r_var_ret_pct'].mean()) / merged['v3_r_var_ret_pct'].std()
z_sig_r = (merged['missing_lead_pearson_r'] - merged['missing_lead_pearson_r'].mean()) / merged['missing_lead_pearson_r'].std()
z_spurious_pen = (np.log1p(merged['spurious_coupling_ratio_v3']) - np.log1p(merged['spurious_coupling_ratio_v3']).mean()) / np.log1p(merged['spurious_coupling_ratio_v3']).std()
z_rtm_pen = (abs(merged['v3_r_presacan_slope']) - abs(merged['v3_r_presacan_slope']).mean()) / abs(merged['v3_r_presacan_slope']).std()

merged['composite_clinical_presacan_score'] = (
    0.30 * z_auroc +
    0.25 * z_precord_var +
    0.20 * z_v3_var +
    0.15 * z_sig_r -
    0.10 * z_spurious_pen
)

print(f"=== JOINT PRESACAN + CLINICAL PARETO BENCHMARK (N = {len(merged)} Models) ===\n")

print("=== TOP 10 OVERALL CHAMPIONS (Ranked by Joint Clinical + Presacan Score) ===")
top10 = merged.sort_values(by='composite_clinical_presacan_score', ascending=False).head(10)
cols_show = ['mask', 'model_id', 'composite_clinical_presacan_score', 'mean_ptbxl_auroc', 'avg_precordial_var_ret_pct', 'v3_r_var_ret_pct', 'v3_r_presacan_slope', 'missing_lead_pearson_r', 'interlead_r2_recon_I_V3']
print(top10[cols_show].to_string(index=False))

print("\n=== TOP 5 BY CLINICAL DIAGNOSTIC AUROC (With Presacan Metrics) ===")
top_clin = merged.sort_values(by='mean_ptbxl_auroc', ascending=False).head(5)
print(top_clin[cols_show].to_string(index=False))

print("\n=== TOP 5 BY PRECORDIAL VOLTAGE RETENTION (With Clinical AUROC) ===")
top_var = merged.sort_values(by='avg_precordial_var_ret_pct', ascending=False).head(5)
print(top_var[cols_show].to_string(index=False))

print("\n=== BOTTOM 5 (WORST JOINT PERFORMERS) ===")
bot5 = merged.sort_values(by='composite_clinical_presacan_score', ascending=True).head(5)
print(bot5[cols_show].to_string(index=False))
