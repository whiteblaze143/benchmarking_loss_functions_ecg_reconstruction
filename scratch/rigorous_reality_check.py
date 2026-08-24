import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("results/clinical_biomarkers_multids/clinical_metrics.db")

df_clin = pd.read_sql_query("""
    SELECT model_id, target, auroc, auprc, f1, mae, pearson_r, r2
    FROM clinical_metrics
    WHERE dataset = 'ptb_xl' AND evaluation_version = 'missing_leads_v2'
""", conn)

df_pres = pd.read_sql_query("""
    SELECT model_id,
           v3_r_presacan_r2, v3_r_presacan_slope, v3_r_var_ret_pct,
           v6_r_presacan_r2, v6_r_presacan_slope, v6_r_var_ret_pct,
           interlead_r2_real_I_V3, interlead_r2_recon_I_V3,
           spurious_coupling_ratio_v3, avg_precordial_var_ret_pct
    FROM presacan_model_summary
    WHERE evaluation_version = 'missing_leads_v2'
""", conn)

conn.close()

# Pivot metrics
piv_auroc = df_clin.pivot(index='model_id', columns='target', values='auroc')
piv_r = df_clin.pivot(index='model_id', columns='target', values='pearson_r')
piv_mae = df_clin.pivot(index='model_id', columns='target', values='mae')

missing_sig = ['Signal_Lead_III', 'Signal_Lead_aVR', 'Signal_Lead_aVL', 'Signal_Lead_aVF', 'Signal_Lead_V1', 'Signal_Lead_V3', 'Signal_Lead_V4', 'Signal_Lead_V5', 'Signal_Lead_V6']
founder_targets = [c for c in piv_auroc.columns if c.startswith('ECGFounder_')]

summary = pd.DataFrame(index=piv_auroc.index)
summary['ecgfounder_macro_auroc'] = piv_auroc['ECGFounder_Macro_150']
summary['missing_lead_pearson_r'] = piv_r[missing_sig].mean(axis=1)
summary['missing_lead_mae_mv'] = piv_mae[missing_sig].mean(axis=1)

merged = pd.merge(summary.reset_index(), df_pres, on='model_id', how='left')

# Filter for U-Net models that have Presacan data
unet_df = merged[merged['model_id'].str.startswith('f_10') & merged['v3_r_var_ret_pct'].notnull()].copy()

def get_mask(mid):
    parts = mid.split('_')
    for p in parts:
        if len(p) == 7 and p.isdigit():
            return p
    return mid

unet_df['mask'] = unet_df['model_id'].apply(get_mask)

# Reference Pure MSE
mse_row = unet_df[unet_df['mask'] == '1000000'].iloc[0]

# Compute deltas relative to Pure MSE
unet_df['delta_macro_auroc'] = unet_df['ecgfounder_macro_auroc'] - mse_row['ecgfounder_macro_auroc']
unet_df['delta_pearson_r'] = unet_df['missing_lead_pearson_r'] - mse_row['missing_lead_pearson_r']
unet_df['delta_v3_var_ret_pct'] = unet_df['v3_r_var_ret_pct'] - mse_row['v3_r_var_ret_pct']
unet_df['delta_precord_var_ret_pct'] = unet_df['avg_precordial_var_ret_pct'] - mse_row['avg_precordial_var_ret_pct']
unet_df['delta_presacan_slope'] = unet_df['v3_r_presacan_slope'] - mse_row['v3_r_presacan_slope']

cols = [
    'mask', 'model_id',
    'ecgfounder_macro_auroc', 'delta_macro_auroc',
    'missing_lead_pearson_r', 'delta_pearson_r',
    'v3_r_var_ret_pct', 'delta_v3_var_ret_pct',
    'avg_precordial_var_ret_pct', 'delta_precord_var_ret_pct',
    'v3_r_presacan_slope'
]

print("=== COLD REALITY CHECK: ALL U-NET FACTORIAL MODELS VS PURE MSE (1000000) ===")
print(f"Pure MSE Baseline: AUROC = {mse_row['ecgfounder_macro_auroc']:.4f} | Pearson r = {mse_row['missing_lead_pearson_r']:.4f} | V3 Var Ret = {mse_row['v3_r_var_ret_pct']:.2f}% | Slope = {mse_row['v3_r_presacan_slope']:.3f}\n")

print(unet_df.sort_values(by='delta_macro_auroc', ascending=False)[cols].to_string(index=False))

print("\n" + "="*90)
print("=== STATISTICAL SUMMARY OF DELTAS ACROSS ALL 40+ U-NET FACTORIAL VARIANTS ===")
print("="*90)
print("Delta Macro AUROC (vs MSE):")
print(unet_df['delta_macro_auroc'].describe())

print("\nDelta Missing Lead Pearson r (vs MSE):")
print(unet_df['delta_pearson_r'].describe())

print("\nDelta V3 Variance Retention % (vs MSE):")
print(unet_df['delta_v3_var_ret_pct'].describe())

print("\nDelta Precordial Variance Retention % (vs MSE):")
print(unet_df['delta_precord_var_ret_pct'].describe())
