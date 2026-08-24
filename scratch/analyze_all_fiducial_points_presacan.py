import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("results/clinical_biomarkers_multids/clinical_metrics.db")

df_pres_detail = pd.read_sql_query("""
    SELECT model_id, lead, feature, real_mean, real_sd, recon_mean, recon_sd,
           p_mean, p_var, var_ret_pct, bland_bias, loa_low, loa_high,
           presacan_r2, presacan_slope, direct_r2, direct_slope
    FROM presacan_clinical_metrics
    WHERE evaluation_version = 'missing_leads_v2'
""", conn)

conn.close()

def get_mask(mid):
    parts = mid.split('_')
    for p in parts:
        if len(p) == 7 and p.isdigit():
            return p
    return mid

df_pres_detail['mask'] = df_pres_detail['model_id'].apply(get_mask)

# Filter for missing precordial leads: V1, V3, V4, V5, V6 (exclude V2 which is observed)
missing_detail = df_pres_detail[df_pres_detail['lead'].isin(['V1', 'V3', 'V4', 'V5', 'V6'])].copy()

print(f"==========================================================================================")
print(f"=== PRESACAN ET AL. COMPREHENSIVE FIDUCIAL POINT BENCHMARK (R, S, T, ST ACROSS V1-V6) ===")
print(f"==========================================================================================")
print(f"Total Rows: {len(missing_detail)} (Models: {missing_detail['model_id'].nunique()})\n")

# 1. Pure MSE Baseline (1000000) Across All Fiducial Points & Leads
mse_detail = missing_detail[missing_detail['mask'] == '1000000'].sort_values(by=['feature', 'lead'])
print("=== 1. PURE MSE BASELINE (1000000) FULL FIDUCIAL POINT BREAKDOWN ===")
cols_show = ['lead', 'feature', 'real_mean', 'recon_mean', 'var_ret_pct', 'bland_bias', 'loa_low', 'loa_high', 'presacan_slope', 'presacan_r2', 'direct_r2']
print(mse_detail[cols_show].to_string(index=False))

# 2. Fiducial Point Summary Table (Averaged across unobserved leads V1, V3, V4, V5, V6)
fiducial_by_mask = missing_detail.groupby(['mask', 'feature']).agg({
    'var_ret_pct': 'mean',
    'presacan_slope': 'mean',
    'presacan_r2': 'mean',
    'direct_r2': 'mean',
    'bland_bias': lambda x: float(np.mean(np.abs(x)))
}).reset_index().rename(columns={'bland_bias': 'abs_bland_bias_mv'})

print("\n" + "="*95)
print("=== 2. FIDUCIAL POINT AGGREGATES ACROSS TOP LOSS MASKS (V1, V3, V4, V5, V6 Average) ===")
print("="*95)

key_masks = ['1000000', '1000003', '1000004', '1000002', '1001002', '1001004', '1000010', '1000110']
sub_fids = fiducial_by_mask[fiducial_by_mask['mask'].isin(key_masks)].sort_values(by=['feature', 'var_ret_pct'], ascending=[True, False])

for feat in ['R', 'S', 'T', 'ST']:
    feat_df = sub_fids[sub_fids['feature'] == feat]
    print(f"\n--- Fiducial Waveform Point: {feat}-Wave / Segment ---")
    print(feat_df.to_string(index=False))

# 3. Factorial Main Effects on Each Fiducial Point
missing_detail['corr_loss'] = missing_detail['mask'].apply(lambda m: int(m[1]) if len(m)==7 else 0)
missing_detail['deriv_loss'] = missing_detail['mask'].apply(lambda m: int(m[2]) if len(m)==7 else 0)
missing_detail['vcg_loss'] = missing_detail['mask'].apply(lambda m: int(m[3]) if len(m)==7 else 0)
missing_detail['energy_loss'] = missing_detail['mask'].apply(lambda m: int(m[4]) if len(m)==7 else 0)
missing_detail['lead_cons_loss'] = missing_detail['mask'].apply(lambda m: int(m[5]) if len(m)==7 else 0)
missing_detail['mmd_kernel'] = missing_detail['mask'].apply(lambda m: int(m[6]) if len(m)==7 else 0)

print("\n" + "="*95)
print("=== 3. FACTORIAL EFFECT ON FIDUCIAL POINT VARIANCE RETENTION (%) ===")
print("="*95)
piv_factorial = missing_detail.pivot_table(index=['lead_cons_loss', 'energy_loss', 'vcg_loss'], columns='feature', values='var_ret_pct', aggfunc='mean')
print(piv_factorial.to_string())

# 4. Fiducial Point Regression-to-the-Mean Slope (Ideal: 0.0, Collapse: -1.0)
print("\n" + "="*95)
print("=== 4. FACTORIAL EFFECT ON REGRESSION-TO-THE-MEAN SLOPE (Ideal: 0.0) ===")
print("="*95)
piv_slope = missing_detail.pivot_table(index=['lead_cons_loss', 'energy_loss', 'vcg_loss'], columns='feature', values='presacan_slope', aggfunc='mean')
print(piv_slope.to_string())
