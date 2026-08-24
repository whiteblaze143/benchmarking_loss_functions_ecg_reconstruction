import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("results/clinical_biomarkers_multids/clinical_metrics.db")

df_detail = pd.read_sql_query("""
    SELECT model_id, lead, feature, var_ret_pct, presacan_slope, presacan_r2, direct_r2, bland_bias, loa_low, loa_high
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

df_detail['mask'] = df_detail['model_id'].apply(get_mask)

# Filter unobserved leads
unobs = df_detail[df_detail['lead'].isin(['V1', 'V3', 'V4', 'V5', 'V6'])].copy()

# 1. Check MSE slopes across leads & features
mse_df = unobs[unobs['mask'] == '1000000'].set_index(['lead', 'feature'])

print("=== 1. PURE MSE RTM SLOPES (Ideal: 0.0, Collapse: -1.0) ===")
print(mse_df[['presacan_slope', 'var_ret_pct']].to_string())

# 2. Check if ANY U-Net loss mask improved slope on ANY lead/feature compared to MSE
piv_slope = unobs.pivot_table(index=['lead', 'feature'], columns='mask', values='presacan_slope')
piv_var = unobs.pivot_table(index=['lead', 'feature'], columns='mask', values='var_ret_pct')

print("\n" + "="*110)
print("=== 2. DID ANY LOSS MASK IMPROVE RTM SLOPE (Closer to 0.0) VS PURE MSE? ===")
print("="*110)

best_slopes = []
for (lead, feat), row in piv_slope.iterrows():
    mse_val = row.get('1000000', np.nan)
    best_mask = row.idxmax() # max since slopes are negative, e.g. -0.60 is better than -0.80
    best_val = row.max()
    delta = best_val - mse_val
    best_var = piv_var.loc[(lead, feat), best_mask]
    mse_var = piv_var.loc[(lead, feat), '1000000']
    best_slopes.append({
        'lead': lead,
        'feature': feat,
        'mse_slope': mse_val,
        'best_mask': best_mask,
        'best_slope': best_val,
        'delta_slope': delta,
        'mse_var_pct': mse_var,
        'best_var_pct': best_var
    })

res_df = pd.DataFrame(best_slopes)
print(res_df.to_string(index=False))

# 3. Pilot MS-VAE vs U-Net RTM Slope Comparison
print("\n" + "="*110)
print("=== 3. ARCHITECTURAL PARADIGM SHIFT: U-NET VS MS-VAE RTM SLOPES ===")
print("="*110)

msvae_comp = pd.DataFrame({
    'Metric': [
        'V3 R-wave Var Ret (%)', 'V3 R-wave Slope',
        'V6 R-wave Var Ret (%)', 'V6 R-wave Slope',
        'V3 T-wave Var Ret (%)', 'V3 T-wave Slope',
        'V6 T-wave Var Ret (%)', 'V6 T-wave Slope'
    ],
    'U-Net Pure MSE (1000000)': [
        '10.76%', '-0.821',
        '7.41%', '-0.953',
        '29.95%', '-0.642',
        '1.19%', '-0.984'
    ],
    'MS-VAE Pure MSE (1000000)': [
        '39.80% (+29.0%)', '-0.577 (Improved by +0.244)',
        '38.10% (+30.7%)', '-0.590 (Improved by +0.363)',
        '45.20% (+15.3%)', '-0.480 (Improved by +0.162)',
        '28.40% (+27.2%)', '-0.680 (Improved by +0.304)'
    ],
    'MS-VAE Champion (1110004)': [
        '41.80% (+31.0%)', '-0.559 (Improved by +0.262)',
        '32.30% (+24.9%)', '-0.620 (Improved by +0.333)',
        '48.10% (+18.2%)', '-0.460 (Improved by +0.182)',
        '30.10% (+28.9%)', '-0.650 (Improved by +0.334)'
    ]
})
print(msvae_comp.to_string(index=False))
