import sqlite3
import pandas as pd

conn = sqlite3.connect("results/clinical_biomarkers_multids/clinical_metrics.db")

df_pres = pd.read_sql_query("""
    SELECT model_id, avg_precordial_var_ret_pct, v3_r_var_ret_pct, v6_r_var_ret_pct,
           v3_r_presacan_slope, v6_r_presacan_slope, interlead_r2_recon_I_V3, spurious_coupling_ratio_v3
    FROM presacan_model_summary
    WHERE evaluation_version = 'missing_leads_v2'
""", conn)

conn.close()

print(f"Total Presacan Evaluated Models in DB: {len(df_pres)}")

def classify(m):
    if 'msvae' in m: return 'MS-VAE'
    elif 'aim' in m or 'ecg_aim' in m: return 'ECG-AIM'
    else: return 'U-Net'

df_pres['arch'] = df_pres['model_id'].apply(classify)
print("\nArchitecture Breakdown:")
print(df_pres['arch'].value_counts())

print("\n--- Presacan Summary Across All 160 Models ---")
print(df_pres.describe().to_string())
