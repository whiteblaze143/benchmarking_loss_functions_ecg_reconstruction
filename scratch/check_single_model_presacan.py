import sqlite3
import pandas as pd

conn = sqlite3.connect('results/clinical_biomarkers_multids/clinical_metrics.db')
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM presacan_model_summary WHERE evaluation_version='missing_leads_v2'")
summary_cnt = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM presacan_clinical_metrics WHERE evaluation_version='missing_leads_v2'")
detail_cnt = cursor.fetchone()[0]

print(f"=== PRESACAN AUDIT VERIFICATION ===")
print(f"Total Model Summaries Saved: {summary_cnt} / 180")
print(f"Total Granular Feature Rows Saved: {detail_cnt} (24 rows per model: 6 leads x 4 features)")

# Show detailed metrics for pure MSE baseline
df = pd.read_sql_query("""
    SELECT lead, feature, real_mean, real_sd, recon_mean, recon_sd,
           p_mean, p_var, var_ret_pct, bland_bias, loa_low, loa_high,
           presacan_r2, presacan_slope, direct_r2, direct_slope
    FROM presacan_clinical_metrics
    WHERE model_id = 'f_1000000_s42' AND evaluation_version='missing_leads_v2'
    ORDER BY lead, feature
""", conn)
conn.close()

if len(df) > 0:
    print("\n--- Detailed Lead x Feature Table for Pure MSE Baseline (f_1000000_s42) ---")
    print(df.to_string(index=False))
else:
    print("\nf_1000000_s42 not yet committed or still in progress.")
