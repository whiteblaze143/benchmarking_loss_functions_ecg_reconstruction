import sqlite3
import pandas as pd

conn = sqlite3.connect("results/clinical_biomarkers_multids/clinical_metrics.db")

print("=== 1. CLINICAL METRICS TABLE STATUS ===")
df_summary = pd.read_sql_query("""
    SELECT dataset, evaluation_version, COUNT(DISTINCT model_id) as n_models, COUNT(*) as n_rows
    FROM clinical_metrics
    GROUP BY dataset, evaluation_version
""", conn)
print(df_summary.to_string(index=False))

print("\n=== 2. PRESACAN MODEL SUMMARY STATUS ===")
df_pres_sum = pd.read_sql_query("""
    SELECT evaluation_version, COUNT(DISTINCT model_id) as n_models, COUNT(*) as n_rows
    FROM presacan_model_summary
    GROUP BY evaluation_version
""", conn)
print(df_pres_sum.to_string(index=False))

print("\n=== 3. PRESACAN CLINICAL METRICS (DETAILED LEAD x FEATURE) STATUS ===")
df_pres_det = pd.read_sql_query("""
    SELECT evaluation_version, COUNT(DISTINCT model_id) as n_models, COUNT(*) as n_rows
    FROM presacan_clinical_metrics
    GROUP BY evaluation_version
""", conn)
print(df_pres_det.to_string(index=False))

print("\n=== 4. INTEGRITY AUDIT: NULLS, INFS, INVALID P-VALUES ===")
c = conn.cursor()
c.execute("""
    SELECT COUNT(*) FROM clinical_metrics 
    WHERE auroc < 0 OR auroc > 1 OR (r2 < -100 AND r2 IS NOT NULL)
""")
invalid_clin = c.fetchone()[0]

c.execute("""
    SELECT COUNT(*) FROM presacan_model_summary 
    WHERE v3_r_presacan_r2 > 1.0 OR v3_r_var_ret_pct < 0
""")
invalid_pres_sum = c.fetchone()[0]

c.execute("""
    SELECT COUNT(*) FROM presacan_clinical_metrics 
    WHERE var_ret_pct < 0 OR presacan_r2 > 1.0
""")
invalid_pres_det = c.fetchone()[0]

print(f"Invalid Clinical Metric Anomalies: {invalid_clin}")
print(f"Invalid Presacan Summary Anomalies: {invalid_pres_sum}")
print(f"Invalid Presacan Detailed Anomalies: {invalid_pres_det}")

conn.close()
