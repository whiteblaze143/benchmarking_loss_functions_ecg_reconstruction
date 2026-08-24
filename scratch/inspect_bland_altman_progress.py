import sqlite3
import pandas as pd
import numpy as np

db_path = "results/clinical_biomarkers_multids/clinical_metrics.db"
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("""
    SELECT model_id, bland_v1_r2, bland_v3_r2, bland_v6_r2, 
           bland_v3_slope, bland_v3_var_ret, bland_v6_var_ret, interlead_r2
    FROM clinical_metrics 
    WHERE bland_v3_r2 IS NOT NULL
""", conn)
conn.close()

df = df.drop_duplicates(subset=["model_id"])
print(f"=== Total Models with Bland-Altman Computed: {len(df)} ===")

if len(df) > 0:
    print("\n--- Top 15 by V3 R2 ---")
    print(df.sort_values(by="bland_v3_r2", ascending=False).head(15).to_string(index=False))

    print("\n--- Bottom 10 by V3 R2 ---")
    print(df.sort_values(by="bland_v3_r2", ascending=True).head(10).to_string(index=False))

    print("\n--- Top 10 by V3 Variance Retention (%) ---")
    print(df.sort_values(by="bland_v3_var_ret", ascending=False).head(10).to_string(index=False))

    print("\n--- Distribution Summary ---")
    print(df[["bland_v1_r2", "bland_v3_r2", "bland_v6_r2", "bland_v3_slope", "bland_v3_var_ret", "bland_v6_var_ret", "interlead_r2"]].describe().to_string())
