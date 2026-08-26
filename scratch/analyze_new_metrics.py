import sqlite3
import pandas as pd

conn = sqlite3.connect("file:results/onelead_rdb_semiseg_blinded/compact.sqlite?mode=ro", uri=True)

df_evals = pd.read_sql_query("SELECT model_id, status, primary_mean_delta_f1 FROM evaluations WHERE status='complete'", conn)
print(f"Total completed models evaluated: {len(df_evals)}")
print("\nTop 5 models by mean precordial Delta F1:")
print(df_evals.sort_values("primary_mean_delta_f1", ascending=False).head(5).to_string(index=False))

print("\n--- Example Lead-Specific Boundary Results for Best Model ---")
if len(df_evals) > 1:
    best_model = df_evals[df_evals["model_id"] != "__original__"].sort_values("primary_mean_delta_f1", ascending=False).iloc[0]["model_id"]
    df_bounds = pd.read_sql_query(f"""
        SELECT lead_name, boundary, orig_f1_20ms, model_f1_20ms, delta_f1_20ms, retention_pct, 
               preserved_events, recovered_events, lost_events, delta_fp 
        FROM boundary_summaries 
        WHERE model_id='{best_model}' AND lead_name IN ('lead_1', 'lead_7') -- II and V2
        ORDER BY lead_name, boundary
    """, conn)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_bounds.to_string(index=False))
else:
    print("No models evaluated yet.")

conn.close()
