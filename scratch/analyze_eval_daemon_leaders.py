import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("results/clinical_biomarkers_multids/clinical_metrics.db")

df_all = pd.read_sql_query("""
    SELECT dataset, model_id, target, auroc, auprc, f1, mae, pearson_r, r2
    FROM clinical_metrics
    WHERE evaluation_version = 'missing_leads_v2'
""", conn)
conn.close()

models = df_all['model_id'].unique()
datasets = df_all['dataset'].unique()

print(f"=== EVAL DAEMON STATUS (missing_leads_v2) ===")
print(f"Total Unique Models Evaluated: {len(models)}")
print(f"Datasets present: {list(datasets)}")
print(f"Total metric rows: {len(df_all)}\n")

for ds in datasets:
    sub = df_all[df_all['dataset'] == ds]
    targets = sub['target'].unique()
    print(f"--- Dataset: {ds} (Models: {len(sub['model_id'].unique())}) ---")
    print(f"Targets ({len(targets)}): {list(targets)[:8]}...")

# PTB-XL
df_ptb = df_all[df_all['dataset'] == 'ptb_xl'].copy()
if len(df_ptb) > 0:
    ptb_pivot_auroc = df_ptb.pivot(index='model_id', columns='target', values='auroc').dropna(how='all', axis=1)
    ptb_pivot_r = df_ptb.pivot(index='model_id', columns='target', values='pearson_r').dropna(how='all', axis=1)
    ptb_pivot_mae = df_ptb.pivot(index='model_id', columns='target', values='mae').dropna(how='all', axis=1)

    print("\n=== TOP 10 MODELS ON PTB-XL (AUROC Metrics) ===")
    if len(ptb_pivot_auroc.columns) > 0:
        ptb_pivot_auroc['mean_auroc'] = ptb_pivot_auroc.mean(axis=1)
        sorted_auroc = ptb_pivot_auroc.sort_values(by='mean_auroc', ascending=False)
        print(sorted_auroc.head(10).to_string())

    print("\n=== TOP 10 MODELS ON PTB-XL (Signal Pearson r) ===")
    if len(ptb_pivot_r.columns) > 0:
        ptb_pivot_r['mean_r'] = ptb_pivot_r.mean(axis=1)
        sorted_r = ptb_pivot_r.sort_values(by='mean_r', ascending=False)
        print(sorted_r.head(10).to_string())

# EchoNext
df_echo = df_all[df_all['dataset'] == 'echonext'].copy()
if len(df_echo) > 0:
    echo_pivot = df_echo.pivot(index='model_id', columns='target', values='auroc').dropna(how='all', axis=1)
    print("\n=== TOP 10 MODELS ON ECHONEXT STRUCTURAL HEART DISEASE (AUROC) ===")
    if len(echo_pivot.columns) > 0:
        echo_pivot['mean_echo_auroc'] = echo_pivot.mean(axis=1)
        print(echo_pivot.sort_values(by='mean_echo_auroc', ascending=False).head(10).to_string())

# LUDB
df_ludb = df_all[df_all['dataset'] == 'ludb'].copy()
if len(df_ludb) > 0:
    ludb_pivot = df_ludb.pivot(index='model_id', columns='target', values='mae').dropna(how='all', axis=1)
    print("\n=== TOP 10 MODELS ON LUDB DELINEATION (MAE ms) ===")
    if len(ludb_pivot.columns) > 0:
        ludb_pivot['mean_ludb_mae'] = ludb_pivot.mean(axis=1)
        print(ludb_pivot.sort_values(by='mean_ludb_mae', ascending=True).head(10).to_string())
