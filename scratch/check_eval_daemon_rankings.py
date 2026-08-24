import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("results/clinical_biomarkers_multids/clinical_metrics.db")

# Query PTB-XL diagnostic classification performance
df_ptb = pd.read_sql_query("""
    SELECT model_id, target, auroc, auprc, f1, mae, pearson_r
    FROM clinical_metrics
    WHERE dataset = 'ptb_xl' AND evaluation_version = 'missing_leads_v2'
""", conn)

# Query EchoNext performance
df_echo = pd.read_sql_query("""
    SELECT model_id, target, auroc
    FROM clinical_metrics
    WHERE dataset = 'echonext' AND evaluation_version = 'missing_leads_v2'
""", conn)

# Query LUDB performance
df_ludb = pd.read_sql_query("""
    SELECT model_id, target, mae
    FROM clinical_metrics
    WHERE dataset = 'ludb' AND evaluation_version = 'missing_leads_v2'
""", conn)

conn.close()

print("=== PTB-XL DIAGNOSTIC CLASSIFIER SUMMARY ===")
print("Models evaluated on PTB-XL:", len(df_ptb['model_id'].unique()))
diag_targets = [t for t in df_ptb['target'].unique() if not t.startswith('Signal_')]
print("Diagnostic Targets:", diag_targets)

if len(diag_targets) > 0:
    ptb_diag = df_ptb[df_ptb['target'].isin(diag_targets)].copy()
    ptb_piv = ptb_diag.pivot(index='model_id', columns='target', values='auroc')
    ptb_piv['mean_diagnostic_auroc'] = ptb_piv.mean(axis=1)
    
    print("\n--- TOP 10 MODELS BY PTB-XL MEAN DIAGNOSTIC AUROC ---")
    sorted_ptb = ptb_piv.sort_values(by='mean_diagnostic_auroc', ascending=False)
    print(sorted_ptb.head(10).to_string())

# Signal Quality
sig_targets = [t for t in df_ptb['target'].unique() if t.startswith('Signal_')]
if len(sig_targets) > 0:
    ptb_sig = df_ptb[df_ptb['target'].isin(sig_targets)].copy()
    sig_piv_r = ptb_sig.pivot(index='model_id', columns='target', values='pearson_r')
    sig_piv_mae = ptb_sig.pivot(index='model_id', columns='target', values='mae')
    
    # Filter out observed leads (I, II, V2) to get missing leads average
    missing_lead_targets = [t for t in sig_targets if t not in ['Signal_Lead_I', 'Signal_Lead_II', 'Signal_Lead_V2']]
    
    sig_summary = pd.DataFrame({
        'missing_lead_pearson_r': sig_piv_r[missing_lead_targets].mean(axis=1),
        'missing_lead_mae': sig_piv_mae[missing_lead_targets].mean(axis=1)
    }).sort_values(by='missing_lead_pearson_r', ascending=False)
    
    print("\n--- TOP 10 MODELS BY MISSING LEAD SIGNAL RECONSTRUCTION (Pearson r) ---")
    print(sig_summary.head(10).to_string())
