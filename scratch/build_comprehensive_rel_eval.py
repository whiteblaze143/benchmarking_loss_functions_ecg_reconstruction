import sqlite3
import pandas as pd
import numpy as np

def build_full_report():
    conn = sqlite3.connect("results/clinical_biomarkers_multids/clinical_metrics.db")
    
    # 1. Load Presacan model summary (160 models)
    df_pres_sum = pd.read_sql_query("""
        SELECT model_id, avg_precordial_var_ret_pct, v3_r_var_ret_pct, v6_r_var_ret_pct,
               v3_r_presacan_slope, v6_r_presacan_slope, v3_r_presacan_r2, v6_r_presacan_r2,
               interlead_r2_real_I_V3, interlead_r2_recon_I_V3, spurious_coupling_ratio_v3
        FROM presacan_model_summary
        WHERE evaluation_version = 'missing_leads_v2'
    """, conn)
    
    # 2. Load Presacan detailed lead x feature rows (3,840 rows)
    df_pres_det = pd.read_sql_query("""
        SELECT model_id, lead, feature, real_mean, real_sd, recon_mean, recon_sd,
               p_mean, p_var, var_ret_pct, bland_bias, loa_low, loa_high,
               presacan_r2, presacan_slope, direct_r2, direct_slope
        FROM presacan_clinical_metrics
        WHERE evaluation_version = 'missing_leads_v2'
        ORDER BY model_id, lead, feature
    """, conn)
    
    # 3. Load Clinical Downstream (PTB-XL, EchoNext, etc.)
    df_clin = pd.read_sql_query("""
        SELECT model_id, target, auroc, auprc, f1, mae, pearson_r, r2
        FROM clinical_metrics
        WHERE dataset = 'ptb_xl' AND evaluation_version = 'missing_leads_v2'
    """, conn)
    
    conn.close()
    
    def get_mask(mid):
        parts = mid.split('_')
        for p in parts:
            if len(p) == 7 and p.isdigit():
                return p
        return mid

    df_pres_sum['mask'] = df_pres_sum['model_id'].apply(get_mask)
    df_pres_det['mask'] = df_pres_det['model_id'].apply(get_mask)
    
    # Pivot clinical metrics
    piv_auroc = df_clin.pivot(index='model_id', columns='target', values='auroc')
    piv_r = df_clin.pivot(index='model_id', columns='target', values='pearson_r')
    piv_mae = df_clin.pivot(index='model_id', columns='target', values='mae')
    
    missing_sig = ['Signal_Lead_III', 'Signal_Lead_aVR', 'Signal_Lead_aVL', 'Signal_Lead_aVF', 'Signal_Lead_V1', 'Signal_Lead_V3', 'Signal_Lead_V4', 'Signal_Lead_V5', 'Signal_Lead_V6']
    
    clin_sum = pd.DataFrame(index=piv_auroc.index)
    if 'ECGFounder_Macro_150' in piv_auroc.columns:
        clin_sum['macro_auroc'] = piv_auroc['ECGFounder_Macro_150']
    clin_sum['missing_sig_r'] = piv_r[[c for c in missing_sig if c in piv_r.columns]].mean(axis=1)
    if 'QRS_Overall' in piv_mae.columns:
        clin_sum['qrs_mae'] = piv_mae['QRS_Overall']
    if 'LVH_SokolowLyon' in piv_mae.columns:
        clin_sum['lvh_mae'] = piv_mae['LVH_SokolowLyon']
        
    master = pd.merge(df_pres_sum, clin_sum.reset_index(), on='model_id', how='left')
    
    # Save the exhaustive 3,840-row table to a dedicated companion CSV
    csv_path = "results/clinical_biomarkers_multids/presacan_granular_3840_rows.csv"
    df_pres_det.to_csv(csv_path, index=False)
    print(f"Saved {len(df_pres_det)} granular rows to {csv_path}")
    
    return master, df_pres_det

master, df_pres_det = build_full_report()
print(f"Master Summary Models: {len(master)}")
print(f"Granular Rows: {len(df_pres_det)}")
