import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("results/clinical_biomarkers_multids/clinical_metrics.db")

df_clin = pd.read_sql_query("""
    SELECT model_id, target, auroc, auprc, f1, sens, spec, ppv, npv, mae, pearson_r, r2
    FROM clinical_metrics
    WHERE dataset = 'ptb_xl' AND evaluation_version = 'missing_leads_v2'
""", conn)

df_pres = pd.read_sql_query("""
    SELECT model_id,
           v3_r_presacan_r2, v3_r_presacan_slope, v3_r_var_ret_pct,
           v6_r_presacan_r2, v6_r_presacan_slope, v6_r_var_ret_pct,
           interlead_r2_real_I_V3, interlead_r2_recon_I_V3,
           spurious_coupling_ratio_v3, avg_precordial_var_ret_pct
    FROM presacan_model_summary
    WHERE evaluation_version = 'missing_leads_v2'
""", conn)

conn.close()

# 1. Exact Disease Subcategories
infarct_targets = [
    'ECGFounder_ANTERIOR_INFARCT', 'ECGFounder_ANTEROLATERAL_INFARCT',
    'ECGFounder_ANTEROLATERAL_LEADS', 'ECGFounder_ANTEROSEPTAL_INFARCT',
    'ECGFounder_INFERIOR_INFARCT', 'ECGFounder_LATERAL_INFARCT', 'ECGFounder_SEPTAL_INFARCT'
]

conduction_targets = [
    'ECGFounder_LEFT_BUNDLE_BRANCH_BLOCK', 'ECGFounder_RIGHT_BUNDLE_BRANCH_BLOCK',
    'ECGFounder_LEFT_ANTERIOR_FASCICULAR_BLOCK', 'ECGFounder_LEFT_POSTERIOR_FASCICULAR_BLOCK',
    'ECGFounder_NONSPECIFIC_INTRAVENTRICULAR_BLOCK', 'ECGFounder_WITH_1ST_DEGREE_AV_BLOCK',
    'ECGFounder_WITH_QRS_WIDENING', 'ECGFounder_WOLFF_PARKINSON_WHITE'
]

arrhythmia_targets = [
    'ECGFounder_ATRIAL_FIBRILLATION', 'ECGFounder_ATRIAL_FLUTTER',
    'ECGFounder_PREMATURE_VENTRICULAR_COMPLEXES', 'ECGFounder_SUPRAVENTRICULAR_TACHYCARDIA',
    'ECGFounder_VENTRICULAR_TACHYCARDIA', 'ECGFounder_SINUS_BRADYCARDIA',
    'ECGFounder_SINUS_TACHYCARDIA', 'ECGFounder_SINUS_RHYTHM'
]

hypertrophy_targets = [
    'ECGFounder_LEFT_VENTRICULAR_HYPERTROPHY', 'ECGFounder_RIGHT_VENTRICULAR_HYPERTROPHY',
    'ECGFounder_LEFT_ATRIAL_ENLARGEMENT', 'ECGFounder_RIGHT_ATRIAL_ENLARGEMENT',
    'ECGFounder_LOW_VOLTAGE_QRS', 'ECGFounder_QT_HAS_LENGTHENED'
]

missing_sig_targets = [
    'Signal_Lead_III', 'Signal_Lead_aVR', 'Signal_Lead_aVL', 'Signal_Lead_aVF',
    'Signal_Lead_V1', 'Signal_Lead_V3', 'Signal_Lead_V4', 'Signal_Lead_V5', 'Signal_Lead_V6'
]

# Pivot AUROCs
piv_auroc = df_clin.pivot(index='model_id', columns='target', values='auroc')
piv_f1 = df_clin.pivot(index='model_id', columns='target', values='f1')
piv_mae = df_clin.pivot(index='model_id', columns='target', values='mae')
piv_r = df_clin.pivot(index='model_id', columns='target', values='pearson_r')

summary_df = pd.DataFrame(index=piv_auroc.index)

# Compute Category Macro AUROCs
summary_df['ecgfounder_macro_150_auroc'] = piv_auroc['ECGFounder_Macro_150']
summary_df['infarct_macro_auroc'] = piv_auroc[infarct_targets].mean(axis=1)
summary_df['conduction_macro_auroc'] = piv_auroc[conduction_targets].mean(axis=1)
summary_df['arrhythmia_macro_auroc'] = piv_auroc[arrhythmia_targets].mean(axis=1)
summary_df['hypertrophy_macro_auroc'] = piv_auroc[hypertrophy_targets].mean(axis=1)

# Specific Critical Diseases
summary_df['afib_auroc'] = piv_auroc['ECGFounder_ATRIAL_FIBRILLATION']
summary_df['lbbb_auroc'] = piv_auroc['ECGFounder_LEFT_BUNDLE_BRANCH_BLOCK']
summary_df['anterior_mi_auroc'] = piv_auroc['ECGFounder_ANTERIOR_INFARCT']
summary_df['lvh_auroc'] = piv_auroc['ECGFounder_LEFT_VENTRICULAR_HYPERTROPHY']

# Biomarker & Signal Quality
summary_df['qrs_duration_error_ms'] = piv_mae['QRS_Overall']
summary_df['sokolow_lyon_mae_mv'] = piv_mae['LVH_SokolowLyon']
summary_df['missing_lead_pearson_r'] = piv_r[missing_sig_targets].mean(axis=1)
summary_df['missing_lead_mae_mv'] = piv_mae[missing_sig_targets].mean(axis=1)

# Merge with Presacan
full_df = pd.merge(summary_df.reset_index(), df_pres, on='model_id', how='left')

def get_mask(mid):
    parts = mid.split('_')
    for p in parts:
        if len(p) == 7 and p.isdigit():
            return p
    return mid

full_df['mask'] = full_df['model_id'].apply(get_mask)

# Save Matrix
full_df.to_csv("scratch/all_classifiers_comprehensive_matrix.csv", index=False)

# Compute Grand Unified Multi-Domain Score
def zscore(s):
    std = s.std()
    return (s - s.mean()) / (std if std > 1e-8 else 1.0)

# Weights across all clinical categories and Presacan physical metrics:
# 1. Foundation Model Macro AUROC (15%)
# 2. Myocardial Infarctions (15%)
# 3. Conduction Defects (10%)
# 4. Arrhythmias (10%)
# 5. Hypertrophy & Low Voltage (15%)
# 6. Presacan Precordial Variance Retention (15%)
# 7. Presacan V3 Variance Retention (10%)
# 8. Missing Lead Pearson r (10%)

full_df['grand_unified_score'] = (
    0.15 * zscore(full_df['ecgfounder_macro_150_auroc']) +
    0.15 * zscore(full_df['infarct_macro_auroc']) +
    0.10 * zscore(full_df['conduction_macro_auroc']) +
    0.10 * zscore(full_df['arrhythmia_macro_auroc']) +
    0.15 * zscore(full_df['hypertrophy_macro_auroc']) +
    0.15 * zscore(full_df['avg_precordial_var_ret_pct'].fillna(full_df['avg_precordial_var_ret_pct'].median())) +
    0.10 * zscore(full_df['v3_r_var_ret_pct'].fillna(full_df['v3_r_var_ret_pct'].median())) +
    0.10 * zscore(full_df['missing_lead_pearson_r'])
)

sorted_full = full_df.sort_values(by='grand_unified_score', ascending=False)

print("\n" + "="*135)
print("=== GRAND UNIFIED MULTI-CLASSIFIER + PRESACAN LEADERBOARD (TOP 15 MODELS) ===")
print("="*135)
cols_show = [
    'mask', 'model_id', 'grand_unified_score',
    'ecgfounder_macro_150_auroc', 'infarct_macro_auroc', 'conduction_macro_auroc', 'arrhythmia_macro_auroc', 'hypertrophy_macro_auroc',
    'avg_precordial_var_ret_pct', 'v3_r_var_ret_pct', 'missing_lead_pearson_r'
]
print(sorted_full[cols_show].head(15).to_string(index=False))

print("\n" + "="*135)
print("=== CLINICAL SUB-DOMAIN CHAMPIONS ===")
print("="*135)

print("\n1. 🫀 Myocardial Infarction Champions (Anterior, Inferior, Lateral, Septal):")
print(full_df.sort_values(by='infarct_macro_auroc', ascending=False)[['mask', 'model_id', 'infarct_macro_auroc', 'anterior_mi_auroc', 'avg_precordial_var_ret_pct']].head(5).to_string(index=False))

print("\n2. ⚡ Conduction Defect Champions (LBBB, RBBB, LAFB, WPW):")
print(full_df.sort_values(by='conduction_macro_auroc', ascending=False)[['mask', 'model_id', 'conduction_macro_auroc', 'lbbb_auroc', 'qrs_duration_error_ms']].head(5).to_string(index=False))

print("\n3. 💓 Arrhythmia Champions (AFIB, AFLT, PVC, SVT, VT):")
print(full_df.sort_values(by='arrhythmia_macro_auroc', ascending=False)[['mask', 'model_id', 'arrhythmia_macro_auroc', 'afib_auroc', 'missing_lead_pearson_r']].head(5).to_string(index=False))

print("\n4. 🏋️ Hypertrophy & Overload Champions (LVH, RVH, LAE, Low Voltage):")
print(full_df.sort_values(by='hypertrophy_macro_auroc', ascending=False)[['mask', 'model_id', 'hypertrophy_macro_auroc', 'lvh_auroc', 'sokolow_lyon_mae_mv', 'v3_r_var_ret_pct']].head(5).to_string(index=False))
