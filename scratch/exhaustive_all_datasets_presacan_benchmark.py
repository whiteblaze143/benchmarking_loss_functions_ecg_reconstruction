import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("results/clinical_biomarkers_multids/clinical_metrics.db")

# 1. Load ALL rows from clinical_metrics under missing_leads_v2
df_clin = pd.read_sql_query("""
    SELECT dataset, model_id, target, auroc, auprc, f1, sens, spec, ppv, npv, mae, pearson_r, r2, bland_bias, loa_low, loa_high
    FROM clinical_metrics
    WHERE evaluation_version = 'missing_leads_v2'
""", conn)

# 2. Load Presacan Nature Benchmark
df_pres = pd.read_sql_query("""
    SELECT model_id,
           v3_r_presacan_r2, v3_r_presacan_slope, v3_r_var_ret_pct, v3_r_direct_r2, v3_r_direct_slope,
           v6_r_presacan_r2, v6_r_presacan_slope, v6_r_var_ret_pct,
           interlead_r2_real_I_V3, interlead_r2_recon_I_V3,
           spurious_coupling_ratio_v3, avg_precordial_var_ret_pct
    FROM presacan_model_summary
    WHERE evaluation_version = 'missing_leads_v2'
""", conn)

# 3. Load Detailed Presacan metrics
df_pres_detail = pd.read_sql_query("""
    SELECT model_id, lead, feature, var_ret_pct, presacan_r2, presacan_slope, direct_r2, p_var, p_mean
    FROM presacan_clinical_metrics
    WHERE evaluation_version = 'missing_leads_v2'
""", conn)

conn.close()

models = df_clin['model_id'].unique()
print(f"==================================================================================")
print(f"=== EXHAUSTIVE MULTI-DATASET + PRESACAN BENCHMARK AUDIT (N = {len(models)} Models) ===")
print(f"==================================================================================")

# A. PTB-XL ECGFounder 150-task Foundation Classifier
ptb_df = df_clin[(df_clin['dataset'] == 'ptb_xl') & (~df_clin['target'].str.startswith('Signal_'))]
ptb_summary = ptb_df.groupby('model_id').agg(
    ptbxl_mean_auroc=('auroc', 'mean'),
    ptbxl_mean_auprc=('auprc', 'mean'),
    ptbxl_mean_f1=('f1', 'mean')
).reset_index()

# Extract key PTB-XL clinical superclasses: MI, HYP, STTC, CD, NORM
ptb_piv_auroc = ptb_df.pivot(index='model_id', columns='target', values='auroc')
for sc in ['MI', 'HYP', 'STTC', 'CD', 'NORM', 'LVH', 'AFIB', 'LBBB', 'RBBB']:
    if sc in ptb_piv_auroc.columns:
        ptb_summary[f'ptbxl_{sc}_auroc'] = ptb_summary['model_id'].map(ptb_piv_auroc[sc])

# B. EchoNext Structural Heart Disease (SHD 12 phenotypes)
echo_df = df_clin[df_clin['dataset'] == 'echonext']
echo_shd_targets = [t for t in echo_df['target'].unique() if 'SHD' in t or 'lvef' in t or 'lvwt' in t or 'stenosis' in t or 'regurgitation' in t]
echo_sub = echo_df[echo_df['target'].isin(echo_shd_targets)]
echo_summary = echo_sub.groupby('model_id').agg(
    echonext_mean_auroc=('auroc', 'mean'),
    echonext_mean_auprc=('auprc', 'mean'),
    echonext_mean_f1=('f1', 'mean')
).reset_index()

echo_piv = echo_sub.pivot(index='model_id', columns='target', values='auroc')
for shd_t in ['EchoNextSHD_Macro_12', 'EchoNextSHD_lvef_lte_45', 'EchoNextSHD_lvwt_gte_13', 'EchoNextSHD_aortic_stenosis_moderate_or_greater']:
    if shd_t in echo_piv.columns:
        echo_summary[f'echo_{shd_t}'] = echo_summary['model_id'].map(echo_piv[shd_t])

# C. LUDB Waveform Delineation Precision (ms error & Dice)
ludb_df = df_clin[df_clin['dataset'] == 'ludb']
ludb_bound_targets = [t for t in ludb_df['target'].unique() if t.startswith('Boundary_')]
ludb_morph_targets = [t for t in ludb_df['target'].unique() if t.startswith('Morphology_')]

ludb_bound_sub = ludb_df[ludb_df['target'].isin(ludb_bound_targets)]
ludb_morph_sub = ludb_df[ludb_df['target'].isin(ludb_morph_targets)]

ludb_summary = ludb_df.groupby('model_id').agg(
    ludb_overall_mae=('mae', 'mean')
).reset_index()

if len(ludb_bound_sub) > 0:
    ludb_bound_piv = ludb_bound_sub.groupby('model_id')['mae'].mean().reset_index(name='ludb_mean_boundary_error_ms')
    ludb_summary = pd.merge(ludb_summary, ludb_bound_piv, on='model_id', how='left')

if len(ludb_morph_sub) > 0:
    ludb_morph_piv = ludb_morph_sub.groupby('model_id')['f1'].mean().reset_index(name='ludb_mean_morphology_dice')
    ludb_summary = pd.merge(ludb_summary, ludb_morph_piv, on='model_id', how='left')

# D. External Generalizability: ISP & Zhejiang Datasets
isp_df = df_clin[df_clin['dataset'] == 'isp']
isp_summary = isp_df.groupby('model_id').agg(
    isp_mean_auroc=('auroc', 'mean'),
    isp_mean_mae=('mae', 'mean')
).reset_index()

zhe_df = df_clin[df_clin['dataset'] == 'zhejiang']
zhe_summary = zhe_df.groupby('model_id').agg(
    zhejiang_mean_auroc=('auroc', 'mean'),
    zhejiang_mean_mae=('mae', 'mean')
).reset_index()

sunny_df = df_clin[df_clin['dataset'] == 'sunnybrook']
sunny_summary = sunny_df.groupby('model_id').agg(
    sunnybrook_mean_mae=('mae', 'mean')
).reset_index()

# E. Signal Reconstruction Quality (Missing Leads Pearson r, MAE, SNR, DTW)
sig_df = df_clin[(df_clin['dataset'] == 'ptb_xl') & (df_clin['target'].str.startswith('Signal_'))]
missing_sig = sig_df[~sig_df['target'].isin(['Signal_Lead_I', 'Signal_Lead_II', 'Signal_Lead_V2'])]
sig_summary = missing_sig.groupby('model_id').agg(
    missing_lead_pearson_r=('pearson_r', 'mean'),
    missing_lead_mae=('mae', 'mean')
).reset_index()

# F. Merge ALL Datasets with Presacan Nature Benchmark
master = pd.merge(ptb_summary, echo_summary, on='model_id', how='left')
master = pd.merge(master, ludb_summary, on='model_id', how='left')
master = pd.merge(master, isp_summary, on='model_id', how='left')
master = pd.merge(master, zhe_summary, on='model_id', how='left')
master = pd.merge(master, sunny_summary, on='model_id', how='left')
master = pd.merge(master, sig_summary, on='model_id', how='left')
master = pd.merge(master, df_pres, on='model_id', how='left')

def get_mask(mid):
    parts = mid.split('_')
    for p in parts:
        if len(p) == 7 and p.isdigit():
            return p
    return mid

master['mask'] = master['model_id'].apply(get_mask)

# G. Compute Comprehensive Multi-Domain Grand Index
# Domain 1: Diagnostic Classification Power (PTB-XL + EchoNext + ISP + Zhejiang AUROC)
# Domain 2: Biomechanical Waveform Fidelity (Presacan Precordial VarRet%, V3 VarRet%, Presacan Slope)
# Domain 3: Delineation Precision (LUDB Boundary Error MS & Morphology Dice)
# Domain 4: Signal Reconstruction (Missing Leads Pearson r & MAE)
# Domain 5: Anti-Spurious Isolation (Resistance to Lead I -> V3 Spurious Coupling)

def zscore(s):
    std = s.std()
    return (s - s.mean()) / (std if std > 1e-8 else 1.0)

z_diag = zscore(master['ptbxl_mean_auroc'].fillna(master['ptbxl_mean_auroc'].median()))
z_echo = zscore(master['echonext_mean_auroc'].fillna(master['echonext_mean_auroc'].median()))
z_precord_var = zscore(master['avg_precordial_var_ret_pct'].fillna(master['avg_precordial_var_ret_pct'].median()))
z_v3_var = zscore(master['v3_r_var_ret_pct'].fillna(master['v3_r_var_ret_pct'].median()))
z_slope_rtm = -zscore(abs(master['v3_r_presacan_slope'].fillna(master['v3_r_presacan_slope'].median())))
z_sig_r = zscore(master['missing_lead_pearson_r'].fillna(master['missing_lead_pearson_r'].median()))
z_spurious_pen = -zscore(np.log1p(master['spurious_coupling_ratio_v3'].fillna(master['spurious_coupling_ratio_v3'].median())))

master['grand_comprehensive_score'] = (
    0.25 * z_diag +
    0.20 * z_echo +
    0.20 * z_precord_var +
    0.15 * z_v3_var +
    0.10 * z_sig_r +
    0.05 * z_slope_rtm +
    0.05 * z_spurious_pen
)

master_sorted = master.sort_values(by='grand_comprehensive_score', ascending=False)

print("\n" + "="*115)
print("=== GRAND EXHAUSTIVE MULTI-DATASET + PRESACAN LEADERBOARD (TOP 12 MODELS) ===")
print("="*115)
cols_table = [
    'mask', 'model_id', 'grand_comprehensive_score',
    'ptbxl_mean_auroc', 'echonext_mean_auroc',
    'avg_precordial_var_ret_pct', 'v3_r_var_ret_pct', 'v3_r_presacan_slope',
    'missing_lead_pearson_r', 'interlead_r2_recon_I_V3'
]
print(master_sorted[cols_table].head(12).to_string(index=False))

print("\n" + "="*115)
print("=== TOP PERFORMERS BY DOMAIN ===")
print("="*115)

print("\n1. Top 5 Models on EchoNext Structural Heart Disease (SHD AUROC):")
print(master.sort_values(by='echonext_mean_auroc', ascending=False)[['mask', 'model_id', 'echonext_mean_auroc', 'ptbxl_mean_auroc', 'avg_precordial_var_ret_pct']].head(5).to_string(index=False))

print("\n2. Top 5 Models on PTB-XL ECGFounder Foundation Model (150 Tasks AUROC):")
print(master.sort_values(by='ptbxl_mean_auroc', ascending=False)[['mask', 'model_id', 'ptbxl_mean_auroc', 'echonext_mean_auroc', 'v3_r_var_ret_pct']].head(5).to_string(index=False))

print("\n3. Top 5 Models on Presacan Precordial Voltage Retention (%):")
print(master.sort_values(by='avg_precordial_var_ret_pct', ascending=False)[['mask', 'model_id', 'avg_precordial_var_ret_pct', 'v3_r_var_ret_pct', 'ptbxl_mean_auroc']].head(5).to_string(index=False))

print("\n4. Top 5 Models on Missing Lead Pearson Signal Correlation (r):")
print(master.sort_values(by='missing_lead_pearson_r', ascending=False)[['mask', 'model_id', 'missing_lead_pearson_r', 'missing_lead_mae', 'ptbxl_mean_auroc']].head(5).to_string(index=False))

print("\n" + "="*115)
print("=== BOTTOM 5 OVERALL MODELS (WORST PERFORMERS ACROSS ALL DATASETS) ===")
print("="*115)
print(master_sorted[cols_table].tail(5).to_string(index=False))
