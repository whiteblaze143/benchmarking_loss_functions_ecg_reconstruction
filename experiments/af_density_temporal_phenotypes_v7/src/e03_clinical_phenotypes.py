"""
Experiment E03: Clinical Characterization of Temporal AF Phenotypes in SHDB
Grounded in:
- Inoue et al., Circulation Journal 2026 (Low-burden AF characterization)
- Rosman et al., European Heart Journal 2026 & Daghlas & Marcus 2026 (AF density & clinical substrate)
- Standardized 12-hour monitoring horizon anchored to acquisition start
- Echocardiographic structural outcomes (LAD in mm, LVEF in %) across latency thresholds (<=30d, <=90d, <=180d)
- LOPO Cook's D leverage analysis
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import OLSInfluence

# Ensure af_density metrics are accessible
sys.path.insert(0, os.path.abspath('experiments/af_density'))
try:
    from metrics import charitos_density
except ImportError:
    def charitos_density(intervals, observation_seconds):
        return None

def compute_12h_metrics(record_id, patient_id, runs_df, horizon_s=43200.0):
    """
    Standardize observation duration to horizon_s (default 12 hours = 43200 seconds).
    Extracts all AF runs within [0, horizon_s] and computes exact temporal metrics.
    """
    runs_rec = runs_df[runs_df['record_id'] == record_id].sort_values('start_time_s')
    
    intervals = []
    for _, r_row in runs_rec.iterrows():
        s = max(0.0, min(horizon_s, float(r_row['start_time_s'])))
        e = max(0.0, min(horizon_s, float(r_row['end_time_s'])))
        if e > s:
            intervals.append((s, e))
            
    n_runs = len(intervals)
    af_dur = sum(e - s for s, e in intervals)
    burden = af_dur / horizon_s
    
    if n_runs == 0:
        return {
            'record_id': record_id,
            'patient_id': patient_id,
            'horizon_s': horizon_s,
            'n_runs_12h': 0,
            'af_dur_12h_s': 0.0,
            'af_burden_12h': 0.0,
            'longest_ep_12h_s': 0.0,
            's_max_12h': 0.0,
            'density_12h': None,
            'n_brief_10s_12h': 0,
            'n_brief_30s_12h': 0,
            'n_brief_60s_12h': 0,
            'n_brief_2m_12h': 0,
            'n_interm_6m_30m_12h': 0,
            'n_prolonged_30m_12h': 0,
            'median_gap_12h': None,
            'architecture_type': 'Zero_AF'
        }
        
    durs = [e - s for s, e in intervals]
    longest_ep = max(durs)
    s_max = longest_ep / af_dur if af_dur > 0 else 0.0
    
    n_b10 = sum(1 for d in durs if d < 10.0)
    n_b30 = sum(1 for d in durs if d < 30.0)
    n_b60 = sum(1 for d in durs if d < 60.0)
    n_b2m = sum(1 for d in durs if d < 120.0)
    n_int = sum(1 for d in durs if 360.0 <= d < 1800.0)
    n_pro = sum(1 for d in durs if d >= 1800.0)
    
    if n_runs > 1:
        gaps = [intervals[i+1][0] - intervals[i][1] for i in range(n_runs - 1)]
        median_gap = float(np.median(gaps))
        try:
            density = charitos_density(intervals, horizon_s)
        except Exception:
            density = None
        arch = 'Multi_Predominant' if s_max >= 0.50 else 'Multi_Dispersed'
    else:
        median_gap = None
        density = 1.0  # By definition single episode has density 1.0
        arch = 'Single_Episode'
        
    return {
        'record_id': record_id,
        'patient_id': patient_id,
        'horizon_s': horizon_s,
        'n_runs_12h': n_runs,
        'af_dur_12h_s': af_dur,
        'af_burden_12h': burden,
        'longest_ep_12h_s': longest_ep,
        's_max_12h': s_max,
        'density_12h': density,
        'n_brief_10s_12h': n_b10,
        'n_brief_30s_12h': n_b30,
        'n_brief_60s_12h': n_b60,
        'n_brief_2m_12h': n_b2m,
        'n_interm_6m_30m_12h': n_int,
        'n_prolonged_30m_12h': n_pro,
        'median_gap_12h': median_gap,
        'architecture_type': arch
    }


def load_and_standardize_shdb(additional_data_csv, canonical_runs_parquet, horizon_s=43200.0):
    """
    Loads SHDB AdditionalData.csv and merges with canonical runs clipped to horizon_s.
    """
    df_add = pd.read_csv(additional_data_csv)
    df_add['record_id'] = df_add['Data_ID'].apply(lambda x: f'{int(x):03d}')
    df_add['patient_id'] = df_add['Subject_ID'].apply(lambda x: f'shdb_{int(x)}')
    df_add['Date_Holter_dt'] = pd.to_datetime(df_add['Date_Holter'])
    df_add['Echo_Date_dt'] = pd.to_datetime(df_add['Echo_Date'])
    df_add['echo_latency_days'] = (df_add['Echo_Date_dt'] - df_add['Date_Holter_dt']).dt.days
    df_add['echo_latency_abs_days'] = df_add['echo_latency_days'].abs()
    
    # Sex binary (1 = Male, 0 = Female)
    df_add['sex_male'] = (df_add['Sex'].astype(str).str.strip().str.upper() == 'M').astype(int)
    
    df_runs = pd.read_parquet(canonical_runs_parquet)
    shdb_runs = df_runs[df_runs['dataset'].str.lower().str.contains('shdb')].copy()
    
    horizon_metrics = []
    for _, row in df_add.iterrows():
        m = compute_12h_metrics(row['record_id'], row['patient_id'], shdb_runs, horizon_s=horizon_s)
        horizon_metrics.append(m)
        
    df_12h = pd.DataFrame(horizon_metrics)
    merged = pd.merge(df_add, df_12h, on=['record_id', 'patient_id'])
    return merged


def analyze_low_burden_subgroups(df):
    """
    Analyzes low-burden AF subgroups (<5%, 5-10%, 10-30%, >30% and nested <5%, <10%, 10-30%).
    """
    # AF positive only
    af_pos = df[df['af_burden_12h'] > 0].copy()
    
    strata = {
        'Nested: <5%': af_pos[af_pos['af_burden_12h'] < 0.05],
        'Nested: <10%': af_pos[af_pos['af_burden_12h'] < 0.10],
        'Nested: 10-30%': af_pos[(af_pos['af_burden_12h'] >= 0.10) & (af_pos['af_burden_12h'] <= 0.30)],
        'Exclusive: <5%': af_pos[af_pos['af_burden_12h'] < 0.05],
        'Exclusive: 5-10%': af_pos[(af_pos['af_burden_12h'] >= 0.05) & (af_pos['af_burden_12h'] < 0.10)],
        'Exclusive: 10-30%': af_pos[(af_pos['af_burden_12h'] >= 0.10) & (af_pos['af_burden_12h'] <= 0.30)],
        'Exclusive: >30%': af_pos[af_pos['af_burden_12h'] > 0.30],
    }
    
    summary = []
    for name, sub in strata.items():
        n = len(sub)
        if n == 0:
            continue
        paf_cnt = (sub['AF_Type'] == 'PAF').sum()
        longest_under_6m = (sub['longest_ep_12h_s'] < 360.0).sum()
        longest_6_60m = ((sub['longest_ep_12h_s'] >= 360.0) & (sub['longest_ep_12h_s'] < 3600.0)).sum()
        longest_over_60m = (sub['longest_ep_12h_s'] >= 3600.0).sum()
        
        valid_density = sub['density_12h'].dropna()
        mean_density = valid_density.mean() if len(valid_density) > 0 else np.nan
        std_density = valid_density.std() if len(valid_density) > 1 else np.nan
        
        summary.append({
            'stratum': name,
            'n_patients': n,
            'paf_count': paf_cnt,
            'mean_burden': sub['af_burden_12h'].mean(),
            'longest_ep_lt_6m': longest_under_6m,
            'longest_ep_6_60m': longest_6_60m,
            'longest_ep_gt_60m': longest_over_60m,
            'mean_brief_30s_count': sub['n_brief_30s_12h'].mean(),
            'mean_s_max': sub['s_max_12h'].mean(),
            'n_with_defined_density': len(valid_density),
            'mean_density': mean_density,
            'std_density': std_density
        })
        
    return pd.DataFrame(summary)


def fit_structural_models(df, outcome='Echo_LAD', latency_thresholds=[30, 90, 180, 9999], n_boot=2000):
    """
    Fits nested structural OLS models for outcome (Echo_LAD or Echo_LVEF) across latency thresholds.
    Models compared:
    - M1: Outcome ~ Age_at_Holter + sex_male + BMI
    - M2: M1 + af_burden_12h
    - M3A: M2 + density_12h
    - M3B: M2 + log(longest_ep_12h_s + 1)
    - M3C: M2 + density_12h + log(longest_ep_12h_s + 1)
    """
    model_records = []
    
    # Evaluate across latency thresholds
    for th in latency_thresholds:
        th_label = f'<={th}d' if th < 9999 else 'Unrestricted'
        cohort_th = df[df['echo_latency_abs_days'] <= th].copy()
        
        # We evaluate on PAF with multi-run AF (Primary target)
        cohorts = {
            'PAF_MultiRun': cohort_th[(cohort_th['AF_Type'] == 'PAF') & (cohort_th['n_runs_12h'] >= 2)],
            'PAF_All': cohort_th[cohort_th['AF_Type'] == 'PAF'],
            'All_Annotated': cohort_th[cohort_th['Annotated'] == True]
        }
        
        for c_name, sub in cohorts.items():
            # Prepare complete cases for all variables
            cols_needed = ['Age_at_Holter', 'sex_male', 'BMI', 'af_burden_12h', outcome]
            if c_name == 'PAF_MultiRun':
                cols_needed += ['density_12h', 'longest_ep_12h_s']
            complete = sub.dropna(subset=cols_needed).copy()
            n = len(complete)
            if n < 10:
                continue  # Skip under-supported strata
                
            complete['log_longest_ep'] = np.log(complete['longest_ep_12h_s'] + 1.0)
            
            # M1
            m1 = ols(f'{outcome} ~ Age_at_Holter + sex_male + BMI', data=complete).fit()
            # M2
            m2 = ols(f'{outcome} ~ Age_at_Holter + sex_male + BMI + af_burden_12h', data=complete).fit()
            
            record = {
                'outcome': outcome,
                'latency_threshold': th_label,
                'latency_th_days': th,
                'cohort': c_name,
                'n_patients': n,
                'm1_r2': m1.rsquared,
                'm1_aic': m1.aic,
                'm2_r2': m2.rsquared,
                'm2_aic': m2.aic,
                'm2_burden_coef': m2.params.get('af_burden_12h', np.nan),
                'm2_burden_p': m2.pvalues.get('af_burden_12h', np.nan),
            }
            
            # If density is defined and varies (e.g. PAF_MultiRun)
            if 'density_12h' in complete.columns and complete['density_12h'].nunique() > 1:
                # M3A
                m3a = ols(f'{outcome} ~ Age_at_Holter + sex_male + BMI + af_burden_12h + density_12h', data=complete).fit()
                # M3B
                m3b = ols(f'{outcome} ~ Age_at_Holter + sex_male + BMI + af_burden_12h + log_longest_ep', data=complete).fit()
                # M3C
                m3c = ols(f'{outcome} ~ Age_at_Holter + sex_male + BMI + af_burden_12h + density_12h + log_longest_ep', data=complete).fit()
                
                # F-test M2 vs M3A
                f_test_m3a = m3a.compare_f_test(m2)
                f_stat, f_p = f_test_m3a[0], f_test_m3a[1]
                
                # Density effect per 0.10 increase
                density_coef = m3a.params.get('density_12h', np.nan)
                density_se = m3a.bse.get('density_12h', np.nan)
                density_p = m3a.pvalues.get('density_12h', np.nan)
                effect_per_010 = density_coef * 0.10
                ci_lo = (density_coef - 1.96 * density_se) * 0.10
                ci_hi = (density_coef + 1.96 * density_se) * 0.10
                
                # Partial R2 of density: (SSR(M2) - SSR(M3A)) / SSR(M2)
                partial_r2_density = (m2.ssr - m3a.ssr) / m2.ssr if m2.ssr > 0 else np.nan
                
                # Patient bootstrap CI for effect_per_010
                boot_effects = []
                np.random.seed(42)
                for _ in range(n_boot):
                    idx = np.random.choice(complete.index, size=n, replace=True)
                    boot_df = complete.loc[idx]
                    try:
                        m_boot = ols(f'{outcome} ~ Age_at_Holter + sex_male + BMI + af_burden_12h + density_12h', data=boot_df).fit()
                        boot_effects.append(m_boot.params.get('density_12h', np.nan) * 0.10)
                    except Exception:
                        pass
                boot_effects = [x for x in boot_effects if np.isfinite(x)]
                if len(boot_effects) > 100:
                    boot_ci_lo = float(np.percentile(boot_effects, 2.5))
                    boot_ci_hi = float(np.percentile(boot_effects, 97.5))
                else:
                    boot_ci_lo, boot_ci_hi = ci_lo, ci_hi
                    
                record.update({
                    'm3a_r2': m3a.rsquared,
                    'm3a_aic': m3a.aic,
                    'm3a_partial_r2_density': partial_r2_density,
                    'm3a_f_test_p': f_p,
                    'density_coef': density_coef,
                    'density_se': density_se,
                    'density_p': density_p,
                    'effect_per_010': effect_per_010,
                    'analytical_ci_95': f'[{ci_lo:.3f}, {ci_hi:.3f}]',
                    'bootstrap_ci_95': f'[{boot_ci_lo:.3f}, {boot_ci_hi:.3f}]',
                    'm3b_r2': m3b.rsquared,
                    'm3b_aic': m3b.aic,
                    'm3b_longest_p': m3b.pvalues.get('log_longest_ep', np.nan),
                    'm3c_r2': m3c.rsquared,
                    'm3c_aic': m3c.aic,
                    'm3c_density_p': m3c.pvalues.get('density_12h', np.nan),
                    'm3c_longest_p': m3c.pvalues.get('log_longest_ep', np.nan)
                })
                
            model_records.append(record)
            
    return pd.DataFrame(model_records)


def lopo_cooks_distance_analysis(df, latency_th=90, outcome='Echo_LAD'):
    """
    Computes Cook's distance leverage diagnostics and re-fits models excluding outliers.
    """
    sub = df[(df['echo_latency_abs_days'] <= latency_th) & 
             (df['AF_Type'] == 'PAF') & 
             (df['n_runs_12h'] >= 2)].copy()
             
    cols = ['Age_at_Holter', 'sex_male', 'BMI', 'af_burden_12h', 'density_12h', outcome, 'patient_id', 'record_id']
    complete = sub.dropna(subset=cols).copy()
    n = len(complete)
    if n < 5:
        return {'status': 'insufficient_sample', 'n': n}
        
    m3a = ols(f'{outcome} ~ Age_at_Holter + sex_male + BMI + af_burden_12h + density_12h', data=complete).fit()
    influence = OLSInfluence(m3a)
    cooks_d = influence.cooks_distance[0]
    complete['cooks_d'] = cooks_d
    
    threshold_4_n = 4.0 / n
    outliers = complete[complete['cooks_d'] > threshold_4_n]
    
    # Re-fit without outliers
    clean = complete[complete['cooks_d'] <= threshold_4_n].copy()
    if len(clean) >= 6:
        m3a_clean = ols(f'{outcome} ~ Age_at_Holter + sex_male + BMI + af_burden_12h + density_12h', data=clean).fit()
        coef_clean = m3a_clean.params.get('density_12h', np.nan)
        p_clean = m3a_clean.pvalues.get('density_12h', np.nan)
        effect_clean_010 = coef_clean * 0.10
    else:
        m3a_clean = None
        coef_clean, p_clean, effect_clean_010 = np.nan, np.nan, np.nan
        
    return {
        'n_total': n,
        'threshold_4_n': threshold_4_n,
        'n_outliers': len(outliers),
        'outlier_records': outliers[['record_id', 'patient_id', 'cooks_d', outcome, 'af_burden_12h', 'density_12h']].to_dict('records'),
        'original_m3a_density_coef': m3a.params.get('density_12h', np.nan),
        'original_m3a_density_p': m3a.pvalues.get('density_12h', np.nan),
        'original_effect_per_010': m3a.params.get('density_12h', np.nan) * 0.10,
        'clean_n': len(clean),
        'clean_m3a_density_coef': coef_clean,
        'clean_m3a_density_p': p_clean,
        'clean_effect_per_010': effect_clean_010
    }
