"""
Generalized Additive Models (GAMs) vs Linear OLS Models for Echo_LAD and Echo_LVEF
on the standardized 12-hour SHDB dataset across latency horizons.
"""

import os
import json
import numpy as np
import pandas as pd
from pygam import LinearGAM, s, f, l
import statsmodels.formula.api as smf

def evaluate_cohort_models(complete, outcome, n_sp):
    """
    Fits both OLS and GAM for M1, M2, M3A, M3B, M3C on a complete dataset.
    """
    y = complete[outcome].values
    X_m1 = complete[['Age_at_Holter', 'sex_male', 'BMI']].values
    X_m2 = complete[['Age_at_Holter', 'sex_male', 'BMI', 'af_burden_12h']].values
    
    # Linear OLS M1, M2
    ols_m1 = smf.ols(f'{outcome} ~ Age_at_Holter + sex_male + BMI', data=complete).fit()
    ols_m2 = smf.ols(f'{outcome} ~ Age_at_Holter + sex_male + BMI + af_burden_12h', data=complete).fit()
    
    # GAM M1, M2
    gam_m1 = LinearGAM(s(0, n_splines=n_sp) + f(1) + s(2, n_splines=n_sp)).fit(X_m1, y)
    gam_m2 = LinearGAM(s(0, n_splines=n_sp) + f(1) + s(2, n_splines=n_sp) + s(3, n_splines=n_sp)).fit(X_m2, y)
    
    res = {
        'n': len(complete),
        'ols_m1_r2': ols_m1.rsquared,
        'ols_m1_aic': ols_m1.aic,
        'gam_m1_dev_expl': gam_m1.statistics_['pseudo_r2']['explained_deviance'],
        'gam_m1_aic': gam_m1.statistics_['AIC'],
        
        'ols_m2_r2': ols_m2.rsquared,
        'ols_m2_aic': ols_m2.aic,
        'ols_m2_burden_p': ols_m2.pvalues.get('af_burden_12h', np.nan),
        'gam_m2_dev_expl': gam_m2.statistics_['pseudo_r2']['explained_deviance'],
        'gam_m2_aic': gam_m2.statistics_['AIC'],
        'gam_m2_burden_p': gam_m2.statistics_['p_values'][3] if len(gam_m2.statistics_['p_values']) > 3 else np.nan,
    }
    
    # If density is defined and varies
    if 'density_12h' in complete.columns and complete['density_12h'].nunique() > 1:
        complete_dens = complete.dropna(subset=['density_12h', 'longest_ep_12h_s']).copy()
        if len(complete_dens) >= 10 and complete_dens['density_12h'].nunique() > 1:
            complete_dens['log_longest_ep'] = np.log(complete_dens['longest_ep_12h_s'] + 1.0)
            y_d = complete_dens[outcome].values
            X_m3a = complete_dens[['Age_at_Holter', 'sex_male', 'BMI', 'af_burden_12h', 'density_12h']].values
            X_m3b = complete_dens[['Age_at_Holter', 'sex_male', 'BMI', 'af_burden_12h', 'log_longest_ep']].values
            X_m3c = complete_dens[['Age_at_Holter', 'sex_male', 'BMI', 'af_burden_12h', 'density_12h', 'log_longest_ep']].values
            
            # OLS
            ols_m3a = smf.ols(f'{outcome} ~ Age_at_Holter + sex_male + BMI + af_burden_12h + density_12h', data=complete_dens).fit()
            ols_m3b = smf.ols(f'{outcome} ~ Age_at_Holter + sex_male + BMI + af_burden_12h + log_longest_ep', data=complete_dens).fit()
            ols_m3c = smf.ols(f'{outcome} ~ Age_at_Holter + sex_male + BMI + af_burden_12h + density_12h + log_longest_ep', data=complete_dens).fit()
            
            # GAM
            sp_d = 4 if len(complete_dens) < 25 else (6 if len(complete_dens) < 60 else 10)
            gam_m3a = LinearGAM(s(0, n_splines=sp_d) + f(1) + s(2, n_splines=sp_d) + s(3, n_splines=sp_d) + s(4, n_splines=sp_d)).fit(X_m3a, y_d)
            gam_m3b = LinearGAM(s(0, n_splines=sp_d) + f(1) + s(2, n_splines=sp_d) + s(3, n_splines=sp_d) + s(4, n_splines=sp_d)).fit(X_m3b, y_d)
            gam_m3c = LinearGAM(s(0, n_splines=sp_d) + f(1) + s(2, n_splines=sp_d) + s(3, n_splines=sp_d) + s(4, n_splines=sp_d) + s(5, n_splines=sp_d)).fit(X_m3c, y_d)
            
            res.update({
                'n_density': len(complete_dens),
                'ols_m3a_r2': ols_m3a.rsquared,
                'ols_m3a_aic': ols_m3a.aic,
                'ols_m3a_density_p': ols_m3a.pvalues.get('density_12h', np.nan),
                'gam_m3a_dev_expl': gam_m3a.statistics_['pseudo_r2']['explained_deviance'],
                'gam_m3a_aic': gam_m3a.statistics_['AIC'],
                'gam_m3a_density_p': gam_m3a.statistics_['p_values'][4] if len(gam_m3a.statistics_['p_values']) > 4 else np.nan,
                'gam_m3a_delta_dev': gam_m3a.statistics_['pseudo_r2']['explained_deviance'] - res['gam_m2_dev_expl'],
                
                'ols_m3b_r2': ols_m3b.rsquared,
                'ols_m3b_aic': ols_m3b.aic,
                'ols_m3b_longest_p': ols_m3b.pvalues.get('log_longest_ep', np.nan),
                'gam_m3b_dev_expl': gam_m3b.statistics_['pseudo_r2']['explained_deviance'],
                'gam_m3b_aic': gam_m3b.statistics_['AIC'],
                'gam_m3b_longest_p': gam_m3b.statistics_['p_values'][4] if len(gam_m3b.statistics_['p_values']) > 4 else np.nan,
                
                'ols_m3c_r2': ols_m3c.rsquared,
                'ols_m3c_aic': ols_m3c.aic,
                'gam_m3c_dev_expl': gam_m3c.statistics_['pseudo_r2']['explained_deviance'],
                'gam_m3c_aic': gam_m3c.statistics_['AIC'],
                'gam_m3c_density_p': gam_m3c.statistics_['p_values'][4] if len(gam_m3c.statistics_['p_values']) > 4 else np.nan,
                'gam_m3c_longest_p': gam_m3c.statistics_['p_values'][5] if len(gam_m3c.statistics_['p_values']) > 5 else np.nan
            })
            
    return res

def run_gam_benchmark(df, outcome='Echo_LAD'):
    records = []
    for th in [30, 90, 180, 9999]:
        th_label = f'<={th}d' if th < 9999 else 'Unrestricted'
        cohort_th = df[df['echo_latency_abs_days'] <= th].copy()
        
        cohorts = {
            'PAF_MultiRun': cohort_th[(cohort_th['AF_Type'] == 'PAF') & (cohort_th['n_runs_12h'] >= 2)],
            'PAF_All': cohort_th[cohort_th['AF_Type'] == 'PAF'],
            'All_Annotated': cohort_th[cohort_th['Annotated'] == True]
        }
        
        for c_name, sub in cohorts.items():
            cols = ['Age_at_Holter', 'sex_male', 'BMI', 'af_burden_12h', outcome]
            if c_name == 'PAF_MultiRun':
                cols += ['density_12h', 'longest_ep_12h_s']
            complete = sub.dropna(subset=cols).copy()
            n = len(complete)
            if n < 10:
                continue
                
            n_sp = 4 if n < 25 else (6 if n < 60 else 10)
            metrics = evaluate_cohort_models(complete, outcome, n_sp)
            metrics.update({
                'outcome': outcome,
                'latency_threshold': th_label,
                'latency_th_days': th,
                'cohort': c_name,
                'n_splines': n_sp
            })
            records.append(metrics)
            
    return pd.DataFrame(records)

def main():
    csv_path = 'experiments/af_density_temporal_phenotypes_v7/results/e03_shdb/shdb_12h_standardized_cohort.csv'
    df = pd.read_csv(csv_path)
    out_dir = 'experiments/af_density_temporal_phenotypes_v7/results/e03_shdb'
    
    print('Evaluating GAM vs Linear OLS for Echo_LAD...')
    df_lad = run_gam_benchmark(df, 'Echo_LAD')
    df_lad.to_csv(os.path.join(out_dir, 'shdb_gam_vs_ols_lad.csv'), index=False)
    
    print('Evaluating GAM vs Linear OLS for Echo_LVEF...')
    df_lvef = run_gam_benchmark(df, 'Echo_LVEF')
    df_lvef.to_csv(os.path.join(out_dir, 'shdb_gam_vs_ols_lvef.csv'), index=False)
    
    # Mirror
    codex_dir = '/data/mithunmanivannan/codex_artifacts/af_density_temporal_phenotypes_v7/results/e03_shdb'
    os.makedirs(codex_dir, exist_ok=True)
    df_lad.to_csv(os.path.join(codex_dir, 'shdb_gam_vs_ols_lad.csv'), index=False)
    df_lvef.to_csv(os.path.join(codex_dir, 'shdb_gam_vs_ols_lvef.csv'), index=False)
    print('Complete! Files written to results/e03_shdb and mirrored to codex_artifacts.')

if __name__ == '__main__':
    main()
