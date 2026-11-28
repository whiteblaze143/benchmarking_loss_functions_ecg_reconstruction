"""Generate standalone publication-ready LaTeX tables and figure include snippets from verified CSV/JSON files."""
import json
from pathlib import Path
import pandas as pd

OUT_DIR = Path('reports/figures_hrs')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_table_1():
    inv = pd.read_csv('experiments/af_density_temporal_phenotypes_v7/results/E01_duration_class_inventory.csv')
    runs = pd.read_parquet('experiments/af_density_continuity_v5/manifests/canonical_af_runs.parquet')
    coh_inv = pd.read_csv('experiments/af_density_continuity_v5/manifests/canonical_cohort_inventory_reconciled.csv')

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cohort Recording Characteristics and Episode-Duration Architecture Across Continuous Ambulatory Datasets.}",
        r"\label{tab:cohort_architecture}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Cohort & Patients ($n$) & Monitoring (h) & AF Runs ($n$) & Brief Runs $<30$ s (\%) & Prolonged $\ge 1$ h Count (\%) & Prolonged $\ge 1$ h Burden Share (\%) \\",
        r"\midrule",
    ]
    for ds in ['IRIDIA-AF', 'LTAFDB', 'SHDB-AF', 'ALL']:
        c_sub = coh_inv if ds == 'ALL' else coh_inv[coh_inv['dataset'] == ds]
        n_pts = int(c_sub['patient_id'].nunique())
        mon_h = float(c_sub['monitoring_duration_h'].sum())
        d_sub = inv[inv['dataset'] == ds]
        tot_runs = int(d_sub['n_episodes'].sum())
        brief_pct = 100.0 * d_sub[d_sub['duration_class'].isin(['S0', 'S1'])]['n_episodes'].sum() / tot_runs
        prol_sub = d_sub[d_sub['duration_class'].isin(['L2', 'L3'])]
        prol_cnt_pct = 100.0 * prol_sub['n_episodes'].sum() / tot_runs
        prol_burd_pct = 100.0 * prol_sub['total_duration_s'].sum() / d_sub['total_duration_s'].sum()
        ds_lbl = r"\textbf{Combined}" if ds == 'ALL' else ds
        lines.append(
            f"{ds_lbl} & {n_pts} & {mon_h:,.1f} & {tot_runs:,} & {brief_pct:.1f}\\% & {prol_cnt_pct:.1f}\\% & {prol_burd_pct:.1f}\\% \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    (OUT_DIR / 'TABLE_1_cohort_architecture.tex').write_text('\n'.join(lines) + '\n')
    print('Saved TABLE_1_cohort_architecture.tex')


def build_table_2():
    rr = pd.read_csv('experiments/af_density_temporal_phenotypes_v7/results/E01_neighbourhood_rate_ratios.csv')
    gam = pd.read_csv('experiments/af_density_temporal_phenotypes_v7/results/E02_gam_vs_ols_duration.csv')
    with open('experiments/af_density_temporal_phenotypes_v7/results/e03_shdb/shdb_lopo_cooks_distance.json') as f:
        lopo = json.load(f)

    r_pre = rr[(rr['dataset'] == 'ALL') & (rr['exposure_type'] == 'risk_time_h') & (rr['comparison'] == 'pre_0_5m_vs_remote')].iloc[0]
    r_post = rr[(rr['dataset'] == 'ALL') & (rr['exposure_type'] == 'risk_time_h') & (rr['comparison'] == 'post_0_5m_vs_remote')].iloc[0]
    r_shdb_post = rr[(rr['dataset'] == 'SHDB-AF') & (rr['exposure_type'] == 'risk_time_h') & (rr['comparison'] == 'post_0_5m_vs_remote')].iloc[0]
    g_h3 = gam[(gam['cohort'] == 'ALL') & (gam['history_horizon'] == 'H3 (3h)')].iloc[0]
    l90 = lopo['threshold_90d']

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Summary of Temporal Neighbourhood Surges (E01), Non-Linear Episode Persistence (E02), and Structural Substrate Models (E03).}",
        r"\label{tab:e01_e03_findings}",
        r"\small",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Investigation & Clinical Endpoint / Contrast & Sample ($n$) & Effect Estimate (95\% CI) & $P$ Value \\",
        r"\midrule",
        f"E01 Neighbourhoods & Pre-onset 0--5 m brief-run rate ratio (Pooled) & {int(r_pre['n_patients'])} pts & {r_pre['rate_ratio']:.2f} ({r_pre['ci_lower']:.2f}--{r_pre['ci_upper']:.2f}) & {r_pre['p_value']:.2e} \\\\",
        f"E01 Neighbourhoods & Post-termination 0--5 m rate ratio (Pooled) & {int(r_post['n_patients'])} pts & {r_post['rate_ratio']:.2f} ({r_post['ci_lower']:.2f}--{r_post['ci_upper']:.2f}) & {r_post['p_value']:.2e} \\\\",
        f"E01 Neighbourhoods & Post-termination 0--5 m rate ratio (SHDB-AF) & {int(r_shdb_post['n_patients'])} pts & {r_shdb_post['rate_ratio']:.2f} ({r_shdb_post['ci_lower']:.2f}--{r_shdb_post['ci_upper']:.2f}) & {r_shdb_post['p_value']:.2e} \\\\",
        r"\midrule",
        f"E02 Persistence & 3 h density spline vs linear OLS (log duration) & {int(g_h3['n_patients'])} pts & Dev. Expl. {g_h3['gam_m4_dev_expl']*100:.1f}\\% vs $R^2$ {g_h3['ols_m4_r2']*100:.1f}\\% & {g_h3['gam_m4_density_p']:.2e} \\\\",
        r"\midrule",
        f"E03 SHDB Substrate & LAD vs 12 h AF density ($\\le 90$ d PAF multi-run) & {int(l90['n_total'])} pts & +{l90['original_effect_per_010']:.2f} mm / 0.10 density & {l90['original_m3a_density_p']:.3f} \\\\",
        f"E03 SHDB Sensitivity & LAD vs 12 h AF density (excluding 3 Cook's $D$ outliers) & {int(l90['clean_n'])} pts & {l90['clean_effect_per_010']:.2f} mm / 0.10 density & {l90['clean_m3a_density_p']:.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    (OUT_DIR / 'TABLE_2_e01_e03_clinical_findings.tex').write_text('\n'.join(lines) + '\n')
    print('Saved TABLE_2_e01_e03_clinical_findings.tex')


def build_table_3():
    ab = pd.read_csv('experiments/af_density_prediction_v7p/results/ablation_matrix_results.csv')
    fm = pd.read_csv('experiments/af_density_prediction_v7p/results/foundation_model_probe_benchmark.csv')
    cal = pd.read_csv('experiments/af_density_prediction_v7p/results/V7P_calibration_results.csv')

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Patient-Disjoint 5-Fold Cross-Validated Predictive Performance Across Clinical Tasks (P01--P04).}",
        r"\label{tab:v7p_benchmarks}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Task & Representation / Model & AUROC & AUPRC & Brier Score & Balanced Acc. \\",
        r"\midrule",
    ]

    entries = [
        ('P01', 'Burden alone (X02)', ab[(ab['task_id'] == 'P01') & (ab['stage'] == 'X02')]),
        ('P01', 'History + Density + HR/HRV (X07)', ab[(ab['task_id'] == 'P01') & (ab['stage'] == 'X07_history_density_rr')]),
        ('P01', 'Best ECG Foundation Model (ECGFounder)', fm[(fm['task_id'] == 'P01') & (fm['model_type'] == 'ecgfounder')]),
        ('P02', 'Burden alone (X02)', ab[(ab['task_id'] == 'P02') & (ab['stage'] == 'X02')]),
        ('P02', 'History + Density + HR/HRV (X07)', ab[(ab['task_id'] == 'P02') & (ab['stage'] == 'X07_history_density_rr')]),
        ('P02', 'Best ECG Foundation Model (EchoNext)', fm[(fm['task_id'] == 'P02') & (fm['model_type'] == 'echonext')]),
        ('P02', 'SetOperator F2 (Downstream Head)', fm[(fm['task_id'] == 'P02') & (fm['model_type'] == 'f2')]),
        ('P03', 'History + Density + HR/HRV (X07)', ab[(ab['task_id'] == 'P03') & (ab['stage'] == 'X07_history_density_rr')]),
        ('P03', 'Best ECG Foundation Model (ECGFounder)', fm[(fm['task_id'] == 'P03') & (fm['model_type'] == 'ecgfounder')]),
        ('P04', 'Conventional HR/HRV Tabular (X07)', ab[(ab['task_id'] == 'P04') & (ab['stage'] == 'X07_history_density_rr')]),
        ('P04', 'SetOperator F2 (Downstream Head)', fm[(fm['task_id'] == 'P04') & (fm['model_type'] == 'f2')]),
        ('P04', 'Best ECG Foundation Model (ECGFounder)', fm[(fm['task_id'] == 'P04') & (fm['model_type'] == 'ecgfounder')]),
    ]

    task_titles = {
        'P01': 'P01 Onset Forecast',
        'P02': 'P02 Persistence (60s)',
        'P03': 'P03 Survival (15m)',
        'P04': 'P04 Noise Rhythm',
    }
    prev_t = None
    for tid, label, df_sub in entries:
        best = df_sub.sort_values('auroc', ascending=False).iloc[0]
        if prev_t is not None and tid != prev_t:
            lines.append(r"\midrule")
        prev_t = tid
        lines.append(
            f"{task_titles[tid]} & {label} & {best['auroc']:.3f} & {best['auprc']:.3f} & {best['brier_score']:.3f} & {best['balanced_accuracy']:.3f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    (OUT_DIR / 'TABLE_3_v7p_prediction_summary.tex').write_text('\n'.join(lines) + '\n')
    print('Saved TABLE_3_v7p_prediction_summary.tex')


def build_latex_includes():
    figs = [
        ('Fig01_temporal_phenotypes_and_ecg.pdf',
         'Divergent temporal AF phenotypes at matched cumulative burden. (A) Paired 24-hour episode timelines illustrating fragmented brief-run AF (14.9\\% burden, 354 brief runs, density 0.81) versus consolidated prolonged AF (17.5\\% burden, single 4.2-hour episode, density 1.00). (B) Authentic 10-second Lead I ambulatory strip from IRIDIA-AF (25 mm/s, 10 mm/mV) capturing the abrupt transition into irregular AF. (C) Joint burden--density distribution across IRIDIA-AF ($n=152$), LTAFDB ($n=83$), and SHDB-AF ($n=72$) colored by longest-episode share.',
         'fig:temporal_phenotypes'),
        ('Fig02_cohort_duration_architecture.pdf',
         'Episode-duration architecture across 349 ambulatory recordings (10,982 hours). (A) Log-scale episode duration density by cohort. (B) Contrasting share of episode count (59\\% $<30$ s) versus cumulative AF burden (91\\% $\\ge 1$ h). (C) Divergence between episode-weighted and patient-averaged median durations.',
         'fig:duration_architecture'),
        ('Fig03_neighbourhood_rate_surges.pdf',
         'Temporal clustering of brief AF runs ($<30$ s) surrounding prolonged AF episodes ($\\ge 30$ min). (A) Pooled risk-time rate ratios (95\\% CI) showing pre-onset escalation (7.42-fold at 0--5 min) and immediate post-termination surge (6.13-fold at 0--5 min) relative to remote baseline. (B) Cohort stratification (0--15 min windows). (C) Artifactual attenuation (0.57-fold) when using physical time rather than non-AF risk time in LTAFDB.',
         'fig:neighbourhood_surges'),
        ('Fig04_conditional_null_validation.pdf',
         'Conditional randomization validation across 58 eligible patients. (A) Patient-level clustering Z-scores relative to spatially shuffled brief runs (Null A). (B) Episode-duration sequence permutation test (Null B), confirming duration ordering does not exceed gap-duration marginals.',
         'fig:conditional_nulls'),
        ('Fig05_acute_physiology_and_persistence.pdf',
         'Electrophysiological evolution and subsequent episode persistence (E02). (A) Fixed elapsed-time ventricular rate stratified by high versus low preceding 6-hour AF density. (B) Non-linear generalized additive model (GAM) deviance explained versus linear regression for subsequent episode duration. (C) Cox proportional hazards termination ratios ($n=6,683$ episodes, 244 patients).',
         'fig:ep_persistence'),
        ('Fig06_shdb_clinical_substrate.pdf',
         'Low-burden phenotype divergence and echocardiographic left atrial substrate in SHDB-AF (E03). (A) Longest-episode dominance across standardized 12-hour burden strata. (B) Left atrial diameter (LAD) across clinical subtypes. (C) Non-significant association between LAD and 12-hour AF density in contemporaneous paroxysmal AF ($\\le 90$ days, $n=18$), highlighting 3 high-leverage patients (Cook\'s $D > 0.22$).',
         'fig:shdb_substrate'),
        ('Fig07_noise_observability_and_ecg.pdf',
         'Impact of authentic ambulatory ECG artifact on HRV and AF density (E04). (A) Calibrated 25 mm/s, 10 mm/mV Holter strips across Clean AF, mild artifact (N1), and severe artifact (N2). (B) Collapse of RR irregularity (CVRR) discrimination between sinus rhythm and AF from Clean (AUROC 0.94) to N2 noise (0.49). (C) Cumulative burden resilience ($<0.2\\%$ error) versus density distortion (up to 18.2\\% error) upon omitting brief runs.',
         'fig:noise_observability'),
        ('Fig08_ablation_and_bootstrap_contrasts.pdf',
         'Incremental value of longitudinal rhythm history versus instantaneous ECG representations (V7-P). (A) Cross-validated AUROC progression across clinical information tiers for P01--P04. (B) Patient-clustered bootstrap contrasts (95\\% CI) separating longitudinal prognostic gain from instantaneous noisy-strip classification.',
         'fig:ablation_contrasts'),
        ('Fig09_foundation_models_and_survival.pdf',
         'Evaluation of 12 pretrained ECG foundation models and continuous SetOperators. (A) Cross-validated AUROC across instantaneous rhythm detection (P04) versus longitudinal episode prognosis (P01--P03). (B) 5-fold Harrell\'s C-index for dynamic episode termination survival (P03).',
         'fig:fm_and_survival'),
        ('Fig10_calibration_dca_and_alerts.pdf',
         'Clinical calibration, decision curve net benefit, and continuous warning trade-off. (A) Out-of-fold calibration curves across 3.05 million predictions. (B) Decision curve analysis (dcurves) showing Net Benefit for 60-second episode persistence triage (P02). (C) Alert sensitivity versus false-alert burden for imminent sustained AF forecasting (P01).',
         'fig:calibration_dca'),
    ]
    snippets = ["% Auto-generated HRS Manuscript Figure & Table Includes\n"]
    for fname, cap, lbl in figs:
        snippets.append(
            f"\\begin{{figure*}}[t]\n"
            f"    \\centering\n"
            f"    \\includegraphics[width=0.95\\textwidth]{{figures_hrs/{fname}}}\n"
            f"    \\caption{{{cap}}}\n"
            f"    \\label{{{lbl}}}\n"
            f"\\end{{figure*}}\n"
        )
    (OUT_DIR / 'latex_includes.tex').write_text('\n'.join(snippets))
    print('Saved latex_includes.tex')


if __name__ == '__main__':
    build_table_1()
    build_table_2()
    build_table_3()
    build_latex_includes()
