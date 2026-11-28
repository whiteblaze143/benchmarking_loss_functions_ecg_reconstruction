"""Figure 10: Clinical Calibration, dcurves Decision Curve Analysis & Alert Frontier."""
import sys
sys.path.insert(0, 'reports/figures_hrs')
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from dcurves import dca
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import paper_plot_style as pps

pps.apply_hrs_style()

oof = pd.read_parquet('experiments/af_density_prediction_v7p/results/V7P_out_of_fold_predictions.parquet')
alerts = pd.read_csv('experiments/af_density_prediction_v7p/results/V7P_alert_benchmarks.csv')

fig = plt.figure(figsize=(11.0, 4.3))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.1, 1.0], wspace=0.34)

# --- Panel A: Calibration Curves from 3.05M Out-Of-Fold Predictions ---
ax_a = fig.add_subplot(gs[0, 0])
ax_a.plot([0, 1], [0, 1], '--', color=pps.PALETTE['slate'], lw=1.0, label='Perfect calibration')

cal_configs = [
    ('P02', 'elastic_net', 'X04_f0_f1_f2_f3_density', 'P02 Persistence (BSS 0.40)', pps.PALETTE['navy'], 10),
    ('P03', 'elastic_net', 'X04_f0_f1_f2_f3_density', 'P03 Survival (BSS 0.44)', pps.PALETTE['teal'], 10),
    ('P01', 'elastic_net', 'X04_f0_f1_f2_f3_density', 'P01 Forecast (Prev 1.7%)', pps.PALETTE['amber'], 8),
]
for tid, mname, ftrack, lbl, col, nbins in cal_configs:
    sub = oof[(oof['task_id'] == tid) & (oof['model_name'] == mname) & (oof['feature_track'] == ftrack)]
    prob_true, prob_pred = calibration_curve(sub['target_true'], sub['predicted_prob'], n_bins=nbins, strategy='quantile')
    ax_a.plot(prob_pred, prob_true, '-o', color=col, lw=1.9, markersize=4.5, label=lbl)

ax_a.set_xlabel('Mean Predicted Probability')
ax_a.set_ylabel('Observed Event Frequency')
ax_a.set_xlim(-0.02, 1.02)
ax_a.set_ylim(-0.02, 1.02)
pps.add_subtle_ygrid(ax_a)
ax_a.legend(frameon=False, loc='upper left', handlelength=1.2, fontsize=pps.FONT_SIZE - 1.7)
pps.add_panel_label(ax_a, 'A', x=-0.18, y=1.06)

# --- Panel B: Clinical Decision Curve Analysis (dcurves) for P02 Persistence ---
ax_b = fig.add_subplot(gs[0, 1])
p02_dens = oof[(oof['task_id'] == 'P02') & (oof['model_name'] == 'elastic_net') & (oof['feature_track'] == 'X04_f0_f1_f2_f3_density')].reset_index(drop=True)
p02_burd = oof[(oof['task_id'] == 'P02') & (oof['model_name'] == 'elastic_net') & (oof['feature_track'] == 'X02_f0_f1_burden')].reset_index(drop=True)

dca_in = pd.DataFrame({
    'y': p02_dens['target_true'].astype(int),
    'DensityModel': p02_dens['predicted_prob'].values,
    'BurdenOnly': p02_burd['predicted_prob'].values,
})
dca_res = dca(data=dca_in, outcome='y', modelnames=['DensityModel', 'BurdenOnly'], thresholds=np.arange(0.05, 0.71, 0.02))

for mname, lbl, col, ls in [
    ('all', 'Treat All', pps.PALETTE['light_gray'], ':'),
    ('none', 'Treat None', pps.PALETTE['slate'], '--'),
    ('BurdenOnly', 'Burden Only', pps.PALETTE['amber'], '-.'),
    ('DensityModel', 'History + Density', pps.PALETTE['navy'], '-'),
]:
    m_sub = dca_res[dca_res['model'] == mname]
    ax_b.plot(m_sub['threshold'] * 100.0, m_sub['net_benefit'], linestyle=ls, color=col, lw=2.0, label=lbl)

ax_b.set_xlabel('Risk Threshold Probability (%)')
ax_b.set_ylabel('Net Clinical Benefit (P02)')
ax_b.set_xlim(5, 70)
ax_b.set_ylim(-0.02, 0.20)
pps.add_subtle_ygrid(ax_b)
ax_b.legend(frameon=False, loc='upper right', handlelength=1.4, fontsize=pps.FONT_SIZE - 1.5)
pps.add_panel_label(ax_b, 'B', x=-0.18, y=1.06)

# --- Panel C: P01 Continuous Warning Alert Trade-Off ---
ax_c = fig.add_subplot(gs[0, 2])
al_g = alerts[alerts['model_name'] == 'gbdt_lightgbm'].sort_values('false_alerts_per_monitoring_hour')
ax_c.plot(al_g['false_alerts_per_monitoring_hour'], al_g['episodes_alerted_fraction'] * 100.0,
          '-o', color=pps.PALETTE['vermilion'], lw=2.0, markersize=5.5)

# Annotate only 3 well-separated clinical operating thresholds cleanly
label_offsets = {
    0.05: (-0.015, 0.8, 'right'),
    0.10: (0.025, -0.3, 'left'),
    0.25: (0.035, 0.3, 'left'),
}
for _, row in al_g.iterrows():
    thr = round(float(row['operating_threshold']), 2)
    if thr in label_offsets:
        dx, dy, ha = label_offsets[thr]
        ax_c.text(
            row['false_alerts_per_monitoring_hour'] + dx,
            row['episodes_alerted_fraction'] * 100.0 + dy,
            f"p>={thr:.2f} ({row['median_lead_time_minutes']:.1f}m lead)",
            fontsize=pps.FONT_SIZE - 1.8, color='#222222', va='center', ha=ha
        )

ax_c.set_xlabel('False Alerts / Monitoring Hour')
ax_c.set_ylabel('Sustained AF Episodes Alerted (%)')
ax_c.set_xlim(0.08, 0.56)
ax_c.set_ylim(0, 13.5)
pps.add_subtle_ygrid(ax_c)
pps.add_panel_label(ax_c, 'C', x=-0.18, y=1.06)

pps.save_hrs_fig(fig, 'Fig10_calibration_dca_and_alerts')
