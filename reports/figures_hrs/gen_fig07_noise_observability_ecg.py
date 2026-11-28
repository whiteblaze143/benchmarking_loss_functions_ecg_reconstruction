"""Figure 7: Authentic ECG Artifact Blunts HRV & Distorts Density (Noise Holter, E04)."""
import sys, json
sys.path.insert(0, 'reports/figures_hrs')
import numpy as np
import pandas as pd
import seaborn as sns
import wfdb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statannotations.Annotator import Annotator
import paper_plot_style as pps

pps.apply_hrs_style()

deg = pd.read_csv('experiments/af_density_temporal_phenotypes_v7/results/e04_noise/noise_holter_feature_degradation.csv')
mat = pd.read_csv('experiments/af_density_temporal_phenotypes_v7/results/e04_noise/noise_holter_feature_matrix.csv')
srec = pd.read_csv('experiments/af_density_temporal_phenotypes_v7/results/e04_noise/sensing_dropout_simulation_records.csv')
with open('experiments/af_density_temporal_phenotypes_v7/results/e04_noise/sensing_dropout_simulation_summary.json') as f:
    ssum = json.load(f)

fig = plt.figure(figsize=(11.2, 7.6))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.25, 1.0], hspace=0.36, wspace=0.36)

# --- Panel A: Three Authentic 7s Noise-Holter Strips on Clinical 25 mm/s, 10 mm/mV Paper ---
sub_gs = gs[0, :].subgridspec(3, 1, hspace=0.22)
strips = [
    ('002-020', 1, 'Clean AF (Strip 002-020)', 0.0, 7.0),
    ('001-002', 0, 'N1 Mild Artifact, Sinus (Strip 001-002)', 0.0, 7.0),
    ('001-001', 0, 'N2 Severe Artifact, Sinus (Strip 001-001)', 0.0, 7.0),
]
wfdb_dir = '/data/mithunmanivannan/datasets/noise_holter_database/wfdb_files'
for i, (rec_name, ch_idx, lbl, t0, dur) in enumerate(strips):
    ax_s = fig.add_subplot(sub_gs[i, 0])
    rec = wfdb.rdrecord(f'{wfdb_dir}/{rec_name}')
    fs = float(rec.fs)
    raw = rec.p_signal[int(t0 * fs):int((t0 + dur) * fs), ch_idx]
    sig_mv = pps.highpass_center_ecg(raw, fs=fs)
    pps.draw_clinical_ecg_strip(ax_s, np.clip(sig_mv, -1.35, 1.55), fs=fs, t_start=0.0, label=lbl, y_lim=(-1.4, 1.6))
    if i < 2:
        ax_s.set_xticklabels([])
    else:
        ax_s.set_xlabel('Elapsed Strip Time (s)')
    if i == 0:
        pps.add_panel_label(ax_s, 'A', x=-0.04, y=1.16)

# --- Panel B: CVRR Separation (Sinus vs AF) Across Clean, N1, N2 ---
ax_b = fig.add_subplot(gs[1, 0])
sub_m = mat[mat['is_af'] | mat['is_sinus']].copy()
sub_m['Rhythm'] = np.where(sub_m['is_af'], 'AF', 'Sinus')
sub_m = sub_m[sub_m['cvrr'].notna() & (sub_m['cvrr'] < 0.8)]
order_q = ['Clean', 'N1', 'N2']

sns.boxplot(
    data=sub_m, x='quality_class', y='cvrr', hue='Rhythm', order=order_q,
    hue_order=['Sinus', 'AF'], ax=ax_b,
    palette={'Sinus': pps.PALETTE['navy'], 'AF': pps.PALETTE['vermilion']},
    width=0.58, showfliers=False, boxprops=dict(alpha=0.75)
)
deg_idx = deg.set_index('quality_class')
for idx, q in enumerate(order_q):
    auc_q = deg_idx.loc[q, 'cvrr_auroc']
    ax_b.text(idx, 0.57, f'AUROC {auc_q:.2f}', ha='center', va='bottom',
              fontsize=pps.FONT_SIZE - 1.3, fontweight='bold',
              color=pps.PALETTE['teal'] if auc_q > 0.9 else (pps.PALETTE['amber'] if auc_q > 0.7 else pps.PALETTE['slate']))

ax_b.set_xticks([0, 1, 2])
ax_b.set_xticklabels(['Clean\n(n=2,665)', 'N1 Mild\n(n=2,202)', 'N2 Severe\n(n=1,263)'])
ax_b.set_xlabel('Expert Signal Quality Stratum')
ax_b.set_ylabel('RR Irregularity (CVRR)')
ax_b.set_ylim(-0.02, 0.68)
pps.add_subtle_ygrid(ax_b)
ax_b.legend(frameon=False, loc='upper left', ncol=2, bbox_to_anchor=(0.0, 1.02), handlelength=1.2)
pps.add_panel_label(ax_b, 'B', x=-0.14, y=1.08)

# --- Panel C: Measurement Sensitivity When Brief Runs (<30s) Are Missed ---
ax_c = fig.add_subplot(gs[1, 1])
has_brief = srec[srec['true_brief_30s'] > 0].copy()
bur_abs = has_brief['s1a_burden_err'].abs().values * 100.0
dens_abs = has_brief['s1a_dens_err'].abs().values * 100.0
err_df = pd.DataFrame({
    'Metric': ['AF Burden\nError'] * len(has_brief) + ['AF Density\nError'] * len(has_brief),
    'AbsErr': np.concatenate([bur_abs, dens_abs])
})
sns.boxplot(
    data=err_df, x='Metric', y='AbsErr', hue='Metric', ax=ax_c,
    palette=[pps.PALETTE['sky'], pps.PALETTE['vermilion']],
    width=0.45, showfliers=False, boxprops=dict(alpha=0.40), legend=False
)
sns.stripplot(
    data=err_df, x='Metric', y='AbsErr', hue='Metric', ax=ax_c,
    palette=[pps.PALETTE['sky'], pps.PALETTE['vermilion']],
    size=3.8, alpha=0.7, jitter=0.14, legend=False
)
ax_c.text(0, 16.5, f"Mean {np.nanmean(bur_abs):.2f}%\n(max {np.nanmax(bur_abs):.1f}%, n={len(has_brief)})",
          ha='center', va='top', fontsize=pps.FONT_SIZE - 1.5, color=pps.PALETTE['navy'])
ax_c.text(1, 16.5, f"Mean {np.nanmean(dens_abs):.2f}%\n(up to {np.nanmax(dens_abs):.1f}%)",
          ha='center', va='top', fontsize=pps.FONT_SIZE - 1.5, fontweight='bold', color=pps.PALETTE['vermilion'])
ax_c.text(0.5, 20.6, f"20% dropout: Manski width {ssum['s3_intermittent_dropout_20pct']['mean_manski_width']*100:.1f}%",
          ha='center', va='top', fontsize=pps.FONT_SIZE - 1.5, color='#333333',
          bbox=dict(boxstyle='round,pad=0.2', facecolor='#F5F7FA', edgecolor='#CCCCCC'))
ax_c.set_xlabel('')
ax_c.set_ylabel('Absolute Error (% pts)')
ax_c.set_ylim(-0.8, 22.0)
pps.add_subtle_ygrid(ax_c)
pps.add_panel_label(ax_c, 'C', x=-0.18, y=1.08)

pps.save_hrs_fig(fig, 'Fig07_noise_observability_and_ecg')
