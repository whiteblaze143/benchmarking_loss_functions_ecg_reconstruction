"""Figure 5: Acute Electrophysiology & Subsequent Episode Persistence (E02)."""
import sys
sys.path.insert(0, 'reports/figures_hrs')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import paper_plot_style as pps

pps.apply_hrs_style()

rr = pd.read_parquet('experiments/af_density_temporal_phenotypes_v7/results/E02_fixed_elapsed_rr_features.parquet')
gam = pd.read_csv('experiments/af_density_temporal_phenotypes_v7/results/E02_gam_vs_ols_duration.csv')
cox = pd.read_csv('experiments/af_density_temporal_phenotypes_v7/results/E02_subsequent_episode_models.csv')

fig = plt.figure(figsize=(11.5, 4.3))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.15], wspace=0.52)

# --- Panel A: Within-episode HR by Preceding 6h AF Density ---
ax_a = fig.add_subplot(gs[0, 0])
v = rr[(rr['is_valid_physio'] == True) & rr['h6_density'].notna()].copy()
v['dens_tier'] = pd.qcut(v['h6_density'], q=2, labels=['Low Prior Density', 'High Prior Density'])
win_order = ['win_30_180s', 'win_5_7_5m', 'win_15_17_5m', 'win_30_32_5m']
win_lbls = ['0.5–3m', '5–7.5m', '15–17.5m', '30–32.5m']
x = np.arange(len(win_order))

for tier, col, marker in [('High Prior Density', pps.PALETTE['vermilion'], 'o'),
                          ('Low Prior Density', pps.PALETTE['navy'], 's')]:
    sub = v[v['dens_tier'] == tier].groupby('window_name')['rate_bpm'].agg(['mean', 'sem']).loc[win_order]
    ax_a.errorbar(x, sub['mean'], yerr=sub['sem'], fmt=f'-{marker}', color=col, lw=2.0, capsize=3.5, markersize=5.5, label=tier)

ax_a.set_xticks(x)
ax_a.set_xticklabels(win_lbls, rotation=22, ha='right', fontsize=pps.FONT_SIZE - 1.3)
ax_a.set_xlabel('Elapsed Time in AF')
ax_a.set_ylabel('Mean Ventricular Rate (bpm)')
ax_a.set_ylim(78, 112)
pps.add_subtle_ygrid(ax_a)
ax_a.legend(frameon=False, loc='upper right', handlelength=1.4)
pps.add_panel_label(ax_a, 'A', x=-0.18, y=1.06)

# --- Panel B: Linear OLS vs Non-Linear GAM Explaining Subsequent Episode Duration ---
ax_b = fig.add_subplot(gs[0, 1])
all_gam = gam[gam['cohort'] == 'ALL'].copy()
horizons = ['H1 (1h)', 'H3 (3h)', 'H6 (6h, Primary)']
all_gam = all_gam.set_index('history_horizon').loc[horizons]

xb = np.arange(len(horizons))
wb = 0.35
b_ols = ax_b.bar(xb - wb/2, all_gam['ols_m4_r2'] * 100, wb, color=pps.PALETTE['light_gray'], edgecolor='#222222', lw=0.6, label='Linear OLS ($R^2$)')
b_gam = ax_b.bar(xb + wb/2, all_gam['gam_m4_dev_expl'] * 100, wb, color=pps.PALETTE['purple'], edgecolor='#222222', lw=0.6, label='Non-linear GAM')

for i, (r2v, dv, pv) in enumerate(zip(all_gam['ols_m4_r2'] * 100, all_gam['gam_m4_dev_expl'] * 100, all_gam['gam_m4_density_p'])):
    ax_b.text(xb[i] - wb/2, r2v + 0.8, f'{r2v:.1f}%', ha='center', va='bottom', fontsize=pps.FONT_SIZE - 1.8)
    ax_b.text(xb[i] + wb/2, dv + 0.8, f'{dv:.1f}%\n(p<1e-4)' if pv < 1e-4 else f'{dv:.1f}%', ha='center', va='bottom', fontsize=pps.FONT_SIZE - 1.8, fontweight='bold', color='#5A1846')

ax_b.set_xticks(xb)
ax_b.set_xticklabels(['1 h Window', '3 h Window', '6 h Window'], fontsize=pps.FONT_SIZE - 1.3)
ax_b.set_ylabel('Variance / Deviance Explained (%)')
ax_b.set_ylim(0, 56)
pps.add_subtle_ygrid(ax_b)
ax_b.legend(frameon=False, loc='upper left', handlelength=1.2)
pps.add_panel_label(ax_b, 'B', x=-0.18, y=1.06)

# --- Panel C: Cox Termination Hazard Ratios (Model 4, n=6,683 episodes) ---
ax_c = fig.add_subplot(gs[0, 2])
m4 = cox[cox['model_name'] == 'Model_4_Full_With_Density'].copy()
pred_map = {
    'last_gap_h': 'Prior Gap (h)',
    'last_dur_h': 'Prior Episode (h)',
    'h6_burden': '6h Burden',
    'h6_run_count': '6h Brief Runs',
    'hist_density_filled': 'Linear 6h Density',
}
m4['label'] = m4['predictor'].map(pred_map)
m4 = m4.iloc[::-1].reset_index(drop=True)  # top-to-bottom order

yc = np.arange(len(m4))
ax_c.axvline(1.0, color=pps.PALETTE['slate'], linestyle='--', lw=1.0, zorder=1)
for i, row in m4.iterrows():
    col = pps.PALETTE['navy'] if row['p_value'] < 0.05 and row['hazard_ratio'] < 1.0 else (
        pps.PALETTE['vermilion'] if row['p_value'] < 0.05 and row['hazard_ratio'] > 1.0 else pps.PALETTE['slate']
    )
    ax_c.errorbar(
        row['hazard_ratio'], i,
        xerr=[[row['hazard_ratio'] - row['ci_lower']], [row['ci_upper'] - row['hazard_ratio']]],
        fmt='o', color=col, capsize=3.5, markersize=5.5, lw=1.8, zorder=3
    )
    ax_c.text(max(row['ci_upper'], 1.0) + 0.03, i, f"{row['hazard_ratio']:.2f}", va='center', fontsize=pps.FONT_SIZE - 1.6, color=col, fontweight='bold')

ax_c.set_yticks(yc)
ax_c.set_yticklabels(m4['label'], fontsize=pps.FONT_SIZE - 1.2)
ax_c.set_xlabel('Termination Hazard Ratio (95% CI)')
ax_c.set_xlim(0.42, 1.85)
ax_c.text(0.46, -0.55, '<- Longer AF', fontsize=pps.FONT_SIZE - 2.0, color=pps.PALETTE['navy'], fontweight='bold')
ax_c.text(1.78, -0.55, 'Earlier Term ->', fontsize=pps.FONT_SIZE - 2.0, color=pps.PALETTE['vermilion'], ha='right', fontweight='bold')
ax_c.set_ylim(-0.8, len(m4) - 0.3)
pps.add_panel_label(ax_c, 'C', x=-0.28, y=1.06)

pps.save_hrs_fig(fig, 'Fig05_acute_physiology_and_persistence')
