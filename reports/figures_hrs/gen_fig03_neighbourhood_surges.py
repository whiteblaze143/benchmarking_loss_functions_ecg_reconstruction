"""Figure 3: Brief AF Runs Surge Before and After Prolonged Episodes (E01)."""
import sys
sys.path.insert(0, 'reports/figures_hrs')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import paper_plot_style as pps

pps.apply_hrs_style()

rr = pd.read_csv('experiments/af_density_temporal_phenotypes_v7/results/E01_neighbourhood_rate_ratios.csv')
rt = pd.read_csv('experiments/af_density_temporal_phenotypes_v7/results/E01_risk_time_summary.csv')

fig = plt.figure(figsize=(10.5, 4.1))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.55, 0.95, 0.85], wspace=0.35)

# --- Panel A: Pooled Risk-Time Rate Ratios across pre & post windows ---
ax_a = fig.add_subplot(gs[0, 0])
pre_order = ['pre_30_60m_vs_remote', 'pre_15_30m_vs_remote', 'pre_5_15m_vs_remote', 'pre_0_5m_vs_remote']
post_order = ['post_0_5m_vs_remote', 'post_5_15m_vs_remote', 'post_15_30m_vs_remote', 'post_30_60m_vs_remote']
all_rt = rr[(rr['dataset'] == 'ALL') & (rr['exposure_type'] == 'risk_time_h')].set_index('comparison')

pre_df = all_rt.loc[pre_order]
post_df = all_rt.loc[post_order]

x_pre = np.array([0, 1, 2, 3])
x_post = np.array([4.3, 5.3, 6.3, 7.3])
x_ticks = np.concatenate([x_pre, x_post])
x_labels = ['-60:-30m', '-30:-15m', '-15:-5m', '-5:0m', '0:+5m', '+5:+15m', '+15:+30m', '+30:+60m']

ax_a.axhline(1.0, color=pps.PALETTE['slate'], linestyle='--', linewidth=1.0, zorder=1)
ax_a.axvspan(3.35, 3.95, color='#ECEFF4', zorder=0)
ax_a.text(3.65, 9.2, 'Prolonged AF (>=30m)', ha='center', va='center', rotation=90,
          fontsize=pps.FONT_SIZE - 1.8, color='#333333', fontweight='bold')

ax_a.errorbar(
    x_pre, pre_df['rate_ratio'],
    yerr=[pre_df['rate_ratio'] - pre_df['ci_lower'], pre_df['ci_upper'] - pre_df['rate_ratio']],
    fmt='-o', color=pps.PALETTE['navy'], lw=2.0, capsize=3.5, markersize=5.5, label='Pre-Onset', zorder=3
)
ax_a.errorbar(
    x_post, post_df['rate_ratio'],
    yerr=[post_df['rate_ratio'] - post_df['ci_lower'], post_df['ci_upper'] - post_df['rate_ratio']],
    fmt='-s', color=pps.PALETTE['vermilion'], lw=2.0, capsize=3.5, markersize=5.5, label='Post-Termination', zorder=3
)
ax_a.text(2.85, pre_df.loc['pre_0_5m_vs_remote', 'rate_ratio'] + 0.6,
          f"{pre_df.loc['pre_0_5m_vs_remote', 'rate_ratio']:.2f}x", ha='right', va='bottom',
          color=pps.PALETTE['navy'], fontweight='bold', fontsize=pps.FONT_SIZE - 1.2)
ax_a.text(4.45, post_df.loc['post_0_5m_vs_remote', 'rate_ratio'] + 0.6,
          f"{post_df.loc['post_0_5m_vs_remote', 'rate_ratio']:.2f}x", ha='left', va='bottom',
          color=pps.PALETTE['vermilion'], fontweight='bold', fontsize=pps.FONT_SIZE - 1.2)

ax_a.set_xticks(x_ticks)
ax_a.set_xticklabels(x_labels, rotation=32, ha='right', fontsize=pps.FONT_SIZE - 1.5)
ax_a.set_ylabel('Brief-Run Rate Ratio (vs Remote)')
ax_a.set_ylim(0, 17.5)
pps.add_subtle_ygrid(ax_a)
ax_a.legend(frameon=False, loc='upper left', handlelength=1.4)
pps.add_panel_label(ax_a, 'A', x=-0.14, y=1.06)

# --- Panel B: Cohort Stratification (0-15m Pre & Post) ---
ax_b = fig.add_subplot(gs[0, 1])
coh_rows = rr[(rr['dataset'].isin(['LTAFDB', 'SHDB-AF'])) &
              (rr['exposure_type'] == 'risk_time_h') &
              (rr['comparison'].isin(['pre_0_15m_vs_remote', 'post_0_15m_vs_remote']))].copy()
labels_b = ['LTAFDB\nPre 0–15m', 'LTAFDB\nPost 0–15m', 'SHDB-AF\nPre 0–15m', 'SHDB-AF\nPost 0–15m']
keys_b = [('LTAFDB', 'pre_0_15m_vs_remote'), ('LTAFDB', 'post_0_15m_vs_remote'),
          ('SHDB-AF', 'pre_0_15m_vs_remote'), ('SHDB-AF', 'post_0_15m_vs_remote')]
cols_b = [pps.PALETTE['navy'], pps.PALETTE['vermilion'], pps.PALETTE['navy'], pps.PALETTE['vermilion']]

vals_b, err_lo, err_hi = [], [], []
for ds, comp in keys_b:
    row = coh_rows[(coh_rows['dataset'] == ds) & (coh_rows['comparison'] == comp)].iloc[0]
    vals_b.append(row['rate_ratio'])
    err_lo.append(row['rate_ratio'] - row['ci_lower'])
    err_hi.append(row['ci_upper'] - row['rate_ratio'])

xb = np.arange(4)
ax_b.axhline(1.0, color=pps.PALETTE['slate'], linestyle='--', linewidth=1.0)
for i in range(4):
    ax_b.errorbar(xb[i], vals_b[i], yerr=[[err_lo[i]], [err_hi[i]]], fmt='o', color=cols_b[i], capsize=4, markersize=6, lw=1.8)
    ax_b.text(xb[i] + 0.12, vals_b[i], f'{vals_b[i]:.1f}x', va='center', fontsize=pps.FONT_SIZE - 1.5, fontweight='bold', color=cols_b[i])
ax_b.set_yscale('log')
ax_b.set_ylim(0.5, 100)
ax_b.set_xlim(-0.45, 3.75)
ax_b.set_xticks(xb)
ax_b.set_xticklabels(labels_b, rotation=28, ha='right', fontsize=pps.FONT_SIZE - 1.5)
ax_b.set_ylabel('Rate Ratio (log scale)')
pps.add_subtle_ygrid(ax_b)
pps.add_panel_label(ax_b, 'B', x=-0.22, y=1.06)

# --- Panel C: Risk-Time vs Physical-Time Denominator Bias (LTAFDB Post 0-15m) ---
ax_c = fig.add_subplot(gs[0, 2])
ltaf_bias = rr[(rr['dataset'] == 'LTAFDB') & (rr['comparison'] == 'post_0_15m_vs_remote')]
r_risk = ltaf_bias[ltaf_bias['exposure_type'] == 'risk_time_h'].iloc[0]
r_phys = ltaf_bias[ltaf_bias['exposure_type'] == 'physical_time_h'].iloc[0]

xc = [0, 1]
rc_vals = [r_phys['rate_ratio'], r_risk['rate_ratio']]
rc_lo = [r_phys['rate_ratio'] - r_phys['ci_lower'], r_risk['rate_ratio'] - r_risk['ci_lower']]
rc_hi = [r_phys['ci_upper'] - r_phys['rate_ratio'], r_risk['ci_upper'] - r_risk['rate_ratio']]
ax_c.axhline(1.0, color=pps.PALETTE['slate'], linestyle='--', linewidth=1.0)
ax_c.bar(xc, rc_vals, width=0.52, color=[pps.PALETTE['light_gray'], pps.PALETTE['vermilion']], edgecolor='#222222', lw=0.7)
ax_c.errorbar(xc, rc_vals, yerr=[rc_lo, rc_hi], fmt='none', ecolor='#111111', capsize=4, lw=1.4)
ax_c.text(0, rc_vals[0] + rc_hi[0] + 0.35, f"{rc_vals[0]:.2f}x\n(NS)", ha='center', va='bottom', fontsize=pps.FONT_SIZE - 1.5, color='#555555')
ax_c.text(1, rc_vals[1] + rc_hi[1] + 0.35, f"{rc_vals[1]:.2f}x\n(p=0.001)", ha='center', va='bottom', fontsize=pps.FONT_SIZE - 1.5, fontweight='bold', color=pps.PALETTE['vermilion'])
ax_c.set_xticks(xc)
ax_c.set_xticklabels(['Physical Time\n(Confounded)', 'Non-AF Risk\nTime (Exact)'], fontsize=pps.FONT_SIZE - 1.3)
ax_c.set_ylabel('Post 0–15m Rate Ratio')
ax_c.set_ylim(0, 11.2)
pps.add_subtle_ygrid(ax_c)
pps.add_panel_label(ax_c, 'C', x=-0.22, y=1.06)

pps.save_hrs_fig(fig, 'Fig03_neighbourhood_rate_surges')
