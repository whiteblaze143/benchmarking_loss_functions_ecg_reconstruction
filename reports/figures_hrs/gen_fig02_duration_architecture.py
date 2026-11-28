"""Figure 2: Cohort Episode-Duration Architecture (349 Recordings, 10,982 Hours)."""
import sys
sys.path.insert(0, 'reports/figures_hrs')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import paper_plot_style as pps

pps.apply_hrs_style()

runs = pd.read_parquet('experiments/af_density_continuity_v5/manifests/canonical_af_runs.parquet')
inv = pd.read_csv('experiments/af_density_temporal_phenotypes_v7/results/E01_duration_class_inventory.csv')

fig = plt.figure(figsize=(10.5, 4.2))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.0, 0.95], wspace=0.34)

# --- Panel A: Log10 Episode Duration KDE ---
ax_a = fig.add_subplot(gs[0, 0])
for ds in ['IRIDIA-AF', 'LTAFDB', 'SHDB-AF']:
    grp = runs[runs['dataset'] == ds]
    n_pts = grp['patient_id'].nunique()
    dur_m = (grp['duration_s'] / 60.0).clip(lower=0.02)
    sns.kdeplot(
        np.log10(dur_m), ax=ax_a,
        color=pps.COHORT_COLORS[ds], lw=2.0,
        label=f'{ds} ({n_pts} pts)'
    )
ax_a.set_xticks([-1, 0, 1, np.log10(60), np.log10(360), np.log10(1440)])
ax_a.set_xticklabels(['6s', '1m', '10m', '1h', '6h', '24h'])
ax_a.set_xlabel('Episode Duration (log scale)')
ax_a.set_ylabel('Probability Density')
ax_a.legend(frameon=False, loc='upper right', handlelength=1.4)
pps.add_panel_label(ax_a, 'A', x=-0.16, y=1.06)

# --- Panel B: Count Share vs Burden Share by Duration Class (pooled) ---
ax_b = fig.add_subplot(gs[0, 1])
pooled = inv[inv['dataset'] == 'ALL'].copy()
# Group into 5 clinician-clear tiers
tier_map = {
    'S0': '<30 s', 'S1': '<30 s',
    'S2': '30 s–6 m', 'S3': '30 s–6 m',
    'M1': '6–30 m',
    'L1': '30 m–6 h',
    'L2': '>=6 h', 'L3': '>=6 h',
}
# Wait: inspect exact duration_class names cleanly in pooled
order = ['S0', 'S1', 'S2', 'S3', 'M1', 'L1', 'L2', 'L3']
# Map dynamically using duration classes present
if set(pooled['duration_class']).issuperset({'S0', 'S1', 'S2', 'S3', 'M1', 'L1', 'L2', 'L3'}):
    groups = [
        ('<30 s', ['S0', 'S1'], pps.PALETTE['sky']),
        ('30 s–6 m', ['S2', 'S3'], pps.PALETTE['teal']),
        ('6–30 m', ['M1'], pps.PALETTE['amber']),
        ('30 m–6 h', ['L1'], pps.PALETTE['navy']),
        ('>=6 h', ['L2', 'L3'], pps.PALETTE['vermilion']),
    ]
else:
    groups = [(c, [c], pps.PALETTE['navy']) for c in pooled['duration_class']]

tot_ep = pooled['n_episodes'].sum()
tot_dur = pooled['total_duration_s'].sum()

left_c, left_b = 0.0, 0.0
for lbl, codes, col in groups:
    sub = pooled[pooled['duration_class'].isin(codes)]
    pct_c = 100.0 * sub['n_episodes'].sum() / tot_ep
    pct_b = 100.0 * sub['total_duration_s'].sum() / tot_dur
    ax_b.barh(1.0, pct_c, left=left_c, height=0.45, color=col, edgecolor='white', linewidth=0.8, label=lbl)
    ax_b.barh(0.0, pct_b, left=left_b, height=0.45, color=col, edgecolor='white', linewidth=0.8)
    if pct_c >= 9.0:
        ax_b.text(left_c + pct_c / 2, 1.0, f'{pct_c:.0f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=pps.FONT_SIZE - 1.2)
    if pct_b >= 9.0:
        ax_b.text(left_b + pct_b / 2, 0.0, f'{pct_b:.0f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=pps.FONT_SIZE - 1.2)
    left_c += pct_c
    left_b += pct_b

ax_b.set_yticks([0.0, 1.0])
ax_b.set_yticklabels(['Cumulative\nAF Burden', 'Episode\nCount'])
ax_b.set_xlim(0, 100)
ax_b.set_xlabel('Proportion of Total (%)')
ax_b.spines['left'].set_visible(False)
ax_b.tick_params(axis='y', length=0)
ax_b.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.48, 1.22), ncol=3, columnspacing=0.9, handlelength=1.0, fontsize=pps.FONT_SIZE - 1.6)
pps.add_panel_label(ax_b, 'B', x=-0.18, y=1.06)

# --- Panel C: Episode-Weighted vs Patient-Averaged Median Duration ---
ax_c = fig.add_subplot(gs[0, 2])
cohorts = ['IRIDIA-AF', 'LTAFDB', 'SHDB-AF']
ep_med = [runs[runs['dataset'] == d]['duration_s'].median() / 60.0 for d in cohorts]
pt_med = [runs[runs['dataset'] == d].groupby('patient_id')['duration_s'].median().median() / 60.0 for d in cohorts]

x = np.arange(len(cohorts))
w = 0.36
b1 = ax_c.bar(x - w/2, ep_med, w, color=pps.PALETTE['navy'], label='Episode-weighted')
b2 = ax_c.bar(x + w/2, pt_med, w, color=pps.PALETTE['amber'], label='Patient-median')
ax_c.set_yscale('log')
ax_c.set_ylim(0.1, 1500)
ax_c.set_xticks(x)
ax_c.set_xticklabels(cohorts, fontsize=pps.FONT_SIZE - 1.2)
ax_c.set_ylabel('Median Duration (min)')
pps.add_subtle_ygrid(ax_c)
for bar in list(b1) + list(b2):
    val = bar.get_height()
    txt = f'{val:.1f}m' if val < 60 else f'{val/60:.1f}h'
    ax_c.text(bar.get_x() + bar.get_width()/2, val * 1.18, txt, ha='center', va='bottom', fontsize=pps.FONT_SIZE - 2.0)
ax_c.legend(frameon=False, loc='upper right', handlelength=1.2)
pps.add_panel_label(ax_c, 'C', x=-0.20, y=1.06)

pps.save_hrs_fig(fig, 'Fig02_cohort_duration_architecture')
