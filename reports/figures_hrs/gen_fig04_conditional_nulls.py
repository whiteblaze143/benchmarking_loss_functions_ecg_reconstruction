"""Figure 4: Conditional Null Validation (Spatially Shuffled Runs & Duration Ordering)."""
import sys
sys.path.insert(0, 'reports/figures_hrs')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import paper_plot_style as pps

pps.apply_hrs_style()

cn = pd.read_parquet('experiments/af_density_temporal_phenotypes_v7/results/E01_conditional_null_results.parquet')

fig = plt.figure(figsize=(8.6, 4.0))
gs = GridSpec(1, 2, figure=fig, wspace=0.30)

# --- Panel A: Null A (Spatially Shuffled Brief Runs) Pre & Post Z-scores ---
ax_a = fig.add_subplot(gs[0, 0])
na = cn[cn['null_type'] == 'Null_A']
plot_a = pd.DataFrame({
    'Window': ['Pre-Onset\n0–15 m'] * len(na) + ['Post-Termination\n0–15 m'] * len(na),
    'Z': np.concatenate([na['z_score_pre'].values, na['z_score_post'].values])
}).dropna()

ax_a.axhline(0.0, color=pps.PALETTE['slate'], linestyle='--', linewidth=1.0, label='Null mean (Z = 0)')
ax_a.axhline(1.96, color=pps.PALETTE['vermilion'], linestyle=':', linewidth=1.1, label='Z = 1.96 (p < 0.05)')

sns.violinplot(
    data=plot_a, x='Window', y='Z', hue='Window', ax=ax_a,
    palette=[pps.PALETTE['navy'], pps.PALETTE['vermilion']],
    inner=None, cut=0, alpha=0.25, linewidth=0.9, legend=False
)
sns.stripplot(
    data=plot_a, x='Window', y='Z', hue='Window', ax=ax_a,
    palette=[pps.PALETTE['navy'], pps.PALETTE['vermilion']],
    size=4.2, alpha=0.75, jitter=0.18, legend=False
)
# Annotate patient summary stats cleanly
n_pre = na['z_score_pre'].notna().sum()
m_pre = na['z_score_pre'].mean()
m_post = na['z_score_post'].mean()
ax_a.text(0, 17.5, f'Mean Z = +{m_pre:.2f}\n(n = {n_pre} pts)', ha='center', va='top', fontsize=pps.FONT_SIZE - 1.4, color=pps.PALETTE['navy'], fontweight='bold')
ax_a.text(1, 17.5, f'Mean Z = +{m_post:.2f}\n(n = {n_pre} pts)', ha='center', va='top', fontsize=pps.FONT_SIZE - 1.4, color=pps.PALETTE['vermilion'], fontweight='bold')

ax_a.set_xlabel('')
ax_a.set_ylabel('Patient Clustering Z-Score vs Spatial Null')
ax_a.set_ylim(-7.5, 22.0)
pps.add_subtle_ygrid(ax_a)
ax_a.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03), handlelength=1.2)
pps.add_panel_label(ax_a, 'A', x=-0.16, y=1.06)

# --- Panel B: Null B (Permuted Episode Ordering) Z-scores ---
ax_b = fig.add_subplot(gs[0, 1])
nb = cn[cn['null_type'] == 'Null_B'].dropna(subset=['z_score_ordering'])
ax_b.axhline(0.0, color=pps.PALETTE['slate'], linestyle='--', linewidth=1.0)
ax_b.axhline(1.96, color=pps.PALETTE['vermilion'], linestyle=':', linewidth=1.1)
ax_b.axhline(-1.96, color=pps.PALETTE['vermilion'], linestyle=':', linewidth=1.1)

sns.violinplot(
    y=nb['z_score_ordering'], ax=ax_b, color=pps.PALETTE['teal'],
    inner='box', cut=0, alpha=0.28, linewidth=1.0, width=0.45
)
sns.stripplot(
    y=nb['z_score_ordering'], ax=ax_b, color=pps.PALETTE['teal'],
    size=4.5, alpha=0.75, jitter=0.12
)
ax_b.text(0.28, 1.5, f'Median Z = {nb["z_score_ordering"].median():.2f}\n(Null not rejected;\nn = {len(nb)} pts)',
          ha='left', va='top', fontsize=pps.FONT_SIZE - 1.3, color='#222222')
ax_b.set_xlim(-0.5, 0.9)
ax_b.set_ylim(-3.5, 3.2)
ax_b.set_xticks([0])
ax_b.set_xticklabels(['Duration-Ordering\nPermutation Null'])
ax_b.set_ylabel('Sequence Ordering Z-Score')
pps.add_subtle_ygrid(ax_b)
pps.add_panel_label(ax_b, 'B', x=-0.18, y=1.06)

pps.save_hrs_fig(fig, 'Fig04_conditional_null_validation')
