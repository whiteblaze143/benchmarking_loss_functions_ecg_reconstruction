"""Figure 6: Low-Burden Phenotype Divergence & Null Structural Substrate Link (SHDB-AF, E03)."""
import sys, json
sys.path.insert(0, 'reports/figures_hrs')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statannotations.Annotator import Annotator
import paper_plot_style as pps

pps.apply_hrs_style()

sub_df = pd.read_csv('experiments/af_density_temporal_phenotypes_v7/results/e03_shdb/shdb_low_burden_subgroups.csv')
coh_df = pd.read_csv('experiments/af_density_temporal_phenotypes_v7/results/e03_shdb/shdb_12h_standardized_cohort.csv')
with open('experiments/af_density_temporal_phenotypes_v7/results/e03_shdb/shdb_lopo_cooks_distance.json') as f:
    lopo = json.load(f)

fig = plt.figure(figsize=(11.0, 4.3))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.0, 1.15], wspace=0.34)

# --- Panel A: Low-Burden Strata Episode Dominance ---
ax_a = fig.add_subplot(gs[0, 0])
strata_keys = ['Nested: <5%', 'Nested: <10%', 'Nested: 10-30%']
strata_lbls = ['<5%\n(n=22)', '<10%\n(n=26)', '10–30%\n(n=15)']
s_sub = sub_df.set_index('stratum').loc[strata_keys]

p_lt6 = 100.0 * s_sub['longest_ep_lt_6m'] / s_sub['n_patients']
p_6_60 = 100.0 * s_sub['longest_ep_6_60m'] / s_sub['n_patients']
p_gt60 = 100.0 * s_sub['longest_ep_gt_60m'] / s_sub['n_patients']

xa = np.arange(len(strata_keys))
ax_a.bar(xa, p_lt6, width=0.52, color=pps.PALETTE['sky'], edgecolor='white', label='Longest <6m')
ax_a.bar(xa, p_6_60, width=0.52, bottom=p_lt6, color=pps.PALETTE['amber'], edgecolor='white', label='Longest 6–60m')
ax_a.bar(xa, p_gt60, width=0.52, bottom=p_lt6 + p_6_60, color=pps.PALETTE['vermilion'], edgecolor='white', label='Longest >60m')

for i in range(len(xa)):
    if p_lt6.iloc[i] >= 12:
        ax_a.text(xa[i], p_lt6.iloc[i] / 2, f'{p_lt6.iloc[i]:.0f}%', ha='center', va='center', color='#111111', fontweight='bold', fontsize=pps.FONT_SIZE - 1.3)
    if p_6_60.iloc[i] >= 12:
        ax_a.text(xa[i], p_lt6.iloc[i] + p_6_60.iloc[i] / 2, f'{p_6_60.iloc[i]:.0f}%', ha='center', va='center', color='#111111', fontweight='bold', fontsize=pps.FONT_SIZE - 1.3)
    if p_gt60.iloc[i] >= 12:
        ax_a.text(xa[i], p_lt6.iloc[i] + p_6_60.iloc[i] + p_gt60.iloc[i] / 2, f'{p_gt60.iloc[i]:.0f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=pps.FONT_SIZE - 1.3)

ax_a.set_xticks(xa)
ax_a.set_xticklabels(strata_lbls)
ax_a.set_xlabel('12h Standardized AF Burden')
ax_a.set_ylabel('Share of Patients (%)')
ax_a.set_ylim(0, 118)
ax_a.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.48, 1.15), columnspacing=0.8, handlelength=1.0, fontsize=pps.FONT_SIZE - 1.7)
pps.add_panel_label(ax_a, 'A', x=-0.18, y=1.08)

# --- Panel B: Echocardiographic LAD by AF Clinical Subtype with statannotations ---
ax_b = fig.add_subplot(gs[0, 1])
echo_df = coh_df.dropna(subset=['Echo_LAD', 'AF_Type']).copy()
order_b = ['non-AF', 'PAF', 'PerAF']
lbl_b = ['Non-AF', 'Paroxysmal', 'Persistent']
sns.boxplot(
    data=echo_df, x='AF_Type', y='Echo_LAD', hue='AF_Type', order=order_b, ax=ax_b,
    palette=[pps.PALETTE['sky'], pps.PALETTE['teal'], pps.PALETTE['vermilion']],
    width=0.48, showfliers=False, boxprops=dict(alpha=0.35), legend=False
)
sns.stripplot(
    data=echo_df, x='AF_Type', y='Echo_LAD', hue='AF_Type', order=order_b, ax=ax_b,
    palette=[pps.PALETTE['sky'], pps.PALETTE['teal'], pps.PALETTE['vermilion']],
    size=4.0, alpha=0.8, jitter=0.16, legend=False
)
annot = Annotator(ax_b, [('non-AF', 'PAF'), ('non-AF', 'PerAF')], data=echo_df, x='AF_Type', y='Echo_LAD', order=order_b)
annot.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=0, fontsize=pps.FONT_SIZE - 1.0)
annot.apply_and_annotate()

ax_b.set_xticks([0, 1, 2])
ax_b.set_xticklabels(lbl_b)
ax_b.set_xlabel('Clinical Subtype')
ax_b.set_ylabel('Left Atrial Diameter (mm)')
pps.add_subtle_ygrid(ax_b)
pps.add_panel_label(ax_b, 'B', x=-0.18, y=1.08)

# --- Panel C: LAD vs Density in <=90d PAF Multi-Run (n=18) + LOPO Cook's D ---
ax_c = fig.add_subplot(gs[0, 2])
paf90 = coh_df[
    (coh_df['Annotated'] == True) &
    (coh_df['AF_Type'] == 'PAF') &
    (coh_df['echo_latency_abs_days'] <= 90) &
    (coh_df['n_runs_12h'] >= 2) &
    (coh_df['Echo_LAD'].notna()) &
    (coh_df['density_12h'].notna())
].copy()

outliers = {o['patient_id'] for o in lopo['threshold_90d']['outlier_records']}
inliers = paf90[~paf90['patient_id'].isin(outliers)]
outs = paf90[paf90['patient_id'].isin(outliers)]

ax_c.scatter(inliers['density_12h'], inliers['Echo_LAD'], s=48, color=pps.PALETTE['navy'], edgecolor='#222222', lw=0.6, label=f'Inliers (n={len(inliers)})', zorder=3)
ax_c.scatter(outs['density_12h'], outs['Echo_LAD'], s=68, marker='D', color=pps.PALETTE['vermilion'], edgecolor='#111111', lw=0.8, label=f"Cook's D > 0.22 (n={len(outs)})", zorder=4)

# Plot simple unadjusted trend lines for visual contrast vs reported adjusted p-values
x_grid = np.linspace(0.42, 1.02, 50)
if len(paf90) >= 3:
    z_all = np.polyfit(paf90['density_12h'], paf90['Echo_LAD'], 1)
    ax_c.plot(x_grid, np.polyval(z_all, x_grid), '--', color=pps.PALETTE['vermilion'], lw=1.6, label=f"All (adj p={lopo['threshold_90d']['original_m3a_density_p']:.2f})")
if len(inliers) >= 3:
    z_clean = np.polyfit(inliers['density_12h'], inliers['Echo_LAD'], 1)
    ax_c.plot(x_grid, np.polyval(z_clean, x_grid), '-', color=pps.PALETTE['navy'], lw=1.6, label=f"Clean (adj p={lopo['threshold_90d']['clean_m3a_density_p']:.2f})")

ax_c.set_xlabel('12h AF Density')
ax_c.set_ylabel('Left Atrial Diameter (mm)')
ax_c.set_xlim(0.38, 1.05)
ax_c.set_ylim(24, 58)
pps.add_subtle_ygrid(ax_c)
ax_c.legend(frameon=False, loc='upper left', handlelength=1.2, fontsize=pps.FONT_SIZE - 1.8)
pps.add_panel_label(ax_c, 'C', x=-0.18, y=1.08)

pps.save_hrs_fig(fig, 'Fig06_shdb_clinical_substrate')
