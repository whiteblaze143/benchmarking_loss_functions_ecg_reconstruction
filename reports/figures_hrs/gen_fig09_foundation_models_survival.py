"""Figure 9: 12 Foundation Models & SetOperators + Dynamic Survival (V7-P)."""
import sys
sys.path.insert(0, 'reports/figures_hrs')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import paper_plot_style as pps

pps.apply_hrs_style()

fm = pd.read_csv('experiments/af_density_prediction_v7p/results/foundation_model_probe_benchmark.csv')
surv = pd.read_csv('experiments/af_density_prediction_v7p/results/V7P_survival_benchmarks.csv')

fig = plt.figure(figsize=(10.6, 5.2))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.55, 0.95], wspace=0.34)

# --- Panel A: Grouped Dot Plot across all 12 Pretrained Encoders & SetOperators ---
ax_a = fig.add_subplot(gs[0, 0])
best_fm = fm.groupby(['model_type', 'task_id'])['auroc'].max().unstack()
name_map = {
    'ecgfounder': 'ECGFounder (1-Lead)',
    'echonext': 'EchoNext Minimodel',
    'ecgfm_ked': 'ECG-FM (KED)',
    'f2': 'SetOperator F2 (Cont.)',
    'f0': 'SetOperator F0 (Cat.)',
    'g1': 'SetOperator G1 (Geom.)',
    'control': 'SetOp Control Capacity',
    'nefcore': 'NEFCore Full',
    'dinosr': 'DINO-SR (S4)',
    'jepa': 'JEPA (S4)',
    'clef': 'CLEF Medium',
    'csfm': 'CSFM Tiny',
}
best_fm['label'] = best_fm.index.map(name_map)
best_fm = best_fm.sort_values('P04', ascending=True).reset_index()

ya = np.arange(len(best_fm))
for tid, lbl, col, marker in [
    ('P04', 'P04 Noise Rhythm', pps.PALETTE['vermilion'], 'D'),
    ('P02', 'P02 Persistence', pps.PALETTE['navy'], 'o'),
    ('P03', 'P03 Survival (15m)', pps.PALETTE['teal'], 's'),
    ('P01', 'P01 AF Forecast', pps.PALETTE['amber'], '^'),
]:
    ax_a.scatter(best_fm[tid], ya, s=46, color=col, marker=marker, label=lbl, zorder=3, edgecolors='#222222', linewidths=0.4)

for i, row in best_fm.iterrows():
    ax_a.plot([row['P01'], max(row['P02'], row['P04'])], [i, i], color='#E0E0E0', lw=1.0, zorder=1)
    if row['model_type'] == 'ecgfounder':
        ax_a.text(row['P04'] - 0.015, i, f"{row['P04']:.3f}", ha='right', va='center',
                  fontsize=pps.FONT_SIZE - 1.5, fontweight='bold', color=pps.PALETTE['vermilion'])

ax_a.axvline(0.50, color=pps.PALETTE['slate'], linestyle='--', lw=0.9, zorder=1)
ax_a.set_yticks(ya)
ax_a.set_yticklabels(best_fm['label'], fontsize=pps.FONT_SIZE - 1.0)
ax_a.set_xlabel('5-Fold Cross-Validated AUROC (Instantaneous ECG Alone)')
ax_a.set_xlim(0.47, 1.01)
ax_a.legend(frameon=False, loc='lower right', handlelength=1.2, fontsize=pps.FONT_SIZE - 1.4)
pps.add_panel_label(ax_a, 'A', x=-0.30, y=1.04)

# --- Panel B: Dynamic Episode Termination Survival C-Index (P03, 5 Folds) ---
ax_b = fig.add_subplot(gs[0, 1])
cox_vals = surv['cox_c_index_val'].values
wei_vals = surv['weibull_c_index_val'].values
fm_best_p03 = best_fm['P03'].max()

means = [fm_best_p03, wei_vals.mean(), cox_vals.mean()]
stds = [0.0, wei_vals.std(), cox_vals.std()]
lbls_b = ['Best ECG FM\nAlone', 'Weibull AFT\n(History+Dens)', 'Cox PH\n(History+Dens)']
cols_b = [pps.PALETTE['light_gray'], pps.PALETTE['teal'], pps.PALETTE['navy']]

xb = np.arange(3)
bars = ax_b.bar(xb, means, width=0.52, color=cols_b, edgecolor='#222222', lw=0.7, yerr=stds, capsize=4)
for b, val in zip(bars, means):
    ax_b.text(b.get_x() + b.get_width()/2, val + 0.025, f'{val:.3f}', ha='center', va='bottom', fontsize=pps.FONT_SIZE - 1.2, fontweight='bold')

ax_b.axhline(0.50, color=pps.PALETTE['slate'], linestyle='--', lw=0.9)
ax_b.set_xticks(xb)
ax_b.set_xticklabels(lbls_b, fontsize=pps.FONT_SIZE - 1.3)
ax_b.set_ylabel("Harrell's C-Index (5-Fold Validation)")
ax_b.set_ylim(0.45, 0.84)
pps.add_subtle_ygrid(ax_b)
pps.add_panel_label(ax_b, 'B', x=-0.20, y=1.04)

pps.save_hrs_fig(fig, 'Fig09_foundation_models_and_survival')
