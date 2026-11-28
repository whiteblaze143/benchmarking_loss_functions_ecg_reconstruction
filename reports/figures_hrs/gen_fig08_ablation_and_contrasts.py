"""Figure 8: Longitudinal Episode History vs Instantaneous ECG Foundation Models (V7-P)."""
import sys, json
sys.path.insert(0, 'reports/figures_hrs')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import paper_plot_style as pps

pps.apply_hrs_style()

ab = pd.read_csv('experiments/af_density_prediction_v7p/results/ablation_matrix_results.csv')
with open('experiments/af_density_prediction_v7p/results/mandatory_contrasts_audit.json') as f:
    contrasts = json.load(f)

fig = plt.figure(figsize=(11.4, 4.7))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.25, 1.0], wspace=0.48)

# --- Panel A: Clinical Information Progression Across Tasks ---
ax_a = fig.add_subplot(gs[0, 0])
stage_order = [
    ('X00_prior_only', 'Prior'),
    ('X02', 'Burden'),
    ('X03', 'Episode\nArch'),
    ('X04', '+Density'),
    ('X07_history_density_rr', '+HR / HRV\n(Conv. Full)'),
    ('X08_f5_embeddings_only', 'ECG FM\nAlone'),
    ('X12_full_multimodal_fusion', 'Multimodal\nFusion'),
]
best = ab.groupby(['task_id', 'stage'])['auroc'].max().unstack(0)

task_styles = [
    ('P02', 'P02 Persistence (60s)', pps.PALETTE['navy'], '-o'),
    ('P03', 'P03 Survival (15m)', pps.PALETTE['teal'], '-s'),
    ('P01', 'P01 AF Onset Forecast', pps.PALETTE['amber'], '-^'),
    ('P04', 'P04 Noise Rhythm (10s)', pps.PALETTE['vermilion'], '-D'),
]

x = np.arange(len(stage_order))
stages_raw = [s[0] for s in stage_order]
stages_lbl = [s[1] for s in stage_order]

ax_a.axvline(4.5, color=pps.PALETTE['light_gray'], linestyle='--', lw=1.0, zorder=1)
for tid, lbl, col, fmt in task_styles:
    vals = [best.loc[st, tid] if st in best.index else np.nan for st in stages_raw]
    ax_a.plot(x, vals, fmt, color=col, lw=2.0, markersize=5.5, label=lbl, zorder=3)

ax_a.set_xticks(x)
ax_a.set_xticklabels(stages_lbl, fontsize=pps.FONT_SIZE - 1.5)
ax_a.set_ylabel('5-Fold Cross-Validated AUROC')
ax_a.set_ylim(0.36, 1.03)
pps.add_subtle_ygrid(ax_a)
ax_a.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.48, 1.15), ncol=2, columnspacing=1.0, handlelength=1.4, fontsize=pps.FONT_SIZE - 1.4)
pps.add_panel_label(ax_a, 'A', x=-0.12, y=1.10)

# --- Panel B: Patient-Clustered Bootstrap Contrasts (Delta AUROC, 95% CI) ---
ax_b = fig.add_subplot(gs[0, 1])
contrast_rows = [
    ('P04', 'C4_X07_vs_X12', 'P04: ECG FM vs Tabular'),
    ('P01', 'C2_X06_vs_X07', 'P01: +HR/HRV vs History'),
    ('P02', 'C1_X02_vs_X03', 'P02: Arch vs Burden'),
    ('P03', 'C1_X02_vs_X03', 'P03: Arch vs Burden'),
    ('P01', 'C5_X11_vs_X12', 'P01: +Density over FM+RR'),
    ('P03', 'C3_X06_vs_X11', 'P03: FM+RR vs History'),
    ('P02', 'C3_X06_vs_X11', 'P02: FM+RR vs History'),
]
rows_b = []
for tid, cid, label in contrast_rows:
    d = contrasts[tid][cid]
    rows_b.append({
        'label': label,
        'delta': d['delta_auroc_mean'],
        'lo': d['delta_auroc_ci95_low'],
        'hi': d['delta_auroc_ci95_high'],
        'sig': d['is_statistically_significant'],
    })
cdf = pd.DataFrame(rows_b).iloc[::-1].reset_index(drop=True)

yb = np.arange(len(cdf))
ax_b.axvline(0.0, color=pps.PALETTE['slate'], linestyle='--', lw=1.0, zorder=1)
for i, r in cdf.iterrows():
    col = pps.PALETTE['teal'] if (r['sig'] and r['delta'] > 0) else (
        pps.PALETTE['vermilion'] if (r['sig'] and r['delta'] < 0) else pps.PALETTE['slate']
    )
    ax_b.errorbar(
        r['delta'], i,
        xerr=[[r['delta'] - r['lo']], [r['hi'] - r['delta']]],
        fmt='o', color=col, capsize=3.5, markersize=5.5, lw=1.8, zorder=3
    )
    ax_b.text(r['hi'] + 0.015 if r['delta'] >= 0 else r['lo'] - 0.015, i,
              f"{r['delta']:+.3f}", va='center', ha='left' if r['delta'] >= 0 else 'right',
              fontsize=pps.FONT_SIZE - 1.6, fontweight='bold', color=col)

ax_b.set_yticks(yb)
ax_b.set_yticklabels(cdf['label'], fontsize=pps.FONT_SIZE - 1.3)
ax_b.set_xlabel('Change in AUROC (95% Patient Bootstrap CI)')
ax_b.set_xlim(-0.18, 0.49)
pps.add_panel_label(ax_b, 'B', x=-0.38, y=1.06)

pps.save_hrs_fig(fig, 'Fig08_ablation_and_bootstrap_contrasts')
