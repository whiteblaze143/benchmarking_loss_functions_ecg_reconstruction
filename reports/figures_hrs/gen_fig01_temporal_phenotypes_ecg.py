"""Figure 1: Divergent Temporal Phenotypes at Matched AF Burden + Authentic ECG Grid."""
import sys
sys.path.insert(0, 'reports/figures_hrs')
sys.path.insert(0, 'experiments/af_density_temporal_phenotypes_v7/src')
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import paper_plot_style as pps
from density_continuous import compute_continuous_density

pps.apply_hrs_style()

runs = pd.read_parquet('experiments/af_density_continuity_v5/manifests/canonical_af_runs.parquet')
inv = pd.read_csv('experiments/af_density_continuity_v5/manifests/canonical_cohort_inventory_reconciled.csv')
arch = pd.read_csv('experiments/af_density_temporal_phenotypes_v7/results/E01_density_architecture_results.csv')


def merge_overlapping_intervals(intervals):
    if not intervals:
        return []
    s = sorted(intervals)
    merged = [list(s[0])]
    for a, b in s[1:]:
        if a <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


# Compute exact patient-level burden, Charitos density & longest-episode share for all 3 cohorts
# (using primary record per patient or total recording duration with exact continuous solver)
pt_rows = []
for (ds, pt), p_inv in inv.groupby(['dataset', 'patient_id']):
    # Take longest recording for multi-record patients so intervals sit on a single contiguous timeline
    rec_row = p_inv.sort_values('monitoring_duration_h', ascending=False).iloc[0]
    rec_id = str(rec_row['record_id'])
    obs_s = float(rec_row['monitoring_duration_h']) * 3600.0
    p_runs = runs[(runs['dataset'] == ds) & (runs['record_id'].astype(str) == rec_id)]
    if p_runs.empty or obs_s <= 0:
        continue
    ivs = merge_overlapping_intervals([
        (max(0.0, float(a)), min(obs_s, float(b)))
        for a, b in zip(p_runs['start_time_s'], p_runs['end_time_s']) if b > a
    ])
    tot_af = sum(b - a for a, b in ivs)
    burden = tot_af / obs_s
    dens = compute_continuous_density(ivs, obs_s)
    if dens is None:
        continue
    longest_share = max(b - a for a, b in ivs) / tot_af if tot_af > 0 else 0.0
    pt_rows.append({
        'dataset': ds, 'patient_id': pt,
        'burden': burden, 'density': dens, 'longest_share': longest_share
    })
pt_df = pd.DataFrame(pt_rows)

fig = plt.figure(figsize=(10.5, 8.2))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.05, 1.0, 1.35], hspace=0.45, wspace=0.28)

# --- Panel A: Paired 24h timelines at matched ~15-17% burden ---
ax_a = fig.add_subplot(gs[0, :])
p_disp = arch[arch['patient_id'] == 'ltafdb_53'].iloc[0]
p_cons = arch[arch['patient_id'] == 'ltafdb_118'].iloc[0]
r_disp = runs[runs['patient_id'] == 'ltafdb_53']
r_cons = runs[runs['patient_id'] == 'ltafdb_118']

obs_disp_h = p_disp['observation_duration_h']
obs_cons_h = p_cons['observation_duration_h']

ax_a.barh(1.0, obs_disp_h, height=0.42, color='#F2F4F7', edgecolor='#CCCCCC', linewidth=0.7)
for _, row in r_disp.iterrows():
    s_h = row['start_time_s'] / 3600.0
    d_h = max(row['duration_s'] / 3600.0, 0.06)
    ax_a.barh(1.0, d_h, left=s_h, height=0.42, color=pps.PALETTE['vermilion'], edgecolor='none')

ax_a.barh(0.0, obs_cons_h, height=0.42, color='#F2F4F7', edgecolor='#CCCCCC', linewidth=0.7)
for _, row in r_cons.iterrows():
    s_h = row['start_time_s'] / 3600.0
    d_h = max(row['duration_s'] / 3600.0, 0.06)
    ax_a.barh(0.0, d_h, left=s_h, height=0.42, color=pps.PALETTE['navy'], edgecolor='none')

ax_a.set_yticks([0.0, 1.0])
ax_a.set_yticklabels([
    f"Consolidated\n(Burden {p_cons['af_burden']*100:.1f}%, Density {p_cons['continuous_density']:.2f},\n1 episode = {p_cons['longest_episode_h']:.1f} h)",
    f"Fragmented\n(Burden {p_disp['af_burden']*100:.1f}%, Density {p_disp['continuous_density']:.2f},\n{int(p_disp['n_brief_runs'])} brief runs, max {p_disp['longest_episode_h']*60:.1f} m)"
], fontsize=pps.FONT_SIZE - 0.8)
ax_a.set_xlabel('Ambulatory Recording Time (h)')
ax_a.set_xlim(0, 24.2)
ax_a.set_ylim(-0.45, 1.55)
ax_a.spines['left'].set_visible(False)
ax_a.tick_params(axis='y', length=0)
pps.add_panel_label(ax_a, 'A', x=-0.16, y=1.06)

# --- Panel B: Authentic 10s Holter strip crossing AF onset (IRIDIA record_095) ---
ax_b = fig.add_subplot(gs[1, :])
with h5py.File('/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1/ecg_hdf5/record_095/record_095_ecg_00.h5', 'r') as f:
    fs = 200.0
    t_win_start = 827.5
    raw_seg = f['ecg'][int(t_win_start * fs):int((t_win_start + 10.0) * fs), 0]
seg_mv = pps.highpass_center_ecg(raw_seg, fs=fs)
pps.draw_clinical_ecg_strip(
    ax_b, seg_mv, fs=fs, t_start=0.0,
    label='IRIDIA-AF Holter Transition (Lead I)',
    highlight_span=(831.90 - t_win_start, 10.0),
    highlight_label='AF Onset (Irregular RR)'
)
ax_b.set_xlabel('Elapsed Strip Time (s)')
pps.add_panel_label(ax_b, 'B', x=-0.04, y=1.10)

# --- Panel C: Burden vs Charitos Density across all 3 cohorts ---
sub_gs = gs[2, :].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.055], wspace=0.20)
cohorts = ['IRIDIA-AF', 'LTAFDB', 'SHDB-AF']
sc_ref = None
for i, ds in enumerate(cohorts):
    ax_c = fig.add_subplot(sub_gs[0, i])
    sub = pt_df[pt_df['dataset'] == ds]
    sc_ref = ax_c.scatter(
        sub['burden'] * 100.0,
        sub['density'],
        c=sub['longest_share'],
        cmap='viridis', vmin=0, vmax=1,
        s=32, alpha=0.88, edgecolors='#222222', linewidths=0.45, zorder=3
    )
    pps.add_subtle_ygrid(ax_c)
    ax_c.set_xlim(-4, 104)
    ax_c.set_ylim(-0.04, 1.06)
    ax_c.set_xlabel('AF Burden (%)')
    if i == 0:
        ax_c.set_ylabel('AF Density')
        pps.add_panel_label(ax_c, 'C', x=-0.22, y=1.10)
    else:
        ax_c.set_yticklabels([])
    ax_c.text(0.04, 0.05, f'{ds} (n={len(sub)})', transform=ax_c.transAxes,
              ha='left', va='bottom', fontsize=pps.FONT_SIZE - 1.3, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='#DDDDDD', alpha=0.92))

cax = fig.add_subplot(sub_gs[0, 3])
cbar = fig.colorbar(sc_ref, cax=cax)
cbar.set_label('Longest-Episode Share', fontsize=pps.FONT_SIZE - 0.8)
cbar.ax.tick_params(labelsize=pps.FONT_SIZE - 1.4)

pps.save_hrs_fig(fig, 'Fig01_temporal_phenotypes_and_ecg')
