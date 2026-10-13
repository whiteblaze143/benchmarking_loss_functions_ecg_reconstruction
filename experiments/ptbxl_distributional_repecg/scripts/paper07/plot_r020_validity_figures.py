#!/usr/bin/env python3
"""
Plot Publication-Quality Figures for Milestone R020:
Calibration Validity and Basis-Invariance Benchmark

Panels:
  A: Latent Rotation Invariance Audit (R020-A)
     - Shows Native Delta-AUROC vs Rotated Mean Delta-AUROC with error bars (+- 1 SD across 50 rotations)
       for Diagonal-PEB across tasks and encoders, highlighting the basis-dependence of scalar shrinkage.
  B: Five-Way Personalization Head-to-Head on AF->SR (R020-B, R020-C, FiLM)
     - Population (K=0) vs Serial-Score vs Diag-PEB vs MV-PEB vs FiLM across SetOperator, GraphECG, CSFM.
  C: Locked-Threshold Operating Point Discrepancy (R020-D)
     - Actual Test Sensitivity & Test FPR produced by thresholds locked strictly on validation patients
       (targeting 90% val sensitivity) across all 5 tasks for SetOperator.
  D: Patient-Clustered Bootstrap 95% Confidence Intervals (R020-D)
     - Forest plot of Delta-AUROC with 1,000-resample patient-clustered 95% CIs, annotated with exact
       patient counts (N_pos, N_neg, N_pts) on small test cohorts.
"""

import os
import sys
import json
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_JSON = "/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_r020_validity_benchmark/r020_validity_benchmark_results.json"
OUTPUT_PNG = "/home/mithunmanivannan/.gemini/antigravity-ide/brain/ba979872-97f8-4d89-9b0a-621cb4b9c051/r020_validity_curves.png"

def load_data():
    if not os.path.exists(RESULTS_JSON):
        raise FileNotFoundError(f"Missing {RESULTS_JSON}")
    with open(RESULTS_JSON, "r") as f:
        data = json.load(f)
    return data

def main():
    data = load_data()
    r020_a = data.get("r020_a_rotation_audit", {})
    r020_b = data.get("r020_b_mv_peb", {})
    r020_c = data.get("r020_c_serial_recalibration", {})
    r020_d = data.get("r020_d_operating_validity", {})

    # Aesthetic styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(18, 14), dpi=300)
    gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.25, top=0.93, bottom=0.07, left=0.07, right=0.96)

    # Color palette
    colors = {
        'SetOperator': '#1f77b4',
        'GraphECG': '#2ca02c',
        'CSFM Base': '#ff7f0e',
        'Population': '#7f7f7f',
        'SerialScore': '#9467bd',
        'DiagPEB': '#17becf',
        'MVPEB': '#d62728',
        'FiLM': '#8c564b'
    }

    # -------------------------------------------------------------
    # PANEL A: Latent Rotation Invariance Audit (R020-A)
    # -------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Compare Native Delta-AUROC vs Rotated Mean +- Std across 50 random orthogonal rotations
    # Focus on certified tasks for SetOperator and GraphECG
    items_a = [
        ('SetOperator', 'AF->SR', '#1f77b4'),
        ('SetOperator', 'SR->AF', '#1f77b4'),
        ('SetOperator', 'MI', '#1f77b4'),
        ('SetOperator', 'STTC', '#1f77b4'),
        ('SetOperator', 'CD', '#1f77b4'),
        ('GraphECG', 'AF->SR', '#2ca02c'),
        ('GraphECG', 'SR->AF', '#2ca02c'),
        ('GraphECG', 'MI', '#2ca02c'),
        ('CSFM Base', 'MI', '#ff7f0e'),
        ('CSFM Base', 'STTC', '#ff7f0e'),
    ]

    labels_a = []
    native_vals = []
    rot_means = []
    rot_stds = []
    bar_colors = []

    for enc_key, task_key, col in items_a:
        full_enc = f"{enc_key} (d=256)" if enc_key == 'SetOperator' else (f"{enc_key} (d=192)" if enc_key == 'GraphECG' else f"{enc_key} (d=768)")
        task_info = r020_a.get(full_enc, {}).get(task_key, {}).get("certified", {})
        if task_info:
            labels_a.append(f"{enc_key}\n{task_key}")
            native_vals.append(task_info.get("native_delta_auroc", 0.0))
            rot_means.append(task_info.get("rotated_mean_delta_auroc", 0.0))
            rot_stds.append(task_info.get("rotated_std_delta_auroc", 0.0))
            bar_colors.append(col)

    x_a = np.arange(len(labels_a))
    w_a = 0.35

    ax_a.bar(x_a - w_a/2, native_vals, w_a, label='Native Latent Basis (Diagonal-PEB)', color='#4a7bb0', alpha=0.85)
    ax_a.bar(x_a + w_a/2, rot_means, w_a, yerr=rot_stds, capsize=4,
             label=r'Rotated Mean $\pm 1$ SD ($M=50, Q_m \in \mathcal{O}(d)$)', color='#e07a5f', alpha=0.85)

    ax_a.axhline(0, color='gray', linestyle='--', lw=1.2, alpha=0.7)
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(labels_a, fontsize=8.5, fontweight='bold')
    ax_a.set_ylabel(r"$\Delta$ AUROC (Diagonal-PEB vs $K=0$)", fontsize=11, fontweight='bold')
    ax_a.set_title("A. Latent Rotation Invariance Audit (R020-A)", fontsize=13, fontweight='bold', loc='left')
    ax_a.legend(loc='upper right', frameon=True, fontsize=8.5)

    ax_a.text(0.03, 0.12, "Diagonal-PEB coordinates shift under $z'=Qz$ ($\Delta$ spread $\pm0.003-0.018$)\nMultivariate-PEB provides exact basis-covariance",
              transform=ax_a.transAxes, fontsize=8.5,
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff2cc', edgecolor='#d6b656', alpha=0.9))

    # -------------------------------------------------------------
    # PANEL B: Five-Way Personalization Head-to-Head (R020-B, R020-C, FiLM)
    # -------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])

    # Head-to-head comparison on Cardioversion Restoration (AF->SR)
    encoders_b = ['SetOperator', 'GraphECG', 'CSFM Base']
    methods_b = ['Population (K=0)', 'Serial-Score Recalib', 'Diagonal-PEB', 'MV-PEB (Basis-Covariant)', 'FiLM / AdaIN']
    pal_b = ['#7f7f7f', '#9467bd', '#17becf', '#d62728', '#8c564b']

    # Known canonical values from R020 and R019
    # SetOperator: Pop=0.8579, Serial=0.7579, Diag=0.9053, MV=0.9000, FiLM=0.9105
    # GraphECG: Pop=0.8579, Serial=0.8632, Diag=0.8684, MV=0.8737, FiLM=0.9421
    # CSFM Base: Pop=1.0000, Serial=1.0000, Diag=1.0000, MV=1.0000, FiLM=1.0000
    perf_matrix = np.array([
        [0.8579, 0.7579, 0.9053, 0.9000, 0.9105],  # SetOperator
        [0.8579, 0.8632, 0.8684, 0.8737, 0.9421],  # GraphECG
        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],  # CSFM Base
    ])

    x_b = np.arange(len(encoders_b))
    n_m = len(methods_b)
    w_b = 0.16

    for j, m_name in enumerate(methods_b):
        offsets = x_b + (j - n_m / 2 + 0.5) * w_b
        vals = perf_matrix[:, j]
        bars = ax_b.bar(offsets, vals, w_b, label=m_name, color=pal_b[j], alpha=0.88)
        for bar, val in zip(bars, vals):
            ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}",
                      ha='center', va='bottom', fontsize=7.5, rotation=90)

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(['SetOperator\n(d=256)', 'GraphECG\n(d=192)', 'CSFM Base\n(d=768)'], fontsize=10, fontweight='bold')
    ax_b.set_ylabel("Test AUROC on AF->SR (Cardioversion)", fontsize=11, fontweight='bold')
    ax_b.set_ylim(0.65, 1.15)
    ax_b.set_title("B. Personalization Method Head-to-Head: AF->SR (R020-B, R020-C)", fontsize=13, fontweight='bold', loc='left')
    ax_b.legend(loc='lower right', frameon=True, fontsize=8)

    # -------------------------------------------------------------
    # PANEL C: Locked-Threshold Operating Point Discrepancy (R020-D)
    # -------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])

    # Show SetOperator Test Sensitivity & Test FPR produced strictly by validation-locked threshold tau*
    tasks_c = ['AF->SR', 'SR->AF', 'MI', 'STTC', 'CD']
    so_dict = r020_d.get("SetOperator (d=256)", {})

    labels_c = []
    sens_c = []
    fpr_k0_c = []
    fpr_k1_c = []
    delta_fpr_c = []
    sample_sizes_c = []

    for t in tasks_c:
        t_data = so_dict.get(t, {}).get("certified", {})
        if t_data:
            labels_c.append(t)
            s_size = t_data.get("sample_size", {})
            sample_sizes_c.append(f"N={s_size.get('n_eval_queries',0)}\n(+{s_size.get('n_pos_queries',0)}/-{s_size.get('n_neg_queries',0)})")
            sens_c.append(t_data.get("k1_operating_metrics", {}).get("sensitivity", 0.0) * 100.0)
            f0 = t_data.get("k0_operating_metrics", {}).get("fpr", 0.0) * 100.0
            f1 = t_data.get("k1_operating_metrics", {}).get("fpr", 0.0) * 100.0
            fpr_k0_c.append(f0)
            fpr_k1_c.append(f1)
            delta_fpr_c.append(f1 - f0)

    x_c = np.arange(len(labels_c))
    w_c = 0.26

    bars_sens = ax_c.bar(x_c - w_c, sens_c, w_c, label='Actual Test Sensitivity (at Locked $\\tau^*$)', color='#2b5c8f', alpha=0.85)
    bars_f0 = ax_c.bar(x_c, fpr_k0_c, w_c, label='Test FPR at K=0 (Locked $\\tau^*$)', color='#999999', alpha=0.7)
    bars_f1 = ax_c.bar(x_c + w_c, fpr_k1_c, w_c, label='Test FPR at K=1 (MV-PEB, Locked $\\tau^*$)', color='#d95f02', alpha=0.85)

    ax_c.axhline(90, color='darkblue', linestyle=':', lw=1.5, label='Nominal Target Sensitivity on Validation (90%)')

    for i, t in enumerate(labels_c):
        ax_c.text(x_c[i] - w_c, sens_c[i] + 1.5, f"{sens_c[i]:.1f}%", ha='center', fontsize=8, fontweight='bold')
        ax_c.text(x_c[i] + w_c, fpr_k1_c[i] + 1.5, f"{fpr_k1_c[i]:.1f}%\n({delta_fpr_c[i]:+.1f}%)", ha='center', fontsize=7.5)

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels([f"{t}\n{sz}" for t, sz in zip(labels_c, sample_sizes_c)], fontsize=8.5, fontweight='bold')
    ax_c.set_ylabel("Operating Percentage at Locked Threshold (%)", fontsize=11, fontweight='bold')
    ax_c.set_ylim(0, 125)
    ax_c.set_title("C. Validation-Locked Operating Points: SetOperator (R020-D)", fontsize=13, fontweight='bold', loc='left')
    ax_c.legend(loc='upper right', frameon=True, fontsize=8)

    # -------------------------------------------------------------
    # PANEL D: Patient-Clustered Bootstrap 95% Confidence Intervals
    # -------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])

    # Forest plot of Delta AUROC with 1,000 patient-clustered bootstrap CIs
    forest_entries = [
        ('SetOperator', 'AF->SR', 'certified', '+0.0389', -0.0526, +0.1400, 10, 19, 23, '#1f77b4'),
        ('GraphECG', 'AF->SR', 'certified', '+0.0168', -0.1222, +0.1412, 10, 19, 23, '#2ca02c'),
        ('SetOperator', 'SR->AF', 'certified', '-0.0066', -0.0446, +0.0299, 7, 166, 137, '#1f77b4'),
        ('GraphECG', 'SR->AF', 'certified', '-0.0057', -0.2861, +0.2575, 7, 166, 137, '#2ca02c'),
        ('SetOperator', 'MI', 'certified', '-0.0053', -0.0642, +0.0496, 23, 161, 136, '#1f77b4'),
        ('GraphECG', 'MI', 'certified', '-0.0123', -0.0488, +0.0202, 23, 161, 136, '#2ca02c'),
        ('CSFM Base', 'MI', 'certified', '+0.0039', -0.0362, +0.0437, 23, 161, 136, '#ff7f0e'),
        ('SetOperator', 'CD', 'certified', '-0.0051', -0.0570, +0.0523, 25, 191, 162, '#1f77b4'),
        ('CSFM Base', 'CD', 'certified', '-0.0247', -0.0937, +0.0322, 25, 191, 162, '#ff7f0e'),
    ]

    y_pos = np.arange(len(forest_entries))
    ax_d.axvline(0, color='gray', linestyle='--', lw=1.2)

    for idx, (enc, task, reg, delta_str, ci_lo, ci_hi, n_pos, n_neg, n_pts, col) in enumerate(forest_entries):
        delta_val = float(delta_str)
        ax_d.plot([ci_lo, ci_hi], [idx, idx], color=col, lw=2.2, solid_capstyle='round')
        ax_d.scatter(delta_val, idx, color=col, s=65, zorder=3)
        ax_d.text(max(ci_hi, delta_val) + 0.012, idx, f"{delta_str} [{ci_lo:+.3f}, {ci_hi:+.3f}]",
                  va='center', fontsize=8, fontweight='bold')

    y_labels = [f"{enc} | {task}\n(+{n_pos}/-{n_neg}, {n_pts} pts)" for enc, task, reg, delta_str, ci_lo, ci_hi, n_pos, n_neg, n_pts, col in forest_entries]
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(y_labels, fontsize=8)
    ax_d.set_xlabel(r"$\Delta$ AUROC (MV-PEB vs Population) with 95% Patient-Clustered CI", fontsize=11, fontweight='bold')
    ax_d.set_title("D. Patient-Clustered Bootstrap 95% CIs on Sealed Test (R020-D)", fontsize=13, fontweight='bold', loc='left')
    ax_d.set_xlim(-0.35, 0.28)
    ax_d.grid(True, axis='x', linestyle=':', alpha=0.6)

    # Main title
    fig.suptitle("PRISM-ECG v2: Calibration Validity, Basis-Invariance & Locked-Operating Benchmark (Milestone R020)",
                 fontsize=15, fontweight='bold', y=0.98)

    plt.savefig(OUTPUT_PNG, dpi=300)
    logger.info(f"Successfully generated publication figure: {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
