#!/usr/bin/env python3
"""Generate publication-grade 12-lead clinical ECG gallery overlays
for all 24 selected RDB cases (8 rhythms x 3 tiers: worst, median, best),
plus tri-way failure mode diagnostic plots for the 8 worst cases.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results/rdb_gallery"
OUT_FIGURES_DIR = ROOT / "book/figures/gallery"
OUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_GRID_POS = {
    # 4 columns x 3 rows
    # Col 0: Limb leads I, II, III
    "I": (0, 0),
    "II": (1, 0),
    "III": (2, 0),
    # Col 1: Augmented limb leads aVR, aVL, aVF
    "aVR": (0, 1),
    "aVL": (1, 1),
    "aVF": (2, 1),
    # Col 2: Anterior/Septal precordial V1, V2, V3
    "V1": (0, 2),
    "V2": (1, 2),
    "V3": (2, 2),
    # Col 3: Lateral precordial V4, V5, V6
    "V4": (0, 3),
    "V5": (1, 3),
    "V6": (2, 3),
}

RHYTHM_FULL_NAMES = {
    "AF": "Atrial Flutter",
    "AFIB": "Atrial Fibrillation",
    "AT": "Atrial Tachycardia",
    "SA": "Sinus Arrhythmia",
    "SB": "Sinus Bradycardia",
    "SR": "Normal Sinus Rhythm",
    "ST": "Sinus Tachycardia",
    "SVT": "Supraventricular Tachycardia",
}


def draw_ecg_grid(ax: plt.Axes, t_min: float, t_max: float, v_min: float, v_max: float) -> None:
    """Draw authentic medical ECG grid lines (25 mm/s, 10 mm/mV)."""
    # 1 mm = 0.04 s horizontally, 0.1 mV vertically
    # 5 mm = 0.20 s horizontally, 0.5 mV vertically
    t_start = math.floor(t_min / 0.04) * 0.04
    t_end = math.ceil(t_max / 0.04) * 0.04
    v_start = math.floor(v_min / 0.1) * 0.1
    v_end = math.ceil(v_max / 0.1) * 0.1

    # Minor vertical grid (0.04 s)
    t_minor = np.arange(t_start, t_end + 0.01, 0.04)
    for t in t_minor:
        is_major = np.isclose(t % 0.20, 0.0, atol=1e-3) or np.isclose(t % 0.20, 0.20, atol=1e-3)
        ax.axvline(
            t,
            color="#F48FB1" if is_major else "#FCE4EC",
            linewidth=0.75 if is_major else 0.35,
            zorder=0,
        )

    # Horizontal grid (0.1 mV)
    v_minor = np.arange(v_start, v_end + 0.01, 0.1)
    for v in v_minor:
        is_major = np.isclose(v % 0.5, 0.0, atol=1e-3) or np.isclose(v % 0.5, 0.5, atol=1e-3)
        ax.axhline(
            v,
            color="#F48FB1" if is_major else "#FCE4EC",
            linewidth=0.75 if is_major else 0.35,
            zorder=0,
        )


def plot_12lead_overlay(
    case_meta: dict,
    true_12: np.ndarray,
    pred_12: np.ndarray,
    out_prefix: Path,
    t_span: float = 3.5,  # show first 3.5 seconds in 12-lead grid
    fs: int = 500,
) -> None:
    """Plot standard 12-lead clinical ECG layout with overlay."""
    n_samples_box = int(t_span * fs)
    t_box = np.arange(n_samples_box) / fs
    t_full = np.arange(5000) / fs

    rhythm = case_meta["rhythm"]
    tier = case_meta["tier"]
    rec_id = case_meta["record_id"]
    pat_id = case_meta["patient_id"]
    mean_r = case_meta["a0_mean_missing_r"]
    r_limb = case_meta["a0_r_limb_missing"]
    r_prec = case_meta["a0_r_precordial"]

    # Overall figure size: 16 x 10 inches
    fig = plt.figure(figsize=(16, 10.5), facecolor="#FFFDFD", dpi=300)
    gs = fig.add_gridspec(
        nrows=4,
        ncols=4,
        height_ratios=[1.0, 1.0, 1.0, 0.75],
        hspace=0.22,
        wspace=0.15,
        left=0.05,
        right=0.98,
        top=0.90,
        bottom=0.06,
    )

    # Header title banner
    title_tier_color = "#C62828" if tier == "worst" else ("#1565C0" if tier == "best" else "#2E7D32")
    full_rhy_name = RHYTHM_FULL_NAMES.get(rhythm, rhythm)
    title_text = (
        f"RDB Case Study: {rhythm} ({full_rhy_name}) — [{tier.upper()} CASE]  •  Record: {rec_id} (Patient: {pat_id})\n"
        f"Missing-Lead Mean Pearson r = {mean_r:.4f}  |  Limb Leads r = {r_limb:.4f}  |  Precordial Leads r = {r_prec:.4f}"
    )
    fig.suptitle(
        title_text,
        fontsize=13,
        fontweight="bold",
        color=title_tier_color,
        y=0.96,
        ha="center",
    )

    # Voltage display limits
    y_min, y_max = -1.2, 1.8

    # Plot 12 leads in 3x4 grid
    for l_idx, l_name in enumerate(LEAD_NAMES):
        row, col = LEAD_GRID_POS[l_name]
        ax = fig.add_subplot(gs[row, col])

        # Draw medical ECG grid
        draw_ecg_grid(ax, 0.0, t_span, y_min, y_max)

        # Signal curves
        sig_true = true_12[l_idx, :n_samples_box]
        sig_pred = pred_12[l_idx, :n_samples_box]

        # Pearson r for this lead
        var_t = np.var(sig_true)
        var_p = np.var(sig_pred)
        if var_t > 1e-12 and var_p > 1e-12:
            r_val = float(np.corrcoef(sig_true, sig_pred)[0, 1])
        else:
            r_val = 0.0

        if l_name == "I":
            # Observed lead input
            ax.plot(t_box, sig_true, color="#00838F", linewidth=1.4, label="Lead I (Observed Input)", zorder=3)
            rect = patches.Rectangle(
                (0.01, 0.02),
                0.98,
                0.96,
                transform=ax.transAxes,
                fill=False,
                edgecolor="#00838F",
                linewidth=1.8,
                linestyle="--",
                zorder=4,
            )
            ax.add_patch(rect)
            badge_text = f"Lead {l_name} [OBSERVED SENSOR]"
            badge_color = "#00838F"
        else:
            # Ground truth
            ax.plot(t_box, sig_true, color="#212121", linewidth=1.1, alpha=0.9, label="Ground Truth (True)", zorder=2)
            # Reconstruction
            ax.plot(t_box, sig_pred, color="#D32F2F", linewidth=1.2, alpha=0.85, label="Reconstruction (A0_raw)", zorder=3)
            badge_text = f"Lead {l_name} (r = {r_val:.3f})"
            badge_color = "#212121" if r_val >= 0.70 else ("#E65100" if r_val >= 0.40 else "#B71C1C")

        # Lead label badge
        ax.text(
            0.03,
            0.88,
            badge_text,
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            color=badge_color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor=badge_color, alpha=0.9),
            zorder=5,
        )

        ax.set_xlim(0, t_span)
        ax.set_ylim(y_min, y_max)
        ax.set_facecolor("#FFFDFD")
        ax.tick_params(labelsize=7.5, direction="in", length=2, color="#BDBDBD")
        if row != 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (seconds)", fontsize=8, color="#424242")

    # Bottom Row: Full 10-second Lead II Rhythm Strip
    ax_rhythm = fig.add_subplot(gs[3, :])
    draw_ecg_grid(ax_rhythm, 0.0, 10.0, -1.2, 1.8)

    sig_ii_true = true_12[1]
    sig_ii_pred = pred_12[1]
    r_ii = float(np.corrcoef(sig_ii_true, sig_ii_pred)[0, 1])

    ax_rhythm.plot(t_full, sig_ii_true, color="#212121", linewidth=0.95, alpha=0.85, label="Ground Truth True Lead II", zorder=2)
    ax_rhythm.plot(t_full, sig_ii_pred, color="#D32F2F", linewidth=1.05, alpha=0.85, label="Reconstructed Lead II (A0_raw)", zorder=3)

    ax_rhythm.text(
        0.01,
        0.82,
        f"Continuous 10-Second Rhythm Strip — Lead II (r = {r_ii:.3f})",
        transform=ax_rhythm.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color="#212121",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#BDBDBD", alpha=0.95),
        zorder=5,
    )
    ax_rhythm.set_xlim(0, 10.0)
    ax_rhythm.set_ylim(y_min, y_max)
    ax_rhythm.set_facecolor("#FFFDFD")
    ax_rhythm.set_xlabel("Full 10-Second Continuous Recording Time (seconds)  •  Grid: 25 mm/s, 10 mm/mV", fontsize=8.5, color="#424242")
    ax_rhythm.tick_params(labelsize=8, direction="in", length=2, color="#BDBDBD")

    # Legend
    lines = [
        plt.Line2D([0], [0], color="#00838F", lw=2, linestyle="--", label="Input Lead I (Observed Sensor)"),
        plt.Line2D([0], [0], color="#212121", lw=1.5, label="Ground Truth 12-Lead Physical Waveform"),
        plt.Line2D([0], [0], color="#D32F2F", lw=1.5, label="Synthesized Reconstructed Waveform (conv15e_A0_raw)"),
    ]
    ax_rhythm.legend(handles=lines, loc="upper right", fontsize=8.5, framealpha=0.95, facecolor="#FFFFFF")

    # Save PNG and SVG
    png_path = out_prefix.with_suffix(".png")
    svg_path = out_prefix.with_suffix(".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {png_path.name}")


def plot_triway_worst_comparison(
    case_meta: dict,
    true_12: np.ndarray,
    a0_12: np.ndarray,
    r7_12: np.ndarray,
    out_path: Path,
    fs: int = 500,
) -> None:
    """Plot side-by-side tri-way comparison for worst-case failures:
    Ground Truth vs A0_raw vs R7_wavelet across key diagnostic leads: I, II, III, aVF, V1, V5.
    """
    leads_to_plot = ["I", "II", "III", "aVF", "V1", "V5"]
    n_samples = 1500  # 3.0 seconds
    t = np.arange(n_samples) / fs

    rhythm = case_meta["rhythm"]
    rec_id = case_meta["record_id"]
    a0_r = case_meta["a0_mean_missing_r"]
    r7_r = case_meta.get("r7_mean_missing_r", a0_r)

    fig, axes = plt.subplots(len(leads_to_plot), 1, figsize=(13, 11), sharex=True, facecolor="#FFFDFD", dpi=300)
    fig.subplots_adjust(top=0.91, bottom=0.06, hspace=0.30, left=0.08, right=0.96)

    full_rhy_name = RHYTHM_FULL_NAMES.get(rhythm, rhythm)
    fig.suptitle(
        f"Failure Mode Diagnostic: {rhythm} ({full_rhy_name}) — Worst Case [{rec_id}]\n"
        f"A0_raw (Missing r = {a0_r:.4f}) vs. R7_wavelet (Missing r = {r7_r:.4f}) vs. Ground Truth",
        fontsize=12,
        fontweight="bold",
        color="#B71C1C",
        y=0.96,
    )

    for i, l_name in enumerate(leads_to_plot):
        ax = axes[i]
        l_idx = LEAD_NAMES.index(l_name)

        draw_ecg_grid(ax, 0.0, 3.0, -1.2, 1.8)

        t_sig = true_12[l_idx, :n_samples]
        a0_sig = a0_12[l_idx, :n_samples]
        r7_sig = r7_12[l_idx, :n_samples]

        r_a0 = float(np.corrcoef(t_sig, a0_sig)[0, 1]) if l_name != "I" else 1.0
        r_r7 = float(np.corrcoef(t_sig, r7_sig)[0, 1]) if l_name != "I" else 1.0

        if l_name == "I":
            ax.plot(t, t_sig, color="#00838F", linewidth=1.5, label="Lead I (Observed Input)")
            badge = "Lead I [Observed Input]"
            ax.text(0.015, 0.80, badge, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color="#00838F",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#00838F"))
        else:
            ax.plot(t, t_sig, color="#212121", linewidth=1.3, label="Ground Truth True", zorder=2)
            ax.plot(t, a0_sig, color="#D32F2F", linewidth=1.2, linestyle="-", label=f"A0_raw (r={r_a0:.3f})", zorder=3)
            ax.plot(t, r7_sig, color="#1976D2", linewidth=1.2, linestyle="--", label=f"R7_wavelet (r={r_r7:.3f})", zorder=4)
            badge = f"Lead {l_name}  |  A0 r = {r_a0:.3f}  |  R7 r = {r_r7:.3f}"
            ax.text(0.015, 0.80, badge, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color="#212121",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#BDBDBD"))

        ax.set_xlim(0, 3.0)
        ax.set_ylim(-1.2, 1.8)
        ax.set_facecolor("#FFFDFD")
        ax.set_ylabel("mV", fontsize=8)
        ax.tick_params(labelsize=8, direction="in", length=2, color="#BDBDBD")

    axes[-1].set_xlabel("Time (seconds)", fontsize=9, color="#424242")
    axes[0].legend(loc="upper right", fontsize=8.5, framealpha=0.95, facecolor="#FFFFFF")

    png_path = out_path.with_suffix(".png")
    svg_path = out_path.with_suffix(".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved Diagnostic] {png_path.name}")


def main() -> None:
    cases_file = RESULTS_DIR / "rdb_gallery_selected_cases.json"
    waveforms_file = RESULTS_DIR / "rdb_gallery_selected_waveforms.npz"

    if not cases_file.exists() or not waveforms_file.exists():
        raise FileNotFoundError("Run evaluate_rdb_case_gallery.py first!")

    with open(cases_file) as f:
        cases = json.load(f)

    npz = np.load(waveforms_file)

    print(f"[Gallery Plotter] Generating clinical 12-lead overlays for {len(cases)} cases...")

    for c in cases:
        rhy = c["rhythm"]
        tier = c["tier"]
        rec_id = c["record_id"]
        key = f"{rhy}_{tier}"

        true_12 = npz[f"{key}_true"]
        a0_12 = npz[f"{key}_a0"]

        out_prefix = OUT_FIGURES_DIR / f"ecg_12lead_{rhy}_{tier}_{rec_id}"
        plot_12lead_overlay(c, true_12, a0_12, out_prefix)

        # For worst cases, generate tri-way comparative diagnostic plot
        if tier == "worst" and f"{key}_r7" in npz:
            r7_12 = npz[f"{key}_r7"]
            diag_prefix = OUT_FIGURES_DIR / f"failure_diagnostic_{rhy}_{rec_id}"
            plot_triway_worst_comparison(c, true_12, a0_12, r7_12, diag_prefix)

    print("[Gallery Plotter] All 24 cases and diagnostic plots generated successfully!")


if __name__ == "__main__":
    main()
