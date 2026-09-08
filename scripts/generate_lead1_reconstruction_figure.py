#!/usr/bin/env python3
"""
generate_lead1_reconstruction_figure.py
Generates the publication-grade 12-lead ECG reconstruction figure for the champion ECG-AIM model
(conv15e_A0_wave_noSSL_gated_add_s42_l0) conditioned on single-lead Lead I input (Smartwatch vector)
on PTB-XL Record #10283.
"""

import argparse
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.bootstrap_paths import setup_import_paths
setup_import_paths(include_fairseq=True)
from scripts.train_1lead_wavelet_ssl_mtl import build_model, forward_model

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_GRID_POS = {
    # 4 columns x 3 rows
    # Col 0: Limb leads
    "I": (0, 0),
    "II": (1, 0),
    "III": (2, 0),
    # Col 1: Augmented leads
    "aVR": (0, 1),
    "aVL": (1, 1),
    "aVF": (2, 1),
    # Col 2: Anterior/Septal chest leads
    "V1": (0, 2),
    "V2": (1, 2),
    "V3": (2, 2),
    # Col 3: Lateral chest leads
    "V4": (0, 3),
    "V5": (1, 3),
    "V6": (2, 3),
}

def draw_ecg_grid(ax, t_min=0.0, t_max=2.5, v_min=-1.5, v_max=1.5):
    """Draw authentic medical ECG grid lines (25 mm/s, 10 mm/mV)."""
    # Minor vertical grid (0.04 s)
    t_minor = np.arange(t_min, t_max + 0.01, 0.04)
    for t in t_minor:
        is_major = np.isclose(t % 0.20, 0.0, atol=1e-3) or np.isclose(t % 0.20, 0.20, atol=1e-3)
        ax.axvline(
            t,
            color="#FFD1D1" if is_major else "#FFEBEB",
            linewidth=0.75 if is_major else 0.35,
            zorder=0,
        )

    # Horizontal grid (0.1 mV)
    v_minor = np.arange(v_min, v_max + 0.01, 0.1)
    for v in v_minor:
        is_major = np.isclose(v % 0.5, 0.0, atol=1e-3) or np.isclose(v % 0.5, 0.5, atol=1e-3)
        ax.axhline(
            v,
            color="#FFD1D1" if is_major else "#FFEBEB",
            linewidth=0.75 if is_major else 0.35,
            zorder=0,
        )

def main():
    print("Loading champion Lead I model checkpoint...")
    ckpt_path = _ROOT / "refine-logs/convergence_10e/runs/conv15e_A0_wave_noSSL_gated_add_s42_l0/best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = argparse.Namespace(**ckpt["config"])

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    print("Loading PTB-XL record #10283...")
    rec_path = _ROOT / "data/ptb_xl/tensors/test/10283.pt"
    rec = torch.load(rec_path, map_location="cpu", weights_only=False)
    if isinstance(rec, dict):
        y = rec["waveform"].unsqueeze(0)
    else:
        y = rec.unsqueeze(0) if rec.ndim == 2 else rec

    # Forward pass with Lead 0 (Lead I) observed
    obs = [0]
    with torch.no_grad():
        res = forward_model(model, y, obs, compute_delineation=False, compute_ssl=False)
    y_pred = res["y_pred"]

    # Compute correlations
    r_dict = {}
    for i, l in enumerate(LEAD_NAMES):
        p = y_pred[0, i] - y_pred[0, i].mean()
        t = y[0, i] - y[0, i].mean()
        r = float(F.cosine_similarity(p.unsqueeze(0), t.unsqueeze(0)))
        r_dict[l] = r

    missing_leads = [l for l in LEAD_NAMES if l != "I"]
    mean_missing_r = float(np.mean([r_dict[l] for l in missing_leads]))
    print(f"Mean missing lead Pearson r: {mean_missing_r:.4f}")

    # Set up figure matching 16 x 8.5 inches (widescreen 16:9 format)
    fig, axes = plt.subplots(3, 4, figsize=(16, 8.5), dpi=300, sharex=True, sharey=True)
    plt.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.10, hspace=0.18, wspace=0.14)

    fs = 500
    n_samples = int(2.5 * fs)  # 2.5 seconds = 1250 samples
    time_axis = np.arange(n_samples) / fs

    true_np = y[0, :, :n_samples].numpy()
    pred_np = y_pred[0, :, :n_samples].numpy()

    for lead_idx, lead in enumerate(LEAD_NAMES):
        row, col = LEAD_GRID_POS[lead]
        ax = axes[row, col]
        draw_ecg_grid(ax, t_min=0.0, t_max=2.5, v_min=-1.5, v_max=1.5)

        t_sig = true_np[lead_idx]
        p_sig = pred_np[lead_idx]

        if lead == "I":
            # Acquired Lead I (Smartwatch vector)
            ax.plot(time_axis, t_sig, color="#2B6CB0", linewidth=1.5, label="Acquired Lead I (Smartwatch)", zorder=3)
            status_text = "ACQUIRED"
            status_color = "#2B6CB0"
        else:
            # Reconstructed missing lead
            ax.plot(time_axis, t_sig, color="#1A202C", linewidth=1.2, label="Ground truth", zorder=2)
            ax.plot(time_axis, p_sig, color="#E53E3E", linewidth=1.3, linestyle="--", label="ECG-AIM reconstruction", zorder=4)
            status_text = f"RECON (r={r_dict[lead]:.3f})"
            status_color = "#E53E3E"

        # Lead label box inside subplot
        ax.text(
            0.04, 0.88, lead,
            transform=ax.transAxes,
            fontsize=13, fontweight="bold",
            color="#1A365D", va="top"
        )
        ax.text(
            0.04, 0.72, status_text,
            transform=ax.transAxes,
            fontsize=9.5, fontweight="bold",
            color=status_color, va="top"
        )

        ax.set_xlim(0.0, 2.5)
        ax.set_ylim(-1.5, 1.5)

        # Formatting ticks
        if col == 0:
            ax.set_ylabel("Amplitude (mV)", fontsize=10, color="#1A202C")
            ax.set_yticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
            ax.tick_params(axis="y", labelsize=8.5, colors="#4A5568")
        else:
            ax.tick_params(axis="y", labelleft=False)

        if row == 2:
            ax.set_xlabel("Time (seconds)", fontsize=10, color="#1A202C")
            ax.set_xticks(np.arange(0.0, 2.6, 0.2))
            ax.tick_params(axis="x", labelsize=8.5, colors="#4A5568", rotation=0)
        else:
            ax.tick_params(axis="x", labelbottom=False)

        for spine in ax.spines.values():
            spine.set_color("#CBD5E0")
            spine.set_linewidth(0.8)

    # Suptitle and subtitle
    fig.suptitle(
        "ECG-AIM One-Lead Reconstruction (conv15e_A0_wave_noSSL_gated_add_s42_l0 | PTB-XL #10283)",
        fontsize=15, fontweight="bold", color="#1A365D", y=0.965
    )
    fig.text(
        0.5, 0.925,
        f"Acquired Lead I (Smartwatch Vector) $\\rightarrow$ 11 Missing Leads Inferred (Record Mean Missing-Lead r = {mean_missing_r:.3f})",
        ha="center", fontsize=11.5, fontweight="semibold", color="#4A5568"
    )

    # Custom unified legend at bottom center
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2B6CB0", lw=2, label="Acquired Lead I (Smartwatch)"),
        Line2D([0], [0], color="#1A202C", lw=1.5, label="Ground truth"),
        Line2D([0], [0], color="#E53E3E", lw=1.5, linestyle="--", label="ECG-AIM reconstruction"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E0",
        fontsize=10.5,
        bbox_to_anchor=(0.5, 0.02)
    )

    out_png1 = _ROOT / "figures/best_ecg_aim_lead1_reconstruction.png"
    out_pdf1 = _ROOT / "figures/best_ecg_aim_lead1_reconstruction.pdf"
    out_png2 = _ROOT / "slides/figures/best_ecg_aim_lead1_reconstruction.png"
    out_pdf2 = _ROOT / "slides/figures/best_ecg_aim_lead1_reconstruction.pdf"

    plt.savefig(out_png1, dpi=300)
    plt.savefig(out_pdf1)
    plt.savefig(out_png2, dpi=300)
    plt.savefig(out_pdf2)
    plt.close(fig)

    print(f"Saved: {out_png1}")
    print(f"Saved: {out_pdf1}")
    print(f"Saved: {out_png2}")
    print(f"Saved: {out_pdf2}")

if __name__ == "__main__":
    main()
