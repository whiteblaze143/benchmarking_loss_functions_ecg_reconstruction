#!/usr/bin/env python3
"""
generate_wearable_spatial_figure.py
Generates a publication-grade, high-resolution diagram illustrating cardiac spatial dipoles,
comparing the 12-lead hospital standard, 1-lead smartwatch/patch projections,
and the 3-lead diagnostic triad basis.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    out_dir = Path("slides/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6.0), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Title
    ax.text(6.0, 5.7, "Cardiac Electrical Dipole Projections: 12-Lead Standard vs. Ambulatory Wearables",
            fontsize=13, fontweight="bold", color="#0F2042", ha="center")
    ax.text(6.0, 5.35, "Physical spatial vectors dictate what information is observable vs. mathematically ambiguous",
            fontsize=10, color="#64748B", ha="center")

    # 4 Comparison Columns
    col_w = 2.6
    col_gap = 0.3
    start_x = 0.5
    
    configs = [
        ("A. Hospital 12-Lead Standard", "#0F2042", "#EFF6FF", [
            ("Electrodes", "10 physical skin electrodes"),
            ("Spatial Vectors", "12 leads (6 limb, 6 precordial)"),
            ("Dipole Coverage", "Complete 3D Space (X, Y, Z)"),
            ("Diagnostic Reach", "STEMI, NSTEMI, LVH, QTc"),
            ("Clinical Reality", "Confined to hospital carts")
        ], "Full 3D Heart View"),
        
        ("B. Smartwatch (Lead I)", "#0284C7", "#F0F9FF", [
            ("Electrodes", "2 dry metal contacts (wrists)"),
            ("Spatial Vector", "Single horizontal axis (X)"),
            ("Dipole Coverage", "1D transverse projection only"),
            ("Diagnostic Reach", "Atrial Fibrillation, HR, Rhythm"),
            ("Blind Spot", "Blind to vertical & posterior")
        ], "Horizontal 1D Projection"),
        
        ("C. Multi-Channel Patch", "#8B5CF6", "#FAF5FF", [
            ("Examples", "CardioSTAT, Zio XT, Icentia"),
            ("Spatial Vectors", "2 dual-channel vectors (X, Y)"),
            ("Dipole Coverage", "Planar 2D frontal coverage"),
            ("Diagnostic Reach", "Complex arrhythmia, flutters"),
            ("Blind Spot", "Limited anterior chest depth")
        ], "Planar 2D Projection"),
        
        ("D. 3-Lead Triad (I, II, V2)", "#059669", "#ECFDF5", [
            ("Electrodes", "3-wire clinical configuration"),
            ("Spatial Vectors", "X: Lead I, Y: Lead II, Z: V2"),
            ("Dipole Coverage", "Orthogonal 3D Basis (X, Y, Z)"),
            ("Diagnostic Reach", "Near-lossless 12L recovery"),
            ("Clinical Power", "In-clinic triage with 12L power")
        ], "Orthogonal 3D Basis")
    ]

    for idx, (title, stroke_c, bg_c, details, badge) in enumerate(configs):
        cx = start_x + idx * (col_w + col_gap)
        cy = 0.5
        ch = 4.5
        
        # Outer card box
        rect = patches.FancyBboxPatch((cx, cy), col_w, ch,
                                     boxstyle="round,pad=0.1,rounding_size=0.15",
                                     linewidth=1.5, edgecolor=stroke_c,
                                     facecolor=bg_c, zorder=1)
        ax.add_patch(rect)
        
        # Header banner inside card
        header_rect = patches.FancyBboxPatch((cx + 0.05, cy + ch - 0.75), col_w - 0.1, 0.65,
                                            boxstyle="round,pad=0.05,rounding_size=0.1",
                                            linewidth=0.8, edgecolor=stroke_c,
                                            facecolor=stroke_c, zorder=2)
        ax.add_patch(header_rect)
        
        ax.text(cx + col_w/2, cy + ch - 0.42, title,
                fontsize=9.5, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=3)
                
        # Badge
        badge_rect = patches.FancyBboxPatch((cx + 0.2, cy + ch - 1.15), col_w - 0.4, 0.32,
                                           boxstyle="round,pad=0.05,rounding_size=0.08",
                                           linewidth=0.8, edgecolor=stroke_c,
                                           facecolor="#FFFFFF", zorder=2)
        ax.add_patch(badge_rect)
        ax.text(cx + col_w/2, cy + ch - 0.99, badge,
                fontsize=8.5, fontweight="bold", color=stroke_c, ha="center", va="center", zorder=3)

        # Body items
        item_y = cy + ch - 1.5
        for prop, val in details:
            ax.text(cx + 0.15, item_y, f"{prop}:",
                    fontsize=8.5, fontweight="bold", color="#1E293B", va="top")
            ax.text(cx + 0.15, item_y - 0.22, val,
                    fontsize=8, color="#475569", va="top")
            item_y -= 0.58

    # Bottom summary callout
    bot_rect = patches.FancyBboxPatch((0.5, 0.08), 11.0, 0.36,
                                     boxstyle="round,pad=0.05,rounding_size=0.08",
                                     linewidth=1.0, edgecolor="#94A3B8",
                                     facecolor="#F8FAFC", zorder=1)
    ax.add_patch(bot_rect)
    ax.text(6.0, 0.26, "Key Electrophysiological Insight: 3 physical orthogonal dipoles span the cardiac electrical space; 1 lead requires discrete latent regularizers.",
            fontsize=8.5, fontweight="bold", color="#1E293B", ha="center", va="center")

    plt.tight_layout()
    png_path = out_dir / "wearable_spatial_dipoles.png"
    pdf_path = out_dir / "wearable_spatial_dipoles.pdf"
    plt.savefig(png_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(pdf_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    print(f"Generated {png_path} ({png_path.stat().st_size} bytes)")
    print(f"Generated {pdf_path} ({pdf_path.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
