#!/usr/bin/env python3
"""
generate_delineation_figure.py
Generates a publication-grade, high-resolution ECG wave delineation diagram
showing the P wave, PR interval, PR segment, QRS complex, ST segment,
T wave, and QT/QTc interval on an authentic medical ECG background grid.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_ecg_wave(t):
    """
    Generate an authentic synthetic ECG single cardiac cycle using Gaussians.
    t is time in seconds, centered with R-peak at t=0.40s.
    """
    y = np.zeros_like(t)
    
    # Baseline isoelectric = 0.0 mV
    # P wave: positive deflection, center 0.20s, width 0.045s, amp 0.15 mV
    y += 0.16 * np.exp(-((t - 0.20) ** 2) / (2 * (0.024 ** 2)))
    
    # Q wave: small negative deflection, center 0.365s, width 0.010s, amp -0.15 mV
    y -= 0.15 * np.exp(-((t - 0.365) ** 2) / (2 * (0.008 ** 2)))
    
    # R peak: sharp positive deflection, center 0.400s, width 0.014s, amp 1.25 mV
    y += 1.25 * np.exp(-((t - 0.400) ** 2) / (2 * (0.013 ** 2)))
    
    # S wave: negative deflection, center 0.435s, width 0.012s, amp -0.35 mV
    y -= 0.35 * np.exp(-((t - 0.435) ** 2) / (2 * (0.011 ** 2)))
    
    # ST segment slight elevation / normal slope into T wave
    # T wave: broader positive deflection, center 0.620s, width 0.065s, amp 0.32 mV
    # Asymmetric T wave (slower rise, steeper descent)
    t_shift = t - 0.620
    t_width = np.where(t_shift < 0, 0.070, 0.050)
    y += 0.32 * np.exp(-((t_shift) ** 2) / (2 * (t_width ** 2)))
    
    return y

def main():
    out_dir = Path("slides/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    t = np.linspace(0.05, 0.85, 1600)
    v = generate_ecg_wave(t)
    
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    
    # Draw authentic ECG grid (25 mm/s, 10 mm/mV -> 0.04s small box, 0.20s big box; 0.1mV small, 0.5mV big)
    t_min, t_max = 0.05, 0.85
    v_min, v_max = -0.55, 1.45
    
    # Minor grid
    t_minor = np.arange(0.04, 0.88, 0.04)
    for tm in t_minor:
        is_major = np.isclose(tm % 0.20, 0.0, atol=1e-3) or np.isclose(tm % 0.20, 0.20, atol=1e-3)
        ax.axvline(tm, color="#FCA5A5" if is_major else "#FEE2E2",
                   linewidth=0.9 if is_major else 0.4, zorder=0)
                   
    v_minor = np.arange(-0.6, 1.6, 0.1)
    for vm in v_minor:
        is_major = np.isclose(vm % 0.50, 0.0, atol=1e-3) or np.isclose(vm % 0.50, 0.50, atol=1e-3)
        ax.axhline(vm, color="#FCA5A5" if is_major else "#FEE2E2",
                   linewidth=0.9 if is_major else 0.4, zorder=0)
    
    # Zero line
    ax.axhline(0.0, color="#CBD5E1", linestyle="--", linewidth=0.8, zorder=1)
    
    # Soft, subtle highlight bands for the three main phases
    ax.axvspan(0.145, 0.255, color="#3B82F6", alpha=0.08, zorder=1)  # P wave
    ax.axvspan(0.355, 0.455, color="#EF4444", alpha=0.08, zorder=1)  # QRS complex
    ax.axvspan(0.535, 0.730, color="#10B981", alpha=0.08, zorder=1)  # T wave
    
    # Plot the ECG waveform (thick deep navy line)
    ax.plot(t, v, color="#0F172A", linewidth=3.0, zorder=3)
    
    # --- Clean, Simple Labels ---
    # P wave
    ax.text(0.20, 0.26, "P Wave", color="#1D4ED8", fontsize=13,
            fontweight="bold", ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec="#93C5FD", lw=1.2), zorder=4)
            
    # Q, R, S markers
    ax.plot(0.365, -0.15, marker="o", markersize=4.5, color="#DC2626", zorder=4)
    ax.text(0.352, -0.22, "Q", color="#DC2626", fontsize=12, fontweight="bold", ha="right", va="top")
    
    ax.plot(0.400, 1.25, marker="o", markersize=5.5, color="#DC2626", zorder=4)
    ax.text(0.400, 1.32, "R Peak", color="#DC2626", fontsize=13, fontweight="bold", ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#FCA5A5", lw=1.2), zorder=4)
            
    ax.plot(0.435, -0.35, marker="o", markersize=4.5, color="#DC2626", zorder=4)
    ax.text(0.448, -0.38, "S", color="#DC2626", fontsize=12, fontweight="bold", ha="left", va="top")
    
    # ST Segment
    ax.text(0.495, 0.14, "ST Segment", color="#D97706", fontsize=12,
            fontweight="bold", ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFFBEB", ec="#FCD34D", lw=1.2), zorder=4)
            
    # T wave
    ax.text(0.620, 0.42, "T Wave", color="#047857", fontsize=13,
            fontweight="bold", ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec="#6EE7B7", lw=1.2), zorder=4)
            
    # Helper to draw horizontal interval brackets
    def draw_bracket(x1, x2, y_level, label, color):
        tick_h = 0.03
        ax.plot([x1, x1, x2, x2], [y_level + tick_h, y_level, y_level, y_level + tick_h],
                color=color, linewidth=1.8, zorder=4)
        mid_x = (x1 + x2) / 2.0
        ax.text(mid_x, y_level - 0.03, label, color=color, fontsize=11.5,
                fontweight="bold", ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="#FFFFFF", ec=color, lw=1.0), zorder=4)

    # 1. PR Interval (onset of P to onset of QRS)
    draw_bracket(0.145, 0.355, -0.18, "PR Interval", "#2563EB")
    
    # 2. QT Interval (onset of QRS to end of T) - placed cleanly below S-wave
    draw_bracket(0.355, 0.730, -0.44, "QT Interval", "#059669")

    ax.set_xlim(t_min, t_max)
    ax.set_ylim(-0.58, 1.45)
    ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold", color="#334155", labelpad=8)
    ax.set_ylabel("Voltage (mV)", fontsize=12, fontweight="bold", color="#334155", labelpad=8)
    ax.set_title("Standard ECG Cardiac Cycle & Morphological Intervals",
                 fontsize=15, fontweight="bold", color="#0F172A", pad=14)
                 
    ax.tick_params(axis='both', which='major', labelsize=11, colors="#475569")
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(1.0)
        
    plt.tight_layout()
    
    png_path = out_dir / "ecg_delineation_components.png"
    pdf_path = out_dir / "ecg_delineation_components.pdf"
    plt.savefig(png_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(pdf_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print(f"Generated {png_path} ({png_path.stat().st_size} bytes)")
    print(f"Generated {pdf_path} ({pdf_path.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
