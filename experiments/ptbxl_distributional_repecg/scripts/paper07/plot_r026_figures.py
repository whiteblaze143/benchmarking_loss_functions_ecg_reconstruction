#!/usr/bin/env python3
"""Generate publication-quality figures for Milestone R026: CF-SetOperator."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
R026_BENCH_JSON = CODEX_DIR / "paper07_operator/r026_benchmark_results/r026_benchmark_seed42.json"
R025_BENCH_JSON = CODEX_DIR / "paper07_operator/r025_benchmark_results/r025_benchmark_seed42.json"
ARTIFACTS_DIR = Path("/home/mithunmanivannan/.gemini/antigravity-ide/brain/ba979872-97f8-4d89-9b0a-621cb4b9c051")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})

def main():
    with open(R026_BENCH_JSON) as f:
        r026_data = json.load(f)
    with open(R025_BENCH_JSON) as f:
        r025_data = json.load(f)

    cf_results = r026_data["results_cf"]
    orig_results = r026_data["results_orig"]
    r025_arms = r025_data["results_by_arm"]

    configs = [
        ("Q8_full", "Q8 Full (8 leads)"),
        ("S6_precordial", "S6 Precordial (6 leads)"),
        ("S6_limb", "S6 Limb (6 leads)"),
        ("Derived_limb_only", "Derived Limb (4 leads)"),
        ("S3_reduced", "S3 ICU (3 leads)"),
        ("S2_reduced", "S2 Bipolar (2 leads)"),
        ("S1_smartwatch", "S1 Watch (1 lead)"),
        ("Lead_III_single", "Lead III Single"),
        ("aVF_single", "aVF Single"),
        ("S_icm", "S_icm (V3-V2)"),
        ("S_icm_plus_leadI", "S_icm + Lead I"),
        ("Test_obliques_3", "3 Obliques (Multi-lead)"),
        ("All_16_leads", "All 16 Leads (Redundant)"),
    ]

    cfg_keys = [c[0] for c in configs]
    cfg_labels = [c[1] for c in configs]

    cf_scores = [cf_results[k]["macro_auroc"] for k in cfg_keys]
    orig_scores = [orig_results[k]["macro_auroc"] for k in cfg_keys]
    lq_scores = [r025_arms["learned_q"][k]["macro_auroc"] for k in cfg_keys]
    os_scores = [r025_arms["os_only"][k]["macro_auroc"] for k in cfg_keys]
    
    graphecg_scores = []
    for k in cfg_keys:
        val = r025_arms["graphecg"][k].get("macro_auroc")
        graphecg_scores.append(val if val is not None else np.nan)

    # -------------------------------------------------------------
    # Figure 1: Comprehensive Configuration Benchmark Comparison
    # -------------------------------------------------------------
    x = np.arange(len(cfg_keys))
    width = 0.17

    fig, ax = plt.subplots(figsize=(17, 7.5), dpi=300)

    rects1 = ax.bar(x - 2 * width, cf_scores, width, label="CF-SetOperator (Ours, 0 params)", color="#1f77b4", alpha=0.95)
    rects2 = ax.bar(x - 1 * width, orig_scores, width, label="Original SetOperator (Historical)", color="#ff7f0e", alpha=0.85)
    rects3 = ax.bar(x + 0 * width, lq_scores, width, label="Learned-Q Control (R025)", color="#2ca02c", alpha=0.85)
    rects4 = ax.bar(x + 1 * width, os_scores, width, label="OS-Only MLP (R025)", color="#9467bd", alpha=0.85)
    rects5 = ax.bar(x + 2 * width, [s if not np.isnan(s) else 0 for s in graphecg_scores], width, 
                    label="GraphECG (Ansari et al.)", color="#d62728", alpha=0.85)

    for i, s in enumerate(graphecg_scores):
        if np.isnan(s):
            ax.text(x[i] + 2 * width, 0.42, "N/A", ha="center", va="bottom", fontsize=8, color="#d62728", rotation=90)

    # Annotate dramatic gains on Limb leads and Redundant leads
    # 1. S6 Limb
    idx_limb = cfg_keys.index("S6_limb")
    gain_limb = cf_scores[idx_limb] - orig_scores[idx_limb]
    ax.annotate(f"+{gain_limb:.3f}\n(6→2 leads)", 
                xy=(x[idx_limb] - 2 * width, cf_scores[idx_limb]), 
                xytext=(x[idx_limb] - 2 * width, cf_scores[idx_limb] + 0.035),
                arrowprops=dict(facecolor="#1f77b4", arrowstyle="->", lw=1.5),
                ha="center", fontsize=9, fontweight="bold", color="#1f77b4")

    # 2. Derived Limb
    idx_derived = cfg_keys.index("Derived_limb_only")
    gain_derived = cf_scores[idx_derived] - orig_scores[idx_derived]
    ax.annotate(f"+{gain_derived:.3f}\n(4→2 leads)", 
                xy=(x[idx_derived] - 2 * width, cf_scores[idx_derived]), 
                xytext=(x[idx_derived] - 2 * width, cf_scores[idx_derived] + 0.035),
                arrowprops=dict(facecolor="#1f77b4", arrowstyle="->", lw=1.5),
                ha="center", fontsize=9, fontweight="bold", color="#1f77b4")

    # 3. All 16 Leads
    idx_16 = cfg_keys.index("All_16_leads")
    gain_16 = cf_scores[idx_16] - orig_scores[idx_16]
    ax.annotate(f"+{gain_16:.3f}\n(No Bloat)", 
                xy=(x[idx_16] - 2 * width, cf_scores[idx_16]), 
                xytext=(x[idx_16] - 2 * width, cf_scores[idx_16] + 0.035),
                arrowprops=dict(facecolor="#1f77b4", arrowstyle="->", lw=1.5),
                ha="center", fontsize=9, fontweight="bold", color="#1f77b4")

    ax.set_ylabel("Macro AUROC (PTB-XL 5 Superclasses)")
    ax.set_title("Milestone R026: Canonical-Frame SetOperator (CF-SetOperator) Benchmark\n"
                 "Observable-Subspace Canonicalization Composed with Frozen Phase-KME SetTransformer", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cfg_labels, rotation=35, ha="right")
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    plt.tight_layout()
    out_fig1 = ARTIFACTS_DIR / "r026_cf_benchmark_comparison.png"
    plt.savefig(out_fig1)
    plt.close()
    print(f"Saved Figure 1 to {out_fig1}")

    # -------------------------------------------------------------
    # Figure 2: Redundancy Immunity, Invariance, and Continuity
    # -------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    # Panel A: Redundancy Gate (Q8 vs All 16 Leads)
    models = ["Original SetOp", "Learned-Q", "CF-SetOp (Ours)"]
    q8_vals = [orig_results["Q8_full"]["macro_auroc"],
               r025_arms["learned_q"]["Q8_full"]["macro_auroc"],
               cf_results["Q8_full"]["macro_auroc"]]
    all16_vals = [orig_results["All_16_leads"]["macro_auroc"],
                 r025_arms["learned_q"]["All_16_leads"]["macro_auroc"],
                 cf_results["All_16_leads"]["macro_auroc"]]

    x_red = np.arange(len(models))
    w_red = 0.35
    ax1.bar(x_red - w_red/2, q8_vals, w_red, label="Q8 Full (8 Leads)", color="#2ca02c", alpha=0.9)
    ax1.bar(x_red + w_red/2, all16_vals, w_red, label="All 16 Leads (+8 Redundant)", color="#9467bd", alpha=0.9)

    for i in range(len(models)):
        delta = all16_vals[i] - q8_vals[i]
        sign = "+" if delta >= 0 else ""
        col = "green" if abs(delta) < 1e-4 else "red"
        ax1.text(x_red[i] + w_red/2, all16_vals[i] + 0.008, f"{sign}{delta:.4f}", 
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color=col)

    ax1.set_ylabel("Macro AUROC")
    ax1.set_title("Panel A: Redundancy Resilience\n(Q8 vs. All 16 Leads)", fontweight="bold")
    ax1.set_xticks(x_red)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0.82, 0.98)
    ax1.legend(loc="lower left")

    # Panel B: Test F Invariance under 20 Invertible Remixings Across Ranks
    test_f = r026_data["test_f_invertible_remixings"]
    ranks = [1, 2, 3, 4, 6, 8]
    cf_diffs = [max(1e-16, test_f[f"rank_{r}"]["cf_max_logit_diff_across_20_remixes"]) for r in ranks]
    orig_diffs = [test_f[f"rank_{r}"]["orig_max_logit_diff_across_20_remixes"] for r in ranks]

    x_f = np.arange(len(ranks))
    w_f = 0.35
    ax2.bar(x_f - w_f/2, cf_diffs, w_f, label="CF-SetOp (Max Logit Diff)", color="#1f77b4")
    ax2.bar(x_f + w_f/2, orig_diffs, w_f, label="Original SetOp (Max Logit Diff)", color="#ff7f0e")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-17, 1e2)
    ax2.axhline(1e-5, color="red", linestyle=":", label="Tolerance Gate (1e-5)")
    ax2.set_ylabel("Max Absolute Logit Diff (log scale)")
    ax2.set_title("Panel B: Test F Invertible Remixings\n(20 Random Well-Conditioned R Matrices)", fontweight="bold")
    ax2.set_xticks(x_f)
    ax2.set_xticklabels([f"Rank {r}" for r in ranks])
    ax2.legend(loc="upper left")

    # Panel C: Test G Path Continuity
    test_g = r026_data["test_g_path_continuity"]
    frame_diffs = test_g["frame_diffs"]
    latent_diffs = test_g["latent_diffs"]
    steps = np.arange(len(frame_diffs))
    angles = np.linspace(0, 90, len(frame_diffs))

    ax3.plot(angles, frame_diffs, label="||B(α+Δ) - B(α)||", color="#1f77b4", lw=2)
    ax3.plot(angles, latent_diffs, label="||z(α+Δ) - z(α)|| (Latent Jump)", color="#d62728", lw=2, linestyle="--")
    ax3.axvline(test_g["pivot_transition_angle_deg"], color="gray", linestyle=":", 
                label=f"Pivot Boundary ({test_g['pivot_transition_angle_deg']:.1f}°)")
    ax3.set_xlabel("Rotation Angle α (degrees, Lead I → Lead II)")
    ax3.set_ylabel("Step Difference Norm")
    ax3.set_title("Panel C: Test G Path Continuity Audit\n(Continuous Arc between Coordinate Axes)", fontweight="bold")
    ax3.legend(loc="upper left")

    plt.tight_layout()
    out_fig2 = ARTIFACTS_DIR / "r026_redundancy_and_continuity.png"
    plt.savefig(out_fig2)
    plt.close()
    print(f"Saved Figure 2 to {out_fig2}")

if __name__ == "__main__":
    main()
