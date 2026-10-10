#!/usr/bin/env python3
"""Generate publication-quality figures for Milestone R026/R027 Decisive 2x2.

Figure 1: Decisive 2x2 Factorial Benchmark Comparison:
- Panel A: All 13 configurations across the 4 cells:
  1. Original SetOp (Basis-only, Orig inputs)
  2. CF-SetOp R026 (Basis-only, CF quotient)
  3. Span SetOp R024 (Span training, Orig inputs)
  4. CF + Span SetOp (Span training, CF quotient)
  + Native GraphECG (Ansari et al., properly matched 1000-sample inputs)
- Panel B: Aggregate Summary by Measurement Geometry:
  - Canonical Sparse (1..3 leads)
  - Redundant Leads (S6 limb, Derived limb, All 16)
  - Unseen Obliques (S_icm, S_icm+I, 3 Obliques)
  - Overall Mean

Figure 2: Physical Robustness, Conditioning & Path Continuity:
- Panel A: Noise Attenuation & Conditioning:
  - Operator norm ||B_Q Q^+||_2 and Macro AUROC under noise (Clean, 40dB, 30dB, 20dB, 10dB)
- Panel B: Invariance under 20 Invertible Transformations Across Ranks r=1..8
- Panel C: Path Continuity & Pivot Audit:
  - Plot ||B(alpha+Delta) - B(alpha)|| and ||z(alpha+Delta) - z(alpha)||
  - Correctly labeled crossing at alpha = 45 degrees where ||Pe_I|| = ||Pe_II||
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
DECISIVE_DIR = CODEX_DIR / "paper07_operator/r026_decisive_2x2_results"
NOISE_DIR = CODEX_DIR / "paper07_operator/r026_noise_conditioning_audit"
R026_BENCH_JSON = CODEX_DIR / "paper07_operator/r026_benchmark_results/r026_benchmark_seed42.json"
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

GRAPHECG_NATIVE = {
    "Q8_full": 0.9128,
    "S6_precordial": 0.8784,
    "S6_limb": 0.8800,
    "S_icm": 0.7197,
    "S1_smartwatch": 0.7675,
}

CONFIGS = [
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


def main():
    with open(DECISIVE_DIR / "decisive_2x2_results.json") as f:
        d2x2 = json.load(f)
    with open(NOISE_DIR / "r026_noise_conditioning_audit.json") as f:
        d_noise = json.load(f)
    with open(R026_BENCH_JSON) as f:
        d_r026 = json.load(f)

    cells = d2x2["cells"]
    cfg_keys = [c[0] for c in CONFIGS]
    cfg_labels = [c[1] for c in CONFIGS]

    s_orig_basis = [cells["cell_1_1_orig_basis"][k]["macro_auroc"] for k in cfg_keys]
    s_cf_basis = [cells["cell_1_2_cf_basis"][k]["macro_auroc"] for k in cfg_keys]
    s_orig_span = [cells["cell_2_1_orig_span"][k]["macro_auroc"] for k in cfg_keys]
    s_cf_span = [cells["cell_2_2_cf_span"][k]["macro_auroc"] for k in cfg_keys]

    graphecg_scores = [GRAPHECG_NATIVE.get(k, np.nan) for k in cfg_keys]

    # -------------------------------------------------------------
    # Figure 1: Comprehensive 2x2 Factorial Comparison Across Configurations
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), dpi=300, gridspec_kw={"height_ratios": [1.4, 1.0]})

    x = np.arange(len(cfg_keys))
    w = 0.17

    ax1.bar(x - 2 * w, s_orig_basis, w, label="Original SetOp (Basis, Orig)", color="#ff7f0e", alpha=0.85)
    ax1.bar(x - 1 * w, s_cf_basis, w, label="CF-SetOp R026 (Basis, CF)", color="#1f77b4", alpha=0.95)
    ax1.bar(x + 0 * w, s_orig_span, w, label="Span SetOp R024 (Span, Orig)", color="#2ca02c", alpha=0.85)
    ax1.bar(x + 1 * w, s_cf_span, w, label="CF + Span SetOp (Span, CF) [Unified]", color="#9467bd", alpha=0.95)
    ax1.bar(x + 2 * w, [s if not np.isnan(s) else 0 for s in graphecg_scores], w, 
            label="GraphECG Native (Ansari et al.)", color="#d62728", alpha=0.85)

    for i, s in enumerate(graphecg_scores):
        if np.isnan(s):
            ax1.text(x[i] + 2 * w, 0.42, "N/A", ha="center", va="bottom", fontsize=8, color="#d62728", rotation=90)

    # Highlight Limb and Redundant gains
    idx_limb = cfg_keys.index("S6_limb")
    ax1.annotate("+0.105 (CF)\n(6→2 leads)", 
                 xy=(x[idx_limb] - 1 * w, s_cf_basis[idx_limb]), 
                 xytext=(x[idx_limb] - 1 * w, s_cf_basis[idx_limb] + 0.035),
                 arrowprops=dict(facecolor="#1f77b4", arrowstyle="->", lw=1.5),
                 ha="center", fontsize=8.5, fontweight="bold", color="#1f77b4")

    idx_16 = cfg_keys.index("All_16_leads")
    ax1.annotate("+0.073 (CF)\n(Quotient)", 
                 xy=(x[idx_16] - 1 * w, s_cf_basis[idx_16]), 
                 xytext=(x[idx_16] - 1 * w, s_cf_basis[idx_16] + 0.035),
                 arrowprops=dict(facecolor="#1f77b4", arrowstyle="->", lw=1.5),
                 ha="center", fontsize=8.5, fontweight="bold", color="#1f77b4")

    idx_icm = cfg_keys.index("S_icm")
    ax1.annotate("+0.066 (Span)\n(Subspace Support)", 
                 xy=(x[idx_icm] + 0 * w, s_orig_span[idx_icm]), 
                 xytext=(x[idx_icm] + 0 * w, s_orig_span[idx_icm] + 0.045),
                 arrowprops=dict(facecolor="#2ca02c", arrowstyle="->", lw=1.5),
                 ha="center", fontsize=8.5, fontweight="bold", color="#2ca02c")

    ax1.set_ylabel("Macro AUROC (PTB-XL 5 Superclasses)")
    ax1.set_title("Panel A: 2x2 Factorial Performance Across All 13 Evaluation Configurations\n"
                  "(Factor 1: Basis vs. Span Support Training  |  Factor 2: Original Inputs vs. Canonical Frame Quotient)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cfg_labels, rotation=30, ha="right")
    ax1.set_ylim(0.40, 1.05)
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.95, ncol=2)

    # Panel B: Aggregates by Domain
    aggs = d2x2["aggregates"]
    domains = [
        ("canonical_sparse_1_to_3", "Canonical Sparse\n(S3, S2, S1)"),
        ("redundant_leads", "Redundant Leads\n(S6 Limb, Derived, All 16)"),
        ("oblique_unseen", "Unseen Oblique\n(ICM, ICM+I, 3 Obliques)"),
        ("overall_macro_mean", "Overall Mean\n(All 13 Configs)"),
    ]
    dom_keys = [d[0] for d in domains]
    dom_labels = [d[1] for d in domains]

    x_dom = np.arange(len(dom_keys))
    w_dom = 0.20

    y_1_1 = [aggs["cell_1_1_orig_basis"][k] for k in dom_keys]
    y_1_2 = [aggs["cell_1_2_cf_basis"][k] for k in dom_keys]
    y_2_1 = [aggs["cell_2_1_orig_span"][k] for k in dom_keys]
    y_2_2 = [aggs["cell_2_2_cf_span"][k] for k in dom_keys]

    ax2.bar(x_dom - 1.5 * w_dom, y_1_1, w_dom, label="Cell (1,1): Original SetOp (Basis, Orig)", color="#ff7f0e", alpha=0.85)
    ax2.bar(x_dom - 0.5 * w_dom, y_1_2, w_dom, label="Cell (1,2): CF-SetOp R026 (Basis, CF)", color="#1f77b4", alpha=0.95)
    ax2.bar(x_dom + 0.5 * w_dom, y_2_1, w_dom, label="Cell (2,1): Span SetOp R024 (Span, Orig)", color="#2ca02c", alpha=0.85)
    ax2.bar(x_dom + 1.5 * w_dom, y_2_2, w_dom, label="Cell (2,2): CF + Span SetOp (Span, CF) [Unified]", color="#9467bd", alpha=0.95)

    for i in range(len(dom_keys)):
        # print scores on top of bars
        ax2.text(x_dom[i] - 1.5 * w_dom, y_1_1[i] + 0.005, f"{y_1_1[i]:.3f}", ha="center", va="bottom", fontsize=8)
        ax2.text(x_dom[i] - 0.5 * w_dom, y_1_2[i] + 0.005, f"{y_1_2[i]:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1f77b4")
        ax2.text(x_dom[i] + 0.5 * w_dom, y_2_1[i] + 0.005, f"{y_2_1[i]:.3f}", ha="center", va="bottom", fontsize=8)
        ax2.text(x_dom[i] + 1.5 * w_dom, y_2_2[i] + 0.005, f"{y_2_2[i]:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#9467bd")

    ax2.set_ylabel("Macro AUROC")
    ax2.set_title("Panel B: Aggregate Performance by Measurement Subspace Regime", fontweight="bold")
    ax2.set_xticks(x_dom)
    ax2.set_xticklabels(dom_labels)
    ax2.set_ylim(0.70, 0.98)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.95)

    plt.tight_layout()
    out1 = ARTIFACTS_DIR / "r026_decisive_2x2_benchmark_comparison.png"
    plt.savefig(out1)
    plt.close()
    print(f"Saved Figure 1 to {out1}")

    # -------------------------------------------------------------
    # Figure 2: Noise Robustness, Invariance & Path Continuity
    # -------------------------------------------------------------
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(19, 5.5), dpi=300)

    # Panel A: Noise Attenuation on Redundant Leads (S6 Limb)
    noise_s6 = d_noise["empirical_noise_stress_test"]["S6_limb"]
    snr_keys = ["clean", "40dB", "30dB", "20dB", "10dB"]
    snr_labels = ["Clean", "40 dB", "30 dB", "20 dB", "10 dB"]
    cf_noise_auroc = [noise_s6[k]["cf_auroc"] for k in snr_keys]
    orig_noise_auroc = [noise_s6[k]["orig_auroc"] for k in snr_keys]
    cf_noise_drift = [noise_s6[k]["cf_rmse_logit_drift"] for k in snr_keys]
    orig_noise_drift = [noise_s6[k]["orig_rmse_logit_drift"] for k in snr_keys]

    x_snr = np.arange(len(snr_keys))
    ax_a.plot(x_snr, cf_noise_auroc, marker="o", lw=2.5, color="#1f77b4", label="CF-SetOp AUROC")
    ax_a.plot(x_snr, orig_noise_auroc, marker="s", lw=2.5, color="#ff7f0e", linestyle="--", label="Original SetOp AUROC")
    ax_a.set_xticks(x_snr)
    ax_a.set_xticklabels(snr_labels)
    ax_a.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax_a.set_ylabel("Macro AUROC")
    ax_a.set_ylim(0.65, 0.95)
    ax_a.set_title("Panel A: Noise Resilience on S6 Limb\n(||B_Q Q^+||_2 = 0.674 < 1.0 Noise Attenuation)", fontweight="bold")
    ax_a.legend(loc="lower left")

    # Panel B: Invariance under Invertible Remixings
    test_f = d_r026["test_f_invertible_remixings"]
    ranks = [1, 2, 3, 4, 6, 8]
    cf_diffs = [max(1e-16, test_f[f"rank_{r}"]["cf_max_logit_diff_across_20_remixes"]) for r in ranks]
    orig_diffs = [test_f[f"rank_{r}"]["orig_max_logit_diff_across_20_remixes"] for r in ranks]

    x_f = np.arange(len(ranks))
    w_f = 0.35
    ax_b.bar(x_f - w_f/2, cf_diffs, w_f, label="CF-SetOp (Max Logit Diff)", color="#1f77b4")
    ax_b.bar(x_f + w_f/2, orig_diffs, w_f, label="Original SetOp (Max Logit Diff)", color="#ff7f0e")
    ax_b.set_yscale("log")
    ax_b.set_ylim(1e-17, 1e2)
    ax_b.axhline(1e-5, color="red", linestyle=":", label="Tolerance Gate (1e-5)")
    ax_b.set_ylabel("Max Absolute Logit Diff (log scale)")
    ax_b.set_title("Panel B: Invertible Basis Invariance\n(20 Random Well-Conditioned R Matrices)", fontweight="bold")
    ax_b.set_xticks(x_f)
    ax_b.set_xticklabels([f"Rank {r}" for r in ranks])
    ax_b.legend(loc="upper left")

    # Panel C: Path Continuity & Pivot Audit
    test_g = d_r026["test_g_path_continuity"]
    frame_diffs = test_g["frame_diffs"]
    latent_diffs = test_g["latent_diffs"]
    angles = np.linspace(0, 90, len(frame_diffs))

    ax_c.plot(angles, frame_diffs, label="||B(α+Δ) - B(α)|| (Constant Δα)", color="#1f77b4", lw=2)
    ax_c.plot(angles, latent_diffs, label="||z(α+Δ) - z(α)|| (Latent Variation)", color="#d62728", lw=2, linestyle="--")
    # Correct theoretical crossing at alpha = 45 degrees
    ax_c.axvline(45.0, color="gray", linestyle=":", lw=2,
                 label="Theoretical Crossing (α = 45.0°)\n||Pe_I|| = ||Pe_II|| = 1/√2")
    ax_c.set_xlabel("Rotation Angle α (degrees, Lead I → Lead II)")
    ax_c.set_ylabel("Step Difference Norm")
    ax_c.set_title("Panel C: Path Continuity & Pivot Audit\n(Bounded Latent Variation along Arc; Dym et al. 2024)", fontweight="bold")
    ax_c.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    out2 = ARTIFACTS_DIR / "r026_decisive_noise_and_redundancy.png"
    plt.savefig(out2)
    plt.close()
    print(f"Saved Figure 2 to {out2}")


if __name__ == "__main__":
    main()
