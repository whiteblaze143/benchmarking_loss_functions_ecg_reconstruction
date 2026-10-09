#!/usr/bin/env python3
"""Generate publication-quality figures for Milestone R025: Observable-Subspace SetOperator."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
BENCH_JSON = CODEX_DIR / "paper07_operator/r025_benchmark_results/r025_benchmark_seed42.json"
STATS_JSON = CODEX_DIR / "paper07_operator/r025_statistics/r025_bootstrap_statistics_seed42.json"
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
    with open(BENCH_JSON) as f:
        bench_data = f.read()
    data = json.loads(bench_data)

    results = data["results_by_arm"]
    
    # 1. Figure 1: Configuration Comparison Bar Chart
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

    os_scores = [results["os_only"][k]["macro_auroc"] for k in cfg_keys]
    os_nop_scores = [results["os_without_p"][k]["macro_auroc"] for k in cfg_keys]
    lq_scores = [results["learned_q"][k]["macro_auroc"] for k in cfg_keys]
    canon_scores = [results["canonical_setoperator"][k]["macro_auroc"] for k in cfg_keys]
    
    graphecg_scores = []
    for k in cfg_keys:
        val = results["graphecg"][k].get("macro_auroc")
        graphecg_scores.append(val if val is not None else np.nan)

    x = np.arange(len(cfg_keys))
    width = 0.17

    fig, ax = plt.subplots(figsize=(16, 7), dpi=300)

    rects1 = ax.bar(x - 2 * width, os_scores, width, label="OS-Only (Model B, 133k)", color="#1f77b4", alpha=0.9)
    rects2 = ax.bar(x - 1 * width, os_nop_scores, width, label="OS w/o P (Model C, 133k)", color="#aec7e8", alpha=0.8)
    rects3 = ax.bar(x + 0 * width, lq_scores, width, label="Learned-Q Control (Model D, 141k)", color="#2ca02c", alpha=0.9)
    rects4 = ax.bar(x + 1 * width, canon_scores, width, label="Canonical SetOp (Transformer, 6.3M)", color="#ff7f0e", alpha=0.9)
    rects5 = ax.bar(x + 2 * width, [s if not np.isnan(s) else 0 for s in graphecg_scores], width, 
                    label="GraphECG (Ansari et al., 1.2M)", color="#d62728", alpha=0.9)

    # Add hatched indicators for N/A in GraphECG
    for i, s in enumerate(graphecg_scores):
        if np.isnan(s):
            ax.text(x[i] + 2 * width, 0.05, "N/A", ha="center", va="bottom", fontsize=8, color="#d62728", rotation=90)

    ax.set_ylabel("Macro AUROC (PTB-XL Superclass)")
    ax.set_title("Milestone R025: Architectural Inductive Bias Benchmark Across Measurement Configurations\n"
                 "(Full-Lead Matched Exposure Training, Zero-Shot Test Shift)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cfg_labels, rotation=35, ha="right")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Chance Level (0.50)")
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    plt.tight_layout()
    out_fig1 = ARTIFACTS_DIR / "r025_benchmark_comparison.png"
    plt.savefig(out_fig1)
    plt.close()
    print(f"Saved Figure 1 to {out_fig1}")

    # 2. Figure 2: Exact Observational Invariance and Redundant Measurement Resilience
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: E3 Invariance Stress Test Log Errors
    invar_tests = ["Invertible R\n(6x6 Remix)", "Permutation\n(Order Swap)", "Sign Negation\n(q -> -q, y -> -y)"]
    latent_diffs = [9.54e-7, 1e-16, 1e-16]  # 0 clamped to 1e-16 for log plot
    logit_diffs = [2.86e-6, 1e-16, 1e-16]

    x_inv = np.arange(len(invar_tests))
    w_inv = 0.35
    ax1.bar(x_inv - w_inv/2, latent_diffs, w_inv, label="Latent Embedding L-inf Diff", color="#1f77b4")
    ax1.bar(x_inv + w_inv/2, logit_diffs, w_inv, label="Diagnostic Logits L-inf Diff", color="#ff7f0e")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-17, 1e-3)
    ax1.set_ylabel("Maximum L-infinity Absolute Difference (log scale)")
    ax1.set_title("E3: Observation-Basis Invariance Stress Test\n(Exact Physical Quotient Invariance)", fontweight="bold")
    ax1.set_xticks(x_inv)
    ax1.set_xticklabels(invar_tests)
    ax1.axhline(1e-5, color="red", linestyle=":", label="Tolerance Gate (1e-5)")
    ax1.legend(loc="upper left")

    # Panel B: Redundant Measurements Comparison (Q8 vs All 16 Leads)
    models = ["OS-Only", "OS w/o P", "Learned-Q", "Canonical SetOp"]
    q8_vals = [results["os_only"]["Q8_full"]["macro_auroc"],
               results["os_without_p"]["Q8_full"]["macro_auroc"],
               results["learned_q"]["Q8_full"]["macro_auroc"],
               results["canonical_setoperator"]["Q8_full"]["macro_auroc"]]
    all16_vals = [results["os_only"]["All_16_leads"]["macro_auroc"],
                 results["os_without_p"]["All_16_leads"]["macro_auroc"],
                 results["learned_q"]["All_16_leads"]["macro_auroc"],
                 results["canonical_setoperator"]["All_16_leads"]["macro_auroc"]]

    x_red = np.arange(len(models))
    w_red = 0.35
    ax2.bar(x_red - w_red/2, q8_vals, w_red, label="Q8 Full (8 Leads)", color="#2ca02c", alpha=0.9)
    ax2.bar(x_red + w_red/2, all16_vals, w_red, label="All 16 Leads (+8 Redundant)", color="#9467bd", alpha=0.9)
    
    # Annotate deltas
    for i in range(len(models)):
        delta = all16_vals[i] - q8_vals[i]
        sign = "+" if delta >= 0 else ""
        text_col = "green" if abs(delta) < 1e-4 else ("red" if delta < 0 else "blue")
        ax2.text(x_red[i] + w_red/2, all16_vals[i] + 0.01, f"{sign}{delta:.4f}", 
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color=text_col)

    ax2.set_ylabel("Macro AUROC")
    ax2.set_title("Resilience to Redundant Derived Leads\n(Q8 vs. All 16 Leads)", fontweight="bold")
    ax2.set_xticks(x_red)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0.80, 0.98)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    out_fig2 = ARTIFACTS_DIR / "r025_algebraic_invariance.png"
    plt.savefig(out_fig2)
    plt.close()
    print(f"Saved Figure 2 to {out_fig2}")

if __name__ == "__main__":
    main()
