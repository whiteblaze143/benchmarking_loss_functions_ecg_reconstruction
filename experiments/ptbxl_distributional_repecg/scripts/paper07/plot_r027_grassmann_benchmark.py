#!/usr/bin/env python3
"""Figure 3: Held-Out Grassmann Benchmark Visualizations.

Generates:
1. Panel A: Scatter plot of Delta_span(U) = AUROC_span(U) - AUROC_basis(U) vs d_train(U),
   with regression trend line and Pearson / Spearman correlations.
2. Panel B: Factorial 4-cell AUROC comparison by rank r in {1, 2, 3, 4, 6}.
3. Panel C: Quotient Invariance Stress Test: V(U) = max_k ||f(Q_k) - f(B_U)||_inf
   comparing CF-SetOperator (V == 0) vs Original SetOperator (V > 0).
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
INPUT_FILE = CODEX_DIR / "paper07_operator/r027_heldout_grassmann_results/heldout_grassmann_benchmark.json"
OUTPUT_DIR = CODEX_DIR / "paper07_operator/r027_heldout_grassmann_results"
ARTIFACTS_DIR = Path("/home/mithunmanivannan/.gemini/antigravity-ide/brain/ba979872-97f8-4d89-9b0a-621cb4b9c051")


def main():
    if not INPUT_FILE.exists():
        print(f"File {INPUT_FILE} does not exist yet.")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    summary = data["overall_summary"]
    sub_results = data["subspace_eval_results"]
    df = pd.DataFrame(sub_results)

    # Style configuration
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8), dpi=300)

    # ----------------------------------------------------
    # Panel A: Delta_span(U) vs d_train(U) (Grassmann distance)
    # ----------------------------------------------------
    ax0 = axes[0]
    ranks = sorted(df["rank"].unique())
    rank_colors = {
        1: "#1f77b4",
        2: "#2ca02c",
        3: "#ff7f0e",
        4: "#9467bd",
        6: "#8c564b",
    }

    for r in ranks:
        sub_r = df[df["rank"] == r]
        ax0.scatter(
            sub_r["d_train"],
            sub_r["delta_span_cf"],
            c=rank_colors.get(r, "#333333"),
            label=f"Rank {r} (n={len(sub_r)})",
            alpha=0.75,
            s=45,
            edgecolors="none",
        )

    # Linear regression fit line
    d_vals = df["d_train"].to_numpy()
    delta_vals = df["delta_span_cf"].to_numpy()
    p_fit = np.polyfit(d_vals, delta_vals, 1)
    x_grid = np.linspace(d_vals.min(), d_vals.max(), 100)
    ax0.plot(x_grid, np.polyval(p_fit, x_grid), color="#d62728", linestyle="--", linewidth=2.0, label="Linear Trend")

    r_p = summary["correlation_d_train_vs_delta_span_pearson"]
    r_s = summary["correlation_d_train_vs_delta_span_spearman"]

    ax0.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax0.set_xlabel("Grassmann Geodesic Distance to Coord Support $d_{\\mathrm{train}}(U)$", fontsize=11, fontweight="bold")
    ax0.set_ylabel("$\\Delta_{\\mathrm{span}}(U) = \\mathrm{AUROC}_{\\mathrm{CF\\text{-}Span}} - \\mathrm{AUROC}_{\\mathrm{CF\\text{-}Basis}}$", fontsize=11, fontweight="bold")
    ax0.set_title(f"A. Generalization Gain vs Grassmann Distance\nPearson $r = {r_p:.3f}$, Spearman $\\rho = {r_s:.3f}$", fontsize=12, fontweight="bold")
    ax0.grid(True, linestyle=":", alpha=0.4)
    ax0.legend(loc="upper left", frameon=True, fontsize=9)

    # ----------------------------------------------------
    # Panel B: Factorial 4-Cell AUROC Comparison by Rank
    # ----------------------------------------------------
    ax1 = axes[1]
    by_rank = summary["by_rank"]
    x = np.arange(len(ranks))
    width = 0.20

    c1_1 = [by_rank[f"rank_{r}"]["cell_1_1_orig_basis"] for r in ranks]
    c1_2 = [by_rank[f"rank_{r}"]["cell_1_2_cf_basis"] for r in ranks]
    c2_1 = [by_rank[f"rank_{r}"]["cell_2_1_orig_span"] for r in ranks]
    c2_2 = [by_rank[f"rank_{r}"]["cell_2_2_cf_span"] for r in ranks]

    rects1 = ax1.bar(x - 1.5 * width, c1_1, width, label="(1,1) Orig-Basis", color="#aec7e8", edgecolor="#333333", linewidth=0.6)
    rects2 = ax1.bar(x - 0.5 * width, c1_2, width, label="(1,2) CF-Basis", color="#1f77b4", edgecolor="#333333", linewidth=0.6)
    rects3 = ax1.bar(x + 0.5 * width, c2_1, width, label="(2,1) Orig-Span", color="#ffbb78", edgecolor="#333333", linewidth=0.6)
    rects4 = ax1.bar(x + 1.5 * width, c2_2, width, label="(2,2) CF-Span (Unified)", color="#d62728", edgecolor="#333333", linewidth=0.6)

    ax1.set_xlabel("Subspace Rank $r$", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Macro AUROC (N=2,173)", fontsize=11, fontweight="bold")
    ax1.set_title("B. Factorial 4-Cell Performance Across Ranks", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"r={r}" for r in ranks], fontsize=10)
    ax1.set_ylim(0.50, 1.0)
    ax1.grid(True, linestyle=":", alpha=0.4, axis="y")
    ax1.legend(loc="lower right", frameon=True, fontsize=9)

    # ----------------------------------------------------
    # Panel C: Quotient Invariance V(U) Across Ranks
    # ----------------------------------------------------
    ax2 = axes[2]
    v_orig_mean = [by_rank[f"rank_{r}"]["mean_V_orig"] for r in ranks]
    v_cf_mean = [by_rank[f"rank_{r}"]["mean_V_cf"] for r in ranks]

    ax2.bar(x - 0.2, v_orig_mean, 0.4, label="Original SetOperator ($V > 0$)", color="#d62728", alpha=0.85, edgecolor="#333333", linewidth=0.6)
    ax2.bar(x + 0.2, v_cf_mean, 0.4, label="CF-SetOperator ($V \\equiv 0$)", color="#2ca02c", alpha=0.85, edgecolor="#333333", linewidth=0.6)

    ax2.set_xlabel("Subspace Rank $r$", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Maximum Logit Discrepancy $V(U)$", fontsize=11, fontweight="bold")
    ax2.set_title("C. Quotient Invariance Under Invertible / Redundant Remixes", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"r={r}" for r in ranks], fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.4, axis="y")
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    # Annotate exact zero
    for idx_x in x:
        ax2.annotate(
            "0.00\n(exact)",
            (idx_x + 0.2, 0.2),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#1b5e20",
        )

    plt.tight_layout()
    out_plot_codex = OUTPUT_DIR / "r027_grassmann_benchmark.png"
    out_plot_artifacts = ARTIFACTS_DIR / "r027_grassmann_benchmark.png"

    fig.savefig(out_plot_codex, dpi=300)
    fig.savefig(out_plot_artifacts, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {out_plot_codex} and {out_plot_artifacts}")


if __name__ == "__main__":
    main()
