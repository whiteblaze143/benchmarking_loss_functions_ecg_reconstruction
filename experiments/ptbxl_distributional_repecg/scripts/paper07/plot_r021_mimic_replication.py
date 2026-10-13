#!/usr/bin/env python3
"""
Plot Milestone R021: Preregistered External Confirmation on MIMIC-III-Ext-PPG
Figure Artifact: r021_mimic_replication_curves.png
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RESULTS_JSON = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_r021_real_waveforms/r021_real_replication_results.json")
OUTPUT_PNG = Path("/home/mithunmanivannan/.gemini/antigravity-ide/brain/ba979872-97f8-4d89-9b0a-621cb4b9c051/r021_mimic_replication_curves.png")


def main():
    if not RESULTS_JSON.exists():
        fallback_json = Path("/home/mithunmanivannan/.gemini/antigravity-ide/brain/ba979872-97f8-4d89-9b0a-621cb4b9c051/r021_real_replication_results.json")
        if fallback_json.exists():
            results_path = fallback_json
        else:
            raise FileNotFoundError(f"Neither {RESULTS_JSON} nor {fallback_json} exists yet.")
    else:
        results_path = RESULTS_JSON

    with open(results_path, "r") as f:
        data = json.load(f)

    encoders = ["SetOperator (d=256)", "GraphECG (d=192)", "CSFM Base (d=768)"]
    display_names = ["SetOperator\n(d=256, Contract-Matched)", "GraphECG\n(d=192, Sparse Lead)", "CSFM Base\n(d=768, 30s-Pooled)"]

    # Support new schema with primary_cohort_strict_sqi
    source = data.get("primary_cohort_strict_sqi", data.get("encoders", {}))

    k0_aucs = [source[e]["auroc_k0"] if "auroc_k0" in source[e] else source[e]["sealed_test_metrics"]["auroc_k0"] for e in encoders]
    k1_aucs = [source[e]["auroc_k1"] if "auroc_k1" in source[e] else source[e]["sealed_test_metrics"]["auroc_k1"] for e in encoders]
    delta_aucs = [source[e]["delta_auroc"] if "delta_auroc" in source[e] else source[e]["sealed_test_metrics"]["delta_auroc"] for e in encoders]
    
    ci_auc_low = [
        source[e]["delta_auroc_ci"]["ci_low"] if "delta_auroc_ci" in source[e]
        else source[e]["clustered_bootstrap"]["delta_auroc"]["ci_low"] for e in encoders
    ]
    ci_auc_high = [
        source[e]["delta_auroc_ci"]["ci_high"] if "delta_auroc_ci" in source[e]
        else source[e]["clustered_bootstrap"]["delta_auroc"]["ci_high"] for e in encoders
    ]

    k0_fprs = [
        (source[e]["fpr_tau_k0"] if "fpr_tau_k0" in source[e] else source[e]["sealed_test_metrics"]["fpr_locked_k0"]) * 100 for e in encoders
    ]
    k1_fprs = [
        (source[e]["fpr_tau_k1"] if "fpr_tau_k1" in source[e] else source[e]["sealed_test_metrics"]["fpr_locked_k1"]) * 100 for e in encoders
    ]
    delta_fprs = [
        (source[e]["delta_fpr_tau"] if "delta_fpr_tau" in source[e] else source[e]["sealed_test_metrics"]["delta_fpr_locked"]) * 100 for e in encoders
    ]
    ci_fpr_low = [
        (source[e]["delta_fpr_ci"]["ci_low"] if "delta_fpr_ci" in source[e]
         else source[e]["clustered_bootstrap"]["delta_fpr_locked"]["ci_low"]) * 100 for e in encoders
    ]
    ci_fpr_high = [
        (source[e]["delta_fpr_ci"]["ci_high"] if "delta_fpr_ci" in source[e]
         else source[e]["clustered_bootstrap"]["delta_fpr_locked"]["ci_high"]) * 100 for e in encoders
    ]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    # -------------------------------------------------------------
    # Panel A: Test AUROC (K=0 vs K=1 MV-PEB)
    # -------------------------------------------------------------
    ax = axes[0]
    x = np.arange(len(encoders))
    width = 0.35

    rects1 = ax.bar(x - width/2, k0_aucs, width, label="K=0 Population", color="#4A5568", alpha=0.9, edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, k1_aucs, width, label="K=1 MV-PEB", color="#2B6CB0", alpha=0.9, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Test AUROC (Fold 8 Sealed Test)", fontsize=11, fontweight="bold")
    ax.set_title("A. Empirical Replication on Real MIMIC Waveforms\n(Primary Strict SQI=[1,1,1] Cohort: AF $\\to$ SR vs AF $\\to$ AF Controls)", fontsize=10.5, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10)
    ax.set_ylim(0.50, 1.05)
    ax.legend(loc="lower right", frameon=True)

    # Add delta annotations
    for i in range(len(encoders)):
        d_auc = delta_aucs[i]
        low = ci_auc_low[i]
        high = ci_auc_high[i]
        ax.annotate(
            f"$\\Delta$={d_auc:+.4f}\n[{low:+.4f}, {high:+.4f}]",
            xy=(x[i] + width/2, k1_aucs[i]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold",
            color="#1A365D"
        )

    # -------------------------------------------------------------
    # Panel B: Validation-Locked FPR at 90% Sensitivity
    # -------------------------------------------------------------
    ax = axes[1]
    rects1 = ax.bar(x - width/2, k0_fprs, width, label="K=0 Population", color="#718096", alpha=0.9, edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, k1_fprs, width, label="K=1 MV-PEB", color="#C53030", alpha=0.85, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Validation-Locked Test FPR (%) [$\\tau^*$ @ 90% Sens]", fontsize=11, fontweight="bold")
    ax.set_title("B. Prespecified Operating Metric (Locked $\\tau^*$)\n(Target: $\\Delta\\text{FPR}_{\\tau^*} \\leq 0.0$ and Preserved Sensitivity)", fontsize=10.5, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10)
    y_max = max(max(k0_fprs), max(k1_fprs), 40) + 15
    ax.set_ylim(0, y_max)
    ax.legend(loc="upper right", frameon=True)

    # Add delta annotations
    for i in range(len(encoders)):
        d_fpr = delta_fprs[i]
        low = ci_fpr_low[i]
        high = ci_fpr_high[i]
        ax.annotate(
            f"$\\Delta$={d_fpr:+.1f}%\n[{low:+.1f}%, {high:+.1f}%]",
            xy=(x[i] + width/2, k1_fprs[i]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold",
            color="#742A2A"
        )

    # -------------------------------------------------------------
    # Panel C: Cohort Sensitivity Comparison
    # -------------------------------------------------------------
    ax = axes[2]
    # Compare Delta AUROC across Strict [1, 1, 1], Sensitivity (>=2/3), and Harmonized 10s
    sens_source = data.get("sensitivity_cohort_looser_sqi", {})
    harm_source = data.get("harmonized_10s_sensitivity", {})

    has_sens = bool(sens_source and harm_source)

    if has_sens:
        d_strict = [source[e]["delta_auroc"] for e in encoders]
        d_sens = [sens_source[e]["delta_auroc"] for e in encoders]
        d_harm = [harm_source[e]["delta_auroc"] for e in encoders]

        x_c = np.arange(len(encoders))
        w_c = 0.25

        ax.bar(x_c - w_c, d_strict, w_c, label="Primary: Strict SQI [1,1,1]", color="#2B6CB0", alpha=0.9, edgecolor="black")
        ax.bar(x_c, d_sens, w_c, label="Sensitivity 1: Looser SQI ($\\geq 2/3$)", color="#319795", alpha=0.9, edgecolor="black")
        ax.bar(x_c + w_c, d_harm, w_c, label="Sensitivity 2: Harmonized 10-s", color="#805AD5", alpha=0.9, edgecolor="black")

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(0.02, color="green", linestyle=":", linewidth=1.0, label="Utility Target (+0.02)")
        ax.set_ylabel("$\\Delta$ AUROC ($K=1$ vs $K=0$)", fontsize=11, fontweight="bold")
        ax.set_title("C. Robustness Across Quality & Window Regimes\n(Sensitivity Analysis: Strict vs Looser vs 10s Harmonized)", fontsize=10.5, fontweight="bold", pad=10)
        ax.set_xticks(x_c)
        ax.set_xticklabels(display_names, fontsize=10)
        ax.legend(loc="lower left", frameon=True, fontsize=8.5)
    else:
        # Fallback to decision framework if sensitivity not yet present
        ax.axis("off")
        ax.set_title("C. Operational Decision Framework", fontsize=11, fontweight="bold", pad=10)
        bbox_cold = dict(boxstyle="round,pad=0.6", facecolor="#E2E8F0", edgecolor="#4A5568", linewidth=1.5)
        ax.text(0.5, 0.5, "Operational Decision Gate Active", ha="center", va="center", fontsize=11, fontweight="bold", bbox=bbox_cold)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
