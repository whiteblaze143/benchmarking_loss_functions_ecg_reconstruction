#!/usr/bin/env python3
"""Generate publication-ready visualization comparing SetOperator vs CSFM Base geometry."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Load results
results_file = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_m0_fm_patient_geometry/results.json")
data = json.loads(results_file.read_text())
arms = data["arms"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 11})

# Subplot 1: Same-Patient Verification AUROC (Native & Matched)
ax1 = axes[0, 0]
categories = ["Native", "Matched d=256", "Matched d=128"]
so_euc = [arms["SetOperator (Native, d=256)"]["auroc_euclidean"],
          arms["SetOperator (Matched, d=256)"]["auroc_euclidean"],
          arms["SetOperator (Matched, d=128)"]["auroc_euclidean"]]
csfm_euc = [arms["CSFM Base (Native, d=768)"]["auroc_euclidean"],
            arms["CSFM Base (Matched, d=256)"]["auroc_euclidean"],
            arms["CSFM Base (Matched, d=128)"]["auroc_euclidean"]]
so_cos = [arms["SetOperator (Native, d=256)"]["auroc_cosine"],
          arms["SetOperator (Matched, d=256)"]["auroc_cosine"],
          arms["SetOperator (Matched, d=128)"]["auroc_cosine"]]
csfm_cos = [arms["CSFM Base (Native, d=768)"]["auroc_cosine"],
            arms["CSFM Base (Matched, d=256)"]["auroc_cosine"],
            arms["CSFM Base (Matched, d=128)"]["auroc_cosine"]]

x = np.arange(len(categories))
width = 0.2
ax1.bar(x - 1.5*width, so_euc, width, label="SetOperator (Euclidean)", color="#1f77b4")
ax1.bar(x - 0.5*width, csfm_euc, width, label="CSFM Base (Euclidean)", color="#aec7e8")
ax1.bar(x + 0.5*width, so_cos, width, label="SetOperator (Cosine)", color="#2ca02c")
ax1.bar(x + 1.5*width, csfm_cos, width, label="CSFM Base (Cosine)", color="#98df8a")
ax1.set_ylabel("Verification AUROC")
ax1.set_title("A. Same-Patient Pair Verification AUROC", fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.set_ylim(0.80, 1.00)
ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax1.legend(loc="lower right", fontsize=9)
ax1.grid(axis="y", linestyle=":", alpha=0.6)

# Subplot 2: Closed-Set Retrieval (Top-1, Top-5, Top-10)
ax2 = axes[0, 1]
ret_metrics = ["Top-1", "Top-5", "Top-10"]
so_ret = [arms["SetOperator (Matched, d=256)"]["retrieval"]["top1"] * 100,
          arms["SetOperator (Matched, d=256)"]["retrieval"]["top5"] * 100,
          arms["SetOperator (Matched, d=256)"]["retrieval"]["top10"] * 100]
csfm_ret = [arms["CSFM Base (Matched, d=256)"]["retrieval"]["top1"] * 100,
            arms["CSFM Base (Matched, d=256)"]["retrieval"]["top5"] * 100,
            arms["CSFM Base (Matched, d=256)"]["retrieval"]["top10"] * 100]

x2 = np.arange(len(ret_metrics))
w2 = 0.35
ax2.bar(x2 - w2/2, so_ret, w2, label="SetOperator (d=256)", color="#1f77b4")
ax2.bar(x2 + w2/2, csfm_ret, w2, label="CSFM Base (d=256)", color="#ff7f0e")
for i in range(len(ret_metrics)):
    ax2.text(x2[i] - w2/2, so_ret[i] + 1, f"{so_ret[i]:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax2.text(x2[i] + w2/2, csfm_ret[i] + 1, f"{csfm_ret[i]:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax2.set_ylabel("Retrieval Accuracy (%) [N=1,693]")
ax2.set_title("B. Closed-Set Patient Identification (Gallery vs Query)", fontweight="bold")
ax2.set_xticks(x2)
ax2.set_xticklabels(ret_metrics)
ax2.set_ylim(0, 65)
ax2.legend(loc="upper left")
ax2.grid(axis="y", linestyle=":", alpha=0.6)

# Subplot 3: Variance Decomposition (Patient vs Disease R^2)
ax3 = axes[1, 0]
v_arms = ["SetOperator (d=256)", "CSFM Base (d=256)"]
r2_pat = [arms["SetOperator (Matched, d=256)"]["variance_decomposition"]["r2_patient"] * 100,
          arms["CSFM Base (Matched, d=256)"]["variance_decomposition"]["r2_patient"] * 100]
r2_dis = [arms["SetOperator (Matched, d=256)"]["variance_decomposition"]["r2_disease"] * 100,
          arms["CSFM Base (Matched, d=256)"]["variance_decomposition"]["r2_disease"] * 100]

x3 = np.arange(len(v_arms))
ax3.bar(x3 - w2/2, r2_pat, w2, label="Patient Identity Variance (R²)", color="#4c72b0")
ax3.bar(x3 + w2/2, r2_dis, w2, label="Diagnostic Category Variance (R²)", color="#c44e52")
for i in range(len(v_arms)):
    ax3.text(x3[i] - w2/2, r2_pat[i] + 1.5, f"{r2_pat[i]:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax3.text(x3[i] + w2/2, r2_dis[i] + 1.5, f"{r2_dis[i]:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax3.set_ylabel("Explained Variance (%)")
ax3.set_title("C. Latent Variance Decomposition (Repeated Cohort)", fontweight="bold")
ax3.set_xticks(x3)
ax3.set_xticklabels(v_arms)
ax3.set_ylim(0, 105)
ax3.legend(loc="upper right")
ax3.grid(axis="y", linestyle=":", alpha=0.6)

# Subplot 4: Longitudinal Verification Across Time Gaps
ax4 = axes[1, 1]
t_labels = ["< 7 days", "7 - 30 days", "30 - 180 days", "> 180 days"]
so_time = [arms["SetOperator (Matched, d=256)"]["time_breakdown"][lbl]["auroc"] for lbl in t_labels]
csfm_time = [arms["CSFM Base (Matched, d=256)"]["time_breakdown"][lbl]["auroc"] for lbl in t_labels]

x4 = np.arange(len(t_labels))
ax4.plot(x4, so_time, marker="o", linewidth=2.5, markersize=8, label="SetOperator (Matched d=256)", color="#1f77b4")
ax4.plot(x4, csfm_time, marker="s", linewidth=2.5, markersize=8, label="CSFM Base (Matched d=256)", color="#ff7f0e")
for i in range(len(t_labels)):
    ax4.text(x4[i], so_time[i] + 0.005, f"{so_time[i]:.3f}", ha="center", fontsize=9, color="#1f77b4", fontweight="bold")
    ax4.text(x4[i], csfm_time[i] - 0.010, f"{csfm_time[i]:.3f}", ha="center", fontsize=9, color="#ff7f0e", fontweight="bold")
ax4.set_ylabel("Verification AUROC")
ax4.set_title("D. Same-Patient Stability Across Time Strata", fontweight="bold")
ax4.set_xticks(x4)
ax4.set_xticklabels(t_labels)
ax4.set_ylim(0.85, 0.98)
ax4.legend(loc="lower left")
ax4.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
out_fig = Path("/home/mithunmanivannan/.gemini/antigravity-ide/brain/ba979872-97f8-4d89-9b0a-621cb4b9c051/m0_fm_geometry_comparison.png")
plt.savefig(out_fig, dpi=300)
print(f"Saved figure to: {out_fig}")
