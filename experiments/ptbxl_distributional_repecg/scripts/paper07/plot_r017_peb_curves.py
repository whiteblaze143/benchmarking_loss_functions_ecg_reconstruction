#!/usr/bin/env python3
"""Generate publication 4-panel figure for Milestone R017:
Empirical-Bayes Population-to-Personalized Physiological Monitoring.
Panels:
  A: Performance(K) [AUROC vs K]
  B: Δ(K) [AUROC gain over population model K=0]
  C: Coverage(K) [Availability Pareto frontier]
  D: FPR90(K) & False Alarm Reduction [Clinical False Positive Rate at 90% Sensitivity]
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

results_path = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_r017_peb_benchmark/r017_peb_benchmark_results.json")
with open(results_path) as f:
    res = json.load(f)

output_dir = Path("/home/mithunmanivannan/.gemini/antigravity-ide/brain/ba979872-97f8-4d89-9b0a-621cb4b9c051")
out_img = output_dir / "r017_peb_performance_curves.png"

fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
ax_a, ax_b = axes[0, 0], axes[0, 1]
ax_c, ax_d = axes[1, 0], axes[1, 1]

# Palette and styles
colors = {
    "SetOperator (d=256)": "#1f77b4",  # Blue
    "GraphECG (d=192)": "#ff7f0e",     # Orange
    "CSFM Base (d=768)": "#2ca02c",    # Green
}
markers = {
    "SetOperator (d=256)": "o",
    "GraphECG (d=192)": "s",
    "CSFM Base (d=768)": "^",
}

k_vals = [0, 1, 2, 3, 5]

# Focus on AF->SR (Certified Regime) and MI (Certified Regime)
# Panel A: Performance(K) for AF->SR Certified
task = "AF->SR"
reg = "certified"

for enc in ["SetOperator (d=256)", "GraphECG (d=192)", "CSFM Base (d=768)"]:
    k_data = res[enc][task][reg]["k_curves"]
    aurocs = [k_data[str(k)]["auroc"] for k in k_vals]
    ax_a.plot(k_vals, aurocs, marker=markers[enc], color=colors[enc], linewidth=2.5, markersize=8, label=enc)

ax_a.set_title(f"Panel A: Calibration Response Curve ({task} Certified)", fontsize=13, fontweight="bold")
ax_a.set_xlabel("Calibration Budget (Prior Encounters K)", fontsize=11)
ax_a.set_ylabel("AUROC", fontsize=11)
ax_a.set_xticks(k_vals)
ax_a.grid(True, linestyle="--", alpha=0.5)
ax_a.legend(loc="lower right", framealpha=0.9)
ax_a.set_ylim(0.60, 1.02)

# Panel B: Delta(K) = AUROC(K) - AUROC(0) for AF->SR and MI
for enc in ["SetOperator (d=256)", "GraphECG (d=192)", "CSFM Base (d=768)"]:
    k_data_af = res[enc]["AF->SR"]["certified"]["k_curves"]
    delta_af = [k_data_af[str(k)]["delta_auroc_vs_k0"] for k in k_vals]
    ax_b.plot(k_vals, delta_af, marker=markers[enc], color=colors[enc], linewidth=2.5, markersize=8,
              label=f"{enc.split()[0]} (AF->SR)")

ax_b.axhline(0, color="gray", linestyle=":", linewidth=1.5)
ax_b.set_title("Panel B: Δ AUROC vs Population Baseline (K=0)", fontsize=13, fontweight="bold")
ax_b.set_xlabel("Calibration Budget (Prior Encounters K)", fontsize=11)
ax_b.set_ylabel("Δ AUROC (K - 0)", fontsize=11)
ax_b.set_xticks(k_vals)
ax_b.grid(True, linestyle="--", alpha=0.5)
ax_b.legend(loc="lower left", framealpha=0.9)

# Panel C: Coverage(K) Availability Frontier
cov_vals = [res["SetOperator (d=256)"]["AF->SR"]["certified"]["k_curves"][str(k)]["coverage"] * 100 for k in k_vals]
ax_c.plot(k_vals, cov_vals, marker="D", color="#8c564b", linewidth=3, markersize=9, label="Cohort Availability")
for k, cov in zip(k_vals, cov_vals):
    ax_c.annotate(f"{cov:.1f}%", (k, cov), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold")

ax_c.set_title("Panel C: Clinical Deployment Availability Frontier", fontsize=13, fontweight="bold")
ax_c.set_xlabel("Calibration Requirement (Prior Encounters K)", fontsize=11)
ax_c.set_ylabel("Cohort Coverage (% Patients with >= K History)", fontsize=11)
ax_c.set_xticks(k_vals)
ax_c.set_ylim(-5, 115)
ax_c.grid(True, linestyle="--", alpha=0.5)
ax_c.axvspan(-0.5, 1.5, color="#2ca02c", alpha=0.1, label="High-Feasibility Zone (K=0, 1: 100% Coverage)")
ax_c.axvspan(1.5, 5.5, color="#d62728", alpha=0.08, label="Severe Attrition Zone (K>=2: <25% Coverage)")
ax_c.legend(loc="upper right", framealpha=0.9)

# Panel D: False Alarm Reduction (FPR at 90% Sensitivity) for AF->SR Certified
for enc in ["SetOperator (d=256)", "GraphECG (d=192)", "CSFM Base (d=768)"]:
    k_data = res[enc]["AF->SR"]["certified"]["k_curves"]
    fpr90s = [k_data[str(k)]["fpr90"] * 100 for k in k_vals]
    ax_d.plot(k_vals, fpr90s, marker=markers[enc], color=colors[enc], linewidth=2.5, markersize=8, label=enc)

ax_d.set_title("Panel D: Clinical False Alarm Reduction (FPR at 90% Sensitivity)", fontsize=13, fontweight="bold")
ax_d.set_xlabel("Calibration Budget (Prior Encounters K)", fontsize=11)
ax_d.set_ylabel("False Positive Rate (%)", fontsize=11)
ax_d.set_xticks(k_vals)
ax_d.grid(True, linestyle="--", alpha=0.5)
ax_d.legend(loc="upper right", framealpha=0.9)

plt.tight_layout()
plt.savefig(out_img, dpi=300)
print(f"Saved figure to {out_img}")
