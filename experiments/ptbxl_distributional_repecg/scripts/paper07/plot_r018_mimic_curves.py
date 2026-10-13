#!/usr/bin/env python3
"""Generate publication 4-panel figure for Milestone R018:
MIMIC-III-Ext-PPG Event-Level AF Calibration Frontier.
Panels:
  A: Cohort Availability Pareto Frontier (MIMIC ICU vs PTB-XL Outpatient)
  B: Empirical-Bayes Shrinkage & Uncertainty Dynamics (λ_K and V_K / V_0)
  C: Longitudinal Calibration Response Curve (SR->AF Certified)
  D: Clinical False Alarm Reduction (FPR at 90% Sensitivity)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

results_path = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_r018_mimic_calibration/r018_mimic_calibration_results.json")
with open(results_path) as f:
    res = json.load(f)

output_dir = Path("/home/mithunmanivannan/.gemini/antigravity-ide/brain/ba979872-97f8-4d89-9b0a-621cb4b9c051")
out_img = output_dir / "r018_mimic_calibration_frontier.png"

fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
ax_a, ax_b = axes[0, 0], axes[0, 1]
ax_c, ax_d = axes[1, 0], axes[1, 1]

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

# Panel A: Availability Frontier: MIMIC vs PTB-XL
mimic_cov = [res["coverage_frontier"][str(k)]["coverage"] * 100 for k in k_vals]
# From R017: PTB-XL coverage
ptbxl_cov = [100.0, 100.0, 22.5, 4.6, 0.0]

ax_a.plot(k_vals, mimic_cov, marker="o", color="#1f77b4", linewidth=3, markersize=9, label="MIMIC-III Dense ICU Monitoring (5,788 pts)")
ax_a.plot(k_vals, ptbxl_cov, marker="s", color="#d62728", linestyle="--", linewidth=2.5, markersize=8, label="PTB-XL Episodic Outpatient Visits (1,693 pts)")

for k, mc, pc in zip(k_vals, mimic_cov, ptbxl_cov):
    ax_a.annotate(f"{mc:.1f}%", (k, mc), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold", color="#1f77b4")
    ax_a.annotate(f"{pc:.1f}%", (k, pc), textcoords="offset points", xytext=(0, -14), ha="center", fontweight="bold", color="#d62728")

ax_a.set_title("Panel A: Clinical Availability Pareto Frontier (ICU vs Outpatient)", fontsize=13, fontweight="bold")
ax_a.set_xlabel("Required Prior Encounters (Calibration Budget K)", fontsize=11)
ax_a.set_ylabel("Cohort Coverage (% Patients with >= K History)", fontsize=11)
ax_a.set_xticks(k_vals)
ax_a.set_ylim(-8, 115)
ax_a.grid(True, linestyle="--", alpha=0.5)
ax_a.legend(loc="lower left", framealpha=0.9)

# Panel B: Empirical-Bayes Shrinkage & Uncertainty Reduction Dynamics
tau2 = 0.9364
sigma2 = 0.2526
lambdas = [0.0] + [tau2 / (tau2 + sigma2 / k) for k in [1, 2, 3, 5]]
uncertainty_ratio = [1.0] + [(tau2 * sigma2 / (k * tau2 + sigma2)) / tau2 for k in [1, 2, 3, 5]]

ax_b.plot(k_vals, lambdas, marker="D", color="#9467bd", linewidth=2.5, markersize=8, label="Shrinkage Weight λ_K (Patient Weight)")
ax_b.plot(k_vals, uncertainty_ratio, marker="v", color="#8c564b", linewidth=2.5, linestyle="--", markersize=8, label="Posterior Uncertainty Ratio V_K / V_0")

for k, l_val, u_val in zip(k_vals, lambdas, uncertainty_ratio):
    ax_b.annotate(f"{l_val:.2f}", (k, l_val), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold", color="#9467bd")
    ax_b.annotate(f"{u_val:.2f}", (k, u_val), textcoords="offset points", xytext=(0, -14), ha="center", fontweight="bold", color="#8c564b")

ax_b.set_title("Panel B: Empirical-Bayes Shrinkage & Uncertainty Dynamics", fontsize=13, fontweight="bold")
ax_b.set_xlabel("Calibration Budget (Prior Encounters K)", fontsize=11)
ax_b.set_ylabel("Weight / Uncertainty Ratio", fontsize=11)
ax_b.set_xticks(k_vals)
ax_b.set_ylim(-0.05, 1.15)
ax_b.grid(True, linestyle="--", alpha=0.5)
ax_b.legend(loc="center right", framealpha=0.9)

# Panel C: Longitudinal Calibration Response Curve (SR->AF Certified)
for enc in ["SetOperator (d=256)", "GraphECG (d=192)", "CSFM Base (d=768)"]:
    c_k = res["encoders"][enc]["tasks"]["SR->AF"]["certified"]["k_curves"]
    aurocs = [c_k[str(k)]["auroc"] for k in k_vals]
    lbl = enc if not res["encoders"][enc]["csfm_pretrain_domain_overlap"] else f"{enc} [Pretrain Overlap]"
    ax_c.plot(k_vals, aurocs, marker=markers[enc], color=colors[enc], linewidth=2.5, markersize=8, label=lbl)

ax_c.set_title("Panel C: Event-Level AF Calibration Response Curve (SR->AF Certified)", fontsize=13, fontweight="bold")
ax_c.set_xlabel("Calibration Budget (Prior Encounters K)", fontsize=11)
ax_c.set_ylabel("AUROC", fontsize=11)
ax_c.set_xticks(k_vals)
ax_c.set_ylim(0.65, 1.02)
ax_c.grid(True, linestyle="--", alpha=0.5)
ax_c.legend(loc="lower right", framealpha=0.9)

# Panel D: False Alarm Reduction (FPR at 90% Sensitivity)
for enc in ["SetOperator (d=256)", "GraphECG (d=192)", "CSFM Base (d=768)"]:
    c_k = res["encoders"][enc]["tasks"]["SR->AF"]["certified"]["k_curves"]
    fpr90s = [c_k[str(k)]["fpr90"] * 100 for k in k_vals]
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
