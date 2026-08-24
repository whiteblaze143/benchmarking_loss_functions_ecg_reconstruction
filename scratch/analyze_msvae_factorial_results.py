#!/usr/bin/env python3
"""
Parse all 159 completed MS-VAE factorial logs to extract validation performance,
marginal main effects of loss terms, MMD kernel hierarchies, and top Pareto masks.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

log_dir = Path("refine-logs/queue_3arch/jobs")
logs = sorted(list(log_dir.glob("msvae_f_*.log")))

records = []
for p in logs:
    m = re.search(r"msvae_f_(\d{7})_s42\.log", p.name)
    if not m: continue
    mask = m.group(1)
    
    text = p.read_text()
    
    # Extract Best Val Missing Pearson
    m_bp = re.search(r"Best Val Missing Pearson:\s*([0-9\.\-]+)", text)
    best_pearson = float(m_bp.group(1)) if m_bp else np.nan
    
    # Extract Final Val Loss
    m_vl = re.findall(r"Val Loss:\s*([0-9\.\-]+)", text)
    final_val_loss = float(m_vl[-1]) if m_vl else np.nan
    
    # Extract Final Val Missing Pearson
    m_fp = re.findall(r"Val Missing Pearson:\s*([0-9\.\-]+)", text)
    final_pearson = float(m_fp[-1]) if m_fp else np.nan
    
    # Parse loss term components
    desc = []
    if int(mask[1]) == 1: desc.append("Corr")
    if int(mask[2]) == 1: desc.append("Deriv")
    if int(mask[3]) == 1: desc.append("VCG")
    if int(mask[4]) == 1: desc.append("ED")
    if int(mask[5]) == 1: desc.append("Lead")
    if int(mask[6]) > 0: desc.append(f"MMD(k={mask[6]})")
    loss_desc = "MSE + " + " + ".join(desc) if desc else "MSE only"
    
    records.append({
        "model_id": f"msvae_f_{mask}_s42",
        "mask": mask,
        "loss_desc": loss_desc,
        "corr": int(mask[1]),
        "deriv": int(mask[2]),
        "vcg": int(mask[3]),
        "ed": int(mask[4]),
        "lead": int(mask[5]),
        "mmd": int(mask[6]),
        "best_val_missing_pearson": best_pearson,
        "final_val_missing_pearson": final_pearson,
        "final_val_loss": final_val_loss
    })

df = pd.DataFrame(records).dropna(subset=["best_val_missing_pearson"])
df = df.sort_values(by="best_val_missing_pearson", ascending=False).reset_index(drop=True)

print("="*80)
print(f"MS-VAE FACTORIAL QUEUE ANALYSIS ({len(df)} Completed Models)")
print("="*80)

print("\n--- TOP 10 MS-VAE MODELS (Sorted by Missing Lead Pearson r) ---")
print(df[["mask", "best_val_missing_pearson", "final_val_loss", "loss_desc"]].head(10).to_string(index=False))

print("\n--- BOTTOM 5 MS-VAE MODELS ---")
print(df[["mask", "best_val_missing_pearson", "final_val_loss", "loss_desc"]].tail(5).to_string(index=False))

print("\n--- FACTORIAL MARGINAL MAIN EFFECTS (DELTA PEARSON r) ---")
for factor, name in [("corr", "Pearson Corr (corr)"), ("deriv", "Derivative L1 (deriv)"), ("vcg", "Kors 3D VCG (vcg)"), ("ed", "Energy Distance (ed)"), ("lead", "Lead Consistency (lead)")]:
    active = df[df[factor] == 1]["best_val_missing_pearson"].mean()
    inactive = df[df[factor] == 0]["best_val_missing_pearson"].mean()
    delta = active - inactive
    print(f"  {name:30s}: Active={active:.4f} | Inactive={inactive:.4f} | Delta = {delta:+.4f}")

print("\n--- MMD KERNEL HIERARCHY (MS-VAE) ---")
kernel_names = {
    0: "Baseline (No MMD)",
    1: "Gaussian RBF",
    2: "Anatomical Block Laplacian",
    3: "Anatomical Block IMQ Multiscale",
    4: "KMeans Temporal Block"
}
for k in range(5):
    sub = df[df["mmd"] == k]
    if len(sub):
        print(f"  MMD Level {k} ({kernel_names[k]:32s}): N={len(sub):2d} | Mean Pearson r = {sub['best_val_missing_pearson'].mean():.4f} | Best r = {sub['best_val_missing_pearson'].max():.4f}")

print("\n--- SYNERGIES & LOSS INTERACTION PATTERNS ---")
# Compare Pure MSE vs Best Multi-loss
mse_only = df[df["mask"] == "1000000"]["best_val_missing_pearson"].values
if len(mse_only):
    print(f"  Pure Baseline MSE (1000000)      : r = {mse_only[0]:.4f}")
print(f"  Top Multi-Loss Configuration ({df.iloc[0]['mask']}): r = {df.iloc[0]['best_val_missing_pearson']:.4f} ({df.iloc[0]['loss_desc']})")
print(f"  Performance Range Across Factorial Space: [{df['best_val_missing_pearson'].min():.4f} to {df['best_val_missing_pearson'].max():.4f}]")
