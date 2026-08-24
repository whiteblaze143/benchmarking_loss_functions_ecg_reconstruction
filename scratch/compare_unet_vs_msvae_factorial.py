#!/usr/bin/env python3
"""
Comprehensive Head-to-Head Architectural Comparison: U-Net vs MS-VAE
Compares validation missing lead Pearson correlation r, marginal loss main effects,
MMD kernel hierarchies, and loss sensitivity across all 160 factorial masks.
"""

from pathlib import Path
import re, json, yaml
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# 1. PARSE UNET FACTORIAL RESULTS (160 MODELS)
# -------------------------------------------------------------------------
unet_data = {}
wandb_dirs = sorted(list(Path("wandb").glob("run-*")))
for wdir in wandb_dirs:
    sum_file = wdir / "files" / "wandb-summary.json"
    cfg_file = wdir / "files" / "config.yaml"
    if not sum_file.exists() or not cfg_file.exists(): continue
    
    try:
        s = json.load(open(sum_file))
        c = yaml.safe_load(open(cfg_file))
        
        # Get run_name
        run_name = c.get("run_name", {}).get("value", "")
        if not run_name:
            run_name = c.get("wandb_run_name", {}).get("value", "")
        
        m = re.search(r"f_(\d{7})_s42", run_name)
        if not m: continue
        mask = m.group(1)
        
        # Get val_missing_pearson
        val_p = s.get("val_missing_pearson", np.nan)
        val_loss = s.get("val_loss", np.nan)
        
        # If not in summary, search stdout / log
        if np.isnan(val_p):
            out_log = wdir / "files" / "output.log"
            if out_log.exists():
                txt = out_log.read_text()
                mp = re.findall(r"Val Missing Pearson:\s*([0-9\.\-]+)", txt)
                if mp: val_p = float(mp[-1])
                
        unet_data[mask] = {
            "mask": mask,
            "unet_val_pearson": float(val_p) if val_p is not None else np.nan,
            "unet_val_loss": float(val_loss) if val_loss is not None else np.nan
        }
    except Exception as e:
        pass

# Also check logs directory for any missing unet masks
for p in Path("refine-logs/queue/logs").glob("f_*_s42.log"):
    m = re.search(r"f_(\d{7})_s42\.log", p.name)
    if not m: continue
    mask = m.group(1)
    if mask not in unet_data or np.isnan(unet_data[mask]["unet_val_pearson"]):
        txt = p.read_text()
        mp = re.findall(r"Val Missing Pearson:\s*([0-9\.\-]+)", txt)
        vl = re.findall(r"Val Loss:\s*([0-9\.\-]+)", txt)
        val_p = float(mp[-1]) if mp else np.nan
        val_l = float(vl[-1]) if vl else np.nan
        unet_data[mask] = {
            "mask": mask,
            "unet_val_pearson": val_p,
            "unet_val_loss": val_l
        }

# -------------------------------------------------------------------------
# 2. PARSE MS-VAE FACTORIAL RESULTS (154 MODELS)
# -------------------------------------------------------------------------
msvae_data = {}
for p in Path("refine-logs/queue_3arch/jobs").glob("msvae_f_*_s42.log"):
    m = re.search(r"msvae_f_(\d{7})_s42\.log", p.name)
    if not m: continue
    mask = m.group(1)
    txt = p.read_text()
    
    m_bp = re.search(r"Best Val Missing Pearson:\s*([0-9\.\-]+)", txt)
    best_p = float(m_bp.group(1)) if m_bp else np.nan
    
    m_vl = re.findall(r"Val Loss:\s*([0-9\.\-]+)", txt)
    val_l = float(m_vl[-1]) if m_vl else np.nan
    
    msvae_data[mask] = {
        "mask": mask,
        "msvae_val_pearson": best_p,
        "msvae_val_loss": val_l
    }

# -------------------------------------------------------------------------
# 3. MERGE & BUILD COMPARATIVE MATRIX
# -------------------------------------------------------------------------
all_masks = sorted(list(set(unet_data.keys()) | set(msvae_data.keys())))

rows = []
for mask in all_masks:
    u = unet_data.get(mask, {})
    m = msvae_data.get(mask, {})
    
    u_p = u.get("unet_val_pearson", np.nan)
    m_p = m.get("msvae_val_pearson", np.nan)
    
    # Loss components
    desc = []
    if int(mask[1]) == 1: desc.append("Corr")
    if int(mask[2]) == 1: desc.append("Deriv")
    if int(mask[3]) == 1: desc.append("VCG")
    if int(mask[4]) == 1: desc.append("ED")
    if int(mask[5]) == 1: desc.append("Lead")
    if int(mask[6]) > 0: desc.append(f"MMD(k={mask[6]})")
    loss_desc = "MSE + " + " + ".join(desc) if desc else "MSE only"
    
    rows.append({
        "mask": mask,
        "loss_desc": loss_desc,
        "corr": int(mask[1]), "deriv": int(mask[2]), "vcg": int(mask[3]),
        "ed": int(mask[4]), "lead": int(mask[5]), "mmd": int(mask[6]),
        "unet_pearson": u_p,
        "msvae_pearson": m_p,
        "delta_unet_minus_msvae": u_p - m_p if np.isfinite(u_p) and np.isfinite(m_p) else np.nan,
        "unet_val_loss": u.get("unet_val_loss", np.nan),
        "msvae_val_loss": m.get("msvae_val_loss", np.nan)
    })

df = pd.DataFrame(rows)

print("="*90)
print(f"HEAD-TO-HEAD FACTORIAL COMPARISON: U-NET (N={df['unet_pearson'].count()}) vs MS-VAE (N={df['msvae_pearson'].count()})")
print("="*90)

# Summary Stats
valid_pairs = df.dropna(subset=["unet_pearson", "msvae_pearson"])
print(f"\nTotal Overlapping Factorial Masks: {len(valid_pairs)} / 160")
print(f"U-Net Mean Pearson r   : {valid_pairs['unet_pearson'].mean():.4f} (Range: [{valid_pairs['unet_pearson'].min():.4f}, {valid_pairs['unet_pearson'].max():.4f}])")
print(f"MS-VAE Mean Pearson r  : {valid_pairs['msvae_pearson'].mean():.4f} (Range: [{valid_pairs['msvae_pearson'].min():.4f}, {valid_pairs['msvae_pearson'].max():.4f}])")

unet_wins = (valid_pairs["unet_pearson"] > valid_pairs["msvae_pearson"]).sum()
msvae_wins = (valid_pairs["msvae_pearson"] > valid_pairs["unet_pearson"]).sum()
ties = (valid_pairs["unet_pearson"] == valid_pairs["msvae_pearson"]).sum()
print(f"\nMask-by-Mask Win Tally : U-Net Wins = {unet_wins} ({unet_wins/len(valid_pairs)*100:.1f}%) | MS-VAE Wins = {msvae_wins} ({msvae_wins/len(valid_pairs)*100:.1f}%) | Ties = {ties}")

print("\n" + "="*90)
print("TOP 10 MASKS: HEAD-TO-HEAD COMPARISON (Sorted by U-Net Pearson r)")
print("="*90)
top_u = valid_pairs.sort_values(by="unet_pearson", ascending=False).head(10)
print(top_u[["mask", "unet_pearson", "msvae_pearson", "delta_unet_minus_msvae", "loss_desc"]].to_string(index=False))

print("\n" + "="*90)
print("TOP 10 MASKS: HEAD-TO-HEAD COMPARISON (Sorted by MS-VAE Pearson r)")
print("="*90)
top_m = valid_pairs.sort_values(by="msvae_pearson", ascending=False).head(10)
print(top_m[["mask", "msvae_pearson", "unet_pearson", "delta_unet_minus_msvae", "loss_desc"]].to_string(index=False))

print("\n" + "="*90)
print("MARGINAL MAIN EFFECTS (DELTA PEARSON r): U-NET vs MS-VAE")
print("="*90)
print(f"{'Loss Factor':28s} | {'U-Net Delta':14s} | {'MS-VAE Delta':14s} | {'Interaction / Shift'}")
print("-" * 90)

factors = [("corr", "Pearson Corr (corr)"), ("deriv", "Derivative L1 (deriv)"), ("vcg", "Kors 3D VCG (vcg)"), ("ed", "Energy Distance (ed)"), ("lead", "Lead Consistency (lead)")]
for factor, name in factors:
    u_act = valid_pairs[valid_pairs[factor] == 1]["unet_pearson"].mean()
    u_inact = valid_pairs[valid_pairs[factor] == 0]["unet_pearson"].mean()
    u_del = u_act - u_inact
    
    m_act = valid_pairs[valid_pairs[factor] == 1]["msvae_pearson"].mean()
    m_inact = valid_pairs[valid_pairs[factor] == 0]["msvae_pearson"].mean()
    m_del = m_act - m_inact
    
    shift = "Concordant (+)" if (u_del > 0 and m_del > 0) else "Concordant (-)" if (u_del < 0 and m_del < 0) else "Discordant!"
    print(f"{name:28s} | {u_del:+.4f} (r={u_act:.3f}) | {m_del:+.4f} (r={m_act:.3f}) | {shift}")

print("\n" + "="*90)
print("MMD KERNEL HIERARCHY COMPARISON: U-NET vs MS-VAE")
print("="*90)
kernel_names = {
    0: "Baseline (No MMD)",
    1: "Gaussian RBF",
    2: "Anatomical Block Laplacian",
    3: "Anatomical Block IMQ Multiscale",
    4: "KMeans Temporal Block"
}
print(f"{'MMD Kernel Level':38s} | {'U-Net Mean r':14s} | {'MS-VAE Mean r':14s} | {'U-Net vs MS-VAE'}")
print("-" * 90)
for k in range(5):
    u_k = valid_pairs[valid_pairs["mmd"] == k]["unet_pearson"].mean()
    m_k = valid_pairs[valid_pairs["mmd"] == k]["msvae_pearson"].mean()
    d_k = u_k - m_k
    print(f"Level {k} ({kernel_names[k]:30s}) | {u_k:.4f}         | {m_k:.4f}         | U-Net {d_k:+.4f}")
