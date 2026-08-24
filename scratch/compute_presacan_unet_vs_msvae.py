#!/usr/bin/env python3
"""
Exhaustive Head-to-Head Architecture Audit: U-Net vs. MS-VAE
Presacan et al. (Nature Comms Med 2025) Bland-Altman R^2 & Variance Collapse Comparison.
Tests whether Regression-to-the-Mean (R2M) is a U-Net limitation or a fundamental autoencoding ceiling.
Runs strictly on 1 CPU thread.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import json
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
from scripts.evaluate_clinical_biomarkers_multids import (
    load_adapter, SimplePTBDataset, LEAD_NAMES
)
from unified_latents.engineering.experimental.Multi_Scale_VAE import WearECGVAE
from scripts.train_mcma_3lead import MCMAModel

device = torch.device("cpu")

ptb_data_dir = "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/data/ptb_xl/tensors/test"
test_dataset = SimplePTBDataset(ptb_data_dir)

# 300 test records for fast, low-resource evaluation
N_EVAL = min(300, len(test_dataset))
subset_indices = list(range(N_EVAL))
subset_dataset = torch.utils.data.Subset(test_dataset, subset_indices)
test_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=30, shuffle=False)

print(f"Running U-Net vs MS-VAE Presacan Bland-Altman Audit on {N_EVAL} test ECGs...")

# Matched model pairs (U-Net vs MS-VAE)
model_pairs = [
    {
        "mask": "1000000", "loss": "Pure MSE Baseline",
        "unet": {"id": "f_1000000_s42", "kind": "unet", "name": "U-Net Pure MSE"},
        "msvae": {"id": "factorial_msvae_1000000_s42", "kind": "msvae", "name": "MS-VAE Pure MSE"}
    },
    {
        "mask": "1000003", "loss": "MSE + MMD_IMQ",
        "unet": {"id": "f_1000003_s42", "kind": "unet", "name": "U-Net MMD_IMQ"},
        "msvae": {"id": "factorial_msvae_1000003_s42", "kind": "msvae", "name": "MS-VAE MMD_IMQ"}
    },
    {
        "mask": "1000004", "loss": "MSE + MMD_Temporal",
        "unet": {"id": "f_1000004_s42", "kind": "unet", "name": "U-Net MMD_Temporal"},
        "msvae": {"id": "factorial_msvae_1000004_s42", "kind": "msvae", "name": "MS-VAE MMD_Temporal"}
    },
    {
        "mask": "1010000", "loss": "MSE + Kors 3D VCG",
        "unet": {"id": "f_1010000_s42", "kind": "unet", "name": "U-Net Kors VCG"},
        "msvae": {"id": "factorial_msvae_1010000_s42", "kind": "msvae", "name": "MS-VAE Kors VCG"}
    },
    {
        "mask": "1110004", "loss": "Universal Champion (MSE+Corr+Deriv+MMD4)",
        "unet": {"id": "f_1110004_s42", "kind": "unet", "name": "U-Net Champion (1110004)"},
        "msvae": {"id": "factorial_msvae_1110004_s42", "kind": "msvae", "name": "MS-VAE Champion (1110004)"}
    }
]

def extract_amplitudes(sig_12x5000):
    res = {}
    for idx, lead_name in enumerate(LEAD_NAMES):
        lead_sig = sig_12x5000[idx]
        r_amp = float(np.percentile(lead_sig, 99.0))
        s_amp = float(np.percentile(lead_sig, 1.0))
        st_amp = float(np.mean(lead_sig[200:350])) if len(lead_sig) >= 350 else float(np.mean(lead_sig))
        t_amp = float(np.percentile(lead_sig, 90.0))
        res[lead_name] = {"R": r_amp, "S": s_amp, "ST": st_amp, "T": t_amp}
    return res

# Extract Real Ground Truth Amplitudes
real_amps = []
for batch in test_loader:
    target = batch[0].numpy()
    for i in range(target.shape[0]):
        real_amps.append(extract_amplitudes(target[i]))

audit_results = []

for pair in model_pairs:
    mask = pair["mask"]
    loss_desc = pair["loss"]
    print(f"\n========================================================")
    print(f"Evaluating Mask {mask} ({loss_desc}): U-Net vs MS-VAE")
    print(f"========================================================")
    
    pair_report = {"mask": mask, "loss_desc": loss_desc}
    
    for arch_key in ["unet", "msvae"]:
        mspec = pair[arch_key]
        mid = mspec["id"]
        mname = mspec["name"]
        print(f"  -> Loading & Evaluating {mname} ({mid})...")
        
        try:
            adapter = load_adapter(mspec, device)
        except Exception as e:
            print(f"     Failed to load {mid}: {e}")
            pair_report[arch_key] = None
            continue
            
        recon_amps = []
        with torch.inference_mode():
            for batch in test_loader:
                target = batch[0].to(device)
                try:
                    recon = adapter.reconstruct(target).cpu().numpy()
                except Exception as e:
                    print(f"     Reconstruct failed: {e}")
                    break
                for i in range(recon.shape[0]):
                    recon_amps.append(extract_amplitudes(recon[i]))
                    
        if len(recon_amps) != len(real_amps):
            print(f"     Length mismatch for {mname}")
            pair_report[arch_key] = None
            continue
            
        # Compute Presacan Statistics
        lead_stats = {}
        for lead in ["V1", "V2", "V3", "V4", "V5", "V6"]:
            for feature in ["R", "S", "ST", "T"]:
                y_real = np.array([r[lead][feature] for r in real_amps])
                y_recon = np.array([r[lead][feature] for r in recon_amps])
                
                error = y_recon - y_real
                slope, intercept, r_val, p_val, std_err = stats.linregress(y_real, error)
                r2 = r_val ** 2
                
                var_real = np.var(y_real)
                var_recon = np.var(y_recon)
                var_ratio = var_recon / (var_real + 1e-8)
                
                bias = np.mean(error)
                sd = np.std(error)
                loa_low = bias - 1.96 * sd
                loa_high = bias + 1.96 * sd
                
                lead_stats[f"{lead}_{feature}"] = {
                    "r2": r2, "slope": slope, "var_ratio": var_ratio,
                    "bias": bias, "loa_low": loa_low, "loa_high": loa_high
                }
                
        # Inter-lead R^2(I, V3) for R-peak
        real_I_R = np.array([r["I"]["R"] for r in real_amps])
        real_V3_R = np.array([r["V3"]["R"] for r in real_amps])
        recon_I_R = np.array([r["I"]["R"] for r in recon_amps])
        recon_V3_R = np.array([r["V3"]["R"] for r in recon_amps])
        
        _, _, r_real_I_V3, _, _ = stats.linregress(real_I_R, real_V3_R)
        _, _, r_recon_I_V3, _, _ = stats.linregress(recon_I_R, recon_V3_R)
        
        # Signal correlation across unobserved leads
        missing_leads = ["III", "aVR", "aVL", "aVF", "V1", "V3", "V4", "V5", "V6"]
        sig_r_list = []
        for l in missing_leads:
            l_real = np.array([r[l]["R"] for r in real_amps])
            l_recon = np.array([r[l]["R"] for r in recon_amps])
            _, _, r_l, _, _ = stats.linregress(l_real, l_recon)
            sig_r_list.append(r_l)
        mean_sig_r = float(np.mean(sig_r_list))
        
        pair_report[arch_key] = {
            "name": mname, "id": mid,
            "interlead_R2_real": r_real_I_V3 ** 2,
            "interlead_R2_recon": r_recon_I_V3 ** 2,
            "mean_sig_r": mean_sig_r,
            "lead_stats": lead_stats
        }
        
    audit_results.append(pair_report)

# Print Detailed Cross-Architecture Comparison Table
print("\n" + "="*115)
print("=== HEAD-TO-HEAD AUDIT: U-NET vs. MS-VAE (PRESACAN BLAND-ALTMAN R^2 & VARIANCE COLLAPSE) ===")
print("="*115)

header = f"{'Mask':<8} | {'Architecture':<12} | {'Lead V3 R²':<12} | {'V3 Slope':<10} | {'V3 Var Ret%':<12} | {'V1 R²':<10} | {'V6 R²':<10} | {'V6 Var Ret%':<12} | {'Interlead R²(I,V3)':<18}"
print(header)
print("-" * 115)

for p in audit_results:
    mask = p["mask"]
    for arch_key, arch_name in [("unet", "U-Net"), ("msvae", "MS-VAE")]:
        rep = p.get(arch_key)
        if not rep:
            print(f"{mask:<8} | {arch_name:<12} | {'N/A':<12} | {'N/A':<10} | {'N/A':<12} | {'N/A':<10} | {'N/A':<10} | {'N/A':<12} | {'N/A':<18}")
            continue
        v3_st = rep["lead_stats"]["V3_R"]
        v1_st = rep["lead_stats"]["V1_R"]
        v6_st = rep["lead_stats"]["V6_R"]
        
        v3_r2 = f"{v3_st['r2']:.4f}"
        v3_slope = f"{v3_st['slope']:.4f}"
        v3_var = f"{v3_st['var_ratio']*100.0:.1f}%"
        v1_r2 = f"{v1_st['r2']:.4f}"
        v6_r2 = f"{v6_st['r2']:.4f}"
        v6_var = f"{v6_st['var_ratio']*100.0:.1f}%"
        inter_r2 = f"{rep['interlead_R2_recon']:.4f}"
        
        print(f"{mask:<8} | {arch_name:<12} | {v3_r2:<12} | {v3_slope:<10} | {v3_var:<12} | {v1_r2:<10} | {v6_r2:<10} | {v6_var:<12} | {inter_r2:<18}")
    print("-" * 115)

# Save JSON report
out_json_path = Path("results/analysis/unet_vs_msvae_bland_altman_audit.json")
out_json_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_json_path, "w") as f:
    json.dump(audit_results, f, indent=2)
print(f"\nSaved cross-architecture Bland-Altman audit to {out_json_path}!")
