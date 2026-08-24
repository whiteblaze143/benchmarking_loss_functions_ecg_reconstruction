#!/usr/bin/env python3
"""
Low-Resource Presacan et al. (Nature Comms Med 2025) Bland-Altman R^2 & Variance Collapse Evaluator.
Computes:
1. Bland-Altman Error vs Real Amplitude Linear Regression R^2 (R-peak, S-peak, T-peak, ST in V1, V2, V3, V4, V5, V6)
2. Precordial Variance Collapse Ratio: Var(Recon) / Var(Real)
3. Inter-lead Correlation Artificial Inflation: R^2(Lead I, Lead V3) Real vs Reconstructed
Runs strictly on 1 CPU thread to avoid disturbing GPU queue.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from scipy import stats
import neurokit2 as nk

# Import checkpoint loader from codebase
sys.path.insert(0, "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
from scripts.evaluate_clinical_biomarkers_multids import (
    load_adapter, SimplePTBDataset, LEAD_NAMES, OBSERVED_LEAD_INDICES
)

device = torch.device("cpu")

ptb_data_dir = "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/data/ptb_xl/tensors/test"
ptb_csv_path = "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/data/ptb_xl/ptbxl_database.csv"

df_ptb = pd.read_csv(ptb_csv_path, index_col="ecg_id")
test_dataset = SimplePTBDataset(ptb_data_dir)
print(f"Loaded PTB-XL Test Dataset: {len(test_dataset)} records")

# Models to compare
models_to_test = [
    {"id": "f_1000000_s42", "kind": "unet", "name": "Pure MSE Baseline (1000000)"},
    {"id": "f_1000003_s42", "kind": "unet", "name": "MSE + MMD_IMQ Champion (1000003)"},
    {"id": "f_1010000_s42", "kind": "unet", "name": "MSE + Kors 3D VCG (1010000)"},
    {"id": "f_1000004_s42", "kind": "unet", "name": "MSE + MMD_Temporal (1000004)"}
]

# Evaluate on test set (500 records for fast, low-resource evaluation)
N_EVAL = min(500, len(test_dataset))
subset_indices = list(range(N_EVAL))
subset_dataset = torch.utils.data.Subset(test_dataset, subset_indices)
test_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=32, shuffle=False)

print(f"Running Presacan Bland-Altman Audit on {N_EVAL} test ECGs...")

# Helper to extract peak amplitudes from clean 12-lead signal
def extract_amplitudes(sig_12x5000):
    # sig shape: (12, 5000) at 500Hz
    # Returns dictionary of peak amplitudes for V1, V2, V3, V4, V5, V6, and Lead I
    res = {}
    for idx, lead_name in enumerate(LEAD_NAMES):
        lead_sig = sig_12x5000[idx]
        # Peak R amplitude (max in signal or 95th percentile)
        r_amp = float(np.percentile(lead_sig, 99.0))
        s_amp = float(np.percentile(lead_sig, 1.0))
        # Mean ST amplitude (samples around middle)
        st_amp = float(np.mean(lead_sig[200:350])) if len(lead_sig) >= 350 else float(np.mean(lead_sig))
        # T peak amplitude (max in positive wave)
        t_amp = float(np.percentile(lead_sig, 90.0))
        res[lead_name] = {"R": r_amp, "S": s_amp, "ST": st_amp, "T": t_amp}
    return res

# Extract Real Ground Truth Amplitudes
real_amps = []
for batch in test_loader:
    target = batch[0].numpy()
    for i in range(target.shape[0]):
        real_amps.append(extract_amplitudes(target[i]))

results_report = {}

for mspec in models_to_test:
    mid = mspec["id"]
    mname = mspec["name"]
    print(f"\nEvaluating Model: {mname}...")
    try:
        adapter = load_adapter(mspec, device)
    except Exception as e:
        print(f"Could not load {mid}: {e}")
        continue
        
    recon_amps = []
    with torch.inference_mode():
        for batch in test_loader:
            target = batch[0].to(device)
            try:
                recon = adapter.reconstruct(target).cpu().numpy()
            except Exception as e:
                print(f"Reconstruct failed: {e}")
                break
            for i in range(recon.shape[0]):
                recon_amps.append(extract_amplitudes(recon[i]))
                
    if len(recon_amps) != len(real_amps):
        print(f"Length mismatch: {len(recon_amps)} vs {len(real_amps)}")
        continue
        
    # Analyze Presacan Metrics for Leads V1, V2, V3, V4, V5, V6
    lead_stats = {}
    for lead in ["V1", "V2", "V3", "V4", "V5", "V6"]:
        for feature in ["R", "S", "ST", "T"]:
            y_real = np.array([r[lead][feature] for r in real_amps])
            y_recon = np.array([r[lead][feature] for r in recon_amps])
            
            # Reconstruction Error: Recon - Real
            error = y_recon - y_real
            
            # Linear Regression: Error ~ Real (Presacan Bland-Altman Slope & R^2)
            slope, intercept, r_val, p_val, std_err = stats.linregress(y_real, error)
            r2 = r_val ** 2
            
            # Variance Ratio: Var(Recon) / Var(Real)
            var_real = np.var(y_real)
            var_recon = np.var(y_recon)
            var_ratio = var_recon / (var_real + 1e-8)
            
            # Bland Altman Mean Bias and 95% LoA
            bias = np.mean(error)
            sd = np.std(error)
            loa_low = bias - 1.96 * sd
            loa_high = bias + 1.96 * sd
            
            lead_stats[(lead, feature)] = {
                "r2_error_vs_real": r2,
                "slope": slope,
                "var_ratio": var_ratio,
                "bias": bias,
                "loa_low": loa_low,
                "loa_high": loa_high
            }
            
    # Inter-lead correlation R^2(Lead I, Lead V3) Real vs Reconstructed
    real_I_R = np.array([r["I"]["R"] for r in real_amps])
    real_V3_R = np.array([r["V3"]["R"] for r in real_amps])
    recon_I_R = np.array([r["I"]["R"] for r in recon_amps])
    recon_V3_R = np.array([r["V3"]["R"] for r in recon_amps])
    
    _, _, r_real_I_V3, _, _ = stats.linregress(real_I_R, real_V3_R)
    _, _, r_recon_I_V3, _, _ = stats.linregress(recon_I_R, recon_V3_R)
    
    r2_interlead_real = r_real_I_V3 ** 2
    r2_interlead_recon = r_recon_I_V3 ** 2
    
    results_report[mid] = {
        "name": mname,
        "lead_stats": lead_stats,
        "interlead_R2_real": r2_interlead_real,
        "interlead_R2_recon": r2_interlead_recon
    }

print("\n" + "="*80)
print("=== PRESACAN BLAND-ALTMAN R^2 & VARIANCE COLLAPSE AUDIT RESULTS ===")
print("="*80)

for mid, rep in results_report.items():
    print(f"\n>>> {rep['name']} <<<")
    print(f"Inter-lead R^2(Lead I, Lead V3) for R-peak: Real = {rep['interlead_R2_real']:.4f} | Reconstructed = {rep['interlead_R2_recon']:.4f}")
    print(f"{'Lead':<6} | {'Feature':<8} | {'BA Error vs Real R²':<20} | {'Slope':<10} | {'Var Ratio (Recon/Real)':<22} | {'Bias (mV)':<10} | {'95% LoA':<18}")
    print("-" * 105)
    for lead in ["V1", "V3", "V6"]:
        for feat in ["R", "T", "ST"]:
            st = rep["lead_stats"][(lead, feat)]
            loa_str = f"[{st['loa_low']:.2f}, {st['loa_high']:.2f}]"
            print(f"{lead:<6} | {feat:<8} | {st['r2_error_vs_real']:<20.4f} | {st['slope']:<10.4f} | {st['var_ratio']:<22.4f} | {st['bias']:<10.4f} | {loa_str:<18}")

# Save json artifact for analysis
import json
output_json = Path("results/analysis/presacan_bland_altman_audit.json")
output_json.parent.mkdir(parents=True, exist_ok=True)

serializable = {}
for mid, rep in results_report.items():
    serializable[mid] = {
        "name": rep["name"],
        "interlead_R2_real": rep["interlead_R2_real"],
        "interlead_R2_recon": rep["interlead_R2_recon"],
        "lead_stats": {f"{k[0]}_{k[1]}": v for k, v in rep["lead_stats"].items()}
    }
with open(output_json, "w") as f:
    json.dump(serializable, f, indent=2)
print(f"\nSaved raw audit data to {output_json}!")
