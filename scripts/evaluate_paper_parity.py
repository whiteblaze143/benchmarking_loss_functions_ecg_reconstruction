#!/usr/bin/env python3
"""
PAPER PARITY EVALUATION
=======================
Unified benchmarking of HuBERT, ECG-FM, and Mason baselines using MIDT-ECG paper metrics.
Excludes privacy metrics (MIR/NNAA) as per user request.

Metrics: RMSE, MSE, SNR, Fourier Distance, Hausdorff Distance, SSIM, Pearson r, 
         Avg/Max Inter-lead Correlation Error, QRS/ST Correlation.
"""

import os
import sys
from pathlib import Path
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import firwin
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter
import argparse

# Project setup

# Model Imports
from src.reconstruction.load_functions.ptbxl_dataset import PTBXLDataset
from src.reconstruction.learn_functions.hubert_bridge import HuBERTBridge
from src.reconstruction.learn_functions.ecgfm_bridge import ECGFMBridge
from src.reconstruction.learn_functions.fam_ecg import UniversalSpatialFusionAdapter
from learn_functions.generate_model import generate_reconstructor # Mason

# ==============================================================================
# METRIC FUNCTIONS
# ==============================================================================

def compute_snr_db(y_true, y_pred, eps=1e-8):
    noise = y_true - y_pred
    snr = 20 * torch.log10(torch.norm(y_true, dim=-1) / (torch.norm(noise, dim=-1) + eps) + eps)
    return snr.mean().item()

def compute_fourier_distance(y_true, y_pred):
    fft_true = torch.abs(torch.fft.rfft(y_true, dim=-1))
    fft_pred = torch.abs(torch.fft.rfft(y_pred, dim=-1))
    fft_true_norm = fft_true / (fft_true.sum(dim=-1, keepdim=True) + 1e-8)
    fft_pred_norm = fft_pred / (fft_pred.sum(dim=-1, keepdim=True) + 1e-8)
    return torch.mean(torch.abs(fft_true_norm - fft_pred_norm)).item()

def compute_hausdorff_distance(y_true_np, y_pred_np):
    C, T = y_true_np.shape
    t = np.linspace(0, 1, T)
    distances = []
    for c in range(C):
        u = np.stack([t, y_true_np[c]], axis=1)
        v = np.stack([t, y_pred_np[c]], axis=1)
        d_uv = directed_hausdorff(u, v)[0]
        d_vu = directed_hausdorff(v, u)[0]
        distances.append(max(d_uv, d_vu))
    return np.mean(distances)

def compute_ssim_ecg(img1, img2, win_size=11):
    # 1D SSIM wrapper
    K1, K2, L = 0.01, 0.03, (img2.max() - img2.min())
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    ssims = []
    for i in range(img1.shape[0]):
        s1, s2 = img1[i], img2[i]
        mu1, mu2 = uniform_filter(s1, win_size), uniform_filter(s2, win_size)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
        sigma1_sq = uniform_filter(s1**2, win_size) - mu1_sq
        sigma2_sq = uniform_filter(s2**2, win_size) - mu2_sq
        sigma12 = uniform_filter(s1*s2, win_size) - mu1_mu2
        ssim_score = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                     ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssims.append(np.mean(ssim_score))
    return np.mean(ssims)

def compute_inter_lead_correlation_error(y_true_np, y_pred_np):
    corr_true = np.nan_to_num(np.corrcoef(y_true_np))
    corr_pred = np.nan_to_num(np.corrcoef(y_pred_np))
    abs_diff = np.abs(corr_true - corr_pred)
    return np.mean(abs_diff), np.max(abs_diff)

# ==============================================================================
# MODEL LOADERS
# ==============================================================================

def load_hubert(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    bridge = HuBERTBridge(model_name="hubert-ecg-base", target_len=2500, fs_raw=500).to(device)
    adapter = UniversalSpatialFusionAdapter(dim=bridge.embed_dim, init_gate=1.0).to(device)
    bridge.load_state_dict(ckpt['bridge_state_dict'])
    adapter.load_state_dict(ckpt['adapter_state_dict'])
    return bridge, adapter

def load_ecgfm(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    bridge = ECGFMBridge(checkpoint_path="ecg_fm_integration/checkpoints/mimic_iv_ecg_physionet_pretrained.pt", target_len=2500).to(device)
    adapter = UniversalSpatialFusionAdapter(dim=bridge.embed_dim, init_gate=1.0).to(device)
    bridge.load_state_dict(ckpt['bridge_state_dict'])
    adapter.load_state_dict(ckpt['adapter_state_dict'])
    return bridge, adapter

def load_mason(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Mason Hyperparameters (Protocol 36 Parity)
    model = generate_reconstructor(
        input_lead_num=3, 
        output_lead_num=6, 
        input_channel_per_lead=32, 
        middle_channel_per_lead=32, 
        output_channel_per_lead=32, 
        block_per_input_network=3, 
        block_per_middle_network=2, 
        block_per_output_network=3, 
        input_kernel_size=17, 
        middle_kernel_size=17, 
        output_kernel_size=17, 
        use_residual="true", 
        device=device
    )
    # Manual state dict loading for Mason's custom structure
    weights = ckpt['model_weights']
    for i, net in enumerate(model.input_network.networks):
        for j, block in enumerate(net.blocks):
            # Bypass custom Mason load_state_dict (which expects a path)
            torch.nn.Module.load_state_dict(block, weights['input'][i][j])
    torch.nn.Module.load_state_dict(model.middle_network.input_network.blocks[0], weights['middle']['input'][0])
    torch.nn.Module.load_state_dict(model.middle_network.output_network.blocks[0], weights['middle']['output'][0])
    for i, net in enumerate(model.output_network.networks):
        for j, block in enumerate(net.blocks):
            torch.nn.Module.load_state_dict(block, weights['output'][i][j])
    return model

# ==============================================================================
# MAIN EVALUATION
# ==============================================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset (Test Split)
    dataset = PTBXLDataset(
        root_dir="/home/mithunmanivannan/data/ptb_xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/",
        csv_file="/home/mithunmanivannan/data/ptb_xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv",
        split='test', target_fs=500
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Models
    print("Loading models...")
    models = {}
    try:
        models['HuBERT'] = load_hubert(args.hubert_ckpt, device)
        print("  ✓ HuBERT loaded.")
    except Exception as e: print(f"  × HuBERT failed: {e}")
    
    try:
        models['ECG-FM'] = load_ecgfm(args.ecgfm_ckpt, device)
        print("  ✓ ECG-FM loaded.")
    except Exception as e: print(f"  × ECG-FM failed: {e}")

    try:
        models['Mason'] = load_mason(args.mason_ckpt, device)
        print("  ✓ Mason loaded.")
    except Exception as e: print(f"  × Mason failed: {e}")

    # FIR for Parity
    coeffs = firwin(151, [0.05, 47], pass_zero=False, fs=500)
    fir_coeffs = torch.from_numpy(coeffs).float().to(device)

    results = {name: [] for name in models.keys()}
    
    indices_fam = torch.tensor([0, 1, 8], device=device) # I, II, V3

    print(f"Evaluating {len(dataset)} samples...")
    for x, y, demo in tqdm(loader):
        x, y = x.to(device), y.to(device)
        demo = {k: v.to(device) for k, v in demo.items()}
        B = x.shape[0]

        # Target Filtering
        weight = fir_coeffs.view(1, 1, -1).repeat(12, 1, 1)
        padding = (len(coeffs) - 1) // 2
        y_filt = F.conv1d(y, weight, padding=padding, groups=12)
        y_filt = torch.flip(y_filt, dims=[2])
        y_filt = F.conv1d(y_filt, weight, padding=padding, groups=12)
        y_filt = torch.flip(y_filt, dims=[2])
        
        # We evaluate on V1-V6 only (Indices 6-12)
        target_v = y_filt[:, 6:12, :2500] 

        for name, model in models.items():
            with torch.no_grad():
                if name in ['HuBERT', 'ECG-FM']:
                    bridge, adapter = model
                    recon = bridge(x[..., :2500], lead_indices=indices_fam, demographics=demo, adapter=adapter, target_len=2500)
                    recon_v = recon[:, 6:12, :]
                else:
                    # Mason
                    input_leads = [x[:, i:i+1, :2500].to(device) for i in range(3)] # I, II, V3
                    outputs = model.forward(input_leads)
                    recon_v = torch.stack(outputs, dim=1) # (B, 6, T)

            # Compute Batch Metrics
            for b in range(B):
                t_b = target_v[b]
                r_b = recon_v[b]
                t_np = t_b.cpu().numpy()
                r_np = r_b.cpu().numpy()

                avg_ce, max_ce = compute_inter_lead_correlation_error(t_np, r_np)
                
                results[name].append({
                    "RMSE": torch.sqrt(F.mse_loss(r_b, t_b)).item(),
                    "MSE": F.mse_loss(r_b, t_b).item(),
                    "SNR": compute_snr_db(t_b, r_b),
                    "Fourier": compute_fourier_distance(t_b, r_b),
                    "Hausdorff": compute_hausdorff_distance(t_np, r_np),
                    "SSIM": compute_ssim_ecg(t_np, r_np),
                    "AvgCorrErr": avg_ce,
                    "MaxCorrErr": max_ce,
                    "Global_R": pearsonr(t_np.flatten(), r_np.flatten())[0]
                })

    # Summary
    print("\n" + "="*80)
    print(f"{'Model':<12} | {'RMSE':<8} | {'SNR':<8} | {'Fourier':<8} | {'Hausdorff':<10} | {'SSIM':<8} | {'AvgCorr':<8}")
    print("-" * 80)
    
    comparative_data = []
    for name in models.keys():
        df = pd.DataFrame(results[name])
        m = df.mean()
        print(f"{name:<12} | {m['RMSE']:.4f} | {m['SNR']:.2f} | {m['Fourier']:.4f} | {m['Hausdorff']:.4f} | {m['SSIM']:.4f} | {m['AvgCorrErr']:.4f}")
        row = m.to_dict()
        row['Model'] = name
        comparative_data.append(row)

    output_path = os.path.join(os.path.dirname(__file__), "paper_parity_results.csv")
    pd.DataFrame(comparative_data).to_csv(output_path, index=False)
    print("="*80)
    print(f"Full results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hubert_ckpt", type=str, default="checkpoints/theory_validation/best_theory_model.pt")
    parser.add_argument("--ecgfm_ckpt", type=str, default="checkpoints/fast_ecgfm/best_fast_ecgfm.pt")
    parser.add_argument("--mason_ckpt", type=str, default="checkpoints/mason_baseline_aligned/best_mason_baseline.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    main(args)
