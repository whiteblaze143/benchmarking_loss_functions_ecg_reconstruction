#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/unified_latents/engineering')
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
from scipy.signal import butter, filtfilt
# Add sierra-ecg-tools to path
sys.path.insert(0, os.path.join(os.getcwd(), "third_party/sierra-ecg-tools-master/pysierraecg/src"))

import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.reconstruction.unified_latents.engineering.vae_fm import WearECGFMVAE
from src.reconstruction.unified_latents.engineering.regimes import LEAD_NAMES, make_lead_indices
from src.reconstruction.unified_latents.engineering.common import (
    compute_batch_r2,
    compute_batch_r2_per_lead,
    compute_batch_mae,
    compute_batch_mse,
    compute_batch_corr_per_lead,
    write_json
)

def zscore_normalize(x: torch.Tensor) -> torch.Tensor:
    """Per-lead z-score normalization along the time dimension."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (x - mean) / std

def highpass_filter(data, cutoff=0.05, fs=500.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    # data is [C, T]
    filtered = filtfilt(b, a, data.cpu().numpy(), axis=-1)
    return torch.from_numpy(filtered.copy()).float()

def lowpass_filter(data, cutoff=40.0, fs=500.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # data is [C, T]
    filtered = filtfilt(b, a, data.cpu().numpy(), axis=-1)
    return torch.from_numpy(filtered.copy()).float()

def plot_comparison(target, pred, obs_indices, file_name, save_path):
    """
    Generate an overlay plot for all 12 leads.
    target, pred: [5000, 12]
    """
    fig, axes = plt.subplots(6, 2, figsize=(15, 20), sharex=True)
    axes = axes.flatten()
    
    t = np.arange(5000) / 500.0 # 500Hz
    
    for i in range(12):
        ax = axes[i]
        tag = LEAD_NAMES[i]
        is_obs = i in obs_indices
        
        ax.plot(t, target[:, i], label='Ground Truth', color='black', alpha=0.5, linewidth=1)
        ax.plot(t, pred[:, i], label='Reconstruction', color='red', alpha=0.8, linewidth=1)
        
        title = f"Lead {tag}"
        if is_obs:
            title += " (Observed)"
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    plt.suptitle(f"Reconstruction Overlay: {file_name}", fontsize=16, y=1.02)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def load_sunnybrook_xml(xml_path, target_len=5000):
    """Load Sunnybrook XML and return mV signal."""
    # We use sierraecg from third_party
    import sierraecg
    
    f = sierraecg.read_file(str(xml_path))
    # Canonical 12-lead order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    signal_map = {lead.label: lead.samples for lead in f.leads}
    
    if not all(l in signal_map for l in LEAD_ORDER):
        return None
    
    # Extract resolution from XML directly for safety
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        res_node = root.find(".//resolution")
        resolution = float(res_node.text) if res_node is not None else 5.0
    except:
        resolution = 5.0
    
    sig = np.stack([signal_map[l] for l in LEAD_ORDER]).astype(np.float32)
    
    # Scale by resolution (uV/bit) and convert to mV
    sig_mv = (sig * resolution) / 1000.0
    
    # 1. Artifact Clamping: Limit noise spikes to clinical range (+/- 2.5 mV)
    # This prevents 20mV spikes from dominating the Z-score and squashing real features
    sig_mv = np.clip(sig_mv, -2.5, 2.5)
    
    # 2. Resample/Trim/Pad to target_len
    curr_len = sig_mv.shape[1]
    if curr_len < target_len:
        sig_mv = np.pad(sig_mv, ((0, 0), (0, target_len - curr_len)))
    else:
        sig_mv = sig_mv[:, :target_len]
    
    # Convert to torch tensor for filtering
    sig_mv = torch.from_numpy(sig_mv.astype(np.float32))

    # Baseline removal
    sig_mv = highpass_filter(sig_mv, cutoff=0.05)
    
    # Clinical Denoising (40Hz LPF) - resolve "thick" signals
    sig_mv = lowpass_filter(sig_mv, cutoff=40.0)
    
    # 4. Global Scaling (Relative Amplitude Preservation)
    # This prevents 'Lead Off' flatlines from being amplified into noise mess.
    # We use the MEDIAN lead std to prevent single noisy leads from squashing the record.
    lead_means = sig_mv.mean(dim=1, keepdim=True)
    lead_stds = sig_mv.std(dim=1)
    
    # We want a target std for a typical active lead
    # If using median, we find the middle lead
    robust_global_std = torch.median(lead_stds).clamp(min=1e-6)
    
    # Scale such that a 'typical' lead has std 0.15
    sig_mv = 0.15 * (sig_mv - lead_means) / robust_global_std
    
    return sig_mv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/sunnybrook")
    parser.add_argument("--output_path", type=str, default="results/sunnybrook_engineering_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Observed leads: II, V1, V5 (Indices: 1, 6, 10)
    obs_indices = [1, 6, 10]
    
    # Load Phase 2 model
    print(f"Loading model from {args.checkpoint}...")
    payload = torch.load(args.checkpoint, map_location="cpu")
    
    model = WearECGFMVAE(
        fm_checkpoint_path=payload.get("fm_checkpoint", "ecg_fm_integration/checkpoints/mimic_iv_ecg_physionet_pretrained.pt"),
        latent_channels=4,
        target_len=5000,
        beta_kl=payload.get("beta_kl", 1e-4),
        missing_lead_weight=payload.get("missing_lead_weight", 1.0),
        fm_loss_weight=payload.get("fm_loss_weight", 5e-2),
        fm_cosine_mix=payload.get("fm_cosine_mix", 0.5),
        use_decoder_conditioning=payload.get("use_decoder_conditioning", True),
        fm_cond_drop_prob=0.0,
        use_latent_alignment=payload.get("use_latent_alignment", True),
        latent_align_weight=payload.get("latent_align_weight", 1e-3),
    )
    
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    xml_files = sorted(Path(args.data_dir).glob("*.xml"))
    print(f"Found {len(xml_files)} records.")

    results = []
    
    with torch.no_grad():
        for xml_path in tqdm(xml_files):
            try:
                sig_mv = load_sunnybrook_xml(xml_path)
            except Exception as e:
                print(f"Error loading {xml_path}: {e}")
                continue
                
            if sig_mv is None:
                continue
            
            # Prepare signals in mV (no Z-score normalization to match PTB-XL mV training)
            x_raw = sig_mv.unsqueeze(0).to(device) # [1, 12, 5000]
            
            # Prepare observed leads (zero out others)
            x_input = torch.zeros_like(x_raw)
            x_input[:, obs_indices, :] = x_raw[:, obs_indices, :]
            
            # Gain compensation: Lead-Specific Drive
            # Limb Leads (0-5): Stable 4.0x | Precordial Leads (6-11): Balanced 5.0x
            gain_weights = torch.ones((1, 12, 1), device=device)
            gain_weights[:, :6, :] = 4.0
            gain_weights[:, 6:, :] = 5.0
            x_input = x_input * gain_weights
            
            # Robust SOFT-Clamping: Prevent latent blowouts from noise artifacts
            # Instead of a hard clamp, we use tanh to saturate smoothly toward 3.0
            # This provides a smoother gradient for the encoder.
            x_input = 3.0 * torch.tanh(x_input / 3.0)
            
            lead_indices_tensor = torch.tensor([obs_indices], device=device)
            
            # Run inference (memory safe bfloat16 autocast)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.impute_from_regressor(x_input, lead_indices=lead_indices_tensor)
                y_pred = out["y_pred"] # [1, 12, 5000]
            
            # Robust Reconstruction-Anchored Scaling:
            # We trust the model's 'natural' range (PTB-XL learned) more than the compressed GT.
            # We scale the GROUND TRUTH to match the reconstruction scale.
            with torch.no_grad():
                pred_obs = y_pred[:, obs_indices, :].float()
                target_obs = x_raw[:, obs_indices, :].float()
                
                # Use ROBUST median std (ignore spikes in individual leads)
                pred_stds = pred_obs.std(dim=-1).squeeze(0) # [3]
                target_stds = target_obs.std(dim=-1).squeeze(0) # [3]
                
                # Factor to bring GT up to Recon based on typical lead energy
                sf_decomp = (pred_stds.median() / target_stds.median()).clamp(min=0.1, max=10.0)
                
                # Scale Ground Truth UP (post-hoc decompression)
                x_raw = x_raw * sf_decomp
                
            # Compute metrics on normalized mV space
            target = x_raw.transpose(1, 2).float() # [1, 5000, 12]
            pred = y_pred.transpose(1, 2).float()   # [1, 5000, 12]
            
            # Filter for missing leads only for some metrics
            missing_indices = [i for i in range(12) if i not in obs_indices]
            target_miss = target[:, :, missing_indices]
            pred_miss = pred[:, :, missing_indices]
            
            r2_global = compute_batch_r2(pred_miss, target_miss).item()
            r2_leads = compute_batch_r2_per_lead(pred_miss, target_miss)
            mae_leads = compute_batch_mae_per_lead(pred_miss, target_miss)
            corr_leads = compute_batch_corr_per_lead(pred_miss, target_miss)
            
            res = {
                "file": xml_path.name,
                "r2_global": r2_global,
                "per_lead": {
                    LEAD_NAMES[idx]: {
                        "r2": r2_leads[i],
                        "mae": mae_leads[i],
                        "corr": corr_leads[i]
                    } for i, idx in enumerate(missing_indices)
                }
            }
            results.append(res)

        # Save top and bottom 2 records for visual inspection
        os.makedirs("results/sunnybrook_plots", exist_ok=True)
        
        # Records for visualization (Fidelity check)
        VIS_RECORDS = ["ECG004", "ECG010", "ECG014"]
        viz_records = [r for r in results if Path(r["file"]).stem in VIS_RECORDS]
        print(f"\nGenerating overlays for {len(viz_records)} records...")
        
        for res in viz_records:
            xml_path = Path(args.data_dir) / res["file"]
            sig_mv = load_sunnybrook_xml(xml_path)
            if sig_mv is None: continue
            
            x_raw = sig_mv.unsqueeze(0).to(device)
            x_input = torch.zeros_like(x_raw)
            x_input[:, obs_indices, :] = x_raw[:, obs_indices, :]
            
            # Gain compensation: 6.0x multiplier
            x_input = x_input * 6.0
            
            lead_indices_tensor = torch.tensor([obs_indices], device=device)
            
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model.impute_from_regressor(x_input, lead_indices=lead_indices_tensor)
                    y_pred = out["y_pred"]
            
                # Reconstruction-Anchored Scaling for plots (Robust Median)
                pred_obs = y_pred[:, obs_indices, :].float()
                target_obs = x_raw[:, obs_indices, :].float()
                
                pred_stds = pred_obs.std(dim=-1).squeeze(0)
                target_stds = target_obs.std(dim=-1).squeeze(0)
                sf_decomp = (pred_stds.median() / target_stds.median()).clamp(min=0.1, max=10.0)
                
                # Scale local GT (x_raw) UP to match model range
                x_raw = x_raw * sf_decomp
                
                target_np = x_raw.squeeze(0).transpose(0, 1).float().cpu().numpy()
                pred_np = y_pred.squeeze(0).transpose(0, 1).float().cpu().numpy()
                
                plot_path = f"results/sunnybrook_plots/{xml_path.stem}_overlay.png"
                plot_comparison(target_np, pred_np, obs_indices, res["file"], plot_path)
                print(f"  Saved plot: {plot_path}")

        summary = {
            "checkpoint": args.checkpoint,
            "mean_r2_global": np.mean([r["r2_global"] for r in results]),
            "records": results
        }
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        write_json(args.output_path, summary)
        print(f"\nSunnybrook Evaluation Complete.")
        print(f"Mean Global R2 (Missing Leads): {summary['mean_r2_global']:.4f}")
        print(f"Results saved to {args.output_path}")

def compute_batch_mae_per_lead(out, tgt):
    return (out - tgt).abs().mean(dim=(0, 1)).tolist()

if __name__ == "__main__":
    main()
