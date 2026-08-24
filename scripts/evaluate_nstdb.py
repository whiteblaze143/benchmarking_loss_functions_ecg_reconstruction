#!/usr/bin/env python3
"""
Evaluate M0 and M1-HPO on MIT-BIH Noise Stress Test (NSTDB).
Standardized Robustness Evaluation.
Metric: Global Pearson Correlation degradation under "clean vs noisy" conditions.
Noise Sources: Baseline Wander ('bw'), Muscle Artifact ('ma'), Electrode Motion ('em').
SNRs: [24, 12, 6, 0] dB.
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
import argparse
import numpy as np
import torch
import wfdb
from tqdm import tqdm
from scipy.stats import pearsonr
import pandas as pd

from src.data.multi_source_dataset import MultiSourceECGDataset
from torch.utils.data import DataLoader
from comprehensive_metrics import load_model_reconstructor

def load_nstdb_noise(noise_dir, fs_target=500):
    """
    Load NSTDB noise records (bw, em, ma).
    NSTDB original fs=360Hz. We must resample to 500Hz.
    """
    noises = {}
    for name in ['bw', 'em', 'ma']:
        path = os.path.join(noise_dir, name)
        # wfdb reads physical units by default if fmt is handled
        # record.p_signal is (N, 2). Usually we use channel 0 or both.
        # We'll use channel 0 for simplicity or mix.
        try:
            rec = wfdb.rdrecord(path)
            sig = rec.p_signal[:, 0] # Take lead 0
            
            # Resample 360 -> 500
            # signal len L. New len L * 500/360
            num_samples = int(len(sig) * fs_target / 360)
            from scipy.signal import resample
            sig_500 = resample(sig, num_samples)
            
            # Normalize noise to unit variance for easier SNR scaling?
            # Or keep physical units?
            # Best practice: keep relative dynamics, but we usually scale by SNR anyway.
            # Let's standardize to mean=0, std=1 for consistent mixing.
            sig_500 = (sig_500 - np.mean(sig_500)) / (np.std(sig_500) + 1e-8)
            
            noises[name] = sig_500
            print(f"Loaded {name}: {len(sig_500)} samples (resampled 500Hz)")
            
        except Exception as e:
            print(f"Failed to load {name} from {path}: {e}")
            
    return noises

def add_noise_segment(clean_ecg, noise_full, snr_db):
    """
    clean_ecg: (12, 5000) numpy
    noise_full: (L,) numpy long recording
    """
    # Select random segment from noise
    noise_len = len(noise_full)
    sig_len = clean_ecg.shape[1]
    
    if noise_len <= sig_len:
        # Repeat/tile if too short (unlikely for NSTDB)
        noise_seg = np.resize(noise_full, sig_len)
    else:
        start = np.random.randint(0, noise_len - sig_len)
        noise_seg = noise_full[start : start+sig_len]
        
    # Broadcast noise to all leads? Or independent?
    # Real physical noise (like EM) usually correlates, but simpler to add same noise 
    # or random segments to each lead.
    # To be "standardized" typically means adding scalar-scaled noise.
    # We will add the *same* noise morphology to all leads but scaled, 
    # simulating body-wide artifact? Or independent?
    # Let's do independent segments for diversity.
    
    noisy_ecg = np.zeros_like(clean_ecg)
    
    # Calculate Signal Power per lead
    # P_signal = mean(s^2)
    # P_noise_target = P_signal / 10^(SNR/10)
    # We normalized noise_seg to std=1, so P_noise_raw = 1.
    # scale = sqrt(P_noise_target)
    
    # Input clean_ecg is (3, L) for the 3-lead model input.
    # We iterate over the actual number of channels.
    n_channels = clean_ecg.shape[0]
    
    for i in range(n_channels):
        # Fresh noise segment for each lead? Or same?
        # Let's use same type but different crop to avoid perfectly correlated noise (which is rare physically)
        start_i = np.random.randint(0, noise_len - sig_len)
        n = noise_full[start_i : start_i+sig_len]
        
        s = clean_ecg[i]
        p_sig = np.var(s)
        p_noise_target = p_sig / (10**(snr_db/10.0) + 1e-12)
        scale = np.sqrt(p_noise_target)
        
        noisy_ecg[i] = s + scale * n
        
    return noisy_ecg

def eval_nstdb(m0, m1, loader, noises, device, output_file):
    snrs = [24, 12, 6, 0] # dB
    results = []
    
    m0.eval()
    m1.eval()
    
    clean_metrics = {'M0': [], 'M1': []}
    
    # 1. Clean Baseline (First pass)
    print("Evaluating Clean Baseline...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Clean"):
            x_in = batch['input'].to(device) # (B, 3, 5000)
            target = batch['target'].numpy() # (B, 12, 5000)
            
            p0 = m0(x_in)
            if isinstance(p0, tuple): p0 = p0[0]
            p0 = p0.cpu().numpy()
            
            p1 = m1(x_in)
            if isinstance(p1, tuple): p1 = p1[0]
            p1 = p1.cpu().numpy()
            
            for i in range(len(target)):
                r0, _ = pearsonr(target[i].flatten(), p0[i].flatten())
                r1, _ = pearsonr(target[i].flatten(), p1[i].flatten())
                clean_metrics['M0'].append(r0)
                clean_metrics['M1'].append(r1)
                
    avg_clean_m0 = np.mean(clean_metrics['M0'])
    avg_clean_m1 = np.mean(clean_metrics['M1'])
    results.append({'Condition': 'Clean', 'SNR': 'Inf', 'M0_Rho': avg_clean_m0, 'M1_Rho': avg_clean_m1})
    print(f"Clean: M0={avg_clean_m0:.4f}, M1={avg_clean_m1:.4f}")
    
    # 2. Stress Test Loop
    for noise_type, noise_sig in noises.items():
        for snr in snrs:
            cond_name = f"{noise_type}_{snr}dB"
            print(f"Evaluating {cond_name}...")
            
            rhos_m0 = []
            rhos_m1 = []
            
            with torch.no_grad():
                for batch in tqdm(loader, desc=cond_name):
                    # We need to add noise to INPUT (3-lead)? Or Target?
                    # "Reconstruction from Noisy Input". So add to input.
                    # Input is 3 leads (I, II, V2). Target is 12 leads.
                    # We take Clean Input -> Add Noise -> Reconstruct.
                    # Compare to Clean Target.
                    
                    x_clean = batch['input'].numpy() # (B, 3, 5000)
                    target = batch['target'].numpy()
                    
                    x_noisy = np.zeros_like(x_clean)
                    for b in range(len(x_clean)):
                        x_noisy[b] = add_noise_segment(x_clean[b], noise_sig, snr)
                        
                    x_noisy_t = torch.from_numpy(x_noisy).float().to(device)
                    
                    p0 = m0(x_noisy_t)
                    if isinstance(p0, tuple): p0 = p0[0]
                    p0 = p0.cpu().numpy()
                    
                    p1 = m1(x_noisy_t)
                    if isinstance(p1, tuple): p1 = p1[0]
                    p1 = p1.cpu().numpy()
                    
                    for i in range(len(target)):
                        r0, _ = pearsonr(target[i].flatten(), p0[i].flatten())
                        r1, _ = pearsonr(target[i].flatten(), p1[i].flatten())
                        rhos_m0.append(r0)
                        rhos_m1.append(r1)
            
            avg_m0 = np.mean(rhos_m0)
            avg_m1 = np.mean(rhos_m1)
            results.append({'Condition': noise_type, 'SNR': snr, 'M0_Rho': avg_m0, 'M1_Rho': avg_m1})
            print(f"  M0: {avg_m0:.4f}, M1: {avg_m1:.4f}")
            
    # Save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nSaved NSTDB results to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nstdb_dir', default='/home/mithunmanivannan/data/nstdb')
    parser.add_argument('--output', default='/home/mithunmanivannan/results/nstdb_stress_test.csv')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Noise
    noises = load_nstdb_noise(args.nstdb_dir, fs_target=500)
    if not noises:
        print("No noise records found!")
        return

    # 2. Load Models
    m0 = load_model_reconstructor('/home/mithunmanivannan/checkpoints/M0_seed42.pt', device)
    m1 = load_model_reconstructor('/home/mithunmanivannan/checkpoints/M1_Final.pt', device)
    
    # 3. Load PTB-XL Test Set
    data_path = "/home/mithunmanivannan/data/ptbxl_tensors"
    sources = [{"name": "PTB-XL", "path": data_path, "format": "pt"}]
    # Use subset for speed if needed, but reviewer asked for rigorous. 
    # Full test set N=2207 is fine on GPU.
    dataset = MultiSourceECGDataset(sources=sources, split='test', target_len=5000, normalization='min_max')
    loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)
    
    # 4. Run Eval
    eval_nstdb(m0, m1, loader, noises, device, args.output)

if __name__ == '__main__':
    main()
