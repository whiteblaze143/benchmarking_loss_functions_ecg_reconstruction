#!/usr/bin/env python3
"""
Evaluate Robustness to Noise.
Addressing Review Point 3: Stress Test (Clean vs Noisy).
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
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt, resample_poly
from fractions import Fraction
import matplotlib.pyplot as plt


from src.data.multi_source_dataset import MultiSourceECGDataset
from torch.utils.data import DataLoader, Subset
from comprehensive_metrics import load_model_reconstructor

def add_noise(signal, snr_db=20, eps=1e-12):
    """
    Add White Gaussian Noise at specified SNR (AC power).
    signal: (B, C, L)
    """
    sig_ac = signal - signal.mean(dim=-1, keepdim=True)
    sig_var = sig_ac.var(dim=-1, keepdim=True, unbiased=False)

    snr_lin = 10 ** (snr_db / 10.0)
    noise_var = sig_var / (snr_lin + eps)

    noise = torch.randn_like(signal) * torch.sqrt(noise_var + eps)
    return signal + noise

def add_baseline_wander(signal, fs=500, amplitude=0.1):
    """Add low-frequency baseline wander (0.5 Hz)."""
    # signal: (B, C, L) - expected normalized [0, 1]
    # 0.1 is 10% of full dynamic range, very significant for wander.
    t = torch.linspace(0, signal.shape[-1]/fs, signal.shape[-1]).to(signal.device)
    phi = torch.rand(signal.shape[0], 1, 1).to(signal.device) * 2 * np.pi
    bw = amplitude * torch.sin(2 * np.pi * 0.5 * t + phi)
    return signal + bw

def _butter_filter(x, fs, btype, cutoff, order=4):
    if isinstance(cutoff, (list, tuple)):
        wn = [c / (fs / 2.0) for c in cutoff]
    else:
        wn = cutoff / (fs / 2.0)
    b, a = butter(order, wn, btype=btype)
    return filtfilt(b, a, x)

def build_or_load_fitbit_noise_bank(csv_path, cache_dir, fs_target=500, chunk_len=5000):
    """
    Builds (or loads cached) Fitbit-derived noise chunks.
    - csv_path: Fitbit Data.csv
    - cache_dir: where to store .npy caches
    Returns:
      noise_chunks_ac: (Nchunks, chunk_len) float32
      bw_chunks:       (Nchunks, chunk_len) float32
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ac_path = cache_dir / "fitbit_noise_ac.npy"
    bw_path = cache_dir / "fitbit_noise_bw.npy"

    if ac_path.exists() and bw_path.exists():
        noise_chunks_ac = np.load(ac_path)
        bw_chunks = np.load(bw_path)
        return noise_chunks_ac, bw_chunks

    df = pd.read_csv(csv_path)

    # Signal columns: S1..S2800 (robust sort)
    sig_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("S")]
    sig_cols = sorted(sig_cols, key=lambda x: int(x[1:]))

    if len(sig_cols) < 2800:
        raise ValueError(f"Expected ~2800 signal columns S1..S2800, found {len(sig_cols)}")

    fs_fitbit = 2800.0 / 30.0  # ~93.333 Hz

    ratio = Fraction(fs_target / fs_fitbit).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator

    noise_chunks_ac = []
    bw_chunks = []

    for _, row in df.iterrows():
        x = row[sig_cols].to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(x)):
            continue

        # Remove robust DC
        x = x - np.median(x)

        # Resample to 500 Hz
        x_rs = resample_poly(x, up=up, down=down).astype(np.float64)

        # Ensure ~15000 samples (30s * 500Hz)
        if x_rs.shape[0] < 3 * chunk_len:
            continue
        x_rs = x_rs[:3 * chunk_len]

        # Baseline wander (lowpass 0.5 Hz)
        bw = _butter_filter(x_rs, fs_target, btype="lowpass", cutoff=0.5, order=4)

        # "Clean-ish" ECG estimate (bandpass 0.5-40 Hz)
        clean = _butter_filter(x_rs, fs_target, btype="bandpass", cutoff=[0.5, 40.0], order=4)

        # Residual wearable noise
        residual = x_rs - clean

        # AC-only noise (remove remaining drift)
        residual_ac = residual - _butter_filter(residual, fs_target, btype="lowpass", cutoff=0.5, order=4)

        # Chunk into three 10s segments
        for k in range(3):
            s = k * chunk_len
            e = (k + 1) * chunk_len
            noise_chunks_ac.append(residual_ac[s:e].astype(np.float32))
            bw_chunks.append((bw[s:e] - np.mean(bw[s:e])).astype(np.float32))

    noise_chunks_ac = np.stack(noise_chunks_ac, axis=0)
    bw_chunks = np.stack(bw_chunks, axis=0)

    np.save(ac_path, noise_chunks_ac)
    np.save(bw_path, bw_chunks)
    return noise_chunks_ac, bw_chunks

def add_fitbit_noise(signal, noise_bank_ac, snr_db, device, eps=1e-12):
    """
    Inject Fitbit-derived AC noise at target SNR (AC-power scaling).
    signal: (B, C, L) torch
    noise_bank_ac: (N, L) numpy
    """
    B, C, L = signal.shape
    idx = np.random.randint(0, noise_bank_ac.shape[0], size=B)
    n = torch.from_numpy(noise_bank_ac[idx]).to(device).float()  # (B, L)
    n = n.unsqueeze(1).repeat(1, C, 1)  # (B, C, L)

    sig_ac = signal - signal.mean(dim=-1, keepdim=True)
    n_ac = n - n.mean(dim=-1, keepdim=True)

    Px = sig_ac.var(dim=-1, keepdim=True, unbiased=False)
    Pn = n_ac.var(dim=-1, keepdim=True, unbiased=False)

    snr_lin = 10 ** (snr_db / 10.0)
    alpha = torch.sqrt(Px / (snr_lin * (Pn + eps) + eps))

    return signal + alpha * n_ac

def add_fitbit_baseline_wander(signal, bw_bank, device, k=0.25, eps=1e-12):
    """
    Inject Fitbit baseline wander (drift) scaled relative to signal AC std.
    k controls severity; 0.25 is a good starting point.
    """
    B, C, L = signal.shape
    idx = np.random.randint(0, bw_bank.shape[0], size=B)
    bw = torch.from_numpy(bw_bank[idx]).to(device).float()  # (B, L)
    bw = bw.unsqueeze(1).repeat(1, C, 1)

    sig_ac = signal - signal.mean(dim=-1, keepdim=True)
    bw_ac = bw - bw.mean(dim=-1, keepdim=True)

    sig_std = torch.sqrt(sig_ac.var(dim=-1, keepdim=True, unbiased=False) + eps)
    bw_std = torch.sqrt(bw_ac.var(dim=-1, keepdim=True, unbiased=False) + eps)

    alpha = k * (sig_std / (bw_std + eps))
    return signal + alpha * bw_ac

def eval_robustness(model, loader, device, noise_type='clean', noise_source='synthetic',
                    fitbit_noise_ac=None, fitbit_bw=None):
    if noise_source == 'fitbit':
        assert fitbit_noise_ac is not None and fitbit_bw is not None, "Fitbit noise banks not loaded."
    
    model.eval()
    corrs = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch['input'].to(device)
            y = batch['target'].to(device).cpu().numpy()
            
            # Apply Degradation
            if noise_type == 'snr20':
                if noise_source == 'fitbit':
                    x = add_fitbit_noise(x, fitbit_noise_ac, snr_db=20, device=device)
                else:
                    x = add_noise(x, snr_db=20)
            elif noise_type == 'snr10':
                if noise_source == 'fitbit':
                    x = add_fitbit_noise(x, fitbit_noise_ac, snr_db=10, device=device)
                else:
                    x = add_noise(x, snr_db=10)
            elif noise_type == 'bw':
                if noise_source == 'fitbit':
                    x = add_fitbit_baseline_wander(x, fitbit_bw, device=device, k=0.25)
                else:
                    x = add_baseline_wander(x)
                
            out = model(x)
            if isinstance(out, tuple): out = out[0]
            pred = out.cpu().numpy()
            
            # Simple Global Pearson for robustness check
            for i in range(len(pred)):
                r, _ = pearsonr(y[i].flatten(), pred[i].flatten())
                corrs.append(r)
                
    return np.mean(corrs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m0', required=True)
    parser.add_argument('--m1', required=True)
    parser.add_argument('--noise_source', choices=['synthetic', 'fitbit'], default='synthetic')
    parser.add_argument('--fitbit_csv', default='/home/mithunmanivannan/results/Data.csv')
    parser.add_argument('--fitbit_cache_dir', default='results/noise_banks')
    parser.add_argument('--output', default='results/robustness_results.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Seeding for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data (Full Test Set)
    data_path = "/home/mithunmanivannan/data/ptbxl_tensors"
    sources = [{"name": "PTB-XL", "path": data_path, "format": "pt"}]
    dataset = MultiSourceECGDataset(sources=sources, split='test', target_len=5000, normalization='min_max')
    loader = DataLoader(dataset, batch_size=64, num_workers=4)
    
    # Load Models
    m0 = load_model_reconstructor(args.m0, device)
    m1 = load_model_reconstructor(args.m1, device)
    
    # Load Fitbit Bank if requested
    fitbit_noise_ac = None
    fitbit_bw = None
    if args.noise_source == 'fitbit':
        fitbit_noise_ac, fitbit_bw = build_or_load_fitbit_noise_bank(
            csv_path=args.fitbit_csv,
            cache_dir=args.fitbit_cache_dir,
            fs_target=500,
            chunk_len=5000
        )
        print(f"Loaded Fitbit noise bank: AC={fitbit_noise_ac.shape}, BW={fitbit_bw.shape}")

    conditions = ['clean', 'snr20', 'snr10', 'bw']
    results = []
    
    print(f"\n=== Robustness Evaluation (Source: {args.noise_source}) ===")
    for cond in conditions:
        print(f"Testing Condition: {cond}...")
        r0 = eval_robustness(m0, loader, device, cond, 
                             noise_source=args.noise_source,
                             fitbit_noise_ac=fitbit_noise_ac, 
                             fitbit_bw=fitbit_bw)
        r1 = eval_robustness(m1, loader, device, cond,
                             noise_source=args.noise_source,
                             fitbit_noise_ac=fitbit_noise_ac, 
                             fitbit_bw=fitbit_bw)
        results.append({'Condition': cond, 'M0_Rho': r0, 'M1_Rho': r1})
        print(f"  M0: {r0:.4f} | M1: {r1:.4f}")
        
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

if __name__ == '__main__':
    main()
