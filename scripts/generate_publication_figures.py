#!/usr/bin/env python3
"""
PUBLICATION-QUALITY FFT AND BLAND-ALTMAN ANALYSIS
Generates figures directly from model checkpoints (not cached predictions).
"""

import sys
from pathlib import Path
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import torch
from tqdm import tqdm

# Setup paths

from src.reconstruction.load_functions.multi_source_dataset import MultiSourceECGDataset
from src.reconstruction.learn_functions.mason_mmd_variants import MasonMMD_V4_RationalKernel
from torch.utils.data import DataLoader, Subset

# Configuration
OUTPUT_DIR = '/home/mithunmanivannan/presentation/figures'
FS = 500  # Sampling frequency
N_SAMPLES = 500  # Use 500 samples for analysis

# Publication styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})

COLORS = {
    'original': '#2E2E2E',
    'M0': '#4477AA',
    'M1': '#228833',
}


def load_model(checkpoint_path, device):
    """Load a reconstruction model from checkpoint."""
    model = MasonMMD_V4_RationalKernel(
        input_lead_num=3, output_lead_num=12,
        lambda_mse=1.0, lambda_mmd=0.01,
        lambda_deriv=0.1, lambda_corr=0.1, use_dcor=False
    )
    
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    model_keys = set(model.state_dict().keys())
    new_state = {}
    
    for k, v in state.items():
        if k in model_keys:
            new_state[k] = v
        elif k.replace('reconstructor.', '') in model_keys:
            new_state[k.replace('reconstructor.', '')] = v
    
    model.load_state_dict(new_state, strict=False)
    model.to(device).eval()
    return model


def get_predictions(model, loader, device):
    """Get predictions from model."""
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            x_in = batch['input'].to(device)
            y_true = batch['target']
            
            out = model(x_in)
            if isinstance(out, tuple):
                out = out[0]
            
            all_preds.append(out.cpu().numpy())
            all_targets.append(y_true.numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_targets)


def compute_psd(signals, lead_idx=1):
    """Compute mean PSD across samples for a given lead."""
    psd_list = []
    for i in range(signals.shape[0]):
        f, pxx = welch(signals[i, lead_idx], fs=FS, nperseg=1024)
        psd_list.append(pxx)
    return f, np.mean(psd_list, axis=0), np.std(psd_list, axis=0)


def compute_qrs_amplitude(signals, lead_idx=1):
    """Compute QRS amplitude using central window heuristic."""
    amplitudes = []
    for i in range(signals.shape[0]):
        signal = signals[i, lead_idx]
        T = len(signal)
        start, end = int(T * 0.4), int(T * 0.6)
        qrs_amp = signal[start:end].max() - signal[start:end].min()
        amplitudes.append(qrs_amp)
    return np.array(amplitudes)


def create_spectral_figure(target, pred_m0, pred_m1):
    """Create spectral fidelity comparison figure."""
    print("\nComputing spectral analysis...")
    
    f, psd_orig, _ = compute_psd(target)
    _, psd_m0, _ = compute_psd(pred_m0)
    _, psd_m1, _ = compute_psd(pred_m1)
    
    # Frequency bands
    BANDS = {'Low (0-10Hz)': (0, 10), 'QRS (10-25Hz)': (10, 25), 'High (25-50Hz)': (25, 50)}
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Panel A: PSD curves
    ax1 = axes[0]
    ax1.axvspan(0, 10, alpha=0.15, color='#FFEEAA')
    ax1.axvspan(10, 25, alpha=0.15, color='#AADDFF')
    ax1.axvspan(25, 50, alpha=0.15, color='#FFCCCC')
    
    ax1.semilogy(f, psd_orig, color=COLORS['original'], linewidth=2, label='Ground Truth')
    ax1.semilogy(f, psd_m0, color=COLORS['M0'], linewidth=1.8, linestyle='--', label='M0 (MSE-only)')
    ax1.semilogy(f, psd_m1, color=COLORS['M1'], linewidth=1.8, label='M1-HPO')
    
    ax1.set_xlim(0, 50)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density (V²/Hz)')
    ax1.set_title('(A) Spectral Fidelity', fontweight='bold')
    ax1.legend()
    ax1.grid(True, which='both', linestyle=':', alpha=0.3)
    
    # Panel B: Band power retention
    ax2 = axes[1]
    band_names = list(BANDS.keys())
    
    def band_power(psd, freqs, band):
        idx = np.where((freqs >= band[0]) & (freqs < band[1]))[0]
        return np.sum(psd[idx])
    
    power_orig = [band_power(psd_orig, f, BANDS[b]) for b in band_names]
    retention_m0 = [100 * band_power(psd_m0, f, BANDS[b]) / p for b, p in zip(band_names, power_orig)]
    retention_m1 = [100 * band_power(psd_m1, f, BANDS[b]) / p for b, p in zip(band_names, power_orig)]
    
    x = np.arange(len(band_names))
    width = 0.35
    
    bars_m0 = ax2.bar(x - width/2, retention_m0, width, label='M0 (MSE-only)', color=COLORS['M0'])
    bars_m1 = ax2.bar(x + width/2, retention_m1, width, label='M1-HPO', color=COLORS['M1'])
    
    for bar, val in zip(bars_m0, retention_m0):
        ax2.annotate(f'{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars_m1, retention_m1):
        ax2.annotate(f'{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=8)
    
    ax2.axhline(100, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([b.replace(' ', '\n') for b in band_names])
    ax2.set_ylabel('Power Retention (%)')
    ax2.set_title('(B) Band Power Retention', fontweight='bold')
    ax2.set_ylim(0, max(max(retention_m0), max(retention_m1)) * 1.2)
    ax2.legend()
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'figure_spectral_publication.{ext}'))
    plt.close()
    
    # Print metrics
    hf_att_m0 = 10 * np.log10(retention_m0[2] / 100)
    hf_att_m1 = 10 * np.log10(retention_m1[2] / 100)
    print(f"High-Freq Attenuation: M0={hf_att_m0:.2f}dB, M1={hf_att_m1:.2f}dB")
    print(f"Saved spectral figures")


def create_bland_altman_figure(target, pred_m0, pred_m1):
    """Create side-by-side Bland-Altman plots."""
    print("\nComputing Bland-Altman analysis...")
    
    amp_orig = compute_qrs_amplitude(target)
    amp_m0 = compute_qrs_amplitude(pred_m0)
    amp_m1 = compute_qrs_amplitude(pred_m1)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    
    def plot_ba(ax, ref, method, title):
        mean_vals = (ref + method) / 2
        diff_vals = method - ref
        mean_bias = np.mean(diff_vals)
        std_diff = np.std(diff_vals)
        loa_upper = mean_bias + 1.96 * std_diff
        loa_lower = mean_bias - 1.96 * std_diff
        
        ax.scatter(mean_vals, diff_vals, alpha=0.4, s=15, c='gray')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.axhline(mean_bias, color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean Bias: {mean_bias:.4f} mV')
        ax.axhline(loa_upper, color='blue', linestyle=':', linewidth=1.2,
                   label=f'+1.96 SD: {loa_upper:.4f}')
        ax.axhline(loa_lower, color='blue', linestyle=':', linewidth=1.2,
                   label=f'−1.96 SD: {loa_lower:.4f}')
        ax.fill_between([mean_vals.min(), mean_vals.max()], loa_lower, loa_upper,
                        alpha=0.1, color='blue')
        
        ax.set_xlabel('Mean of Ground Truth and Reconstruction (mV)')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.3)
        
        return mean_bias, std_diff
    
    stats_m0 = plot_ba(axes[0], amp_orig, amp_m0, '(A) M0: MSE-Only')
    stats_m1 = plot_ba(axes[1], amp_orig, amp_m1, '(B) M1-HPO: Composite')
    axes[0].set_ylabel('Difference (Recon − GT) (mV)')
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'figure_bland_altman_publication.{ext}'))
    plt.close()
    
    print(f"M0 Bias: {stats_m0[0]:.4f} mV (SD: {stats_m0[1]:.4f})")
    print(f"M1 Bias: {stats_m1[0]:.4f} mV (SD: {stats_m1[1]:.4f})")
    bias_reduction = 100 * (abs(stats_m0[0]) - abs(stats_m1[0])) / abs(stats_m0[0])
    print(f"Bias Reduction: {bias_reduction:.1f}%")
    print("Saved Bland-Altman figures")


def main():
    print("="*60)
    print("PUBLICATION FIGURE GENERATION")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load models
    print("\nLoading models...")
    m0 = load_model('/home/mithunmanivannan/checkpoints/M0_seed42.pt', device)
    m1 = load_model('/home/mithunmanivannan/checkpoints/M1_Final.pt', device)
    
    # Load data
    print("\nLoading dataset...")
    sources = [{'name': 'PTB-XL', 'path': '/home/mithunmanivannan/data/ptbxl_tensors', 'format': 'pt'}]
    dataset = MultiSourceECGDataset(sources=sources, split='test', target_len=5000, normalization='min_max')
    
    # Use deterministic subset
    np.random.seed(42)
    indices = np.random.choice(len(dataset), N_SAMPLES, replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=32, num_workers=4, shuffle=False)
    
    # Get predictions
    print("\nGenerating M0 predictions...")
    pred_m0, target = get_predictions(m0, loader, device)
    
    print("Generating M1 predictions...")
    pred_m1, _ = get_predictions(m1, loader, device)
    
    # Verification
    print(f"\nData check:")
    print(f"  Target var: {np.var(target):.6f}")
    print(f"  M0 var: {np.var(pred_m0):.6f}")
    print(f"  M1 var: {np.var(pred_m1):.6f}")
    
    # Generate figures
    create_spectral_figure(target, pred_m0, pred_m1)
    create_bland_altman_figure(target, pred_m0, pred_m1)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
