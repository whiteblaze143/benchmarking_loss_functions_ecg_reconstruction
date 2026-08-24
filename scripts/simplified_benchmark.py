#!/usr/bin/env python3
"""
Simplified Benchmark Extraction Script - 12 Reliable Metrics Only
Extracts ONLY metrics we can defend with confidence.

NO fake demographics, NO broken QRS/ST metrics, NO dubious SSIM.

Output:
    - results/benchmark/benchmark_results.json
    - results/benchmark/benchmark_table.tex
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
import torch.nn as nn
import numpy as np
import json
import re
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, brier_score_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.reconstruction.learn_functions.mason_mmd_variants import MasonMMD_V4_RationalKernel
from src.reconstruction.learn_functions.classifier import xresnet1d101
from src.data.multi_source_dataset import MultiSourceECGDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Configuration ===
CHECKPOINTS = {
    'M0': '/home/mithunmanivannan/checkpoints/M0_seed42.pt',
    'M1': '/home/mithunmanivannan/checkpoints/M1_seed42.pt',
    'M2-Sup-Only': '/home/mithunmanivannan/checkpoints/M2_sup_only.pt',
    'M2-Fixed': '/home/mithunmanivannan/checkpoints/M2_joint_seed1002.pt',
    'M2-Ramped': '/home/mithunmanivannan/checkpoints/M2_main_seed42.pt',
}

TRAINING_LOGS = {
    'M0': '/home/mithunmanivannan/logs/M0_seed42_training.log',
    'M1': '/home/mithunmanivannan/logs/M1_seed42_training.log',
    'M2-Sup-Only': '/home/mithunmanivannan/logs/gap_filling/3_train_m2_sup_only.log',
    'M2-Fixed': '/home/mithunmanivannan/logs/gap_filling/4_train_m2_fixed.log',
    'M2-Ramped': '/home/mithunmanivannan/logs/M2_main_training.log',
}

ORACLE_PATH = '/home/mithunmanivannan/checkpoints/oracle_original.pt'
OUTPUT_DIR = Path('/home/mithunmanivannan/results/benchmark')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def extract_epoch_aucs(log_file):
    """Parse validation AUROC from training log."""
    if not os.path.exists(log_file):
        return None
    
    epoch_aucs = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                match = re.search(r'Val.*AUROC[=:\s]+(\d+\.\d+)', line, re.IGNORECASE)
                if match:
                    epoch_aucs.append(float(match.group(1)))
    except:
        return None
    
    return np.array(epoch_aucs) if epoch_aucs else None


def load_model(checkpoint_path):
    """Load reconstruction model."""
    if not os.path.exists(checkpoint_path):
        return None
    
    model = MasonMMD_V4_RationalKernel(
        input_lead_num=3, output_lead_num=12,
        lambda_mse=1.0, lambda_mmd=0.05,
        lambda_deriv=0.1, lambda_corr=0.1, use_dcor=False
    )
    
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle JointModel checkpoints
    if any(k.startswith('reconstructor.') for k in state.keys()):
        new_state = {k.replace('reconstructor.', ''): v 
                    for k, v in state.items() if k.startswith('reconstructor.')}
        model.load_state_dict(new_state)
    else:
        model.load_state_dict(state)
    
    return model.to(device).eval()


def extract_metrics(model_name, checkpoint_path, test_loader, oracle):
    """Extract 12 reliable metrics incrementally (low memory footprint)."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(checkpoint_path):
        print(f"  ⚠ Checkpoint not found: {checkpoint_path}")
        return None
    
    model = load_model(checkpoint_path)
    if model is None:
        return None
    
    # Accumulators for metrics
    mse_sum = 0.0
    pearson_global_sum = 0.0
    pearson_per_lead_sum = np.zeros(12)
    n_batches = 0
    total_samples = 0
    
    # Store only preds/labels for AUROC (small memory footprint)
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            x_3lead = batch['input'].to(device)
            x_12lead = batch['target'].to(device)
            labels = batch['label'].to(device)
            
            output = model(x_3lead)
            recon = output[0] if isinstance(output, tuple) else output
            
            # 1. Oracle Predictions
            logits = oracle(recon)
            probs = torch.sigmoid(logits)
            
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # 2. Signal Metrics (Compute on GPU/Optimization to avoid storing signals)
            # Global MSE
            diff = x_12lead - recon
            batch_mse = torch.mean(diff ** 2).item()
            mse_sum += batch_mse
            
            # Global Pearson (approximate by averaging batch Pearsons - faster & lower memory)
            # Flattening huge arrays is bad, so we do it per sample or per batch?
            # Per-sample is slow. Per-batch flatten is reasonable.
            x_flat = x_12lead.view(x_12lead.size(0), -1)
            r_flat = recon.view(recon.size(0), -1)
            
            # Simple batch-wise manual pearson correlation
            vx = x_flat - torch.mean(x_flat, dim=1, keepdim=True)
            vr = r_flat - torch.mean(r_flat, dim=1, keepdim=True)
            denom = torch.sqrt(torch.sum(vx ** 2, dim=1)) * torch.sqrt(torch.sum(vr ** 2, dim=1))
            rho = torch.sum(vx * vr, dim=1) / (denom + 1e-8)
            pearson_global_sum += rho.sum().item()
            
            # Per-lead Pearson
            for i in range(12):
                l_x = x_12lead[:, i, :]
                l_r = recon[:, i, :]
                vx_l = l_x - torch.mean(l_x, dim=1, keepdim=True)
                vr_l = l_r - torch.mean(l_r, dim=1, keepdim=True)
                denom_l = torch.sqrt(torch.sum(vx_l ** 2, dim=1)) * torch.sqrt(torch.sum(vr_l ** 2, dim=1))
                rho_l = torch.sum(vx_l * vr_l, dim=1) / (denom_l + 1e-8)
                pearson_per_lead_sum[i] += rho_l.sum().item()

            n_batches += 1
            total_samples += x_12lead.size(0)

    # Compile Results
    results = {'model': model_name}
    
    # A. Signal Fidelity
    results['rmse_global'] = float(np.sqrt(mse_sum / n_batches))
    results['pearson_global'] = float(pearson_global_sum / total_samples)
    
    results['pearson_per_lead'] = {}
    for i, lead_name in enumerate(LEAD_NAMES):
        results['pearson_per_lead'][lead_name] = float(pearson_per_lead_sum[i] / total_samples)
        
    pearson_vals = list(results['pearson_per_lead'].values())
    results['pearson_min'] = float(min(pearson_vals))
    results['pearson_max'] = float(max(pearson_vals))
    results['worst_lead'] = LEAD_NAMES[np.argmin(pearson_vals)]
    
    print(f"    ✓ Pearson ρ: {results['pearson_global']:.4f} (range: {results['pearson_min']:.4f}–{results['pearson_max']:.4f})")
    print(f"    ✓ RMSE: {results['rmse_global']:.6f}")
    
    # B. Diagnostic Utility
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    aurocs = []
    for c in range(preds.shape[1]):
        if len(np.unique(labels[:, c])) > 1:
            aurocs.append(roc_auc_score(labels[:, c], preds[:, c]))
            
    results['auroc_macro'] = float(np.mean(aurocs))
    results['auroc_min'] = float(min(aurocs))
    results['auroc_max'] = float(max(aurocs))
    print(f"    ✓ AUROC (macro): {results['auroc_macro']:.4f}")
    
    # C. Calibration
    # Brier Score
    brier_scores = []
    for c in range(preds.shape[1]):
        brier_scores.append(brier_score_loss(labels[:, c], preds[:, c]))
    results['brier_score'] = float(np.mean(brier_scores))
    
    # ECE (Simplified)
    n_bins = 10
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (preds_flat >= bin_boundaries[i]) & (preds_flat < bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_conf = preds_flat[in_bin].mean()
            avg_acc = labels_flat[in_bin].mean()
            ece += prop_in_bin * abs(avg_conf - avg_acc)
    results['ece'] = float(ece)
    print(f"    ✓ ECE: {results['ece']:.4f}")
    
    # === D. Stability (from training logs) ===
    print("  Extracting stability metrics...")
    
    log_file = TRAINING_LOGS.get(model_name)
    if log_file:
        epoch_aucs = extract_epoch_aucs(log_file)
        if epoch_aucs is not None and len(epoch_aucs) > 1:
            results['epoch_auroc_std'] = float(np.std(epoch_aucs))
            results['epoch_auroc_range'] = [float(np.min(epoch_aucs)), float(np.max(epoch_aucs))]
            results['best_vs_final_gap'] = float(np.max(epoch_aucs) - epoch_aucs[-1])
            print(f"    ✓ Epoch AUROC Std: {results['epoch_auroc_std']:.4f}")
            print(f"    ✓ Epoch AUROC Range: {results['epoch_auroc_range'][0]:.4f}–{results['epoch_auroc_range'][1]:.4f}")
        else:
            print(f"    ⚠ Could not parse log: {log_file}")
    
    return results


def generate_latex_table(all_results):
    """Generate LaTeX benchmark table."""
    models = ['M0', 'M1', 'M2-Sup-Only', 'M2-Fixed', 'M2-Ramped']
    
    def get_val(model, key, fmt=".4f"):
        if model in all_results and key in all_results[model]:
            val = all_results[model][key]
            return f"{val:{fmt}}"
        return "[TBD]"
    
    latex = r"""\begin{table*}[t]
\centering
\caption{Comprehensive Benchmark: 12 Reliable Metrics Across 5 ECG Reconstruction Objectives. M0 (MSE baseline) is recommended as the reference.}
\label{tab:comprehensive_benchmark}
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Metric} & \textbf{M0 (MSE)} & \textbf{M1 (MMD)} & \textbf{M2-Sup-Only} & \textbf{M2-Fixed} & \textbf{M2-Ramped} \\
\hline
\multicolumn{6}{|l|}{\textit{A. Signal Fidelity}} \\
\hline
"""
    
    # Signal Fidelity rows
    row = "Global Pearson $\\rho$"
    for m in models:
        row += f" & {get_val(m, 'pearson_global')}"
    latex += row + " \\\\\n"
    
    row = "Global RMSE"
    for m in models:
        row += f" & {get_val(m, 'rmse_global', '.6f')}"
    latex += row + " \\\\\n"
    
    row = "Per-Lead $\\rho$ (min--max)"
    for m in models:
        if m in all_results and 'pearson_min' in all_results[m]:
            row += f" & {all_results[m]['pearson_min']:.4f}--{all_results[m]['pearson_max']:.4f}"
        else:
            row += " & [TBD]"
    latex += row + " \\\\\n"
    
    row = "Worst Lead ($\\rho$)"
    for m in models:
        if m in all_results and 'worst_lead' in all_results[m]:
            row += f" & {all_results[m]['worst_lead']} ({all_results[m]['pearson_min']:.4f})"
        else:
            row += " & [TBD]"
    latex += row + " \\\\\n"
    
    latex += r"""\hline
\multicolumn{6}{|l|}{\textit{B. Diagnostic Utility (Frozen Oracle)}} \\
\hline
"""
    
    row = "Macro AUROC"
    for m in models:
        row += f" & {get_val(m, 'auroc_macro')}"
    latex += row + " \\\\\n"
    
    row = "AUROC (min--max)"
    for m in models:
        if m in all_results and 'auroc_min' in all_results[m]:
            row += f" & {all_results[m]['auroc_min']:.4f}--{all_results[m]['auroc_max']:.4f}"
        else:
            row += " & [TBD]"
    latex += row + " \\\\\n"
    
    latex += r"""\hline
\multicolumn{6}{|l|}{\textit{C. Calibration}} \\
\hline
"""
    
    row = "ECE"
    for m in models:
        row += f" & {get_val(m, 'ece')}"
    latex += row + " \\\\\n"
    
    row = "Brier Score"
    for m in models:
        row += f" & {get_val(m, 'brier_score')}"
    latex += row + " \\\\\n"
    
    latex += r"""\hline
\multicolumn{6}{|l|}{\textit{D. Stability (Epoch-to-Epoch)}} \\
\hline
"""
    
    row = "AUROC Std Dev"
    for m in models:
        row += f" & {get_val(m, 'epoch_auroc_std')}"
    latex += row + " \\\\\n"
    
    row = "AUROC Range"
    for m in models:
        if m in all_results and 'epoch_auroc_range' in all_results[m]:
            rng = all_results[m]['epoch_auroc_range']
            row += f" & {rng[0]:.4f}--{rng[1]:.4f}"
        else:
            row += " & [TBD]"
    latex += row + " \\\\\n"
    
    row = "Best--Final Gap"
    for m in models:
        row += f" & {get_val(m, 'best_vs_final_gap')}"
    latex += row + " \\\\\n"
    
    latex += r"""\hline
\end{tabular}
\end{table*}
"""
    return latex


def main():
    print("=" * 70)
    print("SIMPLIFIED BENCHMARK: 12 Reliable Metrics × 5 Models")
    print("NO fake data, NO broken metrics, ONLY defensible results")
    print("=" * 70)
    
    # Load data
    data_path = "/home/mithunmanivannan/data/ptbxl_tensors"
    sources = [{"name": "PTB-XL", "path": data_path, "format": "pt"}]
    
    print("\nLoading test dataset...")
    test_dataset = MultiSourceECGDataset(
        sources=sources, split='test', target_len=5000, normalization='min_max'
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=1)
    print(f"Test samples: {len(test_dataset)}")
    
    # Load oracle
    print("\nLoading Original-ECG Oracle...")
    oracle = xresnet1d101(num_classes=5, input_channels=12).to(device)
    oracle.load_state_dict(torch.load(ORACLE_PATH, map_location=device, weights_only=False))
    oracle.eval()
    
    # Extract metrics for each model
    all_results = {}
    for model_name, checkpoint_path in CHECKPOINTS.items():
        results = extract_metrics(model_name, checkpoint_path, test_loader, oracle)
        if results:
            all_results[model_name] = results
    
    # Save JSON
    json_output = OUTPUT_DIR / 'benchmark_results.json'
    with open(json_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ JSON saved: {json_output}")
    
    # Generate LaTeX table
    latex_table = generate_latex_table(all_results)
    tex_output = OUTPUT_DIR / 'benchmark_table.tex'
    with open(tex_output, 'w') as f:
        f.write(latex_table)
    print(f"✓ LaTeX saved: {tex_output}")
    
    # Summary
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"{'Model':<15} {'Pearson ρ':<12} {'RMSE':<12} {'AUROC':<10} {'ECE':<10} {'Brier':<10}")
    print("-" * 90)
    for model, results in all_results.items():
        print(f"{model:<15} "
              f"{results.get('pearson_global', 0):<12.4f} "
              f"{results.get('rmse_global', 0):<12.6f} "
              f"{results.get('auroc_macro', 0):<10.4f} "
              f"{results.get('ece', 0):<10.4f} "
              f"{results.get('brier_score', 0):<10.4f}")
    
    print("\n✓ Benchmark extraction complete!")


if __name__ == '__main__':
    main()
