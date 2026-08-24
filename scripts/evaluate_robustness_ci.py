#!/usr/bin/env python3
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
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from src.data.multi_source_dataset import MultiSourceECGDataset
from comprehensive_metrics import load_model_reconstructor
from evaluate_robustness import add_fitbit_noise, add_fitbit_baseline_wander, build_or_load_fitbit_noise_bank

def bootstrap_ci(v0, v1, n_resamples=1000, alpha=0.05):
    """
    v0, v1: numpy arrays of results per record.
    Returns: mean diff, ci_low, ci_high (of the difference).
    """
    diffs = v1 - v0
    boot_means = []
    n = len(diffs)
    for _ in range(n_resamples):
        idx = np.random.randint(0, n, size=n)
        boot_means.append(np.mean(diffs[idx]))
    
    ci_low = np.percentile(boot_means, (alpha/2)*100)
    ci_high = np.percentile(boot_means, (1 - alpha/2)*100)
    return np.mean(diffs), ci_low, ci_high

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = '/home/mithunmanivannan/results'
    
    # Load Data (Full Test Set)
    data_path = "/home/mithunmanivannan/data/ptbxl_tensors"
    sources = [{"name": "PTB-XL", "path": data_path, "format": "pt"}]
    dataset = MultiSourceECGDataset(sources=sources, split='test', target_len=5000, normalization='min_max')
    loader = DataLoader(dataset, batch_size=64, num_workers=4)
    
    # Load Models
    m0 = load_model_reconstructor('/home/mithunmanivannan/checkpoints/M0_seed42.pt', device)
    m1 = load_model_reconstructor('/home/mithunmanivannan/checkpoints/M1_Final.pt', device)
    
    # Noise Banks
    fitbit_noise_ac, fitbit_bw = build_or_load_fitbit_noise_bank(
        csv_path='/home/mithunmanivannan/results/Data.csv',
        cache_dir='/home/mithunmanivannan/results/noise_banks',
        fs_target=500,
        chunk_len=5000
    )
    
    conditions = ['clean', 'snr20', 'snr10', 'bw']
    summary_results = []
    
    for cond in conditions:
        print(f"Testing Condition: {cond}...")
        corrs0, corrs1 = [], []
        
        with torch.no_grad():
            for batch in tqdm(loader):
                x_orig = batch['input'].to(device)
                y = batch['target'].to(device).cpu().numpy()
                
                # Apply degradation
                if cond == 'snr20': x = add_fitbit_noise(x_orig, fitbit_noise_ac, snr_db=20, device=device)
                elif cond == 'snr10': x = add_fitbit_noise(x_orig, fitbit_noise_ac, snr_db=10, device=device)
                elif cond == 'bw': x = add_fitbit_baseline_wander(x_orig, fitbit_bw, device=device, k=0.25)
                else: x = x_orig
                
                out0 = m0(x)
                if isinstance(out0, tuple): out0 = out0[0]
                pred0 = out0.cpu().numpy()
                
                out1 = m1(x)
                if isinstance(out1, tuple): out1 = out1[0]
                pred1 = out1.cpu().numpy()
                
                for i in range(len(pred0)):
                    r0, _ = pearsonr(y[i].flatten(), pred0[i].flatten())
                    r1, _ = pearsonr(y[i].flatten(), pred1[i].flatten())
                    corrs0.append(r0)
                    corrs1.append(r1)
        
        c0, c1 = np.array(corrs0), np.array(corrs1)
        mean_diff, ci_low, ci_high = bootstrap_ci(c0, c1)
        
        summary_results.append({
            'Condition': cond,
            'M0_Mean': np.mean(c0),
            'M1_Mean': np.mean(c1),
            'Delta': mean_diff,
            'CI_Low': ci_low,
            'CI_High': ci_high
        })
        print(f"  Gap recovered: {mean_diff:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
        
    df = pd.DataFrame(summary_results)
    df.to_csv('/home/mithunmanivannan/results/robustness_with_ci.csv', index=False)
    print("\nSaved to results/robustness_with_ci.csv")

if __name__ == '__main__':
    main()
