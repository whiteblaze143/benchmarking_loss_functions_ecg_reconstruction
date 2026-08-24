#!/usr/bin/env python3
"""
MULTICLASS AUDIT: Per-Class AUROC Evaluation
Extracts granular diagnostic performance for NORM, MI, STTC, CD, HYP.
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
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset


from src.data.multi_source_dataset import MultiSourceECGDataset
from src.reconstruction.learn_functions.mason_mmd_variants import MasonMMD_V4_RationalKernel
from src.reconstruction.learn_functions.classifier import xresnet1d101

# Constants
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
N_SAMPLES = 2000

def load_reconstructor(path):
    print(f"Loading Reconstructor: {path}")
    model = MasonMMD_V4_RationalKernel(
        input_lead_num=3, output_lead_num=12,
        lambda_mse=1.0, lambda_mmd=0.01,
        lambda_deriv=0.1, lambda_corr=0.1, use_dcor=False
    )
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    
    # Handle prefixes
    new_state = {}
    for k, v in state.items():
        if k.startswith('reconstructor.'):
            new_state[k.replace('reconstructor.', '')] = v
        else:
            new_state[k] = v
            
    model_keys = model.state_dict().keys()
    final_state = {k: v for k, v in new_state.items() if k in model_keys}
    
    model.load_state_dict(final_state, strict=False)
    model.to(DEVICE).eval()
    return model

def load_oracle(path):
    print(f"Loading Oracle: {path}")
    model = xresnet1d101(num_classes=5, input_channels=12).to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model

def main():
    # 1. Dataset
    print("Loading Dataset...")
    sources = [{"name": "PTB-XL", "path": "/home/mithunmanivannan/data/ptbxl_tensors", "format": "pt"}]
    dataset = MultiSourceECGDataset(sources=sources, split='val', target_len=5000, normalization='min_max')
    
    # Subset
    # Use full test set for rigorous evaluation
    # indices = np.random.RandomState(42).choice(len(dataset), N_SAMPLES, replace=False)
    # subset = Subset(dataset, indices)
    print(f"Using full test set: {len(dataset)} samples")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    
    # 2. Collect Inputs/Labels
    print("Collecting inputs...")
    inputs = []
    labels = []
    for batch in loader:
        inputs.append(batch['input'])
        labels.append(batch['label'])
    
    x_in_gpu = torch.cat(inputs).float().to(DEVICE)
    y_true = torch.cat(labels).numpy()
    
    # 3. Models
    models = {
        'M0 (MSE)': '/home/mithunmanivannan/checkpoints/M0_seed42.pt',
        'M1-HPO': '/home/mithunmanivannan/checkpoints/M1_Final.pt',
        'M1-Corr-Only': '/home/mithunmanivannan/checkpoints/ablation_4config_3_MSE_Deriv_Corr_ep10.pt'
    }
    oracle = load_oracle('/home/mithunmanivannan/checkpoints/oracle_original.pt')
    
    results = {}
    
    # 4. Evaluate
    for name, path in models.items():
        print(f"\nEvaluating {name}...")
        recon_model = load_reconstructor(path)
        
        # Reconstruct
        x_recon_list = []
        with torch.no_grad():
            for i in range(0, len(x_in_gpu), BATCH_SIZE):
                batch = x_in_gpu[i:i+BATCH_SIZE]
                out = recon_model(batch)
                if isinstance(out, tuple): out = out[0]
                x_recon_list.append(out)
        x_recon = torch.cat(x_recon_list)
        
        # Oracle Inference
        logits_list = []
        with torch.no_grad():
            for i in range(0, len(x_recon), BATCH_SIZE):
                batch = x_recon[i:i+BATCH_SIZE]
                logits = oracle(batch)
                logits_list.append(logits)
        logits = torch.cat(logits_list).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
        
        # Calculate Per-Class AUROC
        auroc_per_class = roc_auc_score(y_true, probs, average=None)
        macro_auroc = np.mean(auroc_per_class)
        
        results[name] = {
            'Macro': macro_auroc,
            'PerClass': dict(zip(CLASSES, auroc_per_class))
        }
        
    # 5. Report
    print("\n\n=== MULTI-CLASS AUROC AUDIT ===")
    print(f"{'Class':<10} | {'M0 (MSE)':<10} | {'M1-HPO':<10} | {'Delta':<10}")
    print("-" * 46)
    
    for cls in CLASSES:
        m0_val = results['M0 (MSE)']['PerClass'][cls]
        m1_val = results['M1-HPO']['PerClass'][cls]
        diff = m1_val - m0_val
        print(f"{cls:<10} | {m0_val:.4f}     | {m1_val:.4f}     | {diff:+.4f}")
        
    print("-" * 46)
    print(f"{'MACRO':<10} | {results['M0 (MSE)']['Macro']:.4f}     | {results['M1-HPO']['Macro']:.4f}     | {results['M1-HPO']['Macro'] - results['M0 (MSE)']['Macro']:+.4f}")

    if 'M1-Corr-Only' in results:
        print("\n=== ABLATION REPORT: M1-Corr-Only (No MMD) ===")
        print(f"{'Class':<10} | {'M0 (MSE)':<10} | {'M1-Corr':<10} | {'Delta':<10}")
        print("-" * 46)
        for cls in CLASSES:
            m0_val = results['M0 (MSE)']['PerClass'][cls]
            m1_val = results['M1-Corr-Only']['PerClass'][cls]
            diff = m1_val - m0_val
            print(f"{cls:<10} | {m0_val:.4f}     | {m1_val:.4f}     | {diff:+.4f}")
        print("-" * 46)
        print(f"{'MACRO':<10} | {results['M0 (MSE)']['Macro']:.4f}     | {results['M1-Corr-Only']['Macro']:.4f}     | {results['M1-Corr-Only']['Macro'] - results['M0 (MSE)']['Macro']:+.4f}")
    
    # Save to file
    with open('results/multiclass_audit.txt', 'w') as f:
        f.write("=== MULTI-CLASS AUROC AUDIT ===\n")
        f.write(f"{'Class':<10} | {'M0 (MSE)':<10} | {'M1-HPO':<10} | {'Delta':<10}\n")
        for cls in CLASSES:
            m0_val = results['M0 (MSE)']['PerClass'][cls]
            m1_val = results['M1-HPO']['PerClass'][cls]
            diff = m1_val - m0_val
            f.write(f"{cls:<10} | {m0_val:.4f}     | {m1_val:.4f}     | {diff:+.4f}\n")

if __name__ == '__main__':
    main()
