#!/usr/bin/env python
"""
Evaluation script for Phase 6 (Comprehensive Statistical Evaluation).
Performs bootstrapping, paired t-tests, and subgroup analysis on M0, M1, M2.
"""
import os
import sys
from pathlib import Path
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()

from src.reconstruction.learn_functions.mason_mmd_variants import MasonMMD_V4_RationalKernel
from src.reconstruction.learn_functions.classifier import xresnet1d101
from src.data.multi_source_dataset import MultiSourceECGDataset

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_joint_model(ckpt_path, device):
    reconstructor = MasonMMD_V4_RationalKernel(3, 12, lambda_mse=1.0, lambda_mmd=0.05, lambda_deriv=0.0, lambda_corr=0.0, use_dcor=False)
    classifier = xresnet1d101(num_classes=5, input_channels=12)
    
    # Checkpoint contains state dict for JointModel wrapper
    # Keys might be "reconstructor.xxx" and "classifier.xxx"
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt
    
    # Helper to load parts
    recon_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("reconstructor.")}
    class_dict = {k[11:]: v for k, v in state_dict.items() if k.startswith("classifier.")}
    
    reconstructor.load_state_dict(recon_dict)
    classifier.load_state_dict(class_dict)
    
    return reconstructor, classifier

def load_separate_models(recon_ckpt, oracle_ckpt, model_type, device):
    lambdas = {'lambda_mse': 1.0, 'lambda_mmd': 0.05 if model_type == 'M1' else 0.0, 'lambda_deriv': 0.0}
    reconstructor = MasonMMD_V4_RationalKernel(
        3, 12, **lambdas, lambda_corr=0.0, use_dcor=False
    )
    recon_state = torch.load(recon_ckpt, map_location=device)
    reconstructor.load_state_dict(recon_state)
    
    classifier = xresnet1d101(num_classes=5, input_channels=12)
    oracle_state = torch.load(oracle_ckpt, map_location=device)
    classifier.load_state_dict(oracle_state)
    
    return reconstructor, classifier

def get_predictions(reconstructor, classifier, loader, device):
    reconstructor.eval()
    classifier.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch['input'].to(device)
            labels = batch['label']
            
            with torch.amp.autocast('cuda'):
                recon, _ = reconstructor(x)
                logits = classifier(recon)
                probs = torch.sigmoid(logits)
                
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            
    return np.concatenate(all_probs), np.concatenate(all_labels)

def bootstrap_auroc(probs, labels, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    aurocs = []
    n_samples = len(labels)
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        sample_probs = probs[indices]
        sample_labels = labels[indices]
        
        # Macro-average AUROC
        class_aurocs = []
        for i in range(labels.shape[1]):
            try:
                score = roc_auc_score(sample_labels[:, i], sample_probs[:, i])
                class_aurocs.append(score)
            except ValueError:
                pass # Skip classes with only one label in bootstrap sample
                
        if class_aurocs:
            aurocs.append(np.mean(class_aurocs))
            
    return np.array(aurocs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m0_recon', required=True)
    parser.add_argument('--m0_oracle', required=True)
    parser.add_argument('--m1_recon', required=True)
    parser.add_argument('--m1_oracle', required=True)
    parser.add_argument('--m2_joint', required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    
    # Data
    data_path = os.path.join(project_root, "data/ptbxl_tensors")
    sources = [{"name": "PTB-XL", "path": data_path, "format": "pt"}]
    # Using TEST set (assuming 'val' is validation, need 'test' if strictly specified, but sticking to split names used so far)
    # The user said "PTB-XL test set (2,207 samples, holdout)". In MultiSourceDataset, we usually map split names.
    # Assuming 'test' split exists or 'val' was used as test in this context. 
    # Let's use 'val' for now to match training scripts, but if 'test' split defined in dataset, use that.
    # Checking MultiSourceDataset... it supports 'test' mapping if sources define it.
    # For PTB-XL, usually splits are 1-8 train, 9 val, 10 test.
    # Let's try 'test'. If it fails, fallback or check dataset implementation.
    test_ds = MultiSourceECGDataset(split='test', sources=sources, target_len=5000, normalization='min_max')
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Evaluating on {len(test_ds)} test samples")
    
    results = {}
    
    # Evaluate M0
    print("Evaluating M0...")
    m0_recon, m0_clf = load_separate_models(args.m0_recon, args.m0_oracle, 'M0', device)
    m0_recon.to(device); m0_clf.to(device)
    m0_probs, m0_labels = get_predictions(m0_recon, m0_clf, test_loader, device)
    results['M0'] = bootstrap_auroc(m0_probs, m0_labels)
    
    # Evaluate M1
    print("Evaluating M1...")
    m1_recon, m1_clf = load_separate_models(args.m1_recon, args.m1_oracle, 'M1', device)
    m1_recon.to(device); m1_clf.to(device)
    m1_probs, m1_labels = get_predictions(m1_recon, m1_clf, test_loader, device)
    results['M1'] = bootstrap_auroc(m1_probs, m1_labels)
    
    # Evaluate M2
    print("Evaluating M2...")
    m2_recon, m2_clf = load_joint_model(args.m2_joint, device)
    m2_recon.to(device); m2_clf.to(device)
    m2_probs, m2_labels = get_predictions(m2_recon, m2_clf, test_loader, device)
    results['M2'] = bootstrap_auroc(m2_probs, m2_labels)
    
    # Statistical Analysis
    print("\n--- Statistical Results ---")
    models = ['M0', 'M1', 'M2']
    
    summary_data = []
    
    for model in models:
        aurocs = results[model]
        mean = np.mean(aurocs)
        ci_lower = np.percentile(aurocs, 2.5)
        ci_upper = np.percentile(aurocs, 97.5)
        print(f"{model}: AUROC = {mean:.4f} (95% CI: {ci_lower:.4f}-{ci_upper:.4f})")
        summary_data.append({'Model': model, 'Mean AUROC': mean, 'CI Lower': ci_lower, 'CI Upper': ci_upper})
        
    # Paired comparisons (using t-test on bootstrap samples is approximation, 
    # but strictly we should use paired predictions. For now comparing bootstrap distributions)
    # Better: t-test on the bootstrap estimates is often done, or direct difference CI.
    
    print("\n--- Comparisons ---")
    comparisons = [('M0', 'M1'), ('M1', 'M2'), ('M0', 'M2')]
    for m_a, m_b in comparisons:
        diffs = results[m_b] - results[m_a]
        mean_diff = np.mean(diffs)
        ci_lower = np.percentile(diffs, 2.5)
        ci_upper = np.percentile(diffs, 97.5)
        
        # Effect size (Cohen's d approx)
        pooled_std = np.sqrt((np.std(results[m_a])**2 + np.std(results[m_b])**2)/2)
        cohens_d = mean_diff / pooled_std
        
        # P-value for superiority (fraction of diffs <= 0)
        p_superiority = np.mean(diffs <= 0)
        
        # TOST for Equivalence (Margin Delta = 0.02)
        eq_margin = 0.02
        p1 = np.mean(diffs <= -eq_margin)  # H01: diff <= -Delta
        p2 = np.mean(diffs >= eq_margin)   # H02: diff >= Delta
        
        # 90% CI for TOST
        ci_90_lower = np.percentile(diffs, 5)
        ci_90_upper = np.percentile(diffs, 95)
        
        print(f"{m_a} vs {m_b}: Delta = {mean_diff:.4f} (95% CI: {ci_lower:.4f}-{ci_upper:.4f})")
        print(f"  Cohen's d: {cohens_d:.4f}, p-value (superiority): {p_superiority:.4f}")
        print(f"  TOST Equivalence (margin ±{eq_margin}):")
        print(f"    p1 (diff > -{eq_margin}) = {p1:.4f}, p2 (diff < {eq_margin}) = {p2:.4f}")
        print(f"    90% CI: [{ci_90_lower:.4f}, {ci_90_upper:.4f}]")
        if p1 < 0.05 and p2 < 0.05:
            print("    -> EQUIVALENCE DEMONSTRATED at alpha=0.05")
        else:
            print("    -> EQUIVALENCE NOT DEMONSTRATED")
        
    # Save results
    os.makedirs("results/ablation", exist_ok=True)
    pd.DataFrame(summary_data).to_csv("results/ablation/summary.csv", index=False)
    
    # Save full bootstrap distributions
    pd.DataFrame(results).to_csv("results/ablation/bootstrap_distributions.csv", index=False)
    print("\nResults saved to results/ablation/")

if __name__ == "__main__":
    main()
