#!/usr/bin/env python3
"""
GENERATE FINAL REPORT DATA (Streaming)
Computes Bootstrap CIs and Reliability Diagrams for M0, M1, and M1-Final.
Optimized to avoid OOM by streaming batches and computing sample-wise metrics.
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
import gc
import logging
import argparse
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


from src.reconstruction.learn_functions.classifier import xresnet1d101
from comprehensive_metrics import load_model_reconstructor

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def bootstrap_ci_mean(scores, n_boot=1000):
    """
    Computes 95% CI for the mean of a score vector.
    """
    scores = np.array(scores)
    stats = []
    n = len(scores)
    rng = np.random.RandomState(42)
    
    mean_val = np.mean(scores)
    
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        stats.append(np.mean(scores[idx]))
        
    lower = np.percentile(stats, 2.5)
    upper = np.percentile(stats, 97.5)
    return mean_val, lower, upper

def bootstrap_auroc(y_true, y_probs, n_boot=1000):
    try:
        score = roc_auc_score(y_true, y_probs, average='macro')
    except:
        return np.nan, np.nan, np.nan
        
    stats = []
    n = len(y_true)
    indices = np.arange(n)
    rng = np.random.RandomState(42)
    
    for _ in range(n_boot): # Reduce boots for speed if needed, but 1000 is standard
        idx = rng.choice(indices, n, replace=True)
        try:
            val = roc_auc_score(y_true[idx], y_probs[idx], average='macro')
            stats.append(val)
        except:
            pass
            
    lower = np.percentile(stats, 2.5)
    upper = np.percentile(stats, 97.5)
    return score, lower, upper

class OracleEvaluator:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.model = xresnet1d101(num_classes=5, input_channels=12).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def get_probs_batch(self, x_batch):
        x_in = torch.from_numpy(x_batch).float().to(self.device)
        with torch.no_grad():
            logits = self.model(x_in)
        probs = torch.sigmoid(logits).cpu().numpy()
        return probs

def compute_sample_pearson(y_true, y_pred):
    # y_true: (B, 12, 5000)
    # y_pred: (B, 12, 5000)
    # Compute per-sample pearson (flatten leads)
    pearsons = []
    for i in range(len(y_true)):
        p, _ = pearsonr(y_true[i].flatten(), y_pred[i].flatten())
        pearsons.append(p)
    return np.array(pearsons)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m0', default='/home/mithunmanivannan/checkpoints/M0_seed42.pt')
    parser.add_argument('--m1-def', default='/home/mithunmanivannan/checkpoints/M1_seed42.pt')
    parser.add_argument('--m1-final', default='/home/mithunmanivannan/checkpoints/M1_Final.pt')
    parser.add_argument('--oracle', default='/home/mithunmanivannan/checkpoints/oracle_original.pt')
    parser.add_argument('--output-dir', default='/home/mithunmanivannan/results/final_report')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Models
    logger.info("Loading Models...")
    models = {}
    for name, path in [('M0', args.m0), ('M1-Def', args.m1_def), ('M1-HPO', args.m1_final)]:
        m = load_model_reconstructor(path, device)
        if m: models[name] = m
    
    oracle = OracleEvaluator(args.oracle, device)
    
    # Data Loader
    from src.data.multi_source_dataset import MultiSourceECGDataset
    from torch.utils.data import DataLoader
    
    data_path = "/home/mithunmanivannan/data/ptbxl_tensors"
    sources = [{"name": "PTB-XL", "path": data_path, "format": "pt"}]
    dataset = MultiSourceECGDataset(sources=sources, split='test', target_len=5000, normalization='min_max')
    loader = DataLoader(dataset, batch_size=32, num_workers=2, shuffle=False) # 2 workers ok if not storing
    
    # Accumulators
    results_store = {
        name: {'pearsons': [], 'probs': [], 'labels': []} 
        for name in models.keys()
    }
    
    logger.info("Starting Streaming Evaluation...")
    
    for batch in tqdm(loader):
        x_in = batch['input'].to(device) # (B, 3, 5000)
        target = batch['target'].numpy() # (B, 12, 5000) - Keep CPU for Pearson
        labels = batch['label'].numpy()  # (B, 5)
        
        for name, model in models.items():
            # Reconstruct
            with torch.no_grad():
                out = model(x_in)
                if isinstance(out, tuple): out = out[0]
                rec = out.cpu().numpy()
                
            # Pearson (Sample-wise)
            p_scores = compute_sample_pearson(target, rec)
            results_store[name]['pearsons'].extend(p_scores)
            
            # Oracle
            probs = oracle.get_probs_batch(rec)
            results_store[name]['probs'].extend(probs)
            results_store[name]['labels'].extend(labels)
            
            del out, rec
            
    # Compute Final Metrics with CIs
    final_rows = []
    
    for name, data in results_store.items():
        logger.info(f"Bootstrapping {name}...")
        
        # Pearson
        p_est, p_low, p_high = bootstrap_ci_mean(data['pearsons'], n_boot=1000)
        
        # AUROC
        all_probs = np.array(data['probs'])
        all_labels = np.array(data['labels'])
        auc_est, auc_low, auc_high = bootstrap_auroc(all_labels, all_probs, n_boot=1000)
        
        # Brier
        briers = []
        for i in range(5):
            briers.append(brier_score_loss(all_labels[:, i], all_probs[:, i]))
        brier = np.mean(briers)
        
        final_rows.append({
            'Model': name,
            'Pearson_Mean': p_est, 'Pearson_CI_Low': p_low, 'Pearson_CI_High': p_high,
            'AUROC_Mean': auc_est, 'AUROC_CI_Low': auc_low, 'AUROC_CI_High': auc_high,
            'Brier': brier
        })
        
        # Calibration Plot
        plt.figure()
        prob_flat = all_probs.flatten() 
        label_flat = all_labels.flatten()
        frac_pos, mean_pred_val = calibration_curve(label_flat, prob_flat, n_bins=10)
        plt.plot(mean_pred_val, frac_pos, "s-", label=f"{name} (Brier={brier:.3f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("Mean Predicted Value")
        plt.ylabel("Fraction of Positives")
        plt.title(f"Reliability Diagram: {name}")
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, f"calibration_{name}.png"))
        plt.close()

    df = pd.DataFrame(final_rows)
    print("\nFINAL STREAMING REPORT:\n", df.to_string())
    df.to_csv(os.path.join(args.output_dir, 'final_metrics_ci.csv'), index=False)

if __name__ == '__main__':
    main()
