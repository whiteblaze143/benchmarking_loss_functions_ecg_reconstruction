import argparse
import sys
sys.path.insert(0, '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/unified_latents/engineering')
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
from tqdm import tqdm
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from src.models.angle_d.resnet_baseline import ResNetBaseline
from src.scripts.angle_d.train_resnet import ECGDataset

def compute_ssim_1d(y_true, y_pred, window_size=500, stride=100):
    """
    Compute SSIM for 1D ECG signal using sliding windows.
    y_true, y_pred: (length,) numpy arrays
    """
    # Range of signal
    data_range = y_true.max() - y_true.min()
    if data_range == 0:
        data_range = 1.0 # Avoid div by zero
        
    ssim_scores = []
    # If signal shorter than window, compute on full
    if len(y_true) < window_size:
        return ssim(y_true, y_pred, data_range=data_range)
        
    for i in range(0, len(y_true) - window_size + 1, stride):
        win_true = y_true[i:i+window_size]
        win_pred = y_pred[i:i+window_size]
        s = ssim(win_true, win_pred, data_range=data_range)
        ssim_scores.append(s)
        
    return np.mean(ssim_scores)

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    test_dataset = ECGDataset(args.test_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load Model
    model = ResNetBaseline().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # Containers
    metrics_per_lead = []
    
    # Standard lead names
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    print("Evaluating...")
    with torch.no_grad():
        for x, y_true_batch in tqdm(test_loader):
            x = x.to(device)
            y_true_batch = y_true_batch.cpu().numpy()
            
            y_pred_batch = model(x).cpu().numpy()
            
            # Loop over batch
            for i in range(x.size(0)):
                y_true = y_true_batch[i] # (12, 5000)
                y_pred = y_pred_batch[i]
                
                # Compute metrics per lead
                for lead_idx in range(12):
                    lead_true = y_true[lead_idx]
                    lead_pred = y_pred[lead_idx]
                    
                    # 1. SSIM
                    val_ssim = compute_ssim_1d(lead_true, lead_pred)
                    
                    # 2. Correlation
                    val_corr, _ = pearsonr(lead_true, lead_pred)
                    
                    # 3. RMSE
                    val_rmse = np.sqrt(np.mean((lead_true - lead_pred)**2))
                    
                    metrics_per_lead.append({
                        'lead': lead_names[lead_idx],
                        'lead_idx': lead_idx,
                        'ssim': val_ssim,
                        'corr': val_corr,
                        'rmse': val_rmse
                    })
    
    # Aggregate
    df = pd.DataFrame(metrics_per_lead)
    
    # Global Mean
    print("\nGlobal Metrics:")
    print(df[['ssim', 'corr', 'rmse']].mean())
    
    # Per-Lead Mean
    print("\nPer-Lead Metrics:")
    print(df.groupby('lead')[['ssim', 'corr', 'rmse']].mean())
    
    # Save
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved results to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="results/resnet_metrics.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    evaluate(args)
