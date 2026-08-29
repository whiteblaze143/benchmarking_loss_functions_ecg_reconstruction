#!/usr/bin/env python3
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.train_mcma_3lead import PTBXLDataset
from unified_latents.engineering.models.fm_v2 import MCMAModel_FM_V2

def evaluate(model, loader, device, shuffle_z_fm=False):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"Evaluating (Shuffled: {shuffle_z_fm})"):
            x, y = x.to(device), y.to(device)
            # Remove padding when comparing
            out = model(x, shuffle_z_fm=shuffle_z_fm)
            # Shapes are (B, 12, 5120), trim to 5000
            out = out[:, :, :5000]
            y = y[:, :, :5000]
            
            all_preds.append(out.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate overall Pearson correlation
    # Flattening across all batch, leads, and time
    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()
    
    r, _ = pearsonr(preds_flat, targets_flat)
    return r

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/data/ptbxl_tensors"
    val_dataset = PTBXLDataset(f"{data_dir}/val")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = MCMAModel_FM_V2(in_channels=3, fm_type="csfm", use_film=True, use_residual=True)
    ckpt_path = "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/checkpoints/fm_v2_experiments/FM4_CSFM_Full/best_model.pt"
    
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"Checkpoint not found at {ckpt_path}. Using random initialization for testing.")
        
    model = model.to(device)
    
    print("\n--- Running Unshuffled Inference ---")
    r_unshuffled = evaluate(model, val_loader, device, shuffle_z_fm=False)
    
    print("\n--- Running Shuffled Inference ---")
    r_shuffled = evaluate(model, val_loader, device, shuffle_z_fm=True)
    
    print("\n=== RESULTS ===")
    print(f"Unshuffled Pearson r: {r_unshuffled:.4f}")
    print(f"Shuffled Pearson r:   {r_shuffled:.4f}")
    print(f"Delta:                {r_unshuffled - r_shuffled:.4f}")
    
    if r_unshuffled - r_shuffled > 0.005:
        print("Conclusion: Model IS using patient-specific FM information.")
    else:
        print("Conclusion: FM branch is basically functioning as a learned constant/bias.")

if __name__ == "__main__":
    main()
