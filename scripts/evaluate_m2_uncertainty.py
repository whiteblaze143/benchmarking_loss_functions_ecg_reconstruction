#!/usr/bin/env python
import os
import sys
from pathlib import Path
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths, ECG_FM_ROOT
setup_import_paths(include_fairseq=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from datetime import datetime
from glob import glob
from tqdm import tqdm
import json
from train_m2_uncertainty import UncertaintyMCMAModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def load_ptbxl_test_data():
    test_dir = "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/data/ptbxl_tensors/test"
    files = sorted(glob(os.path.join(test_dir, "*.pt")), 
                   key=lambda x: int(os.path.basename(x).replace('.pt', '')))
    
    # Take subset for testing if --test
    if '--test' in sys.argv:
        files = files[:100]
        
    print(f"Loading {len(files)} test files...")
    inputs = []
    targets = []
    INPUT_LEADS = [0, 1, 5] # Wait, original script uses 0, 1, 5 but MCMAModel used [0, 1, 7]? 
    # M0 evaluate script uses 0, 1, 5. Let's stick to 0, 1, 7 which is I, II, V2 to match train_mcma_3lead
    
    for f in tqdm(files, desc="Loading test data"):
        tensor = torch.load(f, map_location='cpu') # or weights_only=True
        tensor = (tensor - tensor.mean(dim=1, keepdim=True)) / (tensor.std(dim=1, keepdim=True) + 1e-8)
        
        # pad to 5120 for UNet 5-level pooling
        pad = torch.zeros(12, 120)
        tensor_padded = torch.cat([tensor, pad], dim=1)
        
        inp = tensor_padded[[0, 1, 7], :]
        tgt = tensor
        inputs.append(inp)
        targets.append(tgt)
        
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return inputs, targets

def load_labels():
    import pandas as pd
    import ast
    ptbxl_path = "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/data/ptb_xl"
    df = pd.read_csv(os.path.join(ptbxl_path, "ptbxl_database.csv"))
    scp_df = pd.read_csv(os.path.join(ptbxl_path, "scp_statements.csv"), index_col=0)
    test_df = df[df['strat_fold'] == 10].copy()
    superclass_map = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}
    
    def get_superclass_labels(scp_codes_str):
        labels = np.zeros(5, dtype=np.float32)
        try:
            scp_codes = ast.literal_eval(scp_codes_str)
            for code, likelihood in scp_codes.items():
                if code in scp_df.index and likelihood >= 50:
                    superclass = scp_df.loc[code, 'diagnostic_class']
                    if superclass in superclass_map:
                        labels[superclass_map[superclass]] = 1.0
        except:
            pass
        return labels
        
    test_df = test_df.sort_values('ecg_id')
    if '--test' in sys.argv:
        test_df = test_df.head(100)
        
    labels = np.array([get_superclass_labels(row['scp_codes']) for _, row in test_df.iterrows()])
    return torch.tensor(labels, dtype=torch.float32)

def fine_tune_and_evaluate(signals, labels, signal_name="M2", epochs=10, batch_size=32):
    from src.ecg_fm_classifier import ECGFMClassifier
    checkpoint_path = str(ECG_FM_ROOT / "checkpoints/mimic_iv_ecg_physionet_pretrained.pt")
    model = ECGFMClassifier(checkpoint_path=checkpoint_path, num_classes=5, dropout=0.3, freeze_backbone=True).to(device)
    
    n = len(signals)
    indices = torch.randperm(n)
    n_train = int(0.8 * n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    train_ds = TensorDataset(signals[train_idx], labels[train_idx])
    val_ds = TensorDataset(signals[val_idx], labels[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auroc = 0
    if '--test' in sys.argv:
        epochs = 1
        
    for epoch in range(epochs):
        model.train()
        model.backbone.eval()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(device))
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(y.numpy())
                
        probs = np.concatenate(all_probs)
        labels_np = np.concatenate(all_labels)
        try:
            auroc = roc_auc_score(labels_np, probs, average='macro')
        except:
            auroc = 0.5
        if auroc > best_auroc:
            best_auroc = auroc
    
    return model, best_auroc

def main():
    print("=" * 70)
    print("M2 UNCERTAINTY-AWARE AUROC EVALUATION")
    print("=" * 70)
    
    m2_checkpoint = '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/checkpoints/M2_uncertainty_seed42.pt'
    if not os.path.exists(m2_checkpoint):
        print(f"Model not found at {m2_checkpoint}. Did you train it?")
        return
        
    recon_model = UncertaintyMCMAModel(in_channels=3, out_channels=12).to(device)
    recon_model.load_state_dict(torch.load(m2_checkpoint, map_location=device))
    
    inputs, targets = load_ptbxl_test_data()
    labels = load_labels()
    
    batch_size = 32
    mc_samples = 10
    
    print("\n[3/5] Generating M2 reconstructions with MC Dropout...")
    reconstructions_mean = []
    uncertainties = []
    
    for i in tqdm(range(0, len(inputs), batch_size), desc="Reconstructing (MC Dropout)"):
        batch = inputs[i:i+batch_size].to(device)
        mc_preds_mean = []
        mc_preds_logvar = []
        
        recon_model.eval()
        enable_dropout(recon_model)
        
        with torch.no_grad():
            for _ in range(mc_samples):
                out = recon_model(batch)
                mc_preds_mean.append(out[:, :12, :])
                mc_preds_logvar.append(out[:, 12:, :])
                
        mc_preds_mean = torch.stack(mc_preds_mean, dim=0) # [MC, B, 12, 5120]
        mc_preds_logvar = torch.stack(mc_preds_logvar, dim=0)
        
        # Predictive mean is the mean of MC means
        pred_mean = mc_preds_mean.mean(dim=0)
        
        # Aleatoric uncertainty: mean of predicted variances
        aleatoric_var = torch.exp(mc_preds_logvar).mean(dim=0)
        # Epistemic uncertainty: variance of predicted means
        epistemic_var = mc_preds_mean.var(dim=0)
        
        total_var = aleatoric_var + epistemic_var
        
        # Crop back to 5000 length
        pred_mean = pred_mean[:, :, :5000]
        total_var = total_var[:, :, :5000]
        
        reconstructions_mean.append(pred_mean.cpu())
        # Aggregate uncertainty to a single score per sample (e.g. mean across leads and time)
        uncertainties.append(total_var.mean(dim=[1, 2]).cpu())

    reconstructions_mean = torch.cat(reconstructions_mean)
    uncertainties = torch.cat(uncertainties)
    
    print(f"Reconstructions shape: {reconstructions_mean.shape}")
    
    mse = F.mse_loss(reconstructions_mean, targets).item()
    rho, _ = pearsonr(reconstructions_mean.numpy().flatten(), targets.numpy().flatten())
    print(f"Overall M2 MSE: {mse:.6f}, Pearson: {rho:.4f}")
    
    # Flagging mechanism: identify top 20% most uncertain segments
    threshold = np.percentile(uncertainties.numpy(), 80)
    high_fidelity_idx = np.where(uncertainties.numpy() <= threshold)[0]
    
    print(f"\nFiltering {len(high_fidelity_idx)} out of {len(inputs)} samples based on uncertainty threshold ({threshold:.4f})")
    
    filtered_recons = reconstructions_mean[high_fidelity_idx]
    filtered_targets = targets[high_fidelity_idx]
    filtered_labels = labels[high_fidelity_idx]
    
    mse_filt = F.mse_loss(filtered_recons, filtered_targets).item()
    rho_filt, _ = pearsonr(filtered_recons.numpy().flatten(), filtered_targets.numpy().flatten())
    print(f"High-fidelity M2 MSE: {mse_filt:.6f}, Pearson: {rho_filt:.4f}")
    
    print("\n[4/5] Fine-tuning ECG-FM on High-Fidelity M2 reconstructions...")
    ecgfm_model, val_auroc = fine_tune_and_evaluate(
        filtered_recons, filtered_labels, signal_name="M2_Filtered", epochs=20
    )
    
    print(f"\nFiltered Validation AUROC: {val_auroc:.4f}")
    
    os.makedirs('results', exist_ok=True)
    with open('results/m2_uncertainty_evaluation.json', 'w') as f:
        json.dump({
            'overall_mse': float(mse),
            'overall_pearson': float(rho),
            'filtered_mse': float(mse_filt),
            'filtered_pearson': float(rho_filt),
            'filtered_val_auroc': float(val_auroc)
        }, f, indent=2)

if __name__ == '__main__':
    main()
