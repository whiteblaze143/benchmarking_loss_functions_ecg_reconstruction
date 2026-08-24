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
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
from glob import glob
from tqdm import tqdm
import json
import wandb
import pandas as pd
import ast

from scripts.train_mcma_3lead import MCMAModel
from src.ecg_fm_classifier import ECGFMClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data_split(split="test"):
    data_dir = f"/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/data/ptbxl_tensors/{split}"
    files = sorted(glob(os.path.join(data_dir, "*.pt")), 
                   key=lambda x: int(os.path.basename(x).replace('.pt', '')))
    
    signals = []
    targets = []
    
    for f in tqdm(files, desc=f"Loading {split} data"):
        tensor = torch.load(f, weights_only=True)
        if tensor.shape[0] != 12:
             continue
             
        tensor = (tensor - tensor.mean(dim=1, keepdim=True)) / (tensor.std(dim=1, keepdim=True) + 1e-8)
        
        # We need the 3-lead input for the reconstructor
        # Assuming lead indices: I=0, II=1, V2=7
        input_3lead = tensor[[0, 1, 7], :]
        
        signals.append(input_3lead)
        targets.append(tensor)
    
    signals = torch.stack(signals)
    targets = torch.stack(targets)
    return signals, targets

def load_labels(split="test"):
    ptbxl_path = "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/data/ptb_xl"
    df = pd.read_csv(os.path.join(ptbxl_path, "ptbxl_database.csv"))
    scp_df = pd.read_csv(os.path.join(ptbxl_path, "scp_statements.csv"), index_col=0)
    
    if split == "test":
        target_df = df[df['strat_fold'] == 10].sort_values('ecg_id')
    else:
        target_df = df[df['strat_fold'] == 9].sort_values('ecg_id')
        
    superclass_map = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}
    
    def get_labels(scp_codes_str):
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
    
    labels = np.array([get_labels(row['scp_codes']) for _, row in target_df.iterrows()])
    return torch.tensor(labels, dtype=torch.float32)

def load_oracle_classifier():
    checkpoint_path = str(ECG_FM_ROOT / "checkpoints/mimic_iv_ecg_physionet_pretrained.pt")
    model = ECGFMClassifier(
        checkpoint_path=checkpoint_path,
        num_classes=5,
        dropout=0.3,
        freeze_backbone=True
    ).to(device)
    state = torch.load('checkpoints/Oracle_Classifier.pt', map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def load_reconstructor(ckpt_path, out_channels=12):
    model = MCMAModel(in_channels=3, out_channels=out_channels).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def evaluate_recons(reconstructor, classifier, signals, labels, is_m2=False):
    dataset = TensorDataset(signals, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x = x.to(device)
            if is_m2:
                out = reconstructor(x)
                recon = out[:, :12, :]
            else:
                recon = reconstructor(x)
            
            # Normalize reconstruction
            recon = (recon - recon.mean(dim=-1, keepdim=True)) / (recon.std(dim=-1, keepdim=True) + 1e-8)
            
            logits = classifier(recon)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())
            
    probs = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)
    
    auroc = roc_auc_score(labels_np, probs, average='macro')
    return auroc

def main():
    print("Loading Test Set...")
    test_signals, test_targets = load_data_split("test")
    test_labels = load_labels("test")
    
    print("Loading Oracle Classifier...")
    classifier = load_oracle_classifier()
    
    # M0
    if os.path.exists('checkpoints/M0_seed42.pt'):
        print("\nEvaluating M0 (MSE)...")
        m0 = load_reconstructor('checkpoints/M0_seed42.pt')
        m0_auroc = evaluate_recons(m0, classifier, test_signals, test_labels)
        print(f"M0 AUROC: {m0_auroc:.4f}")
    
    # M1
    if os.path.exists('checkpoints/M1_Pearson_seed42.pt'):
        print("\nEvaluating M1 (Pearson)...")
        m1 = load_reconstructor('checkpoints/M1_Pearson_seed42.pt')
        m1_auroc = evaluate_recons(m1, classifier, test_signals, test_labels)
        print(f"M1 AUROC: {m1_auroc:.4f}")
        
    # M2
    if os.path.exists('checkpoints/M2_uncertainty_seed42.pt'):
        print("\nEvaluating M2 (Uncertainty)...")
        m2 = load_reconstructor('checkpoints/M2_uncertainty_seed42.pt', out_channels=24)
        m2_auroc = evaluate_recons(m2, classifier, test_signals, test_labels, is_m2=True)
        print(f"M2 AUROC: {m2_auroc:.4f}")

if __name__ == '__main__':
    main()
