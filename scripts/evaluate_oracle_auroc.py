#!/usr/bin/env python
"""
Oracle ECG-FM AUROC Evaluation Script

Evaluates the ORIGINAL (Ground Truth) 12-lead signals on ECG-FM:
1. Loads PTB-XL validation and test sets (targets only)
2. Fine-tunes ECG-FM classification head on Validation Set (Fold 9) targets
3. Computes AUROC on Test Set (Fold 10) targets
4. Logs results to WandB under name "Oracle"

Usage:
    python scripts/evaluate_oracle_auroc.py
"""

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
from datetime import datetime
from glob import glob
from tqdm import tqdm
import json
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# =============================================================================
# Data Loading (Same as evaluate_ecgfm_auroc.py)
# =============================================================================

def load_data_split(split="test"):
    """Load PTB-XL data (Targets Only) for a specific split."""
    data_dir = f"/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/data/ptbxl_tensors/{split}"
    files = sorted(glob(os.path.join(data_dir, "*.pt")), 
                   key=lambda x: int(os.path.basename(x).replace('.pt', '')))
    
    print(f"Loading {len(files)} {split} files...")
    
    targets = []
    
    for f in tqdm(files, desc=f"Loading {split} data"):
        tensor = torch.load(f, weights_only=True)
        # Verify shape
        if tensor.shape[0] != 12:
             continue
             
        tensor = (tensor - tensor.mean(dim=1, keepdim=True)) / (tensor.std(dim=1, keepdim=True) + 1e-8)
        targets.append(tensor)
    
    targets = torch.stack(targets)
    print(f"{split.capitalize()} targets: {targets.shape}")
    return targets

def load_labels(split="test"):
    """Load PTB-XL labels for specific split."""
    import pandas as pd
    import ast
    
    ptbxl_path = "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/data/ptb_xl"
    df = pd.read_csv(os.path.join(ptbxl_path, "ptbxl_database.csv"))
    scp_df = pd.read_csv(os.path.join(ptbxl_path, "scp_statements.csv"), index_col=0)
    
    if split == "test":
        target_df = df[df['strat_fold'] == 10].sort_values('ecg_id')
    elif split == "val":
        target_df = df[df['strat_fold'] == 9].sort_values('ecg_id')
    else:
        raise ValueError("Only val (fold 9) and test (fold 10) splits supported")
    
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
    print(f"Labels ({split}): {labels.shape}, distribution: {labels.sum(axis=0)}")
    return torch.tensor(labels, dtype=torch.float32)

# =============================================================================
# ECG-FM Classifier
# =============================================================================

def load_ecgfm_classifier(num_classes=5):
    """Load ECG-FM with classification head."""
    from src.ecg_fm_classifier import ECGFMClassifier
    
    checkpoint_path = str(ECG_FM_ROOT / "checkpoints/mimic_iv_ecg_physionet_pretrained.pt")
    
    model = ECGFMClassifier(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        dropout=0.3,
        freeze_backbone=True
    )
    return model

# =============================================================================
# Fine-tuning and Evaluation (Identical logic)
# =============================================================================

def fine_tune_ecgfm(signals, labels, model_name, epochs=20, batch_size=32):
    """Fine-tune ECG-FM classification head."""
    
    print(f"\nFine-tuning ECG-FM on {model_name} signals...")
    
    torch.manual_seed(SEED)
    
    n = len(signals)
    indices = torch.randperm(n)
    n_train = int(0.8 * n)
    
    train_ds = TensorDataset(signals[indices[:n_train]], labels[indices[:n_train]])
    val_ds = TensorDataset(signals[indices[n_train:]], labels[indices[n_train:]])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    model = load_ecgfm_classifier(num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auroc = 0
    best_state = None
    
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
        
        # Validate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                probs = torch.sigmoid(model(x))
                all_probs.append(probs.cpu().numpy())
                all_labels.append(y.numpy())
        
        probs = np.concatenate(all_probs)
        labels_np = np.concatenate(all_labels)
        
        try:
            auroc = roc_auc_score(labels_np, probs, average='macro')
        except:
            auroc = 0.5
        
        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Val AUROC = {auroc:.4f} (best={best_auroc:.4f})")
            
        wandb.log({
            f'{model_name}/finetune_epoch': epoch + 1,
            f'{model_name}/val_auroc': auroc,
        })
    
    if best_state:
        model.load_state_dict(best_state)
        # Save the oracle classifier so it can be evaluated on M0/M1 reconstructions
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(best_state, 'checkpoints/Oracle_Classifier.pt')
        print("Saved Oracle Classifier to checkpoints/Oracle_Classifier.pt")
    
    print(f"Best validation AUROC: {best_auroc:.4f}")
    return model

def bootstrap_auroc(model, signals, labels, n_bootstrap=1000, batch_size=32):
    """Compute AUROC with bootstrap 95% CI."""
    
    print(f"\nComputing bootstrap CI ({n_bootstrap} samples)...")
    
    model.eval()
    dataset = TensorDataset(signals, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())
    
    probs = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)
    
    auroc_point = roc_auc_score(labels_np, probs, average='macro')
    
    np.random.seed(SEED)
    aurocs = []
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):
        idx = np.random.choice(len(probs), size=len(probs), replace=True)
        try:
            aurocs.append(roc_auc_score(labels_np[idx], probs[idx], average='macro'))
        except:
            pass
    
    aurocs = np.array(aurocs)
    ci_lower = np.percentile(aurocs, 2.5)
    ci_upper = np.percentile(aurocs, 97.5)
    
    print(f"AUROC: {auroc_point:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return {
        'auroc': float(auroc_point),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper)
    }

# =============================================================================
# Main
# =============================================================================

def main():
    model_name = "Oracle"
    
    # Initialize wandb
    wandb.init(
        entity="mxthunm-carleton-university",
        project="ECG-FM_Tuning",
        name=f"eval_{model_name}",
        config={
            'model_name': model_name,
            'eval_type': 'oracle_auroc',
            'finetune_epochs': 20,
            'bootstrap_samples': 1000,
            'seed': SEED,
        },
        mode="disabled"
    )
    
    print("=" * 70)
    print("ORACLE ECG-FM AUROC EVALUATION")
    print("=" * 70)
    
    # Step 1: Load Data
    print("\n[1/4] Loading Data...")
    print("  Loading Validation Set (Fold 9) for Fine-tuning...")
    val_signals = load_data_split("val")
    val_labels = load_labels("val")
    
    print("  Loading Test Set (Fold 10) for Evaluation...")
    test_signals = load_data_split("test")
    test_labels = load_labels("test")
    
    # Step 2: Fine-tune ECG-FM (On Validation Set)
    print(f"\n[2/4] Fine-tuning ECG-FM on {model_name} signals (Val Set)...")
    ecgfm_model = fine_tune_ecgfm(val_signals, val_labels, model_name)
    
    # Step 3: Bootstrap AUROC (On Test Set)
    print(f"\n[3/4] Computing bootstrap AUROC on HELD-OUT Test Set...")
    auroc_results = bootstrap_auroc(ecgfm_model, test_signals, test_labels)
    
    # Summary
    print("\n" + "=" * 70)
    print(f"{model_name} RESULTS")
    print("=" * 70)
    print(f"Test AUROC: {auroc_results['auroc']:.4f} [{auroc_results['ci_lower']:.4f}, {auroc_results['ci_upper']:.4f}]")
    
    # Log to WandB
    wandb.log({
        f'{model_name}/final_auroc': auroc_results['auroc'],
        f'{model_name}/auroc_ci_lower': auroc_results['ci_lower'],
        f'{model_name}/auroc_ci_upper': auroc_results['ci_upper'],
    })
    
    wandb.run.summary[f'{model_name}_auroc'] = auroc_results['auroc']
    wandb.run.summary[f'{model_name}_auroc_ci'] = f"[{auroc_results['ci_lower']:.4f}, {auroc_results['ci_upper']:.4f}]"
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'auroc': auroc_results,
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/{model_name}_ecgfm_auroc.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: results/{model_name}_ecgfm_auroc.json")
    wandb.finish()

if __name__ == '__main__':
    main()
