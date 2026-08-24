#!/usr/bin/env python
"""
Evaluate M0 AUROC on ECG-FM

This script:
1. Loads M0 reconstruction model
2. Generates reconstructions on PTB-XL test set
3. Fine-tunes ECG-FM classification head
4. Computes AUROC with bootstrap CI

Usage:
    python scripts/evaluate_m0_auroc.py
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

# Add all necessary paths at the very start

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from datetime import datetime
from glob import glob
from tqdm import tqdm
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# U-Net Reconstruction Model
# ============================================================================

class UNetReconstructor(nn.Module):
    """U-Net for 3-lead to 12-lead ECG reconstruction."""
    
    def __init__(self, in_channels=3, out_channels=12, base_filters=64):
        super().__init__()
        
        self.enc1 = self._conv_block(in_channels, base_filters)
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)
        self.enc4 = self._conv_block(base_filters * 4, base_filters * 8)
        
        self.bottleneck = self._conv_block(base_filters * 8, base_filters * 16)
        
        self.upconv4 = nn.ConvTranspose1d(base_filters * 16, base_filters * 8, 4, stride=2, padding=1)
        self.dec4 = self._conv_block(base_filters * 16, base_filters * 8)
        
        self.upconv3 = nn.ConvTranspose1d(base_filters * 8, base_filters * 4, 4, stride=2, padding=1)
        self.dec3 = self._conv_block(base_filters * 8, base_filters * 4)
        
        self.upconv2 = nn.ConvTranspose1d(base_filters * 4, base_filters * 2, 4, stride=2, padding=1)
        self.dec2 = self._conv_block(base_filters * 4, base_filters * 2)
        
        self.upconv1 = nn.ConvTranspose1d(base_filters * 2, base_filters, 4, stride=2, padding=1)
        self.dec1 = self._conv_block(base_filters * 2, base_filters)
        
        self.out_conv = nn.Conv1d(base_filters, out_channels, 1)
        self.pool = nn.MaxPool1d(2)
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.upconv4(b)
        d4 = self._match_size(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.upconv3(d4)
        d3 = self._match_size(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.upconv2(d3)
        d2 = self._match_size(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.upconv1(d2)
        d1 = self._match_size(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        out = self.out_conv(d1)
        
        if out.shape[2] != x.shape[2]:
            out = F.interpolate(out, size=x.shape[2], mode='linear', align_corners=False)
        
        return out
    
    def _match_size(self, x, target):
        if x.shape[2] != target.shape[2]:
            x = F.interpolate(x, size=target.shape[2], mode='linear', align_corners=False)
        return x


# ============================================================================
# Data Loading (Direct from PT files)
# ============================================================================

def load_ptbxl_test_data():
    """Load PTB-XL test data directly from tensor files."""
    
    test_dir = "/home/mithunmanivannan/data/ptb_xl/tensors/test"
    files = sorted(glob(os.path.join(test_dir, "*.pt")), 
                   key=lambda x: int(os.path.basename(x).replace('.pt', '')))
    
    print(f"Loading {len(files)} test files...")
    
    inputs = []  # 3-lead
    targets = []  # 12-lead
    
    # Lead indices: I=0, II=1, aVF=5 for input
    INPUT_LEADS = [0, 1, 5]
    
    for f in tqdm(files, desc="Loading test data"):
        tensor = torch.load(f, weights_only=True)
        
        # Normalize per-lead
        tensor = (tensor - tensor.mean(dim=1, keepdim=True)) / (tensor.std(dim=1, keepdim=True) + 1e-8)
        
        inp = tensor[INPUT_LEADS]  # [3, 5000]
        tgt = tensor  # [12, 5000]
        
        inputs.append(inp)
        targets.append(tgt)
    
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    
    print(f"Test data: inputs={inputs.shape}, targets={targets.shape}")
    return inputs, targets


def load_labels():
    """Load PTB-XL labels for test set."""
    import pandas as pd
    import ast
    
    # Load PTB-XL metadata
    ptbxl_path = "/home/mithunmanivannan/data/ptb_xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    df = pd.read_csv(os.path.join(ptbxl_path, "ptbxl_database.csv"))
    
    # Load scp_statements for label mapping
    scp_df = pd.read_csv(os.path.join(ptbxl_path, "scp_statements.csv"), index_col=0)
    
    # Filter to test fold (fold 10)
    test_df = df[df['strat_fold'] == 10].copy()
    
    # Superclass mapping
    superclass_map = {
        'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4
    }
    
    def get_superclass_labels(scp_codes_str):
        """Convert SCP codes to 5-class multi-label."""
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
    
    # Sort by ecg_id to match file order
    test_df = test_df.sort_values('ecg_id')
    
    labels = np.array([get_superclass_labels(row['scp_codes']) for _, row in test_df.iterrows()])
    
    print(f"Labels shape: {labels.shape}")
    print(f"Class distribution: {labels.sum(axis=0)}")
    
    return torch.tensor(labels, dtype=torch.float32)


# ============================================================================
# ECG-FM Classifier
# ============================================================================

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


# ============================================================================
# Fine-tuning and Evaluation
# ============================================================================

def fine_tune_and_evaluate(signals, labels, signal_name="M0", epochs=20, batch_size=32):
    """Fine-tune ECG-FM and compute AUROC."""
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning ECG-FM on {signal_name} signals")
    print(f"{'='*60}")
    
    # Split into train/val (80/20)
    n = len(signals)
    indices = torch.randperm(n)
    n_train = int(0.8 * n)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    train_ds = TensorDataset(signals[train_idx], labels[train_idx])
    val_ds = TensorDataset(signals[val_idx], labels[val_idx])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Load ECG-FM
    model = load_ecgfm_classifier(num_classes=5)
    model = model.to(device)
    
    # Only train the head
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auroc = 0
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        model.backbone.eval()  # Keep backbone frozen
        
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
                logits = model(x)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(y.numpy())
        
        probs = np.concatenate(all_probs)
        labels_np = np.concatenate(all_labels)
        
        # Compute AUROC
        try:
            auroc = roc_auc_score(labels_np, probs, average='macro')
        except:
            auroc = 0.5
        
        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Val AUROC = {auroc:.4f} (best={best_auroc:.4f})")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\nBest validation AUROC: {best_auroc:.4f}")
    
    return model, best_auroc


def bootstrap_auroc(model, signals, labels, n_bootstrap=1000, batch_size=32):
    """Compute AUROC with bootstrap 95% CI."""
    
    print(f"\nComputing bootstrap CI ({n_bootstrap} samples)...")
    
    model.eval()
    
    # Get all predictions
    dataset = TensorDataset(signals, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())
    
    probs = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)
    
    # Point estimate
    auroc_point = roc_auc_score(labels_np, probs, average='macro')
    
    # Bootstrap
    n_samples = len(probs)
    aurocs = []
    
    np.random.seed(42)
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        try:
            auroc = roc_auc_score(labels_np[idx], probs[idx], average='macro')
            aurocs.append(auroc)
        except:
            pass
    
    aurocs = np.array(aurocs)
    ci_lower = np.percentile(aurocs, 2.5)
    ci_upper = np.percentile(aurocs, 97.5)
    
    print(f"AUROC: {auroc_point:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return {
        'auroc': float(auroc_point),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'std': float(np.std(aurocs))
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("M0 AUROC EVALUATION")
    print("=" * 70)
    
    # Step 1: Load M0 model
    print("\n[1/5] Loading M0 reconstruction model...")
    m0_checkpoint = '/home/mithunmanivannan/checkpoints/M0_baseline_best.pt'
    
    recon_model = UNetReconstructor(in_channels=3, out_channels=12).to(device)
    checkpoint = torch.load(m0_checkpoint, map_location=device)
    recon_model.load_state_dict(checkpoint['model_state_dict'])
    recon_model.eval()
    
    print(f"  Loaded M0 checkpoint (Pearson={checkpoint.get('pearson', 'N/A')})")
    
    # Step 2: Load test data
    print("\n[2/5] Loading PTB-XL test data...")
    inputs, targets = load_ptbxl_test_data()
    labels = load_labels()
    
    # Step 3: Generate M0 reconstructions
    print("\n[3/5] Generating M0 reconstructions...")
    
    reconstructions = []
    batch_size = 64
    
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="Reconstructing"):
            batch = inputs[i:i+batch_size].to(device)
            recon = recon_model(batch)
            reconstructions.append(recon.cpu())
    
    reconstructions = torch.cat(reconstructions)
    print(f"  Reconstructions shape: {reconstructions.shape}")
    
    # Compute reconstruction metrics
    mse = F.mse_loss(reconstructions, targets).item()
    rho, _ = pearsonr(reconstructions.numpy().flatten(), targets.numpy().flatten())
    print(f"  M0 MSE: {mse:.6f}, Pearson: {rho:.4f}")
    
    # Step 4: Fine-tune ECG-FM on M0 reconstructions
    print("\n[4/5] Fine-tuning ECG-FM on M0 reconstructions...")
    ecgfm_model, val_auroc = fine_tune_and_evaluate(
        reconstructions, labels, signal_name="M0", epochs=20
    )
    
    # Step 5: Bootstrap AUROC
    print("\n[5/5] Computing bootstrap AUROC...")
    auroc_results = bootstrap_auroc(ecgfm_model, reconstructions, labels, n_bootstrap=1000)
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"M0 Reconstruction Pearson: {rho:.4f}")
    print(f"M0 AUROC: {auroc_results['auroc']:.4f} [{auroc_results['ci_lower']:.4f}, {auroc_results['ci_upper']:.4f}]")
    print(f"\nComparison with M1 HPO trials (AUROC 0.927-0.928):")
    
    if auroc_results['auroc'] < 0.92:
        print("  ✅ M0 AUROC < M1 AUROC → Morphology losses HELP")
        print("  → Hypothesis CONFIRMED: Multi-objective losses improve FM diagnostics")
    else:
        print("  ⚠️  M0 AUROC ≈ M1 AUROC → Morphology losses may NOT help significantly")
        print("  → Need to investigate further")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'm0_pearson': float(rho),
        'm0_mse': float(mse),
        'm0_auroc': auroc_results,
        'm1_auroc_range': [0.927, 0.928],
        'hypothesis_confirmed': auroc_results['auroc'] < 0.92
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/m0_auroc_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: results/m0_auroc_evaluation.json")
    
    return results


if __name__ == '__main__':
    main()
