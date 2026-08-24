#!/usr/bin/env python3
"""
Phase 39: Diagnostic Utility Test
==================================
Answers the fundamental question: Does FM-augmented reconstruction
preserve diagnostic information better than a raw Mason CNN?

Protocol:
1. Train a ResNet1d diagnostic classifier on ORIGINAL 12-lead PTB-XL
2. Freeze it, then run inference on:
   (a) Original 12-lead ECGs (upper bound)
   (b) FM-reconstructed 12-lead ECGs (our model)
   (c) Mason-CNN-reconstructed 12-lead ECGs (baseline)
3. Compare AUROC across all three

If (b) > (c): FM adds diagnostic value → Option A (clinical validation)
If (b) ≈ (c): FM doesn't help → Option B (FM-as-critic pivot)
If (b) < (c): FM hurts diagnostics → rethink everything
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
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import wfdb
import pandas as pd
from scipy.signal import resample

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src.reconstruction.learn_functions.classifier import ResNet1d, ResBlock1d


# ============================================================
# Dataset: PTB-XL with 5 superclass labels
# ============================================================
class PTBXLDiagnostic(Dataset):
    """Loads 12-lead PTB-XL ECGs with diagnostic superclass labels."""
    def __init__(self, root_dir, labels_csv, split='train', target_fs=500):
        self.root_dir = root_dir
        self.df = pd.read_csv(labels_csv)
        self.target_fs = target_fs

        if split == 'train':
            self.df = self.df[self.df.strat_fold <= 8]
        elif split == 'val':
            self.df = self.df[self.df.strat_fold == 9]
        elif split == 'test':
            self.df = self.df[self.df.strat_fold == 10]

        self.records = self.df.filename_hr.values
        self.label_cols = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        self.labels = self.df[self.label_cols].values

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rel_path = self.records[idx]
        abs_path = os.path.join(self.root_dir, rel_path)
        data, header = wfdb.rdsamp(abs_path)

        sampling_rate = header['fs']
        if sampling_rate != self.target_fs:
            num_samples = int(len(data) * self.target_fs / sampling_rate)
            data = resample(data, num_samples, axis=0)

        # (T, 12) -> (12, T)
        data = torch.from_numpy(data.T).float()
        y = torch.from_numpy(self.labels[idx]).float()
        return data, y


# ============================================================
# Step 1: Train 12-lead diagnostic classifier
# ============================================================
def train_classifier(device, root_dir, labels_csv, save_path, epochs=15, batch_size=64):
    """Train a ResNet1d on original 12-lead PTB-XL for diagnostic classification."""
    print("=" * 60)
    print("STEP 1: Training 12-lead diagnostic classifier")
    print("=" * 60)

    train_ds = PTBXLDiagnostic(root_dir, labels_csv, split='train', target_fs=500)
    val_ds = PTBXLDiagnostic(root_dir, labels_csv, split='val', target_fs=500)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = ResNet1d(ResBlock1d, [2, 2, 2, 2], num_classes=5, input_channels=12).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_auc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                all_logits.append(model(x).cpu().numpy())
                all_labels.append(y.numpy())

        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        aucs = []
        for i in range(5):
            try:
                aucs.append(roc_auc_score(all_labels[:, i], all_logits[:, i]))
            except ValueError:
                aucs.append(0.5)
        mean_auc = np.mean(aucs)

        print(f"  Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Val AUROC={mean_auc:.4f} "
              f"[NORM={aucs[0]:.3f} MI={aucs[1]:.3f} STTC={aucs[2]:.3f} CD={aucs[3]:.3f} HYP={aucs[4]:.3f}]")
        scheduler.step(mean_auc)

        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model saved (AUROC: {best_auc:.4f})")

    print(f"\nClassifier training complete. Best Val AUROC: {best_auc:.4f}")
    return best_auc


# ============================================================
# Step 2: Evaluate diagnostic AUROC on reconstructed ECGs
# ============================================================
def evaluate_diagnostic_auroc(classifier, dataloader, device, label_names):
    """Run frozen classifier on a DataLoader, return per-class AUROC."""
    classifier.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating", leave=False):
            x = x.to(device)
            all_logits.append(classifier(x).cpu().numpy())
            all_labels.append(y.numpy())

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    results = {}
    aucs = []
    for i, name in enumerate(label_names):
        try:
            auc = roc_auc_score(all_labels[:, i], all_logits[:, i])
        except ValueError:
            auc = 0.5
        results[name] = auc
        aucs.append(auc)
    results['mean'] = np.mean(aucs)
    return results


class ReconstructedECGDataset(Dataset):
    """Wraps a PTBXLDiagnostic dataset and replaces ECGs with reconstructions."""
    def __init__(self, base_dataset, reconstruction_model, device, model_type='ecgfm'):
        self.base = base_dataset
        self.model = reconstruction_model
        self.device = device
        self.model_type = model_type

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        original_ecg, label = self.base[idx]
        # original_ecg: (12, T)
        # Extract input leads: I (0), II (1), V3 (8)
        x_input = original_ecg[[0, 1, 8], :].unsqueeze(0).to(self.device)  # (1, 3, T)

        with torch.no_grad():
            if self.model_type == 'ecgfm':
                indices = torch.tensor([0, 1, 8], device=self.device)
                output = self.model(x_input, lead_indices=indices)
                if isinstance(output, tuple):
                    recon = output[0]  # (1, 12, T)
                else:
                    recon = output
            elif self.model_type == 'mason':
                # Mason expects list of [1, 1, T] inputs
                input_leads = [x_input[:, k:k+1, :] for k in range(3)]
                output_leads = self.model.forward(input_leads)
                # Assemble 12-lead: limb leads from original, precordials from reconstruction
                recon = original_ecg.unsqueeze(0).clone().to(self.device)
                for i, lead_out in enumerate(output_leads):
                    recon[0, 6 + i, :] = lead_out.squeeze()

        recon = recon.squeeze(0).cpu()  # (12, T)

        # Ensure same length
        T_orig = original_ecg.shape[-1]
        T_recon = recon.shape[-1]
        if T_recon != T_orig:
            recon = F.interpolate(recon.unsqueeze(0), size=T_orig, mode='linear', align_corners=False).squeeze(0)

        return recon, label


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 39: Diagnostic Utility Test")
    parser.add_argument("--root", default="data/ptb_xl")
    parser.add_argument("--labels_csv", default="data/ptb_xl/ptbxl_comprehensive_labels.csv")
    parser.add_argument("--fm_checkpoint", default="checkpoints/fast_ecgfm_spec_1.0/best_theory_model.pt")
    parser.add_argument("--mason_checkpoint", default="checkpoints/mason_baseline_aligned/best_mason_baseline/")
    parser.add_argument("--classifier_path", default="checkpoints/diagnostic_12lead_best.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--skip_training", action="store_true", help="Skip classifier training if checkpoint exists")
    parser.add_argument("--output", default="results/diagnostic_utility_test.json")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    label_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

    # ---- Step 1: Train/Load Classifier ----
    if args.skip_training and os.path.exists(args.classifier_path):
        print(f"Loading pre-trained classifier from {args.classifier_path}")
    else:
        train_classifier(device, args.root, args.labels_csv, args.classifier_path, 
                        epochs=args.epochs, batch_size=args.batch_size)

    classifier = ResNet1d(ResBlock1d, [2, 2, 2, 2], num_classes=5, input_channels=12).to(device)
    classifier.load_state_dict(torch.load(args.classifier_path, map_location=device, weights_only=False))
    classifier.eval()
    print("✓ Diagnostic classifier loaded.\n")

    # ---- Step 2: Load test set ----
    test_ds = PTBXLDiagnostic(args.root, args.labels_csv, split='test', target_fs=500)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ---- Step 2a: Original 12-lead (upper bound) ----
    print("=" * 60)
    print("EVALUATING: Original 12-lead ECGs (upper bound)")
    print("=" * 60)
    original_auroc = evaluate_diagnostic_auroc(classifier, test_loader, device, label_names)
    print(f"  Mean AUROC: {original_auroc['mean']:.4f}")
    for k, v in original_auroc.items():
        if k != 'mean':
            print(f"    {k}: {v:.4f}")

    # ---- Step 2b: FM-Reconstructed ----
    print("\n" + "=" * 60)
    print("EVALUATING: FM-Reconstructed ECGs (our model)")
    print("=" * 60)
    fm_auroc = {'mean': 0.0}
    try:
        from src.reconstruction.learn_functions.ecgfm_bridge import ECGFMBridge
        from src.reconstruction.learn_functions.fam_ecg import UniversalSpatialFusionAdapter

        ckpt = torch.load(args.fm_checkpoint, map_location=device, weights_only=False)
        bridge = ECGFMBridge(
            checkpoint_path="ecg_fm_integration/checkpoints/mimic_iv_ecg_physionet_pretrained.pt",
            freeze_encoder=True
        ).to(device)
        bridge.load_state_dict(ckpt['bridge_state_dict'], strict=False)
        bridge.eval()

        adapter = UniversalSpatialFusionAdapter(dim=bridge.embed_dim).to(device)
        if 'adapter_state_dict' in ckpt:
            adapter.load_state_dict(ckpt['adapter_state_dict'])
        adapter.eval()

        # Build FM reconstructions
        fm_recon_ds = ReconstructedECGDataset(test_ds, bridge, device, model_type='ecgfm')
        fm_loader = DataLoader(fm_recon_ds, batch_size=1, shuffle=False, num_workers=0)
        fm_auroc = evaluate_diagnostic_auroc(classifier, fm_loader, device, label_names)
        print(f"  Mean AUROC: {fm_auroc['mean']:.4f}")
        for k, v in fm_auroc.items():
            if k != 'mean':
                print(f"    {k}: {v:.4f}")
    except Exception as e:
        print(f"  ⚠ FM evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # ---- Step 2c: Mason-CNN-Reconstructed ----
    print("\n" + "=" * 60)
    print("EVALUATING: Mason-CNN-Reconstructed ECGs (baseline)")
    print("=" * 60)
    mason_auroc = {'mean': 0.0}
    try:
        MASON_ROOT = os.path.join(PROJECT_ROOT, "third_party", "ecg_reconstruction")
        sys.path.insert(0, MASON_ROOT)
        from learn_functions.generate_model import generate_reconstructor

        mason_model = generate_reconstructor(
            3, 6, input_channel_per_lead=32, middle_channel_per_lead=32,
            output_channel_per_lead=32, block_per_input_network=3,
            block_per_middle_network=2, block_per_output_network=3,
            input_kernel_size=17, middle_kernel_size=17, output_kernel_size=17,
            use_residual="true", device=device
        )
        mason_model.load_state_dict(os.path.join(PROJECT_ROOT, args.mason_checkpoint))

        mason_recon_ds = ReconstructedECGDataset(test_ds, mason_model, device, model_type='mason')
        mason_loader = DataLoader(mason_recon_ds, batch_size=1, shuffle=False, num_workers=0)
        mason_auroc = evaluate_diagnostic_auroc(classifier, mason_loader, device, label_names)
        print(f"  Mean AUROC: {mason_auroc['mean']:.4f}")
        for k, v in mason_auroc.items():
            if k != 'mean':
                print(f"    {k}: {v:.4f}")
    except Exception as e:
        print(f"  ⚠ Mason evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("DIAGNOSTIC UTILITY COMPARISON")
    print("=" * 60)
    print(f"{'Source':<25} {'Mean AUROC':<12} {'NORM':<8} {'MI':<8} {'STTC':<8} {'CD':<8} {'HYP':<8}")
    print("-" * 77)
    for name, res in [("Original 12-lead", original_auroc), 
                       ("FM-Reconstructed", fm_auroc),
                       ("Mason-CNN", mason_auroc)]:
        norm = res.get('NORM', 0)
        mi = res.get('MI', 0)
        sttc = res.get('STTC', 0)
        cd = res.get('CD', 0)
        hyp = res.get('HYP', 0)
        print(f"{name:<25} {res['mean']:<12.4f} {norm:<8.4f} {mi:<8.4f} {sttc:<8.4f} {cd:<8.4f} {hyp:<8.4f}")

    # Decision
    if fm_auroc['mean'] > mason_auroc['mean'] + 0.01:
        verdict = "OPTION_A: FM adds diagnostic value. Proceed with clinical validation."
    elif abs(fm_auroc['mean'] - mason_auroc['mean']) <= 0.01:
        verdict = "OPTION_B: FM doesn't help diagnostics. Pivot to FM-as-critic."
    else:
        verdict = "RETHINK: FM hurts diagnostic utility. Fundamental issue."
    print(f"\n🔬 VERDICT: {verdict}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = {
        'original': original_auroc,
        'fm_reconstructed': fm_auroc,
        'mason_cnn': mason_auroc,
        'verdict': verdict
    }
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Results saved to {args.output}")


if __name__ == "__main__":
    main()
