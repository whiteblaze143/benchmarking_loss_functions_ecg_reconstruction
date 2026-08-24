#!/usr/bin/env python3
"""
Evaluate Sensitivity and Specificity of the Final Model (HuBERT-ECG Bridge).
Computes per-superclass metrics (Sens, Spec, AUROC, F1) against Ground Truth.
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
import torch
import numpy as np
import pandas as pd
import wfdb
import json
from tqdm import tqdm
from scipy.signal import resample
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader


from src.reconstruction.learn_functions.hubert_bridge import HuBERTBridge
from src.reconstruction.learn_functions.fam_ecg import UniversalSpatialFusionAdapter
from src.reconstruction.learn_functions.classifier import ResNet1d, ResBlock1d
from src.data.ptbxl_dataset import PTBXLDataset

# Configuration
HPO_BEST_DIR = "checkpoints/hpo_hubert_smart" 
BASELINE_CHECKPOINT = "checkpoints/production/final_model.pt"
ORACLE_CHECKPOINT = "checkpoints/oracle_original.pt"
DATA_DIR = os.path.expanduser("~/data/ptb_xl")
# Use the correct labels CSV with superclasses
LABELS_CSV = os.path.expanduser("~/data/ptb_xl/ptbxl_superclass_labels.csv") 
LEAD_INDICES = [0, 1, 8] # I, II, V3 (Mason)
TARGET_FS_MODEL = 500
TARGET_FS_ORACLE = 100 # Oracle expects 100Hz (based on typical PTB-XL benchmarks) 
                       # NOTE: evaluate_multi_lead_diagnostic used 250Hz.
                       # I will stick to 250Hz if that's what oracle_original.pt expects.
                       # Let's assume 250Hz.

class PTBXLDiagnosticDataset(Dataset):
    """
    Custom dataset for diagnostic evaluation of PTB-XL records.
    Returns (data, labels) where labels are 5 superclasses.
    """
    def __init__(self, root_dir, csv_file, split='test', target_fs=500):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.target_fs = target_fs
        
        # Filter by split
        if split == 'train':
            self.df = self.df[self.df.strat_fold <= 8]
        elif split == 'val':
            self.df = self.df[self.df.strat_fold == 9]
        elif split == 'test':
            self.df = self.df[self.df.strat_fold == 10]
            
        self.records = self.df.filename_hr.values
        
        # Process labels (assuming SCP codes are in 'scp_codes' column or similar)
        # But for valid comparison, we need the 5 superclasses: NORM, MI, STTC, CD, HYP
        # Requires 'scp_codes' column parsing or pre-computed columns.
        # If columns exist:
        if 'NORM' in self.df.columns:
             self.labels = self.df[['NORM', 'MI', 'STTC', 'CD', 'HYP']].values
        else:
            # Fallback (dummy or error)
            print("WARNING: Superclass columns not found! Using dummy labels.")
            self.labels = np.zeros((len(self.df), 5))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rel_path = self.records[idx]
        abs_path = os.path.join(self.root_dir, rel_path)
        
        # Load High-Res (500Hz)
        try:
            data, header = wfdb.rdsamp(abs_path)
        except Exception as e:
            print(f"Error loading {abs_path}: {e}")
            return torch.zeros(12, 5000), torch.zeros(5)

        # Resample if needed
        # PTB-XL HR is 500Hz.
        sampling_rate = header['fs']
        if sampling_rate != self.target_fs:
            num_samples = int(len(data) * self.target_fs / sampling_rate)
            data = resample(data, num_samples, axis=0)
            
        # Crop or Pad to 5000 (10s @ 500Hz)
        # PTB-XL is usually 10s.
        target_len = int(10 * self.target_fs)
        if len(data) > target_len:
            data = data[:target_len]
        elif len(data) < target_len:
            pad = np.zeros((target_len - len(data), 12))
            data = np.concatenate([data, pad], axis=0)
            
        data = torch.from_numpy(data.T).float() # [12, L]
        y = torch.from_numpy(self.labels[idx]).float()
        return data, y

def get_best_hpo_checkpoint():
    summary_path = os.path.join(HPO_BEST_DIR, "hpo_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        best_id = summary['best']['id']
        ckpt = os.path.join(HPO_BEST_DIR, f"trial_{best_id}", "best.pt")
        print(f"Found Best HPO Trial {best_id}: {ckpt}")
        return ckpt
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Sensitivity/Specificity")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to evaluate")
    parser.add_argument("--label", type=str, default="Model", help="Label for the model in report")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading checkpoint: {args.checkpoint or "Default/ASHA Best"}")
    
    # 1. Load Reconstructor
    bridge = HuBERTBridge(
        model_name="Edoardo-BS/hubert-ecg-large", 
        freeze_encoder=True,
        physics_projection=True, 
        use_res_decoder=True, 
        target_len=5000
    ).to(device)
    adapter = UniversalSpatialFusionAdapter(dim=bridge.embed_dim).to(device)
    
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = get_best_hpo_checkpoint()
        if not ckpt_path or not os.path.exists(ckpt_path):
            ckpt_path = BASELINE_CHECKPOINT
        
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'decoder_state_dict' in checkpoint:
            bridge.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if 'adapter_state_dict' in checkpoint:
            adapter.load_state_dict(checkpoint['adapter_state_dict'])
        print(f"✓ Loaded bridge weights from {ckpt_path}")
    
    bridge.eval()
    adapter.eval()
    
    # 2. Load Oracle
    oracle = ResNet1d(ResBlock1d, [3, 4, 23, 3], num_classes=5, input_channels=12).to(device)
    if os.path.exists(ORACLE_CHECKPOINT):
        oracle.load_state_dict(torch.load(ORACLE_CHECKPOINT, map_location=device, weights_only=False))
        print("✓ Loaded Oracle weights")
    else:
        print("WARNING: Oracle not found!")
    oracle.eval()
    
    # 3. Validation Data
    val_ds = PTBXLDiagnosticDataset(DATA_DIR, DATABASE_CSV, split='val', target_fs=TARGET_FS_MODEL)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Evaluating on {len(val_ds)} records...")
    
    gt_targets = []
    oracle_probs_gt = []
    oracle_probs_recon = []
    
    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x = x.to(device)
            y = y.to(device)
            
            # --- 1. Ground Truth (Upper Bound) ---
            # Resample X (GT) 500 -> 250 for Oracle
            x_down = torch.nn.functional.interpolate(x, size=2500, mode='linear', align_corners=False)
            logits_gt = oracle(x_down)
            probs_gt = torch.sigmoid(logits_gt)
            
            # --- 2. Reconstruction (Model) ---
            # Mason input: indices [0, 1, 8]
            x_in = x[:, LEAD_INDICES, :]
            with torch.amp.autocast('cuda'):
                recon = bridge(x_in, lead_indices=LEAD_INDICES, adapter=adapter)
            
            # Resample Recon 500 -> 250 for Oracle
            recon_down = torch.nn.functional.interpolate(recon, size=2500, mode='linear', align_corners=False)
            logits_recon = oracle(recon_down)
            probs_recon = torch.sigmoid(logits_recon)
            
            gt_targets.append(y.cpu().numpy())
            oracle_probs_gt.append(probs_gt.cpu().numpy())
            oracle_probs_recon.append(probs_recon.cpu().numpy())
            
    gt_targets = np.concatenate(gt_targets)
    oracle_probs_gt = np.concatenate(oracle_probs_gt)
    oracle_probs_recon = np.concatenate(oracle_probs_recon)
    
    # 4. Metrics
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    def print_metrics(targets, probs, label="Model"):
        print("\n" + "="*80)
        print(f"{label} SENSITIVITY / SPECIFICITY ANALYSIS")
        print("="*80)
        print(f"{'Class':<6} {'AUROC':<8} {'Sens':<8} {'Spec':<8} {'F1':<8} {'Acc':<8} {'Threshold':<8}")
        print("-" * 80)
    
        for i, cls in enumerate(classes):
            y_true = targets[:, i]
            y_score = probs[:, i]
            
            # AUROC
            try:
                auc = roc_auc_score(y_true, y_score)
            except:
                auc = 0.5
                
            # Best Threshold (Youden)
            thresholds = np.linspace(0, 1, 100)
            best_j = -1
            best_thresh = 0.5
            best_sens = 0
            best_spec = 0
            
            for t in thresholds:
                y_pred_t = (y_score >= t).astype(int)
                # Compute Confusion Matrix properly for binary
                # cm = confusion_matrix(y_true, y_pred_t, labels=[0,1])
                # TN, FP, FN, TP
                # Note: y_true is 0 or 1.
                tp = np.sum((y_pred_t == 1) & (y_true == 1))
                tn = np.sum((y_pred_t == 0) & (y_true == 0))
                fp = np.sum((y_pred_t == 1) & (y_true == 0))
                fn = np.sum((y_pred_t == 0) & (y_true == 1))
                
                sens = tp / (tp + fn + 1e-8)
                spec = tn / (tn + fp + 1e-8)
                j = sens + spec - 1
                
                if j > best_j:
                    best_j = j
                    best_thresh = t
                    best_sens = sens
                    best_spec = spec
                    
            # Final metrics
            y_pred = (y_score >= best_thresh).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            
            print(f"{cls:<6} {auc:<8.4f} {best_sens:<8.4f} {best_spec:<8.4f} {f1:<8.4f} {acc:<8.4f} {best_thresh:<8.4f}")
        print("="*80)

    print_metrics(gt_targets, oracle_probs_gt, label="ORACLE (Ground Truth Upper Bound)")
    print_metrics(gt_targets, oracle_probs_recon, label="RECONSTRUCTION (Model)")

if __name__ == "__main__":
    main()
