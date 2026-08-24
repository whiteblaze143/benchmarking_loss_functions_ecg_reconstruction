#!/usr/bin/env python3
"""
Fairness Evaluation Script
Links model predictions to PTB-XL demographics using filename keys.
Calculates AUROC/ECE for subgroups: Sex (Male/Female), Age (<40, 40-65, >65).
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
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from pathlib import Path

from src.reconstruction.learn_functions.mason_mmd_variants import MasonMMD_V4_RationalKernel
from src.reconstruction.learn_functions.classifier import xresnet1d101
from src.data.multi_source_dataset import MultiSourceECGDataset

# Configuration
METADATA_PATH = '/home/mithunmanivannan/data/ptbxl_database.csv'
CHECKPOINTS = {
    'M0': '/home/mithunmanivannan/checkpoints/M0_seed42.pt',
    'M1': '/home/mithunmanivannan/checkpoints/M1_Final.pt',
    # 'M2': '/home/mithunmanivannan/checkpoints/M2_main_seed42.pt', # Pending
}
ORACLE_PATH = '/home/mithunmanivannan/checkpoints/diagnostic_classifier.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU as requested
print(f"Fairness Evaluation using: {DEVICE}")

def load_metadata():
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata not found at {METADATA_PATH}")
        sys.exit(1)
    
    df = pd.read_csv(METADATA_PATH)
    # Create mapping from filename stem (e.g. 00001_hr or 00001_lr) to demographics
    # We strip 'records.../' and extension
    
    # Actually MultiSourceECGDataset returns simple stem if it matches
    # Let's clean the dataframe filenames to be safe
    # But batch['file'] returns stem (e.g. '00001_hr')
    
    # Map from stem -> row
    meta_map = {}
    for idx, row in df.iterrows():
        # HR
        stem_hr = Path(row['filename_hr']).stem
        meta_map[stem_hr] = row
        # LR
        stem_lr = Path(row['filename_lr']).stem
        meta_map[stem_lr] = row
        # ID as string
        meta_map[str(row['ecg_id'])] = row
        
    print(f"Loaded metadata for {len(meta_map)} keys")
    return meta_map

def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None
    
    model = MasonMMD_V4_RationalKernel(
        input_lead_num=3, output_lead_num=12,
        lambda_mse=1.0, lambda_mmd=0.05,
        lambda_deriv=0.1, lambda_corr=0.1, use_dcor=False
    )
    
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if any(k.startswith('reconstructor.') for k in state.keys()):
        new_state = {k.replace('reconstructor.', ''): v for k, v in state.items() if k.startswith('reconstructor.')}
        model.load_state_dict(new_state)
    else:
        model.load_state_dict(state)
    
    return model.to(DEVICE).eval()

def main():
    print(f"Using device: {DEVICE}")
    meta_map = load_metadata()
    
    # Load Oracle
    oracle = xresnet1d101(num_classes=5, input_channels=12).to(DEVICE)
    oracle.load_state_dict(torch.load(ORACLE_PATH, map_location=DEVICE, weights_only=False))
    oracle.eval()
    
    # Load Data
    data_path = "/home/mithunmanivannan/data/ptbxl_tensors"
    sources = [{"name": "PTB-XL", "path": data_path, "format": "pt"}]
    dataset = MultiSourceECGDataset(sources=sources, split='test', target_len=5000, normalization='min_max')
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2) # 32 is safe for shared GPU
    
    print(f"Eval Set: {len(dataset)}")
    
    # Process each model
    for model_name, ckpt_path in CHECKPOINTS.items():
        print(f"\nProcessing {model_name}...")
        model = load_model(ckpt_path)
        if model is None:
            print("Checkpoint missing.")
            continue
            
        results = []
        
        with torch.no_grad():
            for batch in tqdm(loader):
                inputs = batch['input'].to(DEVICE)
                files = batch['file']
                labels = batch['label'].numpy()
                
                # Recon
                output = model(inputs)
                recon = output[0] if isinstance(output, tuple) else output
                
                # Oracle
                logits = oracle(recon)
                probs = torch.sigmoid(logits).cpu().numpy()
                
                # Link
                for i, f in enumerate(files):
                    if f in meta_map:
                        row = meta_map[f]
                        results.append({
                            'age': row['age'],
                            'sex': row['sex'], # 0=Male, 1=Female usually? Need to check.
                            'probs': probs[i],
                            'label': labels[i]
                        })
        
        # Analyze Fairness
        df_res = pd.DataFrame(results)
        print(f"Matched {len(df_res)} records.")
        
        # Global check
        labels_all = np.vstack(df_res['label'])
        probs_all = np.vstack(df_res['probs'])
        auc_global = roc_auc_score(labels_all, probs_all, average='macro', multi_class='ovr')
        print(f"Global AUROC: {auc_global:.4f} (Should be ~0.90)")

        # Sex (PTB-XL: 0=Male, 1=Female)
        res_male = df_res[df_res['sex'] == 0]
        res_female = df_res[df_res['sex'] == 1]
        
        auc_male = roc_auc_score(np.vstack(res_male['label']), np.vstack(res_male['probs']), average='macro', multi_class='ovr')
        auc_female = roc_auc_score(np.vstack(res_female['label']), np.vstack(res_female['probs']), average='macro', multi_class='ovr')
        
        print(f"Sex Fairness:")
        print(f"  Male (N={len(res_male)}): {auc_male:.4f}")
        print(f"  Female (N={len(res_female)}): {auc_female:.4f}")
        print(f"  Gap: {abs(auc_male - auc_female):.4f}")
        
        # Age
        res_young = df_res[df_res['age'] < 50]
        res_mid = df_res[(df_res['age'] >= 50) & (df_res['age'] <= 75)]
        res_old = df_res[df_res['age'] > 75]
        
        auc_young = roc_auc_score(np.vstack(res_young['label']), np.vstack(res_young['probs']), average='macro', multi_class='ovr')
        auc_mid = roc_auc_score(np.vstack(res_mid['label']), np.vstack(res_mid['probs']), average='macro', multi_class='ovr')
        auc_old = roc_auc_score(np.vstack(res_old['label']), np.vstack(res_old['probs']), average='macro', multi_class='ovr')
        
        print(f"Age Fairness:")
        print(f"  <50 (N={len(res_young)}): {auc_young:.4f}")
        print(f"  50-75 (N={len(res_mid)}): {auc_mid:.4f}")
        print(f"  >75 (N={len(res_old)}): {auc_old:.4f}")

if __name__ == '__main__':
    main()
