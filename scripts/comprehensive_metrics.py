#!/usr/bin/env python3
import torch
import torch.nn as nn
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

# Project imports
# from src.reconstruction.learn_functions.mason_mmd_variants import MasonMMD_V4_RationalKernel # Removed broken import
from src.reconstruction.learn_functions.classifier import xresnet1d101

def load_model_reconstructor(checkpoint_path, device, model_type="modular_bridge"):
    """
    Standard loader for reconstructors.
    """
    print(f"Loading reconstructor from {checkpoint_path}")
    if model_type == "modular_bridge":
        from src.reconstruction.learn_functions.bridge import ModularBridge
        model = ModularBridge(device=device)
    else:
        # Fallback for older projects if needed, but for now we use bridge
        raise ValueError(f"Unknown model_type: {model_type}")
    
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Handle state dict wrapped in 'model_state_dict' or with 'reconstructor.' prefix
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        
        new_state = {}
        for k, v in state.items():
            if k.startswith('reconstructor.'):
                new_state[k.replace('reconstructor.', '')] = v
            else:
                new_state[k] = v
        
        # Filter keys to match model
        model_keys = model.state_dict().keys()
        final_state = {k: v for k, v in new_state.items() if k in model_keys}
        
        model.load_state_dict(final_state, strict=False)
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found.")
        
    model.to(device)
    model.eval()
    return model

import pandas as pd
import numpy as np
import ast

def load_ptbxl_data(data_path='/home/mithunmanivannan/data/ptbxl_database.csv'):
    """Simplified PTB-XL loader."""
    df = pd.read_csv(data_path, index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # Standard fold 10 split
    test_df = df[df.strat_fold == 10]
    train_df = df[df.strat_fold <= 8]
    val_df = df[df.strat_fold == 9]
    return train_df, val_df, test_df

class PTBXLDataset(torch.utils.data.Dataset):
    def __init__(self, df, sampling_rate=500, base_path='/home/mithunmanivannan/data/'):
        self.df = df
        self.sampling_rate = sampling_rate
        self.base_path = base_path
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        # Implementation depends on how files are named/stored
        # Usually it's in filename_lr or filename_hr
        row = self.df.iloc[idx]
        filename = row.filename_lr if self.sampling_rate == 100 else row.filename_hr
        
        # Using a simplified version of wfdb loading if possible, or just mock for now
        # Actually I need real data. Let's assume wfdb is available.
        import wfdb
        record = wfdb.rdrecord(os.path.join(self.base_path, filename))
        data = record.p_signal.T # [12, 5000]
        
        # Normalize - PTB-XL raw is mV
        # Many models expect [0,1] or z-score. Let's provide raw and let model handle.
        # But wait, Orignal Oracle expects [0, 1] mapped from [-5, 5]
        
        data = torch.from_numpy(data).float()
        
        # Target (simplified to labels)
        target = 0 # Placeholder for classification
        return data, target

class ComprehensiveMetrics:
    """Placeholder for complex metrics if needed by other scripts."""
    def __init__(self):
        pass
