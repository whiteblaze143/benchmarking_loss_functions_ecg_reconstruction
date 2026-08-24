#!/usr/bin/env python3
"""
Generate M0 (MSE baseline) and M1 (morphology-aware) reconstructed signals.

This script loads trained reconstruction models and generates reconstructed
12-lead ECGs from 3-lead inputs for the PTB-XL test set.

Usage:
    python src/generate_reconstructions.py --output_dir ~/data/ptb_xl/reconstructed

Outputs:
    - ~/data/ptb_xl/reconstructed/m0/{ecg_id}.pt
    - ~/data/ptb_xl/reconstructed/m1_hpo/{ecg_id}.pt
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project paths
PROJECT_ROOT = Path(os.path.expanduser("~"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "M1_ARISE_Repo"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Import reconstruction models
try:
    from src.reconstruction.learn_functions.mason_mmd_variants import MasonMMD_V4_RationalKernel
    from src.reconstruction.learn_functions.legacy_mason import LegacyMasonReconstructor
    from src.reconstruction.learn_functions.mason_reconstructor import MasonReconstructor
    HAS_MODELS = True
except ImportError as e:
    logger.warning(f"Could not import reconstruction models: {e}")
    HAS_MODELS = False


class PTBXLTestDataset(Dataset):
    """Dataset for loading PTB-XL test signals."""
    
    def __init__(
        self,
        data_dir: str,
        database_csv: str,
        input_leads: list = [0, 1, 7],  # I, II, V2
        target_len: int = 5000
    ):
        self.data_dir = Path(data_dir)
        self.input_leads = input_leads
        self.target_len = target_len
        
        # Load database and filter test set (fold 10)
        df = pd.read_csv(database_csv)
        self.df = df[df['strat_fold'] == 10].reset_index(drop=True)
        
        logger.info(f"Loaded {len(self.df)} test records")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        ecg_id = row['ecg_id']
        
        # Try to load tensor
        tensor_path = self.data_dir / "tensors" / "test" / f"{ecg_id}.pt"
        
        if tensor_path.exists():
            signal = torch.load(tensor_path, weights_only=True)
            if isinstance(signal, dict):
                signal = signal.get('data', signal.get('signal', None))
        else:
            # Return None to skip
            return {
                'signal_12lead': None,
                'signal_3lead': None,
                'ecg_id': ecg_id
            }
        
        # Ensure correct shape (12, 5000)
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        if signal.shape[0] > 12:
            signal = signal[:12]
        elif signal.shape[0] < 12:
            pad = torch.zeros(12 - signal.shape[0], signal.shape[1])
            signal = torch.cat([signal, pad], dim=0)
        
        # Pad/crop to target length
        if signal.shape[1] < self.target_len:
            pad = torch.zeros(12, self.target_len - signal.shape[1])
            signal = torch.cat([signal, pad], dim=1)
        elif signal.shape[1] > self.target_len:
            signal = signal[:, :self.target_len]
        
        # Extract 3-lead input
        signal_3lead = signal[self.input_leads, :]
        
        return {
            'signal_12lead': signal,
            'signal_3lead': signal_3lead,
            'ecg_id': ecg_id
        }


def load_m0_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load M0 (MSE-only baseline) model.
    M0 is typically a vanilla Mason reconstructor trained with MSE loss only.
    """
    if not HAS_MODELS:
        raise RuntimeError("Reconstruction models not available")
    
    # Try to load as LegacyMasonReconstructor first
    try:
        model = LegacyMasonReconstructor(input_lead_num=3, output_lead_num=12)
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        
        logger.info(f"Loaded M0 model from {checkpoint_path}")
        return model.to(device).eval()
        
    except Exception as e:
        logger.warning(f"LegacyMason failed: {e}, trying MasonReconstructor")
        
        model = MasonReconstructor(in_leads=3, out_leads=12)
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        
        return model.to(device).eval()


def load_m1_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load M1 (morphology-aware HPO) model.
    M1 uses MMD + derivative + correlation losses.
    """
    if not HAS_MODELS:
        raise RuntimeError("Reconstruction models not available")
    
    model = MasonMMD_V4_RationalKernel(
        input_lead_num=3,
        output_lead_num=12,
        lambda_mse=1.0,
        lambda_mmd=0.05,
        lambda_deriv=0.1,
        lambda_corr=0.1,
        use_dcor=False
    )
    
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    
    logger.info(f"Loaded M1 model from {checkpoint_path}")
    return model.to(device).eval()


@torch.no_grad()
def generate_reconstructions(
    model: nn.Module,
    data_loader: DataLoader,
    output_dir: Path,
    device: torch.device,
    model_type: str = "m0"
) -> int:
    """Generate reconstructed signals and save to disk."""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    
    for batch in tqdm(data_loader, desc=f"Generating {model_type}"):
        signals_3lead = batch['signal_3lead']
        signals_12lead = batch['signal_12lead']
        ecg_ids = batch['ecg_id']
        
        # Skip None signals
        valid_mask = [s is not None for s in signals_12lead]
        if not any(valid_mask):
            continue
        
        # Filter valid samples
        valid_3lead = torch.stack([s for s, v in zip(signals_3lead, valid_mask) if v]).to(device)
        valid_ids = [i for i, v in zip(ecg_ids, valid_mask) if v]
        
        if len(valid_3lead) == 0:
            continue
        
        # Reconstruct
        output = model(valid_3lead)
        
        # Handle different output formats
        if isinstance(output, tuple):
            recon = output[0]
        else:
            recon = output
        
        # Save each reconstruction
        for i, ecg_id in enumerate(valid_ids):
            recon_signal = recon[i].cpu()
            output_path = output_dir / f"{ecg_id}.pt"
            torch.save(recon_signal, output_path)
            count += 1
    
    return count


def main():
    parser = argparse.ArgumentParser(description="Generate M0/M1 Reconstructions")
    parser.add_argument("--data_dir", type=str, default="~/data/ptb_xl")
    parser.add_argument("--database_csv", type=str, default="~/data/ptbxl_database.csv")
    parser.add_argument("--output_dir", type=str, default="~/data/ptb_xl/reconstructed")
    parser.add_argument("--m0_checkpoint", type=str, 
                       default="~/checkpoints/ablations/baseline.pt",
                       help="Path to M0 (MSE-only) model checkpoint")
    parser.add_argument("--m1_checkpoint", type=str,
                       default="~/checkpoints/M1_seed42.pt",
                       help="Path to M1 (morphology-aware) model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Expand paths
    args.data_dir = os.path.expanduser(args.data_dir)
    args.database_csv = os.path.expanduser(args.database_csv)
    args.output_dir = os.path.expanduser(args.output_dir)
    args.m0_checkpoint = os.path.expanduser(args.m0_checkpoint)
    args.m1_checkpoint = os.path.expanduser(args.m1_checkpoint)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset
    dataset = PTBXLTestDataset(
        data_dir=args.data_dir,
        database_csv=args.database_csv
    )
    
    # Custom collate to handle None values
    def collate_fn(batch):
        return {
            'signal_12lead': [b['signal_12lead'] for b in batch],
            'signal_3lead': [b['signal_3lead'] for b in batch],
            'ecg_id': [b['ecg_id'] for b in batch]
        }
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    output_path = Path(args.output_dir)
    
    # Generate M0 reconstructions
    if os.path.exists(args.m0_checkpoint):
        logger.info("\n--- Generating M0 (MSE baseline) reconstructions ---")
        m0_model = load_m0_model(args.m0_checkpoint, device)
        m0_count = generate_reconstructions(
            m0_model, loader, output_path / "m0", device, "m0"
        )
        logger.info(f"Generated {m0_count} M0 reconstructions")
        del m0_model
        torch.cuda.empty_cache()
    else:
        logger.warning(f"M0 checkpoint not found: {args.m0_checkpoint}")
    
    # Generate M1 reconstructions
    if os.path.exists(args.m1_checkpoint):
        logger.info("\n--- Generating M1 (morphology-aware) reconstructions ---")
        m1_model = load_m1_model(args.m1_checkpoint, device)
        m1_count = generate_reconstructions(
            m1_model, loader, output_path / "m1_hpo", device, "m1_hpo"
        )
        logger.info(f"Generated {m1_count} M1 reconstructions")
    else:
        logger.warning(f"M1 checkpoint not found: {args.m1_checkpoint}")
    
    print("\n" + "=" * 60)
    print("RECONSTRUCTION GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"M0 (MSE baseline): {output_path / 'm0'}")
    print(f"M1 (morphology-aware): {output_path / 'm1_hpo'}")


if __name__ == "__main__":
    main()
