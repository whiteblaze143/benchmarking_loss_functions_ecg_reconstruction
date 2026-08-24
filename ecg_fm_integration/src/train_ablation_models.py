#!/usr/bin/env python3
"""
Train Ablation Models: Systematic Loss Component Study

Trains reconstruction models with different loss configurations to understand
the contribution of each component.

Usage:
    python src/train_ablation_models.py --config M0      # MSE only baseline
    python src/train_ablation_models.py --config M0+MMD  # MSE + MMD
    python src/train_ablation_models.py --config M1      # Full multi-objective
    python src/train_ablation_models.py --all            # Train all configs

Outputs:
    - checkpoints/ablation/{config}_seed{seed}.pt
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add paths
HOME = Path(os.path.expanduser("~"))
sys.path.insert(0, str(HOME))
sys.path.insert(0, str(HOME / 'M1_ARISE_Repo'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Loss Components
# ============================================================================

class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy loss with RBF kernel."""
    
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma
    
    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix."""
        xx = torch.sum(x ** 2, dim=-1, keepdim=True)
        yy = torch.sum(y ** 2, dim=-1, keepdim=True)
        dist = xx + yy.transpose(-2, -1) - 2 * torch.matmul(x, y.transpose(-2, -1))
        return torch.exp(-dist / (2 * self.sigma ** 2))
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD between x and y distributions.
        
        Args:
            x, y: (batch, features) or (batch, time, features)
        """
        if x.dim() == 3:
            # Flatten temporal dimension
            x = x.view(x.size(0), -1)
            y = y.view(y.size(0), -1)
        
        k_xx = self.rbf_kernel(x, x)
        k_yy = self.rbf_kernel(y, y)
        k_xy = self.rbf_kernel(x, y)
        
        n = x.size(0)
        
        # Unbiased estimator
        mmd = (k_xx.sum() - k_xx.diagonal().sum()) / (n * (n - 1))
        mmd += (k_yy.sum() - k_yy.diagonal().sum()) / (n * (n - 1))
        mmd -= 2 * k_xy.mean()
        
        return torch.clamp(mmd, min=0.0)


class DerivativeLoss(nn.Module):
    """Gradient/derivative preservation loss."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 loss on temporal derivatives.
        
        Args:
            pred, target: (batch, leads, time)
        """
        # First derivative
        pred_deriv = pred[:, :, 1:] - pred[:, :, :-1]
        target_deriv = target[:, :, 1:] - target[:, :, :-1]
        
        return F.l1_loss(pred_deriv, target_deriv)


class CorrelationLoss(nn.Module):
    """Pearson correlation loss (1 - correlation)."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation loss.
        
        Args:
            pred, target: (batch, leads, time)
        """
        # Flatten to (batch, features)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Center
        pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
        target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
        
        # Compute correlation
        numerator = (pred_centered * target_centered).sum(dim=1)
        denominator = torch.sqrt(
            (pred_centered ** 2).sum(dim=1) * (target_centered ** 2).sum(dim=1)
        ) + 1e-8
        
        correlation = numerator / denominator
        
        # Return 1 - mean(correlation) so we minimize
        return 1 - correlation.mean()


class CompositeLoss(nn.Module):
    """Composite loss combining MSE, MMD, Derivative, and Correlation."""
    
    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_mmd: float = 0.0,
        lambda_deriv: float = 0.0,
        lambda_corr: float = 0.0
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_mmd = lambda_mmd
        self.lambda_deriv = lambda_deriv
        self.lambda_corr = lambda_corr
        
        self.mse = nn.MSELoss()
        self.mmd = MMDLoss(sigma=1.0) if lambda_mmd > 0 else None
        self.deriv = DerivativeLoss() if lambda_deriv > 0 else None
        self.corr = CorrelationLoss() if lambda_corr > 0 else None
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute composite loss.
        
        Returns:
            total_loss: Weighted sum of all components
            loss_dict: Individual loss values for logging
        """
        loss_dict = {}
        total_loss = 0.0
        
        # MSE
        mse_loss = self.mse(pred, target)
        loss_dict['mse'] = mse_loss.item()
        total_loss = total_loss + self.lambda_mse * mse_loss
        
        # MMD
        if self.mmd is not None and self.lambda_mmd > 0:
            mmd_loss = self.mmd(pred, target)
            loss_dict['mmd'] = mmd_loss.item()
            total_loss = total_loss + self.lambda_mmd * mmd_loss
        
        # Derivative
        if self.deriv is not None and self.lambda_deriv > 0:
            deriv_loss = self.deriv(pred, target)
            loss_dict['deriv'] = deriv_loss.item()
            total_loss = total_loss + self.lambda_deriv * deriv_loss
        
        # Correlation
        if self.corr is not None and self.lambda_corr > 0:
            corr_loss = self.corr(pred, target)
            loss_dict['corr'] = corr_loss.item()
            total_loss = total_loss + self.lambda_corr * corr_loss
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# ============================================================================
# Simple Reconstruction Model
# ============================================================================

class SimpleUNet1D(nn.Module):
    """Simplified 1D U-Net for ECG reconstruction."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 12, base_channels: int = 32):
        super().__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.enc3 = self._block(base_channels * 2, base_channels * 4)
        
        # Bottleneck
        self.bottleneck = self._block(base_channels * 4, base_channels * 8)
        
        # Decoder
        self.up3 = nn.ConvTranspose1d(base_channels * 8, base_channels * 4, 4, 2, 1)
        self.dec3 = self._block(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, 4, 2, 1)
        self.dec2 = self._block(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose1d(base_channels * 2, base_channels, 4, 2, 1)
        self.dec1 = self._block(base_channels * 2, base_channels)
        
        # Output
        self.out = nn.Conv1d(base_channels, out_channels, 1)
        
        self.pool = nn.MaxPool1d(2)
    
    def _block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)


# ============================================================================
# Dataset
# ============================================================================

class PTBXLReconDataset(Dataset):
    """PTB-XL dataset for reconstruction training."""
    
    def __init__(
        self,
        data_dir: str,
        database_csv: str,
        split: str = "train",
        input_leads: List[int] = [0, 1, 7],  # I, II, V2
        target_len: int = 5000
    ):
        self.data_dir = Path(data_dir)
        self.input_leads = input_leads
        self.target_len = target_len
        
        # Load database
        df = pd.read_csv(database_csv)
        
        # Filter by split
        if split == "train":
            self.df = df[~df['strat_fold'].isin([9, 10])]
        elif split == "val":
            self.df = df[df['strat_fold'] == 9]
        else:
            self.df = df[df['strat_fold'] == 10]
        
        self.df = self.df.reset_index(drop=True)
        logger.info(f"Loaded {len(self.df)} records for {split}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        ecg_id = row['ecg_id']
        
        # Load tensor
        tensor_path = self.data_dir / f"{ecg_id}.pt"
        
        if tensor_path.exists():
            signal = torch.load(tensor_path, weights_only=True)
        else:
            # Return zeros
            signal = torch.zeros(12, self.target_len)
        
        # Ensure shape
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        if signal.shape[1] != self.target_len:
            if signal.shape[1] < self.target_len:
                pad = torch.zeros(signal.shape[0], self.target_len - signal.shape[1])
                signal = torch.cat([signal, pad], dim=1)
            else:
                signal = signal[:, :self.target_len]
        
        # Normalize
        mean = signal.mean(dim=1, keepdim=True)
        std = signal.std(dim=1, keepdim=True) + 1e-6
        signal = (signal - mean) / std
        
        # Extract input leads
        input_signal = signal[self.input_leads, :]
        
        return {
            'input': input_signal,
            'target': signal,
            'ecg_id': ecg_id
        }


# ============================================================================
# Training
# ============================================================================

def train_model(
    config_name: str,
    lambda_mse: float,
    lambda_mmd: float,
    lambda_deriv: float,
    lambda_corr: float,
    data_dir: str,
    database_csv: str,
    output_dir: str,
    seed: int = 42,
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = None
) -> Dict:
    """Train a single ablation configuration."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {config_name}")
    logger.info(f"λ_mse={lambda_mse}, λ_mmd={lambda_mmd}, λ_deriv={lambda_deriv}, λ_corr={lambda_corr}")
    logger.info(f"{'='*60}")
    
    # Create datasets
    train_dataset = PTBXLReconDataset(data_dir, database_csv, split="train")
    val_dataset = PTBXLReconDataset(data_dir, database_csv, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = SimpleUNet1D(in_channels=3, out_channels=12).to(device)
    
    # Create loss
    criterion = CompositeLoss(
        lambda_mse=lambda_mse,
        lambda_mmd=lambda_mmd,
        lambda_deriv=lambda_deriv,
        lambda_corr=lambda_corr
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Mixed precision
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss, loss_dict = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss_dict['total'])
            pbar.set_postfix({'loss': np.mean(train_losses[-10:])})
        
        avg_train_loss = np.mean(train_losses)
        
        # Validate
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(inputs)
                loss, loss_dict = criterion(outputs, targets)
                val_losses.append(loss_dict['total'])
        
        avg_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        logger.info(f"Epoch {epoch}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': {
                    'lambda_mse': lambda_mse,
                    'lambda_mmd': lambda_mmd,
                    'lambda_deriv': lambda_deriv,
                    'lambda_corr': lambda_corr
                }
            }, output_path / f"{config_name}_seed{seed}.pt")
        
        scheduler.step(avg_val_loss)
    
    return {
        'config': config_name,
        'best_val_loss': best_val_loss,
        'final_train_loss': avg_train_loss,
        'epochs_trained': num_epochs,
        'history': history
    }


# ============================================================================
# Main
# ============================================================================

# ============================================================================
# Ablation Configurations
# ============================================================================
# 
# IMPORTANT: The M1-HPO weights were NOT arbitrary!
# They came from NSGA-II multi-objective optimization (see m1_multiobj_hpo.py)
#
# Search process:
#   - Method: NSGA-II with population=50, ~10 generations (500 trials)
#   - Objectives: Maximize Pearson, QRS-corr, ST-corr, AUROC
#   - Search space: λ_mmd ∈ [0.02, 0.15], λ_deriv ∈ [0.05, 0.2], λ_corr ∈ [0.05, 0.2]
#   - Constraint: Σλ ≤ 0.5
#
# The selected weights (Trial 5) achieved simultaneous improvement on ALL objectives
# over the MSE baseline - this is the key finding that justifies the multi-objective approach.
#
# Ablation study purpose: Understand which components contribute most to the improvement.
# ============================================================================

CONFIGS = {
    # Baseline: MSE only
    'M0': {
        'lambda_mse': 1.0, 
        'lambda_mmd': 0.0, 
        'lambda_deriv': 0.0, 
        'lambda_corr': 0.0,
        'description': 'Baseline: MSE only (mean-seeking, smooths peaks)'
    },
    
    # Single-component ablations
    'M0+MMD': {
        'lambda_mse': 1.0, 
        'lambda_mmd': 0.05,  # From Pareto HPO
        'lambda_deriv': 0.0, 
        'lambda_corr': 0.0,
        'description': '+MMD: Distributional alignment (prevents regression to mean)'
    },
    'M0+D': {
        'lambda_mse': 1.0, 
        'lambda_mmd': 0.0, 
        'lambda_deriv': 0.10,  # From Pareto HPO
        'lambda_corr': 0.0,
        'description': '+Derivative: Gradient preservation (preserves QRS slopes)'
    },
    'M0+C': {
        'lambda_mse': 1.0, 
        'lambda_mmd': 0.0, 
        'lambda_deriv': 0.0, 
        'lambda_corr': 0.08,  # From Pareto HPO
        'description': '+Correlation: Temporal coherence (maintains phase alignment)'
    },
    
    # Two-component ablations
    'M0+MD': {
        'lambda_mse': 1.0, 
        'lambda_mmd': 0.05, 
        'lambda_deriv': 0.10, 
        'lambda_corr': 0.0,
        'description': '+MMD+Deriv: Distribution + Gradient'
    },
    'M0+MC': {
        'lambda_mse': 1.0, 
        'lambda_mmd': 0.05, 
        'lambda_deriv': 0.0, 
        'lambda_corr': 0.08,
        'description': '+MMD+Corr: Distribution + Temporal'
    },
    'M0+DC': {
        'lambda_mse': 1.0, 
        'lambda_mmd': 0.0, 
        'lambda_deriv': 0.10, 
        'lambda_corr': 0.08,
        'description': '+Deriv+Corr: Gradient + Temporal'
    },
    
    # Full model (Pareto-optimal from NSGA-II Trial 5)
    'M1-HPO': {
        'lambda_mse': 1.0, 
        'lambda_mmd': 0.05,   # Pareto-optimal
        'lambda_deriv': 0.10, # Pareto-optimal
        'lambda_corr': 0.08,  # Pareto-optimal
        'description': 'Full multi-objective (Pareto-optimal from NSGA-II)'
    },
    
    # Naive equal weights (to show why HPO is necessary)
    'M1-Equal': {
        'lambda_mse': 1.0, 
        'lambda_mmd': 1.0, 
        'lambda_deriv': 1.0, 
        'lambda_corr': 1.0,
        'description': 'Equal weights (demonstrates destructive interference)'
    },
}


def main():
    parser = argparse.ArgumentParser(description="Train Ablation Models")
    parser.add_argument("--config", type=str, choices=list(CONFIGS.keys()),
                       help="Configuration to train")
    parser.add_argument("--all", action="store_true", help="Train all configurations")
    parser.add_argument("--data_dir", type=str, default="~/data/ptb_xl/tensors/train")
    parser.add_argument("--database_csv", type=str, default="~/data/ptbxl_database.csv")
    parser.add_argument("--output_dir", type=str, default="~/checkpoints/ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Expand paths
    args.data_dir = os.path.expanduser(args.data_dir)
    args.database_csv = os.path.expanduser(args.database_csv)
    args.output_dir = os.path.expanduser(args.output_dir)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if args.all:
        configs_to_train = list(CONFIGS.keys())
    elif args.config:
        configs_to_train = [args.config]
    else:
        logger.error("Specify --config or --all")
        return
    
    results = []
    
    for config_name in configs_to_train:
        config = CONFIGS[config_name]
        
        result = train_model(
            config_name=config_name,
            data_dir=args.data_dir,
            database_csv=args.database_csv,
            output_dir=args.output_dir,
            seed=args.seed,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            **config
        )
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"{r['config']}: best_val_loss={r['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
