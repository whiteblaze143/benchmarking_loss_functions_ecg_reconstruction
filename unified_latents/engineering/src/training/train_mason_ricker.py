#!/usr/bin/env python
"""
Mason-Ricker (NIWT) Training Script

Trains the Hybrid Mason Encoder + Ricker Wavelet Decoder model.
Uses Composite Loss (MSE + STFT + BandPower) for spectral fidelity.
"""

import argparse
import json
import logging
import os
import sys
sys.path.insert(0, '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/unified_latents/engineering')
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.getcwd())

from src.data.multi_source_dataset import MultiSourceECGDataset

from src.models.mason_ricker import MasonRicker

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


# ==================================================================================
# Advanced Loss Functions (Matched to main.ipynb NIWT specs)
# ==================================================================================

class STFTLoss(nn.Module):
    """Multi-resolution STFT loss for spectral fidelity."""
    def __init__(self, n_fft_list: Optional[List[int]] = None, norm: str = 'l1'):
        super().__init__()
        self.n_fft_list = n_fft_list or [256, 512]  # For 500Hz ECG
        self.norm = norm
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            b, l, s = pred.shape
            pred = pred.reshape(-1, s)
            gt = gt.reshape(-1, s)
        
        total_loss = 0.0
        
        # Force FP32 for STFT stability
        with autocast(enabled=False):
            pred = pred.float()
            gt = gt.float()
            
            for n_fft in self.n_fft_list:
                hop = n_fft // 4
                window = torch.hann_window(n_fft, device=pred.device, dtype=torch.float32)
                
                try:
                    stft_p = torch.stft(pred, n_fft, hop_length=hop, window=window, return_complex=True, center=False)
                    stft_g = torch.stft(gt, n_fft, hop_length=hop, window=window, return_complex=True, center=False)
                    
                    mag_p = torch.abs(stft_p) + 1e-8
                    mag_g = torch.abs(stft_g) + 1e-8
                    
                    if self.norm == 'l1':
                        total_loss += torch.mean(torch.abs(mag_p - mag_g))
                    else:
                        total_loss += torch.mean((mag_p - mag_g) ** 2)
                except Exception:
                    pass
        
        return total_loss

class BandPowerLoss(nn.Module):
    """Frequency band power matching."""
    def __init__(self, fs: int = 500, bands: Optional[List[Tuple[float, float]]] = None):
        super().__init__()
        self.fs = fs
        self.bands = bands or [(0, 5), (5, 40), (40, 100)]  # T-wave, QRS, Noise
        self.n_fft = 512
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            b, l, s = pred.shape
            pred = pred.reshape(-1, s)
            gt = gt.reshape(-1, s)
        
        with autocast(enabled=False):
            p_float = pred.float()
            g_float = gt.float()
            
            window = torch.hann_window(self.n_fft, device=pred.device, dtype=torch.float32)
            hop = self.n_fft // 2
            
            stft_p = torch.stft(p_float, self.n_fft, hop_length=hop, window=window, return_complex=True, center=False)
            stft_g = torch.stft(g_float, self.n_fft, hop_length=hop, window=window, return_complex=True, center=False)
            
            psd_p = torch.mean(torch.abs(stft_p)**2, dim=-1)
            psd_g = torch.mean(torch.abs(stft_g)**2, dim=-1)
            
            freqs = torch.linspace(0, self.fs/2, psd_p.shape[1], device=pred.device)
            
            total_loss = 0.0
            for (f_low, f_high) in self.bands:
                mask = (freqs >= f_low) & (freqs <= f_high)
                if mask.sum() == 0:
                    continue
                
                band_p = torch.mean(psd_p[:, mask], dim=1) + 1e-8
                band_g = torch.mean(psd_g[:, mask], dim=1) + 1e-8
                
                total_loss += torch.mean(torch.abs(torch.log(band_p) - torch.log(band_g)))
        
        return total_loss

class CompositeLoss(nn.Module):
    """Combined loss: MSE + alpha*STFT + beta*BandPower"""
    def __init__(self, mse_weight=1.0, stft_weight=0.1, band_weight=0.05):
        super().__init__()
        self.mse_weight = mse_weight
        self.stft_weight = stft_weight
        self.band_weight = band_weight
        
        self.mse = nn.MSELoss()
        self.stft = STFTLoss()
        self.band = BandPowerLoss()
    
    def forward(self, pred, gt):
        losses = {'mse': self.mse(pred, gt)}
        total = self.mse_weight * losses['mse']
        
        if self.stft_weight > 0:
            losses['stft'] = self.stft(pred, gt)
            total = total + self.stft_weight * losses['stft']
            
        if self.band_weight > 0:
            losses['band'] = self.band(pred, gt)
            total = total + self.band_weight * losses['band']
            
        return total, losses


# ==================================================================================
# Training Logic
# ==================================================================================

def pearson_corr(x, y):
    """Pearson correlation coefficient."""
    vx = x - torch.mean(x, dim=-1, keepdim=True)
    vy = y - torch.mean(y, dim=-1, keepdim=True)
    cov = torch.sum(vx * vy, dim=-1)
    corr = cov / (torch.sqrt(torch.sum(vx ** 2, dim=-1)) * torch.sqrt(torch.sum(vy ** 2, dim=-1)) + 1e-8)
    return corr.mean()


def train_epoch(model, loader, optimizer, criterion, scaler, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    total_corr = 0
    
    # Track components
    log_losses = {'mse': 0.0, 'stft': 0.0, 'band': 0.0}
    
    for batch in tqdm(loader, desc="Training"):
        x = batch['input'].to(device)
        y = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            pred = model(x)
            loss, components = criterion(pred, y)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        for k, v in components.items():
            if k in log_losses:
                log_losses[k] += v.item()
                
        with torch.no_grad():
            total_corr += pearson_corr(pred, y).item()
    
    avg_loss = total_loss / len(loader)
    avg_corr = total_corr / len(loader)
    avg_components = {k: v / len(loader) for k, v in log_losses.items()}
    
    return avg_loss, avg_corr, avg_components


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_corr = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            x = batch['input'].to(device)
            y = batch['target'].to(device)
            
            with autocast():
                pred = model(x)
                loss, _ = criterion(pred, y)
            
            total_loss += loss.item()
            total_corr += pearson_corr(pred, y).item()
    
    return total_loss / len(loader), total_corr / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_mason_ricker.json')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = json.load(f)
    else:
        LOGGER.warning(f"Config {args.config} not found! Using defaults.")
        cfg = {}

    batch_size = cfg.get('training', {}).get('batch_size', 64)
    epochs = cfg.get('training', {}).get('epochs', 50)
    lr = cfg.get('training', {}).get('learning_rate', 1e-3)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"Using device: {device}")
    
    # Create datasets
    ptb_source = [{"name": "PTB-XL", "path": "data/ptbxl_tensors", "format": "pt"}]
    # Note: Seq len 5000 is standard
    train_ds = MultiSourceECGDataset(split='train', sources=ptb_source, target_len=5000, normalization='min_max')
    val_ds = MultiSourceECGDataset(split='val', sources=ptb_source, target_len=5000, normalization='min_max')
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    
    # Model Config
    model_cfg = cfg.get('model', {})
    model = MasonRicker(
        in_leads=model_cfg.get('input_leads', 3),
        out_leads=12,
        seq_len=5000,
        num_wavelets=model_cfg.get('num_wavelets', 512),
        ricker_hidden_dim=model_cfg.get('hidden_dim', 2048),
        sigma_min=model_cfg.get('sigma_min', 0.02),
        sigma_max=model_cfg.get('sigma_max', 0.5)
    ).to(device)

    
    LOGGER.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss Config
    loss_cfg = cfg.get('loss', {})
    criterion = CompositeLoss(
        mse_weight=loss_cfg.get('mse_weight', 1.0),
        stft_weight=loss_cfg.get('stft_weight', 0.1),
        band_weight=loss_cfg.get('band_weight', 0.05)
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=cfg.get('training', {}).get('weight_decay', 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    # Create save directory
    save_dir = Path(cfg.get('save_dir', 'checkpoints/mason_ricker_optimal'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_corr = -1.0
    start_epoch = 0
    
    LOGGER.info("Starting Training...")
    
    for epoch in range(start_epoch, epochs):
        train_loss, train_corr, components = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_corr = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        # Log metrics
        log_msg = f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Pearson: {val_corr:.4f}"
        log_msg += f" | STFT: {components['stft']:.6f}"
        LOGGER.info(log_msg)
        
        # Save last
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'val_corr': val_corr
        }, save_dir / 'last_mason_ricker.pt')
        
        # Save best
        if val_corr > best_corr:
            best_corr = val_corr
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_corr': val_corr
            }, save_dir / 'best_mason_ricker.pt')
            LOGGER.info(f"Saved Best Model (Pearson {best_corr:.4f})")
    
    LOGGER.info("Training Complete!")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        LOGGER.exception("Training failed with exception:")
        sys.exit(1)
