#!/usr/bin/env python
"""
Mason Reconstructor Training Script (Full Architecture)

Trains the Mason CNN-based ECG reconstruction model from 3-lead to 12-lead.
Architecture matches Mason et al. (Input Networks -> Fusion -> Output Networks).
"""

import argparse
import json
import logging
import os
import sys
sys.path.insert(0, '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/unified_latents/engineering')
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.getcwd())

from src.data.multi_source_dataset import MultiSourceECGDataset

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


# ==================================================================================
# Mason Architecture Components (Matching src/models/mason_ricker.py style)
# ==================================================================================

class ConvBlock(nn.Module):
    """Single convolutional block with optional residual connection."""
    def __init__(self, in_ch, out_ch, kernel_size=17, use_residual=True, activation='elu'):
        super().__init__()
        padding = kernel_size // 2
        self.use_residual = use_residual and (in_ch == out_ch)
        
        self.conv_in = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn_in = nn.BatchNorm1d(out_ch)
        self.conv_out = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding)
        self.bn_out = nn.BatchNorm1d(out_ch)
        
        if activation == 'elu':
            self.act = nn.ELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
    
    def forward(self, x):
        identity = x
        out = self.act(self.bn_in(self.conv_in(x)))
        out = self.bn_out(self.conv_out(out))
        if self.use_residual:
            out = out + identity
        return self.act(out)


class ConvNetwork(nn.Module):
    """Sequential convolutional network."""
    def __init__(self, in_ch, out_ch, depth=3, kernel_size=17, use_residual=True):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(depth - 1):
            next_ch = int((ch + out_ch) / 2)
            layers.append(ConvBlock(ch, next_ch, kernel_size, use_residual))
            ch = next_ch
        layers.append(ConvBlock(ch, out_ch, kernel_size, use_residual))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class MasonReconstructor(nn.Module):
    """
    Full Mason et al. architecture.
    1. Input Networks: 3 parallel networks encode 3 leads.
    2. Fusion: Concatenate -> Encoder -> Decoder.
    3. Output Networks: 12 parallel networks decode to 12 leads.
    """
    def __init__(self, 
                 in_leads=3, 
                 out_leads=12,
                 seq_len=5000,
                 input_channel=32,
                 middle_channel=32, 
                 input_depth=3,
                 middle_depth=2,
                 output_depth=3,
                 kernel_size=17,
                 use_residual=True):
        super().__init__()
        
        self.in_leads = in_leads
        self.out_leads = out_leads
        
        # Stage 1: Per-lead input encoding
        self.input_networks = nn.ModuleList([
            ConvNetwork(1, input_channel, input_depth, kernel_size, use_residual)
            for _ in range(in_leads)
        ])
        
        # Stage 2: Cross-lead fusion
        fusion_in = input_channel * in_leads
        fusion_mid = middle_channel * int((in_leads + out_leads) / 2)
        fusion_out = 32 * out_leads  # Matches checkpoint architecture (32 channels per output lead)
        
        self.middle_encoder = ConvNetwork(fusion_in, fusion_mid, middle_depth // 2 or 1, kernel_size, use_residual)
        self.middle_decoder = ConvNetwork(fusion_mid, fusion_out, middle_depth // 2 or 1, kernel_size, use_residual)
        
        # Stage 3: Per-lead output decoding
        # Each output lead has its own network taking the fused representation
        self.output_networks = nn.ModuleList([
            ConvNetwork(fusion_out, 1, output_depth, kernel_size, use_residual)
            for _ in range(out_leads)
        ])
        
    def forward(self, x):
        """
        Args:
            x: (B, in_leads, T)
        Returns:
            out: (B, out_leads, T)
        """
        B, L, T = x.shape
        
        # Stage 1: Encode each input lead
        encoded = []
        for i in range(self.in_leads):
            lead_in = x[:, i:i+1, :]  # (B, 1, T)
            encoded.append(self.input_networks[i](lead_in))
        
        # Concatenate: (B, input_channel * in_leads, T)
        fused = torch.cat(encoded, dim=1)
        
        # Stage 2: Cross-lead fusion
        fused = self.middle_encoder(fused)
        fused = self.middle_decoder(fused)  # (B, fusion_mid, T)
        
        # Stage 3: Decode each output lead
        outputs = []
        for i in range(self.out_leads):
            out_lead = self.output_networks[i](fused)  # (B, 1, T)
            outputs.append(out_lead)
            
        return torch.cat(outputs, dim=1)


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


def train_epoch(model, loader, optimizer, scaler, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    total_corr = 0
    
    for batch in tqdm(loader, desc="Training"):
        x = batch['input'].to(device)
        y = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            pred = model(x)
            loss = F.mse_loss(pred, y)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        with torch.no_grad():
            total_corr += pearson_corr(pred, y).item()
    
    return total_loss / len(loader), total_corr / len(loader)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_corr = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            x = batch['input'].to(device)
            y = batch['target'].to(device)
            
            with autocast():
                pred = model(x)
                loss = F.mse_loss(pred, y)
            
            total_loss += loss.item()
            total_corr += pearson_corr(pred, y).item()
    
    return total_loss / len(loader), total_corr / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_mason.json')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = json.load(f)
    else:
        cfg = {
            "epochs": 50,
            "batch_size": 32, # Lower batch size for complex model
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "save_dir": "checkpoints/mason_baseline"
        }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"Using device: {device}")
    
    # Create datasets
    ptb_source = [{"name": "PTB-XL", "path": "data/ptbxl_tensors", "format": "pt"}]
    train_ds = MultiSourceECGDataset(split='train', sources=ptb_source, target_len=5000, normalization='min_max')
    val_ds = MultiSourceECGDataset(split='val', sources=ptb_source, target_len=5000, normalization='min_max')
    
    train_loader = DataLoader(train_ds, batch_size=cfg.get('batch_size', 32), shuffle=True, 
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get('batch_size', 32), shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    
    LOGGER.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model (Full Architecture)
    model = MasonReconstructor(in_leads=3, out_leads=12).to(device)
    LOGGER.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Resume if checkpoint provided
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        LOGGER.info(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Strip 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            start_epoch = checkpoint.get('epoch', 0)
        else:
            model.load_state_dict(checkpoint)
        LOGGER.info(f"Resumed from epoch {start_epoch}")
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get('learning_rate', 1e-3),
                                   weight_decay=cfg.get('weight_decay', 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.get('epochs', 50))
    scaler = GradScaler()
    
    # Restore optimizer state if available
    if args.resume and os.path.exists(args.resume) and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            LOGGER.info("Optimizer state restored")
        except Exception as e:
            LOGGER.warning(f"Could not restore optimizer state: {e}")
            
    # Create save directory
    save_dir = Path(cfg.get('save_dir', 'checkpoints/mason_baseline'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_corr = -1.0
    
    LOGGER.info("Starting Training...")
    
    for epoch in range(start_epoch, cfg.get('epochs', 50)):
        train_loss, train_corr = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_corr = evaluate(model, val_loader, device)
        scheduler.step()
        
        LOGGER.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Train Pearson: {train_corr:.4f}")
        LOGGER.info(f"Epoch {epoch+1} | Val Loss: {val_loss:.6f} | Val Pearson: {val_corr:.4f}")
        
        # Save last
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_corr': val_corr
        }, save_dir / 'last_mason.pt')
        
        # Save best (Pearson driven)
        if val_corr > best_corr:
            best_corr = val_corr
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_corr': val_corr
            }, save_dir / 'best_mason.pt')
            LOGGER.info(f"Saved Best Model (Pearson {best_corr:.4f})")
    
    LOGGER.info("Training Complete!")
    LOGGER.info(f"Best Val Pearson: {best_corr:.4f}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        LOGGER.exception("Training failed with exception:")
        sys.exit(1)
