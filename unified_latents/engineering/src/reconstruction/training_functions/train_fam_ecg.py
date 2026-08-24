#!/usr/bin/env python3
"""
Strict Mason VAE Training Orchestrator (Phase 44)
================================================================
Implements the generative reconstruction mission using the Strict Mason VAE
architecture bounded to the frozen ECG-FM backbone.

Mirroring the clean structure of train_cnvae.py.
"""

import os
import sys
sys.path.insert(0, '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/unified_latents/engineering')
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# Project root
sys.path.append(os.getcwd())

# Models, Data, and Mason Protocol -- all centralized in fam_ecg.py
from src.reconstruction.learn_functions.fam_ecg import (
    VAEMasonECGFMBridge,
    MASON_INPUT_LIMB_V3,
    PRECORDIAL_INDICES,
    denormalize_mason,
    batch_r2_function
)
from src.reconstruction.load_functions.ptbxl_dataset import PTBXLDataset
import wandb

# === SIGNAL SCALING PIPELINE ===
# PTBXLDataset(use_mason_scaling=True) -> [0, 1] (Mason normalized)
# denormalize_mason(x_norm) -> raw mV (approx -2.5 to +2.5 mV)
# bridge._masked_zscore -> per-lead z-score -> ECG-FM encoder (FP32)
# bridge decoder output -> unbounded raw mV units
# Targets y -> raw mV (via denormalize_mason)
# Loss: Negative R2 (scale-invariant) against raw mV targets
# ============================================================

def get_model(device, target_len=5000):
    """Factory function to instantiate the Strict Mason VAE bridge."""
    print("Initializing Strict Mason VAE Generative World Model Bridge (Phase 44)...")
    model = VAEMasonECGFMBridge(
        checkpoint_path="ecg_fm_integration/checkpoints/mimic_iv_ecg_physionet_pretrained.pt",
        freeze_encoder=True,
        target_len=target_len
    )
    return model.to(device)

def train(args):
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"ecgfm_phase44_strict_{timestamp}"
    save_dir = os.path.abspath(os.path.join(args.save_dir, run_name))
    os.makedirs(save_dir, exist_ok=True)
    print(f"    Save dir: {save_dir}")

    # Dump args
    with open(os.path.join(save_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Initialize WandB
    wandb.init(
        project="Generative-World-Model-Phase44-Strict",
        name=run_name,
        config=vars(args),
        tags=["generative", "phase44", "strict"]
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training Strict Mason VAE (Phase 44) on {device} ---")
    print(f"    Input leads: I, II, V3 (indices {MASON_INPUT_LIMB_V3})")

    # 1. Data Loading
    print("Loading PTB-XL...")
    ptbxl_root = "/home/mithunmanivannan/data/ptb_xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
    ptbxl_csv = os.path.join(ptbxl_root, "ptbxl_database.csv")
    train_ds = PTBXLDataset(ptbxl_root, ptbxl_csv, split='train', target_fs=500, use_mason_scaling=True)
    val_ds = PTBXLDataset(ptbxl_root, ptbxl_csv, split='val', target_fs=500, use_mason_scaling=True)

    kw = dict(batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    if args.num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    # 2. Model Initialization
    bridge = get_model(device, target_len=args.target_len)

    # 3. Optimization
    trainable_params = [p for p in bridge.parameters() if p.requires_grad]
    # Strict Mason Parity: Adam with exact literature hyperparameters
    optimizer = optim.Adam(trainable_params, lr=3e-6, weight_decay=1e-3, eps=0.01)
    
    # LR Warmup
    warmup_epochs = 5
    base_lr = args.lr
    initial_lr = 1e-6
    
    # Loss: Mason Negative R² reconstruction + internalized KL (betas set in bridge)
    scaler = torch.amp.GradScaler('cuda')

    # 4. Training Loop
    best_val_score = float('inf')
    best_epoch = -1
    patience_timer = args.patience

    print(f"=== TRAINING STARTING (Batch Size: {args.batch_size}, Device: {device}) ===")
    _lead_idx_base = torch.tensor(MASON_INPUT_LIMB_V3, device=device)

    for epoch in range(args.epochs):
        bridge.train()
        total_metrics = {"loss": 0, "recon": 0, "kl": 0, "r2": 0}
        
        # LR Warmup Logic
        current_lr = args.lr
        if epoch < warmup_epochs:
            current_lr = initial_lr + (base_lr - initial_lr) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, batch in enumerate(pbar):
            if args.max_steps > 0 and batch_idx >= args.max_steps: break
            
            x_norm, target_norm, _ = batch
            x_norm, target_norm = x_norm.to(device, non_blocking=True), target_norm.to(device, non_blocking=True)
            x = denormalize_mason(x_norm) if getattr(train_loader.dataset, 'output_units', 'mV') == 'mason' else x_norm
            y = denormalize_mason(target_norm)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                lead_indices = _lead_idx_base.unsqueeze(0).expand(x.size(0), -1)
                recon, kl = bridge(x, lead_indices=lead_indices)

                recon_precord = recon[:, PRECORDIAL_INDICES, :].float()
                y_precord = y[:, PRECORDIAL_INDICES, :].float()

                # Reconstruction: Negative R² (Mason parity)
                recon_loss, _ = batch_r2_function(
                    recon_precord.unbind(dim=1), y_precord.unbind(dim=1), 6, x.size(0), False
                )
                loss = recon_loss + kl

            scaler.scale(loss / args.gradient_accumulation_steps).backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Metrics
            loss_r2, _ = batch_r2_function(recon_precord.unbind(dim=1), y_precord.unbind(dim=1), 6, x.size(0), False)
            total_metrics["loss"] += loss.item()
            total_metrics["recon"] += recon_loss.item()
            total_metrics["kl"] += kl.item()
            total_metrics["r2"] += -loss_r2.item()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "KL": f"{kl.item():.4f}", "R2": f"{-loss_r2.item():.4f}"})

        # Validation
        bridge.eval()
        val_metrics = {"loss": 0, "r2": 0}
        with torch.no_grad():
            for batch in val_loader:
                x_norm, target_norm, _ = batch
                x_norm, target_norm = x_norm.to(device, non_blocking=True), target_norm.to(device, non_blocking=True)
                x = denormalize_mason(x_norm) if getattr(val_loader.dataset, 'output_units', 'mV') == 'mason' else x_norm
                y = denormalize_mason(target_norm)

                recon = bridge(x, lead_indices=_lead_idx_base.unsqueeze(0).expand(x.size(0), -1))
                recon_precord = recon[:, PRECORDIAL_INDICES, :].float()
                y_precord = y[:, PRECORDIAL_INDICES, :].float()
                
                v_loss_r2, _ = batch_r2_function(recon_precord.unbind(dim=1), y_precord.unbind(dim=1), 6, x.size(0), False)
                val_metrics["loss"] += v_loss_r2.item()
                val_metrics["r2"] += -v_loss_r2.item()

        avg_val_r2 = val_metrics["r2"] / len(val_loader)
        avg_val_loss = val_metrics["loss"] / len(val_loader)
        print(f"Epoch {epoch+1}: Val Loss={avg_val_loss:.4f}, Val R2={avg_val_r2:.4f}")

        wandb.log({
            "epoch": epoch,
            "train/loss": total_metrics["loss"] / len(train_loader),
            "train/kl": total_metrics["kl"] / len(train_loader),
            "train/r2": total_metrics["r2"] / len(train_loader),
            "val/loss": avg_val_loss,
            "val/r2": avg_val_r2,
            "lr": current_lr
        })

        if avg_val_loss < best_val_score:
            best_val_score = avg_val_loss
            best_epoch = epoch
            patience_timer = args.patience
            torch.save({'bridge_state_dict': bridge.state_dict(), 'val_r2': avg_val_r2}, os.path.join(save_dir, "best_model.pt"))
            print(f"  Saved Best Model (R2: {avg_val_r2:.4f})")
        else:
            patience_timer -= 1
            if patience_timer <= 0: break

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict Mason VAE Training (Phase 44)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints/phase44")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--target_len", type=int, default=5000)
    args = parser.parse_args()
    train(args)
