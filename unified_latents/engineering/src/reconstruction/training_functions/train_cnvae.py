#!/usr/bin/env python3
"""
cNVAE-ECG Generative World Model Training Script (Phase 45)
================================================================
Implements the generative reconstruction mission using the Top-Down Hierarchical
cNVAE architecture bounded to the frozen ECG-FM backbone.

Follows Mason et al. (2024) split protocol adapted for generative modeling.
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

# Models, Data, and Mason Protocol -- all centralized in cnvae_ecg.py
from src.reconstruction.learn_functions.cnvae_ecg import (
    ECGConvECGFMBridge,
    ECG_INPUT_LIMB_V3,
    PRECORDIAL_INDICES,
    denormalize_ecg,
    batch_r2_function
)
from src.reconstruction.load_functions.ptbxl_dataset import PTBXLDataset
import wandb
import matplotlib.pyplot as plt

# --- Debugging Helpers ---
def tstats(name, x, max_print=5):
    if x is None:
        print(f"[{name}] is None")
        return
    x_det = x.detach()
    finite = torch.isfinite(x_det)
    fin_ratio = finite.float().mean().item()
    msg = (
        f"[{name}] shape={tuple(x_det.shape)} dtype={x_det.dtype} device={x_det.device} "
        f"finite%={100*fin_ratio:.2f}"
    )
    print(msg)
    if fin_ratio > 0:
        xf = x_det[finite]
        print(
            f"  min={xf.min().item():.6g} max={xf.max().item():.6g} "
            f"mean={xf.mean().item():.6g} std={xf.std().item():.6g}"
        )
        flat = xf.flatten()
        if flat.numel() > 0:
            idx = torch.linspace(0, flat.numel()-1, steps=min(max_print, flat.numel())).long()
            print(f"  samples={flat[idx].cpu().tolist()}")

def assert_finite(name, x):
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).float().mean().item()
        raise RuntimeError(f"{name} contains NaN/Inf. bad_fraction={bad:.6f}")

# === SIGNAL SCALING PIPELINE ===
# PTBXLDataset(use_mason_scaling=True) -> [0, 1] (Mason normalized)
# denormalize_ecg(x_norm) -> raw mV (approx -2.5 to +2.5 mV)
# bridge._masked_zscore -> per-lead z-score -> ECG-FM encoder (FP32)
# Loss: Negative R2 (scale-invariant) against raw mV targets
# ============================================================

def r2_elbo_loss(recon, target, kl_scales, kl_balanced, T_actual, n_leads=6, 
                 beta=1.0, gamma_per_scale=None):
    """
    Proper ELBO using R² as the reconstruction likelihood (Pass 12).
    
    Derivation: 
    log p(y | z) = (T/2) * sum(R²) + const
    Loss = -(T/2) * sum(R²) + beta * KL_balanced
    """
    # 1. Reconstruction: Per-lead R² scaled by T/2
    # recon: (B, L, T), target: (B, L, T)
    ssr = ((recon - target) ** 2).sum(dim=2) # (B, L)
    sst = ((target - target.mean(dim=2, keepdim=True)) ** 2).sum(dim=2) # (B, L)
    # Use a robust floor for SST (approx 2uV RMS noise floor) to avoid blowouts
    r2_per_lead = 1.0 - ssr / sst.clamp(min=1e-2) # (B, L)
    # CLIP REMOVED (Phase 27): Clipping -20.0 killed gradients early on.
    r2_per_lead = torch.clamp(r2_per_lead, min=-200.0, max=1.0)
    
    # Absolute Parity: Average over time and leads instead of summing.
    # This reduces loss magnitude from ~15,000 to ~1.0
    recon_nll = (1.0 - r2_per_lead).mean(dim=1) # (B,)
    
    # 2. KL: Already balanced by the decoder if kl_balance=True
    # 3. ELBO (negative, to minimize)
    # Theoretically proper ELBO: beta=1.0
    neg_elbo = recon_nll + beta * kl_balanced # (B,)
    loss = neg_elbo.mean()
    
    metrics = {
        "r2_mean_batch": r2_per_lead.mean().item(),
        "recon_nll": recon_nll.mean().item(),
        "kl_total": kl_scales.sum(dim=1).mean().item(),
        "neg_elbo": loss.item()
    }
    return loss, metrics

def aux_r2_loss(aux_signals, target, target_indices, aux_weight=0.1):
    """
    Auxiliary multi-scale R² loss (Likelihood units).
    """
    total_aux = 0.0
    target_precordial = target[:, target_indices, :]
    for s_idx, aux_s in enumerate(aux_signals):
        # Downsample target to match aux resolution
        T_s = aux_s.shape[-1]
        target_ds = F.adaptive_avg_pool1d(target_precordial, T_s)
        
        ssr = ((aux_s - target_ds) ** 2).sum(dim=2)
        sst = ((target_ds - target_ds.mean(dim=2, keepdim=True)) ** 2).sum(dim=2)
        r2_s = 1.0 - ssr / sst.clamp(min=1e-7)
        
        # Absolute Parity: Average dimensionless R2
        scale_weight = aux_weight * (0.5 ** (len(aux_signals) - 1 - s_idx))
        total_aux += scale_weight * (1.0 - r2_s).mean()
    
    return total_aux

def get_model(backbone_name, device, target_len=5000):
    """Factory function to instantiate the correct bridge."""
    if backbone_name == 'ecgfm':
        # Phase 45: cNVAE-ECG option 2 top-down hierarchy
        print("Initializing cNVAE Generative World Model Bridge (Phase 45)...")
        model = ECGConvECGFMBridge(
            checkpoint_path="ecg_fm_integration/checkpoints/mimic_iv_ecg_physionet_pretrained.pt",
            freeze_encoder=True,
            target_len=target_len,
            n_output_leads=6
        ).to(dtype=torch.float64)
        
        # Absolute Parity: Frozen backbone must stay in Float32 (mismatch otherwise)
        model.backbone.to(dtype=torch.float32)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    return model.to(device)

def train(args):
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.backbone}_phase45_{timestamp}"
    save_dir = os.path.abspath(os.path.join(args.save_dir, run_name))
    os.makedirs(save_dir, exist_ok=True)
    print(f"    Save dir: {save_dir}")

    # Dump args
    with open(os.path.join(save_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Initialize WandB
    wandb.init(
        project="cNVAE-ECG-Scientific-Production",
        name=run_name,
        config=vars(args),
        tags=["scientific", "production", "ELBO", args.backbone]
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training {args.backbone} cNVAE Generative Bridge (Phase 45) on {device} ---")
    print(f"    Input leads: I, II, V3 (indices {ECG_INPUT_LIMB_V3})")
    print(f"    Loss: Exact Mason Batch R² (from third_party)")

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
    bridge = get_model(args.backbone, device, target_len=args.target_len)

    # 3. Optimization
    trainable_params = [p for p in bridge.parameters() if p.requires_grad]
    # Phase 28: Adamax Optimizer (Reference Parity)
    optimizer = optim.Adamax(trainable_params, lr=args.lr, weight_decay=1e-3, eps=1e-3)
    
    # Cosine Annealing with Warmup
    warmup_epochs = 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
    
    # --- Checkpoint Resume (Production Continuity) ---
    start_epoch = args.start_epoch
    if args.resume_path and os.path.exists(args.resume_path):
        print(f"Loading checkpoint from {args.resume_path}...")
        # L3: Set weights_only=False to allow argparse.Namespace pickling
        ckpt = torch.load(args.resume_path, map_location=device, weights_only=False)
        if 'bridge_state_dict' in ckpt:
            bridge.load_state_dict(ckpt['bridge_state_dict'], strict=False)
        elif 'model_state_dict' in ckpt:
            bridge.load_state_dict(ckpt['model_state_dict'], strict=False)
        
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        print("!!! DEBUG MODE ENABLED: Anomaly Detection Active !!!")

    DEBUG = args.debug
    DEBUG_STEPS = args.debug_steps
    warmup_epochs = 5
    base_lr = args.lr
    initial_lr = 1e-6
    
    # Loss functions: Principled ELBO (Pass 12)
    # The reconstruction term is scaled by T/2 (~2500 per lead), providing
    # natural balancing. beta=1.0 is the theoretically proper ELBO.
    beta = 1.0

    # 4. Training Loop
    # Define Generative Targets (leads the model must infer, avoiding metric inflation)
    # Target indices in the 12-lead set that are PRECORDIAL but NOT in the input
    gen_target_indices = [idx for idx in PRECORDIAL_INDICES if idx not in ECG_INPUT_LIMB_V3]
    # Map to 0-5 index for the decoder output
    gen_output_indices = [idx - 6 for idx in gen_target_indices]
    print(f"    Generative Targets: {[f'V{idx-5}' for idx in gen_target_indices]} (Leads expected to be inferred)")

    best_val_score = float('inf')
    best_val_r2 = -float('inf')
    best_epoch = -1
    patience_timer = args.patience

    print(f"=== TRAINING STARTING (Batch Size: {args.batch_size}, Device: {device}) ===")
    torch.backends.cudnn.benchmark = True # O3: Faster fixed-length convolutions
    _lead_idx_base = torch.tensor(ECG_INPUT_LIMB_V3, device=device)

    for epoch in range(start_epoch, args.epochs):
        bridge.train()
        total_loss = 0
        total_r2_loss = 0
        total_recon_loss = 0
        total_kl_global = 0
        n_train_batches = 0
        
        # LR Warmup & Scheduler Logic (Reference Parity)
        if epoch < warmup_epochs:
            current_lr = initial_lr + (base_lr - initial_lr) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
        wandb.log({"train/lr": current_lr, "epoch": epoch})

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, batch in enumerate(pbar):
            if args.max_steps > 0 and batch_idx >= args.max_steps:
                break
            
            x_norm, target_norm, _ = batch
            # Absolute Parity: Train in Float64 for deep hierarchical stability
            x_norm = x_norm.to(device, dtype=torch.float64, non_blocking=True)
            target_norm = target_norm.to(device, dtype=torch.float64, non_blocking=True)
            
            x = denormalize_ecg(x_norm) if getattr(train_loader.dataset, 'output_units', 'mV') == 'mason' else x_norm
            y = denormalize_ecg(target_norm)

            if DEBUG and batch_idx < DEBUG_STEPS:
                print(f"\n--- DEBUG Step {batch_idx} ---")
                tstats("x_raw", x)
                tstats("y_raw", y)

            # K2: Zero grads only at the start of accumulation window
            if batch_idx % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            # Forward pass — P2: bridge returns a dictionary
            # Standard input basis: Leads I (0), III (2), V3 (8)
            lead_indices = torch.tensor([ECG_INPUT_LIMB_V3], device=device).expand(x.size(0), -1)
            
            # Absolute Parity: No AMP, use Float64 directly
            outputs = bridge(x, lead_indices=lead_indices, target_len=args.target_len, kl_balance=True)
            recon = outputs["recon"]
            kl_scales = outputs["kl"]
            kl_balanced = outputs["kl_balanced"]
            aux_signals = outputs.get("aux")
            
            # --- Reconstruction Loss: Principled R2-ELBO (Pass 12/14) ---
            # Cast to float64 to match model
            y_prec = y[:, PRECORDIAL_INDICES, :].to(torch.float64)
            r_prec = recon[:, PRECORDIAL_INDICES, :].to(torch.float64) 

            # Calculate primary ELBO loss
            beta_warmup = min(beta, beta * (epoch / max(warmup_epochs, 1)))
            loss, elbo_metrics = r2_elbo_loss(
                r_prec, y_prec, kl_scales, kl_balanced,
                T_actual=args.target_len, n_leads=6, beta=beta_warmup
            )
            
            # --- Multiscale Supv Loss (N6/P1) ---
            if aux_signals is not None:
                loss += aux_r2_loss(aux_signals, y, PRECORDIAL_INDICES, aux_weight=0.1)

            # --- Spectral & Regularization Loss (O4: Fast Caching) ---
            sr_loss = bridge.decoder.spectral_norm_parallel()
            bn_loss = bridge.decoder.batchnorm_loss()
            loss += 1e-2 * sr_loss + 1e-2 * bn_loss
            
            # --- Finiteness Guards (Trapping Epoch 2 Crash) ---
            assert_finite("recon", recon)
            assert_finite("kl_scales", kl_scales)

            # --- Final Loss Guard ---
            assert_finite("loss", loss)
            
            # K2: Scale loss by accumulation steps
            (loss / args.gradient_accumulation_steps).backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Accumulate metrics
            total_loss += loss.item()
            total_r2_loss += elbo_metrics["r2_mean_batch"]
            total_recon_loss += elbo_metrics["recon_nll"]
            total_kl_global += elbo_metrics["kl_total"]
            n_train_batches += 1
            
            if (batch_idx + 1) % 10 == 0: # Log every 10 steps to wandb
                step_log = {
                    "train/step_loss": loss.item(),
                    "train/step_kl": elbo_metrics["kl_total"],
                    "train/step_r2": elbo_metrics["r2_mean_batch"],
                    "train/step_recon_nll": elbo_metrics["recon_nll"],
                    "train/sr_loss": sr_loss.item(),
                    "train/bn_loss": bn_loss.item(),
                    "train/beta_warmup": beta_warmup
                }
                for i in range(kl_scales.shape[1]):
                    step_log[f"train/kl_scale_{i}"] = kl_scales[:, i].mean().item()
                wandb.log(step_log)

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "NLL": f"{elbo_metrics['recon_nll']:.2f}", "R2": f"{elbo_metrics['r2_mean_batch']:.4f}"})

        # L3: Handle final partial accumulation window
        if (batch_idx + 1) % args.gradient_accumulation_steps != 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Validation
        bridge.eval()
        total_val_loss, total_val_r2_loss, total_val_recon_loss, total_val_kl_global = 0, 0, 0, 0
        n_val_batches = 0
        
        # C1 variables
        val_r2_per_lead = {f"V{idx-5}": 0.0 for idx in gen_target_indices}

        with torch.no_grad():
            for batch_idx_val, batch in enumerate(val_loader):
                x_norm, target_norm, _ = batch
                x_norm = x_norm.to(device, dtype=torch.float64, non_blocking=True)
                target_norm = target_norm.to(device, dtype=torch.float64, non_blocking=True)
                x = denormalize_ecg(x_norm) if getattr(val_loader.dataset, 'output_units', 'mV') == 'mason' else x_norm
                y = denormalize_ecg(target_norm)

                lead_indices = _lead_idx_base.unsqueeze(0).expand(x.size(0), -1)
                
                # B2 & B1 & P2: Unpack correctly from dictionary; use_posterior=False
                res = bridge(x, lead_indices=lead_indices, target_len=args.target_len, use_posterior=False, kl_balance=True)
                recon, kl_scales, kl_balanced = res["recon"], res["kl"], res["kl_balanced"]

                # Generative R2: targeting leads NOT provided as input
                y_gen_target = y[:, gen_target_indices, :].to(torch.float64)
                r_gen_target = recon[:, gen_target_indices, :].to(torch.float64)
                
                val_loss, val_elbo_metrics = r2_elbo_loss(
                    r_gen_target, y_gen_target, kl_scales, kl_balanced,
                    T_actual=args.target_len, n_leads=len(gen_target_indices),
                    beta=beta_warmup
                )

                total_val_loss += val_loss.item()
                total_val_r2_loss += val_elbo_metrics["r2_mean_batch"]
                total_val_recon_loss += val_elbo_metrics["recon_nll"]
                total_val_kl_global += val_elbo_metrics["kl_total"]
                n_val_batches += 1

                # C1: Per-lead R2 Logging (Generative Targets Only)
                for idx in gen_target_indices:
                    ssr = (recon[:, idx, :] - y[:, idx, :]).pow(2).sum(dim=1)
                    sst = (y[:, idx, :] - y[:, idx, :].mean(dim=1, keepdim=True)).pow(2).sum(dim=1).clamp(min=1e-2)
                    r2_lead = 1.0 - (ssr / sst)
                    r2_lead = r2_lead.clamp(min=-20.0)
                    val_r2_per_lead[f"V{idx-5}"] += r2_lead.mean().item()

                # C2: Reconstruction Visualization (first batch only, every 5 epochs)
                if batch_idx_val == 0 and epoch % 5 == 0:
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
                    for i in range(6):
                        ch_y = 6 + i
                        axes[i].plot(y[0, ch_y, :].cpu().numpy(), color='blue', label='Target')
                        axes[i].plot(recon[0, i, :].cpu().numpy(), color='red', linestyle='dashed', label='Recon')
                        axes[i].set_ylabel(f'V{i+1} (mV)')
                        if i == 0:
                            axes[i].legend(loc='upper right')
                    axes[-1].set_xlabel('Time (samples)')
                    plt.tight_layout()
                    wandb.log({"val_reconstruction": wandb.Image(fig), "epoch": epoch})
                    plt.close(fig)

        for k in val_r2_per_lead:
            val_r2_per_lead[k] /= n_val_batches

        # Metrics Tracking (L4: Correct denominators for early breaks)
        avg_loss = total_loss / n_train_batches
        avg_r2_loss = total_r2_loss / n_train_batches
        avg_recon_loss = total_recon_loss / n_train_batches
        avg_kl_total = total_kl_global / n_train_batches

        avg_val_loss = total_val_loss / n_val_batches
        avg_val_r2_loss = total_val_r2_loss / n_val_batches
        avg_val_recon_loss = total_val_recon_loss / n_val_batches
        avg_val_kl_total = total_val_kl_global / n_val_batches

        print(f"Epoch {epoch+1}: Val Loss={avg_val_loss:.4f}, Val Gen R2={-avg_val_r2_loss:.4f}, Patience={patience_timer}")

        log_dict = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_recon_nll": avg_recon_loss,
            "train_kl_total": avg_kl_total,
            "train_r2": avg_r2_loss,
            "lr": current_lr,
            "val_loss": avg_val_loss,
            "val_recon_nll": total_val_recon_loss / n_val_batches,
            "val_kl_total": total_val_kl_global / n_val_batches,
            "val_gen_r2": total_val_r2_loss / n_val_batches,
            "train/beta_warmup": beta_warmup
        }
        # Log per-scale KL
        for i in range(kl_scales.shape[1]):
            log_dict[f"val/kl_scale_{i}"] = kl_scales[:, i].mean().item()

        for k, v in val_r2_per_lead.items():
            log_dict[f"val_r2_{k}"] = v
            
        wandb.log(log_dict)

        if avg_val_loss < best_val_score:
            best_val_score = avg_val_loss
            best_val_r2 = total_val_r2_loss / n_val_batches
            best_epoch = epoch
            patience_timer = args.patience
            save_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'bridge_state_dict': bridge.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r2': -avg_val_r2_loss,
                'args': args
            }, save_path)
            print(f"  Saved Best Model (R2: {-avg_val_r2_loss:.4f})")
        else:
            patience_timer -= 1

        if patience_timer <= 0:
            print(f"Early stopping at epoch {epoch+1}")
            break

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cNVAE-ECG Generative Training (Phase 45)")
    parser.add_argument("--backbone", type=str, required=True, choices=['ecgfm'])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)  # Reference default for Adamax
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0) # B4: Default to 1.0
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints/scientific_production_v1")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--n_output_leads", type=int, default=6) # Added from the instruction
    parser.add_argument("--target_len", type=int, default=5000) # Added from the instruction
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_steps", type=int, default=3)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    
    # Phase 26: Performance Optimization Flags
    parser.add_argument("--project", type=str, default="cNVAE-ECG-Scientific-Production")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--r2_weight", type=float, default=0.5)
    parser.add_argument("--kl_balancer", type=str, default="square")
    parser.add_argument("--spectral_norm", type=float, default=0.01)
    parser.add_argument("--bn_reg", type=float, default=0.01)
    parser.add_argument("--amp", action="store_true")
    
    args = parser.parse_args()
    train(args)
