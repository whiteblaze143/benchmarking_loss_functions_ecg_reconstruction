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
from pathlib import Path
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths, ECG_FM_ROOT
setup_import_paths(include_fairseq=True)
import argparse
import json
import hashlib
import random
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datetime import datetime


# Models, Data, and Mason Protocol -- all centralized in cnvae_ecg.py
from scripts.baselines.cnvae_ecg import (
    ECGConvECGFMBridge,
    ECG_INPUT_LIMB_V3,
    PRECORDIAL_INDICES,
    denormalize_ecg,
    batch_r2_function
)
from src.reconstruction.load_functions.ptbxl_dataset import PTBXLDataset
import wandb
import matplotlib.pyplot as plt
from scripts.common_loss import pearson_loss, mmd_loss, derivative_loss
from scripts.experiment_provenance import code_provenance


def split_inventory_hash(split: str) -> dict:
    root = _ROOT / "data/ptb_xl/tensors" / split
    files = sorted(root.glob("*.pt"), key=lambda path: int(path.stem))
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.stem}:{path.stat().st_size}\n".encode())
    return {"count": len(files), "inventory_sha256": digest.hexdigest()}


class CanonicalTensorDataset(Dataset):
    """Exact shared split used by every factorial family, in raw millivolts."""

    output_units = "mV"

    def __init__(self, tensor_root: str, split: str, input_lead_indices: list[int]):
        self.files = sorted((Path(tensor_root) / split).glob("*.pt"), key=lambda path: int(path.stem))
        self.input_lead_indices = input_lead_indices
        if not self.files:
            raise FileNotFoundError(f"No canonical tensors for split={split!r}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        target = torch.load(self.files[index], weights_only=True).float()
        return target[self.input_lead_indices].clone(), target, {}

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
                 beta=1.0, gamma_per_scale=None,
                 lambda_mse=1.0, lambda_corr=0.0, lambda_mmd=0.0, lambda_deriv=0.0):
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
    neg_elbo = lambda_mse * recon_nll + beta * kl_balanced # (B,)
    loss = neg_elbo.mean()
    
    l_corr = torch.tensor(0.0, device=recon.device)
    l_mmd = torch.tensor(0.0, device=recon.device)
    l_deriv = torch.tensor(0.0, device=recon.device)

    if lambda_corr > 0:
        l_corr = pearson_loss(recon, target)
        loss += lambda_corr * l_corr
    if lambda_mmd > 0:
        l_mmd = mmd_loss(recon, target)
        loss += lambda_mmd * l_mmd
    if lambda_deriv > 0:
        l_deriv = derivative_loss(recon, target)
        loss += lambda_deriv * l_deriv
    
    metrics = {
        "r2_mean_batch": r2_per_lead.mean().item(),
        "recon_nll": recon_nll.mean().item(),
        "kl_total": kl_scales.sum(dim=1).mean().item(),
        "neg_elbo": loss.item(),
        "l_corr": l_corr.item(),
        "l_mmd": l_mmd.item(),
        "l_deriv": l_deriv.item(),
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

def get_model(backbone_name, device, target_len=5000, fm_checkpoint=None):
    """Factory function to instantiate the correct bridge."""
    if backbone_name == 'ecgfm':
        # Phase 45: cNVAE-ECG option 2 top-down hierarchy
        print("Initializing cNVAE Generative World Model Bridge (Phase 45)...")
        model = ECGConvECGFMBridge(
            checkpoint_path=fm_checkpoint,
            freeze_encoder=True,
            target_len=target_len,
            n_output_leads=6
        ).to(dtype=torch.float32)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    return model.to(device)

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    input_lead_indices = [int(x.strip()) for x in args.input_leads.split(",") if x.strip()]
    if len(input_lead_indices) != 3 or len(set(input_lead_indices)) != 3:
        raise ValueError("--input_leads must contain exactly three distinct lead indices")
    if any(idx < 0 or idx >= 12 for idx in input_lead_indices):
        raise ValueError("--input_leads indices must be in [0, 11]")
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.name or f"{args.backbone}_phase45_{timestamp}"
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
    print(f"    Input leads: {[lead_names[idx] for idx in input_lead_indices]} (indices {input_lead_indices})")
    print(f"    Loss: Exact Mason Batch R² (from third_party)")

    # 1. Data Loading
    print("Loading PTB-XL...")
    ptbxl_root = os.path.abspath(args.ptbxl_root)
    ptbxl_csv = os.path.join(ptbxl_root, "ptbxl_database.csv")
    if args.tensor_dir:
        train_ds = CanonicalTensorDataset(args.tensor_dir, 'train', input_lead_indices)
        val_ds = CanonicalTensorDataset(args.tensor_dir, 'val', input_lead_indices)
        print(f"Canonical shared split: train={len(train_ds)}, val={len(val_ds)}")
    else:
        train_ds = PTBXLDataset(
            ptbxl_root, ptbxl_csv, split='train', target_fs=500,
            use_mason_scaling=True, input_lead_indices=input_lead_indices,
        )
        val_ds = PTBXLDataset(
            ptbxl_root, ptbxl_csv, split='val', target_fs=500,
            use_mason_scaling=True, input_lead_indices=input_lead_indices,
        )

    kw = dict(batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    if args.num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    # 2. Model Initialization
    bridge = get_model(
        args.backbone,
        device,
        target_len=args.target_len,
        fm_checkpoint=os.path.abspath(args.fm_checkpoint),
    )

    # 3. Optimization
    trainable_params = [p for p in bridge.parameters() if p.requires_grad]
    # Phase 28: Adamax Optimizer (Reference Parity)
    optimizer = optim.Adamax(trainable_params, lr=args.lr, weight_decay=1e-3, eps=1e-3)
    
    # Cosine Annealing with Warmup
    warmup_epochs = min(5, max(args.epochs - 1, 0))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - warmup_epochs, 1), eta_min=1e-6)
    
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
    warmup_epochs = min(5, max(args.epochs - 1, 0))
    base_lr = args.lr
    initial_lr = 1e-6
    
    # Loss functions: Principled ELBO (Pass 12)
    # The reconstruction term is scaled by T/2 (~2500 per lead), providing
    # natural balancing. beta=1.0 is the theoretically proper ELBO.
    beta = args.beta

    # 4. Training Loop
    # Define Generative Targets (leads the model must infer, avoiding metric inflation)
    # Target indices in the 12-lead set that are PRECORDIAL but NOT in the input
    gen_target_indices = [idx for idx in range(12) if idx not in input_lead_indices]
    # Map to 0-5 index for the decoder output
    print(f"    Generative Targets: {[lead_names[idx] for idx in gen_target_indices]} (Leads expected to be inferred)")

    best_val_score = float('inf')
    best_val_pearson = -float('inf')
    best_epoch = -1
    patience_timer = args.patience

    print(f"=== TRAINING STARTING (Batch Size: {args.batch_size}, Device: {device}) ===")
    torch.backends.cudnn.benchmark = True # O3: Faster fixed-length convolutions
    _lead_idx_base = torch.tensor(input_lead_indices, device=device)

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
            x_norm = x_norm.to(device, dtype=torch.float32, non_blocking=True)
            target_norm = target_norm.to(device, dtype=torch.float32, non_blocking=True)
            
            if getattr(train_loader.dataset, 'output_units', 'mV') == 'mason':

            
                x = denormalize_ecg(x_norm)

            
                y = denormalize_ecg(target_norm)

            
            else:

            
                x = x_norm

            
                y = target_norm

            if DEBUG and batch_idx < DEBUG_STEPS:
                print(f"\n--- DEBUG Step {batch_idx} ---")
                tstats("x_raw", x)
                tstats("y_raw", y)

            # K2: Zero grads only at the start of accumulation window
            if batch_idx % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            # Forward pass — P2: bridge returns a dictionary
            # Standard input basis: Leads I (0), III (2), V3 (8)
            lead_indices = torch.tensor([input_lead_indices], device=device).expand(x.size(0), -1)
            
            # Float32/TF32 gives the A100 a high-throughput path while all
            # explicit finiteness guards remain active.
            outputs = bridge(x, lead_indices=lead_indices, target_len=args.target_len, kl_balance=True)
            recon = outputs["recon"]
            kl_scales = outputs["kl"]
            kl_balanced = outputs["kl_balanced"]
            aux_signals = outputs.get("aux")
            
            # --- Reconstruction Loss: Principled R2-ELBO (Pass 12/14) ---
            y_prec = y[:, gen_target_indices, :].to(torch.float32)
            r_prec = recon[:, gen_target_indices, :].to(torch.float32)

            # Calculate primary ELBO loss
            beta_warmup = min(beta, beta * (epoch / max(warmup_epochs, 1)))
            loss, elbo_metrics = r2_elbo_loss(
                r_prec, y_prec, kl_scales, kl_balanced,
                T_actual=args.target_len, n_leads=len(gen_target_indices), beta=beta_warmup,
                lambda_mse=args.lambda_mse, lambda_corr=args.lambda_corr, lambda_mmd=args.lambda_mmd, lambda_deriv=args.lambda_deriv
            )
            
            # --- Multiscale Supv Loss (N6/P1) ---
            if aux_signals is not None:
                loss += aux_r2_loss(aux_signals, y, PRECORDIAL_INDICES, aux_weight=0.1)

            # --- Spectral & Regularization Loss (O4: Fast Caching) ---
            sr_loss = bridge.decoder.spectral_norm_parallel()
            bn_loss = bridge.decoder.batchnorm_loss()
            loss += args.spectral_reg_weight * sr_loss + args.batchnorm_reg_weight * bn_loss
            
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
                    "train/beta_warmup": beta_warmup,
                    "train/l_corr": elbo_metrics["l_corr"],
                    "train/l_mmd": elbo_metrics["l_mmd"],
                    "train/l_deriv": elbo_metrics["l_deriv"],
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
        total_val_missing_sse, total_val_missing_points = 0.0, 0
        total_val_missing_pearson, total_val_missing_records = 0.0, 0
        n_val_batches = 0
        
        # C1 variables
        val_r2_per_lead = {lead_names[idx]: 0.0 for idx in gen_target_indices}

        with torch.no_grad():
            for batch_idx_val, batch in enumerate(val_loader):
                if args.max_steps > 0 and batch_idx_val >= args.max_steps:
                    break
                x_norm, target_norm, _ = batch
                x_norm = x_norm.to(device, dtype=torch.float32, non_blocking=True)
                target_norm = target_norm.to(device, dtype=torch.float32, non_blocking=True)
                if getattr(val_loader.dataset, 'output_units', 'mV') == 'mason':

                    x = denormalize_ecg(x_norm)

                    y = denormalize_ecg(target_norm)

                else:

                    x = x_norm

                    y = target_norm

                lead_indices = _lead_idx_base.unsqueeze(0).expand(x.size(0), -1)
                
                # The conditional posterior sees only the acquired leads. Its
                # mean is therefore a deterministic, leakage-free inference
                # path and avoids stochastic checkpoint selection noise.
                res = bridge(
                    x,
                    lead_indices=lead_indices,
                    target_len=args.target_len,
                    use_posterior=True,
                    deterministic=True,
                    kl_balance=True,
                )
                recon, kl_scales, kl_balanced = res["recon"], res["kl"], res["kl_balanced"]

                # Generative R2: targeting leads NOT provided as input
                y_gen_target = y[:, gen_target_indices, :].to(torch.float32)
                r_gen_target = recon[:, gen_target_indices, :].to(torch.float32)
                
                val_loss, val_elbo_metrics = r2_elbo_loss(
                    r_gen_target, y_gen_target, kl_scales, kl_balanced,
                    T_actual=args.target_len, n_leads=len(gen_target_indices),
                    beta=beta_warmup,
                    lambda_mse=args.lambda_mse, lambda_corr=args.lambda_corr, lambda_mmd=args.lambda_mmd, lambda_deriv=args.lambda_deriv
                )

                total_val_loss += val_loss.item()
                total_val_r2_loss += val_elbo_metrics["r2_mean_batch"]
                total_val_recon_loss += val_elbo_metrics["recon_nll"]
                total_val_kl_global += val_elbo_metrics["kl_total"]
                n_val_batches += 1
                missing_error = r_gen_target - y_gen_target
                total_val_missing_sse += missing_error.square().sum().item()
                total_val_missing_points += missing_error.numel()
                centered_recon = r_gen_target - r_gen_target.mean(dim=-1, keepdim=True)
                centered_target = y_gen_target - y_gen_target.mean(dim=-1, keepdim=True)
                correlations = (centered_recon * centered_target).sum(dim=-1) / torch.sqrt(
                    centered_recon.square().sum(dim=-1).clamp_min(1e-8)
                    * centered_target.square().sum(dim=-1).clamp_min(1e-8)
                )
                total_val_missing_pearson += correlations.mean(dim=1).sum().item()
                total_val_missing_records += correlations.shape[0]

                # C1: Per-lead R2 Logging (Generative Targets Only)
                for idx in gen_target_indices:
                    ssr = (recon[:, idx, :] - y[:, idx, :]).pow(2).sum(dim=1)
                    sst = (y[:, idx, :] - y[:, idx, :].mean(dim=1, keepdim=True)).pow(2).sum(dim=1).clamp(min=1e-2)
                    r2_lead = 1.0 - (ssr / sst)
                    r2_lead = r2_lead.clamp(min=-20.0)
                    val_r2_per_lead[lead_names[idx]] += r2_lead.mean().item()

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
        avg_val_missing_mse = total_val_missing_sse / max(total_val_missing_points, 1)
        avg_val_missing_pearson = total_val_missing_pearson / max(total_val_missing_records, 1)

        print(
            f"Epoch {epoch+1}: Val Loss={avg_val_loss:.4f}, "
            f"Val Recon NLL={avg_val_recon_loss:.4f}, "
            f"Val Missing MSE={avg_val_missing_mse:.6f}, "
            f"Val Missing Pearson={avg_val_missing_pearson:.4f}, "
            f"Val Gen R2={avg_val_r2_loss:.4f}, Patience={patience_timer}"
        )

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
            "val_missing_mse": avg_val_missing_mse,
            "val_missing_pearson": avg_val_missing_pearson,
            "train/beta_warmup": beta_warmup
        }
        # Log per-scale KL
        for i in range(kl_scales.shape[1]):
            log_dict[f"val/kl_scale_{i}"] = kl_scales[:, i].mean().item()

        for k, v in val_r2_per_lead.items():
            log_dict[f"val_r2_{k}"] = v
            
        wandb.log(log_dict)

        # Select checkpoints using reconstruction quality, not the total ELBO.
        # The KL coefficient warms up across epochs, so comparing total ELBO
        # would systematically favor epoch 1 even when reconstruction improves.
        selector_improved = (
            avg_val_missing_mse < best_val_score - 1e-12
            or (
                abs(avg_val_missing_mse - best_val_score) <= 1e-12
                and avg_val_missing_pearson > best_val_pearson
            )
        )
        if selector_improved:
            best_val_score = avg_val_missing_mse
            best_val_pearson = avg_val_missing_pearson
            best_epoch = epoch
            patience_timer = args.patience
            save_path = os.path.join(save_dir, "best_model.pt")
            # The ECG-FM backbone is frozen and is reloaded from fm_checkpoint.
            # It is registered both directly and through decoder.backbone, so
            # exclude both aliases to avoid adding ~2 GB of duplicate immutable
            # weights to every experiment checkpoint.
            state_dict_to_save = {
                key: value
                for key, value in bridge.state_dict().items()
                if not key.startswith(("backbone.", "decoder.backbone."))
            }
            checkpoint = {
                'epoch': epoch,
                'bridge_state_dict': state_dict_to_save,
                'val_r2': avg_val_r2_loss,
                'val_recon_nll': avg_val_recon_loss,
                'val_missing_mse': avg_val_missing_mse,
                'val_missing_pearson': avg_val_missing_pearson,
                'args': args,
                'checkpoint_contains_fm_backbone': False,
                'fm_checkpoint': os.path.abspath(args.fm_checkpoint),
                'checkpoint_selector': 'lowest_missing_lead_val_mse_then_highest_missing_lead_val_pearson',
                'provenance': {
                    'seed': int(args.seed),
                    'loss_implementation': 'scripts.common_loss:{pearson_loss,mmd_loss,derivative_loss}',
                    'loss_weights': {'corr': args.lambda_corr, 'mmd': args.lambda_mmd, 'deriv': args.lambda_deriv},
                    'base_objective_weights': {'beta_kl': args.beta, 'spectral': args.spectral_reg_weight, 'batchnorm': args.batchnorm_reg_weight},
                    'preprocessing': {'sample_rate_hz': 500, 'training_dataset_scaling': 'canonical shared raw-mV tensors'},
                    'numerical_precision': 'float32_with_A100_TF32_enabled',
                    'inference': {
                        'latent': 'conditional_posterior_mean',
                        'inputs': 'observed_leads_only',
                        'deterministic': True,
                        'target_leakage': False,
                    },
                    'split_inventory': {split: split_inventory_hash(split) for split in ('train', 'val', 'test')},
                    'architecture_revision': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=_ROOT, capture_output=True, text=True, check=True).stdout.strip(),
                    'architecture_provenance': code_provenance(_ROOT, [
                        'scripts/baselines/train_cnvae.py',
                        'scripts/baselines/cnvae_ecg.py',
                        'scripts/common_loss.py',
                    ]),
                    'optimizer': 'Adamax',
                    'scheduler': 'warmup+CosineAnnealingLR',
                    'checkpoint_selector': 'lowest_missing_lead_val_mse_then_highest_missing_lead_val_pearson',
                },
            }
            if args.save_optimizer_state:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            torch.save(checkpoint, save_path)
            print(
                f"  Saved Best Model "
                f"(R2: {avg_val_r2_loss:.4f}, Recon NLL: {avg_val_recon_loss:.4f})"
            )
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
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lr", type=float, default=1e-3)  # Reference default for Adamax
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0) # B4: Default to 1.0
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints/scientific_production_v1")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--n_output_leads", type=int, default=6) # Added from the instruction
    parser.add_argument("--target_len", type=int, default=5000) # Added from the instruction
    parser.add_argument(
        "--ptbxl_root",
        type=str,
        default="data/ptb_xl",
        help="PTB-XL root containing ptbxl_database.csv and records500/.",
    )
    parser.add_argument("--tensor_dir", type=str, default=None)
    parser.add_argument(
        "--fm_checkpoint",
        type=str,
        default=str(ECG_FM_ROOT / "checkpoints/mimic_iv_ecg_physionet_pretrained.pt"),
        help="ECG-FM checkpoint used by the frozen cNVAE backbone.",
    )
    parser.add_argument(
        "--input_leads",
        type=str,
        default="0,2,8",
        help="Comma-separated observed lead indices; use 0,1,7 for I-II-V2.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_steps", type=int, default=3)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument(
        "--save_optimizer_state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include optimizer state in best_model.pt for resumable training.",
    )
    
    # Phase 26: Performance Optimization Flags
    parser.add_argument("--project", type=str, default="cNVAE-ECG-Scientific-Production")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--spectral_reg_weight", type=float, default=1e-6)
    parser.add_argument("--batchnorm_reg_weight", type=float, default=1e-6)
    parser.add_argument("--r2_weight", type=float, default=0.5)
    parser.add_argument("--kl_balancer", type=str, default="square")
    parser.add_argument("--spectral_norm", type=float, default=0.01)
    parser.add_argument("--bn_reg", type=float, default=0.01)
    parser.add_argument("--amp", action="store_true")
    
    parser.add_argument("--lambda_mse", type=float, default=1.0, help="Compatibility flag (not used directly in cnVAE, ELBO is used instead)")
    parser.add_argument("--lambda_corr", type=float, default=0.0)
    parser.add_argument("--lambda_mmd", type=float, default=0.0)
    parser.add_argument("--lambda_deriv", type=float, default=0.0)
    
    args = parser.parse_args()
    train(args)
