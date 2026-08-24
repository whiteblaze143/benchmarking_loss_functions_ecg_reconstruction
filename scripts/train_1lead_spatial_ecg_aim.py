#!/usr/bin/env python3
"""
Unified Cross-Architecture Factorial Training Script.
Supports --architecture unet, msvae, ecg_aim across all 128 loss mask combinations.
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths
setup_import_paths(include_fairseq=True)

import os
import json
import logging
import random
import hashlib
import tempfile
import argparse
import importlib.util
import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from scripts.train_mcma_3lead import MCMAModel, PTBXLDataset
from scripts.common_loss import CombinatorialCompositeLoss
from scripts.experiment_provenance import code_provenance

from unified_latents.engineering.experimental.Multi_Scale_VAE import WearECGVAE
from unified_latents.engineering.utils.common import mask_unobserved_leads
from unified_latents.engineering.utils.regimes import make_lead_indices

import wandb

_SPATIAL_MODULE_PATH = (
    _ROOT / "unified_latents" / "engineering" / "experimental" / "aim_1_lead.py"
)
_SPATIAL_SPEC = importlib.util.spec_from_file_location("ecg_aim_spatial_working_copy", _SPATIAL_MODULE_PATH)
if _SPATIAL_SPEC is None or _SPATIAL_SPEC.loader is None:
    raise ImportError(f"Cannot load spatial ECG-AIM working copy: {_SPATIAL_MODULE_PATH}")
_SPATIAL_MODULE = importlib.util.module_from_spec(_SPATIAL_SPEC)
_SPATIAL_SPEC.loader.exec_module(_SPATIAL_MODULE)
build_spatial_ecg_aim = _SPATIAL_MODULE.build_alitok_vae_1d

OBSERVED_LEADS = [0, 1, 7] # Lead I, II, V2

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def atomic_torch_save(payload, destination):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        # Recursively convert floating tensors to FP16.  Structured checkpoints
        # keep provenance and architecture metadata while remaining inference
        # loadable at roughly half the float32 storage footprint.
        def compact(value):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                return value.half()
            if isinstance(value, dict):
                return {k: compact(v) for k, v in value.items()}
            if isinstance(value, list):
                return [compact(v) for v in value]
            if isinstance(value, tuple):
                return tuple(compact(v) for v in value)
            return value

        compact_payload = compact(payload)

        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        torch.save(compact_payload, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

def build_architecture(arch_name, device, args=None):
    if arch_name == "unet":
        model = MCMAModel(in_channels=12, out_channels=12).to(device)
    elif arch_name == "msvae":
        model = WearECGVAE(
            latent_channels=4,
            target_len=5000,
            beta_kl=1e-4,
            missing_lead_weight=1.0
        ).to(device)
    elif arch_name == "ecg_aim":
        model = build_spatial_ecg_aim(
            architecture="ecg_aim_v1",
            target_len=5000,
            patch_size=25,
            encoder_depth=8,
            decoder_depth=4
        ).to(device)
    elif arch_name == "ecg_aim_spatial":
        if args is None:
            raise ValueError("spatial ECG-AIM requires parsed configuration")
        model = build_spatial_ecg_aim(
            architecture="ecg_aim_spatial_v1",
            target_len=5000,
            patch_size=25,
            encoder_depth=8,
            decoder_depth=4,
            lead_conditioning_mode=args.lead_conditioning_mode,
            use_relative_geometry=args.use_relative_geometry,
            use_spatial_film=args.use_spatial_film,
            spatial_gain_init=args.spatial_gain_init,
            geometry_control=args.geometry_control,
        ).to(device)
    elif arch_name == "ecg_aim_panorama_author":
        model = build_spatial_ecg_aim(
            architecture="ecg_aim_panorama_author_v1",
            target_len=5000,
            patch_size=25,
            encoder_depth=8,
            decoder_depth=4,
        ).to(device)
    elif arch_name == "ecg_aim_exact_theta":
        if args is None:
            raise ValueError("exact-Theta ECG-AIM requires parsed configuration")
        model = build_spatial_ecg_aim(
            architecture="ecg_aim_exact_theta_factorial_v1",
            target_len=5000,
            patch_size=25,
            encoder_depth=8,
            decoder_depth=4,
            use_learned_lead_id=args.use_learned_lead_id,
            use_relative_geometry=args.use_relative_geometry,
            use_spatial_film=args.use_spatial_film,
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
        
    if hasattr(torch, "compile") and arch_name != "unet":
        try:
            model = torch.compile(model)
            logging.info(f"Successfully enabled PyTorch 2.0 torch.compile graph fusion for {arch_name}!")
        except Exception as e:
            logging.warning(f"Could not enable torch.compile: {e}")
            
    return model

def forward_pass(model, arch_name, y, device):
    """Unified forward pass producing reconstructed 12-lead tensor out (B, 12, 5000)."""
    if arch_name == "unet":
        masked_y = mask_unobserved_leads(y, OBSERVED_LEADS).contiguous()
        padded = F.pad(masked_y, (0, 120)).contiguous()
        out = model(padded)[:, :12, :5000].contiguous()
        kl_loss = 0.0
    elif arch_name in {"msvae", "ecg_aim", "ecg_aim_spatial", "ecg_aim_panorama_author", "ecg_aim_exact_theta"}:
        masked_y = mask_unobserved_leads(y, OBSERVED_LEADS).contiguous()
        lead_idx = make_lead_indices(OBSERVED_LEADS, y.shape[0], device)
        res = model(masked_y, y_full=y.contiguous(), lead_indices=lead_idx, mode="stage1")
        out = res["y_pred"][:, :12, :5000].contiguous()
        kl_loss = res.get("kl_loss", 0.0)
    return out, kl_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["ecg_aim", "ecg_aim_spatial", "ecg_aim_panorama_author", "ecg_aim_exact_theta"],
        default="ecg_aim_spatial",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--data_dir", type=str, default="data/ptb_xl/tensors")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--factorial_mask", type=str, required=True, help="7-digit mask, e.g. 1000000")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches for smoke testing")
    parser.add_argument("--observed_leads", type=int, nargs="+", default=[0, 1, 7], help="Observed lead indices")
    parser.add_argument("--lead_conditioning_mode", choices=["learned", "panorama", "panorama_hybrid"], default="learned")
    parser.add_argument("--use_learned_lead_id", action="store_true")
    parser.add_argument("--use_relative_geometry", action="store_true")
    parser.add_argument("--use_spatial_film", action="store_true")
    parser.add_argument("--spatial_gain_init", type=float, default=0.1)
    parser.add_argument(
        "--geometry_control",
        choices=["standard", "fixed_random", "permuted"],
        default="standard",
    )
    args = parser.parse_args()

    global OBSERVED_LEADS
    OBSERVED_LEADS = args.observed_leads

    architecture_sources = {
        "unet": ["scripts/train_mcma_3lead.py"],
        "msvae": ["unified_latents/engineering/experimental/Multi_Scale_VAE.py"],
        "ecg_aim": [
            "unified_latents/engineering/experimental/aim_1_lead.py",
            "unified_latents/engineering/third_party/alitok/alitok-main/train_tokenizer/model/vae_stage1.py",
        ],
        "ecg_aim_spatial": [
            "unified_latents/engineering/experimental/aim_1_lead.py",
            "unified_latents/engineering/third_party/alitok/alitok-main/train_tokenizer/model/vae_stage1.py",
        ],
        "ecg_aim_panorama_author": [
            "unified_latents/engineering/experimental/aim_1_lead.py",
            "external/Electrocardio-Panorama-main/codes/network/utils/theta_encoder.py",
            "external/Electrocardio-Panorama-main/codes/network/model_nefnet2.py",
            "external/Electrocardio-Panorama-main/codes/dataset/ptbv2.py",
            "unified_latents/engineering/third_party/alitok/alitok-main/train_tokenizer/model/vae_stage1.py",
        ],
        "ecg_aim_exact_theta": [
            "unified_latents/engineering/experimental/aim_1_lead.py",
            "external/Electrocardio-Panorama-main/codes/network/utils/theta_encoder.py",
            "external/Electrocardio-Panorama-main/codes/dataset/ptbv2.py",
            "unified_latents/engineering/third_party/alitok/alitok-main/train_tokenizer/model/vae_stage1.py",
        ],
    }
    source_paths = [
        "scripts/train_factorial_multimodel.py",
        "scripts/common_loss.py",
        "scripts/experiment_provenance.py",
        *architecture_sources[args.architecture],
    ]
    source_provenance = code_provenance(_ROOT, source_paths)

    # Set architecture-appropriate default batch size.
    # ECG-AIM (axial transformer) can handle larger batches than the conv MSVAE.
    if args.batch_size == 256 and args.architecture == "msvae":
        args.batch_size = 32
    elif args.batch_size == 256 and args.architecture in {"ecg_aim", "ecg_aim_spatial", "ecg_aim_panorama_author", "ecg_aim_exact_theta"}:
        args.batch_size = 64

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda' and hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
        logging.info("Enabled TensorFloat32 (TF32) matmul precision ('high') for NVIDIA A100 Tensor Cores!")
    # MSVAE (strided conv encoder) has known cudnn non-determinism issues; disable it.
    # ECG-AIM (pure transformer) is fully cudnn-compatible and benefits from it.
    if args.architecture == "msvae":
        torch.backends.cudnn.enabled = False
    print(f"Device: {device} | Architecture: {args.architecture} | Mask: {args.factorial_mask} | Seed: {args.seed} | Batch Size: {args.batch_size}")

    data_dir = os.path.abspath(args.data_dir)
    train_dataset = PTBXLDataset(f"{data_dir}/train")
    val_dataset = PTBXLDataset(f"{data_dir}/val")

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, generator=loader_generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.run_name:
        wandb.init(project="ecg-reconstruction-ablation", name=args.run_name, config=vars(args))

    model = build_architecture(args.architecture, device, args)
    
    if args.architecture in {"ecg_aim", "ecg_aim_spatial", "ecg_aim_panorama_author", "ecg_aim_exact_theta"}:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-4)
        max_lr = 5e-4
        pct_start = 0.2
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-4)
        max_lr = 3e-4
        pct_start = 0.1
    criterion = CombinatorialCompositeLoss(args.factorial_mask)
    scaler = torch.amp.GradScaler('cuda')

    steps_per_epoch = len(train_loader) if not args.max_batches else min(len(train_loader), args.max_batches)
    total_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=pct_start
    )

    best_val_loss = float('inf')
    best_pearson = -1.0
    best_epoch = None

    for epoch in range(args.epochs):
        model.train()
        t_loss = 0.0
        # Enable AMP for unet and ecg_aim; MSVAE conv encoder has amp instability.
        use_amp = (args.architecture in {"unet", "ecg_aim", "ecg_aim_spatial", "ecg_aim_panorama_author", "ecg_aim_exact_theta"})
        amp_dtype = torch.bfloat16 if args.architecture in {"ecg_aim", "ecg_aim_spatial", "ecg_aim_panorama_author", "ecg_aim_exact_theta"} else torch.float16

        for batch_idx, (x_raw, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            y = y[..., :5000].to(device, non_blocking=True).contiguous()
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                out, kl_loss = forward_pass(model, args.architecture, y, device)
                min_len = min(out.shape[-1], y.shape[-1])
                out_crop = out[..., :min_len]
                y_crop = y[..., :min_len]
                loss, l_mse, l_corr, l_deriv, l_vcg, l_ed, l_lead, l_mmd = criterion(out_crop, y_crop)
                total_loss = loss + 1e-4 * kl_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            t_loss += loss.item()

        # Validation
        model.eval()
        v_loss = 0.0
        v_corr_list = []
        with torch.inference_mode():
            for v_idx, (x_raw, y) in enumerate(val_loader):
                if args.max_batches and v_idx >= args.max_batches:
                    break
                y = y[..., :5000].to(device, non_blocking=True).contiguous()
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    out, _ = forward_pass(model, args.architecture, y, device)
                    min_len = min(out.shape[-1], y.shape[-1])
                    out_crop = out[..., :min_len]
                    y_crop = y[..., :min_len]
                    loss, _, _, _, _, _, _, _ = criterion(out_crop, y_crop)
                v_loss += loss.item()
                # Missing leads Pearson correlation (zero-centered)
                missing_mask = torch.ones(12, dtype=torch.bool, device=device)
                missing_mask[OBSERVED_LEADS] = False
                pred_miss = out_crop[:, missing_mask, :]
                true_miss = y_crop[:, missing_mask, :]
                pred_centered = pred_miss - pred_miss.mean(dim=-1, keepdim=True)
                true_centered = true_miss - true_miss.mean(dim=-1, keepdim=True)
                r_val = F.cosine_similarity(pred_centered.flatten(1), true_centered.flatten(1), dim=1).mean().item()
                v_corr_list.append(r_val)

        mean_v_loss = v_loss / len(val_loader)
        mean_v_corr = float(np.mean(v_corr_list))

        print(f"Epoch {epoch+1} | Val Loss: {mean_v_loss:.4f} | Val Missing Pearson: {mean_v_corr:.4f}")
        if args.run_name:
            wandb.log({"epoch": epoch+1, "val_loss": mean_v_loss, "val_missing_pearson": mean_v_corr})

        if mean_v_corr > best_pearson:
            best_pearson = mean_v_corr
            best_epoch = epoch + 1
            uncompiled_model = getattr(model, "_orig_mod", model)
            provenance = {
                "seed": args.seed,
                "factorial_mask": args.factorial_mask,
                "architecture": args.architecture,
                "preprocessing": {
                    "sample_rate_hz": 500,
                    "units": "mV",
                    "normalization": "none",
                    "target_samples": 5000,
                    "observed_leads": OBSERVED_LEADS,
                },
                "architecture_revision": source_provenance["git_commit"],
                "architecture_provenance": source_provenance,
                "checkpoint_selector": "highest_missing_lead_validation_pearson",
                "optimizer": "AdamW(lr=3e-4,weight_decay=1e-4)",
            }
            checkpoint_payload = {
                "schema_version": 1,
                "architecture": args.architecture,
                "model_state_dict": uncompiled_model.state_dict(),
                "provenance": provenance,
                "target_len": 5000,
                "latent_channels": 4 if args.architecture == "msvae" else None,
                "alitok_architecture": {
                    "ecg_aim": "ecg_aim_v1",
                    "ecg_aim_spatial": "ecg_aim_spatial_v1",
                    "ecg_aim_panorama_author": "ecg_aim_panorama_author_v1",
                    "ecg_aim_exact_theta": "ecg_aim_exact_theta_factorial_v1",
                }[args.architecture],
                "alitok_patch_size": 25,
                "alitok_encoder_depth": 8,
                "alitok_decoder_depth": 4,
                "lead_conditioning_mode": args.lead_conditioning_mode,
                "use_learned_lead_id": args.use_learned_lead_id,
                "use_relative_geometry": args.use_relative_geometry,
                "use_spatial_film": args.use_spatial_film,
                "spatial_gain_init": args.spatial_gain_init,
                "geometry_control": args.geometry_control,
                "best_epoch": best_epoch,
                "best_val_loss": mean_v_loss,
                "best_val_missing_pearson": best_pearson,
            }
            atomic_torch_save(checkpoint_payload, args.checkpoint_path)

    print(f"Training Complete for {args.run_name}. Best Val Missing Pearson: {best_pearson:.4f}")

if __name__ == "__main__":
    main()
