#!/usr/bin/env python3
"""Stage A 3D Theta Spatial Reconstruction Ablation Training Script (ECG-AIM-3Dθ).

Implements the 6 controlled experimental cells:
- D0_current_id_currentloss: learned additive lead ID, current MSE objective
- D1_theta_mul_currentloss: exact theta, multiplicative conditioning, current MSE objective
- D2_current_id_l1: learned additive lead ID, genuine missing-lead L1 objective
- D3_theta_mul_l1: exact theta, multiplicative conditioning, genuine missing-lead L1 (PRIMARY 3DRECON)
- D4_learned12_mul_l1: learned 12x12 code, multiplicative conditioning, genuine missing-lead L1 (Capacity control)
- D5_permuted_theta_mul_l1: permuted theta, multiplicative conditioning, genuine missing-lead L1 (Geometry control)

Strict Deployment Normalization:
Input is raw physical millivolts of the observed lead.
Zero target normalization leakage: statistics from the 11 hidden leads are NEVER computed or leaked.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from unified_latents.engineering.models.three_d_theta_reconstruction import (
    ECGAIM_THETA,
    LEAD_NAMES,
    ThreeDThetaECGAIM,
    ThetaEncoder,
    missing_lead_l1,
)

CELL_CONFIGS = {
    "D0_current_id_currentloss": {
        "code_mode": "learned_additive",
        "fusion": "add",
        "recon_loss": "current",
        "description": "Frozen additive learned lead-ID anchor under current MSE objective",
    },
    "D1_theta_mul_currentloss": {
        "code_mode": "theta",
        "fusion": "mul",
        "recon_loss": "current",
        "description": "Physical theta multiplicative conditioning under current MSE objective",
    },
    "D2_current_id_l1": {
        "code_mode": "learned_additive",
        "fusion": "add",
        "recon_loss": "l1",
        "description": "Additive learned lead-ID anchor under genuine missing-lead L1 objective",
    },
    "D3_theta_mul_l1": {
        "code_mode": "theta",
        "fusion": "mul",
        "recon_loss": "l1",
        "description": "Primary 3DRECON-QT cell: exact physical theta, multiplicative, genuine L1",
    },
    "D4_learned12_mul_l1": {
        "code_mode": "learned",
        "fusion": "mul",
        "recon_loss": "l1",
        "description": "Capacity control: learned 12-D code, multiplicative, genuine L1",
    },
    "D5_permuted_theta_mul_l1": {
        "code_mode": "permuted_theta",
        "fusion": "mul",
        "recon_loss": "l1",
        "description": "Geometry control: fixed derangement of physical theta, multiplicative, genuine L1",
    },
    "D6_theta_add_l1": {
        "code_mode": "theta",
        "fusion": "add",
        "recon_loss": "l1",
        "description": "Matched fusion control: exact physical theta, additive, genuine L1",
    },
    "D7_learned12_add_l1": {
        "code_mode": "learned",
        "fusion": "add",
        "recon_loss": "l1",
        "description": "Matched learned-code fusion control: learned 12-D code, additive, genuine L1",
    },
    "D8_random12_mul_l1": {
        "code_mode": "random_fixed",
        "fusion": "mul",
        "recon_loss": "l1",
        "description": "Code structure control: fixed random 12-D code, multiplicative, genuine L1",
    },
}

CHEST_INDICES = [6, 7, 8, 9, 10, 11]  # V1 to V6


class SimplePTBXLDataset(Dataset):
    """Zero-leakage PTB-XL dataset loading raw physical millivolt waveforms."""

    def __init__(self, data_dir: str | Path, max_samples: int | None = None):
        self.data_dir = Path(data_dir)
        self.files = sorted(
            list(self.data_dir.glob("*.pt")),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
        )
        if max_samples is not None:
            self.files = self.files[:max_samples]
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            waveform = torch.as_tensor(obj["waveform"], dtype=torch.float32)
        elif isinstance(obj, torch.Tensor):
            waveform = obj.float()
        else:
            raise TypeError(f"Unexpected data type in {path}: {type(obj)}")
        return waveform[..., :5000]


def compute_lead_pearson(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Vectorized Pearson correlation per sample along temporal dimension.

    pred, target: [B, 5000]
    returns: [B]
    """
    p_cent = pred - pred.mean(dim=-1, keepdim=True)
    t_cent = target - target.mean(dim=-1, keepdim=True)
    cov = (p_cent * t_cent).sum(dim=-1)
    p_std = torch.sqrt((p_cent ** 2).sum(dim=-1).clamp_min(eps))
    t_std = torch.sqrt((t_cent ** 2).sum(dim=-1).clamp_min(eps))
    r = cov / (p_std * t_std)
    return r.clamp(-1.0, 1.0)


def evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    observed_lead: int,
    device: torch.device,
) -> Dict[str, float]:
    """Exhaustive validation evaluation matching the scientific protocol."""
    model.eval()
    missing_indices = [l for l in range(12) if l != observed_lead]

    all_missing_pearsons = []
    chest_pearsons = []
    transition_pearsons = []
    total_l1 = 0.0
    total_mse = 0.0
    total_samples = 0

    with torch.inference_mode():
        for batch in val_loader:
            targets = batch.to(device)  # [B, 12, 5000]
            B = targets.shape[0]

            x_source = targets[:, observed_lead : observed_lead + 1, :]  # [B, 1, 5000]
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                out = model(x_source, obs_lead_idx=observed_lead)
                preds = out["y_pred"].float()  # [B, 12, 5000]

            # 1. Missing-lead Pearson
            p_miss = preds[:, missing_indices]  # [B, 11, 5000]
            t_miss = targets[:, missing_indices]  # [B, 11, 5000]
            p_miss_cent = p_miss - p_miss.mean(dim=-1, keepdim=True)
            t_miss_cent = t_miss - t_miss.mean(dim=-1, keepdim=True)
            agg_r = F.cosine_similarity(p_miss_cent.flatten(1), t_miss_cent.flatten(1), dim=1)
            all_missing_pearsons.extend(agg_r.cpu().tolist())

            # 2. Precordial V1-V6 Pearson
            p_chest = preds[:, CHEST_INDICES]  # [B, 6, 5000]
            t_chest = targets[:, CHEST_INDICES]
            p_chest_cent = p_chest - p_chest.mean(dim=-1, keepdim=True)
            t_chest_cent = t_chest - t_chest.mean(dim=-1, keepdim=True)
            chest_r = F.cosine_similarity(p_chest_cent.flatten(1), t_chest_cent.flatten(1), dim=1)
            chest_pearsons.extend(chest_r.cpu().tolist())

            # 3. Precordial Transition Delta V_k = V_{k+1} - V_k
            # chest leads: V1=6, V2=7, V3=8, V4=9, V5=10, V6=11
            p_trans = preds[:, CHEST_INDICES[1:]] - preds[:, CHEST_INDICES[:-1]]  # [B, 5, 5000]
            t_trans = targets[:, CHEST_INDICES[1:]] - targets[:, CHEST_INDICES[:-1]]
            p_tr_cent = p_trans - p_trans.mean(dim=-1, keepdim=True)
            t_tr_cent = t_trans - t_trans.mean(dim=-1, keepdim=True)
            tr_r = F.cosine_similarity(p_tr_cent.flatten(1), t_tr_cent.flatten(1), dim=1)
            transition_pearsons.extend(tr_r.cpu().tolist())

            # 4. Losses
            l1_loss = missing_lead_l1(preds, targets, observed_lead=observed_lead)
            mse_loss = F.mse_loss(preds[:, missing_indices], targets[:, missing_indices])

            total_l1 += l1_loss.item() * B
            total_mse += mse_loss.item() * B
            total_samples += B

    return {
        "val_missing_pearson": float(np.mean(all_missing_pearsons)),
        "val_missing_pearson_p05": float(np.quantile(all_missing_pearsons, 0.05)),
        "val_chest_pearson": float(np.mean(chest_pearsons)),
        "val_precordial_transition_pearson": float(np.mean(transition_pearsons)),
        "val_l1_recon": float(total_l1 / total_samples),
        "val_mse_recon": float(total_mse / total_samples),
    }


def main():
    p = argparse.ArgumentParser(description="Train 3D Theta Spatial ECG-AIM Ablation Cell")
    p.add_argument("--cell", required=True, choices=list(CELL_CONFIGS.keys()), help="Stage A cell name")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--observed-lead", type=int, default=0, help="Observed lead index (0 = Lead I)")
    p.add_argument("--epochs", type=int, default=10, help="Epochs to train (default 10)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    p.add_argument("--data-dir", default="data/ptb_xl/tensors", help="Path to ptb_xl tensor directory")
    p.add_argument("--runs-dir", default="refine-logs/convergence_10e/runs", help="Output directory for runs")
    p.add_argument("--smoke", action="store_true", help="Run 2-batch smoke test")
    args = p.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    cell_cfg = CELL_CONFIGS[args.cell]
    run_name = f"{args.cell}_s{args.seed}_l{args.observed_lead}"
    output_dir = Path(args.runs_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Launching Run: {run_name}")
    print(f"  Description: {cell_cfg['description']}")
    print(f"  Code Mode:   {cell_cfg['code_mode']} | Fusion: {cell_cfg['fusion']} | Loss: {cell_cfg['recon_loss']}")

    # Build Model
    model = ThreeDThetaECGAIM(
        code_mode=cell_cfg["code_mode"],
        fusion=cell_cfg["fusion"],
        patch_size=25,
        width=768,
        encoder_depth=8,
        decoder_depth=4,
        heads=12,
        use_delineation_head=False,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Capture 12-D Theta Features for Provenance
    theta_enc = ThetaEncoder(encoder_len=1)
    encoded_theta = theta_enc(ECGAIM_THETA.unsqueeze(0))[0].tolist()
    theta_provenance = {
        name: [round(v, 6) for v in encoded_theta[i]] for i, name in enumerate(LEAD_NAMES)
    }

    # Save Config
    config_dict = {
        "cell": args.cell,
        "run_name": run_name,
        "seed": args.seed,
        "observed_lead": args.observed_lead,
        "observed_lead_name": LEAD_NAMES[args.observed_lead],
        "code_mode": cell_cfg["code_mode"],
        "fusion": cell_cfg["fusion"],
        "recon_loss": cell_cfg["recon_loss"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "theta_provenance": theta_provenance,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (output_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    # Datasets
    data_dir = Path(args.data_dir)
    max_s = 64 if args.smoke else None
    train_ds = SimplePTBXLDataset(data_dir / "train", max_samples=max_s)
    val_ds = SimplePTBXLDataset(data_dir / "val", max_samples=max_s)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    metrics_file = output_dir / "metrics.jsonl"
    best_pt_file = output_dir / "best.pt"
    summary_file = output_dir / "summary.json"
    success_file = output_dir / "_SUCCESS.json"

    best_val_r = -1.0
    best_epoch = 0

    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Starting Training ({args.epochs} epochs)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_acc = 0.0
        train_samples = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=True)

        for batch in pbar:
            targets = batch.to(device)  # [B, 12, 5000]
            B = targets.shape[0]

            x_source = targets[:, args.observed_lead : args.observed_lead + 1, :]
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                out = model(x_source, obs_lead_idx=args.observed_lead)
                preds = out["y_pred"]

                if cell_cfg["recon_loss"] == "l1":
                    loss = missing_lead_l1(preds, targets, observed_lead=args.observed_lead)
                else:
                    # Current objective: MSE on missing leads
                    missing_mask = torch.ones(12, dtype=torch.bool, device=device)
                    missing_mask[args.observed_lead] = False
                    loss = F.mse_loss(preds[:, missing_mask], targets[:, missing_mask])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_acc += loss.item() * B
            train_samples += B
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss_acc / train_samples

        # Validation
        val_metrics = evaluate_validation(model, val_loader, args.observed_lead, device)
        val_metrics["epoch"] = epoch
        val_metrics["train_recon_loss"] = avg_train_loss

        with open(metrics_file, "a") as f:
            f.write(json.dumps(val_metrics) + "\n")

        print(
            f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Missing r: {val_metrics['val_missing_pearson']:.4f} (p05: {val_metrics['val_missing_pearson_p05']:.4f}) | "
            f"Chest (V1-V6) r: {val_metrics['val_chest_pearson']:.4f} | "
            f"Precordial Δ r: {val_metrics['val_precordial_transition_pearson']:.4f} | "
            f"L1: {val_metrics['val_l1_recon']:.4f}"
        )

        if val_metrics["val_missing_pearson"] > best_val_r:
            best_val_r = val_metrics["val_missing_pearson"]
            best_epoch = epoch
            # Atomic save with fp16 weights to guarantee disk safety (<180 MB)
            fp16_state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
            tmp_path = best_pt_file.with_suffix(".tmp")
            torch.save({"model_state_dict": fp16_state, "epoch": epoch, "best_val_r": best_val_r}, tmp_path)
            tmp_path.replace(best_pt_file)

    # Save summary
    summary_data = {
        "cell": args.cell,
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val_missing_pearson": best_val_r,
        "epochs_completed": args.epochs,
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        **val_metrics,
    }
    summary_file.write_text(json.dumps(summary_data, indent=2))
    success_file.write_text(json.dumps(summary_data, indent=2))
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Run {run_name} completed successfully! Best Val r: {best_val_r:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
