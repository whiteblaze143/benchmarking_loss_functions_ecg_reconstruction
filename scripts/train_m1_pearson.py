#!/usr/bin/env python3
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()

import os
import json
import random
import hashlib
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
from scripts.train_mcma_3lead import MCMAModel, PTBXLDataset
from scripts.common_loss import CompositeLoss, FactorialLossConfig, PaperParityCompositeLoss, pearson_loss
from scripts.experiment_provenance import code_provenance

import argparse
import wandb


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def split_inventory_hash(path):
    digest = hashlib.sha256()
    for item in sorted(Path(path).glob("*.pt"), key=lambda value: int(value.stem)):
        digest.update(f"{item.stem}:{item.stat().st_size}\n".encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    parser.add_argument("--lambda_corr", type=float, default=0.1)
    parser.add_argument("--lambda_mmd", type=float, default=0.0)
    parser.add_argument("--lambda_deriv", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/M1_pearson_seed42.pt")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data/ptb_xl/tensors")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--factorial_mask", choices=[f"{value:04b}" for value in range(16)] + [f"{value:05b}" for value in range(16, 32)])
    parser.add_argument("--loss_protocol", choices=["extended", "paper_parity"], default="extended")
    parser.add_argument("--metadata_path", type=str, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ckpt_dir = "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    data_dir = os.path.abspath(args.data_dir)
    print("Loading datasets...")
    train_dataset = PTBXLDataset(f"{data_dir}/train")
    val_dataset = PTBXLDataset(f"{data_dir}/val")
    
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, generator=loader_generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    wandb.init(project="ecg-reconstruction-ablation", name=args.run_name, config=vars(args))
    
    model = MCMAModel(in_channels=3, out_channels=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    # Using the parsed arguments
    if args.loss_protocol == "paper_parity":
        criterion = PaperParityCompositeLoss(args.factorial_mask or "1111")
        recorded_weights = PaperParityCompositeLoss.WEIGHTS
        recorded_normalizers = PaperParityCompositeLoss.NORMALIZERS
    else:
        if args.factorial_mask:
            config = FactorialLossConfig.from_mask(
                args.factorial_mask,
                lambda_mse=args.lambda_mse,
                lambda_corr=args.lambda_corr,
                lambda_mmd=args.lambda_mmd,
                lambda_deriv=args.lambda_deriv,
            )
            effective = config.effective_weights()
        else:
            effective = {
                "lambda_mse": args.lambda_mse,
                "lambda_corr": args.lambda_corr,
                "lambda_mmd": args.lambda_mmd,
                "lambda_deriv": args.lambda_deriv,
            }
        criterion = CompositeLoss(**effective)
        recorded_weights = effective
        recorded_normalizers = None
    scaler = torch.amp.GradScaler('cuda')
    
    epochs = args.epochs 
    
    best_selector = None
    best_epoch = None
    best_metrics = None
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_mse = 0
        train_corr = 0
        train_mmd = 0
        train_deriv = 0
        print(f"Epoch {epoch+1}/{epochs}")
        for x, y in tqdm(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss, l_mse, l_corr, l_mmd, l_deriv = criterion(out, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_mse += l_mse.item()
            train_corr += l_corr.item()
            train_mmd += l_mmd.item()
            train_deriv += l_deriv.item()
            
        model.eval()
        val_loss = 0
        val_mse = 0
        val_corr = 0
        val_mmd = 0
        val_deriv = 0
        val_missing_mse = 0
        val_missing_pearson = 0
        missing_indices = [2, 3, 4, 5, 6, 8, 9, 10, 11]
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss, l_mse, l_corr, l_mmd, l_deriv = criterion(out, y)
                val_loss += loss.item()
                val_mse += l_mse.item()
                val_corr += l_corr.item()
                val_mmd += l_mmd.item()
                val_deriv += l_deriv.item()
                val_missing_mse += F.mse_loss(out[:, missing_indices], y[:, missing_indices]).item()
                val_missing_pearson += (1.0 - pearson_loss(out[:, missing_indices], y[:, missing_indices]).item())
                
        t_loss = train_loss / len(train_loader)
        t_mse = train_mse / len(train_loader)
        t_corr = train_corr / len(train_loader)
        t_mmd = train_mmd / len(train_loader)
        t_deriv = train_deriv / len(train_loader)
        v_loss = val_loss / len(val_loader)
        v_mse = val_mse / len(val_loader)
        v_corr = val_corr / len(val_loader)
        v_mmd = val_mmd / len(val_loader)
        v_deriv = val_deriv / len(val_loader)
        v_missing_mse = val_missing_mse / len(val_loader)
        v_missing_pearson = val_missing_pearson / len(val_loader)
        
        print(f"Train Loss: {t_loss:.4f} (MSE: {t_mse:.4f}, Corr: {t_corr:.4f}, MMD: {t_mmd:.4f}, Deriv: {t_deriv:.4f})")
        print(f"Val Loss:   {v_loss:.4f} (MSE: {v_mse:.4f}, Corr: {v_corr:.4f}, MMD: {v_mmd:.4f}, Deriv: {v_deriv:.4f})")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": t_loss,
            "train_mse": t_mse,
            "train_corr": t_corr,
            "train_mmd": t_mmd,
            "train_deriv": t_deriv,
            "val_loss": v_loss,
            "val_mse": v_mse,
            "val_corr": v_corr,
            "val_mmd": v_mmd,
            "val_deriv": v_deriv,
        })
        
        selector = (-v_missing_mse, v_missing_pearson)
        if best_selector is None or selector > best_selector:
            best_selector = selector
            best_epoch = epoch + 1
            best_metrics = {
                "val_missing_mse": v_missing_mse,
                "val_missing_pearson": v_missing_pearson,
                "val_all_lead_mse": v_mse,
                "val_composite": v_loss,
            }
            checkpoint_path = os.path.abspath(args.checkpoint_path)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            source_provenance = code_provenance(_ROOT, [
                "scripts/train_m1_pearson.py",
                "scripts/train_mcma_3lead.py",
                "scripts/common_loss.py",
            ])
            provenance = {
                "seed": args.seed,
                "factorial_mask": args.factorial_mask,
                "loss_protocol": args.loss_protocol,
                "loss_implementation": "scripts.common_loss",
                "mmd_implementation": (
                    "paper_parity_rational_quadratic"
                    if args.loss_protocol == "paper_parity"
                    else "adaptive_multiscale_rbf_mean_squared_distance_v2"
                ),
                "weights": recorded_weights,
                "normalizers": recorded_normalizers,
                "preprocessing": {"sample_rate_hz": 500, "units": "mV", "normalization": "none"},
                "split_inventory_sha256": {
                    split: split_inventory_hash(Path(data_dir) / split)
                    for split in ("train", "val", "test")
                },
                "architecture_revision": subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True, text=True, check=True
                ).stdout.strip(),
                "architecture_provenance": source_provenance,
                "optimizer": "AdamW(lr=3e-4,weight_decay=1e-4)",
                "scheduler": None,
                "checkpoint_selector": "lowest_missing_lead_val_mse_then_highest_missing_lead_val_pearson",
            }
            torch.save({"model_state_dict": model.state_dict(), "provenance": provenance}, checkpoint_path)
            print(f"Saved new best checkpoint: {checkpoint_path}")

    metadata_path = Path(args.metadata_path) if args.metadata_path else Path(args.checkpoint_path).with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    source_provenance = code_provenance(_ROOT, [
        "scripts/train_m1_pearson.py",
        "scripts/train_mcma_3lead.py",
        "scripts/common_loss.py",
    ])
    metadata_path.write_text(json.dumps({
        "schema_version": 2,
        "run_name": args.run_name,
        "family": "unet",
        "seed": args.seed,
        "factorial_mask": args.factorial_mask,
        "loss_protocol": args.loss_protocol,
        "mmd_implementation": (
            "paper_parity_rational_quadratic"
            if args.loss_protocol == "paper_parity"
            else "adaptive_multiscale_rbf_mean_squared_distance_v2"
        ),
        "weights": recorded_weights,
        "normalizers": recorded_normalizers,
        "data_dir": data_dir,
        "split_inventory_sha256": {
            "train": split_inventory_hash(Path(data_dir) / "train"),
            "val": split_inventory_hash(Path(data_dir) / "val"),
            "test": split_inventory_hash(Path(data_dir) / "test"),
        },
        "architecture_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True, text=True, check=True).stdout.strip(),
        "architecture_provenance": source_provenance,
        "optimizer": "AdamW(lr=3e-4,weight_decay=1e-4)",
        "scheduler": None,
        "checkpoint_selector": "lowest_missing_lead_val_mse_then_highest_missing_lead_val_pearson",
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "checkpoint": os.path.abspath(args.checkpoint_path),
    }, indent=2, allow_nan=False))
            
    wandb.finish()

if __name__ == "__main__":
    main()
