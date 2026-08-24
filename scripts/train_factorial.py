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
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
from scripts.train_mcma_3lead import MCMAModel, PTBXLDataset
from scripts.common_loss import CombinatorialCompositeLoss, pearson_loss
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

def split_content_root(path):
    root = hashlib.sha256()
    records = 0
    for item in sorted(Path(path).glob("*.pt"), key=lambda value: int(value.stem)):
        digest = file_sha256(item)
        root.update(f"{item.stem}:{item.stat().st_size}:{digest}\n".encode())
        records += 1
    return {"records": records, "content_root_sha256": root.hexdigest()}

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def _fsync_directory(path):
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

def atomic_torch_save(payload, destination):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

def atomic_write_json(payload, destination):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/factorial.pt")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data/ptb_xl/tensors")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--factorial_mask", type=str, required=True, help="7-digit mask, e.g. 1000000")
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches for smoke testing")
    parser.add_argument(
        "--training_contract",
        type=str,
        default="refine-logs/factorial_training_contract.json",
    )
    args = parser.parse_args()

    source_paths = [
        "scripts/train_factorial.py",
        "scripts/train_mcma_3lead.py",
        "scripts/common_loss.py",
        "scripts/experiment_provenance.py",
    ]
    source_provenance = code_provenance(_ROOT, source_paths)
    contract_path = (_ROOT / args.training_contract).resolve()
    contract = json.loads(contract_path.read_text())
    if contract.get("schema_version") != 1:
        raise RuntimeError("Unsupported factorial training contract schema")
    expected_bundle = contract["approved_source_bundle_sha256"]
    if source_provenance["source_bundle_sha256"] != expected_bundle:
        raise RuntimeError(
            "Training source bundle is not the pinned factorial contract: "
            f"observed={source_provenance['source_bundle_sha256']} "
            f"expected={expected_bundle}"
        )
    if source_provenance["source_file_sha256"] != contract["source_file_sha256"]:
        raise RuntimeError("Training source-file hashes disagree with the contract")
    if contract["state_precision"] != "float16":
        raise RuntimeError("This trainer requires the pinned float16 state policy")
    if contract["preprocessing"] != {
        "sample_rate_hz": 500,
        "units": "mV",
        "normalization": "none",
    }:
        raise RuntimeError("Unsupported preprocessing policy in training contract")
    seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ckpt_dir = "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    data_dir = os.path.abspath(args.data_dir)

    def assert_data_contract():
        for relative, expected_digest in contract["data_artifact_sha256"].items():
            observed_digest = file_sha256(_ROOT / relative)
            if observed_digest != expected_digest:
                raise RuntimeError(
                    f"Pinned data artifact changed: {relative} "
                    f"observed={observed_digest} expected={expected_digest}"
                )
        observed_name_size = {
            split: split_inventory_hash(Path(data_dir) / split)
            for split in ("train", "val", "test")
        }
        observed_content = {
            split: split_content_root(Path(data_dir) / split)
            for split in ("train", "val", "test")
        }
        if observed_name_size != contract["split_inventory_name_size_sha256"]:
            raise RuntimeError("PTB-XL tensor name/size inventory violates contract")
        if observed_content != contract["split_content_roots"]:
            raise RuntimeError("PTB-XL tensor byte-content roots violate contract")
        return observed_name_size, observed_content

    split_name_size_sha256, split_content_roots = assert_data_contract()
    print("Loading datasets...")
    train_dataset = PTBXLDataset(f"{data_dir}/train")
    val_dataset = PTBXLDataset(f"{data_dir}/val")
    
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, generator=loader_generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    wandb.init(project="ecg-reconstruction-ablation", name=args.run_name, config=vars(args))
    
    model = MCMAModel(in_channels=3, out_channels=12).to(device)
    reference_state = model.state_dict()
    state_schema = [
        {
            "key": key,
            "shape": list(tensor.shape),
            "stored_dtype": str(torch.float16),
            "reference_shape": list(tensor.shape),
            "reference_dtype": str(tensor.dtype),
        }
        for key, tensor in sorted(reference_state.items())
    ]
    observed_state_schema = hashlib.sha256(
        json.dumps(
            state_schema, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if observed_state_schema != contract["state_schema_sha256"]:
        raise RuntimeError(
            "Model/state precision schema disagrees with training contract"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    criterion = CombinatorialCompositeLoss(args.factorial_mask)
    scaler = torch.amp.GradScaler('cuda')
    
    epochs = args.epochs 
    
    best_selector = None
    best_epoch = None
    best_metrics = None
    best_provenance = None
    
    for epoch in range(epochs):
        model.train()
        t_stats = {k: 0.0 for k in ['loss', 'mse', 'corr', 'deriv', 'vcg', 'ed', 'lead', 'mmd']}
        
        print(f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
            if args.max_batches and batch_idx >= args.max_batches:
                break
                
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss, l_mse, l_corr, l_deriv, l_vcg, l_ed, l_lead, l_mmd = criterion(out, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            t_stats['loss'] += loss.item()
            t_stats['mse'] += l_mse.item()
            t_stats['corr'] += l_corr.item()
            t_stats['deriv'] += l_deriv.item()
            t_stats['vcg'] += l_vcg.item()
            t_stats['ed'] += l_ed.item()
            t_stats['lead'] += l_lead.item()
            t_stats['mmd'] += l_mmd.item()
            
        model.eval()
        v_stats = {k: 0.0 for k in ['loss', 'mse', 'corr', 'deriv', 'vcg', 'ed', 'lead', 'mmd', 'missing_mse', 'missing_pearson']}
        missing_indices = [2, 3, 4, 5, 6, 8, 9, 10, 11]
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                if args.max_batches and batch_idx >= args.max_batches:
                    break
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss, l_mse, l_corr, l_deriv, l_vcg, l_ed, l_lead, l_mmd = criterion(out, y)
                    
                v_stats['loss'] += loss.item()
                v_stats['mse'] += l_mse.item()
                v_stats['corr'] += l_corr.item()
                v_stats['deriv'] += l_deriv.item()
                v_stats['vcg'] += l_vcg.item()
                v_stats['ed'] += l_ed.item()
                v_stats['lead'] += l_lead.item()
                v_stats['mmd'] += l_mmd.item()
                v_stats['missing_mse'] += F.mse_loss(out[:, missing_indices], y[:, missing_indices]).item()
                v_stats['missing_pearson'] += (1.0 - pearson_loss(out[:, missing_indices], y[:, missing_indices]).item())
                
        train_batches = args.max_batches if args.max_batches else len(train_loader)
        val_batches = args.max_batches if args.max_batches else len(val_loader)
        
        # Normalize
        for k in t_stats: t_stats[k] /= train_batches
        for k in v_stats: v_stats[k] /= val_batches
        
        print(f"Train Loss: {t_stats['loss']:.4f} (MSE: {t_stats['mse']:.4f}, MMD: {t_stats['mmd']:.4f}, ED: {t_stats['ed']:.4f})")
        print(f"Val Loss:   {v_stats['loss']:.4f} (MSE: {v_stats['mse']:.4f}, MMD: {v_stats['mmd']:.4f}, ED: {v_stats['ed']:.4f})")
        
        wandb.log({
            "epoch": epoch + 1,
            **{f"train_{k}": v for k, v in t_stats.items()},
            **{f"val_{k}": v for k, v in v_stats.items()}
        })
        
        selector = (-v_stats['missing_mse'], v_stats['missing_pearson'])
        if best_selector is None or selector > best_selector:
            best_selector = selector
            best_epoch = epoch + 1
            best_metrics = {
                "val_missing_mse": v_stats['missing_mse'],
                "val_missing_pearson": v_stats['missing_pearson'],
                "val_all_lead_mse": v_stats['mse'],
                "val_composite": v_stats['loss'],
            }
            checkpoint_path = os.path.abspath(args.checkpoint_path)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            live_source_provenance = code_provenance(_ROOT, source_paths)
            if (
                live_source_provenance["source_bundle_sha256"]
                != source_provenance["source_bundle_sha256"]
            ):
                raise RuntimeError(
                    "Training sources changed after process startup; refusing "
                    "to write a checkpoint with misleading provenance"
                )
            assert_data_contract()
            provenance = {
                "seed": args.seed,
                "factorial_mask": args.factorial_mask,
                "loss_implementation": "scripts.common_loss",
                "preprocessing": {"sample_rate_hz": 500, "units": "mV", "normalization": "none"},
                "split_inventory_name_size_sha256": split_name_size_sha256,
                "split_content_roots": split_content_roots,
                "architecture_revision": source_provenance["git_commit"],
                "architecture_provenance": source_provenance,
                "training_contract": contract,
                "optimizer": "AdamW(lr=3e-4,weight_decay=1e-4)",
                "checkpoint_selector": "lowest_missing_lead_val_mse_then_highest_missing_lead_val_pearson",
            }
            model_state = {k: v.to(torch.float16) for k, v in model.state_dict().items()}
            atomic_torch_save(
                {"model_state_dict": model_state, "provenance": provenance},
                checkpoint_path,
            )
            best_provenance = provenance
            print(f"Saved new best checkpoint (float16): {checkpoint_path}")

    if best_metrics is None or best_provenance is None:
        raise RuntimeError("Training completed without selecting a checkpoint")
    assert_data_contract()
    metadata_path = Path(args.metadata_path) if args.metadata_path else Path(args.checkpoint_path).with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint_path).resolve()
    atomic_write_json({
        "schema_version": 3,
        "run_name": args.run_name,
        "family": "unet",
        "seed": args.seed,
        "factorial_mask": args.factorial_mask,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "state_precision": "float16",
        "split_inventory_name_size_sha256": best_provenance[
            "split_inventory_name_size_sha256"
        ],
        "split_content_roots": best_provenance["split_content_roots"],
        "preprocessing": best_provenance["preprocessing"],
        "architecture_revision": best_provenance["architecture_revision"],
        "architecture_provenance": best_provenance["architecture_provenance"],
        "checkpoint_selector": best_provenance["checkpoint_selector"],
        "provenance": best_provenance,
    }, metadata_path)
            
    wandb.finish()

if __name__ == "__main__":
    main()
