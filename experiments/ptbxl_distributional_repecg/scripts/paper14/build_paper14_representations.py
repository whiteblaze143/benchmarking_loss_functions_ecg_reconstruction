#!/usr/bin/env python3
"""Build 5-view paired representations for Paper 14 ERM-vs-MMD experiment.

Generates:
  - representation_train.npz: Folds 1-6 (13,068 records * 5 views = 65,340 samples)
  - representation_val.npz: Fold 7 (2,176 records * 5 views = 10,880 samples)
  - representation_fold8.npz: Fold 8 (2,173 records * 5 views = 10,865 samples)

Environments:
  0: clean (reference)
  1: gain_0.8 (0.8x amplitude)
  2: gain_1.2 (1.2x amplitude)
  3: noise_20db (additive sensor noise, 20 dB SNR)
  4: resample_250hz (2x downsampling/upsampling along temporal/phase axis)

Keys in each output file:
  - linear: float32, (5*N, 16, 8)
  - kernel: float32, (5*N, 16, 256)
  - labels: float32, (5*N, 5)
  - ecg_ids: int64, (5*N,)
  - patient_ids: int64, (5*N,)
  - environment: int64, (5*N,)
  - record_ids: int64, (5*N,)
  - strat_fold: int64, (5*N,)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apply_transforms(
    x: torch.Tensor,
    seed: int = 42,
) -> dict[int, torch.Tensor]:
    """Apply the frozen 5-view acquisition perturbation bank.
    
    Args:
        x: Tensor of shape (N, T, C)
        seed: Random seed for noise generation
    
    Returns:
        dict mapping env_idx (0..4) to transformed tensor of shape (N, T, C)
    """
    N, T, C = x.shape
    rng = torch.Generator().manual_seed(seed)

    # 0. Clean reference
    x_clean = x.clone()

    # 1. Gain 0.8x
    x_gain08 = x * 0.8

    # 2. Gain 1.2x
    x_gain12 = x * 1.2

    # 3. Additive sensor noise (20 dB SNR)
    # SNR_dB = 20 * log10(std_signal / std_noise) => std_noise = std_signal / 10.0
    sig_std = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    noise_std = sig_std / 10.0
    noise = torch.randn(N, T, C, generator=rng, dtype=x.dtype, device=x.device) * noise_std
    x_noise = x + noise

    # 4. Bandwidth reduction (resample 500 -> 250 -> 500 Hz, 2x along phase/time axis)
    down = F.interpolate(x.transpose(1, 2), scale_factor=0.5, mode="linear", align_corners=False)
    x_resample = F.interpolate(down, size=T, mode="linear", align_corners=False).transpose(1, 2)

    return {
        0: x_clean,
        1: x_gain08,
        2: x_gain12,
        3: x_noise,
        4: x_resample,
    }


def _build_dataset(
    linear: np.ndarray,
    kernel: np.ndarray,
    labels: np.ndarray,
    ecg_ids: np.ndarray,
    patient_ids: np.ndarray,
    strat_fold: np.ndarray,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    N = len(ecg_ids)
    
    linear_t = torch.from_numpy(linear)
    kernel_t = torch.from_numpy(kernel)

    linear_views = _apply_transforms(linear_t, seed=seed)
    kernel_views = _apply_transforms(kernel_t, seed=seed + 100)

    # Assemble 5-view arrays ordered by environment
    all_linear = np.concatenate([linear_views[e].numpy() for e in range(5)], axis=0)
    all_kernel = np.concatenate([kernel_views[e].numpy() for e in range(5)], axis=0)
    all_labels = np.tile(labels, (5, 1))
    all_ecg_ids = np.tile(ecg_ids, 5)
    all_patient_ids = np.tile(patient_ids, 5)
    all_strat_fold = np.tile(strat_fold, 5)
    all_environment = np.repeat(np.arange(5, dtype=np.int64), N)
    all_record_ids = np.tile(np.arange(N, dtype=np.int64), 5)

    return {
        "linear": all_linear.astype(np.float32),
        "kernel": all_kernel.astype(np.float32),
        "labels": all_labels.astype(np.float32),
        "ecg_ids": all_ecg_ids.astype(np.int64),
        "patient_ids": all_patient_ids.astype(np.int64),
        "strat_fold": all_strat_fold.astype(np.int64),
        "environment": all_environment.astype(np.int64),
        "record_ids": all_record_ids.astype(np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-representations",
        type=Path,
        default=Path("outputs/paper02_kernel_mean/development_representations"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("../../data/ptbxl/ptbxl_database.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/paper14_representations/development_representations"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading source representations from {args.source_representations}...")
    with np.load(args.source_representations / "representation_train.npz") as item:
        train_src = {k: np.asarray(item[k]) for k in item.files}
    with np.load(args.source_representations / "representation_val.npz") as item:
        val_src = {k: np.asarray(item[k]) for k in item.files}

    print(f"Loading metadata from {args.metadata}...")
    df = pd.read_csv(args.metadata)
    fold_map = dict(zip(df.ecg_id, df.strat_fold))

    train_folds = np.array([fold_map[i] for i in train_src["ecg_ids"]])
    val_folds = np.array([fold_map[i] for i in val_src["ecg_ids"]])

    # Folds 1-6 for training fit
    mask_train = np.isin(train_folds, range(1, 7))
    # Fold 7 for validation & lambda tuning
    mask_val = (train_folds == 7)
    # Fold 8 for frozen test evaluation
    mask_fold8 = (val_folds == 8)

    print(f"Split breakdown:")
    print(f"  Train (Folds 1-6): {mask_train.sum()} records -> {mask_train.sum() * 5} views")
    print(f"  Val (Fold 7):      {mask_val.sum()} records -> {mask_val.sum() * 5} views")
    print(f"  Test (Fold 8):     {mask_fold8.sum()} records -> {mask_fold8.sum() * 5} views")

    print("\nBuilding Train (Folds 1-6) paired dataset...")
    train_data = _build_dataset(
        linear=train_src["linear"][mask_train],
        kernel=train_src["kernel"][mask_train],
        labels=train_src["labels"][mask_train],
        ecg_ids=train_src["ecg_ids"][mask_train],
        patient_ids=train_src["patient_ids"][mask_train],
        strat_fold=train_folds[mask_train],
        seed=args.seed,
    )
    train_path = args.output / "representation_train.npz"
    np.savez_compressed(train_path, **train_data)
    print(f"  Saved {train_path} ({os.path.getsize(train_path) / 1024**2:.1f} MB)")

    print("\nBuilding Val (Fold 7) paired dataset...")
    val_data = _build_dataset(
        linear=train_src["linear"][mask_val],
        kernel=train_src["kernel"][mask_val],
        labels=train_src["labels"][mask_val],
        ecg_ids=train_src["ecg_ids"][mask_val],
        patient_ids=train_src["patient_ids"][mask_val],
        strat_fold=train_folds[mask_val],
        seed=args.seed + 1000,
    )
    val_path = args.output / "representation_val.npz"
    np.savez_compressed(val_path, **val_data)
    print(f"  Saved {val_path} ({os.path.getsize(val_path) / 1024**2:.1f} MB)")

    print("\nBuilding Test (Fold 8) paired dataset...")
    fold8_data = _build_dataset(
        linear=val_src["linear"][mask_fold8],
        kernel=val_src["kernel"][mask_fold8],
        labels=val_src["labels"][mask_fold8],
        ecg_ids=val_src["ecg_ids"][mask_fold8],
        patient_ids=val_src["patient_ids"][mask_fold8],
        strat_fold=val_folds[mask_fold8],
        seed=args.seed + 2000,
    )
    fold8_path = args.output / "representation_fold8.npz"
    np.savez_compressed(fold8_path, **fold8_data)
    print(f"  Saved {fold8_path} ({os.path.getsize(fold8_path) / 1024**2:.1f} MB)")

    manifest = {
        "dataset": "ptbxl",
        "description": "Paper 14 5-view paired acquisition perturbation representations",
        "environments": {
            "0": "clean",
            "1": "gain_0.8",
            "2": "gain_1.2",
            "3": "noise_20db",
            "4": "resample_250hz",
        },
        "seed": args.seed,
        "splits": {
            "train_folds_1_6": {
                "records": int(mask_train.sum()),
                "samples": len(train_data["ecg_ids"]),
                "sha256": _sha256(train_path),
            },
            "val_fold_7": {
                "records": int(mask_val.sum()),
                "samples": len(val_data["ecg_ids"]),
                "sha256": _sha256(val_path),
            },
            "test_fold_8": {
                "records": int(mask_fold8.sum()),
                "samples": len(fold8_data["ecg_ids"]),
                "sha256": _sha256(fold8_path),
            },
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
