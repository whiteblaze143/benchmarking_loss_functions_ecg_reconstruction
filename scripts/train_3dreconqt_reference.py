#!/usr/bin/env python3
"""
Training Script for 3DRECON-QT Architectural Reference (3DRECONQT_REFERENCE).

Faithfully adheres to Ansari et al. (2026) paper settings:
  - Architecture: SE-ResNeXt-1D -> Feature Fusion (Z1/Z2) -> Shared 1D Conv Decoder
  - Spatial Conditioning: Multiplicative query gating with Panorama 12-D ThetaEncoder
  - Optimizer: SGD(lr=1e-3, momentum=0.9, weight_decay=1e-5)
  - Scheduler: CosineAnnealingWarmRestarts(T_0=10, T_mult=1, eta_min=1e-6)
  - Batch size: 128 (physical batch 32 with 4 gradient accumulation steps)
  - Loss: Full 12-lead L1 loss (no Lead I overwriting)
  - Selection: Validation missing-11 Pearson r_missing on PTB-XL fold 9
"""

import argparse
import datetime as dt
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from unified_latents.engineering.models.reconqt_reference import (
    LEAD_NAMES,
    STANDARD_ANGLES,
    ThreeDReconQTReference,
    extract_source_vector,
    normalize_record,
)

CELL_CONFIGS = {
    # Primary Reference Cells: Sampled Target Query (closest to public Panorama / Figure 2)
    "RQ1Q_theta_l1": {
        "code_mode": "theta",
        "source_mode": "lead_I",
        "target_training": "sampled_query",
        "description": "PRIMARY REFERENCE: true physical spherical angles, sampled single-query training, full-12 eval, Lead I",
    },
    "RQ2Q_permuted_theta_l1": {
        "code_mode": "permuted_theta",
        "source_mode": "lead_I",
        "target_training": "sampled_query",
        "description": "PRIMARY SPATIAL CONTROL: fixed permuted derangement (arange(12)+5)%12, sampled single-query training, full-12 eval, Lead I",
    },
    "RQ3Q_learned12_l1": {
        "code_mode": "learned",
        "source_mode": "lead_I",
        "target_training": "sampled_query",
        "description": "LEARNED TARGET CONTROL: 12-D learned continuous query codes, sampled single-query training, full-12 eval, Lead I",
    },
    "RQ4Q_random12_l1": {
        "code_mode": "random_fixed",
        "source_mode": "lead_I",
        "target_training": "sampled_query",
        "description": "FIXED RANDOM CONTROL: fixed standardized random 12-D codes, sampled single-query training, full-12 eval, Lead I",
    },
    "RQ0Q_constant_l1": {
        "code_mode": "constant",
        "source_mode": "lead_I",
        "target_training": "sampled_query",
        "description": "IDENTIFIABILITY NEGATIVE CONTROL: zero condition vector, sampled single-query training, full-12 eval, Lead I",
    },
    # Stage R2 Source Vector Interaction (ICM Subcutaneous Vector V3 - V2)
    "RQ1Q_theta_l1_v3_v2": {
        "code_mode": "theta",
        "source_mode": "v3_minus_v2",
        "target_training": "sampled_query",
        "description": "PRECORDIAL ICM REFERENCE: true physical spherical angles from subcutaneous vector (V3 - V2), sampled single-query training",
    },
    "RQ2Q_permuted_theta_l1_v3_v2": {
        "code_mode": "permuted_theta",
        "source_mode": "v3_minus_v2",
        "target_training": "sampled_query",
        "description": "PRECORDIAL ICM CONTROL: fixed permuted derangement from subcutaneous vector (V3 - V2), sampled single-query training",
    },
    "RQ3Q_learned12_l1_v3_v2": {
        "code_mode": "learned",
        "source_mode": "v3_minus_v2",
        "target_training": "sampled_query",
        "description": "PRECORDIAL ICM LEARNED: learned 12-D codes from subcutaneous vector (V3 - V2), sampled single-query training",
    },
    # Secondary Simultaneous-12 Architecture Variants (Diagnostic contrast)
    "RQ1A12_theta_l1": {
        "code_mode": "theta",
        "source_mode": "lead_I",
        "target_training": "simultaneous_12",
        "description": "SECONDARY SIMULTANEOUS: true physical angles, simultaneous 12-lead joint decoding & loss, Lead I",
    },
    "RQ2A12_permuted_theta_l1": {
        "code_mode": "permuted_theta",
        "source_mode": "lead_I",
        "target_training": "simultaneous_12",
        "description": "SECONDARY SIMULTANEOUS CONTROL: permuted derangement, simultaneous 12-lead joint decoding & loss, Lead I",
    },
    "RQ3A12_learned12_l1": {
        "code_mode": "learned",
        "source_mode": "lead_I",
        "target_training": "simultaneous_12",
        "description": "SECONDARY SIMULTANEOUS LEARNED: learned 12-D codes, simultaneous 12-lead joint decoding & loss, Lead I",
    },
    "RQ4A12_random12_l1": {
        "code_mode": "random_fixed",
        "source_mode": "lead_I",
        "target_training": "simultaneous_12",
        "description": "SECONDARY VARIANT: permuted angles, simultaneous 12-lead joint training, Lead I",
        "code_mode": "permuted_theta",
        "source_mode": "lead_I",
        "target_training": "simultaneous_12",
    },
}


class IndexedPTBXLDataset(Dataset):
    """Loads PTB-XL preprocessed tensors along with integer sample index and ECG ID."""

    def __init__(self, folder: Path, max_samples: Optional[int] = None):
        self.folder = Path(folder)
        self.files = sorted(list(self.folder.glob("*.pt")))
        if max_samples is not None:
            self.files = self.files[:max_samples]
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt files found in {self.folder}")
        self.sample_ids = [int(p.stem) for p in self.files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.files[idx]
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "waveform" in obj:
            waveform = torch.as_tensor(obj["waveform"], dtype=torch.float32)
        elif isinstance(obj, torch.Tensor):
            waveform = obj.float()
        else:
            raise TypeError(f"Unexpected data type in {path}: {type(obj)}")
        return waveform[..., :5000], idx


# Backwards compatibility alias
PTBXLPreprocessedDataset = IndexedPTBXLDataset


def get_eligible_targets(source_mode: str) -> List[int]:
    """
    Returns eligible supervised target leads for training.
    For Lead I (observed = 0), returns [1, 2, ..., 11] (the 11 missing leads).
    For derived ICM vector (V3 - V2), returns [0, 1, ..., 11] (all 12 standard leads).
    """
    if source_mode == "lead_I":
        # Never train on observed source lead (Lead I = 0)
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    elif source_mode == "v3_minus_v2":
        return list(range(12))
    else:
        raise ValueError(f"Unknown source mode: {source_mode}")


def generate_target_schedule(
    n_samples: int,
    max_epochs: int,
    eligible_targets: List[int],
    seed: int = 42,
) -> torch.Tensor:
    """
    Pre-computes a deterministic target schedule of shape [max_epochs, n_samples].
    0-indexed across epochs [0, ..., max_epochs - 1] and samples [0, ..., n_samples - 1].
    Guarantees that RQ1Q and RQ2Q receive the exact same target lead for every
    sample at every epoch.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    schedule = torch.empty((max_epochs, n_samples), dtype=torch.long)
    eligible_t = torch.tensor(eligible_targets, dtype=torch.long)
    num_eligible = len(eligible_targets)

    for epoch_idx in range(max_epochs):
        choice_indices = torch.randint(0, num_eligible, size=(n_samples,), generator=g)
        schedule[epoch_idx] = eligible_t[choice_indices]

    return schedule


def compute_lead_pearson(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Vectorized Pearson correlation per sample along temporal dimension."""
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
    source_mode: str,
    device: torch.device,
    normalization_mode: str = "strict_deployable",
) -> Dict[str, float]:
    """Exhaustive evaluation on PTB-XL validation split."""
    model.eval()

    all_12_pearsons = []
    missing_11_pearsons = []
    chest_pearsons = []
    transition_pearsons = []
    total_l1 = 0.0
    total_mse = 0.0
    total_samples = 0

    # Physical inconsistency metrics
    viol_III = []
    viol_aVR = []
    viol_aVL = []
    viol_aVF = []

    # Lead indices in standard order
    idx_I = LEAD_NAMES.index("I")
    idx_II = LEAD_NAMES.index("II")
    idx_III = LEAD_NAMES.index("III")
    idx_aVR = LEAD_NAMES.index("aVR")
    idx_aVL = LEAD_NAMES.index("aVL")
    idx_aVF = LEAD_NAMES.index("aVF")
    chest_indices = [LEAD_NAMES.index(f"V{k}") for k in range(1, 7)]
    missing_indices = [l for l in range(12) if l != idx_I] if source_mode == "lead_I" else list(range(12))

    with torch.no_grad():
        for batch in val_loader:
            targets_raw = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
            B = targets_raw.shape[0]

            source_raw = extract_source_vector(targets_raw, source_mode=source_mode)
            source_norm, targets_norm, _ = normalize_record(
                source_raw, targets_raw, mode=normalization_mode
            )

            out = model(source_norm)
            preds = out["y_pred"].float()  # [B, 12, 5000]

            # 1. All-12 mean Pearson
            r_all_12 = []
            for l in range(12):
                r_l = compute_lead_pearson(preds[:, l, :], targets_norm[:, l, :])
                r_all_12.append(r_l)
            r_all_12_stack = torch.stack(r_all_12, dim=1)  # [B, 12]
            all_12_pearsons.extend(r_all_12_stack.mean(dim=1).cpu().tolist())

            # 2. Missing-11 mean Pearson
            r_missing_stack = r_all_12_stack[:, missing_indices]  # [B, len(missing_indices)]
            missing_11_pearsons.extend(r_missing_stack.mean(dim=1).cpu().tolist())

            # 3. Precordial V1-V6 mean Pearson
            r_chest_stack = r_all_12_stack[:, chest_indices]  # [B, 6]
            chest_pearsons.extend(r_chest_stack.mean(dim=1).cpu().tolist())

            # 4. Precordial delta transition progression
            r_trans = []
            for k in range(5):
                pred_diff = preds[:, chest_indices[k + 1], :] - preds[:, chest_indices[k], :]
                tgt_diff = targets_norm[:, chest_indices[k + 1], :] - targets_norm[:, chest_indices[k], :]
                r_trans.append(compute_lead_pearson(pred_diff, tgt_diff))
            r_trans_stack = torch.stack(r_trans, dim=1)  # [B, 5]
            transition_pearsons.extend(r_trans_stack.mean(dim=1).cpu().tolist())

            # 5. Full 12-lead reconstruction L1 & MSE
            l1_loss = F.l1_loss(preds, targets_norm, reduction="sum")
            mse_loss = F.mse_loss(preds, targets_norm, reduction="sum")
            total_l1 += l1_loss.item()
            total_mse += mse_loss.item()
            total_samples += B * 12 * 5000

            # 6. Physical inconsistency violations on predictions
            p_I = preds[:, idx_I, :]
            p_II = preds[:, idx_II, :]
            p_III = preds[:, idx_III, :]
            p_aVR = preds[:, idx_aVR, :]
            p_aVL = preds[:, idx_aVL, :]
            p_aVF = preds[:, idx_aVF, :]

            v_III = torch.abs(p_II - p_I - p_III).mean().item() * 1000.0  # in uV approx
            v_aVR = torch.abs(p_aVR + (p_I + p_II) / 2.0).mean().item() * 1000.0
            v_aVL = torch.abs(p_aVL - p_I + p_II / 2.0).mean().item() * 1000.0
            v_aVF = torch.abs(p_aVF - p_II + p_I / 2.0).mean().item() * 1000.0

            viol_III.append(v_III)
            viol_aVR.append(v_aVR)
            viol_aVL.append(v_aVL)
            viol_aVF.append(v_aVF)

    return {
        "val_all_12_pearson": float(np.mean(all_12_pearsons)),
        "val_missing_pearson": float(np.mean(missing_11_pearsons)),
        "val_missing_pearson_p05": float(np.quantile(missing_11_pearsons, 0.05)),
        "val_chest_pearson": float(np.mean(chest_pearsons)),
        "val_precordial_transition_pearson": float(np.mean(transition_pearsons)),
        "val_l1_recon": float(total_l1 / total_samples),
        "val_mse_recon": float(total_mse / total_samples),
        "val_mean_III_violation": float(np.mean(viol_III)),
        "val_mean_aVR_violation": float(np.mean(viol_aVR)),
        "val_mean_aVL_violation": float(np.mean(viol_aVL)),
        "val_mean_aVF_violation": float(np.mean(viol_aVF)),
    }


def main():
    p = argparse.ArgumentParser(description="Train 3DRECON-QT Architectural Reference Cell")
    p.add_argument("--cell", required=True, choices=list(CELL_CONFIGS.keys()), help="Reference cell name")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--epochs", type=int, default=100, help="Maximum epochs (default 100)")
    p.add_argument("--patience", type=int, default=15, help="Early stopping patience (default 15)")
    p.add_argument("--physical-batch-size", type=int, default=32, help="Physical batch size (default 32)")
    p.add_argument("--effective-batch-size", type=int, default=128, help="Effective batch size (default 128)")
    p.add_argument("--lr", type=float, default=1e-3, help="Initial SGD learning rate (default 1e-3)")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default 0.9)")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="SGD weight decay (default 1e-5)")
    p.add_argument("--data-dir", default="data/ptb_xl/tensors", help="Path to ptb_xl tensor directory")
    p.add_argument("--runs-dir", default="refine-logs/3dreconqt_reference/runs", help="Output directory for runs")
    p.add_argument("--normalization", default="strict_deployable", choices=["strict_deployable", "per_channel_record", "shared_12lead_record"])
    p.add_argument("--target-training", default=None, choices=["sampled_query", "simultaneous_12"], help="Override target training mode (default from cell config)")
    p.add_argument("--smoke", action="store_true", help="Run 2-batch smoke test")
    args = p.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.enabled = False  # Prevent cuDNN 9.x 1D conv crash

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    cell_cfg = CELL_CONFIGS[args.cell]
    target_training_mode = args.target_training if args.target_training is not None else cell_cfg.get("target_training", "sampled_query")
    run_name = f"{args.cell}_s{args.seed}"
    output_dir = Path(args.runs_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    accum_steps = max(1, args.effective_batch_size // args.physical_batch_size)
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Launching Run: {run_name}")
    print(f"  Description:           {cell_cfg['description']}")
    print(f"  Code Mode:             {cell_cfg['code_mode']}")
    print(f"  Source Mode:           {cell_cfg['source_mode']}")
    print(f"  Target Training Mode:  {target_training_mode} ({'PRIMARY' if target_training_mode == 'sampled_query' else 'SECONDARY'})")
    print(f"  Normalization:         {args.normalization}")
    print(f"  Effective Batch Size:  {args.effective_batch_size} (Physical: {args.physical_batch_size} x {accum_steps} Accum Steps)")
    print(f"  SGD Optimizer:         lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay}")

    # Build Model
    model = ThreeDReconQTReference(
        input_channels=1,
        latent_channels=256,
        theta_hidden=128,
        attention_heads=8,
        code_mode=cell_cfg["code_mode"],
        target_length=5000,
        enable_qt_head=False,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model Parameters:      {total_params:,} total, {trainable_params:,} trainable")

    # Save Provenance & Config
    provenance_record = {
        "model_family": "3dreconqt_reference",
        "fidelity_status": "architecture_reference",
        "cell": args.cell,
        "run_name": run_name,
        "seed": args.seed,
        "source_mode": cell_cfg["source_mode"],
        "code_mode": cell_cfg["code_mode"],
        "target_training": target_training_mode,
        "normalization": args.normalization,
        "angle_table": STANDARD_ANGLES.tolist(),
        "lead_names": list(LEAD_NAMES),
        "theta_permutation": ((torch.arange(12) + 5) % 12).tolist(),
        "filter": {
            "low_hz": 0.5,
            "high_hz": 40.0,
            "order": 5,
            "zero_phase": True,
        },
        "encoder": {
            "family": "SE-ResNeXt-1D",
            "layers": [3, 4, 6, 3],
            "groups": 32,
            "width_per_group": 4,
            "paper_confirmed_depth": False,
        },
        "latent_channels": 256,
        "theta_hidden": 128,
        "optimizer": "SGD",
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "effective_batch_size": args.effective_batch_size,
        "physical_batch_size": args.physical_batch_size,
        "accum_steps": accum_steps,
        "max_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "selection_policy": "val_missing_pearson",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (output_dir / "config.json").write_text(json.dumps(provenance_record, indent=2))

    eligible_targets = get_eligible_targets(cell_cfg["source_mode"])
    print(f"  Eligible Target Leads ({len(eligible_targets)}): {eligible_targets} ({[LEAD_NAMES[l] for l in eligible_targets]})")

    # Datasets (folds 1-8 train, fold 9 val; fold 10 untouched test set)
    data_dir = Path(args.data_dir)
    max_s = 64 if args.smoke else None
    train_ds = IndexedPTBXLDataset(data_dir / "train", max_samples=max_s)
    val_ds = IndexedPTBXLDataset(data_dir / "val", max_samples=max_s)
    val_loader = DataLoader(
        val_ds, batch_size=args.physical_batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    n_train_samples = len(train_ds)
    n_val_samples = len(val_ds)

    # Verify fold distribution against ptbxl_database.csv
    meta_csv = Path("data/ptb_xl/ptbxl_database.csv")
    if meta_csv.exists() and not args.smoke:
        df_meta = pd.read_csv(meta_csv).set_index("ecg_id")
        train_folds = df_meta.loc[train_ds.sample_ids, "strat_fold"].value_counts().sort_index().to_dict()
        val_folds = df_meta.loc[val_ds.sample_ids, "strat_fold"].value_counts().sort_index().to_dict()
        assert set(train_folds.keys()) == set(range(1, 9)), f"Train folds mismatch: {set(train_folds.keys())}"
        assert set(val_folds.keys()) == {9}, f"Val folds mismatch: {set(val_folds.keys())}"
        print(f"  len(train_dataset):          {n_train_samples}")
        print(f"  len(val_dataset):            {n_val_samples}")
        print(f"  train_dataset fold counts:   {train_folds}")
        print(f"  val_dataset fold counts:     {val_folds}")
    else:
        print(f"  len(train_dataset):          {n_train_samples}")
        print(f"  len(val_dataset):            {n_val_samples}")

    # Pre-generate deterministic target schedule for all epochs and all training samples
    # Shape is strictly (max_epochs, len(train_dataset)), 0-indexed across epochs and samples
    target_schedule = generate_target_schedule(
        n_samples=n_train_samples,
        max_epochs=args.epochs,
        eligible_targets=eligible_targets,
        seed=args.seed,
    )

    assert target_schedule.shape == (args.epochs, n_train_samples), f"Schedule shape mismatch: {target_schedule.shape}"
    print(f"  target_schedule.shape:       {tuple(target_schedule.shape)}")
    print(f"  max_epochs:                  {args.epochs}")
    print(f"  number of schedule entries:  {target_schedule.numel():,}")

    # Formal assertion over the ENTIRE schedule (no slices or exclusions)
    if cell_cfg["source_mode"] == "lead_I":
        assert (target_schedule == 0).sum().item() == 0, "FATAL: Observed Lead I found in target schedule!"

    # Target counts report across the actual full training schedule
    sched_flat = target_schedule.flatten().tolist()
    target_counts = {LEAD_NAMES[l]: sched_flat.count(l) for l in eligible_targets}
    print(f"  Actual Full Training Target Lead Counts ({target_schedule.numel():,} entries):")
    print(f"  {target_counts}")

    provenance_record["eligible_targets"] = eligible_targets
    provenance_record["eligible_target_names"] = [LEAD_NAMES[l] for l in eligible_targets]
    provenance_record["target_schedule_seed"] = args.seed
    provenance_record["target_schedule_shape"] = list(target_schedule.shape)
    provenance_record["len_train_dataset"] = n_train_samples
    provenance_record["len_val_dataset"] = n_val_samples
    (output_dir / "config.json").write_text(json.dumps(provenance_record, indent=2))

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=1,
        eta_min=1e-6,
    )

    metrics_file = output_dir / "metrics.jsonl"
    best_pt_file = output_dir / "best.pt"
    summary_file = output_dir / "summary.json"
    success_file = output_dir / "_SUCCESS.json"

    best_val_r = -1.0
    best_epoch = 0
    patience_counter = 0

    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Starting Training ({args.epochs} max epochs)...")
    for epoch in range(1, args.epochs + 1):
        epoch_idx = epoch - 1
        model.train()
        train_loss_acc = 0.0
        train_samples = 0
        optimizer.zero_grad()

        # Seeded RandomSampler ensures bitwise identical mini-batch ordering across epochs between runs
        sampler_g = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.physical_batch_size,
            sampler=torch.utils.data.RandomSampler(train_ds, generator=sampler_g),
            num_workers=4,
            pin_memory=True,
        )

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=True)
        for step, batch in enumerate(pbar):
            targets_raw, sample_indices = batch
            targets_raw = targets_raw.to(device)  # [B, 12, 5000]
            B = targets_raw.shape[0]

            source_raw = extract_source_vector(targets_raw, source_mode=cell_cfg["source_mode"])
            source_norm, targets_norm, _ = normalize_record(
                source_raw, targets_raw, mode=args.normalization
            )

            if target_training_mode == "sampled_query":
                # Deterministic scheduled target lead lookup for these exact samples at this exact epoch (0-indexed)
                target_idx = target_schedule[epoch_idx, sample_indices].to(device)
                target_waveform = targets_norm[torch.arange(B, device=device), target_idx, :]  # [B, 5000]
                out = model(source_norm, target_leads=target_idx)
                pred = out["y_pred"]  # [B, 5000]
                loss = F.l1_loss(pred, target_waveform)
            else:
                # Secondary simultaneous-12 variant
                out = model(source_norm, target_leads=None)
                preds = out["y_pred"]  # [B, 12, 5000]
                loss = F.l1_loss(preds, targets_norm)

            loss_scaled = loss / accum_steps
            loss_scaled.backward()

            train_loss_acc += loss.item() * B
            train_samples += B

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})


        scheduler.step()
        avg_train_loss = train_loss_acc / train_samples

        # Validation
        val_metrics = evaluate_validation(
            model,
            val_loader,
            source_mode=cell_cfg["source_mode"],
            device=device,
            normalization_mode=args.normalization,
        )
        val_metrics["epoch"] = epoch
        val_metrics["train_recon_loss"] = avg_train_loss
        val_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]

        with open(metrics_file, "a") as f:
            f.write(json.dumps(val_metrics) + "\n")

        print(
            f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"All-12 r: {val_metrics['val_all_12_pearson']:.4f} | "
            f"Missing r: {val_metrics['val_missing_pearson']:.4f} (p05: {val_metrics['val_missing_pearson_p05']:.4f}) | "
            f"Chest (V1-V6) r: {val_metrics['val_chest_pearson']:.4f} | "
            f"Precordial Δ r: {val_metrics['val_precordial_transition_pearson']:.4f} | "
            f"L1: {val_metrics['val_l1_recon']:.4f}"
        )

        # Model selection on missing-11 Pearson
        if val_metrics["val_missing_pearson"] > best_val_r:
            best_val_r = val_metrics["val_missing_pearson"]
            best_epoch = epoch
            patience_counter = 0

            # Atomic save with fp16 weights to guarantee minimal disk footprint (<55 MB)
            fp16_state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
            tmp_path = best_pt_file.with_suffix(".tmp")
            checkpoint_data = {
                "model_state_dict": fp16_state,
                "epoch": epoch,
                "best_val_missing_pearson": best_val_r,
                "provenance": provenance_record,
            }
            torch.save(checkpoint_data, tmp_path)
            tmp_path.replace(best_pt_file)
        else:
            patience_counter += 1
            if patience_counter >= args.patience and not args.smoke:
                print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Early stopping triggered after {args.patience} epochs without improvement.")
                break

    # Save summary
    summary_data = {
        "cell": args.cell,
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val_missing_pearson": best_val_r,
        "epochs_completed": epoch,
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        **val_metrics,
    }
    summary_file.write_text(json.dumps(summary_data, indent=2))
    success_file.write_text(json.dumps(summary_data, indent=2))
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Run {run_name} completed! Best Missing r: {best_val_r:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
