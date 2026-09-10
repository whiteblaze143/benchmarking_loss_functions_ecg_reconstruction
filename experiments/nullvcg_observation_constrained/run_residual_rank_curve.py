#!/usr/bin/env python3
"""Characterize ranks 1..6 of the training-derived residual PCA oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from geometry import residual_subspace_oracle  # noqa: E402
from run_geometry_specificity_oracle import (  # noqa: E402
    INDEPENDENT_MISSING,
    TensorDirectory,
    correlations,
    fit_training_residual_pca,
    load_semiseg,
    patient_r_array,
    segment,
    summarize,
)

RANKS = (1, 2, 3, 4, 5, 6)
GROSS_LIMB_THRESHOLD_MV = 0.01


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gross_limb_mask(target: np.ndarray) -> np.ndarray:
    residuals = np.stack(
        (
            target[:, 2] - (target[:, 1] - target[:, 0]),
            target[:, 3] + (target[:, 0] + target[:, 1]) / 2,
            target[:, 4] - (target[:, 0] - target[:, 1] / 2),
            target[:, 5] - (target[:, 1] - target[:, 0] / 2),
        ),
        axis=1,
    )
    return np.max(np.abs(residuals), axis=(1, 2)) > GROSS_LIMB_THRESHOLD_MV


def dependent_r2(target: np.ndarray, reconstruction: np.ndarray, keep: np.ndarray) -> dict[str, float]:
    answer = {}
    for index, name in zip((2, 3, 4, 5), ("III", "aVR", "aVL", "aVF"), strict=True):
        truth = target[keep, index].astype(np.float64)
        estimate = reconstruction[keep, index].astype(np.float64)
        answer[name] = float(1 - np.square(truth - estimate).sum() / np.square(truth - truth.mean()).sum())
    return answer


def select_rank(rows: dict[int, dict]) -> tuple[int, str]:
    candidates = []
    for rank in RANKS[:-1]:
        if rows[rank]["training_residual_energy_fraction"] >= 0.90 and rows[rank + 1]["marginal_oracle_gain"] < 0.005:
            candidates.append(rank)
    if candidates:
        rank = min(candidates)
        return rank, f"smallest rank with >=90% training residual energy and G_{rank + 1}<0.005"
    energy_candidates = [rank for rank in RANKS if rows[rank]["training_residual_energy_fraction"] >= 0.90]
    if not energy_candidates:
        raise RuntimeError("none of ranks 1..6 reaches the preregistered 90% residual-energy floor")
    rank = min(energy_candidates)
    return rank, "smallest rank reaching 90% training residual energy; oracle curve had not met the saturation condition"


def report(payload: dict) -> str:
    rows = payload["ranks"]
    table = []
    for rank in RANKS:
        item = rows[str(rank)]
        table.append(
            f"| {rank} | {item['training_residual_energy_fraction']:.6f} | {item['representation_efficiency']:.6f} | "
            f"{item['mean_independent_missing_r']:.6f} | {item['marginal_oracle_gain']:.6f} | "
            f"{item['p05_independent_missing_r']:.6f} | {item['independent_missing_mse']:.8f} | {item['semiseg_miou']:.6f} |"
        )
    scalars = []
    for rank in RANKS:
        scalars.append(f"rank_{rank}_energy = {rows[str(rank)]['training_residual_energy_fraction']:.9f}")
    scalars.append("")
    for rank in RANKS:
        scalars.append(f"rank_{rank}_oracle_mean_r = {rows[str(rank)]['mean_independent_missing_r']:.9f}")
    return f"""# Residual PCA rank characterization

`FIXED_VCG_SPECIFICITY = NOT_SUPPORTED`

| Rank | Training energy | Energy/rank | Oracle mean r | Marginal gain | Oracle p05 r | Independent MSE | SemiSeg mIoU |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

```text
{chr(10).join(scalars)}

K_STAR = {payload['K_STAR']}
K_STAR_SELECTION_REASON = {payload['K_STAR_SELECTION_REASON']}
```

The rank gate uses all 2,183 fold-9 ECGs for independent-lead endpoints. The frozen dependent-limb QC rule (`max_abs_residual > 0.01 mV`) identifies {payload['qc']['gross_violation_ecgs']} records; only dependent-limb sensitivity metrics exclude them.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=ROOT / "data/ptb_xl/tensors/train")
    parser.add_argument("--val-dir", type=Path, default=ROOT / "data/ptb_xl/tensors/val")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/ptb_xl/ptbxl_database.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "refine-logs/predictable_residual_subspace")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke-batches", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the preregistered SemiSeg metric panel")
    device = torch.device("cuda:0")
    train = TensorDirectory(args.train_dir)
    val = TensorDirectory(args.val_dir, expected=2183)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    coefficients, basis6, fit = fit_training_residual_pca(train_loader, device, args.smoke_batches, rank=6)
    eigenvalues = np.asarray(fit["eigenvalues"])
    energy = np.cumsum(eigenvalues) / np.maximum(eigenvalues.sum(), np.finfo(float).tiny)

    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    target_parts, ecg_ids = [], []
    for batch_index, (truth, ids) in enumerate(tqdm(val_loader, desc="Load fold-9 targets")):
        if args.smoke_batches and batch_index >= args.smoke_batches:
            break
        target_parts.append(truth.numpy())
        ecg_ids.extend(ids.tolist())
    target = np.concatenate(target_parts)
    metadata = pd.read_csv(args.metadata, index_col="ecg_id")
    patient_ids = metadata.loc[ecg_ids, "patient_id"].to_numpy()
    semiseg = load_semiseg(device)
    ground_masks = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(target), args.batch_size), desc="Segment ground truth"):
            ground_masks.append(segment(semiseg, torch.from_numpy(target[start : start + args.batch_size]).to(device)).cpu().numpy())
    ground_masks = np.concatenate(ground_masks)
    qc_bad = gross_limb_mask(target)
    rank_rows = {}

    independent = target[:, (0, 1, 6, 7, 8, 9, 10, 11)]
    lead_i = independent[:, 0]
    missing = independent[:, 1:]
    rank0 = coefficients.cpu().numpy()[None, :, None] * lead_i[:, None, :]
    previous_mean = float(patient_r_array(correlations(missing, rank0), patient_ids).mean())

    for rank in RANKS:
        recon_parts, mask_parts = [], []
        with torch.inference_mode():
            for start in tqdm(range(0, len(target), args.batch_size), desc=f"Rank-{rank} oracle"):
                truth = torch.from_numpy(target[start : start + args.batch_size]).to(device)
                reconstruction, _ = residual_subspace_oracle(truth, coefficients, basis6[:, :rank])
                recon_parts.append(reconstruction.cpu().numpy())
                mask_parts.append(segment(semiseg, reconstruction).cpu().numpy())
        reconstruction = np.concatenate(recon_parts)
        masks = np.concatenate(mask_parts)
        metrics = summarize(target, reconstruction, masks, ground_masks, patient_ids)
        current_mean = metrics["mean_independent_missing_r"]
        metrics.update(
            {
                "training_residual_energy_fraction": float(energy[rank - 1]),
                "representation_efficiency": float(energy[rank - 1] / rank),
                "marginal_oracle_gain": float(current_mean - previous_mean),
                "dependent_lead_r2_qc_sensitivity": dependent_r2(target, reconstruction, ~qc_bad),
            }
        )
        rank_rows[rank] = metrics
        previous_mean = current_mean

    k_star, reason = select_rank(rank_rows)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d_%H%M%S")
    payload = {
        "schema_version": 1,
        "created_at": now.isoformat(),
        "protocol": "predictable residual subspace / rank gate",
        "FIXED_VCG_SPECIFICITY": "NOT_SUPPORTED",
        "cohort": "PTB-XL folds 1-8 fit; fold 9 oracle only",
        "n_fit_ecgs": fit["fit_ecgs"],
        "n_validation_ecgs": len(target),
        "n_validation_patients": int(len(np.unique(patient_ids))),
        "smoke_batches": args.smoke_batches,
        "ranks": {str(rank): rank_rows[rank] for rank in RANKS},
        "K_STAR": k_star,
        "K_STAR_SELECTION_REASON": reason,
        "qc": {
            "gross_limb_threshold_mv": GROSS_LIMB_THRESHOLD_MV,
            "gross_violation_ecgs": int(qc_bad.sum()),
            "gross_violation_ecg_ids": [int(ecg_ids[index]) for index in np.flatnonzero(qc_bad)],
            "primary_independent_metrics_exclude_records": False,
            "dependent_limb_sensitivity_excludes_records": True,
        },
        "pca": {"lead_i_coefficients": fit["lead_i_coefficients"], "basis_k_star": basis6[:, :k_star].cpu().tolist()},
    }
    neural_manifest = {
        "schema_version": 1,
        "created_at": now.isoformat(),
        "status": "FROZEN_BEFORE_NEURAL_TRAINING",
        "K_STAR": k_star,
        "K_STAR_SELECTION_REASON": reason,
        "B1": "direct seven-output",
        "C1": f"fixed PCA rank {k_star}",
        "C2": f"learned QR-orthonormal rank {k_star}",
        "seed": 42,
        "epochs": 15,
        "primary_metric": "mean_independent_missing_r",
        "checkpoint_metric": "patient-weighted r_ind,val",
        "bootstrap_seed": 20260910,
        "bootstrap_replicates": 10000,
        "targets": ["II", "V1", "V2", "V3", "V4", "V5", "V6"],
        "folds": {"train": "1-8", "validation": 9, "test": "SEALED"},
        "fixed_pca": {"lead_i_coefficients": fit["lead_i_coefficients"], "basis": basis6[:, :k_star].cpu().tolist()},
        "rank_curve_sha256": None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_timestamped = args.output_dir / f"residual_rank_curve_{stamp}.json"
    rank_text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    rank_timestamped.write_text(rank_text)
    (args.output_dir / "residual_rank_curve.json").write_text(rank_text)
    report_text = report(payload)
    (args.output_dir / f"residual_rank_curve_{stamp}.md").write_text(report_text)
    (args.output_dir / "residual_rank_curve.md").write_text(report_text)
    neural_manifest["rank_curve_sha256"] = file_sha256(rank_timestamped)
    manifest_text = json.dumps(neural_manifest, indent=2, allow_nan=False) + "\n"
    (args.output_dir / f"K_STAR_manifest_{stamp}.json").write_text(manifest_text)
    (args.output_dir / "K_STAR_manifest.json").write_text(manifest_text)
    print(json.dumps({"K_STAR": k_star, "reason": reason, "stamp": stamp}, indent=2))


if __name__ == "__main__":
    main()
