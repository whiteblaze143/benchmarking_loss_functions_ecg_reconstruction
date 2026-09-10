#!/usr/bin/env python3
"""Run the validation-only NullVCG geometry oracle and frozen kill gate."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from geometry import (  # noqa: E402
    constrained_oracle,
    full_oracle,
    geometry_diagnostics,
    independent_from_standard,
)
from scripts.evaluate_sap_v2 import (  # noqa: E402
    Reconstructor,
    load_semiseg,
    patient_r_array,
    pt_groupby_mean,
    qrs_ms,
)


LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
MISSING = np.arange(1, 12)


class ValidationDataset(Dataset):
    def __init__(self, directory: Path):
        if directory.name != "val":
            raise ValueError(f"oracle requires the PTB-XL validation directory, got {directory}")
        self.files = sorted(directory.glob("*.pt"), key=lambda path: int(path.stem))
        if len(self.files) != 2183:
            raise ValueError(f"expected 2,183 validation tensors, found {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        return torch.load(path, map_location="cpu", weights_only=True).float(), int(path.stem)


def segment(model, waveforms):
    resized = F.interpolate(waveforms, size=2500, mode="linear", align_corners=False)
    normalized = (resized - resized.mean(-1, keepdim=True)) / (resized.std(-1, keepdim=True) + 1e-6)
    return model(normalized[:, 1:2])["seg_logits"].argmax(1).repeat_interleave(2, -1)[:, :5000]


def correlations(target, reconstruction):
    target = target - target.mean(-1, keepdims=True)
    reconstruction = reconstruction - reconstruction.mean(-1, keepdims=True)
    denominator = np.sqrt((target**2).sum(-1) * (reconstruction**2).sum(-1)) + 1e-8
    return (target * reconstruction).sum(-1) / denominator


def patient_mean(values, patient_ids):
    return float(pt_groupby_mean(np.asarray(values), patient_ids).mean())


def summarize(target, reconstruction, masks, ground_masks, patient_ids):
    corr = correlations(target, reconstruction)
    patient_missing = patient_r_array(corr, patient_ids, np.isin(np.arange(12), MISSING))
    per_lead_r2 = {}
    for index, lead in enumerate(LEADS):
        truth = target[:, index].astype(np.float64)
        estimate = reconstruction[:, index].astype(np.float64)
        per_lead_r2[lead] = float(1 - np.square(truth - estimate).sum() / np.square(truth - truth.mean()).sum())

    ious = {}
    for name, label in (("p_wave_iou", 1), ("qrs_wave_iou", 2), ("t_wave_iou", 3)):
        per_record = np.array([
            np.logical_and(gt == label, pred == label).sum()
            / max(np.logical_or(gt == label, pred == label).sum(), 1)
            for gt, pred in zip(ground_masks, masks)
        ])
        ious[name] = patient_mean(per_record, patient_ids)
    ious["semiseg_miou"] = float(np.mean(list(ious.values())))

    qrs_true = np.array([qrs_ms(mask) for mask in ground_masks])
    qrs_recon = np.array([qrs_ms(mask) for mask in masks])
    finite = np.isfinite(qrs_true) & np.isfinite(qrs_recon)
    qrs_mae = patient_mean(np.abs(qrs_true[finite] - qrs_recon[finite]), patient_ids[finite])

    sokolow_true = np.abs(target[:, 6, 1000:4000].min(-1)) + target[:, 10, 1000:4000].max(-1)
    sokolow_recon = np.abs(reconstruction[:, 6, 1000:4000].min(-1)) + reconstruction[:, 10, 1000:4000].max(-1)
    lvh_mae = patient_mean(np.abs(sokolow_true - sokolow_recon), patient_ids)
    variance = float(np.mean([
        np.var(reconstruction[:, lead, 1000:4000].max(-1))
        / (np.var(target[:, lead, 1000:4000].max(-1)) + 1e-9)
        for lead in range(6, 12)
    ]))
    return {
        "mean_all_missing_r": float(patient_missing.mean()),
        "p05_all_missing_r": float(np.quantile(patient_missing, 0.05)),
        **ious,
        "qrs_duration_mae_ms": qrs_mae,
        "lvh_sokolowlyon_mae_mv": lvh_mae,
        "precordial_variance_retention": variance,
        "per_lead_r2": per_lead_r2,
    }


def gate(constrained, baseline):
    checks = {
        "mean_all_missing_r": constrained["mean_all_missing_r"] >= baseline["mean_all_missing_r"] + 0.02,
        "p05_all_missing_r": constrained["p05_all_missing_r"] >= baseline["p05_all_missing_r"] + 0.02,
        "semiseg_miou": constrained["semiseg_miou"] >= baseline["semiseg_miou"],
        "p_wave_iou": constrained["p_wave_iou"] >= baseline["p_wave_iou"],
        "t_wave_iou": constrained["t_wave_iou"] >= baseline["t_wave_iou"],
        "qrs_duration_mae_ms": constrained["qrs_duration_mae_ms"] <= baseline["qrs_duration_mae_ms"],
        "lvh_sokolowlyon_mae_mv": constrained["lvh_sokolowlyon_mae_mv"] <= baseline["lvh_sokolowlyon_mae_mv"],
        "precordial_variance_retention": abs(np.log(constrained["precordial_variance_retention"]))
        <= abs(np.log(baseline["precordial_variance_retention"])),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return checks, "PASS" if all(checks.values()) else "FAIL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/ptb_xl/tensors/val")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/ptb_xl/ptbxl_database.csv")
    parser.add_argument("--baseline-checkpoint", type=Path, default=ROOT / "refine-logs/convergence_10e/runs/conv15e_A0_wave_noSSL_gated_add_s42_l0/best.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke-batches", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the baseline and SemiSeg oracle comparison")
    if not args.baseline_checkpoint.is_file():
        raise FileNotFoundError(args.baseline_checkpoint)
    device = torch.device("cuda:0")
    dataset = ValidationDataset(args.data_dir.resolve())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    metadata = pd.read_csv(args.metadata, index_col="ecg_id")
    baseline = Reconstructor("conv15e_A0_wave_noSSL_gated_add_s42_l0", args.baseline_checkpoint, device)
    semiseg = load_semiseg(device)
    waveforms = {name: [] for name in ("ground_truth", "full_oracle", "constrained_oracle", "a0_wavelet")}
    masks = {name: [] for name in waveforms}
    ecg_ids = []

    with torch.inference_mode():
        for batch_index, (truth, ids) in enumerate(tqdm(loader, desc="NullVCG oracle")):
            if args.smoke_batches and batch_index >= args.smoke_batches:
                break
            truth = truth.to(device)
            full, _ = full_oracle(truth)
            constrained, _ = constrained_oracle(truth)
            reconstructed = baseline.reconstruct(truth)
            current = {
                "ground_truth": truth,
                "full_oracle": full,
                "constrained_oracle": constrained,
                "a0_wavelet": reconstructed,
            }
            for name, value in current.items():
                waveforms[name].append(value.cpu().numpy())
                masks[name].append(segment(semiseg, value).cpu().numpy())
            ecg_ids.extend(ids.tolist())

    patient_ids = metadata.loc[ecg_ids, "patient_id"].to_numpy()
    waveforms = {name: np.concatenate(parts) for name, parts in waveforms.items()}
    masks = {name: np.concatenate(parts) for name, parts in masks.items()}
    target = waveforms["ground_truth"]
    metrics = {
        name: summarize(target, value, masks[name], masks["ground_truth"], patient_ids)
        for name, value in waveforms.items()
    }
    constrained_independent = independent_from_standard(torch.from_numpy(waveforms["constrained_oracle"]))
    source_independent = independent_from_standard(torch.from_numpy(target))
    full_independent = independent_from_standard(torch.from_numpy(waveforms["full_oracle"]))
    full_mse = float(torch.mean((source_independent - full_independent) ** 2))
    constrained_mse = float(torch.mean((source_independent - constrained_independent) ** 2))
    checks, decision = gate(metrics["constrained_oracle"], metrics["a0_wavelet"])
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cohort": "PTB-XL validation fold 9",
        "n_ecgs": len(ecg_ids),
        "n_patients": int(len(np.unique(patient_ids))),
        "smoke_batches": args.smoke_batches,
        "geometry": geometry_diagnostics(),
        "error_decomposition_mse": {
            "dipole_model_error_full_oracle": full_mse,
            "observation_constraint_penalty": constrained_mse - full_mse,
            "constrained_oracle_total": constrained_mse,
        },
        "metrics": metrics,
        "gate_checks": checks,
        "NULLVCG_ORACLE_GATE": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=False))
    print(f"NULLVCG_ORACLE_GATE = {decision}")


if __name__ == "__main__":
    main()
