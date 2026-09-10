#!/usr/bin/env python3
"""Fit a training-only residual PCA and run the fold-9 geometry-specificity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
    geometry_diagnostics,
    independent_from_standard,
    residual_subspace_oracle,
)
from scripts.evaluate_sap_v2 import (  # noqa: E402
    Reconstructor,
    load_semiseg,
    patient_r_array,
    pt_groupby_mean,
    qrs_ms,
)

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
INDEPENDENT_MISSING = np.array([1, 6, 7, 8, 9, 10, 11])
ALL_MISSING = np.arange(1, 12)
RANDOM_SEED = 20260910


class TensorDirectory(Dataset):
    def __init__(self, directory: Path, expected: int | None = None):
        self.directory = directory.resolve()
        self.files = sorted(self.directory.glob("*.pt"), key=lambda path: int(path.stem))
        if expected is not None and len(self.files) != expected:
            raise ValueError(f"expected {expected} tensors in {self.directory}, found {len(self.files)}")
        if not self.files:
            raise ValueError(f"no tensors found in {self.directory}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        return torch.load(path, map_location="cpu", weights_only=True).float(), int(path.stem)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_columns(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.clone()
    for column in range(matrix.shape[1]):
        pivot = torch.argmax(torch.abs(matrix[:, column]))
        if matrix[pivot, column] < 0:
            matrix[:, column] *= -1
    return matrix


def fit_training_residual_pca(loader: DataLoader, device: torch.device, limit_batches: int = 0):
    sxx = torch.zeros((), dtype=torch.float64, device=device)
    sxy = torch.zeros(7, dtype=torch.float64, device=device)
    syy = torch.zeros(7, 7, dtype=torch.float64, device=device)
    samples = 0
    ecgs = 0
    with torch.inference_mode():
        for batch_index, (waveforms, _) in enumerate(tqdm(loader, desc="Fit train residual PCA")):
            if limit_batches and batch_index >= limit_batches:
                break
            independent = independent_from_standard(waveforms.to(device=device, dtype=torch.float64))
            x = independent[:, 0].reshape(-1)
            y = independent[:, 1:].permute(0, 2, 1).reshape(-1, 7)
            sxx += torch.dot(x, x)
            sxy += y.T @ x
            syy += y.T @ y
            samples += x.numel()
            ecgs += waveforms.shape[0]
    if sxx <= 0 or samples == 0:
        raise RuntimeError("degenerate training sufficient statistics")
    coefficients = sxy / sxx
    residual_scatter = syy - torch.outer(sxy, sxy) / sxx
    residual_scatter = (residual_scatter + residual_scatter.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(residual_scatter)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    basis = canonicalize_columns(eigenvectors[:, order[:2]])
    explained = eigenvalues[:2].sum() / eigenvalues.clamp_min(0).sum()
    return coefficients.float(), basis.float(), {
        "fit_ecgs": ecgs,
        "fit_time_samples": samples,
        "lead_i_coefficients": coefficients.cpu().tolist(),
        "basis": basis.cpu().tolist(),
        "eigenvalues": eigenvalues.cpu().tolist(),
        "rank2_residual_energy_fraction": float(explained),
        "centering": "uncentered residual second moment",
    }


def correlations(target: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    target_centered = target - target.mean(-1, keepdims=True)
    recon_centered = reconstruction - reconstruction.mean(-1, keepdims=True)
    denominator = np.sqrt((target_centered**2).sum(-1) * (recon_centered**2).sum(-1)) + 1e-8
    return (target_centered * recon_centered).sum(-1) / denominator


def patient_mean(values: np.ndarray, patient_ids: np.ndarray) -> float:
    return float(pt_groupby_mean(np.asarray(values), patient_ids).mean())


def segment(model, waveforms: torch.Tensor) -> torch.Tensor:
    resized = F.interpolate(waveforms, size=2500, mode="linear", align_corners=False)
    normalized = (resized - resized.mean(-1, keepdim=True)) / (resized.std(-1, keepdim=True) + 1e-6)
    return model(normalized[:, 1:2])["seg_logits"].argmax(1).repeat_interleave(2, -1)[:, :5000]


def patient_independent_array(target: np.ndarray, reconstruction: np.ndarray, patient_ids: np.ndarray):
    return patient_r_array(correlations(target, reconstruction), patient_ids, np.isin(np.arange(12), INDEPENDENT_MISSING))


def summarize(target, reconstruction, masks, ground_masks, patient_ids):
    corr = correlations(target, reconstruction)
    independent = patient_r_array(corr, patient_ids, np.isin(np.arange(12), INDEPENDENT_MISSING))
    all_missing = patient_r_array(corr, patient_ids, np.isin(np.arange(12), ALL_MISSING))
    independent_error = target[:, INDEPENDENT_MISSING] - reconstruction[:, INDEPENDENT_MISSING]
    per_lead_r2 = {}
    for index, lead in enumerate(LEADS):
        truth = target[:, index].astype(np.float64)
        estimate = reconstruction[:, index].astype(np.float64)
        per_lead_r2[lead] = float(1 - np.square(truth - estimate).sum() / np.square(truth - truth.mean()).sum())
    ious = {}
    for name, label in (("p_wave_iou", 1), ("qrs_wave_iou", 2), ("t_wave_iou", 3)):
        values = np.array([
            np.logical_and(gt == label, pred == label).sum() / max(np.logical_or(gt == label, pred == label).sum(), 1)
            for gt, pred in zip(ground_masks, masks)
        ])
        ious[name] = patient_mean(values, patient_ids)
    ious["semiseg_miou"] = float(np.mean(list(ious.values())))
    qrs_true = np.array([qrs_ms(mask) for mask in ground_masks])
    qrs_recon = np.array([qrs_ms(mask) for mask in masks])
    finite = np.isfinite(qrs_true) & np.isfinite(qrs_recon)
    sokolow_true = np.abs(target[:, 6, 1000:4000].min(-1)) + target[:, 10, 1000:4000].max(-1)
    sokolow_recon = np.abs(reconstruction[:, 6, 1000:4000].min(-1)) + reconstruction[:, 10, 1000:4000].max(-1)
    variance = float(np.mean([
        np.var(reconstruction[:, lead, 1000:4000].max(-1)) / (np.var(target[:, lead, 1000:4000].max(-1)) + 1e-9)
        for lead in range(6, 12)
    ]))
    return {
        "mean_independent_missing_r": float(independent.mean()),
        "p05_independent_missing_r": float(np.quantile(independent, 0.05)),
        "mean_all_missing_r": float(all_missing.mean()),
        "p05_all_missing_r": float(np.quantile(all_missing, 0.05)),
        "independent_missing_mse": float(np.mean(np.square(independent_error, dtype=np.float64))),
        **ious,
        "qrs_duration_mae_ms": patient_mean(np.abs(qrs_true[finite] - qrs_recon[finite]), patient_ids[finite]),
        "lvh_sokolowlyon_mae_mv": patient_mean(np.abs(sokolow_true - sokolow_recon), patient_ids),
        "precordial_variance_retention": variance,
        "per_lead_r2": per_lead_r2,
    }


def limb_algebra_audit(target: np.ndarray) -> dict:
    expected = {
        "III": target[:, 1] - target[:, 0],
        "aVR": -(target[:, 0] + target[:, 1]) / 2,
        "aVL": target[:, 0] - target[:, 1] / 2,
        "aVF": target[:, 1] - target[:, 0] / 2,
    }
    indices = {"III": 2, "aVR": 3, "aVL": 4, "aVF": 5}
    per_lead = {}
    for name, calculated in expected.items():
        difference = target[:, indices[name]] - calculated
        per_lead[name] = {
            "max_abs": float(np.max(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(np.square(difference, dtype=np.float64)))),
        }
    maximum = max(item["max_abs"] for item in per_lead.values())
    return {
        "per_lead": per_lead,
        "max_abs": maximum,
        "threshold": 0.002,
        "threshold_rationale": "2 uV ceiling for independently quantized 1 uV-resolution mV channels",
        "pass": bool(maximum <= 0.002),
    }


def paired_bootstrap(a, b, seed=RANDOM_SEED, replicates=10_000):
    delta = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 500):
        count = min(500, replicates - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        means[start : start + count] = delta[indices].mean(axis=1)
    return {
        "delta": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "replicates": replicates,
        "seed": seed,
    }


def random_basis_metrics(target, coefficients, patient_ids, count=100, seed=RANDOM_SEED):
    independent = target[:, (0, 1, 6, 7, 8, 9, 10, 11)]
    lead_i = independent[:, 0]
    missing = independent[:, 1:]
    residual = missing - coefficients[None, :, None] * lead_i[:, None, :]
    rng = np.random.default_rng(seed)
    results = []
    for index in tqdm(range(count), desc="Random rank-2 controls"):
        basis, _ = np.linalg.qr(rng.standard_normal((7, 2)))
        signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), np.arange(2)])
        basis *= np.where(signs == 0, 1, signs)
        latent = np.einsum("lr,nlt->nrt", basis, residual, optimize=True)
        estimate = coefficients[None, :, None] * lead_i[:, None, :] + np.einsum("lr,nrt->nlt", basis, latent, optimize=True)
        corr = correlations(missing, estimate)
        patient = patient_r_array(corr, patient_ids)
        results.append({
            "index": index,
            "mean_independent_missing_r": float(patient.mean()),
            "p05_independent_missing_r": float(np.quantile(patient, 0.05)),
            "independent_missing_mse": float(np.mean(np.square(missing - estimate, dtype=np.float64))),
        })
    return results


def clinical_compensation(vcg: dict, pca: dict) -> dict:
    wins = [
        vcg["semiseg_miou"] > pca["semiseg_miou"],
        vcg["qrs_duration_mae_ms"] < pca["qrs_duration_mae_ms"],
        vcg["lvh_sokolowlyon_mae_mv"] < pca["lvh_sokolowlyon_mae_mv"],
        abs(np.log(vcg["precordial_variance_retention"])) < abs(np.log(pca["precordial_variance_retention"])),
    ]
    no_large_loss = [
        vcg["semiseg_miou"] >= pca["semiseg_miou"] - 0.005,
        vcg["qrs_duration_mae_ms"] <= pca["qrs_duration_mae_ms"] + 1.0,
        vcg["lvh_sokolowlyon_mae_mv"] <= pca["lvh_sokolowlyon_mae_mv"] + 0.02,
        abs(np.log(vcg["precordial_variance_retention"])) <= abs(np.log(pca["precordial_variance_retention"])) + 0.05,
    ]
    passed = sum(wins) >= 2 and all(no_large_loss) and pca["mean_independent_missing_r"] - vcg["mean_independent_missing_r"] <= 0.015
    return {"wins": int(sum(wins)), "no_large_loss": [bool(x) for x in no_large_loss], "pass": bool(passed)}


def markdown_report(payload: dict) -> str:
    metrics = payload["metrics"]
    rows = []
    for name in ("G0_fixed_vcg", "P0_train_pca", "T0_patch10"):
        item = metrics[name]
        rows.append(
            f"| {name} | {item['mean_independent_missing_r']:.6f} | {item['p05_independent_missing_r']:.6f} | "
            f"{item['independent_missing_mse']:.8f} | {item['semiseg_miou']:.6f} | {item['qrs_duration_mae_ms']:.3f} |"
        )
    random_summary = payload["random_control_summary"]
    return f"""# Geometry-specificity oracle result

- Decision: **{payload['GEOMETRY_SPECIFICITY_GATE']}**
- Cohort: {payload['cohort']} ({payload['n_ecgs']} ECGs; {payload['n_patients']} patients)
- Limb algebra audit: **{'PASS' if payload['limb_algebra_audit']['pass'] else 'FAIL'}**, max absolute residual {payload['limb_algebra_audit']['max_abs']:.3g}
- Training PCA rank-2 residual energy: {payload['pca_fit']['rank2_residual_energy_fraction']:.4%}

| System | Mean independent r | p05 independent r | Independent MSE | SemiSeg mIoU | QRS MAE ms |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

VCG − PCA paired patient bootstrap: {payload['paired_bootstrap']['G0_minus_P0']['delta']:.6f} "
f"[{payload['paired_bootstrap']['G0_minus_P0']['ci95_low']:.6f}, {payload['paired_bootstrap']['G0_minus_P0']['ci95_high']:.6f}].

The fixed VCG mean-correlation result is at the {random_summary['vcg_percentile_mean_r']:.1f}th percentile of the 100 frozen random rank-2 bases. The training PCA result is at the {random_summary['pca_percentile_mean_r']:.1f}th percentile.

Neural action: {payload['neural_action']}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=ROOT / "data/ptb_xl/tensors/train")
    parser.add_argument("--val-dir", type=Path, default=ROOT / "data/ptb_xl/tensors/val")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/ptb_xl/ptbxl_database.csv")
    parser.add_argument("--trunk-checkpoint", type=Path, default=ROOT / "refine-logs/lean_abl2/runs/lean2_T_patch10_s42_l0/best.pt")
    parser.add_argument("--trunk-config", type=Path, default=ROOT / "refine-logs/lean_abl2/runs/lean2_T_patch10_s42_l0/config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "refine-logs/nullvcg_observation_constrained")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke-batches", type=int, default=0)
    parser.add_argument("--random-bases", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    train = TensorDirectory(args.train_dir)
    val = TensorDirectory(args.val_dir, expected=2183)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    coefficients, basis, pca_fit = fit_training_residual_pca(train_loader, device, args.smoke_batches)
    trunk = Reconstructor("lean2_T_patch10_s42_l0", args.trunk_checkpoint, device)
    semiseg = load_semiseg(device)
    names = ("ground_truth", "G0_fixed_vcg", "P0_train_pca", "T0_patch10")
    waveforms = {name: [] for name in names}
    masks = {name: [] for name in names}
    ecg_ids = []
    with torch.inference_mode():
        for batch_index, (truth, ids) in enumerate(tqdm(val_loader, desc="Fold-9 oracle evaluation")):
            if args.smoke_batches and batch_index >= args.smoke_batches:
                break
            truth = truth.to(device)
            vcg, _ = constrained_oracle(truth)
            pca, _ = residual_subspace_oracle(truth, coefficients, basis)
            direct = trunk.reconstruct(truth)
            current = {"ground_truth": truth, "G0_fixed_vcg": vcg, "P0_train_pca": pca, "T0_patch10": direct}
            for name, value in current.items():
                waveforms[name].append(value.cpu().numpy())
                masks[name].append(segment(semiseg, value).cpu().numpy())
            ecg_ids.extend(ids.tolist())
    waveforms = {name: np.concatenate(parts) for name, parts in waveforms.items()}
    masks = {name: np.concatenate(parts) for name, parts in masks.items()}
    metadata = pd.read_csv(args.metadata, index_col="ecg_id")
    patient_ids = metadata.loc[ecg_ids, "patient_id"].to_numpy()
    target = waveforms["ground_truth"]
    metrics = {name: summarize(target, waveforms[name], masks[name], masks["ground_truth"], patient_ids) for name in names[1:]}
    patient_arrays = {name: patient_independent_array(target, waveforms[name], patient_ids) for name in names[1:]}
    random_results = random_basis_metrics(target, coefficients.cpu().numpy(), patient_ids, args.random_bases)
    random_means = np.array([item["mean_independent_missing_r"] for item in random_results])
    paired = {
        "G0_minus_P0": paired_bootstrap(patient_arrays["G0_fixed_vcg"], patient_arrays["P0_train_pca"]),
        "G0_minus_T0": paired_bootstrap(patient_arrays["G0_fixed_vcg"], patient_arrays["T0_patch10"]),
        "P0_minus_T0": paired_bootstrap(patient_arrays["P0_train_pca"], patient_arrays["T0_patch10"]),
    }
    compensation = clinical_compensation(metrics["G0_fixed_vcg"], metrics["P0_train_pca"])
    useful_g0 = paired["G0_minus_T0"]["delta"] >= 0.005 and paired["G0_minus_T0"]["ci95_low"] > 0
    useful_p0 = paired["P0_minus_T0"]["delta"] >= 0.005 and paired["P0_minus_T0"]["ci95_low"] > 0
    difference = metrics["P0_train_pca"]["mean_independent_missing_r"] - metrics["G0_fixed_vcg"]["mean_independent_missing_r"]
    if not (useful_g0 or useful_p0):
        decision, action = "FAIL", "Stop; neither oracle has preregistered useful headroom over T0."
    elif difference <= 0.005 or compensation["pass"]:
        decision, action = "VCG_SUPPORTED", "Proceed to matched B1/M1/C1 seed-42 training."
    else:
        decision, action = "LOW_RANK_ONLY", "Proceed with B1/C1 for the low-rank claim; M1 is negative-control only."
    audit = limb_algebra_audit(target)
    if not audit["pass"]:
        decision, action = "FAIL", "Stop; source limb-algebra audit exceeded the frozen threshold."
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "schema_version": 2,
        "created_at": now.isoformat(),
        "cohort": "PTB-XL fold 9 only",
        "n_ecgs": len(ecg_ids),
        "n_patients": int(len(np.unique(patient_ids))),
        "smoke_batches": args.smoke_batches,
        "pca_fit": pca_fit,
        "geometry": geometry_diagnostics(),
        "limb_algebra_audit": audit,
        "metrics": metrics,
        "paired_bootstrap": paired,
        "clinical_compensation": compensation,
        "random_controls": random_results,
        "random_control_summary": {
            "count": len(random_results),
            "seed": RANDOM_SEED,
            "mean_r_median": float(np.median(random_means)),
            "mean_r_max": float(np.max(random_means)),
            "vcg_percentile_mean_r": float(100 * np.mean(random_means <= metrics["G0_fixed_vcg"]["mean_independent_missing_r"])),
            "pca_percentile_mean_r": float(100 * np.mean(random_means <= metrics["P0_train_pca"]["mean_independent_missing_r"])),
        },
        "GEOMETRY_SPECIFICITY_GATE": decision,
        "neural_action": action,
    }
    echo_metadata = ROOT / "data/echonext/echonext_metadata_100k.csv"
    echo_frame = pd.read_csv(echo_metadata, usecols=["ecg_key", "patient_key", "split"])
    available = echo_frame.iloc[:5442]
    used_patients = set(available.iloc[:1000]["patient_key"])
    pristine = available.iloc[1000:][~available.iloc[1000:]["patient_key"].isin(used_patients)]
    pristine_ids = sorted(map(int, pristine["ecg_key"].tolist()))
    pristine_hash = hashlib.sha256(("\n".join(map(str, pristine_ids)) + "\n").encode()).hexdigest()
    manifest = {
        "schema_version": 1,
        "created_at": now.isoformat(),
        "protocol": "EXPERIMENT_PLAN_20260910_173054.md",
        "random_seed": RANDOM_SEED,
        "bootstrap_replicates": 10_000,
        "splits": {"pca_fit": "PTB-XL folds 1-8/train", "selection": "PTB-XL fold 9/val", "ptbxl_fold10": "SEALED"},
        "artifacts": {
            "trunk_checkpoint": {"path": str(args.trunk_checkpoint.resolve()), "sha256": sha256(args.trunk_checkpoint)},
            "trunk_config": {"path": str(args.trunk_config.resolve()), "sha256": sha256(args.trunk_config)},
            "ptbxl_metadata": {"path": str(args.metadata.resolve()), "sha256": sha256(args.metadata)},
        },
        "echo_next_metadata_only_audit": {
            "metadata_rows": len(echo_frame), "available_waveform_rows": 5442, "prior_prefix_rows": 1000,
            "patient_disjoint_pristine_ecgs": len(pristine_ids), "sorted_ecg_key_sha256": pristine_hash,
            "status": "FROZEN_FOR_FUTURE; waveforms and labels not evaluated",
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "pca_parameters": {"lead_i_coefficients": pca_fit["lead_i_coefficients"], "basis": pca_fit["basis"]},
        "gate": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = markdown_report(payload)
    for filename, content in (
        (f"geometry_specificity_oracle_{stamp}.json", json.dumps(payload, indent=2, allow_nan=False) + "\n"),
        ("geometry_specificity_oracle.json", json.dumps(payload, indent=2, allow_nan=False) + "\n"),
        (f"geometry_specificity_oracle_{stamp}.md", report),
        ("geometry_specificity_oracle.md", report),
        (f"latent_experiment_manifest_{stamp}.json", json.dumps(manifest, indent=2, allow_nan=False) + "\n"),
        ("latent_experiment_manifest.json", json.dumps(manifest, indent=2, allow_nan=False) + "\n"),
    ):
        (args.output_dir / filename).write_text(content)
    print(json.dumps({"decision": decision, "neural_action": action, "output_stamp": stamp}, indent=2))


if __name__ == "__main__":
    main()
