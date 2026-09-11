#!/usr/bin/env python3
"""Evaluate frozen B1/C1/C2 checkpoints under the predictable-subspace protocol."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nullvcg_observation_constrained.run_geometry_specificity_oracle import (  # noqa: E402
    INDEPENDENT_MISSING,
    TensorDirectory,
    correlations,
    load_semiseg,
    patient_independent_array,
    segment,
    summarize,
)
from experiments.nullvcg_observation_constrained.run_residual_rank_curve import (  # noqa: E402
    dependent_r2,
    gross_limb_mask,
)
from scripts.train_1lead_wavelet_ssl_mtl import (  # noqa: E402
    apply_zscore,
    build_model,
    forward_model,
)

SYSTEMS = ("B1", "C1", "C2")
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_REPLICATES = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = argparse.Namespace(**payload["config"])
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), payload


def latent_sufficient_statistics(predicted: torch.Tensor, oracle: torch.Tensor, state: dict | None = None):
    x = predicted.double().permute(1, 0, 2).reshape(predicted.shape[1], -1)
    y = oracle.double().permute(1, 0, 2).reshape(oracle.shape[1], -1)
    if state is None:
        state = {key: torch.zeros(x.shape[0], dtype=torch.float64) for key in ("sx", "sy", "sxx", "syy", "sxy", "sse")}
        state["n"] = 0
    state["sx"] += x.sum(1).cpu(); state["sy"] += y.sum(1).cpu()
    state["sxx"] += x.square().sum(1).cpu(); state["syy"] += y.square().sum(1).cpu()
    state["sxy"] += (x * y).sum(1).cpu(); state["sse"] += (x - y).square().sum(1).cpu()
    state["n"] += x.shape[1]
    return state


def finish_latent_statistics(state: dict) -> list[dict]:
    n = state["n"]
    cov = state["sxy"] - state["sx"] * state["sy"] / n
    var_x = state["sxx"] - state["sx"].square() / n
    var_y = state["syy"] - state["sy"].square() / n
    correlation = cov / torch.sqrt(var_x * var_y).clamp_min(1e-12)
    r2 = 1 - state["sse"] / var_y.clamp_min(1e-12)
    return [{"coordinate": i + 1, "correlation": float(correlation[i]), "r2": float(r2[i])} for i in range(len(r2))]


def paired_bootstrap(a: np.ndarray, b: np.ndarray):
    if len(a) != len(b):
        raise ValueError("paired patient arrays differ in length")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    mean_delta, p05_delta = [], []
    for _ in range(0, BOOTSTRAP_REPLICATES, 250):
        indices = rng.integers(0, len(a), size=(250, len(a)))
        av, bv = a[indices], b[indices]
        mean_delta.extend((av.mean(1) - bv.mean(1)).tolist())
        p05_delta.extend((np.quantile(av, 0.05, axis=1) - np.quantile(bv, 0.05, axis=1)).tolist())
    def result(samples, point):
        return {"delta": float(point), "ci95_low": float(np.quantile(samples, 0.025)), "ci95_high": float(np.quantile(samples, 0.975))}
    return {
        "mean_independent_r": result(mean_delta, a.mean() - b.mean()),
        "p05_independent_r": result(p05_delta, np.quantile(a, 0.05) - np.quantile(b, 0.05)),
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
    }


def variance_retention_per_lead(target: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    return {
        f"V{lead - 5}": float(np.var(reconstruction[:, lead, 1000:4000].max(-1)) / (np.var(target[:, lead, 1000:4000].max(-1)) + 1e-9))
        for lead in range(6, 12)
    }


def harm_screen(candidate: dict, baseline: dict) -> dict:
    checks = {
        "semiseg_miou": candidate["semiseg_miou"] - baseline["semiseg_miou"] > -0.005,
        "p_wave_iou": candidate["p_wave_iou"] - baseline["p_wave_iou"] > -0.01,
        "t_wave_iou": candidate["t_wave_iou"] - baseline["t_wave_iou"] > -0.01,
        "qrs_duration_mae_ms": candidate["qrs_duration_mae_ms"] - baseline["qrs_duration_mae_ms"] < 1.0,
        "lvh_sokolowlyon_mae_mv": candidate["lvh_sokolowlyon_mae_mv"] - baseline["lvh_sokolowlyon_mae_mv"] < 0.05,
        "precordial_variance_log_error": abs(np.log(candidate["precordial_variance_retention"])) - abs(np.log(baseline["precordial_variance_retention"])) <= 0.05,
    }
    return {"checks": {key: bool(value) for key, value in checks.items()}, "pass": bool(all(checks.values()))}


def markdown(payload: dict) -> str:
    rows = []
    for system in SYSTEMS:
        m = payload["metrics"][system]
        rows.append(f"| {system} | {m['mean_independent_missing_r']:.6f} | {m['p05_independent_missing_r']:.6f} | {m['independent_missing_mse']:.8f} | {m['semiseg_miou']:.6f} | {m['qrs_duration_mae_ms']:.3f} | {m['lvh_sokolowlyon_mae_mv']:.4f} |")
    return f"""# Predictable residual subspace: seed-42 evaluation

| System | Mean independent r | p05 | MSE | SemiSeg mIoU | QRS MAE ms | LVH MAE mV |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

- C1 advance: **{payload['decisions']['C1']['advance']}**
- C2 advance: **{payload['decisions']['C2']['advance']}**
- Multi-seed action: {payload['multi_seed_action']}
"""


def main():
    parser = argparse.ArgumentParser()
    root = ROOT / "refine-logs/predictable_residual_subspace"
    parser.add_argument("--runs-dir", type=Path, default=root / "runs")
    parser.add_argument("--rank-curve", type=Path, default=root / "residual_rank_curve.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/ptb_xl/tensors/val")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/ptb_xl/ptbxl_database.csv")
    parser.add_argument("--output-dir", type=Path, default=root)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    run_names = {"B1": "prs_direct_k5_s42_l0", "C1": "prs_frozen_pca_k5_s42_l0", "C2": "prs_learned_k5_s42_l0"}
    checkpoints = {name: args.runs_dir / run / "best.pt" for name, run in run_names.items()}
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    dataset = TensorDirectory(args.data_dir, expected=2183)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    target_parts, ids = [], []
    for truth, ecg_ids in loader:
        target_parts.append(truth.numpy()); ids.extend(ecg_ids.tolist())
    target = np.concatenate(target_parts)
    patients = pd.read_csv(args.metadata, index_col="ecg_id").loc[ids, "patient_id"].to_numpy()
    semiseg = load_semiseg(device)
    ground_masks = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(target), args.batch_size), desc="Ground-truth segmentation"):
            ground_masks.append(segment(semiseg, torch.from_numpy(target[start:start + args.batch_size]).to(device)).cpu().numpy())
    ground_masks = np.concatenate(ground_masks)
    qc_bad = gross_limb_mask(target)
    metrics, patient_arrays, latent_diagnostics, hashes = {}, {}, {}, {}
    learned_basis = None

    for system in SYSTEMS:
        model, checkpoint_payload = load_model(checkpoints[system], device)
        recon_parts, mask_parts = [], []
        latent_state = None
        pca_basis = torch.as_tensor(checkpoint_payload["config"] and json.loads(Path(checkpoint_payload["config"]["k_star_manifest"]).read_text())["fixed_pca"]["basis"], device=device)
        coefficients = model.lead_i_coefficients if hasattr(model, "lead_i_coefficients") else None
        with torch.inference_mode():
            for start in tqdm(range(0, len(target), args.batch_size), desc=f"Evaluate {system}"):
                truth = torch.from_numpy(target[start:start + args.batch_size]).to(device)
                normalized, mean, scale = apply_zscore(truth)
                result = forward_model(model, normalized, [0], compute_delineation=False, compute_ssl=False)
                reconstruction = result["y_pred"] * scale + mean
                reconstruction[:, 0] = truth[:, 0]
                recon_parts.append(reconstruction.cpu().numpy())
                mask_parts.append(segment(semiseg, reconstruction).cpu().numpy())
                if system == "C1":
                    independent = normalized[:, INDEPENDENT_MISSING]
                    residual = independent - coefficients[None, :, None] * normalized[:, :1]
                    oracle_latent = torch.einsum("lk,blt->bkt", pca_basis, residual)
                    latent_state = latent_sufficient_statistics(result["predicted_residual_latent"], oracle_latent, latent_state)
        reconstruction = np.concatenate(recon_parts); masks = np.concatenate(mask_parts)
        current = summarize(target, reconstruction, masks, ground_masks, patients)
        current["precordial_variance_retention_per_lead"] = variance_retention_per_lead(target, reconstruction)
        current["dependent_lead_r2_qc_sensitivity"] = dependent_r2(target, reconstruction, ~qc_bad)
        metrics[system] = current
        patient_arrays[system] = patient_independent_array(target, reconstruction, patients)
        hashes[system] = {"checkpoint": str(checkpoints[system].resolve()), "sha256": sha256(checkpoints[system]), "best_epoch": checkpoint_payload.get("best_metrics", {}).get("epoch")}
        if system == "C1":
            latent_diagnostics["C1_coordinate_predictability"] = finish_latent_statistics(latent_state)
        if system == "C2":
            learned_basis = model.output_basis().detach().float()
            singular = torch.linalg.svdvals(pca_basis.T @ learned_basis).clamp(-1, 1)
            latent_diagnostics["C2_vs_PCA_principal_angles_degrees"] = torch.rad2deg(torch.acos(singular)).cpu().tolist()
        del model, reconstruction
        torch.cuda.empty_cache()

    comparisons = {
        "C1_minus_B1": paired_bootstrap(patient_arrays["C1"], patient_arrays["B1"]),
        "C2_minus_B1": paired_bootstrap(patient_arrays["C2"], patient_arrays["B1"]),
        "C2_minus_C1": paired_bootstrap(patient_arrays["C2"], patient_arrays["C1"]),
    }
    rank_curve = json.loads(args.rank_curve.read_text())
    oracle_mean = rank_curve["ranks"][str(rank_curve["K_STAR"])]["mean_independent_missing_r"]
    oracle_p05 = rank_curve["ranks"][str(rank_curve["K_STAR"])]["p05_independent_missing_r"]
    decisions = {}
    for system in ("C1", "C2"):
        comparison = comparisons[f"{system}_minus_B1"]
        mean_test = comparison["mean_independent_r"]
        harm = harm_screen(metrics[system], metrics["B1"])
        advance = mean_test["delta"] >= 0.005 and mean_test["ci95_low"] > 0 and harm["pass"]
        decisions[system] = {
            "advance": bool(advance), "harm_screen": harm,
            "oracle_mean_gap_closed": float((metrics[system]["mean_independent_missing_r"] - metrics["B1"]["mean_independent_missing_r"]) / (oracle_mean - metrics["B1"]["mean_independent_missing_r"])),
            "oracle_p05_gap_closed": float((metrics[system]["p05_independent_missing_r"] - metrics["B1"]["p05_independent_missing_r"]) / (oracle_p05 - metrics["B1"]["p05_independent_missing_r"])),
        }
    advancing = [system for system in ("C1", "C2") if decisions[system]["advance"]]
    action = "Stop low-rank development; no seed-42 system advanced." if not advancing else f"Run seeds 43/44 for B1 and advancing systems: {', '.join(advancing)}."
    now = datetime.now(timezone.utc); stamp = now.strftime("%Y%m%d_%H%M%S")
    payload = {
        "schema_version": 1, "created_at": now.isoformat(), "cohort": "PTB-XL fold 9 only",
        "n_ecgs": len(target), "n_patients": int(len(np.unique(patients))), "K_STAR": rank_curve["K_STAR"],
        "checkpoint_artifacts": hashes, "metrics": metrics, "paired_bootstrap": comparisons,
        "latent_diagnostics": latent_diagnostics, "decisions": decisions, "multi_seed_action": action,
        "qc": {"gross_limb_threshold_mv": 0.01, "excluded_from_primary": False, "dependent_sensitivity_excluded_ids": [int(ids[i]) for i in np.flatnonzero(qc_bad)]},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(payload, indent=2, allow_nan=False) + "\n"; md_text = markdown(payload)
    for name, text in ((f"seed42_evaluation_{stamp}.json", json_text), ("seed42_evaluation.json", json_text), (f"seed42_evaluation_{stamp}.md", md_text), ("seed42_evaluation.md", md_text)):
        (args.output_dir / name).write_text(text)
    print(json.dumps({"decisions": decisions, "multi_seed_action": action}, indent=2))


if __name__ == "__main__":
    main()
