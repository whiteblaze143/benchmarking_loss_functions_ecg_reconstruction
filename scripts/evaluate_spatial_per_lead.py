#!/usr/bin/env python3
"""Evaluate Spatial ECG-AIM models on PTB-XL validation cohort strictly on CPU (6 cores).

Computes:
1. Individual Lead Pearson correlation (mean, p05, p50, p95), RMSE, MAE, SNR (dB)
   for all 12 leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.
2. Anatomical Lead Group Aggregations:
   - Precordial / Chest Leads (V1–V6)
   - Limb Leads (excluding observed lead)
   - Septal (V1, V2)
   - Anterior (V3, V4)
   - Lateral Precordial (V5, V6)
   - High Lateral (I, aVL)
   - Inferior (II, III, aVF)
3. Persists evaluations to results/convergence_per_lead_evaluation_v1/compact.sqlite
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.bootstrap_paths import setup_import_paths
setup_import_paths(include_fairseq=True)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.train_3dreconqt_reference import PTBXLPreprocessedDataset
from unified_latents.engineering.experimental.aim_1_lead import build_alitok_vae_1d
from scripts.train_1lead_spatial_ecg_aim import mask_unobserved_leads, make_lead_indices

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CHEST_LEADS = [6, 7, 8, 9, 10, 11]
SEPTAL_LEADS = [6, 7]
ANTERIOR_LEADS = [8, 9]
LATERAL_CHEST_LEADS = [10, 11]
HIGH_LATERAL_LEADS = [0, 4]
INFERIOR_LEADS = [1, 2, 5]


def compute_metrics_arrays(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute per-record, per-lead metrics.
    
    y_true, y_pred: [N, 12, 5000]
    """
    N, L, T = y_true.shape
    r_matrix = np.zeros((N, L), dtype=np.float32)
    mse_matrix = np.zeros((N, L), dtype=np.float32)
    mae_matrix = np.zeros((N, L), dtype=np.float32)
    snr_matrix = np.zeros((N, L), dtype=np.float32)

    for l in range(L):
        yt = y_true[:, l, :]
        yp = y_pred[:, l, :]

        yt_mean = yt.mean(axis=-1, keepdims=True)
        yp_mean = yp.mean(axis=-1, keepdims=True)

        yt_c = yt - yt_mean
        yp_c = yp - yp_mean

        num = (yt_c * yp_c).sum(axis=-1)
        den = np.sqrt((yt_c ** 2).sum(axis=-1) * (yp_c ** 2).sum(axis=-1)) + 1e-8
        r_matrix[:, l] = num / den

        diff = yt - yp
        mse = (diff ** 2).mean(axis=-1)
        mse_matrix[:, l] = mse
        mae_matrix[:, l] = np.abs(diff).mean(axis=-1)

        signal_pwr = (yt ** 2).mean(axis=-1)
        noise_pwr = mse + 1e-8
        snr_matrix[:, l] = 10.0 * np.log10(np.maximum(signal_pwr, 1e-8) / noise_pwr)

    return r_matrix, mse_matrix, mae_matrix, snr_matrix


def evaluate_spatial_model(
    model: torch.nn.Module,
    observed_lead: int,
    data_dir: Path,
    split: str,
    batch_size: int = 32,
    smoke: bool = False,
) -> Dict[str, Any]:
    val_dataset = PTBXLPreprocessedDataset(data_dir / split)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    all_y_true = []
    all_y_pred = []

    model.eval()
    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if isinstance(batch, (tuple, list)):
                waveforms = batch[0]
            elif isinstance(batch, dict) and "waveform" in batch:
                waveforms = batch["waveform"]
            else:
                waveforms = batch

            waveforms = waveforms.float()
            B = waveforms.shape[0]

            masked = mask_unobserved_leads(waveforms, [observed_lead])
            lead_idx = make_lead_indices([observed_lead], B, "cpu")
            res = model(masked, y_full=waveforms, lead_indices=lead_idx, mode="stage1")
            pred = res["y_pred"][:, :12, :5000].cpu().numpy()

            all_y_true.append(waveforms.cpu().numpy())
            all_y_pred.append(pred)

            if smoke and b_idx >= 2:
                print("Smoke mode: stopping after 3 batches.")
                break

    y_true_all = np.concatenate(all_y_true, axis=0)
    y_pred_all = np.concatenate(all_y_pred, axis=0)

    r_mat, mse_mat, mae_mat, snr_mat = compute_metrics_arrays(y_true_all, y_pred_all)

    per_lead_summary = {}
    missing_indices = [l for l in range(12) if l != observed_lead]
    all_missing_pearsons = r_mat[:, missing_indices].flatten()

    for l in range(12):
        arr_r = r_mat[:, l]
        arr_mse = mse_mat[:, l]
        arr_mae = mae_mat[:, l]
        arr_snr = snr_mat[:, l]

        per_lead_summary[l] = {
            "lead_name": LEAD_NAMES[l],
            "is_observed": (l == observed_lead),
            "mean_pearson": float(np.mean(arr_r)),
            "p05_pearson": float(np.quantile(arr_r, 0.05)),
            "p50_pearson": float(np.median(arr_r)),
            "p95_pearson": float(np.quantile(arr_r, 0.95)),
            "rmse": float(np.sqrt(np.mean(arr_mse))),
            "mae": float(np.mean(arr_mae)),
            "snr_db": float(np.mean(arr_snr)),
        }

    chest_missing = [l for l in CHEST_LEADS if l != observed_lead]
    limb_missing = [l for l in range(6) if l != observed_lead]
    septal_missing = [l for l in SEPTAL_LEADS if l != observed_lead]
    anterior_missing = [l for l in ANTERIOR_LEADS if l != observed_lead]
    lat_chest_missing = [l for l in LATERAL_CHEST_LEADS if l != observed_lead]
    high_lat_missing = [l for l in HIGH_LATERAL_LEADS if l != observed_lead]
    inferior_missing = [l for l in INFERIOR_LEADS if l != observed_lead]

    def group_stats(lead_indices):
        if not lead_indices:
            return 0.0, 0.0
        r_means = [per_lead_summary[l]["mean_pearson"] for l in lead_indices]
        p05_means = [per_lead_summary[l]["p05_pearson"] for l in lead_indices]
        return float(np.mean(r_means)), float(np.mean(p05_means))

    mean_chest_r, p05_chest_r = group_stats(chest_missing)
    mean_limb_r, p05_limb_r = group_stats(limb_missing)
    mean_septal_r, _ = group_stats(septal_missing)
    mean_anterior_r, _ = group_stats(anterior_missing)
    mean_lat_chest_r, _ = group_stats(lat_chest_missing)
    mean_high_lat_r, _ = group_stats(high_lat_missing)
    mean_inferior_r, _ = group_stats(inferior_missing)

    return {
        "total_samples": len(all_missing_pearsons),
        "mean_all_missing_r": float(np.mean(all_missing_pearsons)),
        "p05_all_missing_r": float(np.quantile(all_missing_pearsons, 0.05)),
        "mean_chest_r": mean_chest_r,
        "p05_chest_r": p05_chest_r,
        "mean_limb_r": mean_limb_r,
        "p05_limb_r": p05_limb_r,
        "mean_septal_r": mean_septal_r,
        "mean_anterior_r": mean_anterior_r,
        "mean_lateral_chest_r": mean_lat_chest_r,
        "mean_high_lateral_r": mean_high_lat_r,
        "mean_inferior_r": mean_inferior_r,
        "per_lead": per_lead_summary,
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate a spatial model checkpoint on PTB-XL using CPU")
    p.add_argument("--checkpoint-path", required=True, help="Path to materialized .pt checkpoint")
    p.add_argument("--model-id", default=None, help="Model ID (defaults to checkpoint stem)")
    p.add_argument("--data-dir", default="data/ptb_xl/tensors")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--output-db", default="results/convergence_per_lead_evaluation_v1/compact.sqlite")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(args.threads)

    ckpt_path = Path(args.checkpoint_path).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model_id = args.model_id or ckpt_path.stem
    observed_lead = 0
    if "_l1" in model_id:
        observed_lead = 1
    elif "_l7" in model_id:
        observed_lead = 7

    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Evaluating {model_id} on CPU ({args.threads} threads, observed lead {observed_lead})")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = payload.get("architecture", "ecg_aim")
    arch_name = {
        "ecg_aim": "ecg_aim_v1",
        "ecg_aim_spatial": "ecg_aim_spatial_v1",
        "ecg_aim_panorama_author": "ecg_aim_panorama_author_v1",
        "ecg_aim_exact_theta": "ecg_aim_exact_theta_factorial_v1",
    }.get(arch, "ecg_aim_v1")

    model = build_alitok_vae_1d(
        architecture=arch_name,
        target_len=payload.get("target_len", 5000),
        patch_size=payload.get("alitok_patch_size", 25),
        encoder_depth=payload.get("alitok_encoder_depth", 8),
        decoder_depth=payload.get("alitok_decoder_depth", 4),
        lead_conditioning_mode=payload.get("lead_conditioning_mode", "learned"),
        use_learned_lead_id=payload.get("use_learned_lead_id", False),
        use_relative_geometry=payload.get("use_relative_geometry", False),
        use_spatial_film=payload.get("use_spatial_film", False),
        spatial_gain_init=payload.get("spatial_gain_init", 0.1),
        geometry_control=payload.get("geometry_control", "standard"),
    ).to("cpu")

    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    res = evaluate_spatial_model(
        model,
        observed_lead=observed_lead,
        data_dir=_ROOT / args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        smoke=args.smoke,
    )

    db_path = _ROOT / args.output_db
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
        obs_name = LEAD_NAMES[observed_lead]

        cur.execute(
            """
            INSERT OR REPLACE INTO evaluations (
                model_id, run_name, track, architecture, observed_lead, observed_lead_idx,
                split, total_samples, mean_all_missing_r, p05_all_missing_r,
                mean_chest_r, p05_chest_r, mean_limb_r, p05_limb_r,
                mean_septal_r, mean_anterior_r, mean_lateral_chest_r, mean_high_lateral_r, mean_inferior_r,
                details_json, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                model_id,
                "3-Epoch Spatial Architecture Grid",
                arch,
                obs_name,
                observed_lead,
                args.split,
                res["total_samples"],
                res["mean_all_missing_r"],
                res["p05_all_missing_r"],
                res["mean_chest_r"],
                res["p05_chest_r"],
                res["mean_limb_r"],
                res["p05_limb_r"],
                res["mean_septal_r"],
                res["mean_anterior_r"],
                res["mean_lateral_chest_r"],
                res["mean_high_lateral_r"],
                res["mean_inferior_r"],
                json.dumps(res),
                now_iso,
            ),
        )

        for l_idx, pmetrics in res["per_lead"].items():
            cur.execute(
                """
                INSERT OR REPLACE INTO per_lead_metrics (
                    model_id, lead_idx, lead_name, is_observed,
                    mean_pearson, p05_pearson, p50_pearson, p95_pearson,
                    rmse, mae, snr_db
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    l_idx,
                    pmetrics["lead_name"],
                    int(pmetrics["is_observed"]),
                    pmetrics["mean_pearson"],
                    pmetrics["p05_pearson"],
                    pmetrics["p50_pearson"],
                    pmetrics["p95_pearson"],
                    pmetrics["rmse"],
                    pmetrics["mae"],
                    pmetrics["snr_db"],
                ),
            )
        con.commit()

    print(f"  ✓ {model_id}: All-Missing r={res['mean_all_missing_r']:.4f} (p05={res['p05_all_missing_r']:.4f}) | Chest r={res['mean_chest_r']:.4f} | Limb r={res['mean_limb_r']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
