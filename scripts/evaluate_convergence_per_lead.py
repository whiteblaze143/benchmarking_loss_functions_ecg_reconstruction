#!/usr/bin/env python3
"""Evaluate per-lead and chest reconstruction fidelity on PTB-XL evaluation cohort.

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
3. Persists evaluations to SQLite database and JSON summaries.
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import os
import re
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

if hasattr(os, "cpu_count") and os.cpu_count():
    torch.set_num_threads(os.cpu_count())

from scripts.train_mcma_3lead import PTBXLDataset
from scripts.train_1lead_wavelet_ssl_mtl import (
    build_model,
    forward_model,
    waveform_from_batch,
    LEADS,
)

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CHEST_LEADS = [6, 7, 8, 9, 10, 11]  # V1-V6
SEPTAL_LEADS = [6, 7]  # V1, V2
ANTERIOR_LEADS = [8, 9]  # V3, V4
LATERAL_CHEST_LEADS = [10, 11]  # V5, V6
HIGH_LATERAL_LEADS = [0, 4]  # I, aVL
INFERIOR_LEADS = [1, 2, 5]  # II, III, aVF

def init_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                model_id TEXT PRIMARY KEY,
                run_name TEXT,
                track TEXT,
                architecture TEXT,
                observed_lead TEXT,
                observed_lead_idx INTEGER,
                split TEXT,
                total_samples INTEGER,
                mean_all_missing_r REAL,
                p05_all_missing_r REAL,
                mean_chest_r REAL,
                p05_chest_r REAL,
                mean_limb_r REAL,
                p05_limb_r REAL,
                mean_septal_r REAL,
                mean_anterior_r REAL,
                mean_lateral_chest_r REAL,
                mean_high_lateral_r REAL,
                mean_inferior_r REAL,
                details_json TEXT,
                completed_at TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS per_lead_metrics (
                model_id TEXT,
                lead_idx INTEGER,
                lead_name TEXT,
                is_observed INTEGER,
                mean_pearson REAL,
                p05_pearson REAL,
                p50_pearson REAL,
                p95_pearson REAL,
                rmse REAL,
                mae REAL,
                snr_db REAL,
                PRIMARY KEY (model_id, lead_idx)
            )
        """)
        con.commit()

def clean_arch_name(name: str) -> str:
    name = re.sub(r'^conv(10|15)e_', '', name)
    name = re.sub(r'_s\d+_l[01]$', '', name)
    return name

def compute_lead_pearson(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute centered Pearson correlation per sample for a batch of signals [B, T]."""
    p = pred - pred.mean(dim=-1, keepdim=True)
    t = target - target.mean(dim=-1, keepdim=True)
    p_norm = torch.norm(p, dim=-1).clamp_min(1e-8)
    t_norm = torch.norm(t, dim=-1).clamp_min(1e-8)
    dot = (p * t).sum(dim=-1)
    return dot / (p_norm * t_norm)

@torch.no_grad()
def evaluate_model_per_lead(
    model: torch.nn.Module,
    loader: DataLoader,
    observed_lead: int,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Dict[str, Any]:
    model.eval()
    
    # Store per-lead sample metrics: lead_idx -> list of values
    lead_pearsons: Dict[int, List[float]] = {l: [] for l in range(12)}
    lead_mses: Dict[int, List[float]] = {l: [] for l in range(12)}
    lead_maes: Dict[int, List[float]] = {l: [] for l in range(12)}
    lead_snrs: Dict[int, List[float]] = {l: [] for l in range(12)}
    all_missing_pearsons: List[float] = []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        
        y = waveform_from_batch(batch)[..., :5000].to(device) # [B, 12, 5000]
        B = y.shape[0]

        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
            res = forward_model(
                model, y, [observed_lead], compute_delineation=False, compute_ssl=False
            )
            y_pred = res["y_pred"][..., :5000].float() # [B, 12, 5000]
            y_float = y.float()

        # Aggregate missing Pearson across all 11 missing leads concatenated
        m_mask = torch.ones(12, dtype=torch.bool, device=device)
        m_mask[observed_lead] = False
        p_miss = y_pred[:, m_mask]
        t_miss = y_float[:, m_mask]
        p_miss = p_miss - p_miss.mean(-1, keepdim=True)
        t_miss = t_miss - t_miss.mean(-1, keepdim=True)
        agg_r = F.cosine_similarity(p_miss.flatten(1), t_miss.flatten(1), dim=1)
        all_missing_pearsons.extend(agg_r.cpu().tolist())

        # Individual lead calculations
        for l in range(12):
            pred_l = y_pred[:, l, :] # [B, 5000]
            targ_l = y_float[:, l, :] # [B, 5000]
            
            r_l = compute_lead_pearson(pred_l, targ_l).cpu().tolist()
            mse_l = ((pred_l - targ_l) ** 2).mean(dim=-1).cpu().tolist()
            mae_l = (pred_l - targ_l).abs().mean(dim=-1).cpu().tolist()
            
            # SNR = 10 * log10( sum(targ^2) / sum((pred - targ)^2) )
            signal_power = (targ_l ** 2).sum(dim=-1).clamp_min(1e-8)
            noise_power = ((pred_l - targ_l) ** 2).sum(dim=-1).clamp_min(1e-8)
            snr_l = (10.0 * torch.log10(signal_power / noise_power)).cpu().tolist()

            lead_pearsons[l].extend(r_l)
            lead_mses[l].extend(mse_l)
            lead_maes[l].extend(mae_l)
            lead_snrs[l].extend(snr_l)

    # Compute summary statistics per lead
    per_lead_summary = {}
    for l in range(12):
        arr_r = np.array(lead_pearsons[l])
        arr_mse = np.array(lead_mses[l])
        arr_mae = np.array(lead_maes[l])
        arr_snr = np.array(lead_snrs[l])
        
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

    # Anatomical Group Summary Calculations
    missing_leads = [l for l in range(12) if l != observed_lead]
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
        "per_lead": per_lead_summary
    }

def process_single_run(
    run_dir: Path,
    data_dir: Path,
    split: str,
    db_path: Path,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
    smoke: bool = False
) -> bool:
    run_name = run_dir.name
    config_path = run_dir / "config.json"
    best_ckpt = run_dir / "best.pt"
    resume_ckpt = run_dir / "resume.pt"

    ckpt_path = best_ckpt if best_ckpt.is_file() else (resume_ckpt if resume_ckpt.is_file() else None)
    if not config_path.is_file() or ckpt_path is None:
        return False

    with sqlite3.connect(db_path) as con:
        row = con.execute("SELECT model_id FROM evaluations WHERE model_id = ?", (run_name,)).fetchone()
        if row is not None and not smoke:
            return False # Already evaluated

    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Evaluating per-lead for: {run_name} (using {ckpt_path.name})")

    try:
        cfg = json.loads(config_path.read_text())
        args = argparse.Namespace(**cfg)
        
        # Ensure critical flags exist
        if not hasattr(args, "no_delineation_head"): args.no_delineation_head = False
        if not hasattr(args, "no_fiducial_head"): args.no_fiducial_head = False
        if not hasattr(args, "mask_type_mode"): args.mask_type_mode = "legacy"
        if not hasattr(args, "custom_wavelet_asset"): args.custom_wavelet_asset = None
        if not hasattr(args, "view_a_custom_wavelet_asset"): args.view_a_custom_wavelet_asset = None
        if not hasattr(args, "view_b_custom_wavelet_asset"): args.view_b_custom_wavelet_asset = None
        if not hasattr(args, "view_a_bank"): args.view_a_bank = "inherit"
        if not hasattr(args, "view_b_bank"): args.view_b_bank = "inherit"

        observed_lead = args.observed_leads[0] if isinstance(args.observed_leads, list) else int(args.observed_leads)
        
        model = build_model(args).to(device)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(state, strict=False)

        split_dir = data_dir / split
        ds = PTBXLDataset(str(split_dir))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        max_batches = 2 if smoke else None
        res = evaluate_model_per_lead(model, loader, observed_lead, device, max_batches=max_batches)

        # Database Insertion
        track = "15-Epoch" if "conv15e" in run_name else ("10-Epoch" if "conv10e" in run_name else "Screening")
        arch = clean_arch_name(run_name)
        obs_lead_str = LEAD_NAMES[observed_lead]
        now_iso = dt.datetime.now(dt.timezone.utc).isoformat()

        with sqlite3.connect(db_path) as con:
            con.execute("""
                INSERT OR REPLACE INTO evaluations (
                    model_id, run_name, track, architecture, observed_lead, observed_lead_idx,
                    split, total_samples, mean_all_missing_r, p05_all_missing_r,
                    mean_chest_r, p05_chest_r, mean_limb_r, p05_limb_r,
                    mean_septal_r, mean_anterior_r, mean_lateral_chest_r,
                    mean_high_lateral_r, mean_inferior_r, details_json, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_name, run_name, track, arch, obs_lead_str, observed_lead,
                split, res["total_samples"], res["mean_all_missing_r"], res["p05_all_missing_r"],
                res["mean_chest_r"], res["p05_chest_r"], res["mean_limb_r"], res["p05_limb_r"],
                res["mean_septal_r"], res["mean_anterior_r"], res["mean_lateral_chest_r"],
                res["mean_high_lateral_r"], res["mean_inferior_r"], json.dumps(res["per_lead"]), now_iso
            ))

            for l_idx, pmetrics in res["per_lead"].items():
                con.execute("""
                    INSERT OR REPLACE INTO per_lead_metrics (
                        model_id, lead_idx, lead_name, is_observed,
                        mean_pearson, p05_pearson, p50_pearson, p95_pearson,
                        rmse, mae, snr_db
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_name, l_idx, pmetrics["lead_name"], int(pmetrics["is_observed"]),
                    pmetrics["mean_pearson"], pmetrics["p05_pearson"], pmetrics["p50_pearson"], pmetrics["p95_pearson"],
                    pmetrics["rmse"], pmetrics["mae"], pmetrics["snr_db"]
                ))
            con.commit()

        print(f"  ✓ {run_name}: All-Missing r={res['mean_all_missing_r']:.4f} (p05={res['p05_all_missing_r']:.4f}) | Chest (V1-V6) r={res['mean_chest_r']:.4f} | Limb r={res['mean_limb_r']:.4f}")
        return True
    except Exception as e:
        print(f"  ✗ Error evaluating {run_name}: {e}")
        import traceback; traceback.print_exc()
        return False

def main():
    p = argparse.ArgumentParser(description="Per-lead & chest ECG reconstruction evaluator daemon")
    p.add_argument("--runs-dir", default="refine-logs/convergence_10e/runs")
    p.add_argument("--data-dir", default="data/ptb_xl/tensors")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--output-db", default="results/convergence_per_lead_evaluation_v1/compact.sqlite")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--poll-seconds", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force-cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.force_cpu or not torch.cuda.is_available() else "cuda")
    print(f"Starting Per-Lead Evaluator on {device} (split={args.split}, smoke={args.smoke})")

    db_path = _ROOT / args.output_db
    init_db(db_path)

    runs_dir = _ROOT / args.runs_dir
    data_dir = _ROOT / args.data_dir

    while True:
        candidates = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
        evaluated_count = 0
        for cd in candidates:
            success = process_single_run(
                cd, data_dir, args.split, db_path, device,
                batch_size=args.batch_size, num_workers=args.num_workers, smoke=args.smoke
            )
            if success:
                evaluated_count += 1
            if args.smoke and evaluated_count >= 2:
                print("Smoke verification complete on 2 runs.")
                break

        if args.poll_seconds <= 0 or args.smoke:
            break

        time.sleep(args.poll_seconds)

if __name__ == "__main__":
    main()
