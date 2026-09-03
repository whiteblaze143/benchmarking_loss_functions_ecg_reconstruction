#!/usr/bin/env python3
"""Evaluate per-lead and anatomical metrics for Stage A 3D Theta models (D0-D5)
and store them in results/convergence_per_lead_evaluation_v1/compact.sqlite.
"""

from __future__ import annotations
import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from unified_latents.engineering.models.three_d_theta_reconstruction import (
    LEAD_NAMES,
    ThreeDThetaECGAIM,
)
from scripts.train_3dtheta_ablation import SimplePTBXLDataset

DB_PATH = ROOT / "results" / "convergence_per_lead_evaluation_v1" / "compact.sqlite"

CHEST_LEADS = [6, 7, 8, 9, 10, 11]  # V1-V6
SEPTAL_LEADS = [6, 7]  # V1, V2
ANTERIOR_LEADS = [8, 9]  # V3, V4
LATERAL_CHEST_LEADS = [10, 11]  # V5, V6
HIGH_LATERAL_LEADS = [0, 4]  # I, aVL
INFERIOR_LEADS = [1, 2, 5]  # II, III, aVF

CELLS = [
    ("D0_current_id_currentloss_s42_l0", "refine-logs/convergence_10e/runs/D0_current_id_currentloss_s42_l0/best.pt", "learned_additive", "add", "D0_current_id_currentloss"),
    ("D1_theta_mul_currentloss_s42_l0", "refine-logs/convergence_10e/runs/D1_theta_mul_currentloss_s42_l0/best.pt", "theta", "mul", "D1_theta_mul_currentloss"),
    ("D2_current_id_l1_s42_l0", "refine-logs/convergence_10e/runs/D2_current_id_l1_s42_l0/best.pt", "learned_additive", "add", "D2_current_id_l1"),
    ("D3_theta_mul_l1_s42_l0", "refine-logs/convergence_10e/runs/D3_theta_mul_l1_s42_l0/best.pt", "theta", "mul", "D3_theta_mul_l1"),
    ("D4_learned12_mul_l1_s42_l0", "refine-logs/convergence_10e/runs/D4_learned12_mul_l1_s42_l0/best.pt", "learned", "mul", "D4_learned12_mul_l1"),
    ("D5_permuted_theta_mul_l1_s42_l0", "refine-logs/convergence_10e/runs/D5_permuted_theta_mul_l1_s42_l0/best.pt", "permuted_theta", "mul", "D5_permuted_theta_mul_l1"),
    ("D6_theta_add_l1_s42_l0", "refine-logs/convergence_10e/runs/D6_theta_add_l1_s42_l0/best.pt", "theta", "add", "D6_theta_add_l1"),
    ("D7_learned12_add_l1_s42_l0", "refine-logs/convergence_10e/runs/D7_learned12_add_l1_s42_l0/best.pt", "learned", "add", "D7_learned12_add_l1"),
    ("D8_random12_mul_l1_s42_l0", "refine-logs/convergence_10e/runs/D8_random12_mul_l1_s42_l0/best.pt", "random_fixed", "mul", "D8_random12_mul_l1"),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Stage A 3D Theta models on {device}...")

    val_ds = SimplePTBXLDataset(ROOT / "data" / "ptb_xl" / "tensors" / "val")
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
    total_samples = len(val_ds)

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    for mid, ckpt_rel, code_mode, fusion, arch in CELLS:
        ckpt_path = ROOT / ckpt_rel
        if not ckpt_path.exists():
            print(f"Skipping {mid}, checkpoint not found: {ckpt_path}")
            continue

        print(f"\n--- Evaluating {mid} ---")
        model = ThreeDThetaECGAIM(
            code_mode=code_mode,
            fusion=fusion,
            width=768,
            encoder_depth=8,
            decoder_depth=4,
            heads=12,
        ).to(device)

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = {
            k: v.float() if v.is_floating_point() else v
            for k, v in payload["model_state_dict"].items()
        }
        model.load_state_dict(state)
        model.eval()

        lead_pearsons = [[] for _ in range(12)]
        lead_rmses = [[] for _ in range(12)]
        lead_maes = [[] for _ in range(12)]
        lead_snrs = [[] for _ in range(12)]

        with torch.inference_mode():
            for batch in val_loader:
                targets = batch.to(device)
                x_src = targets[:, 0:1, :]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    preds = model(x_src, obs_lead_idx=0)["y_pred"].float()

                # Per-lead statistics
                for l in range(12):
                    p_l = preds[:, l]  # [B, 5000]
                    t_l = targets[:, l]

                    # Centered for Pearson
                    p_c = p_l - p_l.mean(dim=-1, keepdim=True)
                    t_c = t_l - t_l.mean(dim=-1, keepdim=True)
                    cov = (p_c * t_c).sum(dim=-1)
                    std_p = torch.sqrt((p_c**2).sum(dim=-1).clamp_min(1e-8))
                    std_t = torch.sqrt((t_c**2).sum(dim=-1).clamp_min(1e-8))
                    r = (cov / (std_p * std_t)).clamp(-1, 1).cpu().numpy()
                    lead_pearsons[l].extend(r)

                    # Errors
                    diff = p_l - t_l
                    mse = (diff**2).mean(dim=-1).cpu().numpy()
                    rmse = np.sqrt(mse)
                    mae = torch.abs(diff).mean(dim=-1).cpu().numpy()
                    lead_rmses[l].extend(rmse)
                    lead_maes[l].extend(mae)

                    # SNR (dB) = 10 * log10(var(target) / mse)
                    target_var = (t_c**2).mean(dim=-1).cpu().numpy()
                    snr = 10.0 * np.log10(np.clip(target_var / np.clip(mse, 1e-8, None), 1e-6, 1e6))
                    lead_snrs[l].extend(snr)

        # Convert to numpy
        lead_pearsons = [np.array(x) for x in lead_pearsons]
        lead_rmses = [np.array(x) for x in lead_rmses]
        lead_maes = [np.array(x) for x in lead_maes]
        lead_snrs = [np.array(x) for x in lead_snrs]

        # Aggregate missing leads (leads 1..11)
        missing_leads = list(range(1, 12))
        all_missing_r = np.mean([lead_pearsons[l] for l in missing_leads], axis=0)
        chest_r = np.mean([lead_pearsons[l] for l in CHEST_LEADS], axis=0)
        limb_leads = [1, 2, 3, 4, 5]
        limb_r = np.mean([lead_pearsons[l] for l in limb_leads], axis=0)

        septal_r = np.mean([np.mean(lead_pearsons[l]) for l in SEPTAL_LEADS])
        anterior_r = np.mean([np.mean(lead_pearsons[l]) for l in ANTERIOR_LEADS])
        lateral_chest_r = np.mean([np.mean(lead_pearsons[l]) for l in LATERAL_CHEST_LEADS])
        high_lateral_r = np.mean([np.mean(lead_pearsons[l]) for l in HIGH_LATERAL_LEADS])
        inferior_r = np.mean([np.mean(lead_pearsons[l]) for l in INFERIOR_LEADS])

        mean_all_missing = float(np.mean(all_missing_r))
        p05_all_missing = float(np.quantile(all_missing_r, 0.05))
        mean_chest = float(np.mean(chest_r))
        p05_chest = float(np.quantile(chest_r, 0.05))
        mean_limb = float(np.mean(limb_r))
        p05_limb = float(np.quantile(limb_r, 0.05))

        print(f"  Missing r: {mean_all_missing:.4f} | Chest r: {mean_chest:.4f} | Limb r: {mean_limb:.4f}")

        # Insert or replace in evaluations table
        now = dt.datetime.now(dt.timezone.utc).isoformat()
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
                mid,
                mid,
                "Stage A 3D Theta Spatial Benchmark",
                arch,
                "I",
                0,
                "val",
                total_samples,
                mean_all_missing,
                p05_all_missing,
                mean_chest,
                p05_chest,
                mean_limb,
                p05_limb,
                float(septal_r),
                float(anterior_r),
                float(lateral_chest_r),
                float(high_lateral_r),
                float(inferior_r),
                json.dumps({"code_mode": code_mode, "fusion": fusion}),
                now,
            ),
        )

        # Insert per-lead metrics
        for l in range(12):
            cur.execute(
                """
                INSERT OR REPLACE INTO per_lead_metrics (
                    model_id, lead_idx, lead_name, is_observed,
                    mean_pearson, p05_pearson, p50_pearson, p95_pearson,
                    rmse, mae, snr_db
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mid,
                    l,
                    LEAD_NAMES[l],
                    1 if l == 0 else 0,
                    float(np.mean(lead_pearsons[l])),
                    float(np.quantile(lead_pearsons[l], 0.05)),
                    float(np.median(lead_pearsons[l])),
                    float(np.quantile(lead_pearsons[l], 0.95)),
                    float(np.mean(lead_rmses[l])),
                    float(np.mean(lead_maes[l])),
                    float(np.mean(lead_snrs[l])),
                ),
            )

        con.commit()

    con.close()
    print("\nAll Stage A 3D Theta evaluations committed to SQLite successfully!")


if __name__ == "__main__":
    main()
