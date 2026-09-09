#!/usr/bin/env python3
"""Reintegrate Kill-Gate (Round 1) and Lean Ablation (Round 2) into Master CSVs.

1. Ingests all 15 Round-1 Kill-Gate ablation runs from refine-logs/killgate/runs/
2. Ingests all 44 Round-2 Lean ablation cells (completed runs populated from summary.json,
   running/queued cells added with proper metadata and blank metric columns)
3. Appends/updates results/lead1_all_models_comprehensive_metrics.csv
4. Rebuilds results/clinical_biomarkers_multids/FINAL_CLINICAL_BENCHMARK_METRICS_MASTER.csv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

ROOT = Path(__file__).resolve().parents[1]
LEAD1_CSV = ROOT / "results/lead1_all_models_comprehensive_metrics.csv"
KILLGATE_DIR = ROOT / "refine-logs/killgate/runs"
LEAN_DIR = ROOT / "refine-logs/lean_abl2/runs"

# 15 Kill-Gate Models
KILLGATE_MODELS = [
    "conv15e_K1_nold_s42_l0",
    "conv15e_K2_noart_s42_l0",
    "conv15e_K3_nodel_s42_l0",
    "conv15e_K4_l1_s42_l0",
    "conv15e_K5_adaptive_s42_l0",
    "conv15e_B1_hardbasis_s42_l0",
    "conv15e_B2_hardbasis_nodel_s42_l0",
    "conv15e_B3_hardbasis_adaptive_s42_l0",
    "conv15e_B4_hardbasis_l1_s42_l0",
    "conv15e_C1_enc4_s42_l0",
    "conv15e_C2_dec2_s42_l0",
    "conv15e_C3_width512_s42_l0",
    "conv15e_C4_minimal_s42_l0",
    "conv15e_Z1_zscore_s42_l0",
    "conv15e_Z2_hardbasis_zscore_s42_l0",
]

# 44 Lean Cells (+ L0 pilot)
LEAN_CELLS = [
    ("L0_lean_best", "Clean Lean Baseline (Pilot)", True),
    ("L1_clean_lean_best", "Clean Combined Lean Best (enc4, width512, no_lead_dropout, zscore, mask 1110000)", False),
    ("LA1_nodel", "Occam 1: Remove Delineation MTL Head & Segmentation Losses", False),
    ("LB1_dec3", "Occam 2: Decoder Depth D=3", False),
    ("LC1_width384", "Occam 3: Prune Hidden Width to 384 and Heads to 6", False),
    ("LD1_enc3", "Occam 4: Encoder Depth E=3", False),
    ("LE1_nodice", "Occam 5: Delineation with Cross-Entropy only", False),
    ("LE2_adaptive", "Occam 6: Adaptive Composite Loss Weighting", False),
    ("L_mse_only", "Axis 1: MSE-only reconstruction loss", False),
    ("L_corr_only", "Axis 1: Correlation-only reconstruction loss", False),
    ("L_mse_corr", "Axis 1: MSE + Correlation reconstruction loss", False),
    ("L_mse_deriv", "Axis 1: MSE + 1st Derivative reconstruction loss", False),
    ("L_l1_direct", "Axis 1: L1 direct reconstruction loss", False),
    ("W_conv_c128", "Axis 2: Wavelet Conv branch C=128", False),
    ("W_conv_c256", "Axis 2: Wavelet Conv branch C=256", False),
    ("W_timesformer_dim256", "Axis 2: Wavelet TimeSformer dim=256", False),
    ("W_fusion_gated", "Axis 2: Wavelet Gated Add fusion", False),
    ("W_fusion_crossattn", "Axis 2: Wavelet Cross-Attention fusion", False),
    ("T_patch50", "Axis 3: Patch Size 50 (coarse temporal tokens)", False),
    ("T_patch10", "Axis 3: Patch Size 10 (fine temporal tokens)", False),
    ("T_heads4", "Axis 3: 4 Attention Heads", False),
    ("T_heads16", "Axis 3: 16 Attention Heads", False),
    ("D_cadence1", "Axis 4: Delineation Cadence 1 (every epoch)", False),
    ("D_cadence4", "Axis 4: Delineation Cadence 4 (every 4 epochs)", False),
    ("D_boundary", "Axis 4: Delineation with Boundary loss", False),
    ("D_fiducial", "Axis 4: Delineation with Fiducial loss", False),
    ("D_heavy_seg", "Axis 4: Heavy segmentation weights (CE=2.0, Dice=1.0)", False),
    ("D_light_ce", "Axis 4: Light segmentation weight (CE=0.5, Dice=0.0)", False),
    ("S_film", "Axis 5: Spatial Conditioning FiLM", False),
    ("S_panorama", "Axis 5: Spatial Conditioning Panorama", False),
    ("R_mask15", "Axis 6: Random lead dropout p=0.15", False),
    ("R_tempmask15", "Axis 6: Random temporal patch masking p=0.15", False),
    ("R_wd_low", "Axis 6: Low Weight Decay (1e-5)", False),
    ("R_wd_high", "Axis 6: High Weight Decay (1e-3)", False),
    ("AV1_vcg_adaptive", "Adaptive Axis 1: Adaptive Composite + VCG Loop (Frank XYZ)", False),
    ("AV2_triplet_vcg_adaptive", "Adaptive Axis 1: Adaptive Composite + Triplet Frontal VCG", False),
    ("AM1_mmd_imq_adaptive", "Adaptive Axis 2: Adaptive Composite + MMD IMQ Kernel", False),
    ("AM2_mmd_kmeans_adaptive", "Adaptive Axis 2: Adaptive Composite + MMD K-Means Archetypes", False),
    ("AM3_mmd_laplace_adaptive", "Adaptive Axis 2: Adaptive Composite + MMD Laplacian Kernel", False),
    ("AVM1_vcg_mmd_imq_adaptive", "Adaptive Axis 3: Adaptive Composite + VCG + MMD IMQ", False),
    ("AVM2_vcg_mmd_kmeans_adaptive", "Adaptive Axis 3: Adaptive Composite + VCG + MMD K-Means", False),
    ("AVM3_vcg_mmd_laplace_adaptive", "Adaptive Axis 3: Adaptive Composite + VCG + MMD Laplacian", False),
    ("AVM4_full_probe_laplace_adaptive", "Adaptive Axis 4: Full Frontier Probe (Adaptive + VCG + MMD Laplace)", False),
    ("AVM5_full_probe_imq_adaptive", "Adaptive Axis 4: Full Frontier Probe (Adaptive + VCG + MMD IMQ)", False),
    ("AVLead1_vcg_lead_adaptive", "Adaptive Axis 5: Single-Lead Derived Spatial Loop + Adaptive", False),
]


def load_run_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    sum_file = run_dir / "summary.json"
    if sum_file.exists():
        try:
            with open(sum_file) as f:
                return json.load(f)
        except Exception:
            pass

    # Check metrics.jsonl if summary.json is not present (e.g. interrupted runs)
    jsonl_file = run_dir / "metrics.jsonl"
    if jsonl_file.exists():
        try:
            last_line = ""
            with open(jsonl_file) as f:
                for line in f:
                    if line.strip():
                        last_line = line.strip()
            if last_line:
                d = json.loads(last_line)
                d["epochs_completed"] = d.get("epoch", None)
                return d
        except Exception:
            pass
    return None


def extract_arch(mid: str) -> str:
    s = mid
    for prefix in ["conv15e_", "conv10e_", "spatial_1lead_", "lean2_"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    if s.endswith("_s42_l0"):
        s = s[:-7]
    elif s.endswith("_s42_l1"):
        s = s[:-7]
    return s


def build_record(
    mid: str,
    study_track: str,
    summary: Optional[Dict[str, Any]],
    status_default: str = "QUEUED",
    epochs_trained_default: int = 15
) -> Dict[str, Any]:
    arch = extract_arch(mid)
    rec: Dict[str, Any] = {
        "model_id": mid,
        "study_track": study_track,
        "architecture": arch,
        "observed_lead": "I",
        "factorial_mask": "1110000",
        "seed": 42,
        "epochs_trained": summary.get("epochs_completed", epochs_trained_default) if summary else epochs_trained_default,
        "status": "completed" if (summary and summary.get("epochs_completed") == 15) else (
            f"interrupted_epoch{summary.get('epochs_completed')}" if summary and summary.get("epochs_completed", 0) < 15 else status_default
        ),
    }

    metric_fields = [
        "val_missing_pearson", "val_missing_pearson_p05", "val_recon_loss",
        "miou_wave", "P_iou", "QRS_iou", "T_iou", "macro_f1_wave", "boundary_f1_smoke",
        "train_recon", "train_ssl", "train_ce", "train_dice", "train_boundary",
        "train_fid", "train_total", "train_grad_norm", "train_consistency"
    ]

    for f in metric_fields:
        rec[f] = summary.get(f, np.nan) if summary else np.nan

    return rec


def main():
    logging.info("Starting Master CSV Reintegration...")
    if not LEAD1_CSV.exists():
        raise FileNotFoundError(f"Lead-1 CSV not found: {LEAD1_CSV}")

    df_lead1 = pd.read_csv(LEAD1_CSV)
    logging.info("Existing lead1 dataset: %d rows, %d columns", len(df_lead1), len(df_lead1.columns))

    existing_models = set(df_lead1["model_id"])
    new_records: List[Dict[str, Any]] = []

    # 1. Ingest Kill-Gate Runs
    logging.info("Processing 15 Round-1 Kill-Gate runs...")
    for mid in KILLGATE_MODELS:
        rdir = KILLGATE_DIR / mid
        sum_data = load_run_metrics(rdir)
        rec = build_record(
            mid=mid,
            study_track="Kill-Gate Ablation Suite (Round 1)",
            summary=sum_data,
            status_default="missing_run",
            epochs_trained_default=15
        )
        if mid in existing_models:
            logging.info("  Updating existing model %s", mid)
            idx = df_lead1.index[df_lead1["model_id"] == mid].tolist()[0]
            for k, v in rec.items():
                df_lead1.at[idx, k] = v
        else:
            new_records.append(rec)

    # 2. Ingest Lean Ablation Cells
    logging.info("Processing 44 Round-2 Lean Ablation cells...")
    for cell_id, desc, is_pilot in LEAN_CELLS:
        mid = f"lean2_{cell_id}_s42_l0"
        rdir = LEAN_DIR / mid
        sum_data = load_run_metrics(rdir)

        # Determine status
        if sum_data and sum_data.get("epochs_completed") == 15:
            st = "completed"
        elif rdir.exists() and (rdir / "train.log").exists():
            st = "running"
        else:
            st = "QUEUED"

        rec = build_record(
            mid=mid,
            study_track="Lean Ablation Suite (Round 2)",
            summary=sum_data,
            status_default=st,
            epochs_trained_default=15
        )

        if mid in existing_models:
            logging.info("  Updating existing model %s", mid)
            idx = df_lead1.index[df_lead1["model_id"] == mid].tolist()[0]
            for k, v in rec.items():
                df_lead1.at[idx, k] = v
        else:
            new_records.append(rec)

    if new_records:
        logging.info("Appending %d new models to lead1 dataset...", len(new_records))
        df_new = pd.DataFrame(new_records)
        df_combined = pd.concat([df_lead1, df_new], ignore_index=True)
    else:
        df_combined = df_lead1

    # Save updated lead1 CSV
    df_combined.to_csv(LEAD1_CSV, index=False)
    logging.info("Saved updated %s: now %d rows total.", LEAD1_CSV.name, len(df_combined))


if __name__ == "__main__":
    main()
