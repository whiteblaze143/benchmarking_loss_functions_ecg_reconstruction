#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for 3DRECON-QT Reference Models.
Computes per-lead, all-12, missing-11, chest, transition, and physics consistency metrics
across the entire PTB-XL validation split (N = 2,183 patients).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from unified_latents.engineering.models.reconqt_reference import (
    LEAD_NAMES,
    ThreeDReconQTReference,
    extract_source_vector,
    normalize_record,
)
from scripts.train_3dreconqt_reference import PTBXLPreprocessedDataset, compute_lead_pearson


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_dir: Path,
    device: torch.device,
    batch_size: int = 32,
) -> Dict:
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    provenance = ckpt.get("provenance", {})

    code_mode = provenance.get("code_mode", "theta")
    source_mode = provenance.get("source_mode", "lead_I")
    normalization_mode = provenance.get("normalization", "strict_deployable")

    # Instantiate model
    model = ThreeDReconQTReference(
        input_channels=1,
        latent_channels=provenance.get("latent_channels", 256),
        theta_hidden=provenance.get("theta_hidden", 128),
        code_mode=code_mode,
        target_length=5000,
        enable_qt_head=False,
    )
    # Convert fp16 state_dict back to float32 for inference
    float32_state = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}
    model.load_state_dict(float32_state)
    model.to(device)
    model.eval()

    val_ds = PTBXLPreprocessedDataset(data_dir / "val")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    idx_I = LEAD_NAMES.index("I")
    idx_II = LEAD_NAMES.index("II")
    idx_III = LEAD_NAMES.index("III")
    idx_aVR = LEAD_NAMES.index("aVR")
    idx_aVL = LEAD_NAMES.index("aVL")
    idx_aVF = LEAD_NAMES.index("aVF")
    chest_indices = [LEAD_NAMES.index(f"V{k}") for k in range(1, 7)]
    missing_indices = [l for l in range(12) if l != idx_I] if source_mode == "lead_I" else list(range(12))

    patient_lead_r = {name: [] for name in LEAD_NAMES}
    patient_all_12_r = []
    patient_missing_r = []
    patient_chest_r = []
    patient_trans_r = []

    total_l1 = 0.0
    total_mse = 0.0
    total_samples = 0

    viol_III = []
    viol_aVR = []
    viol_aVL = []
    viol_aVF = []

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (tuple, list)):
                targets_raw = batch[0].to(device)
            else:
                targets_raw = batch.to(device)
            B = targets_raw.shape[0]

            source_raw = extract_source_vector(targets_raw, source_mode=source_mode)
            source_norm, targets_norm, _ = normalize_record(
                source_raw, targets_raw, mode=normalization_mode
            )

            out = model(source_norm)
            preds = out["y_pred"].float()  # [B, 12, 5000]

            r_matrix = []
            for l_idx, name in enumerate(LEAD_NAMES):
                r_l = compute_lead_pearson(preds[:, l_idx, :], targets_norm[:, l_idx, :])
                patient_lead_r[name].extend(r_l.cpu().tolist())
                r_matrix.append(r_l)

            r_stack = torch.stack(r_matrix, dim=1)  # [B, 12]
            patient_all_12_r.extend(r_stack.mean(dim=1).cpu().tolist())
            patient_missing_r.extend(r_stack[:, missing_indices].mean(dim=1).cpu().tolist())
            patient_chest_r.extend(r_stack[:, chest_indices].mean(dim=1).cpu().tolist())

            # Transition progression
            r_trans = []
            for k in range(5):
                pred_diff = preds[:, chest_indices[k + 1], :] - preds[:, chest_indices[k], :]
                tgt_diff = targets_norm[:, chest_indices[k + 1], :] - targets_norm[:, chest_indices[k], :]
                r_trans.append(compute_lead_pearson(pred_diff, tgt_diff))
            r_trans_stack = torch.stack(r_trans, dim=1)
            patient_trans_r.extend(r_trans_stack.mean(dim=1).cpu().tolist())

            # L1 & MSE
            l1_val = F.l1_loss(preds, targets_norm, reduction="sum").item()
            mse_val = F.mse_loss(preds, targets_norm, reduction="sum").item()
            total_l1 += l1_val
            total_mse += mse_val
            total_samples += B * 12 * 5000

            # Physics violations in uV approx
            p_I = preds[:, idx_I, :]
            p_II = preds[:, idx_II, :]
            p_III = preds[:, idx_III, :]
            p_aVR = preds[:, idx_aVR, :]
            p_aVL = preds[:, idx_aVL, :]
            p_aVF = preds[:, idx_aVF, :]

            viol_III.extend((torch.abs(p_II - p_I - p_III).mean(dim=-1) * 1000.0).cpu().tolist())
            viol_aVR.extend((torch.abs(p_aVR + (p_I + p_II) / 2.0).mean(dim=-1) * 1000.0).cpu().tolist())
            viol_aVL.extend((torch.abs(p_aVL - p_I + p_II / 2.0).mean(dim=-1) * 1000.0).cpu().tolist())
            viol_aVF.extend((torch.abs(p_aVF - p_II + p_I / 2.0).mean(dim=-1) * 1000.0).cpu().tolist())

    summary = {
        "checkpoint": str(checkpoint_path),
        "code_mode": code_mode,
        "source_mode": source_mode,
        "normalization": normalization_mode,
        "n_patients": len(patient_missing_r),
        "all_12_mean_pearson": float(np.mean(patient_all_12_r)),
        "missing_mean_pearson": float(np.mean(patient_missing_r)),
        "missing_pearson_p05": float(np.quantile(patient_missing_r, 0.05)),
        "chest_v1_v6_mean_pearson": float(np.mean(patient_chest_r)),
        "precordial_transition_pearson": float(np.mean(patient_trans_r)),
        "l1_error": float(total_l1 / total_samples),
        "mse_error": float(total_mse / total_samples),
        "per_lead_pearson": {k: float(np.mean(v)) for k, v in patient_lead_r.items()},
        "physical_inconsistency_uV": {
            "mean_III_violation": float(np.mean(viol_III)),
            "mean_aVR_violation": float(np.mean(viol_aVR)),
            "mean_aVL_violation": float(np.mean(viol_aVL)),
            "mean_aVF_violation": float(np.mean(viol_aVF)),
        },
    }

    patient_level_arrays = {
        "ecg_ids": np.array(val_ds.sample_ids),
        "patient_missing_r": np.array(patient_missing_r),
        "patient_all_12_r": np.array(patient_all_12_r),
        "patient_chest_r": np.array(patient_chest_r),
        "patient_trans_r": np.array(patient_trans_r),
        **{f"lead_r_{k}": np.array(v) for k, v in patient_lead_r.items()},
    }

    return summary, patient_level_arrays


def main():
    p = argparse.ArgumentParser(description="Evaluate 3DRECON-QT Reference Checkpoints")
    p.add_argument("--checkpoint", required=True, help="Path to best.pt")
    p.add_argument("--data-dir", default="data/ptb_xl/tensors", help="Path to ptb_xl tensor directory")
    p.add_argument("--out-dir", default=None, help="Output directory for eval results")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False

    summary, patient_arrays = evaluate_checkpoint(ckpt_path, Path(args.data_dir), device)

    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out_dir / "patient_level_eval.npz", **patient_arrays)

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"All-12 Mean Pearson:         {summary['all_12_mean_pearson']:.4f}")
    print(f"Missing-11 Mean Pearson:     {summary['missing_mean_pearson']:.4f} (p05: {summary['missing_pearson_p05']:.4f})")
    print(f"Chest (V1-V6) Mean Pearson:  {summary['chest_v1_v6_mean_pearson']:.4f}")
    print(f"Precordial Transition Delta: {summary['precordial_transition_pearson']:.4f}")
    print(f"Full 12-Lead L1 Loss:        {summary['l1_error']:.4f}")
    print("\nPer-Lead Pearson Correlations:")
    for lead, r_val in summary["per_lead_pearson"].items():
        print(f"  {lead:<5}: {r_val:.4f}")
    print(f"\nSaved evaluation to: {out_dir / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
