#!/usr/bin/env python3
"""Evaluate all 360 RDB test records using conv15e_A0_raw and conv15e_R7,
compute full missing-lead metrics, rank records within each rhythm class,
and extract Best, Median, and Worst cases for the visual gallery.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_latents.engineering.utils.common import mask_unobserved_leads
from unified_latents.engineering.utils.regimes import make_lead_indices
from unified_latents.engineering.experimental.wavelet_ssl_ecg_aim import build_wavelet_ecg_aim

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
MISSING_LEAD_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # all except Lead I (0)


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = argparse.Namespace(**payload["config"])
    model = build_wavelet_ecg_aim(
        target_len=5000,
        patch_size=cfg.patch_size,
        width=cfg.width,
        encoder_depth=cfg.encoder_depth,
        decoder_depth=cfg.decoder_depth,
        heads=cfg.heads,
        random_mask_ratio=cfg.random_mask_ratio,
        temporal_mask_ratio=cfg.temporal_mask_ratio,
        consistency_weight=cfg.consistency_weight,
        lead_conditioning_mode=cfg.lead_conditioning_mode,
        use_relative_geometry=cfg.use_relative_geometry,
        use_spatial_film=cfg.use_spatial_film,
        spatial_gain_init=cfg.spatial_gain_init,
        geometry_control=cfg.geometry_control,
        use_wavelet_branch=cfg.use_wavelet_branch,
        wavelet_bank=cfg.wavelet_bank,
        custom_wavelet_asset=cfg.custom_wavelet_asset,
        view_a_bank=cfg.view_a_bank,
        view_b_bank=cfg.view_b_bank,
        view_a_custom_wavelet_asset=cfg.view_a_custom_wavelet_asset,
        view_b_custom_wavelet_asset=cfg.view_b_custom_wavelet_asset,
        n_scales=cfg.n_scales,
        min_freq_hz=cfg.min_freq_hz,
        max_freq_hz=cfg.max_freq_hz,
        morlet_cycles=cfg.morlet_cycles,
        view_a=cfg.view_a,
        view_b=cfg.view_b,
        wavelet_encoder=cfg.wavelet_encoder,
        wavelet_dim=cfg.wavelet_dim,
        wavelet_depth=cfg.wavelet_depth,
        wavelet_heads=cfg.wavelet_heads,
        wavelet_conv_hidden=cfg.wavelet_conv_hidden,
        wavelet_fusion=cfg.wavelet_fusion,
        fusion_heads=cfg.fusion_heads,
        inference_view=cfg.inference_view,
        ssl_mode=cfg.ssl_mode,
        ssl_projector_hidden=cfg.ssl_projector_hidden,
        ssl_projector_dim=cfg.ssl_projector_dim,
        ssl_predictor_hidden=cfg.ssl_predictor_hidden,
        byol_tau=cfg.byol_tau,
        use_delineation_head=not cfg.no_delineation_head,
        delineation_hidden=cfg.delineation_hidden,
        delineation_kernel=cfg.delineation_kernel,
        predict_fiducials=not cfg.no_fiducial_head,
        mask_type_mode=cfg.mask_type_mode,
    )
    state = {k.removeprefix("_orig_mod."): v for k, v in payload["model_state_dict"].items()}
    state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def reconstruct_batch(
    model: torch.nn.Module,
    targets: torch.Tensor,
    observed_lead: int = 0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    obs_list = [observed_lead]
    targets = targets.float().to(device)
    masked = mask_unobserved_leads(targets, obs_list).contiguous()
    indices = make_lead_indices(obs_list, targets.shape[0], device)
    with torch.inference_mode():
        result = model(
            masked,
            y_full=targets,
            lead_indices=indices,
            compute_delineation=False,
            compute_ssl=False,
        )["y_pred"].float()
        result = result[:, :12, :5000].clone()
        result[:, observed_lead, :] = targets[:, observed_lead, :]
        return result.cpu()


def compute_metrics(true_12: np.ndarray, pred_12: np.ndarray) -> dict[str, float]:
    """Compute per-lead Pearson r and aggregate metrics on missing leads."""
    lead_rs = {}
    lead_rmses = {}
    missing_rs = []
    missing_rmses = []

    for idx, name in enumerate(LEAD_NAMES):
        t = true_12[idx]
        p = pred_12[idx]
        # Pearson r
        var_t = np.var(t)
        var_p = np.var(p)
        if var_t > 1e-12 and var_p > 1e-12:
            r = float(np.corrcoef(t, p)[0, 1])
            if np.isnan(r):
                r = 0.0
        else:
            r = 0.0
        lead_rs[f"r_{name}"] = r

        rmse = float(np.sqrt(np.mean((t - p) ** 2)))
        lead_rmses[f"rmse_{name}"] = rmse

        if idx in MISSING_LEAD_INDICES:
            missing_rs.append(r)
            missing_rmses.append(rmse)

    mean_r = float(np.mean(missing_rs))
    mean_rmse = float(np.mean(missing_rmses))

    # Lead groups
    limb_missing = [1, 2, 3, 4, 5]  # II, III, aVR, aVL, aVF
    precordial = [6, 7, 8, 9, 10, 11]  # V1-V6

    r_limb = float(np.mean([lead_rs[f"r_{LEAD_NAMES[i]}"] for i in limb_missing]))
    r_precordial = float(np.mean([lead_rs[f"r_{LEAD_NAMES[i]}"] for i in precordial]))

    res = {
        "mean_missing_r": mean_r,
        "mean_missing_rmse": mean_rmse,
        "r_limb_missing": r_limb,
        "r_precordial": r_precordial,
    }
    res.update(lead_rs)
    res.update(lead_rmses)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RDB test cohort and select gallery cases")
    parser.add_argument(
        "--ckpt-a0",
        type=Path,
        default=ROOT / "refine-logs/convergence_10e/runs/conv15e_A0_raw_s42_l0/best.pt",
    )
    parser.add_argument(
        "--ckpt-r7",
        type=Path,
        default=ROOT / "refine-logs/convergence_10e/runs/conv15e_R7_morlet_mag_ueg_real_s42_l0/best.pt",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/rdb_wavelet_delineation_cache",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/rdb_gallery",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[Gallery Eval] Using device: {device}")

    # 1. Load manifest and filter test records
    manifest_file = args.cache_dir / "manifest.json"
    with open(manifest_file) as f:
        manifest = json.load(f)

    test_records = [r for r in manifest["records"] if r.get("split") == "test"]
    print(f"[Gallery Eval] Found {len(test_records)} test records in RDB cache.")

    # 2. Load models
    print(f"[Gallery Eval] Loading conv15e_A0_raw from {args.ckpt_a0}...")
    model_a0 = load_model(args.ckpt_a0, device)

    model_r7 = None
    if args.ckpt_r7.exists():
        print(f"[Gallery Eval] Loading conv15e_R7 from {args.ckpt_r7}...")
        model_r7 = load_model(args.ckpt_r7, device)

    # 3. Load all payloads
    print("[Gallery Eval] Loading test payloads into memory...")
    records_meta = []
    waveforms_true = []
    segmentations = []
    for r in test_records:
        file_path = args.cache_dir / r["output"]
        payload = torch.load(file_path, map_location="cpu", weights_only=True)
        records_meta.append({
            "record_id": r["record_id"],
            "patient_id": r["patient_id"],
            "canonical_rhythm": r["canonical_rhythm"],
            "released_rhythm": r.get("released_rhythm", r["canonical_rhythm"]),
            "path": str(file_path),
        })
        waveforms_true.append(payload["waveform"].float().numpy())
        segmentations.append(payload["segmentation"].numpy())

    waveforms_true = np.stack(waveforms_true)  # (N, 12, 5000)
    print(f"[Gallery Eval] Loaded {len(waveforms_true)} ground-truth waveforms.")

    # 4. Batch reconstruction
    print("[Gallery Eval] Running reconstruction with conv15e_A0_raw...")
    recons_a0 = []
    tensor_true = torch.from_numpy(waveforms_true)
    for offset in range(0, len(tensor_true), args.batch_size):
        batch = tensor_true[offset : offset + args.batch_size]
        pred = reconstruct_batch(model_a0, batch, observed_lead=0, device=device)
        recons_a0.append(pred.numpy())
    recons_a0 = np.concatenate(recons_a0, axis=0)  # (N, 12, 5000)

    recons_r7 = None
    if model_r7 is not None:
        print("[Gallery Eval] Running reconstruction with conv15e_R7...")
        recons_r7 = []
        for offset in range(0, len(tensor_true), args.batch_size):
            batch = tensor_true[offset : offset + args.batch_size]
            pred = reconstruct_batch(model_r7, batch, observed_lead=0, device=device)
            recons_r7.append(pred.numpy())
        recons_r7 = np.concatenate(recons_r7, axis=0)

    # 5. Compute metrics per record
    print("[Gallery Eval] Computing metrics across all records...")
    eval_rows = []
    for i, meta in enumerate(records_meta):
        row = dict(meta)
        m_a0 = compute_metrics(waveforms_true[i], recons_a0[i])
        for k, v in m_a0.items():
            row[f"a0_{k}"] = v

        if recons_r7 is not None:
            m_r7 = compute_metrics(waveforms_true[i], recons_r7[i])
            for k, v in m_r7.items():
                row[f"r7_{k}"] = v
            row["delta_r_r7_minus_a0"] = m_r7["mean_missing_r"] - m_a0["mean_missing_r"]

        eval_rows.append(row)

    df = pd.DataFrame(eval_rows)
    csv_path = args.out_dir / "rdb_all_360_records_eval.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Gallery Eval] Saved all-records evaluation to {csv_path}")

    # Summary by rhythm
    print("\n=== PERFORMANCE BY CANONICAL RHYTHM (conv15e_A0_raw) ===")
    rhythm_summary = df.groupby("canonical_rhythm")["a0_mean_missing_r"].agg(["count", "mean", "std", "min", "median", "max"])
    print(rhythm_summary.to_string())

    # 6. Stratified Case Selection: Best, Median, Worst per rhythm
    selected_cases = []
    rhythms = sorted(df["canonical_rhythm"].unique())
    selected_indices = {}

    for rhy in rhythms:
        sub = df[df["canonical_rhythm"] == rhy].copy()
        sub = sub.sort_values(by="a0_mean_missing_r", ascending=True)

        n = len(sub)
        idx_worst = sub.index[0]
        idx_med = sub.index[n // 2]
        idx_best = sub.index[-1]

        cases = [
            ("worst", idx_worst, sub.loc[idx_worst]),
            ("median", idx_med, sub.loc[idx_med]),
            ("best", idx_best, sub.loc[idx_best]),
        ]

        for tier, global_idx, row_data in cases:
            case_info = {
                "rhythm": rhy,
                "tier": tier,
                "record_id": row_data["record_id"],
                "patient_id": row_data["patient_id"],
                "global_index": int(global_idx),
                "a0_mean_missing_r": float(row_data["a0_mean_missing_r"]),
                "a0_mean_missing_rmse": float(row_data["a0_mean_missing_rmse"]),
                "a0_r_limb_missing": float(row_data["a0_r_limb_missing"]),
                "a0_r_precordial": float(row_data["a0_r_precordial"]),
                "a0_r_II": float(row_data["a0_r_II"]),
                "a0_r_III": float(row_data["a0_r_III"]),
                "a0_r_aVF": float(row_data["a0_r_aVF"]),
                "a0_r_V1": float(row_data["a0_r_V1"]),
                "a0_r_V5": float(row_data["a0_r_V5"]),
            }
            if recons_r7 is not None:
                case_info["r7_mean_missing_r"] = float(row_data["r7_mean_missing_r"])
                case_info["delta_r"] = float(row_data["delta_r_r7_minus_a0"])
            selected_cases.append(case_info)
            selected_indices[f"{rhy}_{tier}"] = int(global_idx)

    json_path = args.out_dir / "rdb_gallery_selected_cases.json"
    with open(json_path, "w") as f:
        json.dump(selected_cases, f, indent=2)
    print(f"[Gallery Eval] Saved 24 selected cases metadata to {json_path}")

    # 7. Save waveforms for selected cases to NPZ for rapid plotting
    npz_data = {
        "selected_indices": json.dumps(selected_indices),
        "lead_names": LEAD_NAMES,
    }
    for key, g_idx in selected_indices.items():
        npz_data[f"{key}_true"] = waveforms_true[g_idx]
        npz_data[f"{key}_a0"] = recons_a0[g_idx]
        if recons_r7 is not None:
            npz_data[f"{key}_r7"] = recons_r7[g_idx]
        npz_data[f"{key}_seg"] = segmentations[g_idx]

    npz_path = args.out_dir / "rdb_gallery_selected_waveforms.npz"
    np.savez_compressed(npz_path, **npz_data)
    print(f"[Gallery Eval] Saved compressed waveforms for plotting to {npz_path}")
    print("[Gallery Eval] Successfully completed Phase 1!")


if __name__ == "__main__":
    main()
