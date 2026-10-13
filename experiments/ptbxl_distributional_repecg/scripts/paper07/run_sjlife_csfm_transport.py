#!/usr/bin/env python3
"""Evaluate frozen CSFM Base on SJLIFE Apple Watch-to-Clinical cross-device transport matching."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import resample
import torch
import torch.nn as nn

# Disable cuDNN to prevent PyTorch 2.6 / A100 driver bug
torch.backends.cudnn.enabled = False

REPO_DIR = Path(__file__).resolve().parents[4]
CSFM_REPO = REPO_DIR / "external" / "Cardiac-Sensing-FM"
if str(CSFM_REPO) not in sys.path:
    sys.path.insert(0, str(CSFM_REPO))

from network.model import CSFM_model


def fast_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact Mann-Whitney U AUROC computation."""
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(y_score)) + 1
    r1 = float(ranks[y_true == 1].sum())
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def resample_and_prepare(
    signal: np.ndarray,
    orig_fs: int,
    target_fs: int = 250,
    target_len: int = 5000,
) -> np.ndarray:
    """Resample 1D signal to target_fs and pad/truncate to target_len."""
    # signal: (T,)
    n_samples_target = int(len(signal) * target_fs / orig_fs)
    resampled = resample(signal, n_samples_target)
    if len(resampled) >= target_len:
        out = resampled[:target_len]
    else:
        out = np.zeros(target_len, dtype=resampled.dtype)
        out[:len(resampled)] = resampled
    # Standardize
    std = np.std(out)
    if std > 1e-6:
        out = (out - np.mean(out)) / std
    return out.astype(np.float32)


def evaluate_retrieval(
    z_watch: np.ndarray,
    z_clin: np.ndarray,
    patient_ids: List[str],
) -> Dict[str, Any]:
    """Compute verification AUROC and retrieval ranks between watch and clinical representations."""
    N = len(patient_ids)
    # Euclidean distance matrix
    diff = z_watch[:, None, :] - z_clin[None, :, :]
    dist_euc = np.linalg.norm(diff, axis=-1)  # (N, N)

    # Cosine distance matrix
    w_norm = z_watch / (np.linalg.norm(z_watch, axis=-1, keepdims=True) + 1e-12)
    c_norm = z_clin / (np.linalg.norm(z_clin, axis=-1, keepdims=True) + 1e-12)
    sim_cos = np.matmul(w_norm, c_norm.T)  # (N, N)
    dist_cos = 1.0 - sim_cos

    same_mask = np.eye(N, dtype=bool)
    diff_mask = ~same_mask

    same_euc = dist_euc[same_mask]
    diff_euc = dist_euc[diff_mask]

    same_cos = sim_cos[same_mask]
    diff_cos_vals = sim_cos[diff_mask]

    # Verification AUROC
    y_verif = np.concatenate([np.ones(N), np.zeros(len(diff_euc))])
    auc_euc = fast_auroc(y_verif, np.concatenate([-same_euc, -diff_euc]))
    auc_cos = fast_auroc(y_verif, np.concatenate([same_cos, diff_cos_vals]))

    # Retrieval: for each watch record, rank all clinical candidates by euclidean distance
    ranks_euc = []
    ranks_cos = []
    for i in range(N):
        order_euc = np.argsort(dist_euc[i])
        rank_euc = int(np.flatnonzero(order_euc == i)[0]) + 1
        ranks_euc.append(rank_euc)

        order_cos = np.argsort(dist_cos[i])
        rank_cos = int(np.flatnonzero(order_cos == i)[0]) + 1
        ranks_cos.append(rank_cos)

    ranks_euc = np.array(ranks_euc)
    ranks_cos = np.array(ranks_cos)

    return {
        "n_patients": N,
        "verification_auroc_euclidean": float(auc_euc),
        "verification_auroc_cosine": float(auc_cos),
        "same_distance_euclidean": {
            "mean": float(same_euc.mean()),
            "median": float(np.median(same_euc)),
        },
        "different_distance_euclidean": {
            "mean": float(diff_euc.mean()),
            "median": float(np.median(diff_euc)),
        },
        "retrieval_euclidean": {
            "top1": float(np.mean(ranks_euc <= 1)),
            "top5": float(np.mean(ranks_euc <= 5)),
            "top10": float(np.mean(ranks_euc <= 10)),
            "median_rank": float(np.median(ranks_euc)),
        },
        "retrieval_cosine": {
            "top1": float(np.mean(ranks_cos <= 1)),
            "top5": float(np.mean(ranks_cos <= 5)),
            "top10": float(np.mean(ranks_cos <= 10)),
            "median_rank": float(np.median(ranks_cos)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SJLIFE CSFM Base Cross-Device Evaluation")
    parser.add_argument("--data", type=Path, default=Path("/data/mithunmanivannan/datasets/sjlife/repository"))
    parser.add_argument("--checkpoint", type=Path, default=Path("external/Cardiac-Sensing-FM/pretrained/CSFM_base.pth"))
    parser.add_argument("--output", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/sjlife_csfm_results.json"))
    args = parser.parse_args()

    table = pd.read_csv(args.data / "shared_paired_data_243.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CSFM Base checkpoint from {args.checkpoint} on {device}...")

    model = CSFM_model("Base")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    encoder_state_dict = {
        k.replace("encoder.", ""): v
        for k, v in ckpt.items()
        if k.startswith("encoder.") and "mlp_head" not in k
    }
    model.load_state_dict(encoder_state_dict, strict=False)
    model.mlp_head = nn.Identity()
    model = model.to(device)
    model.eval()

    watch_embs = []
    clin_lead1_embs = []
    clin_12lead_embs = []
    admitted_ids = []

    print(f"Extracting CSFM embeddings for {len(table)} SJLIFE paired records...")
    t0 = time.time()

    for idx, row in table.iterrows():
        ap = args.data / row["apple_loc_ECG_243"]
        cp = args.data / "ClinicalECGs_full_243" / Path(row["clinical_loc_ECG_243"]).name.replace("apple_", "clinical_")
        if not ap.exists() or not cp.exists():
            continue

        raw_watch = np.load(ap).astype(np.float64) / 1000.0  # uV to mV
        raw_clin = np.load(cp).astype(np.float64) / 1000.0   # (1, 12, 5000) or (12, 5000)
        if raw_clin.ndim == 3:
            raw_clin = raw_clin[0]

        # 1. Watch Lead I: (1, 1, 2500)
        w_resamp = resample_and_prepare(raw_watch, orig_fs=512, target_fs=250, target_len=2500)
        w_tensor = torch.as_tensor(w_resamp[None, None, :], dtype=torch.float32, device=device)

        # 2. Clinical Lead I: (1, 1, 2500)
        c_l1_resamp = resample_and_prepare(raw_clin[0], orig_fs=500, target_fs=250, target_len=2500)
        c_l1_tensor = torch.as_tensor(c_l1_resamp[None, None, :], dtype=torch.float32, device=device)

        # 3. Clinical 12-Lead: (1, 12, 2500)
        c_12l_resamp = np.stack([
            resample_and_prepare(raw_clin[ch], orig_fs=500, target_fs=250, target_len=2500)
            for ch in range(12)
        ], axis=0)
        c_12l_tensor = torch.as_tensor(c_12l_resamp[None, :, :], dtype=torch.float32, device=device)

        with torch.inference_mode():
            z_w = model(w_tensor, np.array([0])).cpu().numpy()[0]
            z_c_l1 = model(c_l1_tensor, np.array([0])).cpu().numpy()[0]
            z_c_12l = model(c_12l_tensor, np.arange(12)).cpu().numpy()[0]

        watch_embs.append(z_w)
        clin_lead1_embs.append(z_c_l1)
        clin_12lead_embs.append(z_c_12l)
        admitted_ids.append(str(row["sudoID"]))

    z_watch_arr = np.array(watch_embs, dtype=np.float32)
    z_clin_l1_arr = np.array(clin_lead1_embs, dtype=np.float32)
    z_clin_12l_arr = np.array(clin_12lead_embs, dtype=np.float32)

    print(f"Extraction complete in {time.time() - t0:.2f}s for {len(admitted_ids)} patients.")

    # Evaluate Configuration 1: Lead I -> Lead I (Hardware transport, matched sensing lead)
    res_lead1_to_lead1 = evaluate_retrieval(z_watch_arr, z_clin_l1_arr, admitted_ids)
    print("\n--- CSFM Base: Apple Watch Lead I -> Clinical Lead I ---")
    print(f"Verification AUROC (Euclidean): {res_lead1_to_lead1['verification_auroc_euclidean']:.4f}")
    print(f"Verification AUROC (Cosine):    {res_lead1_to_lead1['verification_auroc_cosine']:.4f}")
    print(f"Top-1 Retrieval (Euclidean):     {res_lead1_to_lead1['retrieval_euclidean']['top1']*100:.2f}%")
    print(f"Top-5 Retrieval (Euclidean):     {res_lead1_to_lead1['retrieval_euclidean']['top5']*100:.2f}%")
    print(f"Median Rank:                    {res_lead1_to_lead1['retrieval_euclidean']['median_rank']:.1f}")

    # Evaluate Configuration 2: Lead I -> Full 12-Lead (Cross-hardware & cross-lead transport)
    res_lead1_to_12lead = evaluate_retrieval(z_watch_arr, z_clin_12l_arr, admitted_ids)
    print("\n--- CSFM Base: Apple Watch Lead I -> Clinical 12-Lead ---")
    print(f"Verification AUROC (Euclidean): {res_lead1_to_12lead['verification_auroc_euclidean']:.4f}")
    print(f"Verification AUROC (Cosine):    {res_lead1_to_12lead['verification_auroc_cosine']:.4f}")
    print(f"Top-1 Retrieval (Euclidean):     {res_lead1_to_12lead['retrieval_euclidean']['top1']*100:.2f}%")
    print(f"Top-5 Retrieval (Euclidean):     {res_lead1_to_12lead['retrieval_euclidean']['top5']*100:.2f}%")
    print(f"Median Rank:                    {res_lead1_to_12lead['retrieval_euclidean']['median_rank']:.1f}")

    output_payload = {
        "schema_version": "sjlife_csfm_transport_v1",
        "checkpoint": str(args.checkpoint),
        "admitted_patients": len(admitted_ids),
        "lead1_to_lead1": res_lead1_to_lead1,
        "lead1_to_12lead": res_lead1_to_12lead,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved CSFM SJLIFE results to {args.output}")


if __name__ == "__main__":
    main()
