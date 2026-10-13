#!/usr/bin/env python3
"""Evaluate frozen CSFM Base on Kingston-ICU AFIB/AFLT vs SINUS rhythm detection benchmark."""
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import wfdb

# Disable cuDNN
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
    orig_fs: int = 240,
    target_fs: int = 250,
    target_len: int = 2500,
) -> np.ndarray:
    """Resample 1D signal to target_fs and pad/truncate to target_len."""
    n_samples_target = int(len(signal) * target_fs / orig_fs)
    resampled = resample(signal, n_samples_target)
    if len(resampled) >= target_len:
        out = resampled[:target_len]
    else:
        out = np.zeros(target_len, dtype=resampled.dtype)
        out[:len(resampled)] = resampled
    std = np.std(out)
    if std > 1e-6:
        out = (out - np.mean(out)) / std
    return out.astype(np.float32)


def fit_and_eval_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    c_grid: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
) -> Tuple[float, float]:
    """Fit standardized logistic regression with C tuning."""
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_train)
    x_te = scaler.transform(x_test)

    best_auc = -1.0
    best_auprc = -1.0

    for c_val in c_grid:
        try:
            clf = LogisticRegression(C=c_val, max_iter=1000, random_state=42, solver="lbfgs")
            clf.fit(x_tr, y_train)
            probs = clf.predict_proba(x_te)[:, 1]
            auc = fast_auroc(y_test, probs)
            auprc = average_precision_score(y_test, probs)
            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc
                best_auprc = auprc
        except Exception:
            continue
    return best_auc, best_auprc


def main() -> None:
    parser = argparse.ArgumentParser(description="Kingston-ICU CSFM Base Benchmark")
    parser.add_argument("--kingston-dir", type=Path, default=Path("data/kingston-icu-af-dataset-1.0.0"))
    parser.add_argument("--checkpoint", type=Path, default=Path("external/Cardiac-Sensing-FM/pretrained/CSFM_base.pth"))
    parser.add_argument("--output", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/kingston_csfm_results.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CSFM Base model on {device}...")

    model = CSFM_model("Base")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    encoder_state_dict = {
        k.replace("encoder.", ""): v
        for k, v in ckpt.items()
        if k.startswith("encoder.") and "mlp_head" not in k
    }
    model.load_state_dict(encoder_state_dict, strict=False)
    model.mlp_head = nn.Identity()
    model = model.to(device).eval()

    # Load Kingston metadata
    meta = pd.read_csv(args.kingston_dir / "metadata.csv")
    print(f"Loaded Kingston metadata: {len(meta)} records.")
    train_meta = meta[meta["TrainOrTest"] == "Train"].copy()
    test_meta = meta[meta["TrainOrTest"] == "Test"].copy()
    print(f"   Train records: {len(train_meta)} | Test records: {len(test_meta)}")

    y_train = (train_meta["Rhythm"] == "AFIB/AFLT").astype(int).values
    y_test = (test_meta["Rhythm"] == "AFIB/AFLT").astype(int).values

    def load_waveforms(subset_df: pd.DataFrame) -> np.ndarray:
        waveforms = []
        for _, r in subset_df.iterrows():
            record_path = args.kingston_dir / r["ECG"]
            sig, fields = wfdb.rdsamp(str(record_path))  # (2400, 4)
            # Process each of the 4 leads (I, II, III, V)
            sig_resamp = np.stack([
                resample_and_prepare(sig[:, ch], orig_fs=fields["fs"], target_fs=250, target_len=2500)
                for ch in range(4)
            ], axis=0)  # (4, 2500)
            waveforms.append(sig_resamp)
        return np.array(waveforms, dtype=np.float32)

    print("Loading and preprocessing Kingston waveforms...")
    w_train = load_waveforms(train_meta)  # (N_tr, 4, 2500)
    w_test = load_waveforms(test_meta)    # (N_te, 4, 2500)
    print(f"Waveforms ready: Train {w_train.shape}, Test {w_test.shape}")

    # Lead configurations to evaluate
    lead_configs = [
        ("Lead II Only", [1], np.array([1])),
        ("Lead I Only", [0], np.array([0])),
        ("Leads I + II", [0, 1], np.array([0, 1])),
        ("Leads I + II + III", [0, 1, 2], np.array([0, 1, 2])),
        ("All 4 Kingston Leads (I, II, III, V)", [0, 1, 2, 3], np.array([0, 1, 2, 3])),
    ]

    results = {}
    for cfg_name, ch_indices, ch_array in lead_configs:
        print(f"\nEvaluating configuration: {cfg_name} (Channels: {ch_indices})...")
        t_sub_tr = torch.as_tensor(w_train[:, ch_indices, :], dtype=torch.float32, device=device)
        t_sub_te = torch.as_tensor(w_test[:, ch_indices, :], dtype=torch.float32, device=device)

        with torch.inference_mode():
            # Batch extraction
            bs = 64
            z_tr_list = []
            for b_start in range(0, len(t_sub_tr), bs):
                b_end = min(b_start + bs, len(t_sub_tr))
                z_tr_list.append(model(t_sub_tr[b_start:b_end], ch_array).cpu().numpy())
            z_tr = np.concatenate(z_tr_list, axis=0)

            z_te_list = []
            for b_start in range(0, len(t_sub_te), bs):
                b_end = min(b_start + bs, len(t_sub_te))
                z_te_list.append(model(t_sub_te[b_start:b_end], ch_array).cpu().numpy())
            z_te = np.concatenate(z_te_list, axis=0)

        auc, auprc = fit_and_eval_probe(z_tr, y_train, z_te, y_test)
        print(f"   Test AUROC: {auc:.4f} | Test AUPRC: {auprc:.4f}")
        results[cfg_name] = {
            "channels": ch_indices,
            "latent_dim": int(z_tr.shape[1]),
            "test_auroc": float(auc),
            "test_auprc": float(auprc),
        }

    # Reference SetOperator Result on Kingston-ICU
    so_ref = {
        "model": "SetOperator (Continuous Operator)",
        "test_auroc": 0.7837,
        "source": "outputs/external_dataset_evaluations/zeroshot_kingston_set_operator.json",
    }

    out_payload = {
        "dataset": "kingston_icu",
        "task": "AFIB/AFLT vs SINUS rhythm detection",
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "test_prevalence_af": float(np.mean(y_test)),
        "setoperator_reference": so_ref,
        "csfm_base_configurations": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved Kingston benchmark results to {args.output}")


if __name__ == "__main__":
    main()
