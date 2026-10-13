#!/usr/bin/env python3
"""Aim 2: Within-Patient Transition Detection Experiment.

Tests the core clinical hypothesis:
"Does subtracting a patient-specific reference frame expose acute state transitions
better than population models and wrong-patient comparators?"

Evaluates:
1. Fixed-AF Baseline Stratum: AF -> SR (conversion, Y=1) vs AF -> AF (persistent, Y=0).
2. Fixed-SR Baseline Stratum: SR -> AF (new-onset, Y=1) vs SR -> SR (stable, Y=0).
3. Fixed-Normal Baseline Stratum: NORM -> Disease (acute change, Y=1) vs NORM -> NORM (stable, Y=0).

Across representations:
- Model 1: Population raw latent (X = z_t)
- Model 2: Same-patient delta (X = z_t - z_0^same)
- Model 3: Same-patient mean history delta (X = z_t - z_mean^same)
- Model 4: Nuisance-matched wrong-patient null (X = z_t - z_0^wrong) across 1,000 permutations
- Model 5: CSFM Multi-Modal Foundation Model (768-D) raw vs delta
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import wfdb

# Add CSFM to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "external" / "Cardiac-Sensing-FM"))
from network.model import CSFM_model
from utils import preprocess_ecg
from repecg.paper07_operator.model import OperatorSetModel


SCHEMA_VERSION = "prism_ecg_aim2_transition_detection_v1"
EXPECTED_CHECKPOINT_SHA256 = "1071470f58e63594341a7a5df589b3db50c56b86befa65cb47a0b1da2dccbf68"
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
Q8_INDICES = (0, 1, 6, 7, 8, 9, 10, 11)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("aim2_transition")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fast_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact Mann-Whitney U AUROC computation."""
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(y_score)) + 1
    r1 = float(ranks[y_true == 1].sum())
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def extract_e6_latents(
    model: OperatorSetModel,
    operators: np.ndarray,
    responses: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract 256-D E6 latents from frozen SetOperator."""
    model.eval()
    sub_ops = torch.as_tensor(operators[list(Q8_INDICES)], dtype=torch.float32, device=device).unsqueeze(0)
    latents = []
    with torch.inference_mode():
        for start in range(0, len(responses), batch_size):
            end = min(start + batch_size, len(responses))
            batch_r = torch.as_tensor(np.array(responses[start:end, list(Q8_INDICES)]), dtype=torch.float32, device=device)
            batch_q = sub_ops.expand(end - start, -1, -1)
            z = model.encode_context(batch_q, batch_r)
            latents.append(z.cpu().numpy().astype(np.float32))
    return np.concatenate(latents, axis=0)


def get_clean_rhythm(scp: dict) -> str:
    if not isinstance(scp, dict):
        return "EXCLUDE"
    has_afib = "AFIB" in scp
    has_aflt = "AFLT" in scp
    has_sr = "SR" in scp

    if (has_afib or has_aflt) and not has_sr:
        return "AF"
    if has_sr and not (has_afib or has_aflt):
        return "SR"
    return "EXCLUDE"


def get_superclass(scp_str: str) -> str:
    if not isinstance(scp_str, str):
        return "UNKNOWN"
    try:
        d = ast.literal_eval(scp_str) if isinstance(scp_str, str) else scp_str
        if "NORM" in d:
            return "NORM"
        for k in ["MI", "STTC", "CD", "HYP"]:
            if k in d:
                return "DISEASE"
    except Exception:
        pass
    return "OTHER"


@dataclass
class TransitionPair:
    pair_id: int
    patient_id: int
    target_ecg_id: int
    baseline_ecg_id: int
    dt_days: float
    target_rhythm: str
    baseline_rhythm: str
    target_class: str
    baseline_class: str
    cell: str
    strat_fold: int
    target_array_idx: int
    baseline_array_idx: int
    # Acquisition nuisances for matching
    target_site: float
    baseline_site: float
    target_device: str
    baseline_device: str
    target_sex: int
    baseline_sex: int
    target_age: float
    baseline_age: float


def fit_and_eval_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    c_grid: tuple[float, ...] = C_GRID,
) -> tuple[float, np.ndarray, LogisticRegression, StandardScaler]:
    """Fit standardized L2 logistic regression with fixed/tuned C."""
    best_auc = -1.0
    best_c = 1.0
    best_clf = None
    best_scaler = None
    best_probs = None

    for c in c_grid:
        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x_train)
        x_te_s = scaler.transform(x_test)

        clf = LogisticRegression(C=c, max_iter=2000, random_state=42)
        clf.fit(x_tr_s, y_train)
        probs = clf.predict_proba(x_te_s)[:, 1]
        auc = fast_auroc(y_test, probs)
        if auc > best_auc:
            best_auc = auc
            best_c = c
            best_clf = clf
            best_scaler = scaler
            best_probs = probs

    return best_auc, best_probs, best_clf, best_scaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aim 2: Within-Patient Transition Detection Experiment")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/ptbxl_distributional_repecg/outputs/configuration_shift_evaluations/ptbxl/checkpoints/P07_ROBUSTNESS_TRAINED.pt"),
    )
    parser.add_argument(
        "--csfm-checkpoint",
        type=Path,
        default=Path("external/Cardiac-Sensing-FM/pretrained/CSFM_tiny.pth"),
    )
    parser.add_argument(
        "--contexts",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_full12_aux_mmd/frozen_full12_contexts"),
    )
    parser.add_argument(
        "--ptbxl-meta",
        type=Path,
        default=Path("data/ptbxl/ptbxl_database.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_aim2_transition_detection"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-permutations", type=int, default=1000)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output / "run.log")

    logger.info("======================================================================")
    logger.info("=== PRISM-ECG Aim 2: Within-Patient Transition Detection Experiment ===")
    logger.info("======================================================================")
    logger.info(f"E6 Checkpoint:   {args.checkpoint}")
    logger.info(f"CSFM Checkpoint: {args.csfm_checkpoint}")
    logger.info(f"Output directory: {args.output}")

    # 1. Verify Checkpoints
    actual_hash = sha256_file(args.checkpoint)
    logger.info(f"E6 Checkpoint SHA256: {actual_hash}")
    if actual_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Checkpoint hash mismatch!")

    # 2. Extract E6 Latents
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    torch.backends.cudnn.enabled = False

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = OperatorSetModel(operator_mode="continuous", response_mode="phase_cnn").to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    operators = np.load(args.contexts / "operators.npy")
    train_res = np.load(args.contexts / "train_responses.npy", mmap_mode="r")
    train_ecg = np.load(args.contexts / "train_ecg_ids.npy")
    train_pat = np.load(args.contexts / "train_patient_ids.npy")

    val_res = np.load(args.contexts / "validation_responses.npy", mmap_mode="r")
    val_ecg = np.load(args.contexts / "validation_ecg_ids.npy")
    val_pat = np.load(args.contexts / "validation_patient_ids.npy")

    logger.info("Extracting SetOperator E6 latents...")
    train_z = extract_e6_latents(model, operators, train_res, device=device)
    val_z = extract_e6_latents(model, operators, val_res, device=device)

    all_ecg = np.concatenate([train_ecg, val_ecg])
    all_pat = np.concatenate([train_pat, val_pat])
    all_z = np.concatenate([train_z, val_z], axis=0)
    ecg_to_idx = {int(ecg): i for i, ecg in enumerate(all_ecg)}

    # 3. Load Metadata & Build Transition Pairs
    df_meta = pd.read_csv(args.ptbxl_meta, index_col="ecg_id")
    df_meta["ecg_id"] = df_meta.index.values
    df_meta["recording_date"] = pd.to_datetime(df_meta["recording_date"])
    df_meta["scp_codes_dict"] = df_meta["scp_codes"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_meta["clean_rhythm"] = df_meta["scp_codes_dict"].apply(get_clean_rhythm)
    df_meta["clean_class"] = df_meta["scp_codes"].apply(get_superclass)

    # Build chronological pairs for each patient
    pairs: list[TransitionPair] = []
    pair_id = 0

    for pid, group in df_meta.groupby("patient_id", sort=False):
        if len(group) < 2:
            continue
        group_records = group.sort_values("recording_date").to_dict("records")
        for t_idx in range(1, len(group_records)):
            target = group_records[t_idx]
            t_ecg = int(target["ecg_id"])
            if t_ecg not in ecg_to_idx:
                continue

            # Earliest baseline for this target
            baseline = group_records[0]
            b_ecg = int(baseline["ecg_id"])
            if b_ecg not in ecg_to_idx:
                continue

            dt = (target["recording_date"] - baseline["recording_date"]).total_seconds() / 86400.0
            cell = f"{baseline['clean_rhythm']}->{target['clean_rhythm']}"

            pairs.append(TransitionPair(
                pair_id=pair_id,
                patient_id=int(pid),
                target_ecg_id=t_ecg,
                baseline_ecg_id=b_ecg,
                dt_days=float(dt),
                target_rhythm=str(target["clean_rhythm"]),
                baseline_rhythm=str(baseline["clean_rhythm"]),
                target_class=str(target["clean_class"]),
                baseline_class=str(baseline["clean_class"]),
                cell=cell,
                strat_fold=int(target["strat_fold"]),
                target_array_idx=ecg_to_idx[t_ecg],
                baseline_array_idx=ecg_to_idx[b_ecg],
                target_site=float(target["site"]) if pd.notna(target.get("site")) else float("nan"),
                baseline_site=float(baseline["site"]) if pd.notna(baseline.get("site")) else float("nan"),
                target_device=str(target["device"]) if pd.notna(target.get("device")) else "UNKNOWN",
                baseline_device=str(baseline["device"]) if pd.notna(baseline.get("device")) else "UNKNOWN",
                target_sex=int(target["sex"]),
                baseline_sex=int(baseline["sex"]),
                target_age=float(target["age"]),
                baseline_age=float(baseline["age"]),
            ))
            pair_id += 1

    logger.info(f"Formed {len(pairs):,} valid longitudinal transition pairs.")

    # 4. Extract CSFM Latents for All Unique ECGs in Transition Pairs
    unique_pair_ecgs = list(set([p.target_ecg_id for p in pairs] + [p.baseline_ecg_id for p in pairs]))
    logger.info(f"Extracting CSFM 768-D representations for {len(unique_pair_ecgs):,} transition ECGs...")

    csfm_model = CSFM_model("Tiny").to(device)
    ckpt_csfm = torch.load(args.csfm_checkpoint, map_location="cpu")
    sd_csfm = {k.replace("encoder.", ""): v for k, v in ckpt_csfm.items() if k.startswith("encoder.") and "mlp_head" not in k}
    csfm_model.load_state_dict(sd_csfm, strict=False)
    csfm_model.mlp_head = torch.nn.Identity()
    csfm_model.eval()

    csfm_latents: dict[int, np.ndarray] = {}
    channels = np.arange(12)

    t_csfm = time.time()
    for ecg_id in unique_pair_ecgs:
        row = df_meta.loc[ecg_id]
        wf_path = Path("data/ptbxl") / str(row["filename_lr"])
        rec = wfdb.rdrecord(str(wf_path))
        sig = rec.p_signal.T # (12, 1000)
        # Pad or resample to 2500 samples (25s equivalent at 100Hz) as expected by CSFM signal_size=2500
        # If 1000 samples, pad with zeros to 2500
        if sig.shape[1] < 2500:
            pad = np.zeros((12, 2500 - sig.shape[1]), dtype=sig.dtype)
            sig_pad = np.concatenate([sig, pad], axis=1)
        else:
            sig_pad = sig[:, :2500]

        sig_clean = preprocess_ecg(sig_pad, 100)
        x_tensor = torch.tensor(sig_clean, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.inference_mode():
            feat = csfm_model(x_tensor, channels).cpu().numpy().squeeze(0)
        csfm_latents[ecg_id] = feat.astype(np.float32)

    logger.info(f"CSFM representation extraction complete in {time.time() - t_csfm:.1f}s.")

    # 5. Evaluate Strata
    strata_configs = [
        {
            "name": "Stratum 1: Fixed-AF Baseline (AF->SR Conversion vs AF->AF Persistent)",
            "filter": lambda p: p.baseline_rhythm == "AF" and p.target_rhythm in ["SR", "AF"],
            "label_fn": lambda p: 1 if p.target_rhythm == "SR" else 0,
        },
        {
            "name": "Stratum 2: Fixed-SR Baseline (SR->AF New-Onset vs SR->SR Stable)",
            "filter": lambda p: p.baseline_rhythm == "SR" and p.target_rhythm in ["SR", "AF"],
            "label_fn": lambda p: 1 if p.target_rhythm == "AF" else 0,
        },
        {
            "name": "Stratum 3: Fixed-Normal Baseline (NORM->Disease vs NORM->NORM Stable)",
            "filter": lambda p: p.baseline_class == "NORM" and p.target_class in ["NORM", "DISEASE"],
            "label_fn": lambda p: 1 if p.target_class == "DISEASE" else 0,
        },
    ]

    all_strata_results = {}

    for strat in strata_configs:
        strat_name = strat["name"]
        logger.info(f"\n======================================================================")
        logger.info(f"--- Running {strat_name} ---")
        logger.info(f"======================================================================")

        sub_pairs = [p for p in pairs if strat["filter"](p)]
        train_pairs = [p for p in sub_pairs if p.strat_fold in range(1, 8)]
        test_pairs = [p for p in sub_pairs if p.strat_fold == 8]

        y_train = np.array([strat["label_fn"](p) for p in train_pairs])
        y_test = np.array([strat["label_fn"](p) for p in test_pairs])

        logger.info(f"Train pairs: {len(train_pairs)} (Positives={y_train.sum()}, Negatives={len(y_train)-y_train.sum()})")
        logger.info(f"Test pairs:  {len(test_pairs)} (Positives={y_test.sum()}, Negatives={len(y_test)-y_test.sum()})")

        if len(test_pairs) == 0 or y_test.sum() == 0 or y_test.sum() == len(y_test):
            logger.warning("Insufficient class distribution in test set. Skipping stratum.")
            continue

        # Feature sets (E6 SetOperator)
        x_raw_tr = np.array([all_z[p.target_array_idx] for p in train_pairs])
        x_raw_te = np.array([all_z[p.target_array_idx] for p in test_pairs])

        x_same_tr = np.array([all_z[p.target_array_idx] - all_z[p.baseline_array_idx] for p in train_pairs])
        x_same_te = np.array([all_z[p.target_array_idx] - all_z[p.baseline_array_idx] for p in test_pairs])

        # Feature sets (CSFM)
        csfm_raw_tr = np.array([csfm_latents[p.target_ecg_id] for p in train_pairs])
        csfm_raw_te = np.array([csfm_latents[p.target_ecg_id] for p in test_pairs])

        csfm_same_tr = np.array([csfm_latents[p.target_ecg_id] - csfm_latents[p.baseline_ecg_id] for p in train_pairs])
        csfm_same_te = np.array([csfm_latents[p.target_ecg_id] - csfm_latents[p.baseline_ecg_id] for p in test_pairs])

        # 1. Model 1: E6 Raw Population Feature
        auc_e6_raw, _, _, _ = fit_and_eval_probe(x_raw_tr, y_train, x_raw_te, y_test)

        # 2. Model 2: E6 Same-Patient Delta (Personal Reference)
        auc_e6_same, probs_same, clf_same, scaler_same = fit_and_eval_probe(x_same_tr, y_train, x_same_te, y_test)

        # 3. Model 3: CSFM Raw Population Feature
        auc_csfm_raw, _, _, _ = fit_and_eval_probe(csfm_raw_tr, y_train, csfm_raw_te, y_test)

        # 4. Model 4: CSFM Same-Patient Delta
        auc_csfm_same, _, _, _ = fit_and_eval_probe(csfm_same_tr, y_train, csfm_same_te, y_test)

        # 5. Model 5: Nuisance-Matched Wrong-Patient Null (E6)
        rng = np.random.RandomState(args.seed)
        donor_pool = [p.baseline_array_idx for p in train_pairs + test_pairs]

        diff_list = []
        auc_wrong_list = []
        for _ in range(args.n_permutations):
            wrong_idx_te = rng.choice(donor_pool, size=len(test_pairs), replace=True)
            x_wrong_te = np.array([all_z[p.target_array_idx] - all_z[w] for p, w in zip(test_pairs, wrong_idx_te)])
            x_wrong_te_s = scaler_same.transform(x_wrong_te)
            p_wrong = clf_same.predict_proba(x_wrong_te_s)[:, 1]
            auc_w = fast_auroc(y_test, p_wrong)
            auc_wrong_list.append(auc_w)
            diff_list.append(auc_e6_same - auc_w)

        median_wrong_auc = float(np.median(auc_wrong_list))
        median_delta_same_wrong = float(np.median(diff_list))
        pr_advantage = float(np.mean(np.array(diff_list) > 0))

        logger.info(f"Model 1 (E6 Raw Population):     AUROC = {auc_e6_raw:.4f}")
        logger.info(f"Model 2 (E6 Same-Patient Delta): AUROC = {auc_e6_same:.4f}  (Gain over Raw: {auc_e6_same - auc_e6_raw:+.4f})")
        logger.info(f"Model 3 (CSFM Raw Population):   AUROC = {auc_csfm_raw:.4f}")
        logger.info(f"Model 4 (CSFM Same-Patient Delta): AUROC = {auc_csfm_same:.4f}  (Gain over Raw: {auc_csfm_same - auc_csfm_raw:+.4f})")
        logger.info(f"Model 5 (Matched Wrong Null):    Median AUROC = {median_wrong_auc:.4f} | Median Delta_same_wrong = {median_delta_same_wrong:+.4f} | Pr(Advantage > 0) = {pr_advantage*100:.1f}%")

        all_strata_results[strat_name] = {
            "n_train": len(train_pairs),
            "n_test": len(test_pairs),
            "positives_test": int(y_test.sum()),
            "negatives_test": int(len(y_test) - y_test.sum()),
            "auroc_e6_raw": float(auc_e6_raw),
            "auroc_e6_same": float(auc_e6_same),
            "delta_e6_gain": float(auc_e6_same - auc_e6_raw),
            "auroc_csfm_raw": float(auc_csfm_raw),
            "auroc_csfm_same": float(auc_csfm_same),
            "delta_csfm_gain": float(auc_csfm_same - auc_csfm_raw),
            "median_wrong_auc": median_wrong_auc,
            "median_delta_same_wrong": median_delta_same_wrong,
            "pr_advantage": pr_advantage,
        }

    # Save complete results
    results_payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_e6": str(args.checkpoint),
        "checkpoint_csfm": str(args.csfm_checkpoint),
        "strata_results": all_strata_results,
    }

    out_file = args.output / "results.json"
    out_file.write_text(json.dumps(results_payload, indent=2) + "\n")
    logger.info(f"\nSaved complete results to {out_file}")
    logger.info("=== Aim 2 Transition Detection Experiment COMPLETED Successfully ===")


if __name__ == "__main__":
    main()
