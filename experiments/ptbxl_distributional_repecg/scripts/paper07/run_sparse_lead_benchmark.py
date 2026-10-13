#!/usr/bin/env python3
"""Run Milestone R014 / Block 2: Sparse-Lead Representation Resilience Benchmark.

Systematic evaluation of representation geometry preservation as measurement burden drops:
Configurations:
1. 12L (Full standard diagnostic context)
2. Lead II (Standard wearable single-lead)
3. Lead I + II (Frontal limb vector plane)
4. Lead I + II + V1 (3-lead rhythm + right precordial)
5. Lead I + II + V5 (3-lead limb + lateral precordial)

Evaluates:
- SetOperator (continuous measurement functional operator sets) vs CSFM Base (channel index tokens)
- Native representations (SO d=256 vs CSFM d=768)
- Dimensionality-matched representations via PCA (d=128 and d=256, fitted strictly on training patients)
- Verification AUROC (Euclidean & Cosine), Top-1/5/10 retrieval, median rank, R^2 decomposition,
  cross-disease verification, and retention percentage relative to 12L.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import wfdb

# Set cuDNN disabled to avoid conv descriptor bug on this platform
torch.backends.cudnn.enabled = False

# Add external CSFM directory to path
CSFM_REPO = Path(__file__).resolve().parents[4] / "external" / "Cardiac-Sensing-FM"
if str(CSFM_REPO) not in sys.path:
    sys.path.insert(0, str(CSFM_REPO))

from network.model import CSFM_model
from utils.preprocess import preprocess_ecg
from repecg.paper07_operator.model import OperatorSetModel


SCHEMA_VERSION = "prism_ecg_sparse_lead_resilience_v1"

LEAD_CONFIGS = [
    {
        "name": "Lead II",
        "description": "Standard wearable single-lead",
        "so_indices": [1],
        "csfm_channels": [1],
    },
    {
        "name": "Lead I + II",
        "description": "Frontal plane limb leads (Einthoven)",
        "so_indices": [0, 1],
        "csfm_channels": [0, 1],
    },
    {
        "name": "Lead I + II + V1",
        "description": "3-lead limb + right precordial (atrial rhythm / RV)",
        "so_indices": [0, 1, 6],
        "csfm_channels": [0, 1, 6],
    },
    {
        "name": "Lead I + II + V5",
        "description": "3-lead limb + lateral precordial (LV repolarization)",
        "so_indices": [0, 1, 10],
        "csfm_channels": [0, 1, 10],
    },
    {
        "name": "12L",
        "description": "Full standard 12-lead diagnostic context",
        "so_indices": [0, 1, 6, 7, 8, 9, 10, 11],  # 8 independent basis leads
        "csfm_channels": list(range(12)),          # all 12 channels
    },
]


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("sparse_lead_benchmark")
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


def fast_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact Mann-Whitney U AUROC computation."""
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(y_score)) + 1
    r1 = float(ranks[y_true == 1].sum())
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def preprocess_single_file(args_tuple: Tuple[str, str]) -> np.ndarray:
    """Preprocess one raw ECG recording for CSFM input."""
    base_dir, rel_path = args_tuple
    full_path = os.path.join(base_dir, rel_path)
    sig, meta = wfdb.rdsamp(full_path)
    ecg_clean = preprocess_ecg(sig.T, fs=meta["fs"])
    return ecg_clean.astype(np.float32)


def get_preprocessed_waveforms(
    df_multi: pd.DataFrame,
    ptbxl_data_dir: Path,
    cache_path: Path,
    num_workers: int = 8,
    logger: logging.Logger = None,
) -> np.ndarray:
    """Load or preprocess all 12-lead waveforms (4,087, 12, 2500) once."""
    if cache_path.exists():
        if logger:
            logger.info(f"Loading cached preprocessed waveforms from {cache_path}...")
        return np.load(cache_path)["waveforms"]

    if logger:
        logger.info(f"Preprocessing {len(df_multi)} waveforms using {num_workers} workers...")
    t0 = time.time()
    file_tuples = [(str(ptbxl_data_dir), str(row["filename_hr"])) for _, row in df_multi.iterrows()]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        preprocessed_signals = list(executor.map(preprocess_single_file, file_tuples))

    waveforms = np.stack(preprocessed_signals).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, waveforms=waveforms, ecg_ids=df_multi.index.values)
    if logger:
        logger.info(f"Preprocessed waveforms cached to {cache_path} in {time.time() - t0:.2f}s.")
    return waveforms


def extract_so_sparse_latents(
    model: OperatorSetModel,
    operators: np.ndarray,
    responses: np.ndarray,
    so_indices: List[int],
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract SetOperator latents for a specific sparse lead subset."""
    model.eval()
    sub_ops = torch.as_tensor(operators[so_indices], dtype=torch.float32, device=device).unsqueeze(0)
    latents = []
    with torch.inference_mode():
        for start in range(0, len(responses), batch_size):
            end = min(start + batch_size, len(responses))
            batch_r = torch.as_tensor(np.array(responses[start:end, so_indices]), dtype=torch.float32, device=device)
            batch_q = sub_ops.expand(end - start, -1, -1)
            z = model.encode_context(batch_q, batch_r)
            latents.append(z.cpu().numpy().astype(np.float32))
    return np.concatenate(latents, axis=0)


def extract_csfm_sparse_latents(
    model: nn.Module,
    waveforms: np.ndarray,
    csfm_channels: List[int],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract CSFM Base latents for a specific sparse lead subset."""
    model.eval()
    channels_np = np.array(csfm_channels)
    all_embeddings = []
    with torch.inference_mode():
        for start_idx in range(0, len(waveforms), batch_size):
            end_idx = min(start_idx + batch_size, len(waveforms))
            batch_np = waveforms[start_idx:end_idx, csfm_channels, :]
            batch_tensor = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
            z = model(batch_tensor, channels_np)
            all_embeddings.append(z.cpu().numpy().astype(np.float32))
    return np.concatenate(all_embeddings, axis=0)


def evaluate_representation_geometry(
    z_raw: np.ndarray,
    df_multi: pd.DataFrame,
    same_pairs: List[Dict[str, Any]],
    diff_pairs: List[Dict[str, Any]],
    cross_disease_mask: np.ndarray,
    rep_name: str,
) -> Dict[str, Any]:
    """Run verification, retrieval, R^2, and cross-disease audit."""
    pop_mean = np.mean(z_raw, axis=0, keepdims=True)
    z_centered = z_raw - pop_mean
    z_normed = z_raw / (np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-12)

    # Verification (Cosine & Euclidean)
    same_cos = [float(np.dot(z_normed[p["pos1"]], z_normed[p["pos2"]])) for p in same_pairs]
    diff_cos = [float(np.dot(z_normed[p["pos1"]], z_normed[p["pos2"]])) for p in diff_pairs]

    same_euc = [float(np.linalg.norm(z_centered[p["pos1"]] - z_centered[p["pos2"]])) for p in same_pairs]
    diff_euc = [float(np.linalg.norm(z_centered[p["pos1"]] - z_centered[p["pos2"]])) for p in diff_pairs]

    y_true = np.concatenate([np.ones(len(same_cos)), np.zeros(len(diff_cos))])
    scores_cos = np.concatenate([same_cos, diff_cos])
    scores_euc = -np.concatenate([same_euc, diff_euc])

    auc_cos = fast_auroc(y_true, scores_cos)
    auc_euc = fast_auroc(y_true, scores_euc)

    # Closed-Set Retrieval (First vs Last)
    gallery_positions = []
    query_positions = []
    for pid, group in df_multi.groupby("patient_id"):
        recs = group.to_dict("records")
        gallery_positions.append(recs[0]["pos"])
        query_positions.append(recs[-1]["pos"])

    gallery_z = z_normed[gallery_positions]
    query_z = z_normed[query_positions]
    sim_matrix = query_z @ gallery_z.T

    ranks = []
    for i in range(len(query_positions)):
        sorted_indices = np.argsort(-sim_matrix[i])
        rank = int(np.where(sorted_indices == i)[0][0]) + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    top1 = float(np.mean(ranks == 1))
    top5 = float(np.mean(ranks <= 5))
    top10 = float(np.mean(ranks <= 10))
    median_rank = float(np.median(ranks))

    # Variance Decomposition
    total_ss = np.sum((z_raw - np.mean(z_raw, axis=0)) ** 2)
    pat_means = df_multi.groupby("patient_id")["pos"].apply(lambda idxs: np.mean(z_raw[idxs], axis=0))
    pat_ss = sum(len(group) * np.sum((pat_means[pid] - np.mean(z_raw, axis=0)) ** 2) for pid, group in df_multi.groupby("patient_id"))
    r2_patient = float(pat_ss / total_ss) if total_ss > 0 else 0.0

    dis_means = df_multi.groupby("diagnostic_superclass")["pos"].apply(lambda idxs: np.mean(z_raw[idxs], axis=0))
    dis_ss = sum(len(group) * np.sum((dis_means[sc] - np.mean(z_raw, axis=0)) ** 2) for sc, group in df_multi.groupby("diagnostic_superclass"))
    r2_disease = float(dis_ss / total_ss) if total_ss > 0 else 0.0

    # Cross-Disease Verification
    same_cd_cos = [same_cos[i] for i in range(len(same_cos)) if cross_disease_mask[i]]
    diff_cd_cos = diff_cos[:len(same_cd_cos) * 10]
    y_cd = np.concatenate([np.ones(len(same_cd_cos)), np.zeros(len(diff_cd_cos))])
    scores_cd_cos = np.concatenate([same_cd_cos, diff_cd_cos])
    auc_cross_disease = fast_auroc(y_cd, scores_cd_cos)

    return {
        "representation": rep_name,
        "auroc_euclidean": auc_euc,
        "auroc_cosine": auc_cos,
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "top10_accuracy": top10,
        "median_rank": median_rank,
        "r2_patient": r2_patient,
        "r2_disease": r2_disease,
        "cross_disease_auroc": auc_cross_disease,
    }


def compute_patient_bootstrap_differences(
    df_multi: pd.DataFrame,
    same_pairs: List[Dict[str, Any]],
    diff_pairs: List[Dict[str, Any]],
    z_so: np.ndarray,
    z_csfm: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute 1000-sample patient-clustered bootstrap paired difference."""
    rng = np.random.RandomState(seed)
    unique_patients = df_multi["patient_id"].unique()
    n_patients = len(unique_patients)

    pat_to_same_indices = {}
    for idx, p in enumerate(same_pairs):
        pat_to_same_indices.setdefault(p["patient_id"], []).append(idx)

    # Precompute pair scores
    z_so_norm = z_so / (np.linalg.norm(z_so, axis=1, keepdims=True) + 1e-12)
    z_csfm_norm = z_csfm / (np.linalg.norm(z_csfm, axis=1, keepdims=True) + 1e-12)

    so_same_cos = np.array([np.dot(z_so_norm[p["pos1"]], z_so_norm[p["pos2"]]) for p in same_pairs])
    so_diff_cos = np.array([np.dot(z_so_norm[p["pos1"]], z_so_norm[p["pos2"]]) for p in diff_pairs])
    csfm_same_cos = np.array([np.dot(z_csfm_norm[p["pos1"]], z_csfm_norm[p["pos2"]]) for p in same_pairs])
    csfm_diff_cos = np.array([np.dot(z_csfm_norm[p["pos1"]], z_csfm_norm[p["pos2"]]) for p in diff_pairs])

    z_so_cent = z_so - np.mean(z_so, axis=0, keepdims=True)
    z_csfm_cent = z_csfm - np.mean(z_csfm, axis=0, keepdims=True)

    so_same_euc = -np.array([np.linalg.norm(z_so_cent[p["pos1"]] - z_so_cent[p["pos2"]]) for p in same_pairs])
    so_diff_euc = -np.array([np.linalg.norm(z_so_cent[p["pos1"]] - z_so_cent[p["pos2"]]) for p in diff_pairs])
    csfm_same_euc = -np.array([np.linalg.norm(z_csfm_cent[p["pos1"]] - z_csfm_cent[p["pos2"]]) for p in same_pairs])
    csfm_diff_euc = -np.array([np.linalg.norm(z_csfm_cent[p["pos1"]] - z_csfm_cent[p["pos2"]]) for p in diff_pairs])

    diffs_cos = []
    diffs_euc = []

    for _ in range(n_bootstraps):
        sampled_pats = rng.choice(unique_patients, size=n_patients, replace=True)
        boot_same_idxs = []
        for p in sampled_pats:
            if p in pat_to_same_indices:
                boot_same_idxs.extend(pat_to_same_indices[p])

        if len(boot_same_idxs) == 0:
            continue

        boot_same_idxs = np.array(boot_same_idxs)
        n_same = len(boot_same_idxs)
        n_diff = min(n_same * 10, len(diff_pairs))
        boot_diff_idxs = rng.choice(len(diff_pairs), size=n_diff, replace=True)

        y_b = np.concatenate([np.ones(n_same), np.zeros(n_diff)])

        # Cosine AUROC
        auc_so_c = fast_auroc(y_b, np.concatenate([so_same_cos[boot_same_idxs], so_diff_cos[boot_diff_idxs]]))
        auc_csfm_c = fast_auroc(y_b, np.concatenate([csfm_same_cos[boot_same_idxs], csfm_diff_cos[boot_diff_idxs]]))
        diffs_cos.append(auc_so_c - auc_csfm_c)

        # Euclidean AUROC
        auc_so_e = fast_auroc(y_b, np.concatenate([so_same_euc[boot_same_idxs], so_diff_euc[boot_diff_idxs]]))
        auc_csfm_e = fast_auroc(y_b, np.concatenate([csfm_same_euc[boot_same_idxs], csfm_diff_euc[boot_diff_idxs]]))
        diffs_euc.append(auc_so_e - auc_csfm_e)

    diffs_cos = np.array(diffs_cos)
    diffs_euc = np.array(diffs_euc)

    return {
        "delta_cosine_mean": float(np.mean(diffs_cos)),
        "delta_cosine_ci95": [float(np.percentile(diffs_cos, 2.5)), float(np.percentile(diffs_cos, 97.5))],
        "delta_euclidean_mean": float(np.mean(diffs_euc)),
        "delta_euclidean_ci95": [float(np.percentile(diffs_euc, 2.5)), float(np.percentile(diffs_euc, 97.5))],
        "p_value_cosine": float(np.mean(diffs_cos <= 0) if np.mean(diffs_cos) > 0 else np.mean(diffs_cos >= 0)),
        "p_value_euclidean": float(np.mean(diffs_euc <= 0) if np.mean(diffs_euc) > 0 else np.mean(diffs_euc >= 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Milestone R014: Sparse-Lead Resilience Benchmark")
    parser.add_argument("--so-checkpoint", type=Path, default=Path("experiments/ptbxl_distributional_repecg/outputs/configuration_shift_evaluations/ptbxl/checkpoints/P07_ROBUSTNESS_TRAINED.pt"))
    parser.add_argument("--so-contexts", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_full12_aux_mmd/frozen_full12_contexts"))
    parser.add_argument("--csfm-checkpoint", type=Path, default=Path("external/Cardiac-Sensing-FM/pretrained/CSFM_base.pth"))
    parser.add_argument("--ptbxl-data", type=Path, default=Path("data/ptbxl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/sparse_leads"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_sparse_lead_benchmark"))
    parser.add_argument("--n-impostors", type=int, default=10)
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir / "run.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 70)
    logger.info("Milestone R014: Sparse-Lead Representation Resilience Benchmark")
    logger.info(f"Compute Device: {device}")
    logger.info(f"SetOperator Checkpoint: {args.so_checkpoint}")
    logger.info(f"CSFM Base Checkpoint:   {args.csfm_checkpoint}")
    logger.info(f"Output Directory:        {args.output_dir}")
    logger.info("=" * 70)

    # 1. Load Metadata & Repeated Cohort
    logger.info("Loading PTB-XL metadata...")
    df_meta = pd.read_csv(args.ptbxl_data / "ptbxl_database.csv", index_col="ecg_id")
    df_meta["recording_date"] = pd.to_datetime(df_meta["recording_date"])

    train_ecg = np.load(args.so_contexts / "train_ecg_ids.npy")
    val_ecg = np.load(args.so_contexts / "validation_ecg_ids.npy")
    all_ecg = np.concatenate([train_ecg, val_ecg])
    ecg_to_so_idx = {int(ecg_id): i for i, ecg_id in enumerate(all_ecg)}

    df_meta = df_meta.loc[all_ecg].copy()

    def get_superclass(scp_str):
        if not isinstance(scp_str, str):
            return "UNKNOWN"
        try:
            d = ast.literal_eval(scp_str)
            for k in ["NORM", "MI", "STTC", "CD", "HYP"]:
                if k in d:
                    return k
        except Exception:
            pass
        return "OTHER"

    df_meta["diagnostic_superclass"] = df_meta["scp_codes"].apply(get_superclass)
    pat_counts = df_meta["patient_id"].value_counts()
    multi_pats = pat_counts[pat_counts >= 2].index.values
    df_multi = df_meta[df_meta["patient_id"].isin(multi_pats)].sort_values(["patient_id", "recording_date"]).copy()
    df_multi["pos"] = np.arange(len(df_multi))
    logger.info(f"Verified Repeated Cohort: {len(multi_pats):,} patients, {len(df_multi):,} ECG records.")

    # Indices of multi-visit records in the full SetOperator responses array
    so_multi_indices = [ecg_to_so_idx[ecg_id] for ecg_id in df_multi.index.values]

    # 2. Form Identical Evaluation Pairs
    logger.info("Forming longitudinal same-patient and matched impostor pairs...")
    rng = np.random.RandomState(args.seed)
    same_pairs = []
    diff_pairs = []

    for pid, group in df_multi.groupby("patient_id"):
        recs = group.to_dict("records")
        for i in range(len(recs) - 1):
            t1 = recs[i]
            t2 = recs[i + 1]
            dt_days = (t2["recording_date"] - t1["recording_date"]).total_seconds() / 86400.0
            same_class = (t1["diagnostic_superclass"] == t2["diagnostic_superclass"])
            same_pairs.append({
                "patient_id": pid,
                "pos1": t1["pos"],
                "pos2": t2["pos"],
                "dt_days": dt_days,
                "same_class": same_class,
            })

    all_positions = df_multi["pos"].values
    for sp in same_pairs:
        pos1 = sp["pos1"]
        pid1 = sp["patient_id"]
        impostors = []
        while len(impostors) < args.n_impostors:
            cand_pos = rng.choice(all_positions)
            cand_pid = df_multi.iloc[cand_pos]["patient_id"]
            if cand_pid != pid1 and cand_pos not in impostors:
                impostors.append(cand_pos)
        for imp_pos in impostors:
            diff_pairs.append({"pos1": pos1, "pos2": imp_pos})

    same_df = pd.DataFrame(same_pairs)
    cross_disease_mask = (~same_df["same_class"]).values
    logger.info(f"Same-patient pairs: {len(same_pairs):,}, Impostor pairs: {len(diff_pairs):,}")

    # 3. Load Models
    logger.info("Initializing SetOperator model...")
    ckpt_so = torch.load(args.so_checkpoint, map_location="cpu")
    model_so = OperatorSetModel(operator_mode="continuous", response_mode="phase_cnn").to(device)
    model_so.load_state_dict(ckpt_so["state_dict"])
    model_so.eval()

    logger.info("Initializing CSFM Base model...")
    model_csfm = CSFM_model("Base")
    ckpt_csfm = torch.load(args.csfm_checkpoint, map_location="cpu")
    enc_dict = {k.replace("encoder.", ""): v for k, v in ckpt_csfm.items() if k.startswith("encoder.") and "mlp_head" not in k}
    model_csfm.load_state_dict(enc_dict, strict=False)
    model_csfm.mlp_head = nn.Identity()
    model_csfm = model_csfm.to(device)
    model_csfm.eval()

    # 4. Load Raw Responses & Preprocessed Waveforms
    logger.info("Loading precomputed SetOperator responses & operators...")
    operators = np.load(args.so_contexts / "operators.npy")
    train_res = np.load(args.so_contexts / "train_responses.npy", mmap_mode="r")
    val_res = np.load(args.so_contexts / "validation_responses.npy", mmap_mode="r")

    logger.info("Loading preprocessed waveforms for CSFM...")
    waveforms_cache = args.cache_dir / "csfm_preprocessed_waveforms_4087.npz"
    waveforms = get_preprocessed_waveforms(
        df_multi=df_multi,
        ptbxl_data_dir=args.ptbxl_data,
        cache_path=waveforms_cache,
        logger=logger,
    )

    # 5. Execute Sparse-Lead Evaluations across all configurations
    train_mask = (df_multi["strat_fold"] != 10).values
    logger.info(f"Training records for PCA fitting: {train_mask.sum():,} / {len(df_multi):,}")

    all_config_results = []
    so_12l_auc = None
    csfm_12l_auc = None

    for cfg in LEAD_CONFIGS:
        cfg_name = cfg["name"]
        logger.info("\n" + "=" * 70)
        logger.info(f"Processing Configuration: {cfg_name} ({cfg['description']})")
        logger.info("=" * 70)

        # Extract or load cached latents
        cache_so_file = args.cache_dir / f"so_latents_{cfg_name.replace(' ', '_').replace('+', 'plus')}.npy"
        cache_csfm_file = args.cache_dir / f"csfm_latents_{cfg_name.replace(' ', '_').replace('+', 'plus')}.npy"

        if cache_so_file.exists():
            z_so_native = np.load(cache_so_file)
            logger.info(f"Loaded cached SetOperator latents: {z_so_native.shape}")
        else:
            logger.info(f"Extracting SetOperator latents for {cfg_name} (leads={cfg['so_indices']})...")
            t0 = time.time()
            z_so_train = extract_so_sparse_latents(model_so, operators, train_res, cfg["so_indices"], device=device)
            z_so_val = extract_so_sparse_latents(model_so, operators, val_res, cfg["so_indices"], device=device)
            z_so_all = np.concatenate([z_so_train, z_so_val], axis=0)
            z_so_native = z_so_all[so_multi_indices].astype(np.float32)
            np.save(cache_so_file, z_so_native)
            logger.info(f"SetOperator extraction complete: {z_so_native.shape} in {time.time() - t0:.2f}s")

        if cache_csfm_file.exists():
            z_csfm_native = np.load(cache_csfm_file)
            logger.info(f"Loaded cached CSFM Base latents: {z_csfm_native.shape}")
        else:
            logger.info(f"Extracting CSFM Base latents for {cfg_name} (channels={cfg['csfm_channels']})...")
            t0 = time.time()
            z_csfm_native = extract_csfm_sparse_latents(model_csfm, waveforms, cfg["csfm_channels"], device=device)
            np.save(cache_csfm_file, z_csfm_native)
            logger.info(f"CSFM Base extraction complete: {z_csfm_native.shape} in {time.time() - t0:.2f}s")

        # Fit PCA on training patients only for matched comparisons
        logger.info(f"Fitting PCA d=128 on training patients...")
        pca_so_128 = PCA(n_components=128, random_state=args.seed).fit(z_so_native[train_mask])
        z_so_d128 = pca_so_128.transform(z_so_native).astype(np.float32)

        pca_csfm_128 = PCA(n_components=128, random_state=args.seed).fit(z_csfm_native[train_mask])
        z_csfm_d128 = pca_csfm_128.transform(z_csfm_native).astype(np.float32)

        pca_csfm_256 = PCA(n_components=256, random_state=args.seed).fit(z_csfm_native[train_mask])
        z_csfm_d256 = pca_csfm_256.transform(z_csfm_native).astype(np.float32)

        # Evaluate arms
        res_so_nat = evaluate_representation_geometry(z_so_native, df_multi, same_pairs, diff_pairs, cross_disease_mask, "SetOperator (Native d=256)")
        res_csfm_nat = evaluate_representation_geometry(z_csfm_native, df_multi, same_pairs, diff_pairs, cross_disease_mask, "CSFM Base (Native d=768)")
        res_so_128 = evaluate_representation_geometry(z_so_d128, df_multi, same_pairs, diff_pairs, cross_disease_mask, "SetOperator (Matched d=128)")
        res_csfm_128 = evaluate_representation_geometry(z_csfm_d128, df_multi, same_pairs, diff_pairs, cross_disease_mask, "CSFM Base (Matched d=128)")
        res_csfm_256 = evaluate_representation_geometry(z_csfm_d256, df_multi, same_pairs, diff_pairs, cross_disease_mask, "CSFM Base (Matched d=256)")

        if cfg_name == "12L":
            so_12l_auc = res_so_nat["auroc_euclidean"]
            csfm_12l_auc = res_csfm_nat["auroc_euclidean"]

        # Paired bootstrap difference (Native & Matched d=128)
        boot_diff_nat = compute_patient_bootstrap_differences(df_multi, same_pairs, diff_pairs, z_so_native, z_csfm_native, n_bootstraps=args.n_bootstraps, seed=args.seed)
        boot_diff_128 = compute_patient_bootstrap_differences(df_multi, same_pairs, diff_pairs, z_so_d128, z_csfm_d128, n_bootstraps=args.n_bootstraps, seed=args.seed)

        cfg_summary = {
            "config_name": cfg_name,
            "description": cfg["description"],
            "lead_count": len(cfg["so_indices"]),
            "setoperator_native": res_so_nat,
            "csfm_native": res_csfm_nat,
            "setoperator_matched_d128": res_so_128,
            "csfm_matched_d128": res_csfm_128,
            "csfm_matched_d256": res_csfm_256,
            "bootstrap_native": boot_diff_nat,
            "bootstrap_matched_d128": boot_diff_128,
        }
        all_config_results.append(cfg_summary)

        logger.info(f"--- Results for {cfg_name} ---")
        logger.info(f"SetOperator Native: Euclidean AUC = {res_so_nat['auroc_euclidean']:.4f}, Cosine AUC = {res_so_nat['auroc_cosine']:.4f}, Top-1 = {res_so_nat['top1_accuracy']*100:.2f}%, Pat R² = {res_so_nat['r2_patient']*100:.1f}%")
        logger.info(f"CSFM Base Native:   Euclidean AUC = {res_csfm_nat['auroc_euclidean']:.4f}, Cosine AUC = {res_csfm_nat['auroc_cosine']:.4f}, Top-1 = {res_csfm_nat['top1_accuracy']*100:.2f}%, Pat R² = {res_csfm_nat['r2_patient']*100:.1f}%")
        logger.info(f"Matched d=128:      SO Euclidean = {res_so_128['auroc_euclidean']:.4f} vs CSFM = {res_csfm_128['auroc_euclidean']:.4f} (Δ = +{res_so_128['auroc_euclidean'] - res_csfm_128['auroc_euclidean']:.4f})")
        logger.info(f"Paired Bootstrap Δ: Native Eucl = +{boot_diff_nat['delta_euclidean_mean']:.4f} (95% CI: [{boot_diff_nat['delta_euclidean_ci95'][0]:.4f}, {boot_diff_nat['delta_euclidean_ci95'][1]:.4f}], p={boot_diff_nat['p_value_euclidean']:.4f})")

    # Compute Retention Ratios
    for res in all_config_results:
        so_euc = res["setoperator_native"]["auroc_euclidean"]
        csfm_euc = res["csfm_native"]["auroc_euclidean"]
        res["retention_so_euclidean_pct"] = float(((so_euc - 0.5) / (so_12l_auc - 0.5)) * 100.0)
        res["retention_csfm_euclidean_pct"] = float(((csfm_euc - 0.5) / (csfm_12l_auc - 0.5)) * 100.0)

    # 6. Save Structured JSON Results
    results_payload = {
        "schema_version": SCHEMA_VERSION,
        "n_patients": len(multi_pats),
        "n_records": len(df_multi),
        "n_same_pairs": len(same_pairs),
        "n_impostor_pairs": len(diff_pairs),
        "configurations": all_config_results,
    }

    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results_payload, f, indent=2)
    logger.info(f"\nSaved structured results to: {args.output_dir / 'results.json'}")

    # 7. Generate Markdown Summary Table
    md_table = [
        "# Milestone R014: Sparse-Lead Representation Resilience Benchmark",
        "",
        "### Head-to-Head Geometry across Sensing Configurations: SetOperator vs CSFM Base",
        "",
        "| Configuration | Lead Count | SetOperator Native Eucl AUROC | CSFM Base Native Eucl AUROC | Paired Δ (SO − CSFM) [95% CI] | Matched d=128 Δ (SO − CSFM) | SetOperator Top-1 | CSFM Top-1 | SO Retention (% of 12L) | CSFM Retention (% of 12L) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for res in all_config_results:
        name = res["config_name"]
        leads = res["lead_count"]
        so_euc = res["setoperator_native"]["auroc_euclidean"]
        csfm_euc = res["csfm_native"]["auroc_euclidean"]
        b_nat = res["bootstrap_native"]
        so_128 = res["setoperator_matched_d128"]["auroc_euclidean"]
        csfm_128 = res["csfm_matched_d128"]["auroc_euclidean"]
        m_diff = so_128 - csfm_128
        so_top1 = res["setoperator_native"]["top1_accuracy"] * 100.0
        csfm_top1 = res["csfm_native"]["top1_accuracy"] * 100.0
        so_ret = res["retention_so_euclidean_pct"]
        csfm_ret = res["retention_csfm_euclidean_pct"]

        ci_str = f"[{b_nat['delta_euclidean_ci95'][0]:+.4f}, {b_nat['delta_euclidean_ci95'][1]:+.4f}]"
        md_table.append(
            f"| **{name}** | {leads} | **{so_euc:.4f}** | {csfm_euc:.4f} | **{b_nat['delta_euclidean_mean']:+.4f}** {ci_str} | **{m_diff:+.4f}** | **{so_top1:.1f}%** | {csfm_top1:.1f}% | **{so_ret:.1f}%** | {csfm_ret:.1f}% |"
        )

    md_content = "\n".join(md_table) + "\n"
    with open(args.output_dir / "sparse_lead_resilience_table.md", "w") as f:
        f.write(md_content)
    logger.info(f"Saved Markdown table to: {args.output_dir / 'sparse_lead_resilience_table.md'}")
    logger.info("\n" + md_content)

    # 8. Generate Publication Figure
    logger.info("Generating publication figure...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    cfg_labels = [r["config_name"] for r in all_config_results]
    x = np.arange(len(cfg_labels))

    # Panel A: Verification AUROC
    so_euc_vals = [r["setoperator_native"]["auroc_euclidean"] for r in all_config_results]
    csfm_euc_vals = [r["csfm_native"]["auroc_euclidean"] for r in all_config_results]
    so_cos_vals = [r["setoperator_native"]["auroc_cosine"] for r in all_config_results]
    csfm_cos_vals = [r["csfm_native"]["auroc_cosine"] for r in all_config_results]

    ax = axes[0]
    ax.plot(x, so_euc_vals, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="SetOperator (Euclidean)")
    ax.plot(x, csfm_euc_vals, "s--", color="#ff7f0e", linewidth=2.5, markersize=8, label="CSFM Base (Euclidean)")
    ax.plot(x, so_cos_vals, "^:", color="#2ca02c", linewidth=1.8, markersize=7, label="SetOperator (Cosine)")
    ax.plot(x, csfm_cos_vals, "v:", color="#d62728", linewidth=1.8, markersize=7, label="CSFM Base (Cosine)")
    ax.set_xticks(x)
    ax.set_xticklabels(cfg_labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Verification AUROC", fontsize=12, fontweight="bold")
    ax.set_title("A. Same-Patient Verification AUROC", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)

    # Panel B: Top-1 Retrieval Accuracy
    ax = axes[1]
    so_top1_vals = [r["setoperator_native"]["top1_accuracy"] * 100 for r in all_config_results]
    csfm_top1_vals = [r["csfm_native"]["top1_accuracy"] * 100 for r in all_config_results]
    width = 0.35
    ax.bar(x - width/2, so_top1_vals, width, label="SetOperator", color="#1f77b4", alpha=0.9)
    ax.bar(x + width/2, csfm_top1_vals, width, label="CSFM Base", color="#ff7f0e", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(cfg_labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Top-1 Retrieval Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("B. Closed-Set Retrieval Accuracy", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(frameon=True, fontsize=10)

    # Panel C: Information Retention Ratio (% of 12L)
    ax = axes[2]
    so_ret_vals = [r["retention_so_euclidean_pct"] for r in all_config_results]
    csfm_ret_vals = [r["retention_csfm_euclidean_pct"] for r in all_config_results]
    ax.plot(x, so_ret_vals, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="SetOperator")
    ax.plot(x, csfm_ret_vals, "s--", color="#ff7f0e", linewidth=2.5, markersize=8, label="CSFM Base")
    ax.axhline(100, color="gray", linestyle=":", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(cfg_labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Geometry Retention (% of 12L)", fontsize=12, fontweight="bold")
    ax.set_title("C. Sensing Degradation Resilience", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    plot_path = args.output_dir / "sparse_lead_resilience_curves.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved publication figure to: {plot_path}")

    logger.info("\n=== Milestone R014 Execution COMPLETED Successfully ===")


if __name__ == "__main__":
    main()
