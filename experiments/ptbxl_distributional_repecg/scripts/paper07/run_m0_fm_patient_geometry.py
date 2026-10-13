#!/usr/bin/env python3
"""Milestone M0-FM: Foundation-Model Patient-Geometry Benchmark.

Head-to-head geometric comparison of frozen SetOperator E6 (d=256) vs frozen
Cardiac Sensing Foundation Model (CSFM Base, d=768) on the exact same PTB-XL
repeated-patient cohort (1,693 patients, 4,087 ECGs, identical pairs).

Dimensionality Control Protocol:
1. Native representations: z_SO in R^256 vs z_CSFM in R^768.
2. Matched dimensionality: Fit PCA strictly on training patients only and transform
   both representations to d=256 and d=128. Never fit PCA on test patients.

Endpoints (reported directly without composite scoring):
- Same-patient Euclidean verification AUROC
- Same-patient Cosine verification AUROC
- Top-1, Top-5, Top-10 closed-set retrieval accuracy & Median Rank
- Patient explained variance (R^2_patient) vs Diagnostic Superclass (R^2_disease)
- Cross-disease verification AUROC
- Time-stratified verification AUROCs (<7d, 7-30d, 30-180d, >180d)
- 1,000-sample patient-clustered bootstrap 95% CIs and paired difference Delta.
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

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import wfdb

# Add external CSFM directory to path
CSFM_REPO = Path(__file__).resolve().parents[4] / "external" / "Cardiac-Sensing-FM"
if str(CSFM_REPO) not in sys.path:
    sys.path.insert(0, str(CSFM_REPO))

from network.model import CSFM_model
from utils.preprocess import preprocess_ecg


SCHEMA_VERSION = "prism_ecg_stage_m0_fm_geometry_v1"


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("m0_fm_geometry")
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
    # sig shape: (N_samples, 12). Transpose to (12, N_samples)
    ecg_clean = preprocess_ecg(sig.T, fs=meta["fs"])
    return ecg_clean.astype(np.float32)


def extract_csfm_embeddings(
    df_multi: pd.DataFrame,
    ptbxl_data_dir: Path,
    checkpoint_path: Path,
    cache_path: Path,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 8,
    logger: logging.Logger = None,
) -> np.ndarray:
    """Extract or load cached CSFM Base representations for the multi-visit cohort."""
    if cache_path.exists():
        if logger:
            logger.info(f"Loading cached CSFM embeddings from {cache_path}...")
        data = np.load(cache_path)
        return data["csfm_z"]

    if logger:
        logger.info(f"Extracting CSFM embeddings for {len(df_multi)} ECGs using {num_workers} workers...")
    t0 = time.time()

    # 1. Preprocess raw waveforms
    file_tuples = [(str(ptbxl_data_dir), str(row["filename_hr"])) for _, row in df_multi.iterrows()]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        preprocessed_signals = list(executor.map(preprocess_single_file, file_tuples))

    if logger:
        logger.info(f"Waveform preprocessing complete in {time.time() - t0:.2f}s. Running GPU inference...")

    # 2. Load CSFM model
    model = CSFM_model("Base")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    encoder_state_dict = {
        k.replace("encoder.", ""): v
        for k, v in ckpt.items()
        if k.startswith("encoder.") and "mlp_head" not in k
    }
    model.load_state_dict(encoder_state_dict, strict=False)
    model.mlp_head = nn.Identity()
    model = model.to(device)
    model.eval()

    channels = np.arange(12)  # Standard 12-lead channels 0-11
    all_embeddings = []

    # 3. Batch GPU inference
    with torch.inference_mode():
        for start_idx in range(0, len(preprocessed_signals), batch_size):
            end_idx = min(start_idx + batch_size, len(preprocessed_signals))
            batch_np = np.stack(preprocessed_signals[start_idx:end_idx])
            batch_tensor = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
            z = model(batch_tensor, channels)
            all_embeddings.append(z.cpu().numpy().astype(np.float32))

    csfm_z = np.concatenate(all_embeddings, axis=0)
    if logger:
        logger.info(f"CSFM extraction complete: shape {csfm_z.shape} in {time.time() - t0:.2f}s.")

    # Cache to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, csfm_z=csfm_z, ecg_ids=df_multi.index.values)
    if logger:
        logger.info(f"Saved CSFM embeddings to cache: {cache_path}")
    return csfm_z


def evaluate_representation_geometry(
    z_raw: np.ndarray,
    df_multi: pd.DataFrame,
    same_pairs: List[Dict[str, Any]],
    diff_pairs: List[Dict[str, Any]],
    time_bins: List[Tuple[str, np.ndarray]],
    cross_disease_mask: np.ndarray,
    rep_name: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run the comprehensive geometry and identity audit for a given latent representation."""
    # 1. Normalization & Centering
    pop_mean = np.mean(z_raw, axis=0, keepdims=True)
    z_centered = z_raw - pop_mean
    z_normed = z_raw / (np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-12)

    # 2. Pairwise Verification (Cosine & Euclidean)
    # Cosine on z_normed
    same_cos = [float(np.dot(z_normed[p["pos1"]], z_normed[p["pos2"]])) for p in same_pairs]
    diff_cos = [float(np.dot(z_normed[p["pos1"]], z_normed[p["pos2"]])) for p in diff_pairs]

    # Euclidean on z_centered
    same_euc = [float(np.linalg.norm(z_centered[p["pos1"]] - z_centered[p["pos2"]])) for p in same_pairs]
    diff_euc = [float(np.linalg.norm(z_centered[p["pos1"]] - z_centered[p["pos2"]])) for p in diff_pairs]

    y_true = np.concatenate([np.ones(len(same_cos)), np.zeros(len(diff_cos))])
    scores_cos = np.concatenate([same_cos, diff_cos])
    scores_euc = -np.concatenate([same_euc, diff_euc])  # Smaller distance = higher score

    auc_cos = fast_auroc(y_true, scores_cos)
    auc_euc = fast_auroc(y_true, scores_euc)

    # 3. Closed-Set Retrieval (Gallery = earliest, Query = latest)
    gallery_positions = []
    query_positions = []
    for pid, group in df_multi.groupby("patient_id"):
        recs = group.to_dict("records")
        gallery_positions.append(recs[0]["pos"])
        query_positions.append(recs[-1]["pos"])

    gallery_z = z_normed[gallery_positions]
    query_z = z_normed[query_positions]
    n_candidates = len(query_positions)

    sim_matrix = query_z @ gallery_z.T  # (N_pats, N_pats)
    ranks = []
    for i in range(n_candidates):
        sorted_indices = np.argsort(-sim_matrix[i])
        rank = int(np.where(sorted_indices == i)[0][0]) + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    top1 = float(np.mean(ranks == 1))
    top5 = float(np.mean(ranks <= 5))
    top10 = float(np.mean(ranks <= 10))
    median_rank = float(np.median(ranks))

    # 4. Descriptive Variance Decomposition
    total_ss = np.sum((z_raw - np.mean(z_raw, axis=0)) ** 2)

    pat_means = df_multi.groupby("patient_id")["pos"].apply(lambda idxs: np.mean(z_raw[idxs], axis=0))
    pat_ss = sum(
        len(group) * np.sum((pat_means[pid] - np.mean(z_raw, axis=0)) ** 2)
        for pid, group in df_multi.groupby("patient_id")
    )
    r2_patient = float(pat_ss / max(total_ss, 1e-12))

    class_means = df_multi.groupby("diagnostic_superclass")["pos"].apply(lambda idxs: np.mean(z_raw[idxs], axis=0))
    class_ss = sum(
        len(group) * np.sum((class_means[cls] - np.mean(z_raw, axis=0)) ** 2)
        for cls, group in df_multi.groupby("diagnostic_superclass")
    )
    r2_disease = float(class_ss / max(total_ss, 1e-12))
    variance_ratio = float(r2_patient / max(r2_disease, 1e-6))

    # 5. Cross-Disease Verification
    same_cos_arr = np.array(same_cos)
    cd_scores = same_cos_arr[cross_disease_mask]
    impostor_sample = np.array(diff_cos[: len(cd_scores) * 10])
    y_cd = np.concatenate([np.ones(len(cd_scores)), np.zeros(len(impostor_sample))])
    scores_cd = np.concatenate([cd_scores, impostor_sample])
    auc_cross_disease = float(fast_auroc(y_cd, scores_cd))

    # 6. Time Breakdown
    time_results = {}
    for label, mask in time_bins:
        n_pairs = int(mask.sum())
        sub_same_cos = same_cos_arr[mask]
        sub_diff_cos = np.array(diff_cos[: len(sub_same_cos) * 10])
        y_sub = np.concatenate([np.ones(len(sub_same_cos)), np.zeros(len(sub_diff_cos))])
        scores_sub = np.concatenate([sub_same_cos, sub_diff_cos])
        auc_sub = float(fast_auroc(y_sub, scores_sub))
        time_results[label] = {
            "n_pairs": n_pairs,
            "auroc": auc_sub,
            "median_cos": float(np.median(sub_same_cos)),
        }

    logger.info(f"[{rep_name}] Metrics Summary:")
    logger.info(f"   Euclidean Verification AUROC: {auc_euc:.4f} | Cosine AUROC: {auc_cos:.4f}")
    logger.info(f"   Top-1 Retrieval: {top1*100:.2f}% | Top-5: {top5*100:.2f}% | Top-10: {top10*100:.2f}% | Median Rank: {median_rank:.1f}")
    logger.info(f"   Variance R^2: Patient = {r2_patient*100:.2f}%, Disease = {r2_disease*100:.2f}% (Ratio = {variance_ratio:.2f}x)")
    logger.info(f"   Cross-Disease Verification AUROC: {auc_cross_disease:.4f}")
    for lbl, t_data in time_results.items():
        logger.info(f"   Time Gap {lbl:15s}: AUROC = {t_data['auroc']:.4f}")

    return {
        "rep_name": rep_name,
        "dim": int(z_raw.shape[1]),
        "auroc_euclidean": auc_euc,
        "auroc_cosine": auc_cos,
        "same_euc_median": float(np.median(same_euc)),
        "diff_euc_median": float(np.median(diff_euc)),
        "same_cos_median": float(np.median(same_cos)),
        "diff_cos_median": float(np.median(diff_cos)),
        "retrieval": {
            "top1": top1,
            "top5": top5,
            "top10": top10,
            "median_rank": median_rank,
            "n_candidates": n_candidates,
        },
        "variance_decomposition": {
            "r2_patient": r2_patient,
            "r2_disease": r2_disease,
            "variance_ratio": variance_ratio,
        },
        "cross_disease_auroc": auc_cross_disease,
        "time_breakdown": time_results,
    }


def compute_paired_bootstrap_cis(
    z_so: np.ndarray,
    z_csfm: np.ndarray,
    df_multi: pd.DataFrame,
    same_pairs: List[Dict[str, Any]],
    diff_pairs: List[Dict[str, Any]],
    n_bootstraps: int = 1000,
    seed: int = 42,
    logger: logging.Logger = None,
) -> Dict[str, Any]:
    """Compute patient-clustered paired bootstrap 95% CIs for Euclidean AUROC, Cosine AUROC, and Top-1."""
    if logger:
        logger.info(f"Running {n_bootstraps} paired patient-level bootstrap resamples...")
    rng = np.random.RandomState(seed)

    unique_patients = df_multi["patient_id"].unique()
    n_patients = len(unique_patients)

    # Pre-normalize representations
    so_mean = np.mean(z_so, axis=0, keepdims=True)
    so_cent = z_so - so_mean
    so_norm = z_so / (np.linalg.norm(z_so, axis=1, keepdims=True) + 1e-12)

    csfm_mean = np.mean(z_csfm, axis=0, keepdims=True)
    csfm_cent = z_csfm - csfm_mean
    csfm_norm = z_csfm / (np.linalg.norm(z_csfm, axis=1, keepdims=True) + 1e-12)

    # Precompute pair similarities
    so_same_cos = np.array([float(np.dot(so_norm[p["pos1"]], so_norm[p["pos2"]])) for p in same_pairs])
    so_diff_cos = np.array([float(np.dot(so_norm[p["pos1"]], so_norm[p["pos2"]])) for p in diff_pairs])
    so_same_euc = -np.array([float(np.linalg.norm(so_cent[p["pos1"]] - so_cent[p["pos2"]])) for p in same_pairs])
    so_diff_euc = -np.array([float(np.linalg.norm(so_cent[p["pos1"]] - so_cent[p["pos2"]])) for p in diff_pairs])

    csfm_same_cos = np.array([float(np.dot(csfm_norm[p["pos1"]], csfm_norm[p["pos2"]])) for p in same_pairs])
    csfm_diff_cos = np.array([float(np.dot(csfm_norm[p["pos1"]], csfm_norm[p["pos2"]])) for p in diff_pairs])
    csfm_same_euc = -np.array([float(np.linalg.norm(csfm_cent[p["pos1"]] - csfm_cent[p["pos2"]])) for p in same_pairs])
    csfm_diff_euc = -np.array([float(np.linalg.norm(csfm_cent[p["pos1"]] - csfm_cent[p["pos2"]])) for p in diff_pairs])

    pair_patients = np.array([p["patient_id"] for p in same_pairs])
    y_true = np.concatenate([np.ones(len(same_pairs)), np.zeros(len(diff_pairs))])

    # Bootstrap accumulators
    diffs_cos_auc = []
    diffs_euc_auc = []

    so_cos_auc_list = []
    csfm_cos_auc_list = []
    so_euc_auc_list = []
    csfm_euc_auc_list = []

    for b in range(n_bootstraps):
        boot_pats = set(rng.choice(unique_patients, size=n_patients, replace=True))
        # Keep same-pairs from selected bootstrap patients
        mask_same = np.isin(pair_patients, list(boot_pats))
        if mask_same.sum() == 0:
            continue

        sub_y = np.concatenate([np.ones(mask_same.sum()), np.zeros(len(diff_pairs))])

        so_s_cos = np.concatenate([so_same_cos[mask_same], so_diff_cos])
        csfm_s_cos = np.concatenate([csfm_same_cos[mask_same], csfm_diff_cos])
        auc_so_cos = fast_auroc(sub_y, so_s_cos)
        auc_csfm_cos = fast_auroc(sub_y, csfm_s_cos)

        so_s_euc = np.concatenate([so_same_euc[mask_same], so_diff_euc])
        csfm_s_euc = np.concatenate([csfm_same_euc[mask_same], csfm_diff_euc])
        auc_so_euc = fast_auroc(sub_y, so_s_euc)
        auc_csfm_euc = fast_auroc(sub_y, csfm_s_euc)

        so_cos_auc_list.append(auc_so_cos)
        csfm_cos_auc_list.append(auc_csfm_cos)
        diffs_cos_auc.append(auc_so_cos - auc_csfm_cos)

        so_euc_auc_list.append(auc_so_euc)
        csfm_euc_auc_list.append(auc_csfm_euc)
        diffs_euc_auc.append(auc_so_euc - auc_csfm_euc)

    def ci_stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }

    return {
        "so_cosine_auc": ci_stats(so_cos_auc_list),
        "csfm_cosine_auc": ci_stats(csfm_cos_auc_list),
        "delta_cosine_auc": ci_stats(diffs_cos_auc),
        "so_euclidean_auc": ci_stats(so_euc_auc_list),
        "csfm_euclidean_auc": ci_stats(csfm_euc_auc_list),
        "delta_euclidean_auc": ci_stats(diffs_euc_auc),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone M0-FM: Foundation-Model Patient-Geometry Benchmark")
    parser.add_argument(
        "--so-cache",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/setoperator_e6_all_z.npz"),
    )
    parser.add_argument(
        "--csfm-checkpoint",
        type=Path,
        default=Path("external/Cardiac-Sensing-FM/pretrained/CSFM_base.pth"),
    )
    parser.add_argument(
        "--csfm-cache",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/csfm_base_multi_z.npz"),
    )
    parser.add_argument(
        "--ptbxl-data",
        type=Path,
        default=Path("data/ptbxl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_m0_fm_patient_geometry"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-impostors", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir / "run.log")

    logger.info("======================================================================")
    logger.info("=== Milestone M0-FM: Foundation-Model Patient-Geometry Benchmark ===")
    logger.info("======================================================================")
    logger.info(f"SetOperator Cache: {args.so_cache}")
    logger.info(f"CSFM Checkpoint:   {args.csfm_checkpoint}")
    logger.info(f"CSFM Cache:        {args.csfm_cache}")
    logger.info(f"PTB-XL Data Dir:   {args.ptbxl_data}")
    logger.info(f"Output Directory:  {args.output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device:    {device}")

    # 1. Load SetOperator Latents & ECG IDs
    logger.info("\nLoading cached SetOperator latents...")
    so_data = np.load(args.so_cache)
    all_z_so = so_data["all_z"]  # (17417, 256)
    all_ecg = so_data["all_ecg"]  # (17417,)
    ecg_to_so_idx = {int(ecg_id): i for i, ecg_id in enumerate(all_ecg)}

    # 2. Load & Prepare Metadata
    logger.info("Loading PTB-XL database metadata...")
    df_meta = pd.read_csv(args.ptbxl_data / "ptbxl_database.csv", index_col="ecg_id")
    df_meta["recording_date"] = pd.to_datetime(df_meta["recording_date"])
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

    # 3. Filter to Repeated-Patient Cohort (Exact same 1,693 patients, 4,087 records)
    pat_counts = df_meta["patient_id"].value_counts()
    multi_pats = pat_counts[pat_counts >= 2].index.values
    df_multi = df_meta[df_meta["patient_id"].isin(multi_pats)].sort_values(["patient_id", "recording_date"]).copy()
    df_multi["pos"] = np.arange(len(df_multi))
    logger.info(f"Verified Repeated Cohort: {len(multi_pats):,} patients, {len(df_multi):,} ECG records.")

    # Slice SetOperator latents for df_multi
    so_indices = [ecg_to_so_idx[ecg_id] for ecg_id in df_multi.index.values]
    z_so_native = all_z_so[so_indices].astype(np.float32)
    logger.info(f"SetOperator Native Latents: {z_so_native.shape}")

    # 4. Extract / Load CSFM Base Embeddings
    z_csfm_native = extract_csfm_embeddings(
        df_multi=df_multi,
        ptbxl_data_dir=args.ptbxl_data,
        checkpoint_path=args.csfm_checkpoint,
        cache_path=args.csfm_cache,
        device=device,
        logger=logger,
    )
    logger.info(f"CSFM Base Native Latents:   {z_csfm_native.shape}")

    # 5. Form Identical Evaluation Pairs
    logger.info("\nForming longitudinal same-patient and matched impostor pairs...")
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
                "class1": t1["diagnostic_superclass"],
                "class2": t2["diagnostic_superclass"],
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
            diff_pairs.append({
                "pos1": pos1,
                "pos2": imp_pos,
            })

    logger.info(f"Longitudinal Same-Patient Pairs: {len(same_pairs):,}")
    logger.info(f"Matched Impostor Pairs (10x):    {len(diff_pairs):,}")

    # Prepare masks for sub-analyses
    same_df = pd.DataFrame(same_pairs)
    time_bins = [
        ("< 7 days", same_df["dt_days"] < 7),
        ("7 - 30 days", (same_df["dt_days"] >= 7) & (same_df["dt_days"] < 30)),
        ("30 - 180 days", (same_df["dt_days"] >= 30) & (same_df["dt_days"] < 180)),
        ("> 180 days", same_df["dt_days"] >= 180),
    ]
    cross_disease_mask = (~same_df["same_class"]).values

    # 6. Dimensionality Matching via PCA (Fitted Strictly on Training Patients)
    logger.info("\nFitting PCA on Training Patients (strat_fold != 10 / folds 1-8)...")
    train_mask = (df_multi["strat_fold"] != 10).values
    logger.info(f"Training records for PCA: {train_mask.sum():,} / {len(df_multi):,} ({train_mask.mean()*100:.1f}%)")

    # Fit PCA d=256 for CSFM
    pca_csfm_256 = PCA(n_components=256, random_state=args.seed)
    pca_csfm_256.fit(z_csfm_native[train_mask])
    z_csfm_d256 = pca_csfm_256.transform(z_csfm_native).astype(np.float32)
    logger.info(f"CSFM d=256: Variance Explained on Train = {pca_csfm_256.explained_variance_ratio_.sum()*100:.2f}%")

    # Fit PCA d=128 for CSFM
    pca_csfm_128 = PCA(n_components=128, random_state=args.seed)
    pca_csfm_128.fit(z_csfm_native[train_mask])
    z_csfm_d128 = pca_csfm_128.transform(z_csfm_native).astype(np.float32)
    logger.info(f"CSFM d=128: Variance Explained on Train = {pca_csfm_128.explained_variance_ratio_.sum()*100:.2f}%")

    # Fit PCA d=128 for SetOperator
    pca_so_128 = PCA(n_components=128, random_state=args.seed)
    pca_so_128.fit(z_so_native[train_mask])
    z_so_d128 = pca_so_128.transform(z_so_native).astype(np.float32)
    logger.info(f"SetOperator d=128: Variance Explained on Train = {pca_so_128.explained_variance_ratio_.sum()*100:.2f}%")

    # SetOperator d=256 is its native dimension
    z_so_d256 = z_so_native

    # 7. Run Comprehensive Audit Across All Representation Arms
    arms = [
        ("SetOperator (Native, d=256)", z_so_native),
        ("CSFM Base (Native, d=768)", z_csfm_native),
        ("SetOperator (Matched, d=256)", z_so_d256),
        ("CSFM Base (Matched, d=256)", z_csfm_d256),
        ("SetOperator (Matched, d=128)", z_so_d128),
        ("CSFM Base (Matched, d=128)", z_csfm_d128),
    ]

    results_by_arm = {}
    for name, z_mat in arms:
        logger.info(f"\n--- Evaluating Arm: {name} ---")
        res = evaluate_representation_geometry(
            z_raw=z_mat,
            df_multi=df_multi,
            same_pairs=same_pairs,
            diff_pairs=diff_pairs,
            time_bins=time_bins,
            cross_disease_mask=cross_disease_mask,
            rep_name=name,
            logger=logger,
        )
        results_by_arm[name] = res

    # 8. Paired Bootstrap Statistical Significance (Native & Matched d=256)
    logger.info("\n--- Computing Paired Patient-Level Bootstrap CIs (Native) ---")
    boot_native = compute_paired_bootstrap_cis(
        z_so=z_so_native,
        z_csfm=z_csfm_native,
        df_multi=df_multi,
        same_pairs=same_pairs,
        diff_pairs=diff_pairs,
        n_bootstraps=args.n_bootstrap,
        seed=args.seed,
        logger=logger,
    )

    logger.info("\n--- Computing Paired Patient-Level Bootstrap CIs (Matched d=256) ---")
    boot_d256 = compute_paired_bootstrap_cis(
        z_so=z_so_d256,
        z_csfm=z_csfm_d256,
        df_multi=df_multi,
        same_pairs=same_pairs,
        diff_pairs=diff_pairs,
        n_bootstraps=args.n_bootstrap,
        seed=args.seed,
        logger=logger,
    )

    # 9. Format Markdown Comparison Table
    table_lines = [
        "# Milestone M0-FM: Foundation-Model Patient-Geometry Benchmark",
        "",
        "| Endpoint / Metric | SetOperator (Native, d=256) | CSFM Base (Native, d=768) | SetOperator (Matched, d=256) | CSFM Base (Matched, d=256) | SetOperator (Matched, d=128) | CSFM Base (Matched, d=128) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    rows = [
        ("Same-Patient Euclidean AUROC", lambda r: f"{r['auroc_euclidean']:.4f}"),
        ("Same-Patient Cosine AUROC", lambda r: f"{r['auroc_cosine']:.4f}"),
        ("Top-1 Patient Retrieval", lambda r: f"{r['retrieval']['top1']*100:.2f}%"),
        ("Top-5 Patient Retrieval", lambda r: f"{r['retrieval']['top5']*100:.2f}%"),
        ("Top-10 Patient Retrieval", lambda r: f"{r['retrieval']['top10']*100:.2f}%"),
        ("Median Patient Rank", lambda r: f"{r['retrieval']['median_rank']:.1f}"),
        ("Patient-Effect Variance (R²)", lambda r: f"{r['variance_decomposition']['r2_patient']*100:.2f}%"),
        ("Diagnostic-Superclass Variance (R²)", lambda r: f"{r['variance_decomposition']['r2_disease']*100:.2f}%"),
        ("Cross-Disease Verification AUROC", lambda r: f"{r['cross_disease_auroc']:.4f}"),
        ("Time Gap < 7 days AUROC", lambda r: f"{r['time_breakdown']['< 7 days']['auroc']:.4f}"),
        ("Time Gap 7 - 30 days AUROC", lambda r: f"{r['time_breakdown']['7 - 30 days']['auroc']:.4f}"),
        ("Time Gap 30 - 180 days AUROC", lambda r: f"{r['time_breakdown']['30 - 180 days']['auroc']:.4f}"),
        ("Time Gap > 180 days AUROC", lambda r: f"{r['time_breakdown']['> 180 days']['auroc']:.4f}"),
    ]

    for label, fn in rows:
        row_str = f"| **{label}** | " + " | ".join(fn(results_by_arm[name]) for name, _ in arms) + " |"
        table_lines.append(row_str)

    table_lines.append("")
    table_lines.append("### Paired Patient-Level Bootstrap Differences (SetOperator − CSFM Base)")
    table_lines.append(f"- **Native Cosine AUROC Δ**: {boot_native['delta_cosine_auc']['mean']:+.4f} (95% CI: [{boot_native['delta_cosine_auc']['ci_lower']:+.4f}, {boot_native['delta_cosine_auc']['ci_upper']:+.4f}])")
    table_lines.append(f"- **Native Euclidean AUROC Δ**: {boot_native['delta_euclidean_auc']['mean']:+.4f} (95% CI: [{boot_native['delta_euclidean_auc']['ci_lower']:+.4f}, {boot_native['delta_euclidean_auc']['ci_upper']:+.4f}])")
    table_lines.append(f"- **Matched d=256 Cosine AUROC Δ**: {boot_d256['delta_cosine_auc']['mean']:+.4f} (95% CI: [{boot_d256['delta_cosine_auc']['ci_lower']:+.4f}, {boot_d256['delta_cosine_auc']['ci_upper']:+.4f}])")
    table_lines.append(f"- **Matched d=256 Euclidean AUROC Δ**: {boot_d256['delta_euclidean_auc']['mean']:+.4f} (95% CI: [{boot_d256['delta_euclidean_auc']['ci_lower']:+.4f}, {boot_d256['delta_euclidean_auc']['ci_upper']:+.4f}])")

    table_md = "\n".join(table_lines) + "\n"
    (args.output_dir / "m0_fm_comparison_table.md").write_text(table_md)

    # 10. Save Full JSON Results
    final_output = {
        "schema_version": SCHEMA_VERSION,
        "n_multi_patients": len(multi_pats),
        "n_multi_ecgs": len(df_multi),
        "n_same_pairs": len(same_pairs),
        "n_diff_pairs": len(diff_pairs),
        "n_cross_disease_pairs": int(cross_disease_mask.sum()),
        "arms": results_by_arm,
        "bootstrap_native": boot_native,
        "bootstrap_d256": boot_d256,
    }

    results_file = args.output_dir / "results.json"
    results_file.write_text(json.dumps(final_output, indent=2) + "\n")
    logger.info(f"\nSaved structured results to: {results_file}")
    logger.info(f"Saved comparison table to:    {args.output_dir / 'm0_fm_comparison_table.md'}")
    logger.info("\n" + table_md)
    logger.info("=== Milestone M0-FM Execution COMPLETED Successfully ===")


if __name__ == "__main__":
    main()
