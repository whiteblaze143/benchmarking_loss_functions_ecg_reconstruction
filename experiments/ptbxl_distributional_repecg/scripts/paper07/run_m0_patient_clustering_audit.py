#!/usr/bin/env python3
"""Milestone M0 / Aim 1: Patient Representation Clustering & Reference Audit.

Rigorous evaluation of patient-specific identity structure in the frozen SetOperator E6 latent:
1. Patient Verification AUROC: Same-patient vs different-patient pair discrimination (Cosine & Euclidean).
2. Closed-Set Patient Identification (Top-1, Top-5, Top-10 Retrieval Accuracy).
3. Patient vs Disease Variance Decomposition (ANOVA / R^2 explained by Patient vs Superclass).
4. SetOperator Reference Frame Quality: Empirical Mean baseline vs SetOperator baseline.
5. Time-Staleness Stratification: Verification performance across short (<30d), medium (30-180d), and long (>180d) gaps.
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

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import silhouette_score
import torch

from repecg.paper07_operator.model import OperatorSetModel


SCHEMA_VERSION = "prism_ecg_stage_m0_patient_clustering_v1"
EXPECTED_CHECKPOINT_SHA256 = "1071470f58e63594341a7a5df589b3db50c56b86befa65cb47a0b1da2dccbf68"
Q8_INDICES = (0, 1, 6, 7, 8, 9, 10, 11)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("m0_clustering")
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


def compute_setoperator_reference(
    model: OperatorSetModel,
    operators: np.ndarray,
    history_responses: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Pass multiple observations through the SetOperator to construct personal reference baseline b_p.
    
    If history has K items, treats the union of operators and responses across all K observations
    as a set input to the permutation-invariant model.
    """
    model.eval()
    K = len(history_responses)
    # history_responses: (K, 8, L)
    # Replicate operators K times: (K*8, 8)
    sub_ops = operators[list(Q8_INDICES)] # (8, 8)
    stacked_ops = np.tile(sub_ops, (K, 1)) # (K*8, 8)
    stacked_res = history_responses[:, list(Q8_INDICES), :].reshape(K * len(Q8_INDICES), -1) # (K*8, L)

    q = torch.as_tensor(stacked_ops, dtype=torch.float32, device=device).unsqueeze(0) # (1, K*8, 8)
    r = torch.as_tensor(stacked_res, dtype=torch.float32, device=device).unsqueeze(0) # (1, K*8, L)

    with torch.inference_mode():
        b_p = model.encode_context(q, r) # (1, 256)
    return b_p.cpu().numpy().squeeze(0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Milestone M0: Patient Representation Clustering & Reference Audit")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/ptbxl_distributional_repecg/outputs/configuration_shift_evaluations/ptbxl/checkpoints/P07_ROBUSTNESS_TRAINED.pt"),
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
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_m0_patient_clustering"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-impostors-per-pair", type=int, default=10)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output / "run.log")

    logger.info("======================================================================")
    logger.info("=== PRISM-ECG Stage M0: Patient Representation Clustering Audit ===")
    logger.info("======================================================================")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Contexts dir: {args.contexts}")
    logger.info(f"Metadata file: {args.ptbxl_meta}")
    logger.info(f"Output directory: {args.output}")

    # 1. Checkpoint Verification
    t0 = time.time()
    logger.info("Verifying checkpoint SHA256...")
    actual_hash = sha256_file(args.checkpoint)
    logger.info(f"Checkpoint SHA256: {actual_hash}")
    if actual_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Checkpoint hash mismatch! Expected {EXPECTED_CHECKPOINT_SHA256}, got {actual_hash}")
    logger.info("Checkpoint checksum verified.")

    # 2. Extract Latents
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")
    torch.backends.cudnn.enabled = False

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = OperatorSetModel(operator_mode="continuous", response_mode="phase_cnn").to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    logger.info("Loading precomputed operators and responses...")
    operators = np.load(args.contexts / "operators.npy")
    train_res = np.load(args.contexts / "train_responses.npy", mmap_mode="r")
    train_ecg = np.load(args.contexts / "train_ecg_ids.npy")
    train_pat = np.load(args.contexts / "train_patient_ids.npy")

    val_res = np.load(args.contexts / "validation_responses.npy", mmap_mode="r")
    val_ecg = np.load(args.contexts / "validation_ecg_ids.npy")
    val_pat = np.load(args.contexts / "validation_patient_ids.npy")

    logger.info(f"Extracting latents (Train: {len(train_ecg)}, Val: {len(val_ecg)})...")
    train_z = extract_e6_latents(model, operators, train_res, device=device)
    val_z = extract_e6_latents(model, operators, val_res, device=device)

    all_ecg = np.concatenate([train_ecg, val_ecg])
    all_pat = np.concatenate([train_pat, val_pat])
    all_z = np.concatenate([train_z, val_z], axis=0)
    logger.info(f"Total extracted latents: {all_z.shape} in {time.time() - t0:.2f}s")

    # 3. Load & Match Metadata
    logger.info("Loading metadata...")
    df_meta = pd.read_csv(args.ptbxl_meta, index_col="ecg_id")
    df_meta["recording_date"] = pd.to_datetime(df_meta["recording_date"])
    df_meta = df_meta.loc[all_ecg].copy()
    df_meta["array_idx"] = np.arange(len(all_ecg))

    # Parse diagnostic superclass
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

    # 4. Filter to Patients with Multiple Visits
    pat_counts = df_meta["patient_id"].value_counts()
    multi_pats = pat_counts[pat_counts >= 2].index.values
    logger.info(f"Total patients: {len(pat_counts):,}, Patients with >=2 ECGs: {len(multi_pats):,}")

    df_multi = df_meta[df_meta["patient_id"].isin(multi_pats)].sort_values(["patient_id", "recording_date"])
    logger.info(f"Total multi-ECG records: {len(df_multi):,}")

    # 5. Part A: Pairwise Patient Verification (Same vs Different)
    logger.info("\n--- Part A: Pairwise Verification AUROC (Same vs Impostor) ---")
    rng = np.random.RandomState(args.seed)

    same_pairs = []
    diff_pairs = []

    # Construct all consecutive same-patient pairs
    for pid, group in df_multi.groupby("patient_id"):
        recs = group.to_dict("records")
        for i in range(len(recs) - 1):
            t1 = recs[i]
            t2 = recs[i + 1]
            dt_days = (t2["recording_date"] - t1["recording_date"]).total_seconds() / 86400.0
            same_class = (t1["diagnostic_superclass"] == t2["diagnostic_superclass"])
            same_pairs.append({
                "patient_id": pid,
                "idx1": t1["array_idx"],
                "idx2": t2["array_idx"],
                "dt_days": dt_days,
                "same_class": same_class,
                "class1": t1["diagnostic_superclass"],
                "class2": t2["diagnostic_superclass"],
            })

    logger.info(f"Formed {len(same_pairs):,} same-patient longitudinal pairs.")

    # Construct matched impostor pairs (10 per same-patient pair)
    all_multi_indices = df_multi["array_idx"].values
    all_multi_pats = df_multi["patient_id"].values
    pat_to_indices = {pid: df_multi[df_multi["patient_id"] == pid]["array_idx"].values for pid in multi_pats}

    for sp in same_pairs:
        idx1 = sp["idx1"]
        pid1 = sp["patient_id"]
        # Sample impostors from different patients
        impostors = []
        while len(impostors) < args.n_impostors_per_pair:
            cand_idx = rng.choice(all_multi_indices)
            cand_pid = df_meta.iloc[cand_idx]["patient_id"]
            if cand_pid != pid1 and cand_idx not in impostors:
                impostors.append(cand_idx)
        for imp_idx in impostors:
            diff_pairs.append({
                "idx1": idx1,
                "idx2": imp_idx,
            })

    logger.info(f"Formed {len(diff_pairs):,} different-patient (impostor) pairs.")

    # Compute distances under:
    # 1. Raw z
    # 2. Population Mean-Centered z
    # 3. L2 Normalized z
    pop_mean = np.mean(all_z, axis=0, keepdims=True)
    z_centered = all_z - pop_mean
    z_normed = all_z / (np.linalg.norm(all_z, axis=1, keepdims=True) + 1e-12)

    def eval_pair_distances(z_array, name):
        same_cos = []
        same_euc = []
        for p in same_pairs:
            v1 = z_array[p["idx1"]]
            v2 = z_array[p["idx2"]]
            c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
            e = float(np.linalg.norm(v1 - v2))
            same_cos.append(c)
            same_euc.append(e)

        diff_cos = []
        diff_euc = []
        for p in diff_pairs:
            v1 = z_array[p["idx1"]]
            v2 = z_array[p["idx2"]]
            c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
            e = float(np.linalg.norm(v1 - v2))
            diff_cos.append(c)
            diff_euc.append(e)

        y_true = np.concatenate([np.ones(len(same_cos)), np.zeros(len(diff_cos))])
        scores_cos = np.concatenate([same_cos, diff_cos])
        scores_euc = -np.concatenate([same_euc, diff_euc]) # Negative because smaller distance = higher similarity

        auc_cos = fast_auroc(y_true, scores_cos)
        auc_euc = fast_auroc(y_true, scores_euc)

        logger.info(f"[{name}] Pairwise Verification AUROC:")
        logger.info(f"   Cosine Similarity AUROC: {auc_cos:.4f} (Same median={np.median(same_cos):.4f}, Diff median={np.median(diff_cos):.4f})")
        logger.info(f"   Euclidean Distance AUROC: {auc_euc:.4f} (Same median={np.median(same_euc):.4f}, Diff median={np.median(diff_euc):.4f})")
        return {
            "auroc_cosine": auc_cos,
            "auroc_euclidean": auc_euc,
            "same_cos_median": float(np.median(same_cos)),
            "diff_cos_median": float(np.median(diff_cos)),
            "same_euc_median": float(np.median(same_euc)),
            "diff_euc_median": float(np.median(diff_euc)),
        }

    res_raw = eval_pair_distances(all_z, "Raw Latent z_t")
    res_centered = eval_pair_distances(z_centered, "Population Mean-Centered z_t")
    res_normed = eval_pair_distances(z_normed, "Unit L2-Normalized z_t")

    # 6. Part B: Identification / Nearest Neighbor Retrieval
    logger.info("\n--- Part B: Patient Identification (Gallery vs Query Retrieval) ---")
    # For every multi-visit patient: earliest observation = gallery, latest observation = query
    gallery_indices = []
    query_indices = []
    gallery_pats = []
    query_pats = []

    for pid, group in df_multi.groupby("patient_id"):
        recs = group.to_dict("records")
        gallery_indices.append(recs[0]["array_idx"])
        gallery_pats.append(pid)
        query_indices.append(recs[-1]["array_idx"])
        query_pats.append(pid)

    gallery_z = z_normed[gallery_indices] # (N_pats, 256)
    query_z = z_normed[query_indices] # (N_pats, 256)
    gallery_pats = np.array(gallery_pats)
    query_pats = np.array(query_pats)

    sim_matrix = query_z @ gallery_z.T # (N_pats, N_pats)
    ranks = []
    for i in range(len(query_pats)):
        sorted_indices = np.argsort(-sim_matrix[i])
        rank = int(np.where(sorted_indices == i)[0][0]) + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    top1 = float(np.mean(ranks == 1))
    top5 = float(np.mean(ranks <= 5))
    top10 = float(np.mean(ranks <= 10))
    median_rank = float(np.median(ranks))
    random_top1 = 1.0 / len(query_pats)

    logger.info(f"Patient Identification (N = {len(query_pats):,} candidate patients):")
    logger.info(f"   Top-1 Accuracy:  {top1*100:.2f}% (Chance level: {random_top1*100:.3f}%)")
    logger.info(f"   Top-5 Accuracy:  {top5*100:.2f}%")
    logger.info(f"   Top-10 Accuracy: {top10*100:.2f}%")
    logger.info(f"   Median Rank:     {median_rank:.1f} / {len(query_pats)}")

    # 7. Part C: Variance Decomposition (Patient Identity vs Disease Superclass)
    logger.info("\n--- Part C: Variance Decomposition (Patient vs Disease R^2) ---")
    # Across multi-patient ECGs: compute between-patient vs within-patient sum of squares
    df_eval = df_multi.copy()
    z_eval = all_z[df_eval["array_idx"].values]

    # Overall variance
    total_ss = np.sum((z_eval - np.mean(z_eval, axis=0)) ** 2)

    # Patient explained variance
    pat_means = df_eval.groupby("patient_id")["array_idx"].apply(lambda idxs: np.mean(all_z[idxs], axis=0))
    pat_ss = sum(len(group) * np.sum((pat_means[pid] - np.mean(z_eval, axis=0)) ** 2) for pid, group in df_eval.groupby("patient_id"))
    r2_patient = float(pat_ss / total_ss)

    # Disease explained variance
    class_means = df_eval.groupby("diagnostic_superclass")["array_idx"].apply(lambda idxs: np.mean(all_z[idxs], axis=0))
    class_ss = sum(len(group) * np.sum((class_means[cls] - np.mean(z_eval, axis=0)) ** 2) for cls, group in df_eval.groupby("diagnostic_superclass"))
    r2_disease = float(class_ss / total_ss)

    logger.info(f"Latent Variance Decomposition (Total SS = {total_ss:.2e}):")
    logger.info(f"   Patient Identity R^2: {r2_patient*100:.2f}% of latent variance")
    logger.info(f"   Disease Category R^2: {r2_disease*100:.2f}% of latent variance")
    logger.info(f"   Patient / Disease Ratio: {r2_patient / max(r2_disease, 1e-6):.2f}x")

    # 8. Part D: Time-Staleness Breakdown
    logger.info("\n--- Part D: Verification Robustness across Time Gaps ---")
    same_df = pd.DataFrame(same_pairs)
    time_bins = [
        ("< 7 days", same_df["dt_days"] < 7),
        ("7 - 30 days", (same_df["dt_days"] >= 7) & (same_df["dt_days"] < 30)),
        ("30 - 180 days", (same_df["dt_days"] >= 30) & (same_df["dt_days"] < 180)),
        ("> 180 days", same_df["dt_days"] >= 180),
    ]

    time_breakdown = {}
    for label, mask in time_bins:
        n_pairs = int(mask.sum())
        if n_pairs == 0:
            continue
        sub_same = same_df[mask]
        sub_same_cos = [float(np.dot(z_normed[p["idx1"]], z_normed[p["idx2"]])) for _, p in sub_same.iterrows()]
        
        # Sample impostor scores of matching sample size
        sub_diff_cos = [float(np.dot(z_normed[p["idx1"]], z_normed[p["idx2"]])) for p in diff_pairs[:len(sub_same_cos) * 10]]
        
        y_sub = np.concatenate([np.ones(len(sub_same_cos)), np.zeros(len(sub_diff_cos))])
        s_sub = np.concatenate([sub_same_cos, sub_diff_cos])
        auc_sub = fast_auroc(y_sub, s_sub)
        time_breakdown[label] = {
            "pairs": n_pairs,
            "auroc": float(auc_sub),
            "median_cos": float(np.median(sub_same_cos)),
        }
        logger.info(f"   Time Gap {label:15s}: N={n_pairs:4d} pairs | AUROC = {auc_sub:.4f} | Median Cosine = {np.median(sub_same_cos):.4f}")

    # 9. Part E: Disease Transition Robustness (Same Patient, Different Disease vs Impostor)
    logger.info("\n--- Part E: Disease Transition Invariance (Cross-Disease Same-Patient Pairs) ---")
    cross_disease = same_df[~same_df["same_class"]]
    same_disease = same_df[same_df["same_class"]]

    cd_cos = [float(np.dot(z_normed[p["idx1"]], z_normed[p["idx2"]])) for _, p in cross_disease.iterrows()]
    sd_cos = [float(np.dot(z_normed[p["idx1"]], z_normed[p["idx2"]])) for _, p in same_disease.iterrows()]
    diff_sample = [float(np.dot(z_normed[p["idx1"]], z_normed[p["idx2"]])) for p in diff_pairs[:len(cd_cos) * 10]]

    y_cd = np.concatenate([np.ones(len(cd_cos)), np.zeros(len(diff_sample))])
    s_cd = np.concatenate([cd_cos, diff_sample])
    auc_cd = fast_auroc(y_cd, s_cd)

    logger.info(f"   Same Patient, Same Disease:      N={len(same_disease):4d} | Median Cosine = {np.median(sd_cos):.4f}")
    logger.info(f"   Same Patient, DIFFERENT Disease: N={len(cross_disease):4d} | Median Cosine = {np.median(cd_cos):.4f} | Verification AUROC = {auc_cd:.4f}")
    logger.info(f"   Different Patient (Impostor):    N={len(diff_sample):4d} | Median Cosine = {np.median(diff_sample):.4f}")

    # Save summary
    results = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(args.checkpoint),
        "checksum": actual_hash,
        "n_multi_patients": len(multi_pats),
        "n_multi_ecgs": len(df_multi),
        "n_same_pairs": len(same_pairs),
        "n_diff_pairs": len(diff_pairs),
        "verification_raw": res_raw,
        "verification_centered": res_centered,
        "verification_normed": res_normed,
        "identification": {
            "n_candidates": len(query_pats),
            "top1": top1,
            "top5": top5,
            "top10": top10,
            "median_rank": median_rank,
            "random_top1": random_top1,
        },
        "variance_decomposition": {
            "r2_patient": r2_patient,
            "r2_disease": r2_disease,
            "ratio": r2_patient / max(r2_disease, 1e-6),
        },
        "time_breakdown": time_breakdown,
        "cross_disease_verification": {
            "n_cross_disease_pairs": len(cross_disease),
            "median_same_disease_cos": float(np.median(sd_cos)),
            "median_cross_disease_cos": float(np.median(cd_cos)),
            "median_impostor_cos": float(np.median(diff_sample)),
            "auroc_cross_disease": float(auc_cd),
        },
        "elapsed_seconds": time.time() - t0,
    }

    results_file = args.output / "results.json"
    results_file.write_text(json.dumps(results, indent=2) + "\n")
    logger.info(f"\nSaved results to {results_file}")
    logger.info("=== Milestone M0 Audit COMPLETED Successfully ===")


if __name__ == "__main__":
    main()
