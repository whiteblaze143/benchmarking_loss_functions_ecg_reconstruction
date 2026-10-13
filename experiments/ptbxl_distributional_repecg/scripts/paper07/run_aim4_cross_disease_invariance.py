#!/usr/bin/env python3
"""Aim 4: Cross-Disease Identity Invariance Experiment (SetOperator E6).

Tests Claim C3 from the PRD:
"Does the patient reference vector s_p remain patient-specific across diagnostic shifts,
proving that the representation does not collapse into a simple disease signature?"

Evaluates:
1. Verification AUROC under Static Disease (Same-patient y1==y2 vs Diff-patient y1==y2)
2. Verification AUROC across Disease Transition (Same-patient y1!=y2 vs Diff-patient y1!=y2)
3. Cosine similarity and Euclidean distance distributions across all pair strata
4. Per-transition breakdown:
   - NORM -> DISEASE (new-onset pathology)
   - DISEASE -> NORM (clinical resolution)
   - SR -> AF (new-onset arrhythmia)
   - AF -> SR (arrhythmia conversion)
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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch

from repecg.paper07_operator.model import OperatorSetModel


SCHEMA_VERSION = "prism_ecg_aim4_cross_disease_invariance_v1"
EXPECTED_CHECKPOINT_SHA256 = "1071470f58e63594341a7a5df589b3db50c56b86befa65cb47a0b1da2dccbf68"
Q8_INDICES = (0, 1, 6, 7, 8, 9, 10, 11)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("aim4_invariance")
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
    return "OTHER"


def get_superclass(scp_str: str, diag_map: dict[str, str]) -> str:
    if not isinstance(scp_str, str):
        return "UNKNOWN"
    try:
        d = ast.literal_eval(scp_str) if isinstance(scp_str, str) else scp_str
        classes = {diag_map[code] for code in d.keys() if code in diag_map}
        if "NORM" in classes and len(classes) == 1:
            return "NORM"
        if any(c in ["MI", "STTC", "CD", "HYP"] for c in classes):
            return "DISEASE"
    except Exception:
        pass
    return "OTHER"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aim 4: Cross-Disease Identity Invariance Experiment")
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
        "--scp-statements",
        type=Path,
        default=Path("data/ptbxl/scp_statements.csv"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_aim4_cross_disease_invariance"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-impostors-per-pair", type=int, default=10)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output / "run.log")

    logger.info("======================================================================")
    logger.info("=== PRISM-ECG Aim 4: Cross-Disease Identity Invariance Experiment ===")
    logger.info("======================================================================")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output directory: {args.output}")

    # 1. Verify Checkpoint
    actual_hash = sha256_file(args.checkpoint)
    logger.info(f"Checkpoint SHA256: {actual_hash}")
    if actual_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Checkpoint hash mismatch!")

    # 2. Extract or Load Cached Latents
    cache_file = args.cache_dir / "setoperator_e6_all_z.npz"
    if cache_file.exists():
        logger.info(f"Loading cached latents from {cache_file}...")
        npz = np.load(cache_file)
        all_z = npz["all_z"]
        all_ecg = npz["all_ecg"]
        all_pat = npz["all_pat"]
    else:
        logger.info("Extracting SetOperator E6 latents from scratch...")
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

        train_z = extract_e6_latents(model, operators, train_res, device=device)
        val_z = extract_e6_latents(model, operators, val_res, device=device)

        all_ecg = np.concatenate([train_ecg, val_ecg])
        all_pat = np.concatenate([train_pat, val_pat])
        all_z = np.concatenate([train_z, val_z], axis=0)

        # Cache for fast reuse
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, all_z=all_z, all_ecg=all_ecg, all_pat=all_pat)
        logger.info(f"Saved cached latents to {cache_file}")

    ecg_to_idx = {int(ecg): i for i, ecg in enumerate(all_ecg)}
    logger.info(f"Total encoded ECG representations: {len(all_z):,} (Dim: {all_z.shape[1]})")

    # Normalize latents for cosine similarity
    z_norm = all_z / (np.linalg.norm(all_z, axis=1, keepdims=True) + 1e-8)

    # 3. Load Metadata & Class Annotations
    scp_df = pd.read_csv(args.scp_statements, index_col=0)
    diag_map = scp_df["diagnostic_class"].dropna().to_dict()

    df_meta = pd.read_csv(args.ptbxl_meta, index_col="ecg_id")
    df_meta["ecg_id"] = df_meta.index.values
    df_meta["recording_date"] = pd.to_datetime(df_meta["recording_date"])
    df_meta["scp_codes_dict"] = df_meta["scp_codes"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_meta["clean_rhythm"] = df_meta["scp_codes_dict"].apply(get_clean_rhythm)
    df_meta["clean_class"] = df_meta["scp_codes"].apply(lambda s: get_superclass(s, diag_map))

    # 4. Form Evaluation Pairs
    rng = np.random.RandomState(args.seed)

    same_pt_static_pairs = []       # y1 == y2 (same disease class)
    same_pt_transition_pairs = []   # y1 != y2 (disease transition)
    same_pt_rhythm_trans_pairs = [] # rhythm transition (AF<->SR)

    for pid, group in df_meta.groupby("patient_id", sort=False):
        if len(group) < 2:
            continue
        g_recs = group.sort_values("recording_date").to_dict("records")
        for i in range(len(g_recs)):
            for j in range(i + 1, len(g_recs)):
                e1, e2 = int(g_recs[i]["ecg_id"]), int(g_recs[j]["ecg_id"])
                if e1 not in ecg_to_idx or e2 not in ecg_to_idx:
                    continue
                c1, c2 = g_recs[i]["clean_class"], g_recs[j]["clean_class"]
                r1, r2 = g_recs[i]["clean_rhythm"], g_recs[j]["clean_rhythm"]
                dt = (g_recs[j]["recording_date"] - g_recs[i]["recording_date"]).total_seconds() / 86400.0

                pair_info = {
                    "patient_id": int(pid),
                    "idx1": ecg_to_idx[e1],
                    "idx2": ecg_to_idx[e2],
                    "c1": c1, "c2": c2,
                    "r1": r1, "r2": r2,
                    "dt_days": dt,
                }

                if c1 in ["NORM", "DISEASE"] and c2 in ["NORM", "DISEASE"]:
                    if c1 == c2:
                        same_pt_static_pairs.append(pair_info)
                    else:
                        same_pt_transition_pairs.append(pair_info)

                if r1 in ["AF", "SR"] and r2 in ["AF", "SR"] and r1 != r2:
                    same_pt_rhythm_trans_pairs.append(pair_info)

    logger.info(f"Same-Patient Pairs Formed:")
    logger.info(f"  - Static Disease Class (y1 == y2):       {len(same_pt_static_pairs):,}")
    logger.info(f"  - Disease Transition (NORM <-> DISEASE): {len(same_pt_transition_pairs):,}")
    logger.info(f"  - Rhythm Transition (AF <-> SR):         {len(same_pt_rhythm_trans_pairs):,}")

    # Build Verification Datasets with Nuisance-Matched Impostor Pairs
    def evaluate_verification(same_pairs: list[dict], match_disease: bool = True) -> dict:
        y_true = []
        scores_cos = []
        scores_euc = []

        # Positive pairs
        for p in same_pairs:
            z1 = z_norm[p["idx1"]]
            z2 = z_norm[p["idx2"]]
            cos_sim = float(np.dot(z1, z2))
            euc_dist = float(np.linalg.norm(all_z[p["idx1"]] - all_z[p["idx2"]]))

            y_true.append(1)
            scores_cos.append(cos_sim)
            scores_euc.append(-euc_dist)

        all_indices = np.arange(len(all_z))
        class_to_indices = {}
        for idx, row in df_meta.iterrows():
            ecg = int(row["ecg_id"])
            if ecg in ecg_to_idx:
                c = row["clean_class"]
                class_to_indices.setdefault(c, []).append(ecg_to_idx[ecg])

        for p in same_pairs:
            pid = p["patient_id"]
            if match_disease and p["c2"] in class_to_indices:
                cand_pool = class_to_indices[p["c2"]]
            else:
                cand_pool = all_indices

            for _ in range(args.n_impostors_per_pair):
                rand_idx = rng.choice(cand_pool)
                while all_pat[rand_idx] == pid:
                    rand_idx = rng.choice(cand_pool)

                z1 = z_norm[p["idx1"]]
                z_imp = z_norm[rand_idx]
                cos_sim = float(np.dot(z1, z_imp))
                euc_dist = float(np.linalg.norm(all_z[p["idx1"]] - all_z[rand_idx]))

                y_true.append(0)
                scores_cos.append(cos_sim)
                scores_euc.append(-euc_dist)

        y_true = np.array(y_true)
        scores_cos = np.array(scores_cos)
        scores_euc = np.array(scores_euc)

        auc_cos = fast_auroc(y_true, scores_cos)
        auc_euc = fast_auroc(y_true, scores_euc)

        pos_cos = scores_cos[y_true == 1]
        neg_cos = scores_cos[y_true == 0]

        return {
            "n_positive": int(y_true.sum()),
            "n_negative": int(len(y_true) - y_true.sum()),
            "auroc_cosine": float(auc_cos),
            "auroc_euclidean": float(auc_euc),
            "pos_cos_mean": float(np.mean(pos_cos)),
            "pos_cos_std": float(np.std(pos_cos)),
            "neg_cos_mean": float(np.mean(neg_cos)),
            "neg_cos_std": float(np.std(neg_cos)),
            "cosine_separation": float(np.mean(pos_cos) - np.mean(neg_cos)),
        }

    logger.info("\nEvaluating Identity Invariance Across Clinical Strata...")

    res_static = evaluate_verification(same_pt_static_pairs, match_disease=True)
    logger.info(f"1. Static Disease (y1 == y2):")
    logger.info(f"   AUROC (Cosine): {res_static['auroc_cosine']:.4f} | AUROC (Euclidean): {res_static['auroc_euclidean']:.4f}")
    logger.info(f"   Pos Cos Sim: {res_static['pos_cos_mean']:.3f} +/- {res_static['pos_cos_std']:.3f} vs Impostor: {res_static['neg_cos_mean']:.3f} +/- {res_static['neg_cos_std']:.3f} (Delta={res_static['cosine_separation']:+.3f})")

    res_disease_trans = evaluate_verification(same_pt_transition_pairs, match_disease=True)
    logger.info(f"2. Disease Transition (NORM <-> DISEASE):")
    logger.info(f"   AUROC (Cosine): {res_disease_trans['auroc_cosine']:.4f} | AUROC (Euclidean): {res_disease_trans['auroc_euclidean']:.4f}")
    logger.info(f"   Pos Cos Sim: {res_disease_trans['pos_cos_mean']:.3f} +/- {res_disease_trans['pos_cos_std']:.3f} vs Impostor: {res_disease_trans['neg_cos_mean']:.3f} +/- {res_disease_trans['neg_cos_std']:.3f} (Delta={res_disease_trans['cosine_separation']:+.3f})")

    res_rhythm_trans = evaluate_verification(same_pt_rhythm_trans_pairs, match_disease=False)
    logger.info(f"3. Rhythm Transition (AF <-> SR):")
    logger.info(f"   AUROC (Cosine): {res_rhythm_trans['auroc_cosine']:.4f} | AUROC (Euclidean): {res_rhythm_trans['auroc_euclidean']:.4f}")
    logger.info(f"   Pos Cos Sim: {res_rhythm_trans['pos_cos_mean']:.3f} +/- {res_rhythm_trans['pos_cos_std']:.3f} vs Impostor: {res_rhythm_trans['neg_cos_mean']:.3f} +/- {res_rhythm_trans['neg_cos_std']:.3f} (Delta={res_rhythm_trans['cosine_separation']:+.3f})")

    retention_ratio = res_disease_trans["cosine_separation"] / res_static["cosine_separation"]
    logger.info(f"\nIdentity Retention Across Disease Transition: {retention_ratio*100:.1f}%")

    # 5. Plot Comparison Figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    categories = ["Static Disease\n(y1 == y2)", "Disease Transition\n(NORM <-> DISEASE)", "Rhythm Transition\n(AF <-> SR)"]
    aurocs = [res_static["auroc_cosine"], res_disease_trans["auroc_cosine"], res_rhythm_trans["auroc_cosine"]]
    colors = ["#2B6CB0", "#319795", "#805AD5"]

    bars = axes[0].bar(categories, aurocs, color=colors, alpha=0.85, width=0.55, edgecolor="black", linewidth=1.2)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel("Pairwise Patient Verification AUROC", fontsize=11, fontweight="bold")
    axes[0].set_title("(A) Patient Identity Preservation\nAcross Clinical Transitions", fontsize=12, fontweight="bold")
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Random Guess (0.50)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    for b in bars:
        h = b.get_height()
        axes[0].annotate(f"{h:.4f}",
                         xy=(b.get_x() + b.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontsize=11, fontweight="bold")

    bins = np.linspace(-0.2, 1.0, 50)
    axes[1].hist(
        [np.dot(z_norm[p["idx1"]], z_norm[p["idx2"]]) for p in same_pt_static_pairs],
        bins=bins, density=True, alpha=0.5, color="#2B6CB0", label=f"Same Pt (Static, μ={res_static['pos_cos_mean']:.2f})"
    )
    axes[1].hist(
        [np.dot(z_norm[p["idx1"]], z_norm[p["idx2"]]) for p in same_pt_transition_pairs],
        bins=bins, density=True, alpha=0.5, color="#319795", label=f"Same Pt (Disease Trans, μ={res_disease_trans['pos_cos_mean']:.2f})"
    )
    axes[1].axvline(res_static['neg_cos_mean'], color="#E53E3E", linestyle="--", linewidth=2, label=f"Impostor Mean (μ={res_static['neg_cos_mean']:.2f})")

    axes[1].set_xlabel("Cosine Similarity", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Density", fontsize=11, fontweight="bold")
    axes[1].set_title("(B) Latent Alignment Distribution\n(Same Patient vs Impostor)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper left", fontsize=9, frameon=True)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plot_file = args.output / "aim4_identity_invariance_comparison.png"
    plt.savefig(plot_file)
    logger.info(f"Saved publication plot to {plot_file}")

    results_payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(args.checkpoint),
        "static_disease": res_static,
        "disease_transition": res_disease_trans,
        "rhythm_transition": res_rhythm_trans,
        "identity_retention_ratio": float(retention_ratio),
    }
    out_file = args.output / "results.json"
    out_file.write_text(json.dumps(results_payload, indent=2) + "\n")
    logger.info(f"Saved complete results JSON to {out_file}")
    logger.info("=== Aim 4 Cross-Disease Identity Invariance Experiment COMPLETED Successfully ===")


if __name__ == "__main__":
    main()
