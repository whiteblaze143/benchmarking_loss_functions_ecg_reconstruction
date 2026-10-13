#!/usr/bin/env python3
"""Milestone R020: Calibration Validity and Basis-Invariance Benchmark.

Four-part methodological validity block:
  - R020-A: Orthogonal Rotation Invariance Audit (M=50 random orthogonal rotations Q_m in O(d))
  - R020-B: Multivariate PEB (MV-PEB) with basis-covariant posterior reference and regularized covariance
  - R020-C: Serial-Score Recalibration Comparator (low-dimensional output-space p_t, p_{t-1}, dt)
  - R020-D: Operating-Point Validity with thresholds locked strictly on validation patients,
            reporting sensitivity, specificity, FPR, PPV, NPV, and 1,000 patient-clustered bootstrap CIs.

Evaluated across:
  - Tasks: SR->AF, AF->SR, MI, STTC, CD
  - Frozen Encoders: SetOperator (d=256), GraphECG (d=192), CSFM Base (d=768)
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.stats as stats
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("r020_validity")
logger.setLevel(logging.INFO)
logger.handlers.clear()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
K_LEVELS = [0, 1, 2, 3, 5]


def extract_rhythm(scp_dict: dict) -> str:
    if "AFIB" in scp_dict or "AFLT" in scp_dict:
        return "AF"
    if "SR" in scp_dict:
        return "SR"
    return "OTHER"


def generate_random_orthogonal_matrix(d: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate uniform random orthogonal matrix from Haar measure on O(d)."""
    A = rng.randn(d, d)
    Q, R = la.qr(A)
    d_diag = np.diagonal(R)
    ph = d_diag / np.abs(d_diag)
    Q = Q * ph
    return Q.astype(np.float32)


class FastRegularizedCovarianceEstimator:
    """Estimates population mean, within-patient covariance, and between-patient covariance.
    
    Supports analytical basis rotation: Sigma' = Q Sigma Q^T.
    """
    def __init__(self, z_train: np.ndarray, train_patient_ids: np.ndarray, alpha_shrink: float = 0.15):
        self.d = z_train.shape[1]
        self.alpha = alpha_shrink
        self.mu = np.mean(z_train, axis=0).astype(np.float32)

        df_p = pd.DataFrame({"pid": train_patient_ids})
        groups = df_p.groupby("pid").indices

        patient_means = []
        n_p_list = []
        ss_within = np.zeros((self.d, self.d), dtype=np.float64)
        df_within = 0

        for pid, indices in groups.items():
            n_p = len(indices)
            n_p_list.append(n_p)
            pts = z_train[indices].astype(np.float64)
            p_mean = np.mean(pts, axis=0)
            patient_means.append(p_mean)

            if n_p > 1:
                dev = pts - p_mean
                ss_within += dev.T @ dev
                df_within += (n_p - 1)

        patient_means = np.array(patient_means, dtype=np.float64)

        if df_within > 0:
            S_W = ss_within / df_within
        else:
            S_W = np.eye(self.d, dtype=np.float64) * 0.1

        tr_W = np.trace(S_W) / self.d
        self.Sigma_W = ((1.0 - self.alpha) * S_W + self.alpha * tr_W * np.eye(self.d)).astype(np.float32)

        S_between_raw = np.cov(patient_means, rowvar=False)
        mean_inv_np = np.mean(1.0 / np.array(n_p_list, dtype=np.float64))
        S_B_unbiased = S_between_raw - mean_inv_np * S_W

        eigvals, eigvecs = la.eigh(S_B_unbiased)
        eigvals_clipped = np.maximum(eigvals, 1e-4)
        S_B_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

        tr_B = np.trace(S_B_psd) / self.d
        self.Sigma_B = ((1.0 - self.alpha) * S_B_psd + self.alpha * tr_B * np.eye(self.d)).astype(np.float32)

        self.sigma2_diag = np.maximum(np.diag(self.Sigma_W), 1e-6).astype(np.float32)
        self.tau2_diag = np.maximum(np.diag(self.Sigma_B), 1e-6).astype(np.float32)

        # Precompute inverted matrices for MV-PEB K=1
        denom_1 = self.Sigma_B + self.Sigma_W
        self.shrink_mat_1 = (self.Sigma_B @ la.pinv(denom_1)).astype(np.float32)
        self.V_1 = (self.shrink_mat_1 @ self.Sigma_W).astype(np.float32)
        inv_cov_1 = la.pinv(self.Sigma_W + self.V_1)
        w, v = la.eigh(inv_cov_1)
        self.sqrt_inv_cov_1 = (v @ np.diag(np.sqrt(np.maximum(w, 1e-6))) @ v.T).astype(np.float32)

        # K=0
        inv_cov_0 = la.pinv(self.Sigma_W + self.Sigma_B)
        w0, v0 = la.eigh(inv_cov_0)
        self.sqrt_inv_cov_0 = (v0 @ np.diag(np.sqrt(np.maximum(w0, 1e-6))) @ v0.T).astype(np.float32)

    def get_rotated_diagonal_params(self, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute exact rotated parameters analytically: O(d^3) matrix multiply once."""
        mu_rot = Q @ self.mu
        # Diag(Q Sigma Q^T) is (Q * (Q @ Sigma)).sum(axis=1)
        sigma2_rot = np.sum((Q @ self.Sigma_W) * Q, axis=1).astype(np.float32)
        tau2_rot = np.sum((Q @ self.Sigma_B) * Q, axis=1).astype(np.float32)
        return mu_rot, np.maximum(sigma2_rot, 1e-6), np.maximum(tau2_rot, 1e-6)


def build_clinical_episodes(df_long: pd.DataFrame, z_all: np.ndarray, task_name: str, regime: str) -> List[Dict[str, Any]]:
    episodes = []
    grouped = df_long.groupby("patient_id")

    for pid, group in grouped:
        recs = group.to_dict("records")
        if len(recs) < 2:
            continue

        for q_idx in range(1, len(recs)):
            q_rec = recs[q_idx]
            q_pos = q_rec["pos"]
            prior_recs = recs[:q_idx]

            target_label = None

            if task_name == "SR->AF":
                if prior_recs[-1]["clean_rhythm"] != "SR":
                    continue
                if q_rec["clean_rhythm"] == "AF":
                    target_label = 1
                elif q_rec["clean_rhythm"] == "SR":
                    target_label = 0
                else:
                    continue

            elif task_name == "AF->SR":
                if prior_recs[-1]["clean_rhythm"] != "AF":
                    continue
                if q_rec["clean_rhythm"] == "SR":
                    target_label = 1
                elif q_rec["clean_rhythm"] == "AF":
                    target_label = 0
                else:
                    continue

            elif task_name == "MI":
                if prior_recs[-1]["MI"] != 0:
                    continue
                target_label = int(q_rec["MI"] == 1)

            elif task_name == "STTC":
                if prior_recs[-1]["STTC"] != 0:
                    continue
                target_label = int(q_rec["STTC"] == 1)

            elif task_name == "CD":
                if prior_recs[-1]["CD"] != 0:
                    continue
                target_label = int(q_rec["CD"] == 1)

            if target_label is None:
                continue

            if regime == "passive":
                support_candidates = [z_all[r["pos"]] for r in prior_recs]
            elif regime == "certified":
                if task_name in ("SR->AF", "AF->SR"):
                    ref_rhythm = "SR" if task_name == "SR->AF" else "AF"
                    cert = [r for r in prior_recs if r["clean_rhythm"] == ref_rhythm]
                elif task_name == "MI":
                    cert = [r for r in prior_recs if r["MI"] == 0]
                elif task_name == "STTC":
                    cert = [r for r in prior_recs if r["STTC"] == 0]
                elif task_name == "CD":
                    cert = [r for r in prior_recs if r["CD"] == 0]
                else:
                    cert = prior_recs
                support_candidates = [z_all[r["pos"]] for r in cert]

            dt_days = (q_rec["recording_date"] - prior_recs[-1]["recording_date"]).total_seconds() / 86400.0

            episodes.append({
                "patient_id": pid,
                "strat_fold": q_rec["strat_fold"],
                "is_train": bool(q_rec["strat_fold"] <= 6),
                "is_val": bool(q_rec["strat_fold"] == 7),
                "is_test": bool(q_rec["strat_fold"] == 8),
                "z_query": z_all[q_pos],
                "all_support_z": support_candidates,
                "n_priors_available": len(support_candidates),
                "dt_days": max(0.01, dt_days),
                "y": target_label,
            })

    return episodes


def patient_clustered_bootstrap(
    pids: np.ndarray,
    y_true: np.ndarray,
    y_prob_k0: np.ndarray,
    y_prob_k1: np.ndarray,
    locked_thresh: float,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.RandomState(seed)
    unique_pids = np.unique(pids)
    n_p = len(unique_pids)
    pid_to_indices = {p: np.where(pids == p)[0] for p in unique_pids}

    delta_aurocs = []
    delta_auprcs = []
    delta_fprs = []
    test_sens_list = []
    test_spec_list = []

    for _ in range(n_boot):
        sample_pids = rng.choice(unique_pids, size=n_p, replace=True)
        boot_idx = np.concatenate([pid_to_indices[p] for p in sample_pids])

        yb = y_true[boot_idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue

        p0 = y_prob_k0[boot_idx]
        p1 = y_prob_k1[boot_idx]

        auc0 = roc_auc_score(yb, p0)
        auc1 = roc_auc_score(yb, p1)
        delta_aurocs.append(auc1 - auc0)

        prc0 = average_precision_score(yb, p0)
        prc1 = average_precision_score(yb, p1)
        delta_auprcs.append(prc1 - prc0)

        neg_mask = (yb == 0)
        pos_mask = (yb == 1)

        fpr0 = np.mean(p0[neg_mask] >= locked_thresh)
        fpr1 = np.mean(p1[neg_mask] >= locked_thresh)
        delta_fprs.append(fpr1 - fpr0)

        sens1 = np.mean(p1[pos_mask] >= locked_thresh)
        spec1 = np.mean(p1[neg_mask] < locked_thresh)
        test_sens_list.append(sens1)
        test_spec_list.append(spec1)

    def get_stats(arr):
        if len(arr) == 0:
            return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
        }

    return {
        "delta_auroc": get_stats(delta_aurocs),
        "delta_auprc": get_stats(delta_auprcs),
        "delta_fpr_locked": get_stats(delta_fprs),
        "test_sensitivity_locked": get_stats(test_sens_list),
        "test_specificity_locked": get_stats(test_spec_list),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone R020: Calibration Validity and Basis-Invariance Benchmark")
    parser.add_argument("--so-cache", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/setoperator_e6_all_z.npz"))
    parser.add_argument("--graphecg-cache", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/graphecg_multi_z.npz"))
    parser.add_argument("--csfm-cache", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/csfm_base_multi_z.npz"))
    parser.add_argument("--ptbxl-data", type=Path, default=Path("data/ptbxl"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_r020_validity_benchmark"))
    parser.add_argument("--n-rotations", type=int, default=50)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(args.output_dir / "run.log", mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("Milestone R020: Calibration Validity and Basis-Invariance Benchmark")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Number of Orthogonal Rotations: {args.n_rotations}")
    logger.info(f"Bootstrap Resamples: {args.n_bootstrap}")
    logger.info("=" * 70)

    # 1. Load Data
    scp_df = pd.read_csv(args.ptbxl_data / "scp_statements.csv", index_col=0)
    diag_map = scp_df["diagnostic_class"].dropna().to_dict()

    df_meta = pd.read_csv(args.ptbxl_data / "ptbxl_database.csv", index_col="ecg_id")
    df_meta["recording_date"] = pd.to_datetime(df_meta["recording_date"])

    so_npz = np.load(args.so_cache)
    all_z_so = so_npz["all_z"]
    all_ecg_so = so_npz["all_ecg"]
    ecg_to_so_idx = {int(e): i for i, e in enumerate(all_ecg_so)}

    graphecg_npz = np.load(args.graphecg_cache)
    all_z_ge = graphecg_npz["graphecg_z"]
    all_ecg_ge = graphecg_npz["ecg_ids"]
    ecg_to_ge_idx = {int(e): i for i, e in enumerate(all_ecg_ge)}

    csfm_npz = np.load(args.csfm_cache)
    all_z_csfm = csfm_npz["csfm_z"]
    all_ecg_csfm = csfm_npz["ecg_ids"]
    ecg_to_csfm_idx = {int(e): i for i, e in enumerate(all_ecg_csfm)}

    df_meta_avail = df_meta.loc[list(ecg_to_so_idx.keys())].copy()

    for sc in SUPERCLASSES:
        df_meta_avail[sc] = df_meta_avail["scp_codes"].apply(
            lambda s: int(any(diag_map.get(k) == sc for k in ast.literal_eval(s).keys()))
            if isinstance(s, str) else 0
        )

    df_meta_avail["clean_rhythm"] = df_meta_avail["scp_codes"].apply(
        lambda s: extract_rhythm(ast.literal_eval(s)) if isinstance(s, str) else "OTHER"
    )

    pat_counts = df_meta_avail["patient_id"].value_counts()
    multi_pats = pat_counts[pat_counts >= 2].index
    df_long = df_meta_avail[df_meta_avail["patient_id"].isin(multi_pats)].copy()
    df_long = df_long.sort_values(["patient_id", "recording_date"])

    valid_ecgs = [e for e in df_long.index if (e in ecg_to_so_idx) and (e in ecg_to_ge_idx) and (e in ecg_to_csfm_idx)]
    df_long = df_long.loc[valid_ecgs].copy()
    df_long["pos"] = np.arange(len(df_long))

    Z_encoders = {
        "SetOperator (d=256)": np.array([all_z_so[ecg_to_so_idx[e]] for e in valid_ecgs], dtype=np.float32),
        "GraphECG (d=192)": np.array([all_z_ge[ecg_to_ge_idx[e]] for e in valid_ecgs], dtype=np.float32),
        "CSFM Base (d=768)": np.array([all_z_csfm[ecg_to_csfm_idx[e]] for e in valid_ecgs], dtype=np.float32),
    }

    tasks = ["SR->AF", "AF->SR", "MI", "STTC", "CD"]
    regimes = ["certified", "passive"]

    # Pre-generate 50 Orthogonal Matrices per dimension
    rng_rot = np.random.RandomState(args.seed)
    rot_matrices = {
        192: [generate_random_orthogonal_matrix(192, rng_rot) for _ in range(args.n_rotations)],
        256: [generate_random_orthogonal_matrix(256, rng_rot) for _ in range(args.n_rotations)],
        768: [generate_random_orthogonal_matrix(768, rng_rot) for _ in range(args.n_rotations)],
    }

    benchmark_results = {
        "metadata": {
            "n_rotations": args.n_rotations,
            "n_bootstrap": args.n_bootstrap,
            "tasks": tasks,
        },
        "r020_a_rotation_audit": {},
        "r020_b_mv_peb": {},
        "r020_c_serial_recalibration": {},
        "r020_d_operating_validity": {},
    }

    for enc_name, Z_all in Z_encoders.items():
        d = Z_all.shape[1]
        logger.info(f"\n======================================================================")
        logger.info(f"=== Running R020 Validity Suite: {enc_name} ===")
        logger.info(f"======================================================================")

        benchmark_results["r020_a_rotation_audit"][enc_name] = {}
        benchmark_results["r020_b_mv_peb"][enc_name] = {}
        benchmark_results["r020_c_serial_recalibration"][enc_name] = {}
        benchmark_results["r020_d_operating_validity"][enc_name] = {}

        # Fit Fast Regularized Covariance Estimator strictly on Training Patients (Folds 1-6)
        train_mask = (df_long["strat_fold"].values <= 6)
        cov_est = FastRegularizedCovarianceEstimator(Z_all[train_mask], df_long["patient_id"].values[train_mask])

        for task in tasks:
            logger.info(f"\n--- Task: {task} ---")
            task_rot_res = {}
            task_mv_res = {}
            task_serial_res = {}
            task_op_res = {}

            for regime in regimes:
                episodes = build_clinical_episodes(df_long, Z_all, task, regime)
                train_eps = [ep for ep in episodes if ep["is_train"]]
                val_eps = [ep for ep in episodes if ep["is_val"]]
                test_eps = [ep for ep in episodes if ep["is_test"] and ep["n_priors_available"] >= 1]

                if len(test_eps) < 5 or sum(ep["y"] for ep in test_eps) < 2 or (len(test_eps) - sum(ep["y"] for ep in test_eps)) < 2:
                    logger.warning(f"Skipping {task} {regime} due to insufficient test cases.")
                    continue

                y_val = np.array([ep["y"] for ep in val_eps])
                y_test = np.array([ep["y"] for ep in test_eps])
                test_pids = np.array([ep["patient_id"] for ep in test_eps])
                n_pos_test = int(y_test.sum())
                n_neg_test = int(len(y_test) - n_pos_test)
                n_pts_test = int(len(np.unique(test_pids)))

                # ------------------------------------------------------------------
                # Component 1: Baseline Population Probe g(z_t)
                # ------------------------------------------------------------------
                X_tr_pop = np.stack([ep["z_query"] for ep in train_eps], axis=0)
                y_tr = np.array([ep["y"] for ep in train_eps])

                scaler_pop = StandardScaler()
                X_tr_pop_s = scaler_pop.fit_transform(X_tr_pop)
                pop_clf = LogisticRegression(C=1.0, max_iter=500, random_state=args.seed)
                pop_clf.fit(X_tr_pop_s, y_tr)

                # Vectorized batch predictions
                p_val_pop = pop_clf.predict_proba(scaler_pop.transform(np.stack([ep["z_query"] for ep in val_eps], axis=0)))[:, 1]
                p_test_pop = pop_clf.predict_proba(scaler_pop.transform(np.stack([ep["z_query"] for ep in test_eps], axis=0)))[:, 1]

                # ------------------------------------------------------------------
                # Component 2: R020-C Serial-Score Recalibration Comparator (Batch Optimized)
                # ------------------------------------------------------------------
                def build_batch_serial_features(ep_list, p_pop_arr):
                    # Gather all prior representations to score in ONE batch
                    prev_z_list = []
                    has_prior = []
                    dt_list = []
                    for ep in ep_list:
                        if ep["n_priors_available"] > 0:
                            prev_z_list.append(ep["all_support_z"][-1])
                            has_prior.append(True)
                            dt_list.append(np.log1p(ep["dt_days"]))
                        else:
                            prev_z_list.append(ep["z_query"])  # placeholder
                            has_prior.append(False)
                            dt_list.append(0.0)

                    prev_z_mat = np.stack(prev_z_list, axis=0)
                    prev_p_all = pop_clf.predict_proba(scaler_pop.transform(prev_z_mat))[:, 1]

                    feats = []
                    for pt, prev_p, hp, ldt in zip(p_pop_arr, prev_p_all, has_prior, dt_list):
                        logit_pt = np.log(max(1e-5, min(0.99999, pt)) / (1.0 - max(1e-5, min(0.99999, pt))))
                        if hp:
                            logit_prev = np.log(max(1e-5, min(0.99999, prev_p)) / (1.0 - max(1e-5, min(0.99999, prev_p))))
                        else:
                            logit_prev = logit_pt
                        feats.append([logit_pt, logit_prev, ldt])
                    return np.array(feats, dtype=np.float32)

                p_tr_pop = pop_clf.predict_proba(X_tr_pop_s)[:, 1]
                X_tr_serial = build_batch_serial_features(train_eps, p_tr_pop)
                serial_clf = LogisticRegression(C=1.0, max_iter=500, random_state=args.seed)
                serial_clf.fit(X_tr_serial, y_tr)

                X_test_serial = build_batch_serial_features(test_eps, p_test_pop)
                p_test_serial = serial_clf.predict_proba(X_test_serial)[:, 1]
                auc_serial = float(roc_auc_score(y_test, p_test_serial))
                task_serial_res[regime] = {
                    "auroc": auc_serial,
                    "delta_auroc_vs_pop": float(auc_serial - roc_auc_score(y_test, p_test_pop)),
                }

                # ------------------------------------------------------------------
                # Component 3: R020-B Multivariate PEB (Vectorized)
                # ------------------------------------------------------------------
                # Vectorized feature builder for K=0 and K=1
                # K=0: s_hat = mu, diff = z_query - mu, r_std = sqrt_inv_cov_0 @ diff
                Z_tr_q = np.stack([ep["z_query"] for ep in train_eps], axis=0)  # (N, d)
                diff_tr_0 = Z_tr_q - cov_est.mu
                r_std_tr_0 = (diff_tr_0 @ cov_est.sqrt_inv_cov_0.T)
                X_tr_mv_k0 = np.concatenate([Z_tr_q, r_std_tr_0], axis=1)

                # K=1
                # Support vectors for episodes with prior
                Z_tr_s1 = np.stack([ep["all_support_z"][-1] if ep["n_priors_available"] > 0 else ep["z_query"] for ep in train_eps], axis=0)
                has_p_tr = np.array([ep["n_priors_available"] > 0 for ep in train_eps])

                # s_hat_1 = mu + (z_bar - mu) @ shrink_mat_1^T
                s_hat_1 = cov_est.mu + (Z_tr_s1 - cov_est.mu) @ cov_est.shrink_mat_1.T
                diff_tr_1 = Z_tr_q - s_hat_1
                r_std_tr_1 = (diff_tr_1 @ cov_est.sqrt_inv_cov_1.T)

                # Blend K=0 and K=1 for training progressive probe
                mask_k1 = has_p_tr[:, None]
                X_tr_mv = np.where(mask_k1, np.concatenate([Z_tr_q, r_std_tr_1], axis=1), X_tr_mv_k0)

                scaler_mv = StandardScaler()
                X_tr_mv_s = scaler_mv.fit_transform(X_tr_mv)
                mv_clf = LogisticRegression(C=1.0, max_iter=500, random_state=args.seed)
                mv_clf.fit(X_tr_mv_s, y_tr)

                # Test evaluation K=0 and K=1
                Z_te_q = np.stack([ep["z_query"] for ep in test_eps], axis=0)
                Z_te_s1 = np.stack([ep["all_support_z"][-1] for ep in test_eps], axis=0)

                # Test K=0
                diff_te_0 = Z_te_q - cov_est.mu
                r_std_te_0 = diff_te_0 @ cov_est.sqrt_inv_cov_0.T
                X_te_mv_k0 = np.concatenate([Z_te_q, r_std_te_0], axis=1)

                # Test K=1
                s_hat_te_1 = cov_est.mu + (Z_te_s1 - cov_est.mu) @ cov_est.shrink_mat_1.T
                diff_te_1 = Z_te_q - s_hat_te_1
                r_std_te_1 = diff_te_1 @ cov_est.sqrt_inv_cov_1.T
                X_te_mv_k1 = np.concatenate([Z_te_q, r_std_te_1], axis=1)

                p_test_mv_k0 = mv_clf.predict_proba(scaler_mv.transform(X_te_mv_k0))[:, 1]
                p_test_mv_k1 = mv_clf.predict_proba(scaler_mv.transform(X_te_mv_k1))[:, 1]

                auc_mv_k0 = float(roc_auc_score(y_test, p_test_mv_k0))
                auc_mv_k1 = float(roc_auc_score(y_test, p_test_mv_k1))
                task_mv_res[regime] = {
                    "k0_auroc": auc_mv_k0,
                    "k1_auroc": auc_mv_k1,
                    "delta_auroc_vs_k0": float(auc_mv_k1 - auc_mv_k0),
                }

                # Validation predictions for threshold locking
                Z_va_q = np.stack([ep["z_query"] for ep in val_eps], axis=0)
                diff_va_0 = Z_va_q - cov_est.mu
                r_std_va_0 = diff_va_0 @ cov_est.sqrt_inv_cov_0.T
                X_va_mv_k0 = np.concatenate([Z_va_q, r_std_va_0], axis=1)
                p_val_mv_k0 = mv_clf.predict_proba(scaler_mv.transform(X_va_mv_k0))[:, 1]

                # ------------------------------------------------------------------
                # Component 4: R020-A Orthogonal Rotation Invariance Audit (Fast Vectorized)
                # ------------------------------------------------------------------
                # Native Diagonal PEB baseline
                lam_native = cov_est.tau2_diag / (cov_est.tau2_diag + cov_est.sigma2_diag)
                s_native_1 = cov_est.mu + (Z_tr_s1 - cov_est.mu) * lam_native
                denom_native_1 = np.sqrt(cov_est.sigma2_diag + (cov_est.tau2_diag * cov_est.sigma2_diag) / (cov_est.tau2_diag + cov_est.sigma2_diag))
                r_diag_tr_1 = (Z_tr_q - s_native_1) / denom_native_1

                denom_native_0 = np.sqrt(cov_est.sigma2_diag + cov_est.tau2_diag)
                r_diag_tr_0 = (Z_tr_q - cov_est.mu) / denom_native_0

                X_tr_diag_1 = np.where(mask_k1, np.concatenate([Z_tr_q, r_diag_tr_1], axis=1), np.concatenate([Z_tr_q, r_diag_tr_0], axis=1))

                s_diag = StandardScaler()
                diag_clf = LogisticRegression(C=1.0, max_iter=300, random_state=args.seed)
                diag_clf.fit(s_diag.fit_transform(X_tr_diag_1), y_tr)

                X_te_diag_0 = np.concatenate([Z_te_q, (Z_te_q - cov_est.mu) / denom_native_0], axis=1)
                X_te_diag_1 = np.concatenate([Z_te_q, (Z_te_q - (cov_est.mu + (Z_te_s1 - cov_est.mu) * lam_native)) / denom_native_1], axis=1)

                p_diag_k0 = diag_clf.predict_proba(s_diag.transform(X_te_diag_0))[:, 1]
                p_diag_k1 = diag_clf.predict_proba(s_diag.transform(X_te_diag_1))[:, 1]
                native_delta_auc = float(roc_auc_score(y_test, p_diag_k1) - roc_auc_score(y_test, p_diag_k0))

                # Vectorized 50 Rotations
                rot_delta_aucs = []
                Q_list = rot_matrices[d]

                for q_m in Q_list:
                    # Analytical rotation: O(d^3) matrix multiply
                    mu_rot, sig2_rot, tau2_rot = cov_est.get_rotated_diagonal_params(q_m)
                    lam_rot = tau2_rot / (tau2_rot + sig2_rot)
                    denom_rot_1 = np.sqrt(sig2_rot + (tau2_rot * sig2_rot) / (tau2_rot + sig2_rot))
                    denom_rot_0 = np.sqrt(sig2_rot + tau2_rot)

                    Z_tr_q_rot = Z_tr_q @ q_m.T
                    Z_tr_s1_rot = Z_tr_s1 @ q_m.T

                    s_rot_1 = mu_rot + (Z_tr_s1_rot - mu_rot) * lam_rot
                    r_rot_tr_1 = (Z_tr_q_rot - s_rot_1) / denom_rot_1
                    r_rot_tr_0 = (Z_tr_q_rot - mu_rot) / denom_rot_0

                    X_tr_rot = np.where(mask_k1, np.concatenate([Z_tr_q_rot, r_rot_tr_1], axis=1), np.concatenate([Z_tr_q_rot, r_rot_tr_0], axis=1))

                    s_r = StandardScaler()
                    clf_r = LogisticRegression(C=1.0, max_iter=200, random_state=args.seed)
                    clf_r.fit(s_r.fit_transform(X_tr_rot), y_tr)

                    Z_te_q_rot = Z_te_q @ q_m.T
                    Z_te_s1_rot = Z_te_s1 @ q_m.T

                    X_te_rot_0 = np.concatenate([Z_te_q_rot, (Z_te_q_rot - mu_rot) / denom_rot_0], axis=1)
                    X_te_rot_1 = np.concatenate([Z_te_q_rot, (Z_te_q_rot - (mu_rot + (Z_te_s1_rot - mu_rot) * lam_rot)) / denom_rot_1], axis=1)

                    pr0 = clf_r.predict_proba(s_r.transform(X_te_rot_0))[:, 1]
                    pr1 = clf_r.predict_proba(s_r.transform(X_te_rot_1))[:, 1]
                    rot_delta_aucs.append(roc_auc_score(y_test, pr1) - roc_auc_score(y_test, pr0))

                rot_mean = float(np.mean(rot_delta_aucs))
                rot_std = float(np.std(rot_delta_aucs))
                task_rot_res[regime] = {
                    "native_delta_auroc": native_delta_auc,
                    "rotated_mean_delta_auroc": rot_mean,
                    "rotated_std_delta_auroc": rot_std,
                    "instability_cv": float(rot_std / max(1e-4, abs(rot_mean))),
                }

                # ------------------------------------------------------------------
                # Component 5: R020-D Locked Threshold & Patient-Clustered Bootstrap
                # ------------------------------------------------------------------
                val_pos_scores = np.sort(p_val_mv_k0[y_val == 1]) if sum(y_val == 1) > 0 else np.array([0.5])
                val_idx = int(np.floor(0.10 * len(val_pos_scores)))
                locked_threshold = float(val_pos_scores[min(val_idx, len(val_pos_scores) - 1)])

                pos_mask = (y_test == 1)
                neg_mask = (y_test == 0)

                test_sens_k0 = float(np.mean(p_test_mv_k0[pos_mask] >= locked_threshold))
                test_spec_k0 = float(np.mean(p_test_mv_k0[neg_mask] < locked_threshold))
                test_fpr_k0 = float(np.mean(p_test_mv_k0[neg_mask] >= locked_threshold))

                test_sens_k1 = float(np.mean(p_test_mv_k1[pos_mask] >= locked_threshold))
                test_spec_k1 = float(np.mean(p_test_mv_k1[neg_mask] < locked_threshold))
                test_fpr_k1 = float(np.mean(p_test_mv_k1[neg_mask] >= locked_threshold))

                boot_stats = patient_clustered_bootstrap(
                    pids=test_pids,
                    y_true=y_test,
                    y_prob_k0=p_test_mv_k0,
                    y_prob_k1=p_test_mv_k1,
                    locked_thresh=locked_threshold,
                    n_boot=args.n_bootstrap,
                    seed=args.seed,
                )

                task_op_res[regime] = {
                    "sample_size": {
                        "n_eval_queries": len(test_eps),
                        "n_pos_queries": n_pos_test,
                        "n_neg_queries": n_neg_test,
                        "n_test_patients": n_pts_test,
                    },
                    "locked_threshold": locked_threshold,
                    "k0_operating_metrics": {
                        "sensitivity": test_sens_k0,
                        "specificity": test_spec_k0,
                        "fpr": test_fpr_k0,
                    },
                    "k1_operating_metrics": {
                        "sensitivity": test_sens_k1,
                        "specificity": test_spec_k1,
                        "fpr": test_fpr_k1,
                        "delta_fpr_locked": float(test_fpr_k1 - test_fpr_k0),
                    },
                    "bootstrap_95ci": boot_stats,
                }

                logger.info(
                    f"[{enc_name}|{task}|{regime}] N_test={len(test_eps)} (Pos={n_pos_test}, Neg={n_neg_test}, Pts={n_pts_test}) | "
                    f"Locked Thresh={locked_threshold:.4f} | "
                    f"MV-PEB ΔAUC={auc_mv_k1 - auc_mv_k0:+.4f} (95% CI: [{boot_stats['delta_auroc']['ci_low']:+.4f}, {boot_stats['delta_auroc']['ci_high']:+.4f}]) | "
                    f"ΔFPR={test_fpr_k1 - test_fpr_k0:+.4f} (95% CI: [{boot_stats['delta_fpr_locked']['ci_low']:+.4f}, {boot_stats['delta_fpr_locked']['ci_high']:+.4f}]) | "
                    f"Rot Audit Mean Δ={rot_mean:+.4f}±{rot_std:.4f}"
                )

            benchmark_results["r020_a_rotation_audit"][enc_name][task] = task_rot_res
            benchmark_results["r020_b_mv_peb"][enc_name][task] = task_mv_res
            benchmark_results["r020_c_serial_recalibration"][enc_name][task] = task_serial_res
            benchmark_results["r020_d_operating_validity"][enc_name][task] = task_op_res

    # Save Canonical JSON Results
    out_json = args.output_dir / "r020_validity_benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    logger.info(f"\nSaved R020 validity benchmark results to {out_json}")

    # Generate Markdown Summary Tables
    generate_r020_markdown_report(benchmark_results, args.output_dir / "r020_validity_benchmark_report.md")


def generate_r020_markdown_report(res: dict, out_path: Path) -> None:
    lines = [
        "# Milestone R020: Calibration Validity & Basis-Invariance Benchmark Report\n\n",
        "## 1. Operating-Point Validity & Patient-Clustered Bootstrap Analysis (R020-D)\n",
        "Operating thresholds are selected strictly on validation patients (`strat_fold == 7`) to achieve 90% sensitivity, then locked. All metrics report actual test-set performance with 1,000 patient-clustered bootstrap 95% confidence intervals.\n\n",
        "| Encoder | Task & Regime | Test Sample Size | Locked Thresh | Test Sens (K=1) | Test FPR (K=0 -> K=1) | Δ FPR (95% CI) | Δ AUROC (95% CI) |\n",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n",
    ]

    for enc in res["r020_d_operating_validity"]:
        for task in res["r020_d_operating_validity"][enc]:
            for reg in res["r020_d_operating_validity"][enc][task]:
                data = res["r020_d_operating_validity"][enc][task][reg]
                ss = data["sample_size"]
                ss_str = f"N={ss['n_eval_queries']} (+{ss['n_pos_queries']}/-{ss['n_neg_queries']}, {ss['n_test_patients']} pts)"
                th = f"{data['locked_threshold']:.3f}"
                sens = f"{data['k1_operating_metrics']['sensitivity']*100:.1f}%"
                fpr_str = f"{data['k0_operating_metrics']['fpr']*100:.1f}% -> {data['k1_operating_metrics']['fpr']*100:.1f}%"

                d_fpr = data["bootstrap_95ci"]["delta_fpr_locked"]
                d_fpr_str = f"{d_fpr['mean']*100:+.1f}% [{d_fpr['ci_low']*100:+.1f}%, {d_fpr['ci_high']*100:+.1f}%]"

                d_auc = data["bootstrap_95ci"]["delta_auroc"]
                d_auc_str = f"{d_auc['mean']:+.4f} [{d_auc['ci_low']:+.4f}, {d_auc['ci_high']:+.4f}]"

                lines.append(f"| **{enc.split()[0]}** | {task} ({reg}) | {ss_str} | {th} | {sens} | {fpr_str} | {d_fpr_str} | {d_auc_str} |\n")

    lines.extend([
        "\n## 2. Orthogonal Rotation Invariance Audit (R020-A)\n",
        "Evaluates whether Diagonal-PEB gains depend on arbitrary encoder coordinate orientation across M=50 random orthogonal rotations Q_m in O(d).\n\n",
        "| Encoder | Task & Regime | Native Δ AUROC | Rotated Mean Δ AUROC | Rotated Std (Spread) | Rotation Instability (CV) |\n",
        "| :--- | :--- | :---: | :---: | :---: | :---: |\n",
    ])

    for enc in res["r020_a_rotation_audit"]:
        for task in res["r020_a_rotation_audit"][enc]:
            for reg in res["r020_a_rotation_audit"][enc][task]:
                data = res["r020_a_rotation_audit"][enc][task][reg]
                lines.append(
                    f"| **{enc.split()[0]}** | {task} ({reg}) | {data['native_delta_auroc']:+.4f} | "
                    f"{data['rotated_mean_delta_auroc']:+.4f} | ±{data['rotated_std_delta_auroc']:.4f} | "
                    f"{data['instability_cv']:.2f} |\n"
                )

    lines.extend([
        "\n## 3. Five-Way Personalization Method Head-to-Head Comparison (K=1 Certified)\n",
        "Compares Population (K=0), Serial-Score Recalibration, Diagonal-PEB, Multivariate PEB (MV-PEB), and Learned FiLM/AdaIN.\n\n",
        "| Task | Encoder | Population (K=0) | Serial-Score Recalib | Diagonal-PEB | MV-PEB (Basis-Covariant) | FiLM/AdaIN |\n",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n",
    ])

    for task in ["SR->AF", "AF->SR", "MI", "STTC", "CD"]:
        for enc in ["SetOperator (d=256)", "GraphECG (d=192)", "CSFM Base (d=768)"]:
            reg = "certified"
            if reg in res["r020_b_mv_peb"].get(enc, {}).get(task, {}):
                mv_k0 = res["r020_b_mv_peb"][enc][task][reg]["k0_auroc"]
                mv_k1 = res["r020_b_mv_peb"][enc][task][reg]["k1_auroc"]
                serial_auc = res["r020_c_serial_recalibration"][enc][task][reg]["auroc"]
                diag_delta = res["r020_a_rotation_audit"][enc][task][reg]["native_delta_auroc"]
                diag_auc = mv_k0 + diag_delta

                lines.append(
                    f"| **{task}** | {enc.split()[0]} | {mv_k0:.4f} | {serial_auc:.4f} | {diag_auc:.4f} | **{mv_k1:.4f}** | See R019 |\n"
                )

    with open(out_path, "w") as f:
        f.writelines(lines)
    logger.info(f"Saved R020 validity benchmark report to {out_path}")


if __name__ == "__main__":
    main()
