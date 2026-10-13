#!/usr/bin/env python3
"""
Run R017: PTB-XL Empirical-Bayes Serial-Event Personalization Benchmark.
Evaluates SetOperator vs GraphECG vs CSFM Base across clinically anchored transitions
under the Empirical-Bayes (PEB) population-to-personalized shrinkage protocol.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch

torch.backends.cudnn.enabled = False

SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
K_LEVELS = (0, 1, 2, 3, 5)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("r017_peb_benchmark")
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


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1] if i < n_bins - 1 else y_prob <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_fpr_at_sensitivity(y_true: np.ndarray, y_prob: np.ndarray, target_sens: float = 0.90) -> float:
    """Compute False Positive Rate (FPR) at a prespecified sensitivity (e.g. 90%)."""
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Sort thresholds descending
    thresholds = np.sort(y_prob)[::-1]
    best_fpr = 1.0
    for th in thresholds:
        tp = (y_prob[pos_mask] >= th).sum()
        fp = (y_prob[neg_mask] >= th).sum()
        sens = tp / n_pos
        if sens >= target_sens:
            fpr = fp / n_neg
            return float(fpr)
    return float(best_fpr)


class EmpiricalBayesEstimator:
    """Estimates population mean, within-patient variance, and between-patient variance on training cohort."""

    def __init__(self, z_train: np.ndarray, train_patient_ids: np.ndarray):
        self.d = z_train.shape[1]
        self.mu = np.mean(z_train, axis=0).astype(np.float32)  # (d,)

        # Group by patient
        df_p = pd.DataFrame({"pid": train_patient_ids})
        groups = df_p.groupby("pid").indices

        # Compute within-patient variances and patient means
        patient_means = []
        ss_within = np.zeros(self.d, dtype=np.float64)
        df_within = 0
        n_p_list = []

        for pid, indices in groups.items():
            n_p = len(indices)
            n_p_list.append(n_p)
            pts = z_train[indices]
            p_mean = np.mean(pts, axis=0)
            patient_means.append(p_mean)
            if n_p > 1:
                ss_within += np.sum((pts - p_mean) ** 2, axis=0)
                df_within += (n_p - 1)

        # Pooled within-patient variance sigma_j^2
        if df_within > 0:
            self.sigma2 = (ss_within / df_within).astype(np.float32)
        else:
            self.sigma2 = np.var(z_train, axis=0).astype(np.float32) * 0.5
        self.sigma2 = np.maximum(self.sigma2, 1e-6)

        # Between-patient variance tau_j^2 via one-way ANOVA / empirical Bayes
        patient_means = np.array(patient_means, dtype=np.float64)  # (P, d)
        P = len(patient_means)
        var_between_raw = np.var(patient_means, axis=0)
        mean_inv_n = np.mean(1.0 / np.array(n_p_list))
        tau2_unbiased = var_between_raw - self.sigma2 * mean_inv_n
        self.tau2 = np.maximum(tau2_unbiased, 1e-6).astype(np.float32)

    def get_reference_and_deviation(
        self,
        z_query: np.ndarray,
        support_z_list: Optional[List[np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute personal reference s_hat, posterior variance V, and standardized residual r.
        
        Args:
            z_query: (d,) current query embedding
            support_z_list: list of K prior embeddings [(d,), ...], or None / empty for K=0.
            
        Returns:
            s_hat: (d,)
            V: (d,) posterior uncertainty
            r: (d,) standardized patient-relative residual
        """
        K = len(support_z_list) if support_z_list is not None else 0

        if K == 0:
            s_hat = self.mu.copy()
            V = self.tau2.copy()
            r = (z_query - self.mu) / np.sqrt(self.sigma2 + self.tau2)
        else:
            z_bar = np.mean(support_z_list, axis=0).astype(np.float32)
            # lambda_j = tau_j^2 / (tau_j^2 + sigma_j^2 / K)
            lambda_weight = self.tau2 / (self.tau2 + self.sigma2 / float(K))
            s_hat = self.mu + lambda_weight * (z_bar - self.mu)
            # V_j = (1/tau_j^2 + K/sigma_j^2)^-1
            V = (self.tau2 * self.sigma2) / (float(K) * self.tau2 + self.sigma2)
            r = (z_query - s_hat) / np.sqrt(self.sigma2 + V)

        return s_hat, V, r

    def make_feature_vector(
        self,
        z_query: np.ndarray,
        support_z_list: Optional[List[np.ndarray]],
    ) -> Tuple[np.ndarray, float]:
        """Construct X = [z; r; log V] in R^{3d} and scalar uncertainty U."""
        s_hat, V, r = self.get_reference_and_deviation(z_query, support_z_list)
        log_V = np.log(V + 1e-6)
        x_feat = np.concatenate([z_query, r, log_V], axis=0)
        U = float(np.mean(V))
        return x_feat, U


def build_serial_task_episodes(
    df_multi: pd.DataFrame,
    z_all: np.ndarray,
    task_name: str,
    regime: str = "passive",
) -> List[Dict[str, Any]]:
    """Build chronological query episodes with prior support for a clinically anchored task.
    
    Tasks:
        - "SR->AF": query must be AF (label=1) or SR (label=0) when prior was SR
        - "AF->SR": query must be SR (label=1) or AF (label=0) when prior was AF
        - "MI": query is MI+ (label=1) or MI- (label=0) when prior was MI-
        - "STTC": query is STTC+ (label=1) or STTC- (label=0) when prior was STTC-
        - "CD": query is CD+ (label=1) or CD- (label=0) when prior was CD-
    """
    episodes = []

    for pid, group in df_multi.groupby("patient_id"):
        recs = group.to_dict("records")
        if len(recs) < 2:
            continue

        for q_idx in range(1, len(recs)):
            q_rec = recs[q_idx]
            q_pos = q_rec["pos"]
            prior_recs = recs[:q_idx]

            # Determine task eligibility and binary label
            target_label = None
            certified_valid = False

            if task_name == "SR->AF":
                # Eligibility: immediate prior was SR
                if prior_recs[-1]["clean_rhythm"] != "SR":
                    continue
                if q_rec["clean_rhythm"] == "AF":
                    target_label = 1
                elif q_rec["clean_rhythm"] == "SR":
                    target_label = 0
                else:
                    continue

            elif task_name == "AF->SR":
                # Eligibility: immediate prior was AF
                if prior_recs[-1]["clean_rhythm"] != "AF":
                    continue
                if q_rec["clean_rhythm"] == "SR":
                    target_label = 1
                elif q_rec["clean_rhythm"] == "AF":
                    target_label = 0
                else:
                    continue

            elif task_name == "MI":
                # Eligibility: immediate prior was MI-
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

            # Support selection
            if regime == "passive":
                # Most recent prior records regardless of diagnosis
                support_candidates = [z_all[r["pos"]] for r in prior_recs]
            elif regime == "certified":
                # Filter priors to baseline reference state only
                if task_name in ("SR->AF", "AF->SR"):
                    base_rhythm = "SR" if task_name == "SR->AF" else "AF"
                    cert_recs = [r for r in prior_recs if r["clean_rhythm"] == base_rhythm]
                elif task_name == "MI":
                    cert_recs = [r for r in prior_recs if r["MI"] == 0]
                elif task_name == "STTC":
                    cert_recs = [r for r in prior_recs if r["STTC"] == 0]
                elif task_name == "CD":
                    cert_recs = [r for r in prior_recs if r["CD"] == 0]
                else:
                    cert_recs = prior_recs
                support_candidates = [z_all[r["pos"]] for r in cert_recs]

            episodes.append({
                "patient_id": pid,
                "strat_fold": q_rec["strat_fold"],
                "q_pos": q_pos,
                "z_query": z_all[q_pos],
                "y": target_label,
                "all_support_z": support_candidates,  # ordered chronologically oldest to newest
                "n_priors_available": len(support_candidates),
            })

    return episodes


def evaluate_peb_benchmark_for_arm(
    z_all: np.ndarray,
    df_multi: pd.DataFrame,
    rep_name: str,
    logger: logging.Logger,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete PEB Serial-Event Personalization Benchmark for a representation arm."""
    logger.info("\n" + "=" * 70)
    logger.info(f"=== Running R017 PEB Personalization Benchmark: {rep_name} ===")
    logger.info("=" * 70)

    # 1. Fit Empirical-Bayes Estimator strictly on Training Patients (Folds 1-7)
    train_mask = (df_multi["strat_fold"].values <= 7)
    z_tr = z_all[train_mask]
    pid_tr = df_multi["patient_id"].values[train_mask]
    peb = EmpiricalBayesEstimator(z_tr, pid_tr)
    logger.info(f"[{rep_name}] Fitted PEB Prior: d={peb.d}, mean_mu={peb.mu.mean():.4f}, mean_sigma2={peb.sigma2.mean():.4f}, mean_tau2={peb.tau2.mean():.4f}")

    tasks = ["SR->AF", "AF->SR", "MI", "STTC", "CD"]
    regimes = ["passive", "certified"]
    arm_results = {}

    for task in tasks:
        logger.info(f"\n--- Task: {task} ---")
        task_results = {}

        for regime in regimes:
            episodes = build_serial_task_episodes(df_multi, z_all, task_name=task, regime=regime)
            train_episodes = [ep for ep in episodes if ep["strat_fold"] <= 7]
            test_episodes = [ep for ep in episodes if ep["strat_fold"] == 8]

            y_tr = np.array([ep["y"] for ep in train_episodes])
            y_te = np.array([ep["y"] for ep in test_episodes])

            logger.info(f"[{rep_name} | {task} | {regime}] Train Episodes: {len(train_episodes)} (Pos: {y_tr.sum()}) | Test Episodes: {len(test_episodes)} (Pos: {y_te.sum()})")

            if len(train_episodes) < 20 or y_tr.sum() < 5 or (len(y_tr) - y_tr.sum()) < 5:
                logger.warning(f"Insufficient training episodes for {task} ({regime}). Skipping.")
                continue
            if len(test_episodes) < 10 or y_te.sum() < 2 or (len(y_te) - y_te.sum()) < 2:
                logger.warning(f"Insufficient test episodes for {task} ({regime}). Skipping.")
                continue

            # 2. Build Multi-K Training Features: randomly sample K in {0, 1, 2, 3, 5}
            rng = np.random.RandomState(rng_seed)
            X_tr_list = []
            for ep in train_episodes:
                n_avail = ep["n_priors_available"]
                k_choices = [k for k in K_LEVELS if k <= n_avail]
                chosen_k = rng.choice(k_choices)
                if chosen_k == 0:
                    sup = []
                else:
                    sup = ep["all_support_z"][-chosen_k:]
                feat, _ = peb.make_feature_vector(ep["z_query"], sup)
                X_tr_list.append(feat)

            X_tr = np.array(X_tr_list, dtype=np.float32)

            # 3. Train Single Progressive Regularized Predictor
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)

            best_clf = None
            best_val_auc = -1.0
            for c_val in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
                try:
                    clf = LogisticRegression(C=c_val, max_iter=1000, random_state=rng_seed, solver="lbfgs")
                    clf.fit(X_tr_scaled, y_tr)
                    probs_tr = clf.predict_proba(X_tr_scaled)[:, 1]
                    auc_tr = fast_auroc(y_tr, probs_tr)
                    if auc_tr > best_val_auc:
                        best_val_auc = auc_tr
                        best_clf = clf
                except Exception:
                    continue

            if best_clf is None:
                continue

            # 4. Determine Validation Uncertainty Fallback Threshold U* on Train Residuals
            train_u_list = []
            for ep in train_episodes:
                _, u_val = peb.make_feature_vector(ep["z_query"], ep["all_support_z"])
                train_u_list.append(u_val)
            u_threshold = float(np.percentile(train_u_list, 75))  # Fallback when uncertainty is in top quartile

            # 5. Evaluate at each K in {0, 1, 2, 3, 5} on Test Episodes
            # Matched Efficacy: evaluate on test episodes that have at least 1 prior available (or K priors)
            k_eval_results = {}
            coverage_dict = {}

            # Population availability (Coverage) across all test episodes
            total_test_pats = len(test_episodes)
            for k_val in K_LEVELS:
                if k_val == 0:
                    cov = 1.0
                else:
                    cov = float(np.mean([ep["n_priors_available"] >= k_val for ep in test_episodes]))
                coverage_dict[k_val] = cov

            # Fixed Query Matched Evaluation: queries with at least K=1 available
            matched_test_episodes = [ep for ep in test_episodes if ep["n_priors_available"] >= 1]
            y_te_matched = np.array([ep["y"] for ep in matched_test_episodes])

            if len(matched_test_episodes) >= 10 and y_te_matched.sum() >= 2:
                eval_pool = matched_test_episodes
                y_eval = y_te_matched
            else:
                eval_pool = test_episodes
                y_eval = y_te

            # Compute K=0 baseline predictions for all queries
            X_te_k0 = np.array([
                peb.make_feature_vector(ep["z_query"], [])[0]
                for ep in eval_pool
            ], dtype=np.float32)
            p0_preds = best_clf.predict_proba(scaler.transform(X_te_k0))[:, 1]
            auc_k0 = fast_auroc(y_eval, p0_preds)
            auprc_k0 = average_precision_score(y_eval, p0_preds)
            brier_k0 = brier_score_loss(y_eval, p0_preds)
            ece_k0 = compute_ece(y_eval, p0_preds)
            fpr90_k0 = compute_fpr_at_sensitivity(y_eval, p0_preds, target_sens=0.90)

            for k_val in K_LEVELS:
                if k_val == 0:
                    preds = p0_preds
                    gated_preds = p0_preds
                    fallback_frac = 0.0
                else:
                    # Construct feature vector using up to k_val priors
                    X_te_k = []
                    gated_preds_list = []
                    n_fallback = 0

                    for i_ep, ep in enumerate(eval_pool):
                        n_avail = ep["n_priors_available"]
                        take_k = min(k_val, n_avail)
                        sup = ep["all_support_z"][-take_k:] if take_k > 0 else []
                        feat, u_score = peb.make_feature_vector(ep["z_query"], sup)
                        X_te_k.append(feat)

                        # Individual prediction
                        feat_scaled = scaler.transform(feat[None, :])
                        pk = best_clf.predict_proba(feat_scaled)[0, 1]

                        # Uncertainty gating fallback
                        if u_score > u_threshold:
                            gated_preds_list.append(p0_preds[i_ep])
                            n_fallback += 1
                        else:
                            gated_preds_list.append(pk)

                    X_te_k = np.array(X_te_k, dtype=np.float32)
                    preds = best_clf.predict_proba(scaler.transform(X_te_k))[:, 1]
                    gated_preds = np.array(gated_preds_list)
                    fallback_frac = n_fallback / float(len(eval_pool))

                auc_val = fast_auroc(y_eval, preds)
                auprc_val = average_precision_score(y_eval, preds)
                brier_val = brier_score_loss(y_eval, preds)
                ece_val = compute_ece(y_eval, preds)
                fpr90_val = compute_fpr_at_sensitivity(y_eval, preds, target_sens=0.90)

                auc_gated = fast_auroc(y_eval, gated_preds)
                fpr90_gated = compute_fpr_at_sensitivity(y_eval, gated_preds, target_sens=0.90)

                delta_auc = auc_val - auc_k0
                delta_fpr90 = fpr90_val - fpr90_k0

                k_eval_results[k_val] = {
                    "k": k_val,
                    "coverage": coverage_dict[k_val],
                    "auroc": float(auc_val),
                    "auprc": float(auprc_val),
                    "brier": float(brier_val),
                    "ece": float(ece_val),
                    "fpr90": float(fpr90_val),
                    "delta_auroc_vs_k0": float(delta_auc),
                    "delta_fpr90_vs_k0": float(delta_fpr90),
                    "auroc_uncertainty_gated": float(auc_gated),
                    "fpr90_uncertainty_gated": float(fpr90_gated),
                    "fallback_fraction": float(fallback_frac),
                }

                logger.info(
                    f"[{rep_name}|{task}|{regime}] K={k_val} (Cov: {coverage_dict[k_val]*100:.1f}%): "
                    f"AUROC={auc_val:.4f} (Δ={delta_auc:+.4f}), FPR90={fpr90_val:.4f} (Δ={delta_fpr90:+.4f}), "
                    f"Gated AUROC={auc_gated:.4f}, Fallback={fallback_frac*100:.1f}%"
                )

            task_results[regime] = {
                "n_eval_queries": len(eval_pool),
                "n_pos_queries": int(y_eval.sum()),
                "k_curves": k_eval_results,
            }

        arm_results[task] = task_results

    return arm_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R017: PTB-XL Empirical-Bayes Serial-Event Personalization Benchmark")
    parser.add_argument("--so-cache", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/setoperator_e6_all_z.npz"))
    parser.add_argument("--graphecg-cache", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/graphecg_multi_z.npz"))
    parser.add_argument("--csfm-cache", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/csfm_base_multi_z.npz"))
    parser.add_argument("--ptbxl-data", type=Path, default=Path("data/ptbxl"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_r017_peb_benchmark"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir / "run.log")

    logger.info("=" * 70)
    logger.info("Milestone R017: PTB-XL Empirical-Bayes Serial-Event Benchmark")
    logger.info(f"SetOperator Cache: {args.so_cache}")
    logger.info(f"GraphECG Cache:    {args.graphecg_cache}")
    logger.info(f"CSFM Base Cache:   {args.csfm_cache}")
    logger.info(f"Output Directory:  {args.output_dir}")
    logger.info("=" * 70)

    # 1. Load Metadata and Diagnostic Statements
    scp_df = pd.read_csv(args.ptbxl_data / "scp_statements.csv", index_col=0)
    diag_map = scp_df["diagnostic_class"].dropna().to_dict()

    df_meta = pd.read_csv(args.ptbxl_data / "ptbxl_database.csv", index_col="ecg_id")
    df_meta["recording_date"] = pd.to_datetime(df_meta["recording_date"])

    # Load All Latent Files
    so_npz = np.load(args.so_cache)
    all_z_so = so_npz["all_z"]
    all_ecg_so = so_npz["all_ecg"]
    ecg_to_so_idx = {int(ecg_id): i for i, ecg_id in enumerate(all_ecg_so)}

    graphecg_npz = np.load(args.graphecg_cache)
    all_z_ge = graphecg_npz["graphecg_z"]
    all_ecg_ge = graphecg_npz["ecg_ids"]
    ecg_to_ge_idx = {int(ecg_id): i for i, ecg_id in enumerate(all_ecg_ge)}

    csfm_npz = np.load(args.csfm_cache)
    all_z_csfm = csfm_npz["csfm_z"]
    all_ecg_csfm = csfm_npz["ecg_ids"]
    ecg_to_csfm_idx = {int(ecg_id): i for i, ecg_id in enumerate(all_ecg_csfm)}

    # Find common repeated cohort present in all three caches and metadata
    df_meta_avail = df_meta.loc[list(ecg_to_so_idx.keys())].copy()

    for sc in SUPERCLASSES:
        df_meta_avail[sc] = df_meta_avail["scp_codes"].apply(
            lambda s: int(any(diag_map.get(k) == sc for k in ast.literal_eval(s).keys()))
            if isinstance(s, str) else 0
        )

    def get_clean_rhythm(scp_str):
        if not isinstance(scp_str, str):
            return "OTHER"
        try:
            d = ast.literal_eval(scp_str)
            if "AFIB" in d or "AFLT" in d:
                return "AF"
            if "SR" in d:
                return "SR"
        except Exception:
            pass
        return "OTHER"

    df_meta_avail["clean_rhythm"] = df_meta_avail["scp_codes"].apply(get_clean_rhythm)

    pat_counts = df_meta_avail["patient_id"].value_counts()
    multi_pats = pat_counts[pat_counts >= 2].index.values
    df_multi = df_meta_avail[df_meta_avail["patient_id"].isin(multi_pats)].sort_values(["patient_id", "recording_date"]).copy()
    
    # Filter to records present in GraphECG and CSFM
    valid_ecgs = [e for e in df_multi.index.values if e in ecg_to_ge_idx and e in ecg_to_csfm_idx]
    df_multi = df_multi.loc[valid_ecgs].copy()
    df_multi["pos"] = np.arange(len(df_multi))
    logger.info(f"Verified Harmonized Cohort: {df_multi['patient_id'].nunique():,} patients, {len(df_multi):,} ECG records.")

    # Slice embeddings aligned with df_multi
    z_so = np.array([all_z_so[ecg_to_so_idx[e]] for e in df_multi.index.values], dtype=np.float32)
    z_ge = np.array([all_z_ge[ecg_to_ge_idx[e]] for e in df_multi.index.values], dtype=np.float32)
    z_csfm = np.array([all_z_csfm[ecg_to_csfm_idx[e]] for e in df_multi.index.values], dtype=np.float32)

    logger.info(f"Latents Aligned: SetOperator {z_so.shape}, GraphECG {z_ge.shape}, CSFM Base {z_csfm.shape}")

    # Evaluate all three encoders
    encoders = [
        ("SetOperator (d=256)", z_so),
        ("GraphECG (d=192)", z_ge),
        ("CSFM Base (d=768)", z_csfm),
    ]

    benchmark_suite_results = {}
    for enc_name, z_mat in encoders:
        res = evaluate_peb_benchmark_for_arm(z_mat, df_multi, enc_name, logger, rng_seed=args.seed)
        benchmark_suite_results[enc_name] = res

    # Save Results
    results_path = args.output_dir / "r017_peb_benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(benchmark_suite_results, f, indent=2)
    logger.info(f"\nSaved structured benchmark results to: {results_path}")

    # Generate Markdown Summary Table
    md_lines = [
        "# Milestone R017: PTB-XL Empirical-Bayes Serial-Event Personalization Benchmark",
        "",
        "### Primary Estimand: Performance(K) vs K=0, Delta(K), Coverage(K), and Uncertainty Fallback",
        "",
    ]

    for task in ["SR->AF", "MI", "STTC", "CD"]:
        md_lines.append(f"#### Task: {task} (Passive History Regime C1)")
        md_lines.append("| Encoder | K=0 (Pop) AUROC | K=1 AUROC (Δ) | K=2 AUROC (Δ) | K=3 AUROC (Δ) | K=1 FPR90 (Δ) | Coverage (K=1) | Gated AUROC (K=1) |")
        md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for enc_name, _ in encoders:
            if task not in benchmark_suite_results[enc_name] or "passive" not in benchmark_suite_results[enc_name][task]:
                continue
            k_data = benchmark_suite_results[enc_name][task]["passive"]["k_curves"]
            k0_auc = k_data.get(0, {}).get("auroc", float("nan"))
            k1 = k_data.get(1, {})
            k2 = k_data.get(2, {})
            k3 = k_data.get(3, {})

            md_lines.append(
                f"| **{enc_name}** | {k0_auc:.4f} | "
                f"{k1.get('auroc', float('nan')):.4f} ({k1.get('delta_auroc_vs_k0', 0.0):+.4f}) | "
                f"{k2.get('auroc', float('nan')):.4f} ({k2.get('delta_auroc_vs_k0', 0.0):+.4f}) | "
                f"{k3.get('auroc', float('nan')):.4f} ({k3.get('delta_auroc_vs_k0', 0.0):+.4f}) | "
                f"{k1.get('fpr90', float('nan')):.4f} ({k1.get('delta_fpr90_vs_k0', 0.0):+.4f}) | "
                f"{k1.get('coverage', 0.0)*100:.1f}% | "
                f"{k1.get('auroc_uncertainty_gated', float('nan')):.4f} |"
            )
        md_lines.append("")

    table_path = args.output_dir / "r017_peb_benchmark_table.md"
    with open(table_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    logger.info(f"Saved summary table to: {table_path}")
    logger.info("\n=== Milestone R017 Benchmark COMPLETED Successfully ===")


if __name__ == "__main__":
    main()
