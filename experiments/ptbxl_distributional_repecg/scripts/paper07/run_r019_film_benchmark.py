#!/usr/bin/env python3
"""Milestone R019: Support-Conditioned FiLM/AdaIN Latent Personalization Benchmark.

Tests whether learned unlabeled patient conditioning (FiLM/AdaIN) beats statistical
Empirical-Bayes (PEB) shrinkage across all three frozen encoders:
  - SetOperator (d=256)
  - GraphECG (d=192)
  - CSFM Base (d=768)

Protocol:
  - Patient context: c_p^{(K)} = mean(z_{p,1:K})
  - Conditioning: gamma, beta = MLP(c_p)
  - Modulated representation: \tilde{z}_t = (1 + gamma) * LN(z_t) + beta
  - Exact reduction to population model at K=0: gamma = 0, beta = 0
  - Same clinically anchored transitions (SR->AF, AF->SR, MI, STTC, CD)
  - Dual regimes: Passive History (C1) and Certified Reference (C2)
  - Matched-query evaluation across K in {0, 1, 2, 3, 5}
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
import scipy.stats as stats
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# Setup Logger
logger = logging.getLogger("r019_film")
logger.setLevel(logging.INFO)
logger.handlers.clear()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
K_LEVELS = [0, 1, 2, 3, 5]


class FiLMPersonalizedClassifier(nn.Module):
    """Latent FiLM/AdaIN Personalization Module.
    
    Guarantees exact identity (population model) at K=0.
    """
    def __init__(self, d: int, hidden_dim: int = 64):
        super().__init__()
        self.d = d
        self.layer_norm = nn.LayerNorm(d)
        
        # FiLM generator: c_p -> gamma, beta
        self.film_net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * d),
        )
        # Initialize final layer weights and biases to 0
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.zeros_(self.film_net[-1].bias)
        
        # Downstream prediction head
        self.classifier = nn.Sequential(
            nn.Linear(d, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, z_t: torch.Tensor, c_p: torch.Tensor, is_k0: torch.Tensor) -> torch.Tensor:
        z_norm = self.layer_norm(z_t)
        gb = self.film_net(c_p)
        gamma, beta = gb.split(self.d, dim=-1)
        
        # At K=0, force gamma=0, beta=0
        mask = (~is_k0).float().unsqueeze(-1)
        gamma = gamma * mask
        beta = beta * mask
        
        z_tilde = (1.0 + gamma) * z_norm + beta
        logits = self.classifier(z_tilde).squeeze(-1)
        return logits


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = (binids == i)
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += np.sum(mask) / len(y_true) * np.abs(bin_acc - bin_conf)
    return float(ece)


def compute_fpr_at_sensitivity(y_true: np.ndarray, y_prob: np.ndarray, target_sens: float = 0.90) -> float:
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")

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


def extract_rhythm(scp_dict: dict) -> str:
    if "AFIB" in scp_dict or "AFLT" in scp_dict:
        return "AF"
    if "SR" in scp_dict:
        return "SR"
    return "OTHER"


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
                    ref_rhythm = "SR"
                    support_candidates = [z_all[r["pos"]] for r in prior_recs if r["clean_rhythm"] == ref_rhythm]
                elif task_name == "MI":
                    support_candidates = [z_all[r["pos"]] for r in prior_recs if r["MI"] == 0]
                elif task_name == "STTC":
                    support_candidates = [z_all[r["pos"]] for r in prior_recs if r["STTC"] == 0]
                elif task_name == "CD":
                    support_candidates = [z_all[r["pos"]] for r in prior_recs if r["CD"] == 0]
                else:
                    support_candidates = [z_all[r["pos"]] for r in prior_recs]

            episodes.append({
                "patient_id": pid,
                "strat_fold": q_rec["strat_fold"],
                "is_train": bool(q_rec["strat_fold"] <= 7),
                "is_test": bool(q_rec["strat_fold"] >= 8),
                "z_query": z_all[q_pos],
                "all_support_z": support_candidates,
                "n_priors_available": len(support_candidates),
                "y": target_label,
            })

    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone R019: Support-Conditioned FiLM/AdaIN Latent Personalization")
    parser.add_argument("--so-cache", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/setoperator_e6_all_z.npz"))
    parser.add_argument("--graphecg-cache", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/graphecg_multi_z.npz"))
    parser.add_argument("--csfm-cache", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/csfm_base_multi_z.npz"))
    parser.add_argument("--ptbxl-data", type=Path, default=Path("data/ptbxl"))
    parser.add_argument("--output", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_r019_film_benchmark"))
    parser.add_argument("--r017-results", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_r017_peb_benchmark/r017_peb_benchmark_results.json"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(args.output / "run.log", mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("======================================================================")
    logger.info("Milestone R019: Support-Conditioned FiLM/AdaIN Latent Personalization")
    logger.info(f"Output Directory: {args.output}")
    logger.info("======================================================================")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Load Metadata and Align Cohort (Identical to R017)
    scp_df = pd.read_csv(args.ptbxl_data / "scp_statements.csv", index_col=0)
    diag_map = scp_df["diagnostic_class"].dropna().to_dict()

    df_meta = pd.read_csv(args.ptbxl_data / "ptbxl_database.csv", index_col="ecg_id")
    df_meta["recording_date"] = pd.to_datetime(df_meta["recording_date"])

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

    valid_ecgs = []
    for ecg_id in df_long.index:
        if (ecg_id in ecg_to_so_idx) and (ecg_id in ecg_to_ge_idx) and (ecg_id in ecg_to_csfm_idx):
            valid_ecgs.append(ecg_id)

    df_long = df_long.loc[valid_ecgs].copy()
    df_long["pos"] = np.arange(len(df_long))

    Z_encoders = {
        "SetOperator (d=256)": np.array([all_z_so[ecg_to_so_idx[e]] for e in valid_ecgs], dtype=np.float32),
        "GraphECG (d=192)": np.array([all_z_ge[ecg_to_ge_idx[e]] for e in valid_ecgs], dtype=np.float32),
        "CSFM Base (d=768)": np.array([all_z_csfm[ecg_to_csfm_idx[e]] for e in valid_ecgs], dtype=np.float32),
    }

    logger.info(f"Verified Harmonized Cohort: {df_long['patient_id'].nunique():,} patients, {len(df_long):,} records.")

    tasks = ["SR->AF", "AF->SR", "MI", "STTC", "CD"]
    regimes = ["passive", "certified"]
    all_results = {}

    for enc_name, Z_all in Z_encoders.items():
        logger.info(f"\n======================================================================")
        logger.info(f"=== Running FiLM Personalization: {enc_name} ===")
        logger.info(f"======================================================================")
        d = Z_all.shape[1]
        all_results[enc_name] = {}

        for task in tasks:
            logger.info(f"\n--- Task: {task} ---")
            task_results = {}

            for regime in regimes:
                episodes = build_clinical_episodes(df_long, Z_all, task, regime)
                train_episodes = [ep for ep in episodes if ep["is_train"]]
                test_episodes = [ep for ep in episodes if ep["is_test"]]

                matched_test_episodes = [ep for ep in test_episodes if ep["n_priors_available"] >= 1]
                y_eval = np.array([ep["y"] for ep in matched_test_episodes])

                logger.info(f"[{enc_name} | {task} | {regime}] Train Episodes: {len(train_episodes)} | Test Episodes: {len(matched_test_episodes)} (Pos: {y_eval.sum()})")

                if len(matched_test_episodes) < 5 or y_eval.sum() < 2 or (len(matched_test_episodes) - y_eval.sum()) < 2:
                    logger.warning(f"Skipping {task} {regime} due to insufficient test cases.")
                    continue

                # Coverage calculation across all test episodes
                coverage_dict = {}
                for k_val in K_LEVELS:
                    if k_val == 0:
                        cov = 1.0
                    else:
                        cov = float(np.mean([ep["n_priors_available"] >= k_val for ep in test_episodes]))
                    coverage_dict[k_val] = cov

                # Initialize Model
                model = FiLMPersonalizedClassifier(d=d, hidden_dim=64)
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
                criterion = nn.BCEWithLogitsLoss()

                # Train Loop with Random Support Sampling K in {0, 1, 2, 3, 5}
                model.train()
                n_train = len(train_episodes)
                train_indices = np.arange(n_train)

                for epoch in range(args.epochs):
                    np.random.shuffle(train_indices)

                    for start_idx in range(0, n_train, args.batch_size):
                        batch_ids = train_indices[start_idx:start_idx + args.batch_size]
                        b_z_t = []
                        b_c_p = []
                        b_is_k0 = []
                        b_y = []

                        for bi in batch_ids:
                            ep = train_episodes[bi]
                            b_z_t.append(ep["z_query"])
                            b_y.append(ep["y"])

                            k_avail = ep["n_priors_available"]
                            k_sample = np.random.choice([0, 1, 2, 3, 5])

                            if k_sample == 0 or k_avail == 0:
                                b_is_k0.append(True)
                                b_c_p.append(np.zeros(d, dtype=np.float32))
                            else:
                                b_is_k0.append(False)
                                k_use = min(k_sample, k_avail)
                                supp = ep["all_support_z"][-k_use:]
                                b_c_p.append(np.mean(supp, axis=0))

                        t_z_t = torch.tensor(np.stack(b_z_t), dtype=torch.float32)
                        t_c_p = torch.tensor(np.stack(b_c_p), dtype=torch.float32)
                        t_is_k0 = torch.tensor(b_is_k0, dtype=torch.bool)
                        t_y = torch.tensor(b_y, dtype=torch.float32)

                        optimizer.zero_grad()
                        logits = model(t_z_t, t_c_p, t_is_k0)
                        loss = criterion(logits, t_y)
                        loss.backward()
                        optimizer.step()

                # Matched Query Evaluation across K in {0, 1, 2, 3, 5}
                model.eval()
                k_eval_results = {}

                with torch.no_grad():
                    for k_val in K_LEVELS:
                        eval_z_t = []
                        eval_c_p = []
                        eval_is_k0 = []

                        for ep in matched_test_episodes:
                            eval_z_t.append(ep["z_query"])
                            if k_val == 0 or ep["n_priors_available"] == 0:
                                eval_is_k0.append(True)
                                eval_c_p.append(np.zeros(d, dtype=np.float32))
                            else:
                                eval_is_k0.append(False)
                                k_use = min(k_val, ep["n_priors_available"])
                                supp = ep["all_support_z"][-k_use:]
                                eval_c_p.append(np.mean(supp, axis=0))

                        t_eval_z_t = torch.tensor(np.stack(eval_z_t), dtype=torch.float32)
                        t_eval_c_p = torch.tensor(np.stack(eval_c_p), dtype=torch.float32)
                        t_eval_is_k0 = torch.tensor(eval_is_k0, dtype=torch.bool)

                        logits = model(t_eval_z_t, t_eval_c_p, t_eval_is_k0)
                        probs = torch.sigmoid(logits).cpu().numpy()

                        auc_val = float(roc_auc_score(y_eval, probs))
                        auprc_val = float(average_precision_score(y_eval, probs))
                        brier_val = float(brier_score_loss(y_eval, probs))
                        ece_val = compute_ece(y_eval, probs)
                        fpr90_val = compute_fpr_at_sensitivity(y_eval, probs, target_sens=0.90)

                        k_eval_results[str(k_val)] = {
                            "k": k_val,
                            "coverage": coverage_dict[k_val],
                            "auroc": auc_val,
                            "auprc": auprc_val,
                            "brier": brier_val,
                            "ece": ece_val,
                            "fpr90": fpr90_val,
                        }

                k0_auc = k_eval_results["0"]["auroc"]
                k0_fpr = k_eval_results["0"]["fpr90"]
                for k_val in K_LEVELS:
                    k_eval_results[str(k_val)]["delta_auroc_vs_k0"] = k_eval_results[str(k_val)]["auroc"] - k0_auc
                    k_eval_results[str(k_val)]["delta_fpr90_vs_k0"] = k_eval_results[str(k_val)]["fpr90"] - k0_fpr

                    kc = k_eval_results[str(k_val)]
                    logger.info(
                        f"[{enc_name}|{task}|{regime}] FiLM K={k_val} (Cov: {kc['coverage']*100:.1f}%): "
                        f"AUROC={kc['auroc']:.4f} (Δ={kc['delta_auroc_vs_k0']:+.4f}), "
                        f"FPR90={kc['fpr90']:.4f} (Δ={kc['delta_fpr90_vs_k0']:+.4f})"
                    )

                task_results[regime] = {
                    "n_eval_queries": len(matched_test_episodes),
                    "n_pos_queries": int(y_eval.sum()),
                    "k_curves": k_eval_results,
                }

            all_results[enc_name][task] = task_results

    # Save Results
    out_file = args.output / "r019_film_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved R019 FiLM results to {out_file}")

    # Generate Comparison Table vs R017 PEB
    if args.r017_results.exists():
        with open(args.r017_results) as f:
            peb_res = json.load(f)

        table_lines = [
            "# Milestone R019: Learned FiLM/AdaIN vs Statistical PEB Personalization Comparison\n\n",
            "### Head-to-Head Comparison: Statistical PEB vs Learned FiLM/AdaIN at K=1 (Certified Reference Regime C2)\n",
            "| Task | Metric | SetOperator PEB | SetOperator FiLM | GraphECG PEB | GraphECG FiLM | CSFM Base PEB | CSFM Base FiLM |\n",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n",
        ]

        for task_name in ["SR->AF", "AF->SR", "MI", "STTC", "CD"]:
            if task_name in peb_res.get("SetOperator (d=256)", {}):
                reg = "certified"
                # AUROC
                so_peb_auc = peb_res["SetOperator (d=256)"][task_name][reg]["k_curves"]["1"]["auroc"]
                so_film_auc = all_results.get("SetOperator (d=256)", {}).get(task_name, {}).get(reg, {}).get("k_curves", {}).get("1", {}).get("auroc", 0.0)
                ge_peb_auc = peb_res["GraphECG (d=192)"][task_name][reg]["k_curves"]["1"]["auroc"]
                ge_film_auc = all_results.get("GraphECG (d=192)", {}).get(task_name, {}).get(reg, {}).get("k_curves", {}).get("1", {}).get("auroc", 0.0)
                cf_peb_auc = peb_res["CSFM Base (d=768)"][task_name][reg]["k_curves"]["1"]["auroc"]
                cf_film_auc = all_results.get("CSFM Base (d=768)", {}).get(task_name, {}).get(reg, {}).get("k_curves", {}).get("1", {}).get("auroc", 0.0)

                table_lines.append(f"| **{task_name}** | **AUROC** | {so_peb_auc:.4f} | {so_film_auc:.4f} | {ge_peb_auc:.4f} | {ge_film_auc:.4f} | {cf_peb_auc:.4f} | {cf_film_auc:.4f} |\n")

                # FPR90
                so_peb_fpr = peb_res["SetOperator (d=256)"][task_name][reg]["k_curves"]["1"]["fpr90"]
                so_film_fpr = all_results.get("SetOperator (d=256)", {}).get(task_name, {}).get(reg, {}).get("k_curves", {}).get("1", {}).get("fpr90", 0.0)
                ge_peb_fpr = peb_res["GraphECG (d=192)"][task_name][reg]["k_curves"]["1"]["fpr90"]
                ge_film_fpr = all_results.get("GraphECG (d=192)", {}).get(task_name, {}).get(reg, {}).get("k_curves", {}).get("1", {}).get("fpr90", 0.0)
                cf_peb_fpr = peb_res["CSFM Base (d=768)"][task_name][reg]["k_curves"]["1"]["fpr90"]
                cf_film_fpr = all_results.get("CSFM Base (d=768)", {}).get(task_name, {}).get(reg, {}).get("k_curves", {}).get("1", {}).get("fpr90", 0.0)

                table_lines.append(f"| | **FPR90** | {so_peb_fpr:.4f} | {so_film_fpr:.4f} | {ge_peb_fpr:.4f} | {ge_film_fpr:.4f} | {cf_peb_fpr:.4f} | {cf_film_fpr:.4f} |\n")

        table_path = args.output / "r019_film_vs_peb_comparison_table.md"
        with open(table_path, "w") as f:
            f.writelines(table_lines)
        logger.info(f"Saved comparison table to {table_path}")

    logger.info("\n=== Milestone R019 FiLM Benchmark COMPLETE ===")


if __name__ == "__main__":
    main()
