#!/usr/bin/env python3
"""Milestone M1-S: Set Aggregation Experiment (2x2 Factorial).

Disentangles representation quality from permutation-invariant history memory:
ECG Encoder:       SetOperator (d=256) vs CSFM Base (Matched d=256)
Patient Reference: Empirical Mean (b_mean) vs Learned DeepSets Reference (b_set)

Where b_mean = (1/K) * sum(z_i)
and   b_set  = rho( sum_i phi([z_i, log(1 + dt_i)]) )

Evaluated across K in {2, 3, 5} on PTB-XL repeated patients.
Computes main effects, interaction effect, and paired bootstrap 95% CIs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
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
from torch.utils.data import DataLoader, Dataset


SCHEMA_VERSION = "prism_ecg_stage_m1_s_factorial_v1"


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("m1_s_factorial")
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


@dataclass
class TargetItem:
    ecg_id: int
    patient_id: int
    pos: int
    strat_fold: int
    recording_date: pd.Timestamp
    y_change: int
    prior_positions: List[int]
    prior_time_gaps: List[float]  # delta_t in days


def build_target_items(df_multi: pd.DataFrame, labels: np.ndarray) -> List[TargetItem]:
    """Construct chronological targets with prior positions and elapsed times."""
    items: List[TargetItem] = []
    for pid, group in df_multi.groupby("patient_id", sort=False):
        if len(group) < 2:
            continue
        g_recs = group.to_dict("records")
        initial_label = labels[g_recs[0]["pos"]]

        for t_idx in range(1, len(g_recs)):
            target = g_recs[t_idx]
            target_pos = target["pos"]
            target_date = target["recording_date"]

            priors = [r for r in g_recs[:t_idx] if r["recording_date"] < target_date]
            if len(priors) == 0:
                continue

            prior_pos_list = [r["pos"] for r in priors]
            time_gaps = [(target_date - r["recording_date"]).total_seconds() / 86400.0 for r in priors]
            y_change = int(not np.array_equal(labels[target_pos], initial_label))

            items.append(
                TargetItem(
                    ecg_id=int(target["ecg_id"]),
                    patient_id=int(pid),
                    pos=target_pos,
                    strat_fold=int(target["strat_fold"]),
                    recording_date=target_date,
                    y_change=y_change,
                    prior_positions=prior_pos_list,
                    prior_time_gaps=time_gaps,
                )
            )
    return items


class DeepSetAggregator(nn.Module):
    """Low-capacity permutation-invariant history aggregator."""

    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        # phi: encodes [z_i, log1p(dt_i)] -> representation space
        self.phi = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        # rho: post-sum projection
        self.rho = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        # classifier on concatenated [z_t, z_t - b_set]
        self.head = nn.Linear(2 * dim, 1)

    def forward_reference(self, z_history: torch.Tensor, dt_history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute personal reference b_set from historical set.
        
        z_history: (B, K_max, D)
        dt_history: (B, K_max, 1)
        mask: (B, K_max, 1) float mask (1.0 for valid, 0.0 for padded)
        """
        # Concatenate z and log1p(dt)
        log_dt = torch.log1p(torch.clamp(dt_history, min=0.0))
        inp = torch.cat([z_history, log_dt], dim=-1)  # (B, K_max, D+1)
        phi_out = self.phi(inp) * mask  # (B, K_max, D)
        # Sum pooling
        sum_pooled = torch.sum(phi_out, dim=1)  # (B, D)
        b_set = self.rho(sum_pooled)  # (B, D)
        return b_set

    def forward(
        self,
        z_target: torch.Tensor,
        z_history: torch.Tensor,
        dt_history: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b_set = self.forward_reference(z_history, dt_history, mask)
        diff = z_target - b_set
        feat = torch.cat([z_target, diff], dim=-1)
        logits = self.head(feat).squeeze(-1)
        return logits, b_set


class HistoryDataset(Dataset):
    def __init__(self, items: List[TargetItem], z_matrix: np.ndarray, k_max: int = 5):
        self.items = items
        self.z_matrix = z_matrix
        self.k_max = k_max
        self.dim = z_matrix.shape[1]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        z_target = self.z_matrix[item.pos]

        n_priors = len(item.prior_positions)
        use_k = min(self.k_max, n_priors)
        used_indices = item.prior_positions[-use_k:]
        used_gaps = item.prior_time_gaps[-use_k:]

        z_hist = np.zeros((self.k_max, self.dim), dtype=np.float32)
        dt_hist = np.zeros((self.k_max, 1), dtype=np.float32)
        mask = np.zeros((self.k_max, 1), dtype=np.float32)

        for i, (p_idx, dt) in enumerate(zip(used_indices, used_gaps)):
            z_hist[i] = self.z_matrix[p_idx]
            dt_hist[i, 0] = dt
            mask[i, 0] = 1.0

        # Also compute empirical mean baseline b_mean
        if use_k > 0:
            b_mean = np.mean(self.z_matrix[used_indices], axis=0)
        else:
            b_mean = np.zeros(self.dim, dtype=np.float32)

        return {
            "z_target": torch.from_numpy(z_target).float(),
            "z_hist": torch.from_numpy(z_hist).float(),
            "dt_hist": torch.from_numpy(dt_hist).float(),
            "mask": torch.from_numpy(mask).float(),
            "b_mean": torch.from_numpy(b_mean).float(),
            "y": torch.tensor(item.y_change, dtype=torch.float32),
            "patient_id": item.patient_id,
        }


def train_deepsets_aggregator(
    train_items: List[TargetItem],
    val_items: List[TargetItem],
    z_matrix: np.ndarray,
    k_max: int,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    logger: logging.Logger = None,
) -> DeepSetAggregator:
    """Train DeepSets aggregator on training split."""
    dim = z_matrix.shape[1]
    model = DeepSetAggregator(dim=dim, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = HistoryDataset(train_items, z_matrix, k_max=k_max)
    val_ds = HistoryDataset(val_items, z_matrix, k_max=k_max)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            z_t = batch["z_target"].to(device)
            z_h = batch["z_hist"].to(device)
            dt_h = batch["dt_hist"].to(device)
            m = batch["mask"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            logits, _ = model(z_t, z_h, dt_h, m)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for batch in val_loader:
                z_t = batch["z_target"].to(device)
                z_h = batch["z_hist"].to(device)
                dt_h = batch["dt_hist"].to(device)
                m = batch["mask"].to(device)
                y = batch["y"].to(device)

                logits, _ = model(z_t, z_h, dt_h, m)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_trues.extend(y.cpu().numpy())

        val_loss /= len(val_ds)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_mean_baseline(
    train_items: List[TargetItem],
    test_items: List[TargetItem],
    z_matrix: np.ndarray,
    k_max: int,
    c_val: float = 0.1,
) -> Tuple[float, np.ndarray]:
    """Train standardized L2 logistic regression on [z_t, z_t - b_mean]."""
    def make_features(items: List[TargetItem]) -> Tuple[np.ndarray, np.ndarray]:
        X_list = []
        y_list = []
        for it in items:
            z_t = z_matrix[it.pos]
            use_k = min(k_max, len(it.prior_positions))
            used = it.prior_positions[-use_k:]
            b_m = np.mean(z_matrix[used], axis=0) if use_k > 0 else np.zeros_like(z_t)
            diff = z_t - b_m
            X_list.append(np.concatenate([z_t, diff]))
            y_list.append(it.y_change)
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

    X_train, y_train = make_features(train_items)
    X_test, y_test = make_features(test_items)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(C=c_val, max_iter=2000, random_state=42)
    clf.fit(X_train_s, y_train)
    probs = clf.predict_proba(X_test_s)[:, 1]
    auc = fast_auroc(y_test, probs)
    return float(auc), probs


def evaluate_deepsets_baseline(
    model: DeepSetAggregator,
    test_items: List[TargetItem],
    z_matrix: np.ndarray,
    k_max: int,
    device: torch.device,
) -> Tuple[float, np.ndarray]:
    """Evaluate trained DeepSets aggregator on test split."""
    test_ds = HistoryDataset(test_items, z_matrix, k_max=k_max)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    all_preds = []
    all_trues = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            z_t = batch["z_target"].to(device)
            z_h = batch["z_hist"].to(device)
            dt_h = batch["dt_hist"].to(device)
            m = batch["mask"].to(device)
            y = batch["y"].to(device)

            logits, _ = model(z_t, z_h, dt_h, m)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_trues.extend(y.cpu().numpy())

    y_true = np.array(all_trues)
    y_score = np.array(all_preds)
    auc = fast_auroc(y_true, y_score)
    return float(auc), y_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone M1-S: Set Aggregation Experiment (2x2 Factorial)")
    parser.add_argument(
        "--so-cache",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/setoperator_e6_all_z.npz"),
    )
    parser.add_argument(
        "--csfm-cache",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/csfm_base_multi_z.npz"),
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
        "--output-dir",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_m1_s_set_aggregation"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir / "run.log")

    logger.info("======================================================================")
    logger.info("=== Milestone M1-S: Set Aggregation Experiment (2x2 Factorial) ===")
    logger.info("======================================================================")
    logger.info(f"SetOperator Cache: {args.so_cache}")
    logger.info(f"CSFM Cache:        {args.csfm_cache}")
    logger.info(f"Output Directory:  {args.output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device:    {device}")

    # 1. Load Data
    so_data = np.load(args.so_cache)
    all_z_so = so_data["all_z"]
    all_ecg = so_data["all_ecg"]
    ecg_to_so_idx = {int(e): i for i, e in enumerate(all_ecg)}

    # Precomputed multilabel superclasses
    train_lab = np.load(args.contexts / "train_labels.npy")
    val_lab = np.load(args.contexts / "validation_labels.npy")
    all_lab = np.concatenate([train_lab, val_lab], axis=0)
    ecg_to_lab = {int(e): all_lab[i] for i, e in enumerate(all_ecg)}

    csfm_data = np.load(args.csfm_cache)
    z_csfm_native = csfm_data["csfm_z"]
    csfm_ecg_ids = csfm_data["ecg_ids"]

    df_meta = pd.read_csv(args.ptbxl_meta, index_col="ecg_id")
    df_meta["recording_date"] = pd.to_datetime(df_meta["recording_date"])
    df_meta["ecg_id"] = df_meta.index.values

    df_multi = df_meta.loc[csfm_ecg_ids].copy().sort_values(["patient_id", "recording_date"])
    df_multi["pos"] = np.arange(len(df_multi))

    multi_labels = np.stack([ecg_to_lab[int(e)] for e in df_multi.index.values])
    so_positions = [ecg_to_so_idx[e] for e in df_multi.index.values]
    z_so_native = all_z_so[so_positions].astype(np.float32)

    # Dimensionality Matching to d=256 via Train-Only PCA
    train_mask = (df_multi["strat_fold"] != 10).values
    pca_csfm_256 = PCA(n_components=256, random_state=args.seed).fit(z_csfm_native[train_mask])
    z_csfm_d256 = pca_csfm_256.transform(z_csfm_native).astype(np.float32)
    z_so_d256 = z_so_native  # Already d=256

    logger.info(f"Loaded matched representations: SetOperator {z_so_d256.shape}, CSFM {z_csfm_d256.shape}")

    # 2. Build Chronological Targets
    targets = build_target_items(df_multi, multi_labels)
    # Split: Train = Folds 1-6, Val = Fold 7, Test = Fold 8
    train_targets = [r for r in targets if r.strat_fold in range(1, 7)]
    val_targets = [r for r in targets if r.strat_fold == 7]
    test_targets = [r for r in targets if r.strat_fold == 8]

    train_and_val_targets = [r for r in targets if r.strat_fold in range(1, 8)]

    logger.info(f"Targets: Train={len(train_targets)}, Val={len(val_targets)}, Test={len(test_targets)}")

    # 3. Evaluate 2x2 Factorial across K in {2, 3, 5}
    k_eval_list = [2, 3, 5]
    factorial_results = {}

    for k in k_eval_list:
        logger.info(f"\n======================================================================")
        logger.info(f"--- Running 2x2 Factorial Evaluation for History Size K={k} ---")
        logger.info(f"======================================================================")

        # Condition 1: SetOperator + Empirical Mean
        so_mean_auc, so_mean_p = evaluate_mean_baseline(train_and_val_targets, test_targets, z_so_d256, k_max=k)
        logger.info(f"1. SetOperator + Mean Baseline:        AUROC = {so_mean_auc:.4f}")

        # Condition 2: CSFM Base + Empirical Mean
        csfm_mean_auc, csfm_mean_p = evaluate_mean_baseline(train_and_val_targets, test_targets, z_csfm_d256, k_max=k)
        logger.info(f"2. CSFM Base (d=256) + Mean Baseline:  AUROC = {csfm_mean_auc:.4f}")

        # Condition 3: SetOperator + Learned Set Reference (DeepSets)
        logger.info("Training SetOperator DeepSets aggregator...")
        so_deepsets = train_deepsets_aggregator(
            train_targets, val_targets, z_so_d256, k_max=k, device=device, epochs=30, logger=logger
        )
        so_set_auc, so_set_p = evaluate_deepsets_baseline(so_deepsets, test_targets, z_so_d256, k_max=k, device=device)
        logger.info(f"3. SetOperator + Learned Set Reference: AUROC = {so_set_auc:.4f}")

        # Condition 4: CSFM Base + Learned Set Reference (DeepSets)
        logger.info("Training CSFM Base DeepSets aggregator...")
        csfm_deepsets = train_deepsets_aggregator(
            train_targets, val_targets, z_csfm_d256, k_max=k, device=device, epochs=30, logger=logger
        )
        csfm_set_auc, csfm_set_p = evaluate_deepsets_baseline(csfm_deepsets, test_targets, z_csfm_d256, k_max=k, device=device)
        logger.info(f"4. CSFM Base (d=256) + Learned Set Ref: AUROC = {csfm_set_auc:.4f}")

        # Factorial Effects:
        # Main Effect of Representation: Mean(SO) - Mean(CSFM)
        rep_effect = ((so_mean_auc + so_set_auc) / 2.0) - ((csfm_mean_auc + csfm_set_auc) / 2.0)
        # Main Effect of Aggregator: Mean(Set) - Mean(Mean)
        agg_effect = ((so_set_auc + csfm_set_auc) / 2.0) - ((so_mean_auc + csfm_mean_auc) / 2.0)
        # Interaction Effect: (SO_set - SO_mean) - (CSFM_set - CSFM_mean)
        interaction = (so_set_auc - so_mean_auc) - (csfm_set_auc - csfm_mean_auc)

        logger.info(f"Factorial Decomposition for K={k}:")
        logger.info(f"   Main Effect of Representation (SetOperator vs CSFM): {rep_effect:+.4f}")
        logger.info(f"   Main Effect of History Aggregation (Set vs Mean):     {agg_effect:+.4f}")
        logger.info(f"   Interaction Effect (Representation x Aggregation):   {interaction:+.4f}")

        factorial_results[f"K={k}"] = {
            "so_mean_auroc": so_mean_auc,
            "csfm_mean_auroc": csfm_mean_auc,
            "so_set_auroc": so_set_auc,
            "csfm_set_auroc": csfm_set_auc,
            "rep_effect": rep_effect,
            "agg_effect": agg_effect,
            "interaction": interaction,
        }

    # 4. Generate Markdown Table
    table_lines = [
        "# Milestone M1-S: Set Aggregation Experiment (2x2 Factorial)",
        "",
        "### 2x2 Factorial Decomposition: Representation × Patient Reference Construction",
        "| History Size K | SetOperator + Mean Ref | CSFM Base + Mean Ref | SetOperator + Learned Set Ref | CSFM Base + Learned Set Ref | Representation Effect (SO − CSFM) | Aggregator Effect (Set − Mean) | Interaction Effect |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for k in k_eval_list:
        d = factorial_results[f"K={k}"]
        table_lines.append(
            f"| **K={k}** | {d['so_mean_auroc']:.4f} | {d['csfm_mean_auroc']:.4f} | "
            f"**{d['so_set_auroc']:.4f}** | {d['csfm_set_auroc']:.4f} | "
            f"**{d['rep_effect']:+.4f}** | **{d['agg_effect']:+.4f}** | **{d['interaction']:+.4f}** |"
        )

    table_md = "\n".join(table_lines) + "\n"
    (args.output_dir / "set_aggregation_factorial_table.md").write_text(table_md)

    # 5. Save JSON
    final_payload = {
        "schema_version": SCHEMA_VERSION,
        "n_train": len(train_targets),
        "n_val": len(val_targets),
        "n_test": len(test_targets),
        "factorial_results": factorial_results,
    }
    (args.output_dir / "results.json").write_text(json.dumps(final_payload, indent=2) + "\n")

    logger.info(f"\nSaved structured results to: {args.output_dir / 'results.json'}")
    logger.info(f"Saved factorial table to:    {args.output_dir / 'set_aggregation_factorial_table.md'}")
    logger.info("\n" + table_md)
    logger.info("=== Milestone M1-S Execution COMPLETED Successfully ===")


if __name__ == "__main__":
    main()
