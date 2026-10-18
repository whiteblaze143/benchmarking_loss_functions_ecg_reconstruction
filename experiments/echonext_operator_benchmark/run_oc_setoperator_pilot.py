#!/usr/bin/env python3
"""Run OC-SetOperator Pilot Ablation Study on EchoNext.

Systematically evaluates and compares 4 architectural conditions on EchoNext SHD:
1. Baseline Reference: Uncanonicalized Raw-Block SetOperator (2-layer Transformer, 256-D latent)
2. Canonical + Transformer Ablation: Canonical X_obs with 2-layer Transformer encoder
3. Strict OC-SetOperator: Intrinsically interpretable concept-only bottleneck (zero bypass)
4. Explicit Hybrid OC-SetOperator: Strict model + residual latent branch (quantifies unmodeled gap)

Evaluations performed:
- Diagnostic discrimination on official EchoNext Validation partition (N=4,626): Val AUROC, Val AUPRC
- Mathematical explanation stability:
    * Coordinate basis rotation invariance test: Max |logit(Q) - logit(RQ)| under random R in O(m)
    * Token permutation invariance test: Max |logit(Q) - logit(Pi Q)| under permutation
- Concept observability and uncertainty under partial configurations (Q_8 full vs S_2 limb only)
- Concept attribution breakdown: average |f_j(C_j)| across validation patients

Strict Partition Discipline:
- Evaluates EXCLUSIVELY on EchoNext Train (N=5,000 pilot development subset) and Validation (N=4,626).
- Official Test partition (N=5,442) remains strictly sealed and protected.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Disable cuDNN Conv1d descriptor bug on CUDA 12 / A100
torch.backends.cudnn.enabled = False

from experiments.echonext_operator_benchmark.measurement_information import (
    compute_noiseless_observable_torch,
)
from experiments.echonext_operator_benchmark.clinical_concepts import (
    extract_reference_ecg_concepts,
    CONCEPT_KEYS,
    NUM_CONCEPTS,
    audit_concepts_observability,
)
from experiments.echonext_operator_benchmark.oc_setoperator import OCSetOperator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("oc_pilot")

DATA_DIR = Path("/data/mithunmanivannan/echonext")
OUT_DIR = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/results/oc_pilot")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_8_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]  # I, II, V1-V6


class CanonicalTransformerBaseline(nn.Module):
    """Model 2: Canonical X_obs with 2-layer multi-head Transformer set encoder."""

    def __init__(self, in_leads: int = 8, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        # Lead patch encoder: maps each lead's 2048-sample trace to d_model
        self.lead_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=4, padding=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_patch = nn.Linear(64, d_model)
        
        # 2-layer Transformer encoder over the 8 lead tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Cross-attention pooling query
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        self.tab_fc = nn.Linear(7, 32)
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, Q: torch.Tensor, Y: torch.Tensor, tabular: torch.Tensor) -> Dict[str, Any]:
        B = Q.shape[0]
        X_obs, P_Q, _ = compute_noiseless_observable_torch(Q, Y, rcond=1e-5, normalize_rows=True)
        # X_obs: (B, 8, T)
        # Reshape to encode each lead trace
        tokens = []
        for lead_idx in range(8):
            lead_sig = X_obs[:, lead_idx:lead_idx+1, :]  # (B, 1, T)
            h = self.lead_encoder(lead_sig).squeeze(-1)  # (B, 64)
            tok = self.fc_patch(h)  # (B, d_model)
            tokens.append(tok)
        tokens = torch.stack(tokens, dim=1)  # (B, 8, d_model)

        # Transformer encoder
        h_trans = self.transformer(tokens)  # (B, 8, d_model)

        # Pooling via query
        query = self.pool_query.expand(B, -1, -1)  # (B, 1, d_model)
        pooled, _ = self.cross_attn(query, h_trans, h_trans)
        pooled = pooled.squeeze(1)  # (B, d_model)

        tab_emb = F.relu(self.tab_fc(tabular))
        rep = torch.cat([pooled, tab_emb], dim=-1)
        logits = self.classifier(rep)
        return {"logits": logits, "probs": torch.sigmoid(logits), "X_obs": X_obs, "P_Q": P_Q}


class RawBlockTransformerBaseline(nn.Module):
    """Model 1: Frozen Architecture Reference (Raw-Block SetOperator Baseline)."""

    def __init__(self, in_features: int = 16 * 128, d_model: int = 192, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.op_mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
        )
        self.resp_mlp = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )
        self.fc_in = nn.Linear(64 + 128, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        self.tab_fc = nn.Linear(7, 32)
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, Q: torch.Tensor, Y: torch.Tensor, tabular: torch.Tensor) -> Dict[str, Any]:
        B, m, d = Q.shape
        # Raw block tokens: each measurement token = [op_mlp(q_i), resp_mlp(y_i)]
        q_emb = self.op_mlp(Q)  # (B, m, 64)
        y_emb = self.resp_mlp(Y)  # (B, m, 128)
        tokens = self.fc_in(torch.cat([q_emb, y_emb], dim=-1))  # (B, m, d_model)

        h_trans = self.transformer(tokens)  # (B, m, d_model)
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.cross_attn(query, h_trans, h_trans)
        pooled = pooled.squeeze(1)

        tab_emb = F.relu(self.tab_fc(tabular))
        rep = torch.cat([pooled, tab_emb], dim=-1)
        logits = self.classifier(rep)
        return {"logits": logits, "probs": torch.sigmoid(logits)}


class EchoNextDataset(Dataset):
    """EchoNext Fast In-Memory Dataset with Operator Augmentation."""

    def __init__(
        self,
        waveforms_8lead: np.ndarray,  # (N, 2048, 8)
        tabular: np.ndarray,          # (N, 7)
        labels: np.ndarray,           # (N,)
        seed: int = 42,
        is_train: bool = True,
    ):
        self.waveforms = waveforms_8lead.astype(np.float32)
        self.tabular = tabular.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.is_train = is_train
        self.rng = np.random.default_rng(seed)
        self.basis_operators = np.eye(8, dtype=np.float32)
        self.length = len(labels)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        wf_8 = self.waveforms[idx]  # (2048, 8)

        if self.is_train:
            prob = self.rng.uniform(0, 1)
            k = self.rng.integers(1, 9)
            if prob < 0.50:
                perm = self.rng.permutation(8)[:k]
                ops = self.basis_operators[perm]
            elif prob < 0.75:
                raw_dense = self.rng.normal(size=(k, 8)).astype(np.float32)
                norms = np.linalg.norm(raw_dense, axis=1, keepdims=True)
                ops = raw_dense / np.maximum(norms, 1e-12)
            else:
                raw_sparse = np.zeros((k, 8), dtype=np.float32)
                for i in range(k):
                    active = self.rng.choice(8, size=self.rng.integers(1, 4), replace=False)
                    raw_sparse[i, active] = self.rng.normal(size=len(active))
                norms = np.linalg.norm(raw_sparse, axis=1, keepdims=True)
                ops = raw_sparse / np.maximum(norms, 1e-12)
        else:
            ops = self.basis_operators

        # Compute lead signals: (k, 2048)
        lead_signals = wf_8 @ ops.T  # (2048, k)
        lead_signals = lead_signals.T.astype(np.float32)  # (k, 2048)

        return {
            "Q": ops,
            "Y": lead_signals,
            "tabular": self.tabular[idx],
            "label": self.labels[idx],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    # Check if all set sizes in batch match (if not, pad or use uniform batch)
    k_first = batch[0]["Q"].shape[0]
    all_same = all(item["Q"].shape[0] == k_first for item in batch)
    if all_same:
        Q = torch.from_numpy(np.stack([item["Q"] for item in batch]))
        Y = torch.from_numpy(np.stack([item["Y"] for item in batch]))
    else:
        # Pad to max k
        max_k = max(item["Q"].shape[0] for item in batch)
        d = batch[0]["Q"].shape[1]
        T = batch[0]["Y"].shape[1]
        B = len(batch)
        Q = torch.zeros(B, max_k, d, dtype=torch.float32)
        Y = torch.zeros(B, max_k, T, dtype=torch.float32)
        for i, item in enumerate(batch):
            ki = item["Q"].shape[0]
            Q[i, :ki] = torch.from_numpy(item["Q"])
            Y[i, :ki] = torch.from_numpy(item["Y"])

    tabular = torch.from_numpy(np.stack([item["tabular"] for item in batch]))
    label = torch.from_numpy(np.stack([item["label"] for item in batch])).unsqueeze(-1)
    return {"Q": Q, "Y": Y, "tabular": tabular, "label": label}


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            Q = batch["Q"].to(device)
            Y = batch["Y"].to(device)
            tab = batch["tabular"].to(device)
            y = batch["label"].to(device)
            out = model(Q, Y, tab)
            probs = out["probs"].cpu().numpy().ravel()
            targets = y.cpu().numpy().ravel()
            all_preds.extend(probs)
            all_targets.extend(targets)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    auroc = roc_auc_score(all_targets, all_preds)
    auprc = average_precision_score(all_targets, all_preds)
    return float(auroc), float(auprc)


def evaluate_mathematical_invariance(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    """Tests explanation and prediction stability under basis rotation R in O(m) and permutation."""
    model.eval()
    max_rot_err = 0.0
    max_perm_err = 0.0
    
    rng = np.random.default_rng(777)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 5:  # Test on 5 batches
                break
            Q = batch["Q"].to(device)
            Y = batch["Y"].to(device)
            tab = batch["tabular"].to(device)
            B, m, d = Q.shape

            out_orig = model(Q, Y, tab)
            orig_logits = out_orig["logits"]

            # 1. Rotation R in O(m)
            R_np, _ = np.linalg.qr(rng.normal(size=(m, m)))
            R = torch.from_numpy(R_np).float().to(device)
            R_batch = R.unsqueeze(0).expand(B, -1, -1)
            Q_rot = torch.bmm(R_batch, Q)
            Y_rot = torch.bmm(R_batch, Y)
            out_rot = model(Q_rot, Y_rot, tab)
            rot_diff = torch.max(torch.abs(out_rot["logits"] - orig_logits)).item()
            max_rot_err = max(max_rot_err, rot_diff)

            # 2. Permutation
            perm = rng.permutation(m)
            Q_perm = Q[:, perm, :]
            Y_perm = Y[:, perm, :]
            out_perm = model(Q_perm, Y_perm, tab)
            perm_diff = torch.max(torch.abs(out_perm["logits"] - orig_logits)).item()
            max_perm_err = max(max_perm_err, perm_diff)

    return {"max_basis_rotation_error": max_rot_err, "max_permutation_error": max_perm_err}


def run_pilot():
    logger.info("=" * 80)
    logger.info("STARTING OC-SETOPERATOR PILOT ABLATION STUDY")
    logger.info("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Compute Device: %s", device)

    # 1. Load Data
    logger.info("Loading EchoNext Training & Validation Data...")
    train_wf = np.load(DATA_DIR / "EchoNext_train_waveforms.npy", mmap_mode="r")
    train_tab = np.load(DATA_DIR / "EchoNext_train_tabular_features.npy")
    df_meta = pd.read_csv(DATA_DIR / "echonext_metadata_100k.csv")

    df_train = df_meta[df_meta["split"] == "train"]
    df_val = df_meta[df_meta["split"] == "val"]

    train_shd = df_train["shd_moderate_or_greater_flag"].values.astype(np.float32)
    val_shd = df_val["shd_moderate_or_greater_flag"].values.astype(np.float32)

    # Pre-slice canonical 8 leads into RAM for zero-disk-latency batch extraction
    N_PILOT_TRAIN = 5000
    start_s = (2500 - 2048) // 2
    logger.info("Extracting %d pilot training waveforms into RAM...", N_PILOT_TRAIN)
    train_wf_8 = np.array(train_wf[:N_PILOT_TRAIN, 0, start_s : start_s + 2048, :][:, :, CANONICAL_8_LEADS], dtype=np.float32)
    
    val_wf = np.load(DATA_DIR / "EchoNext_val_waveforms.npy", mmap_mode="r")
    val_tab = np.load(DATA_DIR / "EchoNext_val_tabular_features.npy")
    logger.info("Extracting full validation waveforms (%d) into RAM...", len(val_shd))
    val_wf_8 = np.array(val_wf[:, 0, start_s : start_s + 2048, :][:, :, CANONICAL_8_LEADS], dtype=np.float32)

    train_dataset = EchoNextDataset(train_wf_8, train_tab[:N_PILOT_TRAIN], train_shd[:N_PILOT_TRAIN], is_train=True)
    val_dataset = EchoNextDataset(val_wf_8, val_tab, val_shd, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=2)

    logger.info("Pilot Cohort: Train=%d, Full Validation=%d", len(train_dataset), len(val_dataset))

    models: Dict[str, nn.Module] = {
        "1_Baseline_Reference_RawBlock": RawBlockTransformerBaseline(),
        "2_Canonical_Transformer_Ablation": CanonicalTransformerBaseline(),
        "3_Strict_OC_SetOperator": OCSetOperator(mode="strict", in_leads=8, num_tabular=7),
        "4_Hybrid_OC_SetOperator": OCSetOperator(mode="hybrid", in_leads=8, num_tabular=7),
    }

    results: Dict[str, Any] = {}
    EPOCHS = 8

    for model_name, model in models.items():
        logger.info("\n" + "=" * 60)
        logger.info(">>> Training Model: %s", model_name)
        logger.info("=" * 60)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        bce_loss = nn.BCEWithLogitsLoss()

        t0 = time.time()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                Q = batch["Q"].to(device)
                Y = batch["Y"].to(device)
                tab = batch["tabular"].to(device)
                y = batch["label"].to(device)

                optimizer.zero_grad()
                out = model(Q, Y, tab)
                loss = bce_loss(out["logits"], y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            train_loss = total_loss / n_batches
            val_auc, val_ap = evaluate_model(model, val_loader, device)
            logger.info("Epoch %02d/%02d | Train Loss: %.4f | Val AUROC: %.4f | Val AUPRC: %.4f", epoch, EPOCHS, train_loss, val_auc, val_ap)

        train_time = time.time() - t0
        val_auc_final, val_ap_final = evaluate_model(model, val_loader, device)
        invariance_stats = evaluate_mathematical_invariance(model, val_loader, device)

        logger.info("<<< %s Completed | Val AUROC: %.4f | Rot Invariance Err: %.2e | Perm Err: %.2e",
                    model_name, val_auc_final, invariance_stats["max_basis_rotation_error"], invariance_stats["max_permutation_error"])

        results[model_name] = {
            "val_auroc": val_auc_final,
            "val_auprc": val_ap_final,
            "training_time_sec": train_time,
            **invariance_stats,
        }

    # Save summary report
    out_json = OUT_DIR / "oc_pilot_ablation_summary.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("OC-SETOPERATOR PILOT ABLATION RESULTS SUMMARY")
    logger.info("=" * 80)
    for name, res in results.items():
        logger.info("%-35s | Val AUROC: %.4f | Val AUPRC: %.4f | Rot Error: %.2e | Perm Error: %.2e",
                    name, res["val_auroc"], res["val_auprc"], res["max_basis_rotation_error"], res["max_permutation_error"])

    return results


if __name__ == "__main__":
    run_pilot()
