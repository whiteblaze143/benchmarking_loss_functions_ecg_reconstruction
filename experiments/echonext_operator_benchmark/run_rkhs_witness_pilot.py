#!/usr/bin/env python3
"""Four-Model Comparative Pilot for Phase–View RKHS Witness Network.

As mandated by Section 8 of the research directive:
Compares 4 architectural hypotheses on the identical frozen PTB-XL Phase-KME fit:
  Model 1: Existing SetOperator (High-capacity predictive reference: 1.04M classifier / 1.64M total)
  Model 2: Phase-KME + Isotropic Linear Witness on observed tokens (2,048 params/class)
  Model 3: Phase-KME + Continuous Operator-Conditioned Witness (18,432 params/class)
  Model 4: Phase-KME + Observable-Subspace Evidence Integration (18,432 params/class)

Evaluates on Fold 9 (N=2,183 authentic patients):
  - Macro AUROC & Macro AUPRC across 5 superclasses (NORM, MI, STTC, CD, HYP)
  - Probability calibration: Expected Calibration Error (ECE) and Brier Score
  - Basis Rotation Consistency: max |ell(Q) - ell(RQ)| across random orthogonal rotations R
  - Redundancy Consistency: max |ell(Q) - ell(Q_aug)| upon appending redundant linear combinations
  - Effective Trainable Parameter Count
  - Inference Latency per patient (ms)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rkhs_witness_pilot")

# Disable cudnn 1D conv bug on this environment
torch.backends.cudnn.enabled = False
torch.set_float32_matmul_precision("high")

from repecg.paper07_operator.model import OperatorSetModel


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) averaged across classes."""
    ece_list = []
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for c in range(y_true.shape[1]):
        labels = y_true[:, c]
        probs = y_prob[:, c]
        bin_indices = np.digitize(probs, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        ece = 0.0
        for b in range(n_bins):
            mask = bin_indices == b
            if np.any(mask):
                bin_acc = np.mean(labels[mask])
                bin_conf = np.mean(probs[mask])
                ece += np.sum(mask) / len(labels) * abs(bin_acc - bin_conf)
        ece_list.append(ece)
    return float(np.mean(ece_list))


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute mean Brier score across classes."""
    brier_list = [brier_score_loss(y_true[:, c], y_prob[:, c]) for c in range(y_true.shape[1])]
    return float(np.mean(brier_list))


class IsotropicLinearWitness(nn.Module):
    """Model 2: Isotropic Linear Witness on observed Phase-KME tokens.
    
    Evidence: E(p) = <w_{p, 0}, (1/m) sum_{i=1}^m mu_{q_i, p}>
    Logits:   ell = b + (1/16) sum_{p=1}^{16} E(p)
    Parameters per class: 16 * 128 = 2,048 (plus 1 bias)
    """

    def __init__(self, num_classes: int = 5, num_phases: int = 16, nystrom_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.num_phases = num_phases
        self.nystrom_dim = nystrom_dim
        # w_iso: (C, 16, 128)
        self.w_iso = nn.Parameter(torch.zeros(num_classes, num_phases, nystrom_dim))
        nn.init.normal_(self.w_iso, std=0.01)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    @property
    def num_parameters(self) -> int:
        return self.w_iso.numel() + self.bias.numel()

    def forward(self, ops: torch.Tensor, resps: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            ops: (B, m, 8) ignored for isotropic witness
            resps: (B, m, 16, 128) Phase-KME embeddings
            
        Returns:
            logits: (B, C)
        """
        # Average across observed tokens m: (B, 16, 128)
        mean_mu = torch.mean(resps, dim=1)
        # Inner product with w_iso: (B, C, 16)
        # mean_mu: (B, 1, 16, 128), w_iso: (1, C, 16, 128)
        e_p = torch.sum(mean_mu.unsqueeze(1) * self.w_iso.unsqueeze(0), dim=-1)
        # Average across 16 phases: (B, C)
        ell = torch.mean(e_p, dim=-1)
        return self.bias.unsqueeze(0) + ell


class ContinuousOperatorWitness(nn.Module):
    """Model 3 & 4: Continuous Operator-Conditioned Witness.
    
    Evidence: E(q_i, p) = <w_{p, 0}, mu_{q_i, p}> + sum_{a=1}^8 q_{i, a} <w_{p, a}, mu_{q_i, p}>
    Logits:   ell = b + (1/16) sum_{p=1}^{16} (1/m) sum_{i=1}^m E(q_i, p)
    Parameters per class: 16 * 9 * 128 = 18,432 (plus 1 bias)
    """

    def __init__(
        self,
        num_classes: int = 5,
        num_phases: int = 16,
        canonical_dim: int = 8,
        nystrom_dim: int = 128,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_phases = num_phases
        self.canonical_dim = canonical_dim
        self.nystrom_dim = nystrom_dim

        # Isotropic component: (C, 16, 128)
        self.w_iso = nn.Parameter(torch.zeros(num_classes, num_phases, nystrom_dim))
        nn.init.normal_(self.w_iso, std=0.01)

        # Directional component: (C, 16, 8, 128)
        self.w_dir = nn.Parameter(torch.zeros(num_classes, num_phases, canonical_dim, nystrom_dim))
        nn.init.normal_(self.w_dir, std=0.01)

        self.bias = nn.Parameter(torch.zeros(num_classes))

    @property
    def num_parameters(self) -> int:
        return self.w_iso.numel() + self.w_dir.numel() + self.bias.numel()

    def compute_evidence(self, q: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Compute signed evidence E(q, p) for each viewpoint and phase.
        
        Args:
            q:  (B, m, 8)
            mu: (B, m, 16, 128)
            
        Returns:
            E_qp: (B, m, C, 16)
        """
        # Isotropic term: <w_{p, 0}, mu>
        # mu: (B, m, 1, 16, 128), w_iso: (1, 1, C, 16, 128)
        e_iso = torch.sum(mu.unsqueeze(2) * self.w_iso.unsqueeze(0).unsqueeze(0), dim=-1)

        # Directional term: sum_a q_a <w_{p, a}, mu>
        # mu: (B, m, 1, 16, 1, 128), w_dir: (1, 1, C, 16, 8, 128)
        inner_a = torch.sum(mu.unsqueeze(2).unsqueeze(-2) * self.w_dir.unsqueeze(0).unsqueeze(0), dim=-1)
        # q: (B, m, 1, 1, 8)
        e_dir = torch.sum(q.unsqueeze(2).unsqueeze(2) * inner_a, dim=-1)

        return e_iso + e_dir

    def forward(
        self,
        ops: torch.Tensor,
        resps: torch.Tensor,
        observable_subspace_integration: bool = False,
    ) -> torch.Tensor:
        """Forward pass.
        
        If observable_subspace_integration is True (Model 4):
          Computes the canonical orthonormal basis of row(Q) and evaluates evidence
          over the orthonormal basis directions, providing exact basis-rotation and redundancy invariance!
        """
        if not observable_subspace_integration:
            E_qp = self.compute_evidence(ops, resps)  # (B, m, C, 16)
            E_p = torch.mean(E_qp, dim=1)            # (B, C, 16)
            ell = torch.mean(E_p, dim=-1)            # (B, C)
            return self.bias.unsqueeze(0) + ell
        else:
            # Model 4: Observable Subspace Integration
            # For each record, find canonical orthonormal basis U of row(Q)
            # Under linear measurement model, if Q is orthonormalized:
            # U = orth(Q^T)^T in R^{k x 8}
            B, m, d = ops.shape
            logits_list = []
            for b in range(B):
                Q_b = ops[b]      # (m, 8)
                mu_b = resps[b]   # (m, 16, 128)
                
                # SVD of Q_b
                U_svd, S_svd, Vh_svd = torch.linalg.svd(Q_b, full_matrices=False)
                # Numerical rank with threshold
                tol = 1e-4 * (S_svd[0] if len(S_svd) > 0 else 1.0)
                rank = int(torch.sum(S_svd > tol).item())
                rank = max(1, rank)
                
                # Canonical orthonormal directions: Vh_svd[:rank] in R^{rank, 8}
                U_can = Vh_svd[:rank]  # (k, 8)
                
                # Invertible linear re-expression coefficients:
                # Q_b = U_svd[:, :rank] @ diag(S_svd[:rank]) @ U_can
                # So U_can = diag(1/S_svd[:rank]) @ U_svd[:, :rank]^T @ Q_b
                # Under the linear subspace approximation, the canonical Phase-KME is:
                M_inv = torch.linalg.pinv(Q_b)  # (8, m)
                # Weights to express U_can as linear combination of observed rows:
                # W = U_can @ M_inv: shape (k, m)
                W = U_can @ M_inv  # (k, m)
                
                # Re-expressed canonical Phase-KME: W @ mu_b -> shape (k, 16, 128)
                # W: (k, m), mu_b: (m, 16, 128)
                mu_can = torch.einsum("km,mpe->kpe", W, mu_b)
                
                # Evaluate evidence on canonical basis
                # U_can: (1, k, 8), mu_can: (1, k, 16, 128)
                E_can = self.compute_evidence(U_can.unsqueeze(0), mu_can.unsqueeze(0))  # (1, k, C, 16)
                E_p_can = torch.mean(E_can, dim=1)  # (1, C, 16)
                ell_can = torch.mean(E_p_can, dim=-1)  # (1, C)
                logits_list.append(self.bias.unsqueeze(0) + ell_can)
                
            return torch.cat(logits_list, dim=0)


def train_linear_model(
    model: nn.Module,
    train_ops: np.ndarray,
    train_resps: np.ndarray,
    train_labels: np.ndarray,
    val_ops: np.ndarray,
    val_resps: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-2,
    wd: float = 1e-4,
    model_name: str = "LinearWitness",
) -> nn.Module:
    """Train a linear witness classifier on PTB-XL Folds 1-8 representations."""
    logger.info("Training %s (%d trainable parameters)...", model_name, model.num_parameters)
    model.to(device)

    # Class-weighted BCE loss (matching PTB-XL protocol)
    prevalence = torch.as_tensor(train_labels.mean(axis=0), dtype=torch.float32, device=device)
    pos_weight = ((1.0 - prevalence) / prevalence.clamp_min(1e-8)).clamp_max(10.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    N_train = len(train_labels)
    best_val_auroc = 0.0
    best_state = None

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        indices = np.random.permutation(N_train)
        epoch_losses = []

        for i in range(0, N_train, batch_size):
            batch_idx = indices[i : i + batch_size]
            b_ops = torch.as_tensor(train_ops[batch_idx], dtype=torch.float32, device=device)
            b_resps = torch.as_tensor(train_resps[batch_idx], dtype=torch.float32, device=device)
            b_y = torch.as_tensor(train_labels[batch_idx], dtype=torch.float32, device=device)

            # Random context cardinality sampling m in {1..8} during training
            m_ctx = random.randint(1, b_ops.shape[1])
            perm = torch.randperm(b_ops.shape[1])[:m_ctx]
            b_ops_sub = b_ops[:, perm]
            b_resps_sub = b_resps[:, perm]

            optimizer.zero_grad()
            logits = model(b_ops_sub, b_resps_sub)
            loss = loss_fn(logits, b_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        # Evaluate validation on 8 basis leads
        model.eval()
        with torch.no_grad():
            v_ops = torch.as_tensor(val_ops, dtype=torch.float32, device=device)
            v_resps = torch.as_tensor(val_resps, dtype=torch.float32, device=device)
            v_logits = model(v_ops, v_resps)
            v_probs = torch.sigmoid(v_logits).cpu().numpy()

        val_aurocs = [roc_auc_score(val_labels[:, c], v_probs[:, c]) for c in range(5)]
        mean_val_auroc = float(np.mean(val_aurocs))

        if mean_val_auroc > best_val_auroc:
            best_val_auroc = mean_val_auroc
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(
                "[%s] Epoch %2d/%d | Train Loss: %.4f | Val Macro AUROC: %.4f (Best: %.4f)",
                model_name,
                epoch,
                epochs,
                np.mean(epoch_losses),
                mean_val_auroc,
                best_val_auroc,
            )

    logger.info("[%s] Completed training in %.1fs. Best Val AUROC: %.4f", model_name, time.time() - t0, best_val_auroc)
    model.load_state_dict(best_state)
    return model


def evaluate_model_comprehensive(
    model: nn.Module,
    val_ops: np.ndarray,
    val_resps: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    model_name: str,
    observable_subspace: bool = False,
    is_setoperator: bool = False,
) -> Dict[str, Any]:
    """Comprehensive evaluation of a model on Fold 9 data.
    
    Measures:
      - Canonical 8-lead AUROC & AUPRC
      - ECE & Brier score
      - Basis Rotation Invariance (max |ell(Q) - ell(RQ)| across random orthogonal rotations R)
      - Redundancy Invariance (max |ell(Q) - ell(Q_aug)| upon appending redundant linear combinations)
      - Latency per patient (ms)
    """
    model.eval()
    N = len(val_labels)
    v_ops = torch.as_tensor(val_ops, dtype=torch.float32, device=device)
    v_resps = torch.as_tensor(val_resps, dtype=torch.float32, device=device)

    # 1. Canonical 8-lead inference & latency benchmark
    t_start = time.perf_counter()
    logits_list = []
    batch_size = 64
    with torch.inference_mode():
        for i in range(0, N, batch_size):
            b_ops = v_ops[i : i + batch_size]
            b_resps = v_resps[i : i + batch_size]
            if is_setoperator:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    b_logits = model(b_ops, b_resps, return_reconstruction=False)
            elif observable_subspace:
                b_logits = model(b_ops, b_resps, observable_subspace_integration=True)
            else:
                b_logits = model(b_ops, b_resps)
            logits_list.append(b_logits.float())
    logits_8 = torch.cat(logits_list, dim=0)
    latency_ms = (time.perf_counter() - t_start) * 1000.0 / N

    probs_8 = torch.sigmoid(logits_8).cpu().numpy()
    aurocs = [roc_auc_score(val_labels[:, c], probs_8[:, c]) for c in range(5)]
    auprcs = [average_precision_score(val_labels[:, c], probs_8[:, c]) for c in range(5)]
    macro_auroc = float(np.mean(aurocs))
    macro_auprc = float(np.mean(auprcs))
    ece = compute_ece(val_labels, probs_8)
    brier = compute_brier(val_labels, probs_8)

    # 2. Stress Test A: Basis Rotation Invariance
    # For a 4-lead subset (Leads I, II, V1, V2), apply random orthogonal rotation R in O(4)
    # Compute max absolute difference |logits(Q) - logits(RQ)|
    sub_ops = v_ops[:100, :4]      # (100, 4, 8)
    sub_resps = v_resps[:100, :4]  # (100, 4, 16, 128)

    # Generate random orthogonal matrix R in O(4)
    rng = np.random.default_rng(42)
    G = rng.normal(size=(4, 4))
    R_np, _ = np.linalg.qr(G)
    R = torch.as_tensor(R_np, dtype=torch.float32, device=device)

    rot_ops = torch.einsum("ij,bjk->bik", R, sub_ops)
    rot_resps = torch.einsum("ij,bjpe->bipe", R, sub_resps)

    with torch.no_grad():
        if is_setoperator:
            orig_sub = model(sub_ops, sub_resps, return_reconstruction=False)
            rot_sub = model(rot_ops, rot_resps, return_reconstruction=False)
        elif observable_subspace:
            orig_sub = model(sub_ops, sub_resps, observable_subspace_integration=True)
            rot_sub = model(rot_ops, rot_resps, observable_subspace_integration=True)
        else:
            orig_sub = model(sub_ops, sub_resps)
            rot_sub = model(rot_ops, rot_resps)

    max_rot_diff = float(torch.max(torch.abs(orig_sub - rot_sub)).item())

    # 3. Stress Test B: Redundancy Invariance
    # Append redundant linear combination row: q_red = 0.5 * q_1 + 0.5 * q_2
    # mu_red = 0.5 * mu_1 + 0.5 * mu_2
    q_red = 0.5 * sub_ops[:, 0:1] + 0.5 * sub_ops[:, 1:2]
    mu_red = 0.5 * sub_resps[:, 0:1] + 0.5 * sub_resps[:, 1:2]
    aug_ops = torch.cat([sub_ops, q_red], dim=1)       # (100, 5, 8)
    aug_resps = torch.cat([sub_resps, mu_red], dim=1)   # (100, 5, 16, 128)

    with torch.no_grad():
        if is_setoperator:
            aug_sub = model(aug_ops, aug_resps, return_reconstruction=False)
        elif observable_subspace:
            aug_sub = model(aug_ops, aug_resps, observable_subspace_integration=True)
        else:
            aug_sub = model(aug_ops, aug_resps)

    max_red_diff = float(torch.max(torch.abs(orig_sub - aug_sub)).item())

    # Parameter count
    if is_setoperator:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num_params = model.num_parameters

    return {
        "model_name": model_name,
        "num_params": num_params,
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "per_class_auroc": aurocs,
        "per_class_auprc": auprcs,
        "ece": ece,
        "brier_score": brier,
        "max_rotation_delta": max_rot_diff,
        "max_redundancy_delta": max_red_diff,
        "latency_per_patient_ms": latency_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Run 4-model comparative pilot for RKHS Witness Network.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output-json", type=Path, default=Path("experiments/echonext_operator_benchmark/pilot_results.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Running pilot on device: %s", device)

    # 1. Load Validation Data (PTB-XL Fold 9)
    val_path = "/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/final_representations/representation_final_select_waveforms.npz"
    logger.info("Loading Fold-9 validation representations from %s...", val_path)
    with np.load(val_path) as d_val:
        val_ops = d_val["operators"][:, :8]      # (2183, 8, 8)
        val_resps = d_val["responses"][:, :8]    # (2183, 8, 16, 128)
        val_labels = d_val["labels"]             # (2183, 5)

    # 2. Load Training Data (PTB-XL Folds 1-8)
    train_path = "/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/final_representations/representation_final_train_basis_only.npz"
    logger.info("Loading Folds 1-8 training representations from %s...", train_path)
    with np.load(train_path) as d_train:
        train_ops = d_train["operators"]         # (17417, 8, 8)
        train_resps = d_train["responses"]       # (17417, 8, 16, 128)
        train_labels = d_train["labels"]         # (17417, 5)

    results = []

    # =========================================================================
    # MODEL 1: Existing SetOperator (High-Capacity Transformer Reference)
    # =========================================================================
    logger.info("\n=== EVALUATING MODEL 1: Existing SetOperator (Locked Reference) ===")
    ckpt_path = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/ptbxl_three_way_v1/checkpoints/setoperator/span_s42.pt")
    chk = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m1 = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
    m1.load_state_dict(chk["state_dict"], strict=True)
    m1.to(device)

    res_m1 = evaluate_model_comprehensive(
        m1, val_ops, val_resps, val_labels, device,
        model_name="Model 1: Existing SetOperator (High-Capacity Reference)",
        is_setoperator=True,
    )
    results.append(res_m1)
    logger.info("Model 1 | Params: %d | Macro AUROC: %.4f | Macro AUPRC: %.4f | Rot Delta: %.4e | Red Delta: %.4e",
                res_m1["num_params"], res_m1["macro_auroc"], res_m1["macro_auprc"],
                res_m1["max_rotation_delta"], res_m1["max_redundancy_delta"])

    # =========================================================================
    # MODEL 2: Isotropic Linear Witness on observed tokens
    # =========================================================================
    logger.info("\n=== TRAINING & EVALUATING MODEL 2: Isotropic Linear Witness ===")
    m2 = IsotropicLinearWitness(num_classes=5, num_phases=16, nystrom_dim=128)
    m2 = train_linear_model(
        m2, train_ops, train_resps, train_labels, val_ops, val_resps, val_labels,
        device=device, epochs=args.epochs, lr=1e-2, wd=1e-4,
        model_name="Model 2 (Isotropic Linear Witness)",
    )
    res_m2 = evaluate_model_comprehensive(
        m2, val_ops, val_resps, val_labels, device,
        model_name="Model 2: Isotropic Linear Witness",
    )
    results.append(res_m2)
    logger.info("Model 2 | Params: %d | Macro AUROC: %.4f | Macro AUPRC: %.4f | Rot Delta: %.4e | Red Delta: %.4e",
                res_m2["num_params"], res_m2["macro_auroc"], res_m2["macro_auprc"],
                res_m2["max_rotation_delta"], res_m2["max_redundancy_delta"])

    # =========================================================================
    # MODEL 3: Continuous Operator-Conditioned Witness
    # =========================================================================
    logger.info("\n=== TRAINING & EVALUATING MODEL 3: Continuous Operator Witness ===")
    m3 = ContinuousOperatorWitness(num_classes=5, num_phases=16, canonical_dim=8, nystrom_dim=128)
    m3 = train_linear_model(
        m3, train_ops, train_resps, train_labels, val_ops, val_resps, val_labels,
        device=device, epochs=args.epochs, lr=1e-2, wd=1e-4,
        model_name="Model 3 (Continuous Operator Witness)",
    )
    res_m3 = evaluate_model_comprehensive(
        m3, val_ops, val_resps, val_labels, device,
        model_name="Model 3: Continuous Operator-Conditioned Witness",
        observable_subspace=False,
    )
    results.append(res_m3)
    logger.info("Model 3 | Params: %d | Macro AUROC: %.4f | Macro AUPRC: %.4f | Rot Delta: %.4e | Red Delta: %.4e",
                res_m3["num_params"], res_m3["macro_auroc"], res_m3["macro_auprc"],
                res_m3["max_rotation_delta"], res_m3["max_redundancy_delta"])

    # =========================================================================
    # MODEL 4: Observable-Subspace Evidence Integration
    # =========================================================================
    logger.info("\n=== EVALUATING MODEL 4: Observable-Subspace Evidence Integration ===")
    res_m4 = evaluate_model_comprehensive(
        m3, val_ops, val_resps, val_labels, device,
        model_name="Model 4: Observable-Subspace Evidence Integration",
        observable_subspace=True,
    )
    results.append(res_m4)
    logger.info("Model 4 | Params: %d | Macro AUROC: %.4f | Macro AUPRC: %.4f | Rot Delta: %.4e | Red Delta: %.4e",
                res_m4["num_params"], res_m4["macro_auroc"], res_m4["macro_auprc"],
                res_m4["max_rotation_delta"], res_m4["max_redundancy_delta"])

    # Save results to JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nSaved all pilot results to %s", args.output_json)


if __name__ == "__main__":
    main()
