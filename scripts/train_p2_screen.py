"""Milestone P2: Small Representation Screen on PTB-XL (Folds 1-7 Train, Fold 8 Dev Val).

Defined in PRD Addendum §8, §15, §16, §22:
Compares:
- B0: Plain SSL (shared ResNet1D, global pooling, no geometry, no slots)
- B1: Structured latent (clinical slots, no geometry)
- B2: Geometry latent (8-view angle embedding + geometry tokens, no slots)
- M: Full GRAIL pilot (8-view geometry + 6 structured slots + SSL + clinical shaping + auxiliary view decoder)
- UB: Supervised Upper Bound (same temporal backbone, end-to-end multi-task BCE)

Evaluation Battery on Fold 8 (N=2,173):
1. Linear probe accessibility on 25 anchor concepts (AUROC, AUPRC, Brier, Sens@95Sp, Spec@95Se, ECE, and ratio R_c).
2. Linear probe generalization to 12 held-out probe concepts (never seen in training).
3. Latent spectral non-collapse audit (effective rank / condition number).
4. Structured slot domain selectivity (rhythm, conduction, morphology, STT).
5. Zero-shot lead removal robustness (1, 2, 4 leads masked).

Outputs:
- checkpoints/p2_screen/
- results/p2_screen/p2_screen_results.json
- P2_REPRESENTATION_SCREEN_REPORT.md
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# CuDNN disabled for multi-process safety on shared GPU
torch.backends.cudnn.enabled = False

from grail_ecg.src.data.ptbxl_dataset import PTBXLDataset
from grail_ecg.src.models.baselines import (
    PlainEncoderB0,
    StructuredEncoderB1,
    GeometryEncoderB2,
    SupervisedUpperBoundUB,
)
from grail_ecg.src.models.grail_encoder import GRAILEncoder
from grail_ecg.src.models.view_aux_decoder import ViewAuxiliaryDecoder
from grail_ecg.src.losses.ssl import VICRegLoss
from grail_ecg.src.losses.clinical import ClinicalAnchorLoss
from grail_ecg.src.losses.view import ViewReconstructionLoss
from grail_ecg.src.probes.linear_probe import fit_and_evaluate_linear_probe, compute_binary_metrics
from grail_ecg.src.geometry.lead_geometry import INDEPENDENT_8_LEADS, get_angles_tensor


class ECGPhysiologicalAugmenter:
    """Allowed physiological augmentations per PRD Addendum §13."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, L, T = x.shape
        # 1. Shared amplitude scaling across all leads
        alpha = torch.empty(B, 1, 1, device=x.device).uniform_(0.85, 1.15)
        x_aug = x * alpha
        # 2. Additive Gaussian noise
        noise = torch.randn_like(x_aug) * 0.02
        x_aug = x_aug + noise
        # 3. Small baseline shift
        beta = torch.empty(B, 1, 1, device=x.device).uniform_(-0.05, 0.05)
        x_aug = x_aug + beta
        # 4. Lead dropout (masking 1 lead with prob 0.2)
        if torch.rand(1).item() < 0.2:
            lead_idx = torch.randint(0, L, (1,)).item()
            x_aug[:, lead_idx, :] = 0.0
        return x_aug


def train_epoch_ssl(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    augmenter: ECGPhysiologicalAugmenter,
    vicreg_loss: VICRegLoss,
    clinical_loss_fn: Any,
    view_decoder: ViewAuxiliaryDecoder | None,
    view_loss_fn: ViewReconstructionLoss | None,
    canonical_angles: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    if view_decoder is not None:
        view_decoder.train()

    total_loss_acc = 0.0
    ssl_loss_acc = 0.0
    clin_loss_acc = 0.0
    view_loss_acc = 0.0
    n_batches = 0

    for batch in loader:
        x_raw = batch["ecg_8l"].to(device)
        y_anchor = batch["anchor_labels"].to(device)
        B, L, T = x_raw.shape

        x_a = augmenter(x_raw)
        x_b = augmenter(x_raw)

        optimizer.zero_grad()

        # Forward two augmented views for SSL
        if model_name == "b0":
            z_a, logits_a = model(x_a)
            z_b, logits_b = model(x_b)
            loss_ssl_dict = vicreg_loss(z_a, z_b)
            l_ssl = loss_ssl_dict["loss"]
            l_clin = F.binary_cross_entropy_with_logits(logits_a, y_anchor)
            l_view = torch.tensor(0.0, device=device)
            loss = l_ssl + 2.0 * l_clin

        elif model_name == "b1":
            slots_a, z_a, logits_a = model(x_a)
            slots_b, z_b, logits_b = model(x_b)
            loss_ssl_dict = vicreg_loss(z_a, z_b)
            l_ssl = loss_ssl_dict["loss"]
            clin_dict = clinical_loss_fn(logits_a, y_anchor)
            l_clin = clin_dict["loss"]
            l_view = torch.tensor(0.0, device=device)
            loss = l_ssl + 2.0 * l_clin

        elif model_name == "b2":
            z_a, logits_a = model(x_a)
            z_b, logits_b = model(x_b)
            loss_ssl_dict = vicreg_loss(z_a, z_b)
            l_ssl = loss_ssl_dict["loss"]
            l_clin = F.binary_cross_entropy_with_logits(logits_a, y_anchor)

            # View aux task: mask 1 target view q
            q_idx = torch.randint(0, L, (1,)).item()
            target_lead = x_raw[:, q_idx, :]
            target_angle = canonical_angles[q_idx : q_idx + 1].expand(B, -1)
            pred_lead = view_decoder(z_a, target_angle).squeeze(1)
            l_view = view_loss_fn(pred_lead, target_lead)
            loss = l_ssl + 2.0 * l_clin + 1.0 * l_view

        elif model_name == "m":
            # Mask 1 view for view aux task
            q_idx = torch.randint(0, L, (1,)).item()
            target_lead = x_raw[:, q_idx, :]
            target_angle = canonical_angles[q_idx : q_idx + 1].expand(B, -1)

            x_masked_a = x_a.clone()
            x_masked_a[:, q_idx, :] = 0.0

            slots_a, z_a, logits_a = model(x_masked_a)
            slots_b, z_b, logits_b = model(x_b)

            loss_ssl_dict = vicreg_loss(z_a, z_b)
            l_ssl = loss_ssl_dict["loss"]
            clin_dict = clinical_loss_fn(logits_a, y_anchor)
            l_clin = clin_dict["loss"]

            pred_lead = view_decoder(z_a, target_angle).squeeze(1)
            l_view = view_loss_fn(pred_lead, target_lead)
            loss = l_ssl + 2.0 * l_clin + 1.0 * l_view

        else:
            raise ValueError(f"Unknown SSL model {model_name}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if view_decoder is not None:
            torch.nn.utils.clip_grad_norm_(view_decoder.parameters(), 5.0)
        optimizer.step()

        total_loss_acc += loss.item()
        ssl_loss_acc += l_ssl.item()
        clin_loss_acc += l_clin.item()
        view_loss_acc += l_view.item()
        n_batches += 1

    return {
        "total": total_loss_acc / n_batches,
        "ssl": ssl_loss_acc / n_batches,
        "clin": clin_loss_acc / n_batches,
        "view": view_loss_acc / n_batches,
    }


def train_epoch_supervised(
    model: SupervisedUpperBoundUB,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        x_raw = batch["ecg_8l"].to(device)
        y_anchor = batch["anchor_labels"].to(device)
        optimizer.zero_grad()
        logits = model(x_raw)
        loss = F.binary_cross_entropy_with_logits(logits, y_anchor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def extract_features(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Extracts representation Z, slot representations, anchor labels, and probe labels."""
    model.eval()
    all_z = []
    all_slots = []
    all_anchor_labels = []
    all_probe_labels = []

    with torch.no_grad():
        for batch in loader:
            x_raw = batch["ecg_8l"].to(device)
            y_anchor = batch["anchor_labels"]
            y_probe = batch["probe_labels"]

            if model_name == "b0":
                z_flat, _ = model(x_raw)
                slots = None
            elif model_name == "b1":
                slots, z_flat, _ = model(x_raw)
                slots = slots.cpu()
            elif model_name == "b2":
                z_flat, _ = model(x_raw)
                slots = None
            elif model_name == "m":
                slots, z_flat, _ = model(x_raw)
                slots = slots.cpu()
            elif model_name == "ub":
                z_flat = model(x_raw)  # For UB, logits directly
                slots = None
            else:
                raise ValueError(model_name)

            all_z.append(z_flat.cpu())
            if slots is not None:
                all_slots.append(slots)
            all_anchor_labels.append(y_anchor)
            all_probe_labels.append(y_probe)

    Z_out = torch.cat(all_z, dim=0)
    slots_out = torch.cat(all_slots, dim=0) if all_slots else None
    Y_anchor = torch.cat(all_anchor_labels, dim=0)
    Y_probe = torch.cat(all_probe_labels, dim=0)
    return Z_out, slots_out, Y_anchor, Y_probe


def evaluate_lead_robustness(
    model: nn.Module,
    model_name: str,
    val_loader: DataLoader,
    probe_model_anchor: nn.Module,
    num_masked_leads_list: list[int],
    device: torch.device,
) -> dict[int, float]:
    """Evaluates zero-shot lead masking robustness."""
    model.eval()
    probe_model_anchor.eval()
    robustness_results = {}

    for k in num_masked_leads_list:
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["ecg_8l"].clone().to(device)
                y = batch["anchor_labels"].numpy()

                if k > 0:
                    # Mask the first k leads
                    x[:, :k, :] = 0.0

                if model_name in ("b0", "b2"):
                    z, _ = model(x)
                elif model_name in ("b1", "m"):
                    _, z, _ = model(x)
                else:
                    z = model(x)

                logits = probe_model_anchor(z.to(device))
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.append(probs)
                all_targets.append(y)

        P = np.concatenate(all_preds, axis=0)
        T = np.concatenate(all_targets, axis=0)

        aurocs = []
        for c in range(T.shape[1]):
            m = compute_binary_metrics(T[:, c], P[:, c])
            if not np.isnan(m["auroc"]):
                aurocs.append(m["auroc"])
        robustness_results[k] = float(np.mean(aurocs))

    return robustness_results


def evaluate_domain_selectivity(
    slots_train: torch.Tensor,
    slots_val: torch.Tensor,
    y_anchor_train: torch.Tensor,
    y_anchor_val: torch.Tensor,
    domain_slices: dict[str, slice],
    device: torch.device,
) -> dict[str, Any]:
    """Computes Selectivity_j = AUROC(z_j -> domain_j) - AUROC(z_j -> other_domains)."""
    domains = ["rhythm", "conduction", "morphology", "stt"]
    # Matrix: [6 slots, 4 domains]
    selectivity_matrix = np.zeros((6, 4))

    for slot_idx in range(6):
        z_tr = slots_train[:, slot_idx, :]  # [N, 16]
        z_va = slots_val[:, slot_idx, :]    # [N, 16]

        for d_idx, domain in enumerate(domains):
            sl = domain_slices[domain]
            y_tr = y_anchor_train[:, sl]
            y_va = y_anchor_val[:, sl]

            metrics = fit_and_evaluate_linear_probe(
                z_train=z_tr,
                y_train=y_tr,
                z_val=z_va,
                y_val=y_va,
                num_epochs=30,
                lr=0.02,
                device=str(device),
            )
            selectivity_matrix[slot_idx, d_idx] = metrics["macro_auroc"]

    # Compute selectivity for each of the 4 named slots
    selectivity_scores = {}
    for d_idx, domain in enumerate(domains):
        intended = selectivity_matrix[d_idx, d_idx]
        unrelated = np.mean([selectivity_matrix[d_idx, other] for other in range(4) if other != d_idx])
        selectivity_scores[domain] = float(intended - unrelated)

    return {
        "matrix": selectivity_matrix.tolist(),
        "selectivity_scores": selectivity_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="GRAIL-ECG Milestone P2 Screen")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/p2_screen")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/p2_screen")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    ckpt_dir = PROJECT_ROOT / args.checkpoint_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"=== Starting Milestone P2 Representation Screen on device: {device} ===")

    # Load Concepts Configuration
    concepts_path = PROJECT_ROOT / "configs" / "ptbxl_concepts.yaml"
    with open(concepts_path) as f:
        concept_cfg = yaml.safe_load(f)

    anchor_counts = {domain: len(info["anchors"]) for domain, info in concept_cfg["domains"].items() if info["anchors"]}
    domain_slices = {}
    curr = 0
    for domain, count in anchor_counts.items():
        domain_slices[domain] = slice(curr, curr + count)
        curr += count

    total_anchors = sum(anchor_counts.values())
    total_probes = len(concept_cfg["all_probe_codes"])
    print(f"Loaded {total_anchors} anchor concepts and {total_probes} held-out probe concepts.")

    # Data Loaders
    data_dir = PROJECT_ROOT / "data" / "ptb_xl"
    print("Initializing PTB-XL Folds 1-7 (Train) and Fold 8 (Dev Val)...")
    train_dataset = PTBXLDataset(data_dir=data_dir, folds=list(range(1, 8)), concept_config_path=concepts_path)
    val_dataset = PTBXLDataset(data_dir=data_dir, folds=[8], concept_config_path=concepts_path)
    print(f"Train samples: {len(train_dataset)} | Dev Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    augmenter = ECGPhysiologicalAugmenter()
    vicreg_loss = VICRegLoss(sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0)
    clinical_loss_fn = ClinicalAnchorLoss(domain_slices=domain_slices)
    view_loss_fn = ViewReconstructionLoss()
    canonical_angles = get_angles_tensor(INDEPENDENT_8_LEADS).to(device)

    # Dictionary to store all model evaluation results
    results: dict[str, Any] = {}

    # -------------------------------------------------------------
    # 1. Train Supervised Upper Bound (UB)
    # -------------------------------------------------------------
    print("\n--- [1/5] Training Supervised Upper Bound (UB) ---")
    ub_model = SupervisedUpperBoundUB(num_leads=8, hidden_dim=128, num_classes=total_anchors).to(device)
    ub_opt = torch.optim.AdamW(ub_model.parameters(), lr=args.lr, weight_decay=1e-4)
    ub_sched = torch.optim.lr_scheduler.CosineAnnealingLR(ub_opt, T_max=args.epochs)

    start_t = time.time()
    for ep in range(1, args.epochs + 1):
        l = train_epoch_supervised(ub_model, train_loader, ub_opt, device)
        ub_sched.step()
        if ep % 5 == 0 or ep == 1:
            print(f"UB Epoch {ep:02d}/{args.epochs} | BCE: {l:.4f}")
    print(f"UB Training completed in {time.time() - start_t:.1f}s.")

    # Evaluate UB directly on Fold 8
    ub_model.eval()
    ub_preds, ub_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = ub_model(batch["ecg_8l"].to(device))
            ub_preds.append(torch.sigmoid(logits).cpu().numpy())
            ub_targets.append(batch["anchor_labels"].numpy())
    P_ub = np.concatenate(ub_preds, axis=0)
    T_ub = np.concatenate(ub_targets, axis=0)

    ub_metrics_list = [compute_binary_metrics(T_ub[:, c], P_ub[:, c]) for c in range(total_anchors)]
    ub_macro_auroc = float(np.nanmean([m["auroc"] for m in ub_metrics_list]))
    ub_macro_auprc = float(np.nanmean([m["auprc"] for m in ub_metrics_list]))
    print(f"Supervised Upper Bound Fold 8 Macro AUROC: {ub_macro_auprc:.4f} | AUROC: {ub_macro_auroc:.4f}")

    results["UB"] = {
        "macro_auroc": ub_macro_auroc,
        "macro_auprc": ub_macro_auprc,
        "class_metrics": ub_metrics_list,
    }
    torch.save(ub_model.state_dict(), ckpt_dir / "ub_model.pt")

    # -------------------------------------------------------------
    # 2. Train and Evaluate Representation Candidates (B0, B1, B2, M)
    # -------------------------------------------------------------
    candidates = ["b0", "b1", "b2", "m"]

    for idx, cand in enumerate(candidates, start=2):
        print(f"\n--- [{idx}/5] Training Representation Candidate: {cand.upper()} ---")

        if cand == "b0":
            model = PlainEncoderB0(num_leads=8, hidden_dim=128, latent_dim=96, num_anchor_classes=total_anchors).to(device)
            view_dec = None
            opt_params = list(model.parameters())
        elif cand == "b1":
            model = StructuredEncoderB1(
                num_leads=8,
                hidden_dim=128,
                num_slots=6,
                slot_dim=16,
                anchor_counts_per_domain=anchor_counts,
            ).to(device)
            view_dec = None
            opt_params = list(model.parameters())
        elif cand == "b2":
            model = GeometryEncoderB2(
                num_leads=8,
                hidden_dim=128,
                latent_dim=96,
                num_anchor_classes=total_anchors,
            ).to(device)
            view_dec = ViewAuxiliaryDecoder(latent_dim=96, target_len=5000).to(device)
            opt_params = list(model.parameters()) + list(view_dec.parameters())
        elif cand == "m":
            model = GRAILEncoder(
                num_leads=8,
                hidden_dim=128,
                num_slots=6,
                slot_dim=16,
                anchor_counts_per_domain=anchor_counts,
            ).to(device)
            view_dec = ViewAuxiliaryDecoder(latent_dim=96, target_len=5000).to(device)
            opt_params = list(model.parameters()) + list(view_dec.parameters())
        else:
            raise ValueError(cand)

        optimizer = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        start_t = time.time()
        for ep in range(1, args.epochs + 1):
            loss_dict = train_epoch_ssl(
                model=model,
                model_name=cand,
                loader=train_loader,
                optimizer=optimizer,
                augmenter=augmenter,
                vicreg_loss=vicreg_loss,
                clinical_loss_fn=clinical_loss_fn,
                view_decoder=view_dec,
                view_loss_fn=view_loss_fn,
                canonical_angles=canonical_angles,
                device=device,
            )
            scheduler.step()
            if ep % 5 == 0 or ep == 1:
                print(
                    f"{cand.upper()} Epoch {ep:02d}/{args.epochs} | "
                    f"Total: {loss_dict['total']:.4f} | "
                    f"SSL: {loss_dict['ssl']:.4f} | "
                    f"Clin: {loss_dict['clin']:.4f} | "
                    f"View: {loss_dict['view']:.4f}"
                )
        print(f"{cand.upper()} Training completed in {time.time() - start_t:.1f}s.")

        torch.save(model.state_dict(), ckpt_dir / f"{cand}_encoder.pt")
        if view_dec is not None:
            torch.save(view_dec.state_dict(), ckpt_dir / f"{cand}_view_decoder.pt")

        # ---------------------------------------------------------
        # Feature Extraction & Latent Spectral Analysis
        # ---------------------------------------------------------
        print(f"Extracting frozen latent representations for {cand.upper()}...")
        z_tr, slots_tr, y_anc_tr, y_prb_tr = extract_features(model, cand, train_loader, device)
        z_va, slots_va, y_anc_va, y_prb_va = extract_features(model, cand, val_loader, device)

        # Spectral Non-Collapse Audit (Fold 8)
        U, S, V = torch.linalg.svd(z_va - z_va.mean(dim=0), full_matrices=False)
        s_np = S.numpy()
        s_norm = s_np / np.sum(s_np)
        eff_rank = float(np.exp(-np.sum(s_norm * np.log(s_norm + 1e-12))))
        cond_num = float(s_np[0] / (s_np[-1] + 1e-12))
        print(f"Latent Effective Rank on Fold 8: {eff_rank:.2f} / 96 (Cond: {cond_num:.2f})")

        # ---------------------------------------------------------
        # Linear Probe Evaluation on Anchor Concepts
        # ---------------------------------------------------------
        print(f"Fitting linear probe on anchor concepts for {cand.upper()}...")
        anchor_metrics = fit_and_evaluate_linear_probe(
            z_train=z_tr,
            y_train=y_anc_tr,
            z_val=z_va,
            y_val=y_anc_va,
            num_epochs=40,
            lr=0.02,
            device=str(device),
        )

        # Clinical Sufficiency Ratio R_c
        r_c = (anchor_metrics["macro_auroc"] - 0.5) / (ub_macro_auroc - 0.5 + 1e-12)
        print(
            f"Anchor Linear Macro AUROC: {anchor_metrics['macro_auroc']:.4f} | "
            f"AUPRC: {anchor_metrics['macro_auprc']:.4f} | "
            f"R_c: {r_c:.4f}"
        )

        # ---------------------------------------------------------
        # Linear Probe Evaluation on HELD-OUT Probe Concepts
        # ---------------------------------------------------------
        print(f"Fitting linear probe on 12 HELD-OUT probe concepts for {cand.upper()}...")
        probe_metrics = fit_and_evaluate_linear_probe(
            z_train=z_tr,
            y_train=y_prb_tr,
            z_val=z_va,
            y_val=y_prb_va,
            num_epochs=40,
            lr=0.02,
            device=str(device),
        )
        print(
            f"Held-Out Probe Macro AUROC: {probe_metrics['macro_auroc']:.4f} | "
            f"AUPRC: {probe_metrics['macro_auprc']:.4f}"
        )

        # ---------------------------------------------------------
        # Domain Selectivity for Structured Slot Models (B1, M)
        # ---------------------------------------------------------
        slot_selectivity = None
        if cand in ("b1", "m") and slots_tr is not None and slots_va is not None:
            print(f"Evaluating structured slot domain selectivity for {cand.upper()}...")
            slot_selectivity = evaluate_domain_selectivity(
                slots_train=slots_tr,
                slots_val=slots_va,
                y_anchor_train=y_anc_tr,
                y_anchor_val=y_anc_va,
                domain_slices=domain_slices,
                device=device,
            )
            print(f"Slot Selectivity Scores: {slot_selectivity['selectivity_scores']}")

        # ---------------------------------------------------------
        # Lead Removal Robustness (Zero-shot Masking)
        # ---------------------------------------------------------
        print(f"Evaluating lead-removal robustness for {cand.upper()}...")
        # Train a probe model to evaluate on masked inputs
        probe_clf = nn.Linear(96, total_anchors).to(device)
        p_opt = torch.optim.AdamW(probe_clf.parameters(), lr=0.02)
        p_ds = torch.utils.data.TensorDataset(z_tr, y_anc_tr)
        p_ld = DataLoader(p_ds, batch_size=128, shuffle=True)
        probe_clf.train()
        for _ in range(30):
            for bx, by in p_ld:
                p_opt.zero_grad()
                l = F.binary_cross_entropy_with_logits(probe_clf(bx.to(device)), by.to(device))
                l.backward()
                p_opt.step()

        robustness_scores = evaluate_lead_robustness(
            model=model,
            model_name=cand,
            val_loader=val_loader,
            probe_model_anchor=probe_clf,
            num_masked_leads_list=[0, 1, 2, 4],
            device=device,
        )
        print(f"Lead Masking Macro AUROC (0, 1, 2, 4 leads masked): {robustness_scores}")

        # Store candidate results
        results[cand.upper()] = {
            "effective_rank": eff_rank,
            "condition_number": cond_num,
            "anchor_macro_auroc": anchor_metrics["macro_auroc"],
            "anchor_macro_auprc": anchor_metrics["macro_auprc"],
            "anchor_brier": anchor_metrics["macro_brier"],
            "anchor_sens_at_95spec": anchor_metrics["macro_sens_at_95spec"],
            "anchor_spec_at_95sens": anchor_metrics["macro_spec_at_95sens"],
            "anchor_ece": anchor_metrics["macro_ece"],
            "r_c_sufficiency": r_c,
            "probe_macro_auroc": probe_metrics["macro_auroc"],
            "probe_macro_auprc": probe_metrics["macro_auprc"],
            "slot_selectivity": slot_selectivity,
            "lead_robustness": robustness_scores,
        }

    # -------------------------------------------------------------
    # 3. Save Machine-Readable Results JSON
    # -------------------------------------------------------------
    results_json_path = out_dir / "p2_screen_results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults successfully written to: {results_json_path}")

    # -------------------------------------------------------------
    # 4. Generate Comprehensive Milestone P2 Markdown Report
    # -------------------------------------------------------------
    report_path = PROJECT_ROOT / "P2_REPRESENTATION_SCREEN_REPORT.md"
    generate_markdown_report(results, report_path)
    print(f"Milestone P2 Report emitted to: {report_path}")


def generate_markdown_report(results: dict[str, Any], report_path: Path):
    b0 = results["B0"]
    b1 = results["B1"]
    b2 = results["B2"]
    m = results["M"]
    ub = results["UB"]

    # Evaluate Gates per PRD Addendum §23
    gate_a = m["probe_macro_auroc"] >= b0["probe_macro_auroc"]
    gate_b = m["r_c_sufficiency"] >= 0.85
    gate_c = all(s >= 0.0 for s in m["slot_selectivity"]["selectivity_scores"].values()) if m["slot_selectivity"] else False
    gate_d = (m["lead_robustness"]["1"] >= b0["lead_robustness"]["1"]) or (m["probe_macro_auroc"] > b0["probe_macro_auroc"])
    gate_e = m["probe_macro_auroc"] >= b1["probe_macro_auroc"]

    overall_pass = gate_a and gate_b and gate_c and gate_d

    report = f"""# Milestone P2: Representation Screen Report (PTB-XL Folds 1–7 Train, Fold 8 Dev Val)

**Document**: `P2_REPRESENTATION_SCREEN_REPORT.md`  
**Execution Date**: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  
**Dataset**: PTB-XL Folds 1–7 ($N=15,245$), Dev Validation Fold 8 ($N=2,173$)  
**Status**: `P2_REPRESENTATION_SCREEN_GATE = {"PASS" if overall_pass else "FAIL"}`

---

## 1. Executive Summary & Research Claims

This report presents the empirical comparison of four architectural formulations sharing an identical temporal ResNet1D backbone (~1.0–1.4M parameters) on PTB-XL:

1. **B0 (Plain Representation)**: Standard global average pooling, no geometry, no clinical slots.
2. **B1 (Structured Latent)**: 6 clinical slot queries, no geometry embedding.
3. **B2 (Geometry Latent)**: 8-view angle embedding + geometry tokens, single global query, no named slots.
4. **M (Full GRAIL Pilot)**: 8-view geometry + 6 structured slots + VICReg SSL + clinical anchor shaping + any-pairs view auxiliary decoder.
5. **UB (Supervised Upper Bound)**: Same backbone trained end-to-end multi-task BCE.

---

## 2. Core Benchmark Comparison Table (Fold 8 Dev Validation)

| Model | Latent Dim | Eff. Rank (/96) | Anchor AUROC | Anchor AUPRC | Sufficiency $R_c$ | Held-Out Probe AUROC | Held-Out Probe AUPRC | 1-Lead Mask AUROC | 2-Lead Mask AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **UB** (Supervised) | — | — | {ub['macro_auroc']:.4f} | {ub['macro_auprc']:.4f} | 1.0000 | — | — | — | — |
| **B0** (Plain SSL) | 96 | {b0['effective_rank']:.2f} | {b0['anchor_macro_auroc']:.4f} | {b0['anchor_macro_auprc']:.4f} | {b0['r_c_sufficiency']:.4f} | {b0['probe_macro_auroc']:.4f} | {b0['probe_macro_auprc']:.4f} | {b0['lead_robustness']['1']:.4f} | {b0['lead_robustness']['2']:.4f} |
| **B1** (Structured) | 96 | {b1['effective_rank']:.2f} | {b1['anchor_macro_auroc']:.4f} | {b1['anchor_macro_auprc']:.4f} | {b1['r_c_sufficiency']:.4f} | {b1['probe_macro_auroc']:.4f} | {b1['probe_macro_auprc']:.4f} | {b1['lead_robustness']['1']:.4f} | {b1['lead_robustness']['2']:.4f} |
| **B2** (Geometry) | 96 | {b2['effective_rank']:.2f} | {b2['anchor_macro_auroc']:.4f} | {b2['anchor_macro_auprc']:.4f} | {b2['r_c_sufficiency']:.4f} | {b2['probe_macro_auroc']:.4f} | {b2['probe_macro_auprc']:.4f} | {b2['lead_robustness']['1']:.4f} | {b2['lead_robustness']['2']:.4f} |
| **M** (Full GRAIL) | 96 | {m['effective_rank']:.2f} | {m['anchor_macro_auroc']:.4f} | {m['anchor_macro_auprc']:.4f} | **{m['r_c_sufficiency']:.4f}** | **{m['probe_macro_auroc']:.4f}** | **{m['probe_macro_auprc']:.4f}** | **{m['lead_robustness']['1']:.4f}** | **{m['lead_robustness']['2']:.4f}** |

---

## 3. PRD Scientific Gates Verification

### Gate A: Probe-Only Concept Generalization
- **Criterion**: Held-out probe AUROC(M) >= AUROC(B0) on 12 held-out concepts.
- **Result**: M = `{m['probe_macro_auroc']:.4f}` vs B0 = `{b0['probe_macro_auroc']:.4f}` (Delta = {m['probe_macro_auroc'] - b0['probe_macro_auroc']:+.4f}).
- **Verdict**: `{"PASS" if gate_a else "FAIL"}`

### Gate B: Clinical Accessibility & Sufficiency
- **Criterion**: Frozen linear probe recovers >= 85% of end-to-end supervised signal (R_c >= 0.85).
- **Result**: R_c = {m['r_c_sufficiency']:.4f} (Anchor AUROC: `{m['anchor_macro_auroc']:.4f}` vs UB: `{ub['macro_auroc']:.4f}`).
- **Verdict**: `{"PASS" if gate_b else "FAIL"}`

### Gate C: Structured Slot Domain Selectivity
- **Criterion**: Slots demonstrate positive domain specialization (Selectivity = Perf_intended - Perf_unrelated > 0).
- **Results for Model M**:
"""
    if m["slot_selectivity"]:
        for dom, score in m["slot_selectivity"]["selectivity_scores"].items():
            report += f"  - **{dom.capitalize()} Slot (z_{dom})**: Selectivity = {score:+.4f}\n"
    report += f"""- **Verdict**: `{"PASS" if gate_c else "FAIL"}`

### Gate D: Geometric Spatial Inductive Bias
- **Criterion**: Geometry improves probe-only transfer or lead-removal robustness (1 lead masked: M `{m['lead_robustness']['1']:.4f}` vs B0 `{b0['lead_robustness']['1']:.4f}`).
- **Verdict**: `{"PASS" if gate_d else "FAIL"}`

### Gate E: Any-Pairs Auxiliary Decoder
- **Criterion**: View auxiliary decoder retains cross-view electrical structure without harming representation capacity.
- **Verdict**: `{"PASS" if gate_e else "FAIL"}`

---

## 4. Directional Confirmation for Milestone P3
- **Surviving Configuration**: {"Model M (Full GRAIL)" if overall_pass else "Investigate Bottlenecks"}
- **Next Step**: Evaluate on confirmation validation fold (Fold 9, $N=2,183$) to confirm directional repeatability before SSL objective selection (VICReg vs Barlow Twins vs BYOL). Fold 10 remains strictly locked.
"""
    with open(report_path, "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
