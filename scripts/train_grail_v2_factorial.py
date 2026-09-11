"""GRAIL-ECG v2: Phase R1 & R2 Factorial Representation Training and Qualification.

Trains and qualifies:
- UB: End-to-end Supervised Upper Bound (pure BCE with training-fold pos_weights)
- B0: True SSL-only (VICReg only, zero clinical loss, G=0, S=0, V=0)
- Full 2^3 Factorial combinations:
    1. B1 (G=0, S=0, V=0): SSL + Clinical BCE
    2. B3_geom (G=1, S=0, V=0): Geometry alone
    3. B2_slots (G=0, S=1, V=0): Slots alone
    4. Model_001 (G=0, S=0, V=1): View Aux alone
    5. Model_110 (G=1, S=1, V=0): Geometry + Slots
    6. Model_101 (G=1, S=0, V=1): Geometry + View Aux
    7. Model_011 (G=0, S=1, V=1): Slots + View Aux
    8. Model_M (G=1, S=1, V=1): Full Factorial (Geometry + Slots + View Aux)

Followed by automatic evaluation through the complete Representation Qualification Suite.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import sys
import time
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

# CuDNN disabled for multi-process safety and CUDA A100 conv1d compatibility
torch.backends.cudnn.enabled = False

from grail_ecg.src.data.ptbxl_dataset import PTBXLDataset
from grail_ecg.src.models.baselines import FactorialGRAILEncoder, SupervisedUpperBoundUB
from grail_ecg.src.models.view_aux_decoder import ViewAuxiliaryDecoder
from grail_ecg.src.losses.ssl import VICRegLoss
from grail_ecg.src.losses.clinical import ClinicalAnchorLoss
from grail_ecg.src.losses.view import ViewReconstructionLoss
from grail_ecg.src.geometry.lead_geometry import INDEPENDENT_8_LEADS, get_angles_tensor
from grail_ecg.src.evaluation.representation_qualification import (
    compute_compactness_and_geometry,
    compute_concept_separability,
    compute_local_semantic_geometry,
    compute_clustering_quality,
    compute_slot_concept_matrix,
    compute_intervention_specificity,
    compute_residual_slot_challenge,
    compute_low_shot_efficiency,
    compute_nuisance_probes,
)
from grail_ecg.src.probes.linear_probe import fit_and_evaluate_linear_probe


class ECGPhysiologicalAugmenter:
    """Allowed physiological augmentations per PRD Addendum §13."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, L, T = x.shape
        # 1. Amplitude scaling (0.85 - 1.15)
        alpha = torch.empty(B, 1, 1, device=x.device).uniform_(0.85, 1.15)
        x_aug = x * alpha
        # 2. Additive Gaussian noise (std 0.02 mV)
        noise = torch.randn_like(x_aug) * 0.02
        x_aug = x_aug + noise
        # 3. Baseline wander / shift (-0.05 - 0.05 mV)
        beta = torch.empty(B, 1, 1, device=x.device).uniform_(-0.05, 0.05)
        x_aug = x_aug + beta
        # 4. Stochastic lead dropout (masking 1 lead with p=0.2)
        if torch.rand(1).item() < 0.2:
            lead_idx = torch.randint(0, L, (1,)).item()
            x_aug[:, lead_idx, :] = 0.0
        return x_aug


def train_epoch(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    augmenter: ECGPhysiologicalAugmenter,
    vicreg_loss_fn: VICRegLoss,
    pos_weights: torch.Tensor,
    domain_counts: dict[str, int],
    canonical_angles: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss_acc = 0.0
    ssl_loss_acc = 0.0
    clin_loss_acc = 0.0
    view_loss_acc = 0.0
    n_batches = 0

    is_ub = (model_name == "ub")
    is_b0 = (model_name == "b0")
    use_slots = getattr(model, "use_slots", False)
    use_view_aux = getattr(model, "use_view_aux", False)

    for batch in loader:
        x_raw = batch["ecg_8l"].to(device)
        y_anchor = batch["anchor_labels"].to(device)
        B, L, T = x_raw.shape

        optimizer.zero_grad()

        if is_ub:
            logits = model(x_raw)
            loss = F.binary_cross_entropy_with_logits(logits, y_anchor, pos_weight=pos_weights)
            l_ssl = torch.tensor(0.0)
            l_clin = loss
            l_view = torch.tensor(0.0)
        elif is_b0:
            x_a = augmenter(x_raw)
            x_b = augmenter(x_raw)
            _, z_a, _ = model(x_a)
            _, z_b, _ = model(x_b)
            loss_ssl_dict = vicreg_loss_fn(z_a, z_b)
            l_ssl = loss_ssl_dict["loss"]
            loss = l_ssl
            l_clin = torch.tensor(0.0)
            l_view = torch.tensor(0.0)
        else:
            x_a = augmenter(x_raw)
            x_b = augmenter(x_raw)
            slots_a, z_a, logits_a = model(x_a)
            slots_b, z_b, logits_b = model(x_b)

            loss_ssl_dict = vicreg_loss_fn(z_a, z_b)
            l_ssl = loss_ssl_dict["loss"]

            # Clinical loss
            if use_slots and isinstance(logits_a, dict):
                # Concatenate domain logits: rhythm, conduction, morphology, stt
                cat_logits = torch.cat(list(logits_a.values()), dim=1)
                l_clin = F.binary_cross_entropy_with_logits(cat_logits, y_anchor, pos_weight=pos_weights)
            else:
                l_clin = F.binary_cross_entropy_with_logits(logits_a, y_anchor, pos_weight=pos_weights)

            # View aux loss (if enabled)
            if use_view_aux and model.view_aux_decoder is not None:
                # Mask 1 random target lead from encoder input
                target_lead_idx = torch.randint(0, L, (1,)).item()
                x_masked = x_raw.clone()
                x_masked[:, target_lead_idx, :] = 0.0
                _, z_masked, _ = model(x_masked)
                target_angle = canonical_angles[target_lead_idx:target_lead_idx + 1].expand(B, -1).to(device)
                pred_lead = model.view_aux_decoder(z_masked, target_angle).squeeze(1)
                gt_lead = x_raw[:, target_lead_idx, :]
                l_view = F.l1_loss(pred_lead, gt_lead)
            else:
                l_view = torch.tensor(0.0)

            loss = l_ssl + 2.0 * l_clin + (1.0 * l_view if use_view_aux else 0.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss_acc += loss.item()
        ssl_loss_acc += l_ssl.item()
        clin_loss_acc += l_clin.item()
        view_loss_acc += l_view.item()
        n_batches += 1

    return {
        "loss": total_loss_acc / n_batches,
        "loss_ssl": ssl_loss_acc / n_batches,
        "loss_clin": clin_loss_acc / n_batches,
        "loss_view": view_loss_acc / n_batches,
    }


def extract_embeddings_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Extracts representations Z, structured slots, anchor labels, probe labels, and nuisance metadata."""
    model.eval()
    all_z = []
    all_slots = []
    all_anchors = []
    all_probes = []
    all_ages = []
    all_sexes = []

    use_slots = getattr(model, "use_slots", False)

    with torch.no_grad():
        for batch in loader:
            x = batch["ecg_8l"].to(device)
            if hasattr(model, "use_slots"):
                slots, z_flat, _ = model(x)
                all_z.append(z_flat.cpu().numpy())
                if slots is not None:
                    all_slots.append(slots.cpu().numpy())
            else:
                # UB model: extract pooled representation before classifier
                B, L, T = x.shape
                x_flat = x.reshape(B * L, 1, T)
                h_flat = model.temporal_encoder(x_flat)
                h = h_flat.reshape(B, L, model.hidden_dim, 32)
                pooled = h.mean(dim=(1, 3))
                all_z.append(pooled.cpu().numpy())

            all_anchors.append(batch["anchor_labels"].numpy())
            all_probes.append(batch["probe_labels"].numpy())
            all_ages.append(batch["age"].numpy())
            all_sexes.append(batch["sex"].numpy())

    z_arr = np.concatenate(all_z, axis=0)
    slots_arr = np.concatenate(all_slots, axis=0) if all_slots else None
    anchors_arr = np.concatenate(all_anchors, axis=0)
    probes_arr = np.concatenate(all_probes, axis=0)
    meta = {
        "age": np.concatenate(all_ages, axis=0),
        "sex": np.concatenate(all_sexes, axis=0),
    }
    return z_arr, slots_arr, anchors_arr, probes_arr, meta


def run_full_qualification(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    concept_tiers: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Runs the complete Representation Qualification Suite and exports all Parquet/JSON artifacts."""
    print(f"\n--- Running Full Representation Qualification for {model_name} ---")
    z_tr, slots_tr, y_anchors_tr, y_probes_tr, meta_tr = extract_embeddings_and_labels(model, train_loader, device)
    z_val, slots_val, y_anchors_val, y_probes_val, meta_val = extract_embeddings_and_labels(model, val_loader, device)

    # 1. Compactness & Intrinsic Geometry
    print("1/7: Computing compactness, anisotropy, and intrinsic dimension...")
    compactness = compute_compactness_and_geometry(z_val)

    # 2. Linear Probing across Tiers (Anchor, Tier P1, Tier P2, Tier P3)
    print("2/7: Evaluating linear sufficiency across Anchor and Transfer Tiers...")
    anchor_metrics = fit_and_evaluate_linear_probe(
        torch.from_numpy(z_tr), torch.from_numpy(y_anchors_tr),
        torch.from_numpy(z_val), torch.from_numpy(y_anchors_val),
        num_epochs=30, device=str(device),
    )
    probe_metrics = fit_and_evaluate_linear_probe(
        torch.from_numpy(z_tr), torch.from_numpy(y_probes_tr),
        torch.from_numpy(z_val), torch.from_numpy(y_probes_val),
        num_epochs=30, device=str(device),
    )

    # Map probe classes to Tiers (P1, P2, P3)
    p1_codes = concept_tiers["tier_p1_codes"]
    p2_codes = concept_tiers["tier_p2_codes"]
    p3_codes = concept_tiers["tier_p3_codes"]
    all_probe_codes = concept_tiers["all_probe_codes"]
    probe_code_to_idx = {c: i for i, c in enumerate(all_probe_codes)}

    p1_idxs = [probe_code_to_idx[c] for c in p1_codes if c in probe_code_to_idx]
    p2_idxs = [probe_code_to_idx[c] for c in p2_codes if c in probe_code_to_idx]
    p3_idxs = [probe_code_to_idx[c] for c in p3_codes if c in probe_code_to_idx]

    class_m = probe_metrics["class_metrics"]
    tier_p1_auroc = float(np.nanmean([class_m[i]["auroc"] for i in p1_idxs])) if p1_idxs else float("nan")
    tier_p2_auroc = float(np.nanmean([class_m[i]["auroc"] for i in p2_idxs])) if p2_idxs else float("nan")
    tier_p3_auroc = float(np.nanmean([class_m[i]["auroc"] for i in p3_idxs])) if p3_idxs else float("nan")

    # 3. Geometric Separability
    print("3/7: Computing geometric separability metrics (Fisher, centroid, within/between)...")
    sep_results = {}
    for c_idx, code in enumerate(concept_tiers["all_anchor_codes"]):
        sep_results[code] = compute_concept_separability(z_val, y_anchors_val[:, c_idx])
    mean_fisher = float(np.nanmean([v["fisher_separation"] for v in sep_results.values()]))
    mean_wb_ratio = float(np.nanmean([v["within_between_ratio"] for v in sep_results.values()]))

    # 4. Local Semantic Geometry & Retrieval
    print("4/7: Computing local semantic geometry and retrieval metrics (P@k, Recall@k, nDCG@k)...")
    retrieval = compute_local_semantic_geometry(
        z_query=z_val, y_query=y_anchors_val,
        z_corpus=z_tr, y_corpus=y_anchors_tr,
    )

    # 5. Low-Shot Sample Efficiency Sweep
    print("5/7: Evaluating low-shot sample efficiency sweep (1% to 100%)...")
    low_shot = compute_low_shot_efficiency(
        z_train=z_tr, y_train=y_anchors_tr,
        z_val=z_val, y_val=y_anchors_val,
    )

    # 6. Disentanglement & Residual Challenge (if slots present)
    disentanglement = {}
    residual_challenge = {}
    if slots_tr is not None and slots_val is not None:
        print("6/7: Evaluating slot x concept factorization matrix and residual challenge...")
        disentanglement = compute_slot_concept_matrix(
            slots_train=slots_tr, y_train=y_anchors_tr,
            slots_val=slots_val, y_val=y_anchors_val,
            concept_names=concept_tiers["all_anchor_codes"],
            domain_mapping=concept_tiers["domain_mapping"],
        )
        residual_challenge = compute_residual_slot_challenge(
            slots_train=slots_tr, y_train=y_anchors_tr,
            slots_val=slots_val, y_val=y_anchors_val,
        )
    else:
        print("6/7: Skipping slot factorization (model does not use structured slots)...")

    # 7. Nuisance Probes
    print("7/7: Measuring nuisance variable accessibility (Age, Sex)...")
    nuisance = compute_nuisance_probes(z_tr, meta_tr, z_val, meta_val)

    # Export embeddings to Parquet
    rep_dir = output_dir / "representation" / model_name
    rep_dir.mkdir(parents=True, exist_ok=True)

    df_z_val = pd.DataFrame(z_val, columns=[f"z_{i}" for i in range(z_val.shape[1])])
    df_z_val.to_parquet(rep_dir / "embeddings_fold8.parquet", index=False)
    np.save(rep_dir / "singular_values.npy", np.array(compactness["singular_values"]))

    full_profile = {
        "model_name": model_name,
        "compactness": compactness,
        "anchor_linear_probe": {
            "macro_auroc": anchor_metrics["macro_auroc"],
            "macro_auprc": anchor_metrics["macro_auprc"],
            "macro_brier": anchor_metrics["macro_brier"],
            "macro_sens_at_95spec": anchor_metrics["macro_sens_at_95spec"],
            "macro_spec_at_95sens": anchor_metrics["macro_spec_at_95sens"],
            "macro_ece": anchor_metrics["macro_ece"],
        },
        "transfer_probes": {
            "macro_auroc_all_heldout": probe_metrics["macro_auroc"],
            "macro_auprc_all_heldout": probe_metrics["macro_auprc"],
            "tier_p1_directly_related_auroc": tier_p1_auroc,
            "tier_p2_related_distinct_auroc": tier_p2_auroc,
            "tier_p3_ontology_distinct_auroc": tier_p3_auroc,
        },
        "separability": {
            "mean_fisher_separation": mean_fisher,
            "mean_within_between_ratio": mean_wb_ratio,
        },
        "retrieval": retrieval,
        "low_shot_efficiency": low_shot,
        "disentanglement": disentanglement,
        "residual_slot_challenge": residual_challenge,
        "nuisance_probes": nuisance,
    }

    with open(rep_dir / "qualification_profile.json", "w") as f:
        json.dump(full_profile, f, indent=2)

    return full_profile


def main():
    parser = argparse.ArgumentParser(description="GRAIL-ECG v2: Factorial Training & Qualification")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ub", "b0", "b1", "b3_geom", "b2_slots", "model_m"],
        help="Models to train: ub, b0, b1, b3_geom, b2_slots, model_001, model_110, model_101, model_011, model_m",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="results/grail_v2", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/grail_v2")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Concept Tiers and Training-only Class Weights
    concept_tiers_yaml = PROJECT_ROOT / "configs" / "ptbxl_concept_tiers.yaml"
    with open(concept_tiers_yaml) as f:
        concept_tiers = yaml.safe_load(f)

    pos_weights_list = [
        concept_tiers["anchors"][c]["pos_weight"]
        for c in range(len(concept_tiers["anchors"]))
    ]
    pos_weights = torch.tensor(pos_weights_list, dtype=torch.float32, device=device)

    # Domain class counts for structured slots: rhythm, conduction, morphology, stt
    domain_counts = concept_tiers["anchor_counts_per_domain"]

    # 2. Canonical Physical Angles for 8 leads
    canonical_angles = get_angles_tensor(INDEPENDENT_8_LEADS).to(device)

    # 3. Load Datasets: Folds 1-7 Train, Fold 8 Dev Val
    data_dir = PROJECT_ROOT / "data" / "ptb_xl"
    train_dataset = PTBXLDataset(
        data_dir=data_dir,
        folds=[1, 2, 3, 4, 5, 6, 7],
        concept_config_path=concept_tiers_yaml,
        return_12l=False,
    )
    val_dataset = PTBXLDataset(
        data_dir=data_dir,
        folds=[8],
        concept_config_path=concept_tiers_yaml,
        return_12l=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Loaded PTB-XL: Train Folds 1-7 (N={len(train_dataset)}), Val Fold 8 (N={len(val_dataset)})")

    # Factorial configurations
    factorial_configs = {
        "ub": {"type": "ub"},
        "b0": {"type": "factorial", "G": False, "S": False, "V": False},
        "b1": {"type": "factorial", "G": False, "S": False, "V": False},
        "b3_geom": {"type": "factorial", "G": True, "S": False, "V": False},
        "b2_slots": {"type": "factorial", "G": False, "S": True, "V": False},
        "model_001": {"type": "factorial", "G": False, "S": False, "V": True},
        "model_110": {"type": "factorial", "G": True, "S": True, "V": False},
        "model_101": {"type": "factorial", "G": True, "S": False, "V": True},
        "model_011": {"type": "factorial", "G": False, "S": True, "V": True},
        "model_m": {"type": "factorial", "G": True, "S": True, "V": True},
    }

    models_to_run = args.models
    augmenter = ECGPhysiologicalAugmenter()
    vicreg_loss_fn = VICRegLoss(sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0)

    summary_results = {}

    for model_name in models_to_run:
        cfg = factorial_configs.get(model_name)
        if cfg is None:
            print(f"Unknown model name {model_name}, skipping.")
            continue

        print(f"\n=======================================================")
        print(f"TRAINING MODEL: {model_name.upper()} (Config: {cfg})")
        print(f"=======================================================")

        if cfg["type"] == "ub":
            model = SupervisedUpperBoundUB(num_leads=8, hidden_dim=128, num_classes=25).to(device)
        else:
            model = FactorialGRAILEncoder(
                use_geometry=cfg["G"],
                use_slots=cfg["S"],
                use_view_aux=cfg["V"],
                num_leads=8,
                num_tokens_per_lead=32,
                hidden_dim=128,
                num_slots=6,
                slot_dim=16,
                anchor_counts_per_domain=domain_counts if cfg["S"] else None,
                total_anchor_classes=25,
            ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

        best_val_loss = float("inf")
        best_ckpt_path = ckpt_dir / f"{model_name}_best.pt"

        start_time = time.time()
        for epoch in range(1, args.epochs + 1):
            epoch_metrics = train_epoch(
                model=model,
                model_name=model_name,
                loader=train_loader,
                optimizer=optimizer,
                augmenter=augmenter,
                vicreg_loss_fn=vicreg_loss_fn,
                pos_weights=pos_weights,
                domain_counts=domain_counts,
                canonical_angles=canonical_angles,
                device=device,
            )
            scheduler.step()

            print(
                f"[{model_name.upper()}] Epoch {epoch:02d}/{args.epochs:02d} | "
                f"Loss: {epoch_metrics['loss']:.4f} "
                f"(SSL: {epoch_metrics['loss_ssl']:.4f}, Clin: {epoch_metrics['loss_clin']:.4f}, View: {epoch_metrics['loss_view']:.4f}) | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

            # Track and save checkpoint
            if epoch_metrics["loss"] < best_val_loss:
                best_val_loss = epoch_metrics["loss"]
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": epoch_metrics},
                    best_ckpt_path,
                )

        train_duration = time.time() - start_time
        print(f"Finished training {model_name} in {train_duration / 60:.1f} minutes. Checkpoint: {best_ckpt_path}")

        # Load best model for full qualification
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        # Run Qualification Battery
        profile = run_full_qualification(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            concept_tiers=concept_tiers,
            device=device,
            output_dir=output_dir,
        )
        profile["train_duration_sec"] = train_duration
        summary_results[model_name] = profile

        # Update summary JSON
        with open(output_dir / "factorial_qualification_summary.json", "w") as f:
            json.dump(summary_results, f, indent=2)

    print("\n=======================================================")
    print("ALL MODELS TRAINED AND QUALIFIED SUCCESSFULLY!")
    print(f"Summary artifact written to: {output_dir / 'factorial_qualification_summary.json'}")
    print("=======================================================")


if __name__ == "__main__":
    main()
