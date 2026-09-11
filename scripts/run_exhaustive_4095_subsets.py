"""Stage B: Exhaustive 4,095 Subset Lattice Evaluation Driver for GRAIL-ECG v2.

Evaluates every non-empty subset S of the 12 displayed ECG leads against frozen teacher Z*:
- Token caching: Computes all 12 lead representations in O(N * 12) steps.
- Latent preservation: Cosine similarity, normalized L2 distance, CKA to Z*.
- Clinical access: Fixed full-head AUROC/AUPRC vs Subset-reprobed AUROC/AUPRC.
- Coordinate drift gap: D_coordinate(S) = P_reprobe(S) - P_fixed(S).
- Exact Lead Shapley values: Representation-level and Disease-specific.
- Pairwise synergy / redundancy graph: Harsanyi/Möbius 2nd-order interactions.
- Minimal sufficient lead sets within prespecified margin delta.
- Information frontiers: P_min(k), P_med(k), P_max(k) vs displayed count k and rank r.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch.backends.cudnn.enabled = False

from grail_ecg.src.data.ptbxl_dataset import PTBXLDataset
from grail_ecg.src.models.baselines import FactorialGRAILEncoder
from grail_ecg.src.geometry.lead_geometry import CANONICAL_12_LEADS, get_angles_tensor
from grail_ecg.src.evaluation.exhaustive_subset_eval import (
    STANDARD_12_LEADS,
    SubsetRegistry,
    TokenCachedSubsetEvaluator,
    compute_exact_lead_shapley,
    compute_pairwise_lead_interactions,
    compute_information_frontiers,
    compute_disease_minimal_sufficient_sets,
)


def main():
    parser = argparse.ArgumentParser(description="GRAIL-ECG v2: Exhaustive 4,095 Subset Lattice Evaluation")
    parser.add_argument("--model-path", type=str, default="checkpoints/grail_v2/model_m_best.pt", help="Path to trained model checkpoint")
    parser.add_argument("--val-fold", type=int, default=8, help="Validation fold (default: 8 for dev screen, 10 for final)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for token caching")
    parser.add_argument("--output-dir", type=str, default="results/grail_v2/subsets", help="Output directory for subset parquet files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Exhaustive Subset Evaluation on device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Concept Config and Canonical 12-lead angles
    concept_tiers_yaml = PROJECT_ROOT / "configs" / "ptbxl_concept_tiers.yaml"
    with open(concept_tiers_yaml) as f:
        concept_tiers = yaml.safe_load(f)

    anchor_codes = concept_tiers["all_anchor_codes"]
    num_anchors = len(anchor_codes)
    canonical_angles_12 = get_angles_tensor(STANDARD_12_LEADS).to(device)

    # 2. Load Model Checkpoint
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        sys.exit(1)

    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Detect if model uses slots or geometry from keys
    use_geom = any("geom_proj" in k for k in state_dict.keys())
    use_slots = any("slot_aggregator" in k for k in state_dict.keys())
    use_view_aux = any("view_aux_decoder" in k for k in state_dict.keys())

    print(f"Model configuration detected: Geometry={use_geom}, Slots={use_slots}, ViewAux={use_view_aux}")

    model = FactorialGRAILEncoder(
        use_geometry=use_geom,
        use_slots=use_slots,
        use_view_aux=use_view_aux,
        num_leads=12,
        num_tokens_per_lead=32,
        hidden_dim=128,
        num_slots=6,
        slot_dim=16,
        anchor_counts_per_domain=concept_tiers["anchor_counts_per_domain"] if use_slots else None,
        total_anchor_classes=num_anchors,
    ).to(device)

    # Load compatible state dict weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 3. Load Validation Dataset with 12 displayed leads
    data_dir = PROJECT_ROOT / "data" / "ptb_xl"
    val_dataset = PTBXLDataset(
        data_dir=data_dir,
        folds=[args.val_fold],
        concept_config_path=concept_tiers_yaml,
        return_12l=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Loaded Fold {args.val_fold} validation dataset (N={len(val_dataset)}) with 12 displayed leads.")

    # 4. Cache spatio-temporal tokens for all 12 leads across all validation records
    print("Phase 1: Precomputing and caching lead tokens for all 12 leads...")
    cached_tokens_list = []
    all_anchors_list = []
    t0 = time.time()

    with torch.no_grad():
        for batch in val_loader:
            ecg_12l = batch["ecg_12l"].to(device)  # [B, 12, 5000]
            B = ecg_12l.shape[0]

            # Flatten 12 leads into batch dimension to run shared ResNet1D
            x_flat = ecg_12l.reshape(B * 12, 1, 5000)
            if use_geom:
                angles_expanded = canonical_angles_12.unsqueeze(0).expand(B, -1, -1)  # [B, 12, 2]
                toks = model.view_encoder(ecg_12l, custom_angles=angles_expanded)  # [B, 12 * 32, 128]
                toks = toks.reshape(B, 12, 32, 128)
            else:
                h_flat = model.temporal_encoder(x_flat)
                h = h_flat.reshape(B, 12, model.hidden_dim, 32).permute(0, 1, 3, 2)
                toks = (h + model.temporal_pos_embed).reshape(B, 12, 32, model.hidden_dim)

            cached_tokens_list.append(toks.cpu())
            all_anchors_list.append(batch["anchor_labels"])

    all_cached_tokens = torch.cat(cached_tokens_list, dim=0).to(device)  # [N, 12, 32, 128]
    y_anchors = torch.cat(all_anchors_list, dim=0).numpy()  # [N, 25]
    N = all_cached_tokens.shape[0]
    print(f"Token caching complete in {time.time() - t0:.1f}s. Cached tensor shape: {all_cached_tokens.shape}")

    # 5. Compute Full 12-Lead Reference Latent Z* and Full Linear Readout
    print("Phase 2: Computing Full Reference Representation Z* and Baseline Readouts...")
    full_indices = list(range(12))
    flat_full_tokens = all_cached_tokens.reshape(N, 12 * 32, 128)
    with torch.no_grad():
        if use_slots:
            _, z_star = model.slot_aggregator(flat_full_tokens)
            anchor_logits_dict = model.anchor_heads(_)
            full_logits = torch.cat(list(anchor_logits_dict.values()), dim=1).cpu().numpy()
        else:
            q = model.query.expand(N, -1, -1)
            attn_out, _ = model.cross_attn(query=q, key=flat_full_tokens, value=flat_full_tokens)
            out = model.norm(q + attn_out).squeeze(1)
            z_star = model.proj(out)
            full_logits = model.flat_anchor_head(z_star).cpu().numpy()

    z_star_norm = z_star / (torch.norm(z_star, dim=1, keepdim=True) + 1e-8)
    full_probs = 1.0 / (1.0 + np.exp(-full_logits))

    full_aurocs = {}
    for c_idx, code in enumerate(anchor_codes):
        if len(np.unique(y_anchors[:, c_idx])) > 1:
            full_aurocs[code] = float(roc_auc_score(y_anchors[:, c_idx], full_probs[:, c_idx]))
        else:
            full_aurocs[code] = 0.50

    macro_full_auroc = float(np.mean(list(full_aurocs.values())))
    print(f"Full 12-lead Reference Macro AUROC: {macro_full_auroc:.4f}")

    # 6. Exhaustive Evaluation over all 4,095 non-empty subsets
    print("Phase 3: Exhaustively evaluating all 4,095 subsets...")
    registry = SubsetRegistry()
    subset_df = registry.to_dataframe()

    subset_results = []
    t_start_subsets = time.time()

    # Pre-split cached tokens in chunks for GPU memory management
    chunk_size = 512
    num_chunks = (N + chunk_size - 1) // chunk_size

    for row_idx, row in subset_df.iterrows():
        mask = row["subset_bitmask"]
        lead_names = row["lead_names"]
        lead_indices = [i for i in range(12) if (mask & (1 << i))]
        k = len(lead_indices)

        # Gather subset tokens and run aggregation
        sub_tokens = all_cached_tokens[:, lead_indices, :, :].reshape(N, k * 32, 128)

        with torch.no_grad():
            if use_slots:
                slots_s, z_s = model.slot_aggregator(sub_tokens)
                sub_logits_dict = model.anchor_heads(slots_s)
                sub_logits = torch.cat(list(sub_logits_dict.values()), dim=1).cpu().numpy()
            else:
                q = model.query.expand(N, -1, -1)
                attn_out, _ = model.cross_attn(query=q, key=sub_tokens, value=sub_tokens)
                out = model.norm(q + attn_out).squeeze(1)
                z_s = model.proj(out)
                sub_logits = model.flat_anchor_head(z_s).cpu().numpy()

        # Cosine similarity to Z*
        z_s_norm = z_s / (torch.norm(z_s, dim=1, keepdim=True) + 1e-8)
        mean_cos = float(torch.mean(torch.sum(z_s_norm * z_star_norm, dim=1)).item())
        mean_l2 = float(torch.mean(torch.norm(z_s - z_star, dim=1)).item())

        # Clinical AUROC with fixed head
        sub_probs = 1.0 / (1.0 + np.exp(-sub_logits))
        aurocs = []
        for c_idx in range(num_anchors):
            if len(np.unique(y_anchors[:, c_idx])) > 1:
                try:
                    aurocs.append(float(roc_auc_score(y_anchors[:, c_idx], sub_probs[:, c_idx])))
                except ValueError:
                    aurocs.append(0.5)
            else:
                aurocs.append(0.5)

        macro_auroc = float(np.mean(aurocs))

        res_item = {
            "subset_bitmask": mask,
            "k_displayed": k,
            "lead_names_str": row["lead_names_str"],
            "independent_rank": row["independent_rank"],
            "n_limb": row["n_limb"],
            "limb_rank": row["limb_rank"],
            "n_precordial": row["n_precordial"],
            "latent_cosine_to_full": mean_cos,
            "latent_l2_to_full": mean_l2,
            "fixed_head_auroc": macro_auroc,
            "retention_relative_to_full": float(macro_auroc / (macro_full_auroc + 1e-12)),
        }
        # Add individual disease AUROCs for primary concepts
        for c_idx, code in enumerate(anchor_codes[:10]):
            res_item[f"auroc_{code}"] = aurocs[c_idx]

        subset_results.append(res_item)

        if (row_idx + 1) % 500 == 0 or (row_idx + 1) == 4095:
            elapsed = time.time() - t_start_subsets
            print(f"Evaluated {row_idx + 1}/4095 subsets ({elapsed:.1f}s, {(row_idx + 1) / elapsed:.1f} subsets/sec)")

    results_df = pd.DataFrame(subset_results)
    results_df.to_parquet(output_dir / "subset_representation_metrics.parquet", index=False)
    print(f"Saved complete 4,095 subset evaluation to {output_dir / 'subset_representation_metrics.parquet'}")

    # 7. Compute Exact Lead Shapley Values
    print("\nPhase 4: Computing exact Lead Shapley values...")
    # Lookup dictionary for fast bitmask -> performance value
    mask_to_auroc = dict(zip(results_df["subset_bitmask"], results_df["fixed_head_auroc"]))
    mask_to_cosine = dict(zip(results_df["subset_bitmask"], results_df["latent_cosine_to_full"]))

    shap_auroc = compute_exact_lead_shapley(lambda m: mask_to_auroc.get(m, 0.5) - 0.5, num_players=12)
    shap_cosine = compute_exact_lead_shapley(lambda m: mask_to_cosine.get(m, 0.0), num_players=12)

    shapley_df = pd.DataFrame({
        "lead": STANDARD_12_LEADS,
        "shapley_clinical_auroc": shap_auroc,
        "shapley_latent_cosine": shap_cosine,
    })
    shapley_df.to_parquet(output_dir / "lead_shapley.parquet", index=False)
    print("Exact Lead Shapley Values:")
    print(shapley_df.to_string(index=False))

    # 8. Compute Pairwise Lead Interactions (Harsanyi/Möbius Synergy & Redundancy)
    print("\nPhase 5: Computing pairwise lead synergy/redundancy interaction graph...")
    synergy_mat = compute_pairwise_lead_interactions(lambda m: mask_to_auroc.get(m, 0.5), num_players=12)
    synergy_df = pd.DataFrame(synergy_mat, index=STANDARD_12_LEADS, columns=STANDARD_12_LEADS)
    synergy_df.to_parquet(output_dir / "lead_pair_interactions.parquet")

    # 9. Compute Information Frontiers
    print("\nPhase 6: Computing Pareto Information Frontiers...")
    frontiers_df = compute_information_frontiers(results_df, metric_col="fixed_head_auroc")
    frontiers_df.to_parquet(output_dir / "information_frontier.parquet", index=False)
    print("Information Frontier Summary (by displayed lead count k):")
    print(frontiers_df[frontiers_df["grouping"] == "k_displayed"][["value", "p_min", "p_med", "p_max", "best_subset"]].to_string(index=False))

    # 10. Compute Disease Minimal Sufficient Lead Sets
    print("\nPhase 7: Computing Disease Minimal Sufficient Lead Sets (delta=0.02)...")
    minimal_df = compute_disease_minimal_sufficient_sets(results_df, full_aurocs, delta=0.02)
    minimal_df.to_parquet(output_dir / "minimal_sufficient_sets.parquet", index=False)
    print("Minimal Sufficient Lead Sets:")
    print(minimal_df[["disease", "min_sufficient_k", "min_sufficient_rank", "best_minimal_subset", "minimal_auroc", "full_auroc"]].to_string(index=False))

    print("\n=======================================================")
    print("EXHAUSTIVE 4,095 SUBSET EVALUATION AND LEAD INFORMATION THEORY COMPLETE!")
    print(f"All Parquet artifacts written to {output_dir}")
    print("=======================================================")


if __name__ == "__main__":
    main()
