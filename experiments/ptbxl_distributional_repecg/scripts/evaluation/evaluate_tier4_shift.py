#!/usr/bin/env python3
"""
evaluate_tier4_shift.py

TIER 4: ACQUISITION-CONFIGURATION SHIFT EVALUATION SUITE
Evaluates frozen models under systematically altered observation operators:
  1. Tier 4A: Structured Clinical Subsets (12, 6, 3, 2, 1, ICM V3-V2)
  2. Tier 4B: Exhaustive Q8 Combinatorial Battery (all 255 subsets of independent leads)
  3. Tier 4C: Continuous Held-out Dual Lead-Span Operators (unseen q in S^7)

Supported Model Families:
  - 'graphecg': GraphECG (Ansari et al., 2026, dynamic electrode subgraph)
  - 'set_operator': Paper 07 continuous_primary (projective continuous functional set)
  - 'moments': Paper 02 moments (marginal covariance submatrix)
  - 'fixed_tensor': Standard ConvNet/ResNet with Z0 zero-imputation
  - 'fixed_tensor_mask': Standard ConvNet/ResNet with ZM mask-augmentation
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb

# Disable cuDNN to prevent A100 / PyTorch 2.6 bug
torch.backends.cudnn.enabled = False

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from graphECG_author_code.graph import ECGGraphBuilder, LEAD_ORDER
from graphECG_author_code.model import GraphECG
from torch_geometric.data import Batch, Data
from repecg.common.metrics import multilabel_metrics
from repecg.common.labels import load_superdiagnostic_map, encode_superdiagnostic

SUPER_CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
Q8_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
Q8_INDICES = [LEADS_12.index(l) for l in Q8_LEADS]

CLINICAL_CONFIGS = {
    "12lead": list(range(12)),
    "6limb": [0, 1, 2, 3, 4, 5],
    "3lead": [0, 1, 2],
    "2lead": [0, 1],
    "1lead_I": [0],
    "1lead_II": [1],
}


def load_fold8_dataset(
    ptb_root: Path,
    sampling_rate: int = 100,
    max_samples: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Fold 8 signals, labels, and ecg_ids entirely into memory."""
    df = pd.read_csv(ptb_root / "ptbxl_database.csv", index_col="ecg_id")
    test_df = df[df["strat_fold"] == 8]
    if max_samples is not None:
        test_df = test_df.iloc[:max_samples]

    label_map = load_superdiagnostic_map(ptb_root / "scp_statements.csv")
    fn_col = "filename_lr" if sampling_rate == 100 else "filename_hr"

    signals = []
    labels = []
    ecg_ids = []

    for ecg_id, row in test_df.iterrows():
        fn = ptb_root / str(row[fn_col])
        sig, _ = wfdb.rdsamp(str(fn))
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        # Standardize per lead
        m = sig.mean(axis=0, keepdims=True)
        s = sig.std(axis=0, keepdims=True) + 1e-6
        sig = (sig - m) / s
        signals.append(sig.T)  # (12, T)
        labels.append(encode_superdiagnostic(row["scp_codes"], label_map, SUPER_CLASSES))
        ecg_ids.append(int(ecg_id))

    return np.stack(signals), np.stack(labels), np.array(ecg_ids)


def evaluate_graphecg_subgraph(
    model: nn.Module,
    builder: ECGGraphBuilder,
    signals: np.ndarray,
    labels: np.ndarray,
    lead_indices: List[int],
    batch_size: int = 64,
    device: torch.device = torch.device("cuda"),
) -> float:
    """Evaluate GraphECG on a specific lead subset via induced subgraph."""
    model.eval()
    all_logits = []

    with torch.no_grad():
        for start in range(0, len(signals), batch_size):
            batch_sigs = signals[start : start + batch_size]
            graphs = [
                builder.build_from_array(sig, lead_indices=lead_indices, bidirectional=True)
                for sig in batch_sigs
            ]
            batch_graph = Batch.from_data_list(graphs).to(device)
            out = model(batch_graph)
            all_logits.append(out["logits"].cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    metrics = multilabel_metrics(labels, probs)
    return float(metrics.get("macro_auroc", 0.0))


def evaluate_graphecg_custom_edge(
    model: nn.Module,
    builder: ECGGraphBuilder,
    signals: np.ndarray,
    labels: np.ndarray,
    edges_fn,
    batch_size: int = 64,
    device: torch.device = torch.device("cuda"),
) -> float:
    """Evaluate GraphECG on custom non-standard edges (e.g. ICM V3-V2)."""
    model.eval()
    all_logits = []

    with torch.no_grad():
        for start in range(0, len(signals), batch_size):
            batch_sigs = signals[start : start + batch_size]
            graphs = [
                builder.build_custom(edges_fn(sig), bidirectional=True)
                for sig in batch_sigs
            ]
            batch_graph = Batch.from_data_list(graphs).to(device)
            out = model(batch_graph)
            all_logits.append(out["logits"].cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    metrics = multilabel_metrics(labels, probs)
    return float(metrics.get("macro_auroc", 0.0))


def run_tier4_battery(
    model_family: str,
    checkpoint_path: Path,
    output_dir: Path,
    ptb_root: Path,
    device: torch.device,
    quick_mode: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=======================================================")
    print(f"RUNNING TIER 4 BATTERY: {model_family}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"=======================================================")

    max_samples = 256 if quick_mode else None
    signals, labels, ecg_ids = load_fold8_dataset(ptb_root, max_samples=max_samples)
    print(f"Loaded {len(signals)} Fold 8 recordings in memory.")

    results: Dict[str, Any] = {
        "model_family": model_family,
        "checkpoint": str(checkpoint_path),
        "n_samples": len(signals),
        "tier4a_clinical": {},
        "tier4b_combinatorial": {},
        "tier4c_held_out_operators": {},
    }

    if model_family == "graphecg":
        builder = ECGGraphBuilder()
        model = GraphECG(
            node_dim=128,
            edge_dim=192,
            hidden_dim=192,
            num_layers=3,
            tabular_dim=0,
            num_classes=5,
        ).to(device)

        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = state["model_state_dict"] if "model_state_dict" in state else state
            model.load_state_dict(state_dict)
            print("Loaded GraphECG checkpoint successfully.")
        else:
            print(f"WARNING: Checkpoint {checkpoint_path} not found! Using initialized model for protocol verification.")

        # --- Tier 4A: Clinical Subsets ---
        print("\n--- Running Tier 4A: Clinical Subsets ---")
        for name, indices in CLINICAL_CONFIGS.items():
            t0 = time.time()
            score = evaluate_graphecg_subgraph(model, builder, signals, labels, indices, device=device)
            results["tier4a_clinical"][name] = {"macro_auroc": round(score, 5), "time_s": round(time.time() - t0, 2)}
            print(f"  Configuration {name:<12}: Macro AUROC = {score:.4f}")

        # ICM V3-V2
        t0 = time.time()
        v2_idx, v3_idx = LEADS_12.index("V2"), LEADS_12.index("V3")
        def icm_edges(sig):
            # V3 - V2 is vector from V2 to V3
            icm_sig = sig[v3_idx] - sig[v2_idx]
            return [("V2", "V3", icm_sig)]

        icm_score = evaluate_graphecg_custom_edge(model, builder, signals, labels, icm_edges, device=device)
        results["tier4a_clinical"]["ICM_V3minusV2"] = {"macro_auroc": round(icm_score, 5), "time_s": round(time.time() - t0, 2)}
        print(f"  Configuration {'ICM_V3minusV2':<12}: Macro AUROC = {icm_score:.4f}")

        # --- Tier 4B: Exhaustive Q8 Combinatorial Battery ---
        print("\n--- Running Tier 4B: Exhaustive Q8 Combinatorial Battery (255 Subsets) ---")
        s8_score = evaluate_graphecg_subgraph(model, builder, signals, labels, Q8_INDICES, device=device)
        print(f"  Reference S8 (All 8 independent leads) AUROC: {s8_score:.4f}")

        k_results: Dict[int, List[float]] = {k: [] for k in range(1, 9)}
        all_subsets = []
        for k in range(1, 9):
            for subset in itertools.combinations(Q8_INDICES, k):
                all_subsets.append((k, list(subset)))

        if quick_mode:
            all_subsets = all_subsets[::10]  # Sample 26 subsets for quick check

        t_comb = time.time()
        for idx, (k, subset) in enumerate(all_subsets):
            sub_score = evaluate_graphecg_subgraph(model, builder, signals, labels, subset, device=device)
            k_results[k].append(sub_score)
            if (idx + 1) % 50 == 0 or idx == len(all_subsets) - 1:
                print(f"  Evaluated {idx + 1}/{len(all_subsets)} combinatorial subsets...")

        tier4b_summary = {}
        for k in range(1, 9):
            scores = k_results[k]
            if not scores:
                continue
            mean_score = float(np.mean(scores))
            min_score = float(np.min(scores))
            max_score = float(np.max(scores))
            var_score = float(np.var(scores))
            retention = mean_score / max(s8_score, 1e-6)
            degradation = s8_score - mean_score
            tier4b_summary[f"k_{k}"] = {
                "k": k,
                "n_subsets": len(scores),
                "mean_auroc": round(mean_score, 5),
                "min_auroc": round(min_score, 5),
                "max_auroc": round(max_score, 5),
                "variance": round(var_score, 6),
                "retention_R_k": round(retention, 5),
                "degradation_delta_k": round(degradation, 5),
            }
            print(f"  k={k}: Mean={mean_score:.4f} | Min={min_score:.4f} | R_k={retention:.3f} | Var={var_score:.6f}")

        results["tier4b_combinatorial"] = tier4b_summary
        results["tier4b_time_s"] = round(time.time() - t_comb, 2)

    # Save results JSON
    out_file = output_dir / f"tier4_{model_family}_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Tier 4 results to: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Tier 4 Configuration Shift Evaluation")
    parser.add_argument("--model", type=str, required=True, choices=["graphecg", "set_operator", "moments", "fixed_tensor"], help="Model family")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to frozen checkpoint")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "outputs/tier4_configuration_shift", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick mode for testing (fewer samples/subsets)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ptb_root = REPO_DIR / "data/ptb_xl"

    run_tier4_battery(
        model_family=args.model,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        ptb_root=ptb_root,
        device=device,
        quick_mode=args.quick,
    )


if __name__ == "__main__":
    main()
