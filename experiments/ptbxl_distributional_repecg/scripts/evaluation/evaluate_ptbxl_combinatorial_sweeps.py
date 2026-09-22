#!/usr/bin/env python3
"""PTB-XL Combinatorial Single-Lead and Lead-Pair Sweeps (Interpretation B).

Protocol:
    Evaluate frozen models on Fold 8 test set across:
    - All 8 individual single-lead slices: I, II, V1, V2, V3, V4, V5, V6
    - Canonical 2-lead pairs: (I, II), (I, V1), (II, V1), (II, V5), (V1, V2), (V3, V4), (V5, V6)
    Zero probes, zero fine-tuning, strictly evaluating configuration shift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import wfdb

torch.backends.cudnn.enabled = False

from repecg.common.paper_models import create_paper_model
from repecg.common.variants import get_variants_for_paper
from repecg.paper07_operator import OperatorSetModel

REPO_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
EXP_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg"
OUTPUTS_DIR = EXP_DIR / "outputs"
DATA_DIR = REPO_ROOT / "data"
CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")

LEAD_NAMES = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]

# Single lead and pairwise configurations
SWEEP_CONFIGS = {}
for i, name in enumerate(LEAD_NAMES):
    SWEEP_CONFIGS[f"single_{name}"] = {
        "name": f"Single Lead {name}",
        "indices": [i],
    }

PAIRS = [
    ("I", "II", [0, 1]),
    ("I", "V1", [0, 2]),
    ("II", "V1", [1, 2]),
    ("II", "V5", [1, 6]),
    ("V1", "V2", [2, 3]),
    ("V3", "V4", [4, 5]),
    ("V5", "V6", [6, 7]),
]

for name1, name2, idxs in PAIRS:
    SWEEP_CONFIGS[f"pair_{name1}_{name2}"] = {
        "name": f"Pair ({name1}, {name2})",
        "indices": idxs,
    }


def load_fold8_ground_truth() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_full = OUTPUTS_DIR / "ptbxl_masking_control/representation_ptbxl_full_fold8.npz"
    d = np.load(p_full, mmap_mode="r")
    return d["labels"], d["ecg_ids"], d["patient_ids"]


def compute_multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    classwise_auroc = []
    classwise_auprc = []
    for c in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, c], y_prob[:, c])
            ap = average_precision_score(y_true[:, c], y_prob[:, c])
        except Exception:
            auc, ap = 0.5, 0.0
        classwise_auroc.append(float(auc))
        classwise_auprc.append(float(ap))

    return {
        "macro_auroc": float(np.mean(classwise_auroc)),
        "macro_auprc": float(np.mean(classwise_auprc)),
    }


def evaluate_paper07_sweeps(
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, Any]:
    p_reps = CODEX_DIR / "paper07_operator/development_representations/representation_val.npz"
    if not p_reps.exists():
        p_reps = OUTPUTS_DIR / "paper07_operator_reconstruction/development_representations/representation_val.npz"

    d = np.load(p_reps, mmap_mode="r")
    ops_all = d["operators"]
    resps_all = d["responses"]
    labels = d["labels"]

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    model.load_state_dict(sd)
    model.eval()

    results = {}
    for cfg_key, cfg in SWEEP_CONFIGS.items():
        indices = cfg["indices"]
        ops_sub = ops_all[:, indices]
        resps_sub = resps_all[:, indices]

        probs_list = []
        with torch.inference_mode():
            for i in range(0, len(labels), batch_size):
                o_b = torch.from_numpy(ops_sub[i : i + batch_size]).float().to(device)
                r_b = torch.from_numpy(resps_sub[i : i + batch_size]).float().to(device)
                logits = model(o_b, r_b, return_reconstruction=False)
                probs_list.append(torch.sigmoid(logits).float().cpu().numpy())

        probs = np.concatenate(probs_list)
        metrics = compute_multilabel_metrics(labels, probs)
        results[cfg_key] = {
            "name": cfg["name"],
            "indices": indices,
            "macro_auroc": round(metrics["macro_auroc"], 4),
            "macro_auprc": round(metrics["macro_auprc"], 4),
        }
    return results


def evaluate_fixed_tensor_sweeps(
    checkpoint_path: Path,
    model_id: int,
    variant_name: str,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, Any]:
    p_full = OUTPUTS_DIR / "ptbxl_masking_control/representation_ptbxl_full_fold8.npz"
    d = np.load(p_full, mmap_mode="r")
    k_all = d["kernel"]
    labels = d["labels"]

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    var_reg = get_variants_for_paper(model_id)
    var_obj = var_reg.get(variant_name, next(iter(var_reg.values())))
    model = create_paper_model(model_id, input_dim=256, classes=5, variant=var_obj).to(device)
    model.load_state_dict(sd)
    model.eval()

    results = {}
    for cfg_key, cfg in SWEEP_CONFIGS.items():
        indices = cfg["indices"]
        mask = np.zeros(8, dtype=np.float32)
        mask[indices] = 1.0

        probs_list = []
        with torch.inference_mode():
            for i in range(0, len(labels), batch_size):
                x_b = torch.from_numpy(k_all[i : i + batch_size]).clone().float().to(device)
                scale = torch.from_numpy(mask).float().to(device).repeat_interleave(32)
                x_b = x_b * scale.unsqueeze(0).unsqueeze(0)
                logits = model(x_b)
                probs_list.append(torch.sigmoid(logits).float().cpu().numpy())

        probs = np.concatenate(probs_list)
        metrics = compute_multilabel_metrics(labels, probs)
        results[cfg_key] = {
            "name": cfg["name"],
            "indices": indices,
            "macro_auroc": round(metrics["macro_auroc"], 4),
            "macro_auprc": round(metrics["macro_auprc"], 4),
        }
    return results


def evaluate_graphecg_sweeps(
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, Any]:
    from graphECG_author_code.graph import ECGGraphBuilder
    from graphECG_author_code.model import GraphECG
    from torch_geometric.data import Batch

    db_path = DATA_DIR / "ptbxl/ptbxl_database.csv"
    df = pd.read_csv(db_path)
    test_df = df[df["strat_fold"] == 8].reset_index(drop=True)

    LEAD_MAP = {0: 0, 1: 1, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10, 7: 11}
    labels_all = np.load(OUTPUTS_DIR / "ptbxl_masking_control/representation_ptbxl_full_fold8.npz")["labels"]

    builder = ECGGraphBuilder()
    model = GraphECG(node_dim=128, edge_dim=192, hidden_dim=192, num_layers=3, tabular_dim=0, num_classes=5).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()

    signals = []
    for fn in test_df["filename_lr"]:
        sig, _ = wfdb.rdsamp(str(DATA_DIR / "ptbxl" / fn))
        sig = np.nan_to_num(sig, nan=0.0).astype(np.float32)
        sig = (sig - sig.mean(axis=0, keepdims=True)) / (sig.std(axis=0, keepdims=True) + 1e-6)
        signals.append(sig.T)

    results = {}
    for cfg_key, cfg in SWEEP_CONFIGS.items():
        indices = cfg["indices"]
        sub_indices = [LEAD_MAP[i] for i in indices]

        probs_list = []
        with torch.inference_mode():
            for i in range(0, len(signals), batch_size):
                batch_sigs = signals[i : i + batch_size]
                graphs = [builder.build_from_array(s, lead_indices=sub_indices, bidirectional=True) for s in batch_sigs]
                batch_graph = Batch.from_data_list(graphs).to(device)
                out = model(batch_graph)
                logits = out["logits"] if isinstance(out, dict) else out
                probs_list.append(torch.sigmoid(logits).float().cpu().numpy())

        probs = np.concatenate(probs_list)
        metrics = compute_multilabel_metrics(labels_all[: len(probs)], probs)
        results[cfg_key] = {
            "name": cfg["name"],
            "indices": indices,
            "macro_auroc": round(metrics["macro_auroc"], 4),
            "macro_auprc": round(metrics["macro_auprc"], 4),
        }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = OUTPUTS_DIR / "configuration_shift_evaluations/ptbxl"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("PTB-XL Combinatorial Single-Lead & Pair Sweeps")
    print(f"Evaluating {len(SWEEP_CONFIGS)} lead configurations")
    print("=================================================================")

    sweeps_master = {}

    # P07 Robustness Trained
    p07_ckpt = out_dir / "checkpoints/P07_ROBUSTNESS_TRAINED.pt"
    if not p07_ckpt.exists():
        p07_ckpt = OUTPUTS_DIR / "paper07_operator_reconstruction/continuous_primary_best.pt"
    if p07_ckpt.exists():
        print("Evaluating P07_ROBUSTNESS_TRAINED sweeps...")
        sweeps_master["P07_ROBUSTNESS_TRAINED"] = evaluate_paper07_sweeps(p07_ckpt, device)

    # P07 Robustness Trained + Aux Recon
    p07_aux_ckpt = out_dir / "checkpoints/P07_ROBUSTNESS_TRAINED_AUX.pt"
    if not p07_aux_ckpt.exists():
        p07_aux_ckpt = OUTPUTS_DIR / "paper07_operator_reconstruction/continuous_auxiliary_best.pt"
    if p07_aux_ckpt.exists():
        print("Evaluating P07_ROBUSTNESS_TRAINED_AUX sweeps...")
        sweeps_master["P07_ROBUSTNESS_TRAINED_AUX"] = evaluate_paper07_sweeps(p07_aux_ckpt, device)

    # P07 Full-Lead Only
    p07_full_ckpt = out_dir / "checkpoints/P07_FULLLEAD_ONLY.pt"
    if not p07_full_ckpt.exists():
        p07_full_ckpt = OUTPUTS_DIR / "paper07_fulllead_only/continuous_primary_best.pt"
    if p07_full_ckpt.exists():
        print("Evaluating P07_FULLLEAD_ONLY sweeps...")
        sweeps_master["P07_FULLLEAD_ONLY"] = evaluate_paper07_sweeps(p07_full_ckpt, device)

    # GraphECG
    graphecg_ckpt = out_dir / "checkpoints/GraphECG.pt"
    if not graphecg_ckpt.exists():
        graphecg_ckpt = OUTPUTS_DIR / "graphecg/graphecg_ptbxl_best.pt"
    if graphecg_ckpt.exists():
        print("Evaluating GraphECG sweeps...")
        sweeps_master["GraphECG"] = evaluate_graphecg_sweeps(graphecg_ckpt, device)

    # FixedTensor P02
    p02_ckpt = out_dir / "checkpoints/FixedTensor_P02.pt"
    if not p02_ckpt.exists():
        p02_ckpt = OUTPUTS_DIR / "paper02_kernel_mean/development_training/kernel_best.pt"
    if p02_ckpt.exists():
        print("Evaluating FixedTensor_P02 sweeps...")
        sweeps_master["FixedTensor_P02"] = evaluate_fixed_tensor_sweeps(p02_ckpt, 2, "kernel", device)

    # Save JSON
    out_json = out_dir / "combinatorial_sweeps.json"
    with open(out_json, "w") as f:
        json.dump(sweeps_master, f, indent=2)

    # Generate Markdown Table
    md_lines = [
        "# PTB-XL Combinatorial Single-Lead and Pair Sweeps",
        "",
        "| Configuration | SetOperator (Robust) | SetOperator (Aux) | SetOperator (Full-Lead Only) | GraphECG (Stanford) | FixedTensor (P02) | Delta Robust vs GraphECG |",
        "|---|---|---|---|---|---|---|",
    ]

    for cfg_key, cfg in SWEEP_CONFIGS.items():
        name = cfg["name"]
        auc_p07_rob = sweeps_master.get("P07_ROBUSTNESS_TRAINED", {}).get(cfg_key, {}).get("macro_auroc", "—")
        auc_p07_aux = sweeps_master.get("P07_ROBUSTNESS_TRAINED_AUX", {}).get(cfg_key, {}).get("macro_auroc", "—")
        auc_p07_full = sweeps_master.get("P07_FULLLEAD_ONLY", {}).get(cfg_key, {}).get("macro_auroc", "—")
        auc_ge = sweeps_master.get("GraphECG", {}).get(cfg_key, {}).get("macro_auroc", "—")
        auc_fixed = sweeps_master.get("FixedTensor_P02", {}).get(cfg_key, {}).get("macro_auroc", "—")
        if isinstance(auc_p07_rob, float) and isinstance(auc_ge, float):
            diff = auc_p07_rob - auc_ge
            delta_ge = f"{'+' if diff >= 0 else ''}{diff:.4f}"
        else:
            delta_ge = "—"
        md_lines.append(f"| {name} | **{auc_p07_rob}** | {auc_p07_aux} | {auc_p07_full} | {auc_ge} | {auc_fixed} | **{delta_ge}** |")

    out_md = out_dir / "combinatorial_sweeps.md"
    with open(out_md, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n[Done] Combinatorial sweeps saved to {out_md}")


if __name__ == "__main__":
    main()
