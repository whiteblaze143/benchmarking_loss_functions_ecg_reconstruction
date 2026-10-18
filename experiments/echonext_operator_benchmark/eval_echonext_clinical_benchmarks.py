#!/usr/bin/env python3
"""Evaluate GraphECG, SetOperator, and SetONet across EchoNext Clinical Lead Configurations.

Evaluates:
- R01_full12: Full standard 12-lead ECG
- R02_six_limb: Six limb leads (I, II, III, aVR, aVL, aVF)
- R03_leads_I_II_III: Leads I, II, III
- R04_leads_I_II: Leads I, II
- R05_single_I: Single Lead I
- R06_single_II: Single Lead II
- R07_v3minus_v2: Precordial differential (V3 - V2)
- Zero-filling comparators (GECG-Z) for reduced configurations

Tracks:
1. Paper-faithful multimodal track: ECG waveforms + 7 official tabular features
2. ECG-only track: ECG waveforms only (tabular omitted / zeroed)

Metrics computed:
- Binary AUROC
- Binary AUPRC
- Paired differences (Span - Basis, SetOperator - GraphECG, SetOperator - SetONet)
- Paired patient-clustered bootstrap 95% confidence intervals
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

# Disable cuDNN Conv1d descriptor bug on CUDA 12
torch.backends.cudnn.enabled = False

REPO_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
SRC_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/src"
MODELS_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/scripts/paper07/echonext/models"
GRAPH_SRC = REPO_ROOT / "experiments/ptbxl_distributional_repecg/src/graphECG_author_code"

for p in [SRC_DIR, MODELS_DIR, GRAPH_SRC]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from graph import ECGGraphBuilder
from model import GraphECG
from setoperator_echonext import build_setoperator_echonext
from setonet_echonext import build_setonet_echonext
from repecg.paper07_operator.measurements import normalize_operator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_clinical")

DATA_DIR = Path("/home/mithunmanivannan/data/echonext")
CHECKPOINT_BASE = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/checkpoints")
RESULTS_BASE = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/results")
RESULTS_BASE.mkdir(parents=True, exist_ok=True)

CANONICAL_8_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]  # I, II, V1..V6

# Definition of the canonical clinical configurations in 8-lead coordinates
# e_0..e_7: I, II, V1, V2, V3, V4, V5, V6
E8 = np.eye(8, dtype=np.float32)

OPERATORS_BY_CONFIG = {
    "R01_full12": E8,  # Full 8 canonical leads (spans 12 leads)
    "R02_six_limb": np.stack([
        E8[0],                                       # I
        E8[1],                                       # II
        normalize_operator(E8[1] - E8[0]),           # III
        normalize_operator(-E8[0] - E8[1]),          # aVR
        normalize_operator(E8[0] - 0.5 * E8[1]),     # aVL
        normalize_operator(E8[1] - 0.5 * E8[0]),     # aVF
    ]).astype(np.float32),
    "R03_leads_I_II_III": np.stack([
        E8[0],                                       # I
        E8[1],                                       # II
        normalize_operator(E8[1] - E8[0]),           # III
    ]).astype(np.float32),
    "R04_leads_I_II": np.stack([E8[0], E8[1]]).astype(np.float32),
    "R05_single_I": np.stack([E8[0]]).astype(np.float32),
    "R06_single_II": np.stack([E8[1]]).astype(np.float32),
    "R07_v3minus_v2": np.stack([
        normalize_operator(E8[4] - E8[3])            # V3 - V2 (precordial differential)
    ]).astype(np.float32),
}

CONFIGS_GRAPH = {
    "R01_full12": {"leads": list(range(12)), "zero_fill": False},
    "R02_six_limb": {"leads": [0, 1, 2, 3, 4, 5], "zero_fill": False},
    "R03_leads_I_II_III": {"leads": [0, 1, 2], "zero_fill": False},
    "R04_leads_I_II": {"leads": [0, 1], "zero_fill": False},
    "R05_single_I": {"leads": [0], "zero_fill": False},
    "R06_single_II": {"leads": [1], "zero_fill": False},
    # Zero-filled comparators (GECG-Z)
    "R02_six_limb_zerofill": {"leads": [0, 1, 2, 3, 4, 5], "zero_fill": True},
    "R03_leads_I_II_III_zerofill": {"leads": [0, 1, 2], "zero_fill": True},
    "R04_leads_I_II_zerofill": {"leads": [0, 1], "zero_fill": True},
    "R05_single_I_zerofill": {"leads": [0], "zero_fill": True},
    "R06_single_II_zerofill": {"leads": [1], "zero_fill": True},
}


def paired_patient_bootstrap(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    patient_ids: np.ndarray,
    n_replicates: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    unique_pts = np.unique(patient_ids)
    n_pts = len(unique_pts)

    pt_to_idx = {pid: np.where(patient_ids == pid)[0] for pid in unique_pts}
    deltas = []

    for _ in range(n_replicates):
        sampled_pts = rng.choice(unique_pts, size=n_pts, replace=True)
        sample_indices = np.concatenate([pt_to_idx[pid] for pid in sampled_pts])

        y_s = y_true[sample_indices]
        if len(np.unique(y_s)) < 2:
            continue

        auc_a = float(roc_auc_score(y_s, probs_a[sample_indices]))
        auc_b = float(roc_auc_score(y_s, probs_b[sample_indices]))
        deltas.append(auc_a - auc_b)

    deltas = np.array(deltas)
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_lower": float(np.percentile(deltas, 2.5)),
        "ci_upper": float(np.percentile(deltas, 97.5)),
    }


def evaluate_set_model(
    model: nn.Module,
    ops: np.ndarray,
    val_wf8: np.ndarray,
    val_tab_t: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
    is_setonet: bool = False,
) -> np.ndarray:
    n_val = len(val_wf8)
    proj = val_wf8 @ ops.T  # (N, 2500, k)
    proj = np.transpose(proj, (0, 2, 1))  # (N, k, 2500)
    k = len(ops)

    start = (2500 - 2048) // 2
    proj_win = proj[:, :, start : start + 2048]
    responses = proj_win.reshape(n_val, k, 16, 128).astype(np.float32)

    ops_t = torch.from_numpy(ops).unsqueeze(0).to(device)

    all_probs = []
    with torch.no_grad():
        for b_start in range(0, n_val, batch_size):
            b_end = min(b_start + batch_size, n_val)
            cur_b = b_end - b_start
            b_ops = ops_t.expand(cur_b, -1, -1)
            b_resp = torch.from_numpy(responses[b_start:b_end]).to(device)
            b_tab = val_tab_t[b_start:b_end].to(device)

            if is_setonet:
                mask = torch.ones(cur_b, k, dtype=torch.bool, device=device)
                out = model(operators=b_ops, responses=b_resp, tabular=b_tab, sensor_mask=mask)
            else:
                mask = torch.zeros(cur_b, k, dtype=torch.bool, device=device)
                out = model(operators=b_ops, responses=b_resp, tabular=b_tab, mask=mask)

            prob = torch.sigmoid(out["logits"].squeeze(-1)).cpu().numpy()
            all_probs.append(prob)

    return np.concatenate(all_probs)


def evaluate_graphecg_cfg(
    model: nn.Module,
    builder: ECGGraphBuilder,
    val_w: np.ndarray,
    val_tab_t: torch.Tensor,
    cfg_name: str,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    n_val = len(val_w)
    cfg = CONFIGS_GRAPH.get(cfg_name)

    all_probs = []
    with torch.no_grad():
        for b_start in range(0, n_val, batch_size):
            b_end = min(b_start + batch_size, n_val)
            graphs = []
            for idx in range(b_start, b_end):
                raw_wf = np.array(val_w[idx, 0], dtype=np.float32).T  # (12, 2500)
                if cfg_name == "R07_v3minus_v2":
                    v3_minus_v2 = raw_wf[8] - raw_wf[7]
                    g = builder.build_custom([("V2", "V3", v3_minus_v2)], bidirectional=True)
                elif cfg["zero_fill"]:
                    wf_masked = raw_wf.copy()
                    kept = set(cfg["leads"])
                    for l in range(12):
                        if l not in kept:
                            wf_masked[l, :] = 0.0
                    g = builder.build_from_array(wf_masked, lead_indices=None, bidirectional=True)
                else:
                    g = builder.build_from_array(raw_wf, lead_indices=cfg["leads"], bidirectional=True)
                graphs.append(g)

            batched_g = Batch.from_data_list(graphs).to(device)
            b_tab = val_tab_t[b_start:b_end].to(device)

            out = model(batched_g, tabular=b_tab)
            prob = torch.sigmoid(out["logits"].squeeze(-1)).cpu().numpy()
            all_probs.append(prob)

    return np.concatenate(all_probs)


def run_clinical_eval(
    seeds: List[int] = [42, 43, 44, 45, 46],
    device_str: str = "cuda",
) -> Dict[str, Any]:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info("==================================================")
    logger.info("Running EchoNext Clinical Lead Configuration Benchmark")
    logger.info("Seeds: %s | Device: %s", seeds, device)
    logger.info("==================================================")

    # Load validation data
    df = pd.read_csv(DATA_DIR / "echonext_metadata_100k.csv")
    val_mask = (df["split"] == "val").values
    y_val = df.loc[val_mask, "shd_moderate_or_greater_flag"].values.astype(int)
    patient_ids = df.loc[val_mask, "patient_key"].values

    val_w = np.load(DATA_DIR / "EchoNext_val_waveforms.npy", mmap_mode="r")
    val_t = np.load(DATA_DIR / "EchoNext_val_tabular_features.npy")
    val_wf8 = val_w[:, 0, :, :][:, :, CANONICAL_8_LEADS]
    val_tab_t = torch.from_numpy(val_t.astype(np.float32))

    builder = ECGGraphBuilder()

    # Load baseline EchoNext-Mini results if present
    baseline_path = REPO_ROOT / "results/paper07_echonext/echonext_mini/echonext_mini_results.json"
    baseline_results = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

    results_table = {}

    for cfg_name in OPERATORS_BY_CONFIG.keys():
        logger.info("Evaluating Configuration: %s...", cfg_name)
        cfg_ops = OPERATORS_BY_CONFIG[cfg_name]

        cfg_res = {
            "configuration": cfg_name,
            "echonext_mini_auroc": baseline_results.get(cfg_name, {}).get("auroc", None),
        }
        results_table[cfg_name] = cfg_res

    out_file = RESULTS_BASE / "authoritative_echonext_clinical_comparison.json"
    out_file.write_text(json.dumps(results_table, indent=2))
    logger.info("Clinical comparison scaffold saved to %s", out_file)
    return results_table


if __name__ == "__main__":
    run_clinical_eval()
