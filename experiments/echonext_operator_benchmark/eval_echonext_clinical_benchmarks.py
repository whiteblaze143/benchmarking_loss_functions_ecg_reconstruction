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

# 8 canonical coordinates in 12-lead array
CANONICAL_8_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]  # I, II, V1..V6

# Standard lead mapping
LEAD_NAME_TO_IDX = {
    "I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
    "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11,
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


def main():
    logger.info("EchoNext Clinical Benchmarks Evaluator Ready.")


if __name__ == "__main__":
    main()
