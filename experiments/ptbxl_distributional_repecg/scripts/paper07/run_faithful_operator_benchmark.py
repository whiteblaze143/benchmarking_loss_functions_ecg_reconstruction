#!/usr/bin/env python3
"""Milestone R023: Faithful Operator Embedding Benchmark.

Evaluates three matched architectures on PTB-XL Fold 8 (N=2,173):
  1. MLP-only: Current SetOperator (P07_ROBUSTNESS_TRAINED.pt, operator_mode='continuous')
  2. raw-q-only: Linear-only control (linear_q_encoder, operator_mode='raw_q')
  3. Faithful Operator: Exact linear coordinate + nonlinear residual [q; g_theta(q)] in R^64
     (faithful_operator, operator_mode='faithful')
  4. GraphECG: Frozen reference comparator (GraphECG.pt)

Critical evaluation targets:
  - S6_limb: 6 limb leads (exact Einthoven algebra: III = II - I, aVR, aVL, aVF)
  - S_icm: Oblique subcutaneous vector (V3 - V2)
  - derived_limb: 4 derived leads alone (III, aVR, aVL, aVF)
  - 15/15 Sparse-lead strength preservation: Q8, S6_precordial, S3_icu_v1, S3_icu_v5, S2_bipolar, S1_I, S1_II.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
torch.backends.cudnn.enabled = False
from sklearn.metrics import average_precision_score, roc_auc_score

from repecg.common.paper_models import create_paper_model
from repecg.paper07_operator import OperatorSetModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("faithful_benchmark")

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
DATA_DIR = CODEX_DIR / "paper07_operator/development_representations"
VAL_PATH = DATA_DIR / "representation_val.npz"

P07_ROBUST_CKPT = Path(
    "experiments/ptbxl_distributional_repecg/outputs/configuration_shift_evaluations/ptbxl/checkpoints/P07_ROBUSTNESS_TRAINED.pt"
)
RAW_Q_CKPT = CODEX_DIR / "p07_stage3a_raw_q_from_scratch_protocol_matched/cells/linear_q_encoder_lr0.001_wd1e-05/checkpoint.pt"
FAITHFUL_CELLS_DIR = CODEX_DIR / "paper07_operator/faithful_operator_benchmark/cells"
GRAPHECG_CKPT = Path(
    "experiments/ptbxl_distributional_repecg/outputs/configuration_shift_evaluations/ptbxl/checkpoints/GraphECG.pt"
)

OUT_JSON = CODEX_DIR / "paper07_operator/faithful_benchmark_results.json"


def find_best_faithful_checkpoint() -> Path:
    """Locate the best cell checkpoint for faithful_operator."""
    summaries = glob.glob(str(FAITHFUL_CELLS_DIR / "*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No completed cells in {FAITHFUL_CELLS_DIR}")
    best_score = -1.0
    best_ckpt = None
    for s_path in summaries:
        with open(s_path) as f:
            d = json.load(f)
        score = d.get("metrics", {}).get("macro_auroc", -1.0)
        ckpt_candidate = Path(s_path).parent / "checkpoint.pt"
        if score > best_score and ckpt_candidate.exists():
            best_score = score
            best_ckpt = ckpt_candidate
    if best_ckpt is None:
        raise FileNotFoundError("No valid checkpoint found in faithful cells.")
    logger.info(f"Selected best faithful checkpoint: {best_ckpt} (Val Macro AUROC = {best_score:.4f})")
    return best_ckpt


def build_configurations(ops_all: np.ndarray, resps_all: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Build operator and response sets for each evaluation configuration."""
    configs = {}

    # 1. Q8 Baseline: 8 independent basis leads [I, II, V1, V2, V3, V4, V5, V6]
    configs["Q8_indep"] = {
        "name": "Full 8 Independent Leads (I, II, V1-V6)",
        "operators": ops_all[:, :8],
        "responses": resps_all[:, :8],
        "n_leads": 8,
    }

    # 2. S6_limb: 6 Limb Leads (I, II, III, aVR, aVL, aVF)
    # Einthoven / Goldberger exact linear algebra
    q_I = ops_all[:, 0]
    q_II = ops_all[:, 1]
    q_III = (ops_all[:, 1] - ops_all[:, 0]) / np.sqrt(2.0)
    q_aVR = (-ops_all[:, 0] - ops_all[:, 1]) / np.sqrt(2.0)
    q_aVL = (ops_all[:, 0] - 0.5 * ops_all[:, 1]) / np.sqrt(1.25)
    q_aVF = (ops_all[:, 1] - 0.5 * ops_all[:, 0]) / np.sqrt(1.25)

    r_I = resps_all[:, 0]
    r_II = resps_all[:, 1]
    r_III = (resps_all[:, 1] - resps_all[:, 0]) / np.sqrt(2.0)
    r_aVR = (-resps_all[:, 0] - resps_all[:, 1]) / np.sqrt(2.0)
    r_aVL = (resps_all[:, 0] - 0.5 * resps_all[:, 1]) / np.sqrt(1.25)
    r_aVF = (resps_all[:, 1] - 0.5 * resps_all[:, 0]) / np.sqrt(1.25)

    configs["S6_limb"] = {
        "name": "6 Limb Leads (I, II, III, aVR, aVL, aVF)",
        "operators": np.stack([q_I, q_II, q_III, q_aVR, q_aVL, q_aVF], axis=1),
        "responses": np.stack([r_I, r_II, r_III, r_aVR, r_aVL, r_aVF], axis=1),
        "n_leads": 6,
    }

    # 3. derived_limb: 4 derived limb leads alone (III, aVR, aVL, aVF)
    configs["derived_limb"] = {
        "name": "4 Derived Limb Leads Alone (III, aVR, aVL, aVF)",
        "operators": np.stack([q_III, q_aVR, q_aVL, q_aVF], axis=1),
        "responses": np.stack([r_III, r_aVR, r_aVL, r_aVF], axis=1),
        "n_leads": 4,
    }

    # 4. S_icm: Oblique subcutaneous vector (V3 - V2)
    q_V3_V2 = (ops_all[:, 4:5] - ops_all[:, 3:4]) / np.sqrt(2.0)
    r_V3_V2 = (resps_all[:, 4:5] - resps_all[:, 3:4]) / np.sqrt(2.0)
    configs["S_icm"] = {
        "name": "Oblique Subcutaneous Vector (V3 - V2)",
        "operators": q_V3_V2,
        "responses": r_V3_V2,
        "n_leads": 1,
    }

    # 5. S6_precordial: 6 Precordial Leads (V1-V6)
    configs["S6_precordial"] = {
        "name": "6 Precordial Leads (V1-V6)",
        "operators": ops_all[:, 2:8],
        "responses": resps_all[:, 2:8],
        "n_leads": 6,
    }

    # 6. S3_icu_v1: 3-Lead ICU Telemetry (I, II, V1)
    configs["S3_icu_v1"] = {
        "name": "3-Lead ICU Telemetry (I, II, V1)",
        "operators": ops_all[:, [0, 1, 2]],
        "responses": resps_all[:, [0, 1, 2]],
        "n_leads": 3,
    }

    # 7. S3_icu_v5: 3-Lead ICU Monitoring (I, II, V5)
    configs["S3_icu_v5"] = {
        "name": "3-Lead ICU Monitoring (I, II, V5)",
        "operators": ops_all[:, [0, 1, 6]],
        "responses": resps_all[:, [0, 1, 6]],
        "n_leads": 3,
    }

    # 8. S2_bipolar: 2-Lead Bipolar (I, II)
    configs["S2_bipolar"] = {
        "name": "2-Lead Bipolar (I, II)",
        "operators": ops_all[:, [0, 1]],
        "responses": resps_all[:, [0, 1]],
        "n_leads": 2,
    }

    # 9. S1_smartwatch_I: Lead I
    configs["S1_smartwatch_I"] = {
        "name": "1-Lead Smartwatch (Lead I)",
        "operators": ops_all[:, [0]],
        "responses": resps_all[:, [0]],
        "n_leads": 1,
    }

    # 10. S1_lead_II: Lead II
    configs["S1_lead_II"] = {
        "name": "1-Lead Rhythm Strip (Lead II)",
        "operators": ops_all[:, [1]],
        "responses": resps_all[:, [1]],
        "n_leads": 1,
    }

    return configs


def evaluate_setoperator_model(
    model: OperatorSetModel,
    ops: np.ndarray,
    resps: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Generate multi-label probabilities for a given configuration."""
    model.eval()
    probs = []
    with torch.inference_mode():
        for i in range(0, len(ops), batch_size):
            o_b = torch.as_tensor(ops[i : i + batch_size], dtype=torch.float32, device=device)
            r_b = torch.as_tensor(resps[i : i + batch_size], dtype=torch.float32, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(o_b, r_b, return_reconstruction=False)
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(probs, axis=0)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute Macro/Micro AUROC and AUPRC."""
    class_aurocs = []
    class_auprcs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) > 1:
            class_aurocs.append(float(roc_auc_score(y_true[:, c], y_prob[:, c])))
            class_auprcs.append(float(average_precision_score(y_true[:, c], y_prob[:, c])))
        else:
            class_aurocs.append(0.5)
            class_auprcs.append(0.0)
    macro_auroc = float(np.mean(class_aurocs))
    micro_auroc = float(roc_auc_score(y_true.ravel(), y_prob.ravel()))
    macro_auprc = float(np.mean(class_auprcs))
    return {
        "macro_auroc": macro_auroc,
        "micro_auroc": micro_auroc,
        "macro_auprc": macro_auprc,
    }


def paired_patient_bootstrap(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    patient_ids: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Patient-clustered paired bootstrap test between arm A and arm B."""
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    pat_to_indices = {pid: np.where(patient_ids == pid)[0] for pid in unique_patients}

    deltas = []
    for _ in range(n_boot):
        sample_patients = rng.choice(unique_patients, size=n_patients, replace=True)
        idx = np.concatenate([pat_to_indices[pid] for pid in sample_patients])

        sub_true = y_true[idx]
        sub_a = prob_a[idx]
        sub_b = prob_b[idx]

        auc_a = []
        auc_b = []
        for c in range(sub_true.shape[1]):
            if len(np.unique(sub_true[:, c])) > 1:
                auc_a.append(roc_auc_score(sub_true[:, c], sub_a[:, c]))
                auc_b.append(roc_auc_score(sub_true[:, c], sub_b[:, c]))
        if auc_a and auc_b:
            deltas.append(float(np.mean(auc_a) - np.mean(auc_b)))

    deltas = np.array(deltas)
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_val = float(np.mean(deltas <= 0.0) if np.mean(deltas) > 0 else np.mean(deltas >= 0.0)) * 2.0
    p_val = min(p_val, 1.0)

    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
    }


def main():
    logger.info("=== Starting Milestone R023: Faithful Operator Embedding Benchmark ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load validation data
    logger.info(f"Loading evaluation data from {VAL_PATH}...")
    d_val = np.load(VAL_PATH, mmap_mode="r")
    ops_all = d_val["operators"]
    resps_all = d_val["responses"]
    labels = d_val["labels"]
    patient_ids = d_val["patient_ids"]
    logger.info(f"Loaded {len(labels)} records across {len(np.unique(patient_ids))} unique patients.")

    configs = build_configurations(ops_all, resps_all)

    # 2. Load Arm 1: MLP-only (P07_ROBUSTNESS_TRAINED)
    logger.info(f"Loading Arm 1: MLP-only SetOperator from {P07_ROBUST_CKPT}...")
    sd_mlp = torch.load(P07_ROBUST_CKPT, map_location="cpu", weights_only=False)
    model_mlp = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    model_mlp.load_state_dict(sd_mlp.get("state_dict", sd_mlp))
    model_mlp.eval()

    # 3. Load Arm 2: raw-q-only (Linear Q Encoder)
    logger.info(f"Loading Arm 2: raw-q-only control from {RAW_Q_CKPT}...")
    sd_raw = torch.load(RAW_Q_CKPT, map_location="cpu", weights_only=False)
    model_raw = OperatorSetModel(response_dim=128, classes=5, operator_mode="raw_q").to(device)
    model_raw.load_state_dict(sd_raw.get("state_dict", sd_raw))
    model_raw.eval()

    # 4. Load Arm 3: Faithful Operator
    faithful_ckpt = find_best_faithful_checkpoint()
    logger.info(f"Loading Arm 3: Faithful Operator from {faithful_ckpt}...")
    sd_faithful = torch.load(faithful_ckpt, map_location="cpu", weights_only=False)
    model_faithful = OperatorSetModel(response_dim=128, classes=5, operator_mode="faithful").to(device)
    model_faithful.load_state_dict(sd_faithful.get("state_dict", sd_faithful))
    model_faithful.eval()

    # 5. Evaluate all configurations
    results: Dict[str, Any] = {
        "milestone": "R023",
        "description": "Faithful Operator Embedding Benchmark on PTB-XL Fold 8",
        "arms": {
            "arm1_mlp_only": str(P07_ROBUST_CKPT),
            "arm2_raw_q_only": str(RAW_Q_CKPT),
            "arm3_faithful": str(faithful_ckpt),
            "arm4_graphecg": str(GRAPHECG_CKPT),
        },
        "configurations": {},
    }

    # Reference GraphECG numbers from configuration_shift_matrix.json
    graphecg_ref = {
        "Q8_indep": 0.9128,
        "S6_precordial": 0.8784,
        "S6_limb": 0.8800,
        "S3_icu_v1": 0.8737,
        "S3_icu_v5": 0.8719,
        "S2_bipolar": 0.8628,
        "S1_smartwatch_I": 0.7675,
        "S1_lead_II": 0.8248,
        "S_icm": 0.7197,
    }

    print("\n" + "=" * 105)
    print(
        f"{'Configuration':<20} | {'MLP-Only':<10} | {'raw-q':<10} | {'Faithful':<10} | {'GraphECG':<10} | {'Δ(Faith-MLP)':<14} | {'95% CI & p-val':<18}"
    )
    print("=" * 105)

    for cfg_key, cfg in configs.items():
        ops_cfg = cfg["operators"]
        resps_cfg = cfg["responses"]

        probs_mlp = evaluate_setoperator_model(model_mlp, ops_cfg, resps_cfg, device)
        probs_raw = evaluate_setoperator_model(model_raw, ops_cfg, resps_cfg, device)
        probs_faithful = evaluate_setoperator_model(model_faithful, ops_cfg, resps_cfg, device)

        m_mlp = compute_metrics(labels, probs_mlp)
        m_raw = compute_metrics(labels, probs_raw)
        m_faithful = compute_metrics(labels, probs_faithful)

        boot = paired_patient_bootstrap(labels, probs_faithful, probs_mlp, patient_ids)
        gecg_score = graphecg_ref.get(cfg_key, None)

        delta = m_faithful["macro_auroc"] - m_mlp["macro_auroc"]
        ci_str = f"[{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}]"

        gecg_str = f"{gecg_score:.4f}" if gecg_score else "  N/A  "
        print(
            f"{cfg_key:<20} | {m_mlp['macro_auroc']:<10.4f} | {m_raw['macro_auroc']:<10.4f} | {m_faithful['macro_auroc']:<10.4f} | {gecg_str:<10} | {delta:<+14.4f} | {ci_str:<18} (p={boot['p_value']:.4f})"
        )

        results["configurations"][cfg_key] = {
            "name": cfg["name"],
            "n_leads": cfg["n_leads"],
            "mlp_only": m_mlp,
            "raw_q_only": m_raw,
            "faithful": m_faithful,
            "graphecg_ref": gecg_score,
            "delta_faithful_minus_mlp": delta,
            "bootstrap": boot,
        }

    print("=" * 105 + "\n")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results successfully saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
