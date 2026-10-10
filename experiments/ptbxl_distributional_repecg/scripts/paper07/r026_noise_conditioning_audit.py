#!/usr/bin/env python3
"""Milestone R026: Noise and Conditioning Audit for Canonical-Frame SetOperator.

Audits clinical and physical robustness of canonical recovery under measurement inconsistency:
    Y_observed = X Q^T + E
    Y_canon    = Y_observed (B_Q Q^+)^T = X B_Q^T + E (B_Q Q^+)^T

Objectives:
1. Exact Matrix Conditioning Analysis across all 13 benchmark configurations:
   - Condition number kappa(Q) = sigma_max(Q) / sigma_min^+(Q)
   - Operator norm of pseudoinverse mapping ||B_Q Q^+||_2
   - Theoretical noise amplification factor
2. Empirical Noise Stress Test on PTB-XL held-out cohort (N=2,173):
   - Evaluate SNR levels: Clean (inf dB), 40 dB, 30 dB, 20 dB, 10 dB
   - Compare CF-SetOp vs Original SetOp under identical measurement noise E
   - Track macro AUROC, max logit perturbation, and RMSE logit drift
3. Ill-Conditioning Stress Test:
   - Near-collinear operator pairs with separation epsilon in [1e-1, ..., 1e-7]
   - Audit SVD cutoff tolerance in observable_projector (tol=1e-7) preventing noise explosion
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import average_precision_score, roc_auc_score

# Disable cuDNN to avoid PyTorch 2.6 conv1d issue on A100
cudnn.enabled = False

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
import sys
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(EXPERIMENT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from repecg.paper07_operator import OperatorSetModel
from repecg.paper07_operator.canonical_frame import (
    canonical_frame,
    observable_projector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("r026_noise_conditioning_audit")

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
R024_DATA_DIR = CODEX_DIR / "paper07_operator/r024_operator_span"
RESPONSE_FIT = CODEX_DIR / "paper07_operator/development_representations/response_fit.npz"
VAL_CACHE = CODEX_DIR / "paper02_kernel_mean/development_select"
OUTPUT_DIR = CODEX_DIR / "paper07_operator/r026_noise_conditioning_audit"
CHECKPOINT_PATH = CODEX_DIR / "paper07_operator/r024_models/basis_only/checkpoint.pt"


CONFIGURATIONS = {
    "Q8_full": {
        "name": "Full 8 Independent Leads (I, II, V1-V6)",
        "indices": [0, 1, 2, 3, 4, 5, 6, 7],
    },
    "S6_precordial": {
        "name": "6 Precordial Leads (V1-V6)",
        "indices": [2, 3, 4, 5, 6, 7],
    },
    "S6_limb": {
        "name": "6 Limb Leads (I, II, III, aVR, aVL, aVF)",
        "indices": [0, 1, 8, 9, 10, 11],
    },
    "Derived_limb_only": {
        "name": "4 Derived Limb Leads (III, aVR, aVL, aVF)",
        "indices": [8, 9, 10, 11],
    },
    "S3_reduced": {
        "name": "3-Lead ICU Monitoring (I, II, V2)",
        "indices": [0, 1, 3],
    },
    "S2_reduced": {
        "name": "2-Lead Bipolar (I, II)",
        "indices": [0, 1],
    },
    "S1_smartwatch": {
        "name": "1-Lead Smartwatch (Lead I)",
        "indices": [0],
    },
    "Lead_III_single": {
        "name": "1-Lead Single Derived (Lead III)",
        "indices": [8],
    },
    "aVF_single": {
        "name": "1-Lead Single Augmented (aVF)",
        "indices": [11],
    },
    "S_icm": {
        "name": "Oblique Subcutaneous Vector (V3 - V2)",
        "indices": [12],
    },
    "S_icm_plus_leadI": {
        "name": "Wearable Combo: Lead I + (V3 - V2)",
        "indices": [0, 12],
    },
    "Test_obliques_3": {
        "name": "3 Held-Out Oblique Linear Combinations",
        "indices": [13, 14, 15],
    },
    "All_16_leads": {
        "name": "All 16 Available Measurement Channels",
        "indices": list(range(16)),
    },
}


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    class_aurocs = []
    class_auprcs = []
    for c in range(y_true.shape[1]):
        labels_c = y_true[:, c]
        if len(np.unique(labels_c)) > 1:
            class_aurocs.append(float(roc_auc_score(labels_c, y_prob[:, c])))
            class_auprcs.append(float(average_precision_score(labels_c, y_prob[:, c])))
        else:
            class_aurocs.append(0.5)
            class_auprcs.append(0.0)
    return {
        "macro_auroc": float(np.mean(class_aurocs)),
        "micro_auroc": float(roc_auc_score(y_true.ravel(), y_prob.ravel())),
        "macro_auprc": float(np.mean(class_auprcs)),
    }


class WaveformResponseComputer:
    """Computes Phase-KME responses directly from arbitrary 1D waveforms."""
    def __init__(self, fit_path: Path, device: torch.device):
        data = np.load(fit_path)
        self.v_scale = float(data["voltage_scale"])
        self.std = data["physical_std"].astype(np.float32)
        self.mean = data["physical_mean"].astype(np.float32)
        self.landmarks = torch.as_tensor(data["landmarks"], device=device, dtype=torch.float32)
        self.inv_root = torch.as_tensor(data["inverse_root"], device=device, dtype=torch.float32)
        self.c2 = float(data["c2"])
        self.w_mean = torch.as_tensor(data["whitening_mean"], device=device, dtype=torch.float32)
        self.w_comp = torch.as_tensor(data["whitening_components"], device=device, dtype=torch.float32)
        self.w_scale = torch.as_tensor(data["whitening_scales"], device=device, dtype=torch.float32)
        self.device = device

    def compute_record_responses_from_waveforms(self, waveforms: np.ndarray) -> np.ndarray:
        # waveforms shape: (beat, 256, m) -> transpose to (m, beat, 256)
        n_beats, t_len, m_channels = waveforms.shape
        w = waveforms.transpose(2, 0, 1).astype(np.float64)  # [m, beat, 256]
        wf = w / self.v_scale
        der = np.gradient(wf, axis=2)
        atoms = np.stack((wf, der), axis=-1)  # (m, beat, 256, 2)
        cells = atoms.reshape(m_channels, n_beats, 16, 16, 2).transpose(0, 2, 1, 3, 4).reshape(m_channels, 16, -1, 2)
        t_cells = torch.as_tensor(cells, device=self.device, dtype=torch.float32)
        white = (t_cells - self.w_mean) @ self.w_comp * self.w_scale
        d2 = torch.cdist(white, self.landmarks.expand(m_channels, 16, -1, -1)).square() + self.c2
        feat = torch.rsqrt(d2) @ self.inv_root
        return feat.mean(dim=2).cpu().numpy().astype(np.float32)  # (m, 16, 128)


def predict_setoperator(
    model: OperatorSetModel,
    operators: np.ndarray,
    responses: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    N = len(responses)
    is_shared_ops = (operators.ndim == 2)
    if is_shared_ops:
        o_base = torch.as_tensor(operators, dtype=torch.float32, device=device).unsqueeze(0)
    all_logits = []
    with torch.inference_mode():
        for i in range(0, N, batch_size):
            r_b = torch.as_tensor(responses[i : i + batch_size], dtype=torch.float32, device=device)
            B_cur = len(r_b)
            if is_shared_ops:
                o_b = o_base.expand(B_cur, -1, -1)
            else:
                o_b = torch.as_tensor(operators[i : i + batch_size], dtype=torch.float32, device=device)
            latent = model.encode_context(o_b, r_b)
            logits = model.head(latent)
            all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def audit_conditioning(Q: np.ndarray, tol: float = 1e-7) -> Dict[str, Any]:
    """Audit condition number, pseudoinverse operator norm, and amplification factor."""
    P, r = observable_projector(Q, tol=tol)
    B = canonical_frame(P, rank=r, tol=tol)
    Q_pinv = np.linalg.pinv(Q, rcond=tol)
    C_Q = B @ Q_pinv  # [r, m]
    
    # Singular values of Q
    sv = np.linalg.svd(Q, compute_uv=False)
    non_zero_sv = sv[sv > tol]
    kappa = float(non_zero_sv[0] / non_zero_sv[-1]) if len(non_zero_sv) > 0 else 1.0
    
    # Operator 2-norm of C_Q: max singular value of C_Q
    cq_norm = float(np.linalg.norm(C_Q, ord=2))
    cq_frob = float(np.linalg.norm(C_Q, ord="fro"))
    
    return {
        "m_measurements": int(Q.shape[0]),
        "rank_r": int(r),
        "singular_values": [float(x) for x in sv],
        "condition_number_kappa": kappa,
        "c_q_operator_norm_2": cq_norm,
        "c_q_frobenius_norm": cq_frob,
        "noise_amplification_factor": cq_norm,
    }


def run_ill_conditioned_stress_test(tol: float = 1e-7) -> Dict[str, Any]:
    """Test pairs of nearly collinear operators with separation epsilon."""
    logger.info("--- Running Ill-Conditioned Pair Stress Test ---")
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    e = np.eye(8, dtype=np.float64)
    results = {}
    
    for eps in epsilons:
        q1 = e[0]
        q2 = e[0] + eps * e[1]
        Q = np.stack([q1, q2], axis=0)
        
        info = audit_conditioning(Q, tol=tol)
        results[f"eps_{eps:.0e}"] = {
            "eps": float(eps),
            **info,
        }
        logger.info(
            f"Eps: {eps:.0e} | Rank: {info['rank_r']} | "
            f"Kappa: {info['condition_number_kappa']:.2e} | "
            f"||B Q+||_2: {info['c_q_operator_norm_2']:.4f}"
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Milestone R026 Noise & Conditioning Audit")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_eval_records", type=int, default=500)
    args = parser.parse_args()

    device = torch.device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Model & Assets
    logger.info(f"Loading SetOperator checkpoint from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    logger.info(f"Loading WaveformResponseComputer from {RESPONSE_FIT}...")
    computer = WaveformResponseComputer(RESPONSE_FIT, device)

    # Load validation metadata and real beats
    qc_frame = pd.read_csv(VAL_CACHE / "qc.csv").query("eligible").reset_index(drop=True)
    val_meta = np.load(R024_DATA_DIR / "representation_val_real_waveforms.npz")
    labels_all = val_meta["labels"]
    operators_bank = val_meta["operators"][0]  # [16, 8]

    # Subsample records for real-beat noise evaluation
    n_eval = min(args.n_eval_records, len(qc_frame))
    logger.info(f"Auditing real waveforms on N={n_eval} records...")
    beats_list = []
    labels_eval = labels_all[:n_eval]

    for i in range(n_eval):
        p = VAL_CACHE / qc_frame.iloc[i].artifact
        with np.load(p) as item:
            b = np.asarray(item["beats"], dtype=np.float32) * computer.std + computer.mean
            beats_list.append(b)

    # -------------------------------------------------------------
    # Part 1: Matrix Conditioning Audit Across All 13 Configurations
    # -------------------------------------------------------------
    logger.info("=== Part 1: Matrix Conditioning Across Configurations ===")
    conditioning_summary = {}
    for cfg_key, cfg_info in CONFIGURATIONS.items():
        sub_indices = cfg_info["indices"]
        Q = operators_bank[sub_indices]
        cond_info = audit_conditioning(Q, tol=1e-7)
        conditioning_summary[cfg_key] = cond_info
        logger.info(
            f"[{cfg_key:<18}] m={cond_info['m_measurements']} -> r={cond_info['rank_r']} | "
            f"Kappa: {cond_info['condition_number_kappa']:.3f} | "
            f"||B Q+||_2: {cond_info['c_q_operator_norm_2']:.4f}"
        )

    # -------------------------------------------------------------
    # Part 2: Noise Robustness Audit Across SNR Levels
    # -------------------------------------------------------------
    logger.info("=== Part 2: Empirical Measurement Noise Audit ===")
    rng = np.random.default_rng(args.seed)
    
    test_configs = ["Q8_full", "S6_limb", "Derived_limb_only", "All_16_leads", "S2_reduced", "S1_smartwatch", "S_icm"]
    snr_levels_db = [np.inf, 40.0, 30.0, 20.0, 10.0]

    noise_results = {}

    for cfg_key in test_configs:
        sub_indices = CONFIGURATIONS[cfg_key]["indices"]
        Q = operators_bank[sub_indices]
        m = len(sub_indices)
        P, r = observable_projector(Q, tol=1e-7)
        B = canonical_frame(P, rank=r, tol=1e-7)
        Q_pinv = np.linalg.pinv(Q, rcond=1e-7)
        C_Q = B @ Q_pinv  # [r, m]

        cfg_noise_data = {}

        # Observed beats are beats @ Q.T: each b is [beat, 256, 8] -> [beat, 256, m]
        clean_beats_obs = [b @ Q.T for b in beats_list]
        
        # 1. CF-SetOp on clean: Y_canon = Y_obs @ C_Q.T: [beat, 256, r]
        clean_beats_canon = [b_obs @ C_Q.T for b_obs in clean_beats_obs]
        resps_cf_clean = np.stack([computer.compute_record_responses_from_waveforms(b_c) for b_c in clean_beats_canon], axis=0)
        logits_cf_clean = predict_setoperator(model, B, resps_cf_clean, device, batch_size=args.batch_size)
        metrics_cf_clean = compute_metrics(labels_eval, torch.sigmoid(torch.tensor(logits_cf_clean)).numpy())

        # 2. Original SetOp on clean: consumes Y_obs directly
        resps_orig_clean = np.stack([computer.compute_record_responses_from_waveforms(b_obs) for b_obs in clean_beats_obs], axis=0)
        logits_orig_clean = predict_setoperator(model, Q, resps_orig_clean, device, batch_size=args.batch_size)
        metrics_orig_clean = compute_metrics(labels_eval, torch.sigmoid(torch.tensor(logits_orig_clean)).numpy())

        for snr in snr_levels_db:
            snr_tag = "clean" if np.isinf(snr) else f"{int(snr)}dB"
            
            if np.isinf(snr):
                cfg_noise_data[snr_tag] = {
                    "snr_db": float("inf"),
                    "cf_auroc": metrics_cf_clean["macro_auroc"],
                    "orig_auroc": metrics_orig_clean["macro_auroc"],
                    "cf_max_logit_drift": 0.0,
                    "orig_max_logit_drift": 0.0,
                    "cf_rmse_logit_drift": 0.0,
                    "orig_rmse_logit_drift": 0.0,
                }
                continue

            # Generate noise E proportional to signal power
            all_signals = np.concatenate([b_obs.reshape(-1, m) for b_obs in clean_beats_obs], axis=0)
            sig_rms = np.sqrt(np.mean(all_signals**2))
            noise_sigma = sig_rms / (10.0 ** (snr / 20.0))

            noisy_beats_obs = []
            for b_obs in clean_beats_obs:
                noise_e = rng.normal(0.0, noise_sigma, size=b_obs.shape).astype(np.float32)
                noisy_beats_obs.append(b_obs + noise_e)

            # CF: Y_canon = (Y_obs + E) @ C_Q.T
            noisy_beats_canon = [b_noisy @ C_Q.T for b_noisy in noisy_beats_obs]
            resps_cf_noisy = np.stack([computer.compute_record_responses_from_waveforms(b_c) for b_c in noisy_beats_canon], axis=0)
            logits_cf_noisy = predict_setoperator(model, B, resps_cf_noisy, device, batch_size=args.batch_size)
            metrics_cf_noisy = compute_metrics(labels_eval, torch.sigmoid(torch.tensor(logits_cf_noisy)).numpy())

            # Orig: consumes noisy directly
            resps_orig_noisy = np.stack([computer.compute_record_responses_from_waveforms(b_noisy) for b_noisy in noisy_beats_obs], axis=0)
            logits_orig_noisy = predict_setoperator(model, Q, resps_orig_noisy, device, batch_size=args.batch_size)
            metrics_orig_noisy = compute_metrics(labels_eval, torch.sigmoid(torch.tensor(logits_orig_noisy)).numpy())

            cf_drift = logits_cf_noisy - logits_cf_clean
            orig_drift = logits_orig_noisy - logits_orig_clean

            cfg_noise_data[snr_tag] = {
                "snr_db": float(snr),
                "noise_sigma_mv": float(noise_sigma),
                "cf_auroc": metrics_cf_noisy["macro_auroc"],
                "orig_auroc": metrics_orig_noisy["macro_auroc"],
                "cf_max_logit_drift": float(np.max(np.abs(cf_drift))),
                "orig_max_logit_drift": float(np.max(np.abs(orig_drift))),
                "cf_rmse_logit_drift": float(np.sqrt(np.mean(cf_drift**2))),
                "orig_rmse_logit_drift": float(np.sqrt(np.mean(orig_drift**2))),
            }
            logger.info(
                f"[{cfg_key:<16}] SNR: {snr_tag:>5} | "
                f"CF AUROC: {metrics_cf_noisy['macro_auroc']:.4f} (drift RMSE: {np.sqrt(np.mean(cf_drift**2)):.4f}) | "
                f"Orig AUROC: {metrics_orig_noisy['macro_auroc']:.4f} (drift RMSE: {np.sqrt(np.mean(orig_drift**2)):.4f})"
            )

        noise_results[cfg_key] = cfg_noise_data

    # -------------------------------------------------------------
    # Part 3: Ill-Conditioned Pair Stress Test
    # -------------------------------------------------------------
    ill_conditioned_results = run_ill_conditioned_stress_test(tol=1e-7)

    # Compile Final Audit Summary
    final_audit = {
        "milestone": "R026",
        "title": "Noise and Conditioning Audit of Canonical-Frame SetOperator",
        "conditioning_by_config": conditioning_summary,
        "empirical_noise_stress_test": noise_results,
        "ill_conditioned_pair_stress_test": ill_conditioned_results,
    }

    out_file = OUTPUT_DIR / "r026_noise_conditioning_audit.json"
    with open(out_file, "w") as f:
        json.dump(final_audit, f, indent=2)
    logger.info(f"Audit completed successfully. Results saved to {out_file}")


if __name__ == "__main__":
    main()
