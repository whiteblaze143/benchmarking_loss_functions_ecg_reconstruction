#!/usr/bin/env python3
"""Aim 1: Calibration Scaling Curve Experiment.

Characterizes how physiological change detection and diagnostic accuracy scale
with the number of prior same-patient calibration observations:
K in {0, 1, 3, 5, all}

Addresses the core question:
"What is the minimum amount of patient-specific calibration required to make
physiological AI meaningfully more accurate than population models?"

Evaluates:
1. All Available Targets (Opportunistic Deployment)
2. Matched-Cohort Targets (Strict within-patient control to eliminate availability confounds)
3. Patient-Clustered Bootstrap 95% Confidence Intervals
4. Calibration Brier Score across K
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

from repecg.paper07_operator.model import OperatorSetModel


SCHEMA_VERSION = "prism_ecg_aim1_calibration_scaling_v1"
CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
EXPECTED_CHECKPOINT_SHA256 = "1071470f58e63594341a7a5df589b3db50c56b86befa65cb47a0b1da2dccbf68"
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
Q8_INDICES = (0, 1, 6, 7, 8, 9, 10, 11)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("aim1_calibration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fast_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact Mann-Whitney U AUROC computation."""
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(y_score)) + 1
    r1 = float(ranks[y_true == 1].sum())
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def compute_macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, list[float]]:
    class_aurocs = []
    for c in range(y_true.shape[1]):
        auc = fast_auroc(y_true[:, c], y_score[:, c])
        class_aurocs.append(auc)
    valid_aucs = [a for a in class_aurocs if not np.isnan(a)]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return macro_auc, class_aurocs


def extract_e6_latents(
    model: OperatorSetModel,
    operators: np.ndarray,
    responses: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract 256-D E6 latents from frozen SetOperator."""
    model.eval()
    sub_ops = torch.as_tensor(operators[list(Q8_INDICES)], dtype=torch.float32, device=device).unsqueeze(0)
    latents = []
    with torch.inference_mode():
        for start in range(0, len(responses), batch_size):
            end = min(start + batch_size, len(responses))
            batch_r = torch.as_tensor(np.array(responses[start:end, list(Q8_INDICES)]), dtype=torch.float32, device=device)
            batch_q = sub_ops.expand(end - start, -1, -1)
            z = model.encode_context(batch_q, batch_r)
            latents.append(z.cpu().numpy().astype(np.float32))
    return np.concatenate(latents, axis=0)


@dataclass
class TargetInstance:
    target_ecg: int
    patient_id: int
    target_idx: int
    strat_fold: int
    recording_date: pd.Timestamp
    z_target: np.ndarray
    y_multilabel: np.ndarray
    y_change: int  # 1 if diagnostic superclass changed from earliest baseline, 0 if unchanged
    # Historical baselines
    prior_k_means: dict[str, np.ndarray]  # "1", "3", "5", "all" -> baseline vector
    prior_counts: dict[str, int]
    prior_time_gaps: dict[str, float]  # mean delta_t in days


def build_calibration_dataset(
    ecg_ids: np.ndarray,
    patient_ids: np.ndarray,
    labels: np.ndarray,
    latents: np.ndarray,
    df_meta: pd.DataFrame,
    logger: logging.Logger,
) -> list[TargetInstance]:
    """Construct calibration target instances with strictly chronological historical sets."""
    df_sub = df_meta.loc[ecg_ids].copy()
    df_sub["recording_date"] = pd.to_datetime(df_sub["recording_date"])
    df_sub["array_idx"] = np.arange(len(ecg_ids))
    df_sub = df_sub.sort_values(["patient_id", "recording_date"])

    instances: list[TargetInstance] = []

    for pid, group in df_sub.groupby("patient_id", sort=False):
        n_obs = len(group)
        if n_obs < 2:
            continue
        recs = group.to_dict("records")

        # Earliest observation defines patient's initial state
        initial_label = labels[recs[0]["array_idx"]]

        for t_idx in range(1, n_obs):
            target = recs[t_idx]
            t_array_idx = target["array_idx"]
            z_t = latents[t_array_idx]
            y_t = labels[t_array_idx]
            target_date = target["recording_date"]

            # Prior observations strictly before target
            priors = [r for r in recs[:t_idx] if r["recording_date"] < target_date]
            if len(priors) == 0:
                continue

            prior_indices = [r["array_idx"] for r in priors]
            prior_dates = [r["recording_date"] for r in priors]

            # Binary change label: has the multilabel changed compared to baseline?
            y_change = int(not np.array_equal(y_t, initial_label))

            # Build K baselines
            prior_means = {}
            prior_counts = {}
            prior_gaps = {}

            # K = 1 (nearest earlier observation)
            last_idx = prior_indices[-1]
            prior_means["1"] = latents[last_idx]
            prior_counts["1"] = 1
            prior_gaps["1"] = (target_date - prior_dates[-1]).total_seconds() / 86400.0

            # K = 3 (up to 3 priors)
            k3_sub = prior_indices[-3:]
            prior_means["3"] = np.mean(latents[k3_sub], axis=0)
            prior_counts["3"] = len(k3_sub)
            prior_gaps["3"] = np.mean([(target_date - d).total_seconds() / 86400.0 for d in prior_dates[-3:]])

            # K = 5 (up to 5 priors)
            k5_sub = prior_indices[-5:]
            prior_means["5"] = np.mean(latents[k5_sub], axis=0)
            prior_counts["5"] = len(k5_sub)
            prior_gaps["5"] = np.mean([(target_date - d).total_seconds() / 86400.0 for d in prior_dates[-5:]])

            # K = all
            prior_means["all"] = np.mean(latents[prior_indices], axis=0)
            prior_counts["all"] = len(prior_indices)
            prior_gaps["all"] = np.mean([(target_date - d).total_seconds() / 86400.0 for d in prior_dates])

            instances.append(TargetInstance(
                target_ecg=int(target["ecg_id"]),
                patient_id=int(pid),
                target_idx=t_array_idx,
                strat_fold=int(target["strat_fold"]),
                recording_date=target_date,
                z_target=z_t,
                y_multilabel=y_t,
                y_change=y_change,
                prior_k_means=prior_means,
                prior_counts=prior_counts,
                prior_time_gaps=prior_gaps,
            ))

    logger.info(f"Built {len(instances):,} chronological target instances from {len(df_sub['patient_id'].unique()):,} patients.")
    return instances


def evaluate_probe_on_k(
    train_inst: list[TargetInstance],
    test_inst: list[TargetInstance],
    k_key: str | None,
    task: str = "change",  # "change" or "multilabel"
    c_val: float = 0.1,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Train linear probe on train split and evaluate on test split.
    
    If k_key is None -> K=0 (raw z_t).
    If k_key is '1', '3', '5', 'all' -> delta_z = z_t - baseline.
    """
    def make_features(instances: list[TargetInstance]) -> tuple[np.ndarray, np.ndarray]:
        X_list = []
        Y_list = []
        for inst in instances:
            if k_key is None:
                x = inst.z_target
            else:
                x = inst.z_target - inst.prior_k_means[k_key]
            X_list.append(x)
            if task == "change":
                Y_list.append(inst.y_change)
            else:
                Y_list.append(inst.y_multilabel)
        return np.array(X_list, dtype=np.float32), np.array(Y_list)

    X_train, Y_train = make_features(train_inst)
    X_test, Y_test = make_features(test_inst)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if task == "change":
        clf = LogisticRegression(C=c_val, max_iter=2000, random_state=42)
        clf.fit(X_train_s, Y_train)
        probs = clf.predict_proba(X_test_s)[:, 1]
        auc = fast_auroc(Y_test, probs)
        brier = float(np.mean((probs - Y_test) ** 2))
        return float(auc), probs, Y_test, np.array([brier])
    else:
        # Multilabel superclass
        probs = np.zeros_like(Y_test, dtype=np.float32)
        aurocs = []
        briers = []
        for c in range(Y_train.shape[1]):
            clf = LogisticRegression(C=c_val, max_iter=2000, random_state=42)
            clf.fit(X_train_s, Y_train[:, c])
            p = clf.predict_proba(X_test_s)[:, 1]
            probs[:, c] = p
            auc = fast_auroc(Y_test[:, c], p)
            aurocs.append(auc)
            briers.append(float(np.mean((p - Y_test[:, c]) ** 2)))
        macro_auc = float(np.mean(aurocs))
        return macro_auc, probs, Y_test, np.array(briers)


def patient_bootstrap_ci(
    test_inst: list[TargetInstance],
    y_test: np.ndarray,
    probs: np.ndarray,
    task: str = "change",
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute patient-clustered 95% bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    pids = np.array([inst.patient_id for inst in test_inst])
    unique_pats = np.unique(pids)
    pat_to_indices = {pid: np.where(pids == pid)[0] for pid in unique_pats}

    boot_metrics = []
    for _ in range(n_bootstraps):
        sampled_pats = rng.choice(unique_pats, size=len(unique_pats), replace=True)
        sample_indices = np.concatenate([pat_to_indices[p] for p in sampled_pats])

        y_sample = y_test[sample_indices]
        p_sample = probs[sample_indices]

        if task == "change":
            if len(np.unique(y_sample)) < 2:
                continue
            auc = fast_auroc(y_sample, p_sample)
        else:
            aucs = [fast_auroc(y_sample[:, c], p_sample[:, c]) for c in range(y_sample.shape[1])]
            valid = [a for a in aucs if not np.isnan(a)]
            auc = float(np.mean(valid)) if valid else float("nan")

        if not np.isnan(auc):
            boot_metrics.append(auc)

    boot_metrics = np.array(boot_metrics)
    point = float(np.median(boot_metrics))
    ci_low = float(np.percentile(boot_metrics, 2.5))
    ci_high = float(np.percentile(boot_metrics, 97.5))
    return point, ci_low, ci_high


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aim 1: Calibration Scaling Curve Experiment")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/ptbxl_distributional_repecg/outputs/configuration_shift_evaluations/ptbxl/checkpoints/P07_ROBUSTNESS_TRAINED.pt"),
    )
    parser.add_argument(
        "--contexts",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_full12_aux_mmd/frozen_full12_contexts"),
    )
    parser.add_argument(
        "--ptbxl-meta",
        type=Path,
        default=Path("data/ptbxl/ptbxl_database.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_aim1_calibration_scaling"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output / "run.log")

    logger.info("======================================================================")
    logger.info("=== PRISM-ECG Aim 1: Calibration Scaling Curve Experiment ===")
    logger.info("======================================================================")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output directory: {args.output}")

    # 1. Checkpoint Verification
    t0 = time.time()
    actual_hash = sha256_file(args.checkpoint)
    logger.info(f"Checkpoint SHA256: {actual_hash}")
    if actual_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Checkpoint hash mismatch!")

    # 2. Extract Latents
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    torch.backends.cudnn.enabled = False

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = OperatorSetModel(operator_mode="continuous", response_mode="phase_cnn").to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    operators = np.load(args.contexts / "operators.npy")
    train_res = np.load(args.contexts / "train_responses.npy", mmap_mode="r")
    train_ecg = np.load(args.contexts / "train_ecg_ids.npy")
    train_pat = np.load(args.contexts / "train_patient_ids.npy")
    train_lab = np.load(args.contexts / "train_labels.npy")

    val_res = np.load(args.contexts / "validation_responses.npy", mmap_mode="r")
    val_ecg = np.load(args.contexts / "validation_ecg_ids.npy")
    val_pat = np.load(args.contexts / "validation_patient_ids.npy")
    val_lab = np.load(args.contexts / "validation_labels.npy")

    logger.info("Extracting latents...")
    train_z = extract_e6_latents(model, operators, train_res, device=device)
    val_z = extract_e6_latents(model, operators, val_res, device=device)

    all_ecg = np.concatenate([train_ecg, val_ecg])
    all_pat = np.concatenate([train_pat, val_pat])
    all_z = np.concatenate([train_z, val_z], axis=0)
    all_lab = np.concatenate([train_lab, val_lab], axis=0)

    # 3. Load Metadata & Build Instances
    df_meta = pd.read_csv(args.ptbxl_meta, index_col="ecg_id")
    df_meta["ecg_id"] = df_meta.index.values

    instances = build_calibration_dataset(
        all_ecg, all_pat, all_lab, all_z, df_meta, logger
    )

    # Train / Test split: Train = Folds 1-7, Test = Fold 8
    train_instances = [inst for inst in instances if inst.strat_fold in range(1, 8)]
    test_instances = [inst for inst in instances if inst.strat_fold == 8]
    logger.info(f"Train instances (Folds 1-7): {len(train_instances):,}")
    logger.info(f"Test instances (Fold 8):     {len(test_instances):,}")

    k_conditions = [
        ("K=0 (Cold Start)", None),
        ("K=1 (1 Prior)", "1"),
        ("K=3 (3 Priors)", "3"),
        ("K=5 (5 Priors)", "5"),
        ("K=all (All Priors)", "all"),
    ]

    results_summary = {
        "schema_version": SCHEMA_VERSION,
        "n_train_instances": len(train_instances),
        "n_test_instances": len(test_instances),
        "opportunistic_change_detection": {},
        "matched_k3_change_detection": {},
        "diagnostic_superclass": {},
    }

    # 4. EXPERIMENT A: Opportunistic Physiological Change Detection across K
    logger.info("\n======================================================================")
    logger.info("--- Experiment A: Physiological Change Detection Scaling (All Targets) ---")
    logger.info("======================================================================")
    change_curve = {}
    base_auc_change = None

    for label, k_key in k_conditions:
        auc, probs, y_true, brier = evaluate_probe_on_k(
            train_instances, test_instances, k_key, task="change", c_val=0.1
        )
        point, ci_low, ci_high = patient_bootstrap_ci(
            test_instances, y_true, probs, task="change", n_bootstraps=args.n_bootstraps, seed=args.seed
        )
        if base_auc_change is None:
            base_auc_change = auc
            delta = 0.0
        else:
            delta = auc - base_auc_change

        change_curve[label] = {
            "auroc": float(auc),
            "delta_auroc": float(delta),
            "bootstrap_point": point,
            "ci_95": [ci_low, ci_high],
            "brier_score": float(brier[0]),
        }
        logger.info(f"{label:22s} | AUROC: {auc:.4f} [95% CI: {ci_low:.4f}, {ci_high:.4f}] | Delta: {delta:+.4f} | Brier: {brier[0]:.4f}")

    results_summary["opportunistic_change_detection"] = change_curve

    # 5. EXPERIMENT B: Matched-Cohort Control (Subset of patients with >= 3 priors)
    logger.info("\n======================================================================")
    logger.info("--- Experiment B: Matched-Cohort Control (Strict >=3 Priors Subgroup) ---")
    logger.info("======================================================================")
    matched_train = [inst for inst in train_instances if inst.prior_counts["all"] >= 3]
    matched_test = [inst for inst in test_instances if inst.prior_counts["all"] >= 3]
    logger.info(f"Matched Cohort Targets: Train = {len(matched_train)}, Test = {len(matched_test)}")

    matched_curve = {}
    base_auc_matched = None
    for label, k_key in k_conditions:
        auc, probs, y_true, brier = evaluate_probe_on_k(
            matched_train, matched_test, k_key, task="change", c_val=0.1
        )
        point, ci_low, ci_high = patient_bootstrap_ci(
            matched_test, y_true, probs, task="change", n_bootstraps=args.n_bootstraps, seed=args.seed
        )
        if base_auc_matched is None:
            base_auc_matched = auc
            delta = 0.0
        else:
            delta = auc - base_auc_matched

        matched_curve[label] = {
            "auroc": float(auc),
            "delta_auroc": float(delta),
            "ci_95": [ci_low, ci_high],
            "brier_score": float(brier[0]),
        }
        logger.info(f"{label:22s} | Matched AUROC: {auc:.4f} [95% CI: {ci_low:.4f}, {ci_high:.4f}] | Delta: {delta:+.4f}")

    results_summary["matched_k3_change_detection"] = matched_curve

    # 6. EXPERIMENT C: Target Diagnostic Superclass Accuracy across K
    logger.info("\n======================================================================")
    logger.info("--- Experiment C: Target Diagnostic Superclass Accuracy across K ---")
    logger.info("======================================================================")
    diag_curve = {}
    base_auc_diag = None
    for label, k_key in k_conditions:
        macro_auc, probs, y_true, briers = evaluate_probe_on_k(
            train_instances, test_instances, k_key, task="multilabel", c_val=0.1
        )
        point, ci_low, ci_high = patient_bootstrap_ci(
            test_instances, y_true, probs, task="multilabel", n_bootstraps=args.n_bootstraps, seed=args.seed
        )
        if base_auc_diag is None:
            base_auc_diag = macro_auc
            delta = 0.0
        else:
            delta = macro_auc - base_auc_diag

        diag_curve[label] = {
            "macro_auroc": float(macro_auc),
            "delta_auroc": float(delta),
            "ci_95": [ci_low, ci_high],
            "mean_brier": float(np.mean(briers)),
        }
        logger.info(f"{label:22s} | Diagnostic Macro AUROC: {macro_auc:.4f} [95% CI: {ci_low:.4f}, {ci_high:.4f}] | Delta: {delta:+.4f}")

    results_summary["diagnostic_superclass"] = diag_curve

    # 7. Generate Calibration Curve Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Plot A: Change Detection
    k_vals = [0, 1, 3, 5, 8]  # Map 'all' to 8 for visualization
    k_names = ["0", "1", "3", "5", "all"]
    aucs_change = [change_curve[k[0]]["auroc"] for k in k_conditions]
    err_change_low = [change_curve[k[0]]["auroc"] - change_curve[k[0]]["ci_95"][0] for k in k_conditions]
    err_change_high = [change_curve[k[0]]["ci_95"][1] - change_curve[k[0]]["auroc"] for k in k_conditions]

    axes[0].errorbar(k_vals, aucs_change, yerr=[err_change_low, err_change_high], fmt="o-", color="#1f77b4", lw=2, capsize=4, label="Change Detection")
    axes[0].set_xticks(k_vals)
    axes[0].set_xticklabels(k_names)
    axes[0].set_xlabel("Number of Historical Observations (K)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("AUROC", fontsize=11, fontweight="bold")
    axes[0].set_title("A. Physiological State Change Detection", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Plot B: Matched vs Diagnostic
    aucs_matched = [matched_curve[k[0]]["auroc"] for k in k_conditions]
    aucs_diag = [diag_curve[k[0]]["macro_auroc"] for k in k_conditions]

    axes[1].plot(k_vals, aucs_matched, "s--", color="#2ca02c", lw=2, label="Matched Cohort (>=3 priors)")
    axes[1].plot(k_vals, aucs_diag, "^-.", color="#ff7f0e", lw=2, label="Contemporaneous Diagnosis")
    axes[1].set_xticks(k_vals)
    axes[1].set_xticklabels(k_names)
    axes[1].set_xlabel("Number of Historical Observations (K)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Macro AUROC", fontsize=11, fontweight="bold")
    axes[1].set_title("B. Matched Control & Superclass Diagnosis", fontsize=12, fontweight="bold")
    axes[1].legend(loc="best")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_file = args.output / "calibration_scaling_curve.png"
    plt.savefig(plot_file)
    plt.close()
    logger.info(f"\nSaved calibration curve plot to {plot_file}")

    # Save summary json
    results_summary["elapsed_seconds"] = time.time() - t0
    out_json = args.output / "results.json"
    out_json.write_text(json.dumps(results_summary, indent=2) + "\n")
    logger.info(f"Saved complete results to {out_json}")
    logger.info("=== Aim 1 Calibration Scaling Experiment COMPLETED Successfully ===")


if __name__ == "__main__":
    main()
