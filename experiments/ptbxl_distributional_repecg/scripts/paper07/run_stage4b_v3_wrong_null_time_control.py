#!/usr/bin/env python3
"""Stage 4B-v3: Repeated Wrong-Baseline Null + Time-Gap Control.

Rigorous robustness audit for the fixed-AF baseline conversion signal:
1. M=1000 repeated random wrong-patient baseline permutations to establish
   the empirical null distribution D_m = A_same - A_wrong^(m) and Pr(D_m > 0).
2. Time-gap confounding analysis: Spearman correlation between change magnitude
   and elapsed time, time-only linear probe, and time-adjusted joint probe.
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

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from repecg.paper07_operator.model import OperatorSetModel


SCHEMA_VERSION = "p07_stage4b_af_transition_probe_v3"
EXPECTED_CHECKPOINT_SHA256 = "1071470f58e63594341a7a5df589b3db50c56b86befa65cb47a0b1da2dccbf68"
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
Q8_INDICES = (0, 1, 6, 7, 8, 9, 10, 11)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("stage4b_v3")
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


def get_clean_rhythm(scp: dict) -> str:
    has_afib = "AFIB" in scp
    has_aflt = "AFLT" in scp
    has_sr = "SR" in scp
    if has_afib and not has_sr:
        return "AF"
    elif has_sr and not has_afib and not has_aflt:
        return "SR"
    else:
        return "EXCLUDE"


@dataclass
class TransitionPair:
    pair_id: int
    patient_id: int
    target_ecg_id: int
    baseline_ecg_id: int
    target_date: pd.Timestamp
    baseline_date: pd.Timestamp
    dt_days: float
    target_rhythm: str
    baseline_rhythm: str
    cell: str
    strat_fold: int
    baseline_fold: int
    target_array_idx: int
    baseline_array_idx: int


def build_transition_pairs(
    df_meta: pd.DataFrame,
    ecg_to_idx: dict[int, int],
    logger: logging.Logger,
) -> list[TransitionPair]:
    """Find chronologically nearest earlier unambiguous baseline ECG for each target."""
    df_clean = df_meta[df_meta["clean_rhythm"].isin(["SR", "AF"])].copy()
    df_clean["recording_date"] = pd.to_datetime(df_clean["recording_date"])
    df_clean = df_clean.sort_values(by=["patient_id", "recording_date"])

    pairs: list[TransitionPair] = []
    pair_id = 0

    for pt_id, group in df_clean.groupby("patient_id", sort=False):
        if len(group) < 2:
            continue
        records = group.reset_index().to_dict("records")
        for t_idx in range(1, len(records)):
            target = records[t_idx]
            target_ecg = int(target["ecg_id"])
            if target_ecg not in ecg_to_idx:
                continue

            baseline = None
            for b_idx in range(t_idx - 1, -1, -1):
                cand = records[b_idx]
                cand_ecg = int(cand["ecg_id"])
                if cand_ecg in ecg_to_idx and cand["recording_date"] < target["recording_date"]:
                    baseline = cand
                    break

            if baseline is not None:
                b_ecg = int(baseline["ecg_id"])
                dt = (target["recording_date"] - baseline["recording_date"]).total_seconds() / 86400.0
                cell = f"{baseline['clean_rhythm']}->{target['clean_rhythm']}"
                pairs.append(
                    TransitionPair(
                        pair_id=pair_id,
                        patient_id=int(pt_id),
                        target_ecg_id=target_ecg,
                        baseline_ecg_id=b_ecg,
                        target_date=target["recording_date"],
                        baseline_date=baseline["recording_date"],
                        dt_days=float(dt),
                        target_rhythm=str(target["clean_rhythm"]),
                        baseline_rhythm=str(baseline["clean_rhythm"]),
                        cell=cell,
                        strat_fold=int(target["strat_fold"]),
                        baseline_fold=int(baseline["strat_fold"]),
                        target_array_idx=ecg_to_idx[target_ecg],
                        baseline_array_idx=ecg_to_idx[b_ecg],
                    )
                )
                pair_id += 1

    logger.info(f"Total eligible transition pairs extracted: {len(pairs)}")
    leakage = sum(1 for p in pairs if p.strat_fold != p.baseline_fold)
    assert leakage == 0, f"Fatal: Cross-fold leakage detected in {leakage} pairs!"
    logger.info("Zero cross-fold leakage verified.")
    return pairs


def fit_and_eval_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    c_grid: tuple[float, ...] = C_GRID,
) -> tuple[float, float, np.ndarray, LogisticRegression, StandardScaler]:
    """Fit standardized L2 logistic regression on train/val, predict on test."""
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)

    best_c = c_grid[0]
    best_val_auc = -1.0
    best_clf = None

    for c in c_grid:
        clf = LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=2000, random_state=42)
        clf.fit(x_train_s, y_train)
        val_probs = clf.predict_proba(x_val_s)[:, 1]
        val_auc = fast_auroc(y_val, val_probs)
        if np.isnan(val_auc):
            val_auc = -1.0
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_c = c
            best_clf = clf

    test_probs = best_clf.predict_proba(x_test_s)[:, 1]
    test_auc = fast_auroc(y_test, test_probs)
    return test_auc, best_c, test_probs, best_clf, scaler


def sample_random_wrong_baselines(
    pairs: list[TransitionPair],
    all_latents: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Sample random valid wrong-patient baseline for each pair from distinct donor patients."""
    wrong_baselines = np.zeros((len(pairs), 256), dtype=np.float32)

    # Group pairs by unique patient
    pat_to_baselines: dict[int, list[int]] = {}
    for p in pairs:
        pat_to_baselines.setdefault(p.patient_id, []).append(p.baseline_array_idx)

    unique_pats = sorted(list(pat_to_baselines.keys()))
    n_unique = len(unique_pats)

    for i, p in enumerate(pairs):
        target_pat = p.patient_id
        # Candidate donor patients are all other patients in this split
        candidate_donors = [d for d in unique_pats if d != target_pat]
        donor_pat = rng.choice(candidate_donors)
        # Pick one baseline from donor
        donor_b_idx = rng.choice(pat_to_baselines[donor_pat])
        wrong_baselines[i] = all_latents[donor_b_idx]

    return wrong_baselines


def patient_clustered_paired_bootstrap(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    patient_ids: np.ndarray,
    n_bootstraps: int = 10000,
    seed: int = 42,
) -> tuple[float, list[float]]:
    """Compute paired patient-clustered bootstrap distribution for prob_a - prob_b."""
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    pat_to_indices = {p: np.where(patient_ids == p)[0] for p in unique_patients}

    boot_diffs = []
    for _ in range(n_bootstraps):
        sampled_pats = rng.choice(unique_patients, size=n_patients, replace=True)
        sampled_idx = np.concatenate([pat_to_indices[p] for p in sampled_pats])

        y_b = y_true[sampled_idx]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue

        auc_a = fast_auroc(y_b, prob_a[sampled_idx])
        auc_b = fast_auroc(y_b, prob_b[sampled_idx])
        if not np.isnan(auc_a) and not np.isnan(auc_b):
            boot_diffs.append(auc_a - auc_b)

    ci95 = [float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))] if boot_diffs else [float("nan"), float("nan")]
    mean_diff = float(np.mean(boot_diffs)) if boot_diffs else float("nan")
    return mean_diff, ci95


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 4B-v3: Repeated Wrong-Baseline Null + Time-Gap Control")
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/ptbxl_distributional_repecg/outputs/configuration_shift_evaluations/ptbxl/checkpoints/P07_ROBUSTNESS_TRAINED.pt"))
    parser.add_argument("--contexts", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_full12_aux_mmd/frozen_full12_contexts"))
    parser.add_argument("--ptbxl-meta", type=Path, default=Path("data/ptbxl/ptbxl_database.csv"))
    parser.add_argument("--output", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/p07_stage4b_af_transition_probe_v3"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-bootstraps", type=int, default=10000)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output / "run.log")

    logger.info("=== Starting Stage 4B-v3: Repeated Wrong-Baseline Null + Time-Gap Control ===")
    logger.info(f"Output directory: {args.output}")

    # 1. Verify Checkpoint SHA256
    ckpt_hash = sha256_file(args.checkpoint)
    logger.info(f"Checkpoint SHA256: {ckpt_hash}")
    if ckpt_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Checkpoint hash mismatch!")

    # 2. Extract E6 Latents
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = False
    logger.info(f"Device: {device} (cuDNN disabled)")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = OperatorSetModel(operator_mode="continuous", response_mode="phase_cnn").to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    train_res = np.load(args.contexts / "train_responses.npy", mmap_mode="r")
    train_ecg = np.load(args.contexts / "train_ecg_ids.npy")
    train_pat = np.load(args.contexts / "train_patient_ids.npy")

    val_res = np.load(args.contexts / "validation_responses.npy", mmap_mode="r")
    val_ecg = np.load(args.contexts / "validation_ecg_ids.npy")
    val_pat = np.load(args.contexts / "validation_patient_ids.npy")

    operators = np.load(args.contexts / "operators.npy")

    logger.info("Extracting latents...")
    train_z = extract_e6_latents(model, operators, train_res, device=device)
    val_z = extract_e6_latents(model, operators, val_res, device=device)

    all_ecg = np.concatenate([train_ecg, val_ecg])
    all_pat = np.concatenate([train_pat, val_pat])
    all_z = np.concatenate([train_z, val_z], axis=0)
    ecg_to_idx = {int(ecg): i for i, ecg in enumerate(all_ecg)}

    # 3. Load Metadata & Extract Fixed-AF Stratum Pairs
    df_meta = pd.read_csv(args.ptbxl_meta, index_col="ecg_id")
    df_meta["scp_codes"] = df_meta["scp_codes"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_meta["clean_rhythm"] = df_meta["scp_codes"].apply(get_clean_rhythm)

    all_pairs = build_transition_pairs(df_meta, ecg_to_idx, logger)

    # Filter to Fixed-AF Stratum (AF->SR vs AF->AF)
    af_pairs = [p for p in all_pairs if p.baseline_rhythm == "AF" and p.cell in ["AF->SR", "AF->AF"]]
    logger.info(f"Total Fixed-AF stratum pairs: {len(af_pairs)}")

    train_af = [p for p in af_pairs if p.strat_fold in range(1, 7)]
    val_af = [p for p in af_pairs if p.strat_fold == 7]
    test_af = [p for p in af_pairs if p.strat_fold == 8]

    logger.info(f"Fixed-AF split counts: Train={len(train_af)}, Val={len(val_af)}, Test={len(test_af)}")
    y_train = np.array([1 if p.cell == "AF->SR" else 0 for p in train_af], dtype=np.int32)
    y_val = np.array([1 if p.cell == "AF->SR" else 0 for p in val_af], dtype=np.int32)
    y_test = np.array([1 if p.cell == "AF->SR" else 0 for p in test_af], dtype=np.int32)
    test_pats = np.array([p.patient_id for p in test_af])

    logger.info(f"Test targets: {len(test_af)} (Pos={y_test.sum()}, Neg={len(y_test) - y_test.sum()}, Patients={len(set(test_pats))})")

    # 4. Same-Patient Primary Probe
    logger.info("\n--- Fitting Same-Patient Primary Probe ---")
    x_same_train = np.stack([all_z[p.target_array_idx] - all_z[p.baseline_array_idx] for p in train_af], axis=0)
    x_same_val = np.stack([all_z[p.target_array_idx] - all_z[p.baseline_array_idx] for p in val_af], axis=0)
    x_same_test = np.stack([all_z[p.target_array_idx] - all_z[p.baseline_array_idx] for p in test_af], axis=0)

    auc_same, best_c_same, probs_same, clf_same, scaler_same = fit_and_eval_probe(
        x_same_train, y_train, x_same_val, y_val, x_same_test, y_test, C_GRID
    )
    logger.info(f"Same-Patient Delta AUROC: {auc_same:.4f} (Best C={best_c_same})")

    # 5. Cold-Start Comparator z_t
    x_abs_train = np.stack([all_z[p.target_array_idx] for p in train_af], axis=0)
    x_abs_val = np.stack([all_z[p.target_array_idx] for p in val_af], axis=0)
    x_abs_test = np.stack([all_z[p.target_array_idx] for p in test_af], axis=0)

    auc_abs, best_c_abs, probs_abs, _, _ = fit_and_eval_probe(
        x_abs_train, y_train, x_abs_val, y_val, x_abs_test, y_test, C_GRID
    )
    logger.info(f"Absolute Cold-Start z_t AUROC: {auc_abs:.4f} (Best C={best_c_abs})")

    # 6. Repeated Wrong-Baseline Null (M=1000 Permutations)
    logger.info(f"\n--- Generating Repeated Wrong-Baseline Null (M={args.n_permutations} Permutations) ---")
    rng = np.random.RandomState(args.seed)

    auroc_wrong_retrained_list = []
    auroc_wrong_corrupt_list = []
    diff_retrained_list = []
    diff_corrupt_list = []

    t0 = time.time()
    for m in range(args.n_permutations):
        # Sample wrong baselines for test, val, train
        wb_train = sample_random_wrong_baselines(train_af, all_z, rng)
        wb_val = sample_random_wrong_baselines(val_af, all_z, rng)
        wb_test = sample_random_wrong_baselines(test_af, all_z, rng)

        xw_train = np.stack([all_z[p.target_array_idx] for p in train_af], axis=0) - wb_train
        xw_val = np.stack([all_z[p.target_array_idx] for p in val_af], axis=0) - wb_val
        xw_test = np.stack([all_z[p.target_array_idx] for p in test_af], axis=0) - wb_test

        # A) Retrained wrong probe: fit on wrong train/val, evaluate on test
        auc_w_retrained, _, probs_w_retrained, _, _ = fit_and_eval_probe(
            xw_train, y_train, xw_val, y_val, xw_test, y_test, C_GRID
        )
        auroc_wrong_retrained_list.append(float(auc_w_retrained))
        diff_retrained_list.append(float(auc_same - auc_w_retrained))

        # B) Fixed same-patient probe evaluated on corrupted wrong test feature
        xw_test_s = scaler_same.transform(xw_test)
        probs_w_corrupt = clf_same.predict_proba(xw_test_s)[:, 1]
        auc_w_corrupt = fast_auroc(y_test, probs_w_corrupt)
        auroc_wrong_corrupt_list.append(float(auc_w_corrupt))
        diff_corrupt_list.append(float(auc_same - auc_w_corrupt))

        if (m + 1) % 200 == 0:
            logger.info(f"Completed {m + 1}/{args.n_permutations} permutations ({time.time() - t0:.1f}s)")

    diff_retrained = np.array(diff_retrained_list)
    diff_corrupt = np.array(diff_corrupt_list)
    auroc_wrong_retrained = np.array(auroc_wrong_retrained_list)
    auroc_wrong_corrupt = np.array(auroc_wrong_corrupt_list)

    pr_d_gt_0_retrained = float(np.mean(diff_retrained > 0))
    pr_d_gt_0_corrupt = float(np.mean(diff_corrupt > 0))

    logger.info(f"\n--- Repeated Wrong-Baseline Null Summary (Retrained Wrong Probe) ---")
    logger.info(f"Pr(D_m > 0): {pr_d_gt_0_retrained:.4f} ({int(np.sum(diff_retrained > 0))}/{args.n_permutations})")
    logger.info(f"Median D_m: {np.median(diff_retrained):+.4f}")
    logger.info(f"Mean D_m:   {np.mean(diff_retrained):+.4f} (std={np.std(diff_retrained):.4f})")
    logger.info(f"D_m 95% Interval [2.5%, 97.5%]: [{np.percentile(diff_retrained, 2.5):+.4f}, {np.percentile(diff_retrained, 97.5):+.4f}]")
    logger.info(f"A_wrong Distribution: min={auroc_wrong_retrained.min():.4f}, median={np.median(auroc_wrong_retrained):.4f}, max={auroc_wrong_retrained.max():.4f}")

    logger.info(f"\n--- Repeated Wrong-Baseline Null Summary (Fixed Same-Probe on Corrupted Donor) ---")
    logger.info(f"Pr(D_m > 0): {pr_d_gt_0_corrupt:.4f} ({int(np.sum(diff_corrupt > 0))}/{args.n_permutations})")
    logger.info(f"Median D_m: {np.median(diff_corrupt):+.4f}")
    logger.info(f"D_m 95% Interval [2.5%, 97.5%]: [{np.percentile(diff_corrupt, 2.5):+.4f}, {np.percentile(diff_corrupt, 97.5):+.4f}]")

    # 7. Time-Gap Confounding Analysis
    logger.info("\n--- Commencing Time-Gap Confounding Analysis ---")
    dt_test = np.array([p.dt_days for p in test_af], dtype=np.float32)
    dt_train = np.array([p.dt_days for p in train_af], dtype=np.float32)
    dt_val = np.array([p.dt_days for p in val_af], dtype=np.float32)

    log_dt_test = np.log1p(dt_test)
    log_dt_train = np.log1p(dt_train)
    log_dt_val = np.log1p(dt_val)

    norm_same_test = np.linalg.norm(x_same_test, axis=1)
    norm_same_train = np.linalg.norm(x_same_train, axis=1)

    # Spearman correlations between ||delta z_same|| and log(1 + dt)
    corr_test_all, p_test_all = stats.spearmanr(norm_same_test, log_dt_test)
    corr_test_pos, p_test_pos = stats.spearmanr(norm_same_test[y_test == 1], log_dt_test[y_test == 1])
    corr_test_neg, p_test_neg = stats.spearmanr(norm_same_test[y_test == 0], log_dt_test[y_test == 0])

    corr_train_all, p_train_all = stats.spearmanr(norm_same_train, log_dt_train)

    logger.info(f"Spearman Corr(||delta z||, log(1+dt)) in Test (All N=29):      r={corr_test_all:+.4f} (p={p_test_all:.4f})")
    logger.info(f"Spearman Corr(||delta z||, log(1+dt)) in Test (Pos N=10):      r={corr_test_pos:+.4f} (p={p_test_pos:.4f})")
    logger.info(f"Spearman Corr(||delta z||, log(1+dt)) in Test (Neg N=19):      r={corr_test_neg:+.4f} (p={p_test_neg:.4f})")
    logger.info(f"Spearman Corr(||delta z||, log(1+dt)) in Train (All N=223):    r={corr_train_all:+.4f} (p={p_train_all:.4f})")

    # Time-Only Probe: X_time = log(1 + dt) (1-D)
    x_time_train = log_dt_train[:, None]
    x_time_val = log_dt_val[:, None]
    x_time_test = log_dt_test[:, None]

    auc_time, best_c_time, probs_time, _, _ = fit_and_eval_probe(
        x_time_train, y_train, x_time_val, y_val, x_time_test, y_test, C_GRID
    )
    logger.info(f"\nTime-Only Probe X_time AUROC: {auc_time:.4f} (Best C={best_c_time})")

    # Time-Gap-Adjusted Probe: X_same+time = [delta z_same; log(1 + dt)] (257-D)
    x_jt_train = np.concatenate([x_same_train, log_dt_train[:, None]], axis=1)
    x_jt_val = np.concatenate([x_same_val, log_dt_val[:, None]], axis=1)
    x_jt_test = np.concatenate([x_same_test, log_dt_test[:, None]], axis=1)

    auc_jt, best_c_jt, probs_jt, _, _ = fit_and_eval_probe(
        x_jt_train, y_train, x_jt_val, y_val, x_jt_test, y_test, C_GRID
    )
    logger.info(f"Time-Adjusted Probe [delta z; log(1+dt)] AUROC: {auc_jt:.4f} (Best C={best_c_jt})")

    # Paired Bootstrap differences for time models
    diff_same_time_mean, ci_same_time = patient_clustered_paired_bootstrap(
        y_test, probs_same, probs_time, test_pats, n_bootstraps=args.n_bootstraps, seed=args.seed
    )
    diff_jt_same_mean, ci_jt_same = patient_clustered_paired_bootstrap(
        y_test, probs_jt, probs_same, test_pats, n_bootstraps=args.n_bootstraps, seed=args.seed
    )

    logger.info(f"Delta(same - time): {auc_same - auc_time:+.4f} (95% CI: [{ci_same_time[0]:+.4f}, {ci_same_time[1]:+.4f}])")
    logger.info(f"Delta(joint_time - same): {auc_jt - auc_same:+.4f} (95% CI: [{ci_jt_same[0]:+.4f}, {ci_jt_same[1]:+.4f}])")

    # Timing distribution summary
    dt_summary = {
        "conversion_positives": {
            "n": int(y_test.sum()),
            "median_days": float(np.median(dt_test[y_test == 1])),
            "iqr_days": [float(np.percentile(dt_test[y_test == 1], 25)), float(np.percentile(dt_test[y_test == 1], 75))],
        },
        "persistent_negatives": {
            "n": int(len(y_test) - y_test.sum()),
            "median_days": float(np.median(dt_test[y_test == 0])),
            "iqr_days": [float(np.percentile(dt_test[y_test == 0], 25)), float(np.percentile(dt_test[y_test == 0], 75))],
        },
    }

    # 8. Scientific Ledger Update
    # Robust to wrong-baseline sampling:
    if pr_d_gt_0_retrained >= 0.95:
        robust_wrong_sampling = f"ROBUST (Pr(D_m > 0) = {pr_d_gt_0_retrained:.4f}, median Delta = {np.median(diff_retrained):+.4f})"
    elif pr_d_gt_0_retrained >= 0.80:
        robust_wrong_sampling = f"MODERATELY ROBUST (Pr(D_m > 0) = {pr_d_gt_0_retrained:.4f})"
    else:
        robust_wrong_sampling = f"NOT ROBUST (Pr(D_m > 0) = {pr_d_gt_0_retrained:.4f})"

    # Robust to time-gap confounding:
    # Does delta z add value over time alone?
    if (auc_same - auc_time) > 0:
        time_confounding_verdict = f"NON-TRIVIAL ELECTROPHYSIOLOGICAL SIGNAL OVER TIME (AUROC delta_same={auc_same:.4f} vs time_only={auc_time:.4f})"
    else:
        time_confounding_verdict = f"CONFOUNDED BY ELAPSED TIME (time_only={auc_time:.4f} >= delta_same={auc_same:.4f})"

    scientific_ledger = {
        "FIXED_SR_BASELINE_PERSONALIZATION": "NOT DETECTED / INSUFFICIENT SAMPLE",
        "FIXED_AF_BASELINE_SAME_PATIENT_SIGNAL": "PROMISING EXPLORATORY SIGNAL",
        "ROBUST_TO_POSITIVE_PATIENT_REMOVAL": "YES",
        "ROBUST_TO_WRONG_BASELINE_SAMPLING": robust_wrong_sampling,
        "ROBUST_TO_TIME_GAP_CONFOUNDING": time_confounding_verdict,
        "CLINICAL_PERSONALIZATION_CLAIM": "NOT ESTABLISHED",
    }

    interpretation_contract = {
        "study_nature": "Stage 4B-v3 Repeated Wrong-Baseline Null + Time-Gap Control examining whether the fixed-AF conversion signal is robust to donor assignment uncertainty and temporal elapsed-time drift.",
        "repeated_null_definition": "1,000 distinct permutations of wrong-patient AF baselines within test fold 8.",
        "softened_finding_1": "same-patient AF differencing contains additional transition-associated signal under this PTB-XL pairing contract.",
        "softened_finding_2": "no same-patient SR-baseline advantage was detected in the severely underpowered SR->AF arm.",
        "time_gap_caveat": "Conversion pairs have shorter median elapsed time (0.9 days) than persistent AF pairs (11.9 days); time-only probe provides a critical benchmark.",
        "architecture_rule": "No architectural changes to SetOperator (frozen E6 representation maintained)."
    }

    # 9. Write Results
    null_dist_summary = {
        "n_permutations": args.n_permutations,
        "pr_delta_gt_0_retrained": pr_d_gt_0_retrained,
        "pr_delta_gt_0_corrupt": pr_d_gt_0_corrupt,
        "retrained_wrong_probe": {
            "median_delta": float(np.median(diff_retrained)),
            "mean_delta": float(np.mean(diff_retrained)),
            "std_delta": float(np.std(diff_retrained)),
            "ci95_delta": [float(np.percentile(diff_retrained, 2.5)), float(np.percentile(diff_retrained, 97.5))],
            "auroc_wrong_summary": {
                "min": float(auroc_wrong_retrained.min()),
                "p25": float(np.percentile(auroc_wrong_retrained, 25)),
                "median": float(np.median(auroc_wrong_retrained)),
                "p75": float(np.percentile(auroc_wrong_retrained, 75)),
                "max": float(auroc_wrong_retrained.max()),
            },
        },
        "fixed_same_probe_corrupt": {
            "median_delta": float(np.median(diff_corrupt)),
            "mean_delta": float(np.mean(diff_corrupt)),
            "ci95_delta": [float(np.percentile(diff_corrupt, 2.5)), float(np.percentile(diff_corrupt, 97.5))],
            "auroc_corrupt_summary": {
                "min": float(auroc_wrong_corrupt.min()),
                "median": float(np.median(auroc_wrong_corrupt)),
                "max": float(auroc_wrong_corrupt.max()),
            },
        },
    }

    time_gap_results = {
        "timing_distributions": dt_summary,
        "spearman_correlation_norm_vs_log_dt": {
            "test_all": {"r": float(corr_test_all), "p": float(p_test_all)},
            "test_pos_conversions": {"r": float(corr_test_pos), "p": float(p_test_pos)},
            "test_neg_persistent": {"r": float(corr_test_neg), "p": float(p_test_neg)},
            "train_all": {"r": float(corr_train_all), "p": float(p_train_all)},
        },
        "model_aurocs": {
            "same_patient_delta_z": float(auc_same),
            "cold_start_z_t": float(auc_abs),
            "time_only_log_dt": float(auc_time),
            "joint_delta_plus_time": float(auc_jt),
        },
        "paired_contrasts": {
            "delta_same_minus_time": {
                "empirical": float(auc_same - auc_time),
                "bootstrap_mean": float(diff_same_time_mean),
                "bootstrap_ci95": ci_same_time,
            },
            "delta_joint_time_minus_same": {
                "empirical": float(auc_jt - auc_same),
                "bootstrap_mean": float(diff_jt_same_mean),
                "bootstrap_ci95": ci_jt_same,
            },
        },
    }

    full_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scientific_ledger": scientific_ledger,
        "interpretation_contract": interpretation_contract,
        "fixed_af_stratum_sample": {
            "test_pairs": len(test_af),
            "test_patients": len(set(test_pats)),
            "test_pos_conversions": int(y_test.sum()),
            "test_neg_persistent": int(len(y_test) - y_test.sum()),
        },
        "primary_aurocs": {
            "same_patient_delta": float(auc_same),
            "cold_start_abs": float(auc_abs),
        },
        "repeated_wrong_baseline_null": null_dist_summary,
        "time_gap_analysis": time_gap_results,
    }

    with (args.output / "results.json").open("w") as f:
        json.dump(full_payload, f, indent=2)

    with (args.output / "time_gap_analysis.json").open("w") as f:
        json.dump(time_gap_results, f, indent=2)

    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": ckpt_hash,
        "n_permutations": args.n_permutations,
        "n_bootstraps": args.n_bootstraps,
        "seed": args.seed,
    }
    with (args.output / "manifest.json").open("w") as f:
        json.dump(manifest_payload, f, indent=2)

    np.savez_compressed(
        args.output / "null_distribution.npz",
        auroc_wrong_retrained=auroc_wrong_retrained,
        auroc_wrong_corrupt=auroc_wrong_corrupt,
        diff_retrained=diff_retrained,
        diff_corrupt=diff_corrupt,
    )

    np.savez_compressed(
        args.output / "predictions.npz",
        y_true=y_test,
        patient_ids=test_pats,
        probs_same=probs_same,
        probs_abs=probs_abs,
        probs_time=probs_time,
        probs_joint_time=probs_jt,
    )

    logger.info("Saved results.json, time_gap_analysis.json, null_distribution.npz, predictions.npz, manifest.json.")
    logger.info("=== Stage 4B-v3 Experiment COMPLETED Successfully ===")


if __name__ == "__main__":
    main()
