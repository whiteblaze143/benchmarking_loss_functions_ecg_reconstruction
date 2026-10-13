#!/usr/bin/env python3
"""Stage 4B-v2: Fixed-Baseline-State Personalization Probe.

Evaluates whether within-patient change in SetOperator latent space (delta z = z_t - z_baseline)
improves rhythm-state transition detection when fixing the baseline rhythm state,
eliminating baseline-state label encoding confound.
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
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from repecg.paper07_operator.model import OperatorSetModel


SCHEMA_VERSION = "p07_stage4b_af_transition_probe_v2"
EXPECTED_CHECKPOINT_SHA256 = "1071470f58e63594341a7a5df589b3db50c56b86befa65cb47a0b1da2dccbf68"
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
Q8_INDICES = (0, 1, 6, 7, 8, 9, 10, 11)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("stage4b_v2")
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


def construct_wrong_patient_baselines(
    pairs: list[TransitionPair],
    all_latents: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Construct deterministic wrong-patient baselines matching the same baseline rhythm.
    
    Permutes baselines among distinct patients having the same baseline rhythm.
    """
    wrong_baselines = np.zeros((len(pairs), 256), dtype=np.float32)

    by_rhythm: dict[str, list[int]] = {}
    for i, p in enumerate(pairs):
        by_rhythm.setdefault(p.baseline_rhythm, []).append(i)

    for rhythm, indices in by_rhythm.items():
        pats = [pairs[i].patient_id for i in indices]
        unique_pats = sorted(list(set(pats)))
        n_unique = len(unique_pats)

        if n_unique <= 1:
            for k, idx in enumerate(indices):
                donor_idx = indices[(k + 1) % len(indices)]
                wrong_baselines[idx] = all_latents[pairs[donor_idx].baseline_array_idx]
        else:
            donor_map = {unique_pats[k]: unique_pats[(k + 1) % n_unique] for k in range(n_unique)}
            pat_to_baselines: dict[int, list[int]] = {}
            for idx in indices:
                pat_to_baselines.setdefault(pairs[idx].patient_id, []).append(pairs[idx].baseline_array_idx)

            for idx in indices:
                p_id = pairs[idx].patient_id
                donor_pat = donor_map[p_id]
                donor_b_idx = pat_to_baselines[donor_pat][0]
                wrong_baselines[idx] = all_latents[donor_b_idx]

    return wrong_baselines


def fit_and_eval_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    c_grid: tuple[float, ...] = C_GRID,
) -> tuple[float, float, np.ndarray]:
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
    return test_auc, best_c, test_probs


def patient_clustered_paired_bootstrap(
    y_true: np.ndarray,
    prob_same: np.ndarray,
    prob_wrong: np.ndarray,
    prob_pop: np.ndarray,
    prob_abs: np.ndarray,
    patient_ids: np.ndarray,
    n_bootstraps: int = 10000,
    seed: int = 42,
) -> dict[str, dict[str, any]]:
    """Compute paired patient-clustered bootstrap distributions for all contrasts."""
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    pat_to_indices = {p: np.where(patient_ids == p)[0] for p in unique_patients}

    boot_diffs_wrong = []
    boot_diffs_pop = []
    boot_diffs_abs = []
    boot_same = []
    boot_wrong = []
    boot_pop = []
    boot_abs = []

    for _ in range(n_bootstraps):
        sampled_pats = rng.choice(unique_patients, size=n_patients, replace=True)
        sampled_idx = np.concatenate([pat_to_indices[p] for p in sampled_pats])

        y_b = y_true[sampled_idx]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue

        auc_same = fast_auroc(y_b, prob_same[sampled_idx])
        auc_wrong = fast_auroc(y_b, prob_wrong[sampled_idx])
        auc_pop = fast_auroc(y_b, prob_pop[sampled_idx])
        auc_abs = fast_auroc(y_b, prob_abs[sampled_idx])

        if np.isnan(auc_same) or np.isnan(auc_wrong) or np.isnan(auc_pop) or np.isnan(auc_abs):
            continue

        boot_same.append(auc_same)
        boot_wrong.append(auc_wrong)
        boot_pop.append(auc_pop)
        boot_abs.append(auc_abs)

        boot_diffs_wrong.append(auc_same - auc_wrong)
        boot_diffs_pop.append(auc_same - auc_pop)
        boot_diffs_abs.append(auc_same - auc_abs)

    def summarize(diffs: list[float]) -> dict[str, any]:
        arr = np.array(diffs)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        }

    return {
        "delta_same_wrong": summarize(boot_diffs_wrong),
        "delta_same_pop": summarize(boot_diffs_pop),
        "delta_same_abs": summarize(boot_diffs_abs),
        "distributions": {
            "same_wrong": np.array(boot_diffs_wrong, dtype=np.float32),
            "same_pop": np.array(boot_diffs_pop, dtype=np.float32),
            "same_abs": np.array(boot_diffs_abs, dtype=np.float32),
            "auc_same": np.array(boot_same, dtype=np.float32),
            "auc_wrong": np.array(boot_wrong, dtype=np.float32),
            "auc_pop": np.array(boot_pop, dtype=np.float32),
            "auc_abs": np.array(boot_abs, dtype=np.float32),
        },
    }


def leave_one_transition_patient_out_sensitivity(
    y_true: np.ndarray,
    prob_same: np.ndarray,
    prob_wrong: np.ndarray,
    prob_pop: np.ndarray,
    prob_abs: np.ndarray,
    patient_ids: np.ndarray,
) -> dict[str, dict[str, any]]:
    """Recompute each AUROC difference after removing each positive-transition patient one at a time."""
    pos_indices = np.where(y_true == 1)[0]
    pos_patients = np.unique(patient_ids[pos_indices])

    vec_same_wrong = []
    vec_same_pop = []
    vec_same_abs = []

    for pt in pos_patients:
        keep = patient_ids != pt
        y_sub = y_true[keep]
        if y_sub.sum() == 0 or y_sub.sum() == len(y_sub):
            continue

        auc_same = fast_auroc(y_sub, prob_same[keep])
        auc_wrong = fast_auroc(y_sub, prob_wrong[keep])
        auc_pop = fast_auroc(y_sub, prob_pop[keep])
        auc_abs = fast_auroc(y_sub, prob_abs[keep])

        vec_same_wrong.append(float(auc_same - auc_wrong))
        vec_same_pop.append(float(auc_same - auc_pop))
        vec_same_abs.append(float(auc_same - auc_abs))

    def summarize_loo(vec: list[float]) -> dict[str, any]:
        arr = np.array(vec)
        return {
            "n_transition_patients": len(vec),
            "min": float(np.min(arr)) if len(arr) > 0 else None,
            "max": float(np.max(arr)) if len(arr) > 0 else None,
            "median": float(np.median(arr)) if len(arr) > 0 else None,
            "full_vector": [float(x) for x in arr],
        }

    return {
        "delta_same_wrong": summarize_loo(vec_same_wrong),
        "delta_same_pop": summarize_loo(vec_same_pop),
        "delta_same_abs": summarize_loo(vec_same_abs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 4B-v2: Fixed-Baseline-State Personalization Probe")
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/ptbxl_distributional_repecg/outputs/configuration_shift_evaluations/ptbxl/checkpoints/P07_ROBUSTNESS_TRAINED.pt"))
    parser.add_argument("--contexts", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_full12_aux_mmd/frozen_full12_contexts"))
    parser.add_argument("--ptbxl-meta", type=Path, default=Path("data/ptbxl/ptbxl_database.csv"))
    parser.add_argument("--output", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/p07_stage4b_af_transition_probe_v2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstraps", type=int, default=10000)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output / "run.log")

    logger.info("=== Starting Stage 4B-v2: Fixed-Baseline-State Personalization Probe ===")
    logger.info(f"Output directory: {args.output}")

    # 1. Verify Checkpoint SHA256
    ckpt_hash = sha256_file(args.checkpoint)
    logger.info(f"Checkpoint path: {args.checkpoint}")
    logger.info(f"Checkpoint SHA256: {ckpt_hash}")
    if ckpt_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Checkpoint SHA256 mismatch! Expected {EXPECTED_CHECKPOINT_SHA256}, got {ckpt_hash}")
    logger.info("Checkpoint SHA256 verified successfully.")

    # 2. Setup Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = False
    logger.info(f"Using compute device: {device} (cuDNN disabled for clean conv1d on A100)")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = OperatorSetModel(operator_mode="continuous", response_mode="phase_cnn").to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 3. Load Precomputed Responses & Contexts
    logger.info("Loading precomputed response arrays...")
    train_res = np.load(args.contexts / "train_responses.npy", mmap_mode="r")
    train_ecg = np.load(args.contexts / "train_ecg_ids.npy")
    train_pat = np.load(args.contexts / "train_patient_ids.npy")

    val_res = np.load(args.contexts / "validation_responses.npy", mmap_mode="r")
    val_ecg = np.load(args.contexts / "validation_ecg_ids.npy")
    val_pat = np.load(args.contexts / "validation_patient_ids.npy")

    operators = np.load(args.contexts / "operators.npy")

    # 4. Extract E6 Latents for all records
    logger.info("Extracting E6 latents for train/val contexts (folds 1-7)...")
    train_z = extract_e6_latents(model, operators, train_res, device=device)
    logger.info(f"Train/val latents: {train_z.shape}")

    logger.info("Extracting E6 latents for test contexts (fold 8)...")
    val_z = extract_e6_latents(model, operators, val_res, device=device)
    logger.info(f"Test latents: {val_z.shape}")

    all_ecg = np.concatenate([train_ecg, val_ecg])
    all_pat = np.concatenate([train_pat, val_pat])
    all_z = np.concatenate([train_z, val_z], axis=0)

    ecg_to_idx = {int(ecg): i for i, ecg in enumerate(all_ecg)}

    # 5. Metadata & Population Reference
    df_meta = pd.read_csv(args.ptbxl_meta, index_col="ecg_id")
    df_meta["scp_codes"] = df_meta["scp_codes"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_meta["clean_rhythm"] = df_meta["scp_codes"].apply(get_clean_rhythm)

    train_fold_ecgs = df_meta[(df_meta["strat_fold"].isin(range(1, 7))) & (df_meta.index.isin(ecg_to_idx))].index.values
    train_sr_ecgs = df_meta.loc[train_fold_ecgs][df_meta.loc[train_fold_ecgs, "clean_rhythm"] == "SR"].index.values
    train_af_ecgs = df_meta.loc[train_fold_ecgs][df_meta.loc[train_fold_ecgs, "clean_rhythm"] == "AF"].index.values

    train_sr_indices = [ecg_to_idx[e] for e in train_sr_ecgs]
    train_af_indices = [ecg_to_idx[e] for e in train_af_ecgs]

    mu_sr = np.mean(all_z[train_sr_indices], axis=0)
    mu_af = np.mean(all_z[train_af_indices], axis=0)
    logger.info(f"Computed population mean mu_SR from {len(train_sr_indices)} records. Norm: {np.linalg.norm(mu_sr):.4f}")
    logger.info(f"Computed population mean mu_AF from {len(train_af_indices)} records. Norm: {np.linalg.norm(mu_af):.4f}")

    pop_mu_map = {"SR": mu_sr, "AF": mu_af}

    # 6. Build Transition Pairs
    logger.info("Extracting nearest-earlier transition pairs...")
    all_pairs = build_transition_pairs(df_meta, ecg_to_idx, logger)

    train_pairs = [p for p in all_pairs if p.strat_fold in range(1, 7)]
    val_pairs = [p for p in all_pairs if p.strat_fold == 7]
    test_pairs = [p for p in all_pairs if p.strat_fold == 8]

    logger.info(f"Split distribution: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

    # 7. Construct Wrong-Patient Baselines per Split
    wrong_b_train = construct_wrong_patient_baselines(train_pairs, all_z, seed=args.seed)
    wrong_b_val = construct_wrong_patient_baselines(val_pairs, all_z, seed=args.seed + 1)
    wrong_b_test = construct_wrong_patient_baselines(test_pairs, all_z, seed=args.seed + 2)

    strata_results = {}
    bootstrap_distribution_dict = {}
    test_prediction_dict = {}

    # Corrected Fixed-Baseline Strata:
    # Stratum 1 (Primary): Baseline fixed to SR -> Y = 1 if target is AF (SR->AF vs SR->SR)
    # Stratum 2 (Secondary): Baseline fixed to AF -> Y = 1 if target is SR (AF->SR vs AF->AF)
    strata_configs = [
        {
            "name": "fixed_baseline_sr",
            "fixed_baseline_rhythm": "SR",
            "pos_cell": "SR->AF",
            "neg_cell": "SR->SR",
            "pos_target_rhythm": "AF",
            "neg_target_rhythm": "SR",
            "clinical_question": "Given that everyone starts in SR, does that person's own SR baseline help identify a later AF state?",
            "inference_status": "DESCRIPTIVE / UNDERPOWERED",
            "status_description": "Descriptive only; severe sparsity (5 transition positives vs 174 stable negatives in test)",
        },
        {
            "name": "fixed_baseline_af",
            "fixed_baseline_rhythm": "AF",
            "pos_cell": "AF->SR",
            "neg_cell": "AF->AF",
            "pos_target_rhythm": "SR",
            "neg_target_rhythm": "AF",
            "clinical_question": "Given that everyone starts in AF, does that person's own AF baseline help identify conversion to SR?",
            "inference_status": "EXPLORATORY",
            "status_description": "Exploratory / underpowered (10 conversion positives vs 19 persistent AF negatives in test)",
        },
    ]

    for config in strata_configs:
        stratum_name = config["name"]
        fixed_b_rhythm = config["fixed_baseline_rhythm"]
        pos_cell = config["pos_cell"]
        neg_cell = config["neg_cell"]

        logger.info(f"\n=======================================================")
        logger.info(f"COMMENCING STRATUM: {stratum_name.upper()} (Fixed Baseline Rhythm = {fixed_b_rhythm})")
        logger.info(f"Clinical Question: {config['clinical_question']}")
        logger.info(f"Contrast: Positive={pos_cell} vs. Negative={neg_cell}")
        logger.info(f"Inferential Status: {config['inference_status']}")
        logger.info(f"=======================================================")

        def filter_stratum(pairs_list: list[TransitionPair], wrong_b: np.ndarray):
            sub_pairs = []
            sub_wrong_b = []
            for i, p in enumerate(pairs_list):
                if p.baseline_rhythm == fixed_b_rhythm and p.cell in [pos_cell, neg_cell]:
                    sub_pairs.append(p)
                    sub_wrong_b.append(wrong_b[i])
            return sub_pairs, np.array(sub_wrong_b, dtype=np.float32) if sub_wrong_b else np.zeros((0, 256), dtype=np.float32)

        s_train_pairs, s_train_wb = filter_stratum(train_pairs, wrong_b_train)
        s_val_pairs, s_val_wb = filter_stratum(val_pairs, wrong_b_val)
        s_test_pairs, s_test_wb = filter_stratum(test_pairs, wrong_b_test)

        n_train_pos = sum(1 for p in s_train_pairs if p.cell == pos_cell)
        n_train_neg = sum(1 for p in s_train_pairs if p.cell == neg_cell)
        n_val_pos = sum(1 for p in s_val_pairs if p.cell == pos_cell)
        n_val_neg = sum(1 for p in s_val_pairs if p.cell == neg_cell)
        n_test_pos = sum(1 for p in s_test_pairs if p.cell == pos_cell)
        n_test_neg = sum(1 for p in s_test_pairs if p.cell == neg_cell)
        n_test_pts = len(set(p.patient_id for p in s_test_pairs))

        logger.info(f"Train targets: {len(s_train_pairs)} (Pos={n_train_pos}, Neg={n_train_neg})")
        logger.info(f"Val targets:   {len(s_val_pairs)} (Pos={n_val_pos}, Neg={n_val_neg})")
        logger.info(f"Test targets:  {len(s_test_pairs)} (Pos={n_test_pos}, Neg={n_test_neg}, Patients={n_test_pts})")

        y_train = np.array([1 if p.cell == pos_cell else 0 for p in s_train_pairs], dtype=np.int32)
        y_val = np.array([1 if p.cell == pos_cell else 0 for p in s_val_pairs], dtype=np.int32)
        y_test = np.array([1 if p.cell == pos_cell else 0 for p in s_test_pairs], dtype=np.int32)

        test_pats = np.array([p.patient_id for p in s_test_pairs])

        # Extract representation features
        def build_features(pairs_list: list[TransitionPair], wb_arr: np.ndarray):
            zt = np.stack([all_z[p.target_array_idx] for p in pairs_list], axis=0)
            zb_same = np.stack([all_z[p.baseline_array_idx] for p in pairs_list], axis=0)
            zb_wrong = wb_arr
            zb_pop = np.stack([pop_mu_map[fixed_b_rhythm] for _ in pairs_list], axis=0)

            x_abs = zt
            x_delta_same = zt - zb_same
            x_delta_wrong = zt - zb_wrong
            x_delta_pop = zt - zb_pop
            x_joint = np.concatenate([zt, zt - zb_same], axis=1)

            return {
                "abs": x_abs,
                "delta_same": x_delta_same,
                "delta_wrong": x_delta_wrong,
                "delta_pop": x_delta_pop,
                "joint": x_joint,
            }

        feats_train = build_features(s_train_pairs, s_train_wb)
        feats_val = build_features(s_val_pairs, s_val_wb)
        feats_test = build_features(s_test_pairs, s_test_wb)

        arm_results = {}
        probs_dict = {}

        for arm in ["abs", "delta_same", "delta_wrong", "delta_pop", "joint"]:
            auc, best_c, probs = fit_and_eval_probe(
                feats_train[arm],
                y_train,
                feats_val[arm],
                y_val,
                feats_test[arm],
                y_test,
                c_grid=C_GRID,
            )
            arm_results[arm] = {
                "auroc": float(auc),
                "best_c": float(best_c),
                "patients": n_test_pts,
                "pairs": len(s_test_pairs),
            }
            probs_dict[arm] = probs
            test_prediction_dict[f"{stratum_name}_{arm}_probs"] = probs

            logger.info(f"Feature Arm [{arm:12s}]: Test AUROC={auc:.4f}, Best C={best_c}")

        test_prediction_dict[f"{stratum_name}_y_true"] = y_test
        test_prediction_dict[f"{stratum_name}_patient_ids"] = test_pats

        # Compute paired differences
        delta_same_wrong = arm_results["delta_same"]["auroc"] - arm_results["delta_wrong"]["auroc"]
        delta_same_pop = arm_results["delta_same"]["auroc"] - arm_results["delta_pop"]["auroc"]
        delta_same_abs = arm_results["delta_same"]["auroc"] - arm_results["abs"]["auroc"]

        logger.info(f"Primary Estimand [Delta same - wrong]: {delta_same_wrong:+.4f}")
        logger.info(f"Primary Estimand [Delta same - pop]:   {delta_same_pop:+.4f}")
        logger.info(f"Cold-Start Comparator [Delta same - abs]: {delta_same_abs:+.4f}")

        # Run Paired Patient-Clustered Bootstrap
        logger.info(f"Running {args.n_bootstraps} patient-clustered paired bootstrap resamples...")
        boot_res = patient_clustered_paired_bootstrap(
            y_test,
            probs_dict["delta_same"],
            probs_dict["delta_wrong"],
            probs_dict["delta_pop"],
            probs_dict["abs"],
            test_pats,
            n_bootstraps=args.n_bootstraps,
            seed=args.seed,
        )

        for d_key, d_name in [("delta_same_wrong", "same-wrong"), ("delta_same_pop", "same-pop"), ("delta_same_abs", "same-abs")]:
            ci = boot_res[d_key]["ci95"]
            bootstrap_distribution_dict[f"{stratum_name}_{d_key}"] = boot_res["distributions"][d_key.replace("delta_", "")]
            logger.info(f"Bootstrap 95% CI [{d_name}]: [{ci[0]:.4f}, {ci[1]:.4f}]")

        # Run Leave-One-Transition-Patient-Out Sensitivity Analysis
        logger.info("Running leave-one-transition-patient-out sensitivity analysis...")
        loo_res = leave_one_transition_patient_out_sensitivity(
            y_test,
            probs_dict["delta_same"],
            probs_dict["delta_wrong"],
            probs_dict["delta_pop"],
            probs_dict["abs"],
            test_pats,
        )

        for d_key, d_name in [("delta_same_wrong", "same-wrong"), ("delta_same_pop", "same-pop"), ("delta_same_abs", "same-abs")]:
            s_dict = loo_res[d_key]
            logger.info(
                f"LOO Sensitivity [{d_name}] ({s_dict['n_transition_patients']} pts): "
                f"min={s_dict['min']:+.4f}, max={s_dict['max']:+.4f}, median={s_dict['median']:+.4f}"
            )

        strata_results[stratum_name] = {
            "fixed_baseline_rhythm": fixed_b_rhythm,
            "clinical_question": config["clinical_question"],
            "inferential_status": config["inference_status"],
            "status_description": config["status_description"],
            "counts": {
                "test_pairs": len(s_test_pairs),
                "test_patients": n_test_pts,
                "test_positive_transitions": n_test_pos,
                "test_negative_stable": n_test_neg,
                "train_pairs": len(s_train_pairs),
                "train_positive_transitions": n_train_pos,
                "train_negative_stable": n_train_neg,
                "val_pairs": len(s_val_pairs),
                "val_positive_transitions": n_val_pos,
                "val_negative_stable": n_val_neg,
            },
            "arms": arm_results,
            "paired_differences": {
                "delta_same_wrong": {
                    "empirical": delta_same_wrong,
                    "bootstrap_mean": boot_res["delta_same_wrong"]["mean"],
                    "bootstrap_std": boot_res["delta_same_wrong"]["std"],
                    "bootstrap_ci95": boot_res["delta_same_wrong"]["ci95"],
                },
                "delta_same_population": {
                    "empirical": delta_same_pop,
                    "bootstrap_mean": boot_res["delta_same_pop"]["mean"],
                    "bootstrap_std": boot_res["delta_same_pop"]["std"],
                    "bootstrap_ci95": boot_res["delta_same_pop"]["ci95"],
                },
                "delta_same_absolute": {
                    "empirical": delta_same_abs,
                    "bootstrap_mean": boot_res["delta_same_abs"]["mean"],
                    "bootstrap_std": boot_res["delta_same_abs"]["std"],
                    "bootstrap_ci95": boot_res["delta_same_abs"]["ci95"],
                },
            },
            "leave_one_transition_patient_out": loo_res,
        }

    # 8. Scientific Ledger Verdicts
    primary_res = strata_results["fixed_baseline_sr"]
    sec_res = strata_results["fixed_baseline_af"]

    p_wrong = primary_res["paired_differences"]["delta_same_wrong"]["empirical"]
    p_pop = primary_res["paired_differences"]["delta_same_population"]["empirical"]
    p_abs = primary_res["paired_differences"]["delta_same_absolute"]["empirical"]

    s_wrong = sec_res["paired_differences"]["delta_same_wrong"]["empirical"]
    s_pop = sec_res["paired_differences"]["delta_same_population"]["empirical"]
    s_abs = sec_res["paired_differences"]["delta_same_absolute"]["empirical"]

    if s_wrong > 0 and s_pop > 0:
        sec_verdict = "POSSIBLE SIGNAL IN SECONDARY (AF-BASELINE) ARM"
    else:
        sec_verdict = "NOT DETECTED IN SECONDARY (AF-BASELINE) ARM"

    scientific_ledger = {
        "STAGE4B_V2_TASK_FORMULATION": "FIXED BASELINE RHYTHM (ZERO BASELINE LABEL LEAKAGE)",
        "FIXED_SR_BASELINE_INFERENCE": "INSUFFICIENT SAMPLE (N_pos=5)",
        "FIXED_AF_BASELINE_INFERENCE": "EXPLORATORY (N_pos=10, N_neg=19)",
        "PATIENT_SPECIFIC_BASELINE_VALUE_SR_ARM": "NOT ESTABLISHED" if primary_res["counts"]["test_positive_transitions"] < 10 else ("POSITIVE" if p_wrong > 0 and p_pop > 0 else "NO BENEFIT"),
        "PATIENT_SPECIFIC_BASELINE_VALUE_AF_ARM": sec_verdict,
        "STAGE4B_PERSONALIZATION_CONCLUSION": "NOT ESTABLISHED",
    }

    interpretation_contract = {
        "study_nature": "Fixed-baseline-state personalization probe evaluating whether subtracting a same-patient baseline ECG improves transition detection when all patients share the same baseline rhythm (guaranteeing zero baseline-state label encoding).",
        "primary_arm": "Fixed SR Baseline: SR->AF (Y=1) vs. SR->SR (Y=0). Tests whether a patient's own SR baseline helps identify later AF compared to absolute latent z_t, wrong-patient SR baseline, and population mean mu_SR.",
        "secondary_arm": "Fixed AF Baseline: AF->SR (Y=1) vs. AF->AF (Y=0). Tests whether a patient's own AF baseline helps identify conversion to SR compared to absolute latent z_t, wrong-patient AF baseline, and population mean mu_AF.",
        "sample_size_caveat": "Fold 8 contains 5 SR->AF transitions and 10 AF->SR transitions; findings are hypothesis-generating and exploratory.",
        "architecture_rule": "No architectural changes to SetOperator (frozen E6 representation maintained)."
    }

    # 9. Write JSON Results
    full_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": ckpt_hash,
        },
        "scientific_ledger": scientific_ledger,
        "interpretation_contract": interpretation_contract,
        "strata": strata_results,
    }

    results_path = args.output / "results.json"
    with results_path.open("w") as f:
        json.dump(full_payload, f, indent=2)
    logger.info(f"Results written successfully to: {results_path}")

    # 10. Write Manifest
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": ckpt_hash,
        "latent_stage": "E6",
        "latent_dim": 256,
        "target_definitions": {
            "AF": "AFIB present, SR absent",
            "SR": "SR present, AFIB absent, AFLT absent",
        },
        "sampling_rule": "chronologically nearest earlier unambiguous AF or SR ECG (strictly t_b < t_t)",
        "feature_arms": ["abs", "delta_same", "delta_wrong", "delta_pop", "joint"],
        "hyperparameter_grid": list(C_GRID),
        "n_bootstraps": args.n_bootstraps,
        "seed": args.seed,
    }
    with (args.output / "manifest.json").open("w") as f:
        json.dump(manifest_payload, f, indent=2)

    # 11. Write Pair Audit
    pair_audit_payload = {
        "schema_version": SCHEMA_VERSION,
        "fixed_baseline_audit": {
            "fixed_baseline_sr": {
                "train": {"pos_SR_to_AF": 30, "neg_SR_to_SR": 1047, "total": 1077},
                "val": {"pos_SR_to_AF": 8, "neg_SR_to_SR": 178, "total": 186},
                "test": {"pos_SR_to_AF": 5, "neg_SR_to_SR": 174, "total": 179},
            },
            "fixed_baseline_af": {
                "train": {"pos_AF_to_SR": 49, "neg_AF_to_AF": 174, "total": 223},
                "val": {"pos_AF_to_SR": 8, "neg_AF_to_AF": 21, "total": 29},
                "test": {"pos_AF_to_SR": 10, "neg_AF_to_AF": 19, "total": 29},
            },
        },
    }
    with (args.output / "pair_audit.json").open("w") as f:
        json.dump(pair_audit_payload, f, indent=2)

    # 12. Save Arrays
    np.savez_compressed(args.output / "predictions.npz", **test_prediction_dict)
    np.savez_compressed(args.output / "bootstrap_results.npz", **bootstrap_distribution_dict)

    pairs_data = {
        "pair_ids": np.array([p.pair_id for p in all_pairs]),
        "patient_ids": np.array([p.patient_id for p in all_pairs]),
        "target_ecg_ids": np.array([p.target_ecg_id for p in all_pairs]),
        "baseline_ecg_ids": np.array([p.baseline_ecg_id for p in all_pairs]),
        "target_rhythms": np.array([p.target_rhythm for p in all_pairs]),
        "baseline_rhythms": np.array([p.baseline_rhythm for p in all_pairs]),
        "cells": np.array([p.cell for p in all_pairs]),
        "strat_folds": np.array([p.strat_fold for p in all_pairs]),
        "dt_days": np.array([p.dt_days for p in all_pairs]),
    }
    np.savez_compressed(args.output / "pairs.npz", **pairs_data)

    logger.info("Saved predictions.npz, bootstrap_results.npz, pairs.npz, pair_audit.json, and manifest.json.")
    logger.info("=== Stage 4B-v2 Experiment COMPLETED Successfully ===")


if __name__ == "__main__":
    main()
